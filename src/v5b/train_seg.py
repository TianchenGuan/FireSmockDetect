"""V5b segmentation trainer (torchrun, DDP, bf16 autocast). One process group per
config; the pod entry runs several configs concurrently on disjoint GPU sets.

Per-epoch eval on the VAL role (heinola only): smoke-pixel global IoU / Dice /
precision / recall over valid pixels, mean per-image IoU on positive frames,
negative-frame pixel-FP rate and frame false-alarm rate, and mask->box recall
against the authors' boxes (one large box per frame) under three criteria.
best.pt = highest global smoke IoU on VAL. Final pass with --dump-probs writes
val_probs/<id>.png (uint8 prob map at stride 4) + val_per_image.json."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parents[1]):  # flat pod dir / repo root
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from src.v5b.seg_head import (build_model, seg_loss, saved_state_dict,
                                  load_saved_state_dict, param_counts)
    from src.v5b.seg_data import SegDataset, collate
except ImportError:
    from seg_head import (build_model, seg_loss, saved_state_dict,  # type: ignore
                          load_saved_state_dict, param_counts)
    from seg_data import SegDataset, collate  # type: ignore


def ddp_env():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    return 0, 1, 0


def log(rank, *a):
    if rank == 0:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------- connected components (torch, label propagation) -----------------
def components(fg: torch.Tensor, max_iter=2000):
    """fg: [H,W] bool -> int32 labels (0 = bg). Max-pool label propagation."""
    h, w = fg.shape
    lab = torch.arange(1, h * w + 1, device=fg.device, dtype=torch.float32).view(1, 1, h, w)
    lab = lab * fg.float()
    for _ in range(max_iter):
        new = F.max_pool2d(lab, 3, 1, 1) * fg.float()
        if torch.equal(new, lab):
            break
        lab = new
    return lab.view(h, w).long()


def comp_boxes(fg: torch.Tensor, min_area: int):
    lab = components(fg)
    ids, cnt = torch.unique(lab, return_counts=True)
    boxes = []
    for i, c in zip(ids.tolist(), cnt.tolist()):
        if i == 0 or c < min_area:
            continue
        ys, xs = torch.nonzero(lab == i, as_tuple=True)
        boxes.append([xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1, c])
    return boxes


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def mask_box_stats(fg_low: torch.Tensor, gt_boxes_px, min_area):
    """fg_low: [h,w] bool at eval grid; gt boxes in the same pixel grid.
    Returns per-GT-box hits under: A comp-IoU>=0.4, B union-bbox IoU>=0.4,
    C 'hit' (>=30% of predicted pixels inside GT box and GT box >=10% covered)."""
    comps = comp_boxes(fg_low, min_area)
    res = []
    npix = int(fg_low.sum())
    for g in gt_boxes_px:
        a = any(box_iou(c[:4], g) >= 0.4 for c in comps)
        if comps:
            ub = [min(c[0] for c in comps), min(c[1] for c in comps),
                  max(c[2] for c in comps), max(c[3] for c in comps)]
            b = box_iou(ub, g) >= 0.4
        else:
            b = False
        x1, y1, x2, y2 = [int(round(v)) for v in g]
        inside = int(fg_low[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].sum())
        area = max(1, (x2 - x1) * (y2 - y1))
        c = npix > 0 and inside / max(npix, 1) >= 0.3 and inside / area >= 0.10
        res.append({"A": bool(a), "B": bool(b), "C": bool(c)})
    return res, len(comps)


# ---------------- evaluation -----------------
@torch.no_grad()
def evaluate(model, loader, device, rank, world, thr=0.5, dump_dir=None, amp=None):
    model.eval()
    per_img = []
    for xs, tgt, valid, hm, metas in loader:
        xs = xs.to(device, non_blocking=True)
        if amp:
            with amp:
                _, low = model(xs)
        else:
            _, low = model(xs)
        prob_low = torch.sigmoid(low.float())  # [B,1,h,w] stride 4
        h, w = prob_low.shape[-2:]
        tgt_low = F.interpolate(tgt.to(device), size=(h, w), mode="nearest")
        val_low = F.interpolate(valid.to(device), size=(h, w), mode="nearest") > 0.5
        gt_low = tgt_low > 0.5
        pred = (prob_low >= thr) & val_low
        for b, m in enumerate(metas):
            p, g, v = pred[b, 0], gt_low[b, 0] & val_low[b, 0], val_low[b, 0]
            tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
            nvalid = int(v.sum())
            rec = {"id": m["id"], "burn": m["burn"], "flight": m["flight"],
                   "provenance": m["provenance"], "is_negative": m["is_negative"],
                   "tp": tp, "fp": fp, "fn": fn, "n_valid": nvalid, "gt_pix": int(g.sum()),
                   "pred_pix": int(p.sum())}
            rec["iou"] = tp / max(tp + fp + fn, 1) if (tp + fp + fn) else 1.0
            min_area = max(8, int(0.0005 * nvalid))
            # frame false alarm: any component >= 0.05% of valid pixels
            comps = comp_boxes(p, min_area)
            rec["n_comp"] = len(comps)
            rec["max_comp_frac"] = (max(c[4] for c in comps) / max(nvalid, 1)) if comps else 0.0
            if m["boxes_xyxy_norm"]:
                s = m["scale"]  # letterbox: boxes normalized to orig -> eval grid (stride 4)
                gb = [[bx[0] * m["orig_w"] * s / 4 + m["ox"] / 4, bx[1] * m["orig_h"] * s / 4 + m["oy"] / 4,
                       bx[2] * m["orig_w"] * s / 4 + m["ox"] / 4, bx[3] * m["orig_h"] * s / 4 + m["oy"] / 4]
                      for bx in m["boxes_xyxy_norm"]]
                rec["box_hits"], _ = mask_box_stats(p, gb, min_area)
                rec["box_hits_gtmask"], _ = mask_box_stats(g, gb, min_area)  # SAM/manual mask reference
            if dump_dir is not None:
                arr = (prob_low[b, 0].clamp(0, 1) * 255).round().byte().cpu().numpy()
                Image.fromarray(arr).save(Path(dump_dir) / f"{m['id']}.png")
            per_img.append(rec)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, per_img)
        per_img = [r for g in gathered for r in g]
    model.train()
    return summarize(per_img), per_img


def summarize(per_img):
    # de-dup (DistributedSampler pads)
    seen, rows = set(), []
    for r in per_img:
        if r["id"] not in seen:
            seen.add(r["id"]); rows.append(r)
    pos = [r for r in rows if not r["is_negative"]]
    neg = [r for r in rows if r["is_negative"]]
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    out = {"n": len(rows), "n_pos": len(pos), "n_neg": len(neg),
           "iou": tp / max(tp + fp + fn, 1), "dice": 2 * tp / max(2 * tp + fp + fn, 1),
           "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
           "miou_pos": float(np.mean([r["iou"] for r in pos])) if pos else 0.0}
    for prov in ("sam_model_assisted", "manual_human"):
        sub = [r for r in pos if r["provenance"] == prov]
        if sub:
            t = sum(r["tp"] for r in sub); f = sum(r["fp"] for r in sub); n = sum(r["fn"] for r in sub)
            out[f"iou_{prov}"] = t / max(t + f + n, 1)
            out[f"miou_{prov}"] = float(np.mean([r["iou"] for r in sub]))
            out[f"n_{prov}"] = len(sub)
    if neg:
        out["neg_fp_pixel_frac"] = sum(r["fp"] for r in neg) / max(sum(r["n_valid"] for r in neg), 1)
        out["neg_frame_false_alarm"] = float(np.mean([r["n_comp"] > 0 for r in neg]))
        out["neg_frame_fa_1pct"] = float(np.mean([r["max_comp_frac"] >= 0.01 for r in neg]))
    for key, name in (("box_hits", "box_recall"), ("box_hits_gtmask", "box_recall_gtmask_ref")):
        hits = [h for r in pos for h in r.get(key, [])]
        if hits:
            out[f"{name}_A_iou04"] = float(np.mean([h["A"] for h in hits]))
            out[f"{name}_B_union_iou04"] = float(np.mean([h["B"] for h in hits]))
            out[f"{name}_C_hit"] = float(np.mean([h["C"] for h in hits]))
            out[f"{name}_n_boxes"] = len(hits)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["dino_frozen", "dino_lora", "r50"])
    ap.add_argument("--backbone-dir", default=None)
    ap.add_argument("--random-spec", default=None, help="CPU wiring tests only")
    ap.add_argument("--r50-weights", default=None)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=None)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=736)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--time-budget-s", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every-epochs", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--dump-probs", action="store_true", help="final: dump val prob maps")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    rank, world, local = ddp_env()
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    model = build_model(args.kind, args.backbone_dir, args.random_spec, args.lora, args.lora_alpha,
                        args.grad_ckpt, args.r50_weights).to(device)
    pc = param_counts(model)
    for k, v in pc.items():
        log(rank, f"param_counts[{k}]: trainable={v['trainable']} total={v['total']}")
    if rank == 0:
        (outdir / "param_counts.json").write_text(json.dumps(pc, indent=1))

    train_ds = SegDataset(args.manifest, args.data_root, "train", args.width, args.height, True,
                          args.max_train, args.seed)
    val_ds = SegDataset(args.manifest, args.data_root, "val", args.width, args.height, False,
                        args.max_val, args.seed)
    log(rank, f"train rows {len(train_ds)} | val rows {len(val_ds)} | world {world}")
    sampler = torch.utils.data.distributed.DistributedSampler(train_ds, world, rank, shuffle=True,
                                                              drop_last=True) if world > 1 else None
    vsampler = torch.utils.data.distributed.DistributedSampler(val_ds, world, rank, shuffle=False) \
        if world > 1 else None
    train_ld = torch.utils.data.DataLoader(train_ds, args.bs, shuffle=sampler is None, sampler=sampler,
                                           num_workers=args.workers, collate_fn=collate, drop_last=True,
                                           pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    val_ld = torch.utils.data.DataLoader(val_ds, max(1, args.bs), shuffle=False, sampler=vsampler,
                                         num_workers=args.workers, collate_fn=collate)
    steps_per_epoch = max(1, len(train_ld))
    total_steps = int(math.ceil(args.epochs * steps_per_epoch)) if args.max_steps < 0 else args.max_steps

    bb_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    groups = [{"params": head_params, "lr": args.lr, "base_lr": args.lr}]
    if bb_params:
        groups.append({"params": bb_params, "lr": args.backbone_lr, "base_lr": args.backbone_lr})
    log(rank, f"lr groups: head {args.lr} ({sum(p.numel() for p in head_params)}) | "
              f"backbone {args.backbone_lr if bb_params else '-'} ({sum(p.numel() for p in bb_params)})")
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)

    def lr_at(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        p = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    start_step, best_iou, best_step = 0, -1.0, -1
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        load_saved_state_dict(model, ck["model"]); opt.load_state_dict(ck["opt"])
        start_step, best_iou, best_step = ck["step"], ck.get("best_iou", -1.0), ck.get("best_step", -1)
        log(rank, f"resumed from {args.resume} @ step {start_step}")

    net = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local] if device.type == "cuda" else None,
            find_unused_parameters=False)

    def save(path, step, with_opt=True, extra=None):
        if rank != 0:
            return
        d = {"model": saved_state_dict(net), "step": step, "best_iou": best_iou, "best_step": best_step,
             "args": vars(args), "kind": args.kind, "param_counts": pc,
             "backbone_dir": args.backbone_dir, "input_wh": [args.width, args.height]}
        if with_opt:
            d["opt"] = opt.state_dict()
        if extra:
            d.update(extra)
        torch.save(d, str(path) + ".tmp"); os.replace(str(path) + ".tmp", path)

    mf = (outdir / "metrics.jsonl").open("a") if rank == 0 else None

    def emit(rec):
        if mf:
            mf.write(json.dumps(rec) + "\n"); mf.flush()

    amp = torch.autocast("cuda", dtype=torch.bfloat16) if (device.type == "cuda" and not args.no_amp) else None
    step, epoch = start_step, start_step // steps_per_epoch
    t_log, acc, n_acc = time.time(), {"loss": 0.0, "bce": 0.0, "dice": 0.0}, 0
    stopped = False
    while step < total_steps and not stopped:
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        for xs, tgt, valid, hm, metas in train_ld:
            if step >= total_steps:
                break
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_at(step)
            xs, tgt, valid, hm = (t.to(device, non_blocking=True) for t in (xs, tgt, valid, hm))
            if amp:
                with amp:
                    logits, _ = model(xs)
            else:
                logits, _ = model(xs)
            losses = seg_loss(logits, tgt, valid, hm)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 5.0)
            opt.step()
            for k in acc:
                acc[k] += float(losses[k].detach())
            n_acc += 1; step += 1
            if step % 10 == 0 or step == total_steps:
                dt = (time.time() - t_log) / max(1, n_acc)
                log(rank, f"ep {epoch} step {step}/{total_steps} loss {acc['loss']/n_acc:.4f} "
                          f"bce {acc['bce']/n_acc:.4f} dice {acc['dice']/n_acc:.4f} "
                          f"lr {opt.param_groups[0]['lr']:.2e} {dt:.2f}s/it")
                emit({"step": step, "epoch": epoch, **{k: v / n_acc for k, v in acc.items()},
                      "lr": opt.param_groups[0]["lr"], "s_per_it": dt, "t": time.time()})
                t_log, acc, n_acc = time.time(), {k: 0.0 for k in acc}, 0
            if args.time_budget_s and time.time() - t_start > args.time_budget_s:
                log(rank, "time budget reached -> checkpoint and stop")
                save(outdir / "last.pt", step); stopped = True
                break
        if stopped:
            break
        epoch += 1
        if epoch % args.eval_every_epochs == 0 or step >= total_steps:
            m, _ = evaluate(model, val_ld, device, rank, world, args.thr, None, amp)
            log(rank, f"eval@ep{epoch} step{step}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                                     for k, v in m.items()}))
            emit({"step": step, "epoch": epoch, "eval": m, "t": time.time()})
            if m["iou"] > best_iou:
                best_iou, best_step = m["iou"], step
                save(outdir / "best.pt", step, with_opt=False, extra={"val_metrics": m, "epoch": epoch})
        save(outdir / "last.pt", step)
        if world > 1:
            dist.barrier()

    if not stopped:
        # final: reload best, dump probs + per-image metrics
        if world > 1:
            dist.barrier()
        if (outdir / "best.pt").exists():
            ck = torch.load(outdir / "best.pt", map_location="cpu", weights_only=False)
            load_saved_state_dict(net, ck["model"])
        dump = None
        if args.dump_probs:
            dump = outdir / "val_probs"; dump.mkdir(exist_ok=True)
        m, per = evaluate(model, val_ld, device, rank, world, args.thr, dump, amp)
        if rank == 0:
            (outdir / "val_metrics_best.json").write_text(json.dumps(
                {"summary": m, "best_step": best_step, "best_iou": best_iou, "thr": args.thr}, indent=1))
            seen = set(); uniq = [r for r in per if not (r["id"] in seen or seen.add(r["id"]))]
            (outdir / "val_per_image.json").write_text(json.dumps(uniq))
            if dump is not None:
                with tarfile.open(outdir / "val_probs.tar", "w") as tf:
                    tf.add(dump, arcname="val_probs")
                    tf.add(outdir / "val_per_image.json", arcname="val_per_image.json")
            (outdir / "DONE").write_text(json.dumps({"step": step, "best_iou": best_iou, "best_step": best_step}))
        log(rank, "SEG_TRAIN_DONE", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}))
    if mf:
        mf.close()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
