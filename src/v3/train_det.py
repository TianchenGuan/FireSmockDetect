"""Track B/C detector trainer (DDP, resumable, offline-pod-safe).

Modes:
  --backbone-dir <hf snapshot>          end-to-end (frozen or partial-unfreeze)
  --feat-root <dir> --embed-dim <C>     C1 cached-feature head training
  --random-spec dinov3_vitl16           random-init wiring test (no weights)

Backbone adaptation (v4, mutually exclusive; default stays frozen):
  --unfreeze-top N                      trainable top N blocks (B3)
  --unfreeze-all                        entire ViT trainable
  --lora R [--lora-alpha A]             hand-rolled LoRA on attn q/k/v/o
  --backbone-lr LR                      absolute backbone-group LR
  --grad-ckpt                           HF gradient checkpointing

Prints DET_TRAIN_DONE on completion; writes metrics.jsonl, last.pt, best.pt
(trainable-only state; frozen backbone params come from_pretrained on load).
Run under torchrun; single-process works too (LOCAL_RANK absent).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from det_data import CachedFeatureDataset, CocoDetDataset, collate  # noqa: E402
from det_modeling import (ViTBackbone, build_detector, forward_with_features,  # noqa: E402
                          load_trainable_state_dict, param_group_counts,
                          trainable_state_dict)


def ddp_env():
    if "LOCAL_RANK" in os.environ:
        local = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


@torch.no_grad()
def evaluate(model, loader, device, cached, max_dets=100):
    """COCO bbox mAP on the loader's annotations (single class)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    net = model.module if hasattr(model, "module") else model
    net.eval()
    ds = loader.dataset
    results, img_ids = [], []
    for batch in loader:
        xs, targets = batch
        img_ids += [int(t["image_id"]) for t in targets]
        if cached:
            fmaps = torch.stack([x for x in xs]).to(device)
            shapes = [(ds.th, ds.tw)] * len(xs)
            _, dets = forward_with_features(net, fmaps, shapes)
        else:
            xs = [x.to(device) for x in xs]
            dets = net(xs)
        for t, d in zip(targets, dets):
            r = float(t["scale"])
            for box, score in zip(d["boxes"].cpu(), d["scores"].cpu()):
                x1, y1, x2, y2 = [v / r for v in box.tolist()]  # back to orig px
                results.append({"image_id": int(t["image_id"]), "category_id": 1,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": float(score)})
    # gather across ranks
    if dist.is_initialized():
        all_r = [None] * dist.get_world_size()
        all_i = [None] * dist.get_world_size()
        dist.all_gather_object(all_r, results)
        dist.all_gather_object(all_i, img_ids)
        results = [r for part in all_r for r in part]
        img_ids = [i for part in all_i for i in part]
    if (dist.is_initialized() and dist.get_rank() != 0):
        return {}
    # pycocotools 2.0.10 loadRes deepcopies dataset['info'] -> must exist
    gt = {"info": {}, "licenses": [],
          "images": [{"id": it["id"], "width": it["w"], "height": it["h"]}
                     for it in ds.items if it["id"] in set(img_ids)],
          "annotations": [], "categories": [{"id": 1, "name": "smoke"}]}
    aid = 1
    for it in ds.items:
        if it["id"] not in set(img_ids):
            continue
        for b in it["boxes"]:
            gt["annotations"].append({"id": aid, "image_id": it["id"],
                                      "category_id": 1, "bbox": b,
                                      "area": b[2] * b[3], "iscrowd": 0})
            aid += 1
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(open(os.devnull, "w")):
        coco.createIndex()
        if not results:
            return {"mAP": 0.0, "AP50": 0.0, "AR100": 0.0, "n_dets": 0}
        cocoDt = coco.loadRes(results)
        ev = COCOeval(coco, cocoDt, "bbox")
        ev.params.maxDets = [1, 10, max_dets]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"mAP": float(ev.stats[0]), "AP50": float(ev.stats[1]),
            "AR100": float(ev.stats[8]), "n_dets": len(results)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-dir", default=None)
    ap.add_argument("--random-spec", default=None,
                    help="dinov3_vitl16|dinov3_vith16plus|dinov3_vit7b16 (random init wiring test)")
    ap.add_argument("--feat-root", default=None, help="cached-feature (C1) root")
    ap.add_argument("--embed-dim", type=int, default=None, help="required with --feat-root")
    ap.add_argument("--unfreeze-top", type=int, default=0, help="B3: trainable top blocks")
    ap.add_argument("--unfreeze-all", action="store_true",
                    help="entire ViT trainable (incl. embeddings/cls/registers)")
    ap.add_argument("--lora", type=int, default=0, metavar="R",
                    help="LoRA rank on ViT attention q/k/v/o (backbone weights "
                         "frozen except LoRA params); 0 = off")
    ap.add_argument("--lora-alpha", type=float, default=None,
                    help="LoRA scaling alpha (default 2*R)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="HF gradient checkpointing on the backbone (for "
                         "--unfreeze-all / --lora memory; --unfreeze-top "
                         "already always enables it)")
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--val-json", default=None)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=736)
    ap.add_argument("--epochs", type=float, default=12)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--bs", type=int, default=4, help="per-GPU batch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=None,
                    help="absolute LR for trainable backbone params (LoRA / "
                         "unfreeze); overrides --backbone-lr-scale. Default "
                         "None keeps the v3 behavior lr*backbone-lr-scale")
    ap.add_argument("--backbone-lr-scale", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--resume", default=None, help="path to last.pt")
    ap.add_argument("--save-every-steps", type=int, default=500)
    ap.add_argument("--eval-max-images", type=int, default=1500)
    ap.add_argument("--eval-every-epochs", type=int, default=1)
    ap.add_argument("--overfit-n", type=int, default=0,
                    help="train on N images (>=half positives) and eval on the SAME N")
    ap.add_argument("--time-budget-s", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t_start = time.time()
    rank, world, local = ddp_env()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cached = args.feat_root is not None

    # ---- model
    if sum([args.unfreeze_top > 0, args.unfreeze_all, args.lora > 0]) > 1:
        raise SystemExit("--unfreeze-top / --unfreeze-all / --lora are mutually exclusive")
    if cached:
        assert args.embed_dim, "--embed-dim required with --feat-root"
        assert not (args.unfreeze_top or args.unfreeze_all or args.lora), \
            "backbone adaptation flags need an actual backbone (not --feat-root)"
        backbone = None
        model = build_detector(None, embed_dim=args.embed_dim,
                               image_size=(args.width, args.height))
    else:
        backbone = ViTBackbone(args.backbone_dir,
                               frozen=args.unfreeze_top == 0,
                               unfreeze_top=args.unfreeze_top,
                               unfreeze_all=args.unfreeze_all,
                               lora_r=args.lora, lora_alpha=args.lora_alpha,
                               grad_ckpt=args.grad_ckpt,
                               random_init_spec=args.random_spec)
        model = build_detector(backbone, image_size=(args.width, args.height))
        if args.lora:
            log(rank, f"lora: r={args.lora} alpha={backbone.lora_alpha} on "
                      f"{len(backbone.lora_modules)} attention projections "
                      f"(e.g. {backbone.lora_modules[0]})")
    model.to(device)
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_p = sum(p.numel() for p in model.parameters())
    log(rank, f"params: trainable {n_train_p/1e6:.1f}M / total {n_total_p/1e6:.1f}M "
              f"(grid {args.width//16}x{args.height//16} @patch16-equivalent)")
    pc = param_group_counts(model)  # REQUIRED output: exact per-group counts
    for g in ("backbone", "neck", "head", "all"):
        log(rank, f"param_counts[{g}]: trainable={pc[g]['trainable']} "
                  f"total={pc[g]['total']}")

    # ---- data
    size = (args.width, args.height)
    if cached:
        train_ds = CachedFeatureDataset(args.train_json, args.image_root,
                                        args.feat_root, size=size)
    else:
        train_ds = CocoDetDataset(args.train_json, args.image_root, size=size, train=True)
    if args.overfit_n:
        pos = [it["id"] for it in train_ds.items if it["boxes"]][: args.overfit_n * 3 // 4]
        neg = [it["id"] for it in train_ds.items if not it["boxes"]][: args.overfit_n - len(pos)]
        keep = set(pos + neg)
        train_ds.items = [it for it in train_ds.items if it["id"] in keep]
        log(rank, f"overfit mode: {len(pos)} pos + {len(neg)} neg images")
    val_ds = None
    if args.val_json:
        if cached:
            val_ds = CachedFeatureDataset(args.val_json, args.image_root, args.feat_root,
                                          size=size, max_images=args.eval_max_images,
                                          seed=args.seed)
        else:
            val_ds = CocoDetDataset(args.val_json, args.image_root, size=size,
                                    train=False, max_images=args.eval_max_images,
                                    seed=args.seed)
    elif args.overfit_n:
        if cached:
            val_ds = CachedFeatureDataset(args.train_json, args.image_root,
                                          args.feat_root, size=size)
        else:
            val_ds = CocoDetDataset(args.train_json, args.image_root, size=size,
                                    train=False)
        val_ds.items = list(train_ds.items)

    sampler = (torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
               if world > 1 else None)
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=args.bs, shuffle=sampler is None, sampler=sampler,
        num_workers=args.workers, collate_fn=collate, drop_last=len(train_ds) > args.bs,
        persistent_workers=args.workers > 0)
    val_ld = None
    if val_ds is not None:
        vsampler = (torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
                    if world > 1 else None)
        val_ld = torch.utils.data.DataLoader(
            val_ds, batch_size=max(1, args.bs), sampler=vsampler, shuffle=False,
            num_workers=args.workers, collate_fn=collate)

    # ---- optimizer: separate LR for (partially) trainable backbone
    # (unfreeze-top / unfreeze-all blocks, or LoRA params — all live under
    # backbone.backbone.*). --backbone-lr sets it absolutely; otherwise the
    # original v3 behavior lr * backbone-lr-scale applies.
    head_params, bb_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb_params if n.startswith("backbone.backbone.") else head_params).append(p)
    bb_base_lr = (args.backbone_lr if args.backbone_lr is not None
                  else args.lr * args.backbone_lr_scale)
    group_base_lrs = [args.lr]
    groups = [{"params": head_params, "lr": args.lr}]
    if bb_params:
        groups.append({"params": bb_params, "lr": bb_base_lr})
        group_base_lrs.append(bb_base_lr)
        log(rank, f"lr groups: head {args.lr:.2e} ({sum(p.numel() for p in head_params)} "
                  f"params) | backbone {bb_base_lr:.2e} "
                  f"({sum(p.numel() for p in bb_params)} params)")
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)

    steps_per_epoch = max(1, len(train_ld))
    total_steps = (args.max_steps if args.max_steps > 0
                   else int(steps_per_epoch * args.epochs))

    def lr_at(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        p = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    start_step, best_map = 0, -1.0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        load_trainable_state_dict(model, ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step, best_map = ck["step"], ck.get("best_map", -1.0)
        log(rank, f"resumed from {args.resume} @ step {start_step}")

    # cached route bypasses model.forward (drives RPN/RoI internals directly),
    # which is incompatible with DDP's reducer -> plain model + manual grad
    # all-reduce there; standard DDP wrap for the end-to-end route.
    if world > 1 and not cached:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local] if device.type == "cuda" else None)
    net = model.module if hasattr(model, "module") else model

    def save(path, step):
        if rank != 0:
            return
        torch.save({"model": trainable_state_dict(net), "opt": opt.state_dict(),
                    "step": step, "best_map": best_map,
                    "args": vars(args)}, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    metrics_f = (outdir / "metrics.jsonl").open("a") if rank == 0 else None

    def emit(rec):
        if metrics_f:
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()

    # ---- train loop
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else None
    step = start_step
    epoch = step // steps_per_epoch
    done = False
    t_log, loss_acc, n_acc = time.time(), 0.0, 0
    while step < total_steps and not done:
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        for xs, targets in train_ld:
            if step >= total_steps:
                break
            for g_i, g in enumerate(opt.param_groups):
                g["lr"] = group_base_lrs[g_i] * lr_at(step)
            if cached:
                fmaps = torch.stack(list(xs)).to(device)
                shapes = [(args.height, args.width)] * len(xs)
                if amp:
                    with amp:
                        losses, _ = forward_with_features(net, fmaps, shapes, list(targets))
                else:
                    losses, _ = forward_with_features(net, fmaps, shapes, list(targets))
                loss = sum(losses.values())
            else:
                xs = [x.to(device) for x in xs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                if amp:
                    with amp:
                        losses = model(xs, targets)
                else:
                    losses = model(xs, targets)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if world > 1 and cached:
                # DDP was bypassed (internals called directly): all-reduce grads
                for p in net.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad], 10.0)
            opt.step()
            loss_acc += float(loss.detach())
            n_acc += 1
            step += 1
            if step % 20 == 0 or step == total_steps:
                dt = time.time() - t_log
                # v5b: peak GPU memory (GiB, this rank) for the B200 batch-size audit
                mem_gib = (torch.cuda.max_memory_allocated() / 2**30
                           if device.type == "cuda" else 0.0)
                log(rank, f"step {step}/{total_steps} loss {loss_acc/max(1,n_acc):.4f} "
                          f"lr {opt.param_groups[0]['lr']:.2e} {dt/max(1,n_acc):.2f}s/it "
                          f"mem {mem_gib:.1f}G")
                emit({"step": step, "loss": loss_acc / max(1, n_acc),
                      "lr": opt.param_groups[0]["lr"], "s_per_it": dt / max(1, n_acc),
                      "peak_mem_gib": round(mem_gib, 2), "t": time.time()})
                t_log, loss_acc, n_acc = time.time(), 0.0, 0
            if args.save_every_steps and step % args.save_every_steps == 0:
                save(outdir / "last.pt", step)
            if args.time_budget_s and time.time() - t_start > args.time_budget_s:
                log(rank, "time budget reached — checkpoint and stop")
                save(outdir / "last.pt", step)
                if metrics_f:
                    metrics_f.close()
                return 0
        epoch += 1
        if val_ld is not None and (epoch % args.eval_every_epochs == 0 or step >= total_steps):
            m = evaluate(model, val_ld, device, cached)
            if rank == 0 and m:
                log(rank, f"eval@step{step}: {m}")
                emit({"step": step, "eval": m, "t": time.time()})
                if m["mAP"] > best_map:
                    best_map = m["mAP"]
                    save(outdir / "best.pt", step)
        if world > 1:
            dist.barrier()
    save(outdir / "last.pt", step)
    if rank == 0:  # file marker: console prints are unreliable under DDP (v1 lesson)
        (outdir / "DONE").write_text(json.dumps({"step": step, "best_map": best_map}))
    log(rank, "DET_TRAIN_DONE")
    if metrics_f:
        metrics_f.close()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
