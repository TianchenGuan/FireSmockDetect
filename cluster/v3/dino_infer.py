"""Pod-side Track B/C detector inference over an image list (one shard/GPU).

Same JSONL contract as rfdetr_infer.py so the offline sweep
(src/eval/eval_detection.py --preds-jsonl) is shared:

    {"image": "<path as listed>", "w": W, "h": H,
     "preds": [[x1n, y1n, x2n, y2n, score], ...], "latency_ms": ...}

Preprocessing matches src/v3/det_data.py exactly (PIL RGB, bilinear letterbox
to WxH on a 114-gray canvas, ImageNet norm); boxes are mapped back to original
pixels via the letterbox scale, then normalized by original W/H. GT is never
read here.

Modes:
  --backbone-dir <hf snapshot> --ckpt <best.pt>     end-to-end (B1/B2/B3; C1 test)
  --feat-root <dir> --embed-dim <C> --ckpt <best.pt> cached features (C1 val)

Usage (per shard):
  python dino_infer.py --backbone-dir /scratch/.../dinov3-vitl16 \
      --ckpt /scratch/.../runs_v3/B1.../best.pt --list list.txt --shard 0/8 \
      --device-index 0 --root-map "<local prefix>=<pod prefix>" \
      --out /tmp/out/shard0.jsonl [--latency-n 50 --latency-out lat0.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-dir", default=None)
    ap.add_argument("--feat-root", default=None)
    ap.add_argument("--embed-dim", type=int, default=None)
    ap.add_argument("--ckpt", required=True, help="trainable-state checkpoint (best.pt)")
    ap.add_argument("--list", required=True)
    ap.add_argument("--shard", default="0/1", help="i/N")
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=736)
    ap.add_argument("--bf16-backbone", action="store_true",
                    help="load backbone weights in bf16 (7B route)")
    ap.add_argument("--lora", type=int, default=0, metavar="R",
                    help="v4: inject LoRA (rank R) on attention q/k/v/o before "
                         "loading a LoRA-trained checkpoint (its lora_A/B keys "
                         "are otherwise 'unexpected'). Must match training.")
    ap.add_argument("--lora-alpha", type=float, default=None,
                    help="LoRA alpha (default 2*R, same as train_det.py)")
    ap.add_argument("--lora-merge", action="store_true",
                    help="after loading, fold W += (alpha/r) * B @ A into the base "
                         "linears and drop the LoRA branch (deployment form; "
                         "identical function, plain-backbone latency)")
    ap.add_argument("--conf-floor", type=float, default=0.02)
    ap.add_argument("--root-map", default=None, help="localprefix=podprefix")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--latency-n", type=int, default=0)
    ap.add_argument("--latency-out", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))            # flat per-job layout
    sys.path.insert(0, str(here.parents[1] / "src/v3"))  # devbox layout
    from det_data import IMAGENET_MEAN, IMAGENET_STD, letterbox_params  # noqa: E402
    from det_modeling import (ViTBackbone, build_detector,  # noqa: E402
                              forward_with_features, load_trainable_state_dict)

    dev = torch.device("cpu" if args.device == "cpu" else
                       f"{args.device}:{args.device_index}")
    cached = args.feat_root is not None
    if cached:
        assert args.embed_dim, "--embed-dim required with --feat-root"
        model = build_detector(None, embed_dim=args.embed_dim,
                               image_size=(args.width, args.height),
                               box_score_thresh=args.conf_floor)
    else:
        # full-unfreeze checkpoints load into a plain backbone (same keys);
        # LoRA checkpoints need the same injection as training first.
        bb = ViTBackbone(args.backbone_dir, frozen=True,
                         dtype=torch.bfloat16 if args.bf16_backbone else None,
                         lora_r=args.lora, lora_alpha=args.lora_alpha)
        model = build_detector(bb, image_size=(args.width, args.height),
                               box_score_thresh=args.conf_floor)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    load_trainable_state_dict(model, ck["model"])
    model.to(dev).eval()
    n_merged = 0
    if args.lora_merge:
        from det_modeling import LoRALinear  # noqa: E402
        import torch.nn as nn
        for name, mod in list(model.named_modules()):
            if not isinstance(mod, LoRALinear):
                continue
            base = mod.base
            with torch.no_grad():
                w = base.weight.float() + (mod.lora_B.float() @ mod.lora_A.float()) * mod.scaling
                # model may already live on the GPU: build the replacement in place
                lin = nn.Linear(base.in_features, base.out_features,
                                bias=base.bias is not None, dtype=base.weight.dtype,
                                device=base.weight.device)
                lin.weight.copy_(w.to(base.weight.dtype))
                if base.bias is not None:
                    lin.bias.copy_(base.bias)
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.rsplit(".", 1)[-1], lin)
            n_merged += 1
        print(f"lora merged into {n_merged} base linears", flush=True)
    n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    print(f"loaded ckpt step={ck.get('step')} best_map={ck.get('best_map')} "
          f"ckpt_keys={len(ck['model'])} lora_params={n_lora}", flush=True)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tw, th = args.width, args.height

    def preprocess(img: Image.Image):
        iw, ih = img.size
        r, nw, nh = letterbox_params(iw, ih, tw, th)
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (tw, th), (114, 114, 114))
        canvas.paste(img, (0, 0))
        x = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255.0
        x = (x - mean) / std
        return x, r

    i, n = (int(x) for x in args.shard.split("/"))
    listed = [l.strip() for l in Path(args.list).read_text().splitlines() if l.strip()]
    if args.max_images:
        listed = listed[: args.max_images]
    mine = listed[i::n]

    # comma-separated "src=dst" pairs so ONE job can eval lists that live under
    # several roots (v4: sdis_val under smoke_data_v2 + v4_evalval sets under
    # v4data). First matching prefix wins; unmatched paths pass through.
    maps: list[tuple[str, str]] = []
    if args.root_map:
        for spec in args.root_map.split(","):
            if not spec:
                continue
            s, d = spec.split("=", 1)
            maps.append((s, d))

    def resolve(p: str) -> str:
        for s, d in maps:
            if p.startswith(s):
                return d + p[len(s):]
        return p

    feat_root = Path(args.feat_root) if cached else None

    def flat_name(listed_path: str) -> str:
        # relpath under the data root == COCO file_name -> flat_file_name
        for s, _ in maps:
            if listed_path.startswith(s):
                return listed_path[len(s):].lstrip("/").replace("/", "__") + ".pt"
        return Path(listed_path).name + ".pt"

    @torch.inference_mode()
    def detect(ip: str, listed_path: str):
        """returns (preds_normalized, timing_ms dict, (w, h)) or None."""
        try:
            img = Image.open(ip).convert("RGB")
        except Exception:
            return None
        iw, ih = img.size
        t0 = time.perf_counter()
        if cached:
            fp = feat_root / flat_name(listed_path)
            fmap = torch.load(fp, map_location="cpu", weights_only=True).float()
            r, _, _ = letterbox_params(iw, ih, tw, th)
            t1 = time.perf_counter()
            _, dets = forward_with_features(model, fmap[None].to(dev), [(th, tw)])
        elif args.bf16_backbone:
            # 7B route: reproduce the precompute+cached-head computation
            # exactly — bf16 backbone forward on bf16 input, feature map
            # rounded to bf16 (as the cache stores it), fp32 head.
            x, r = preprocess(img)
            t1 = time.perf_counter()
            vitb = model.backbone.backbone
            fmap = vitb(x[None].to(dev, torch.bfloat16))
            fmap = fmap.to(torch.bfloat16).float()
            _, dets = forward_with_features(model, fmap, [(th, tw)])
        else:
            x, r = preprocess(img)
            t1 = time.perf_counter()
            dets = model([x.to(dev)])
        d = dets[0]
        preds = []
        for box, score in zip(d["boxes"].cpu(), d["scores"].cpu()):
            if float(score) < args.conf_floor:
                continue
            x1, y1, x2, y2 = [v / r for v in box.tolist()]
            preds.append([round(x1 / iw, 6), round(y1 / ih, 6),
                          round(x2 / iw, 6), round(y2 / ih, 6), round(float(score), 6)])
        t2 = time.perf_counter()
        timing = {"latency_ms": (t2 - t0) * 1000,
                  "preprocess_ms": (t1 - t0) * 1000,
                  "infer_ms": (t2 - t1) * 1000}
        return preds, timing, (iw, ih)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    with out.open("w") as fh:
        for k, listed_path in enumerate(mine):
            res = detect(resolve(listed_path), listed_path)
            if res is None:
                fh.write(json.dumps({"image": listed_path, "error": "unreadable"}) + "\n")
                continue
            preds, timing, (w, h) = res
            fh.write(json.dumps({"image": listed_path, "w": w, "h": h,
                                 "preds": preds,
                                 "latency_ms": timing["latency_ms"]}) + "\n")
            if (k + 1) % 200 == 0:
                el = time.time() - t_start
                print(f"shard {i}: {k+1}/{len(mine)} ({el:.0f}s, "
                      f"{(k+1)/el:.2f} img/s)", flush=True)

    if args.latency_n and args.latency_out:
        # batch-1 latency INCLUDING preprocessing (decode excluded) — same
        # protocol as the rfdetr/yolo measurements in reports/v3/rfdetr_eval.md
        for _ in range(3):
            detect(resolve(mine[0]), mine[0])
        lat, pre, inf = [], [], []
        for listed_path in mine[: args.latency_n]:
            res = detect(resolve(listed_path), listed_path)
            if res is None:
                continue
            _, tm, _ = res
            lat.append(tm["latency_ms"])
            pre.append(tm["preprocess_ms"])
            inf.append(tm["infer_ms"])
        Path(args.latency_out).write_text(json.dumps({
            "device": str(dev), "width": args.width, "height": args.height,
            "n": len(lat),
            "batch1_ms_mean": round(statistics.mean(lat), 2),
            "batch1_ms_median": round(statistics.median(lat), 2),
            "batch1_ms_p90": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 2),
            "batch1_ms_min": round(min(lat), 2), "batch1_ms_max": round(max(lat), 2),
            "preprocess_ms_median": round(statistics.median(pre), 2),
            "infer_ms_median": round(statistics.median(inf), 2),
        }, indent=2))
    el = time.time() - t_start
    print(f"shard {i} DONE {len(mine)} imgs in {el:.0f}s "
          f"({len(mine)/max(el,1):.2f} img/s)", flush=True)


if __name__ == "__main__":
    main()
