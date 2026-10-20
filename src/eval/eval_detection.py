"""Image-level detection evaluation on a frozen split.

Outputs, per model/config:
  - PR points across confidence sweep (custom matching, IoU>=0.4 default —
    smoke extent is fuzzy; strict IoU understates useful detections)
  - recall by GT box size bucket: tiny (<0.1% img area), small (0.1-1%),
    medium (1-5%), large (>5%)
  - image-level false-positive rate on annotated-negative images
  - threshold table for the recall/false-alert tradeoff (selected on VAL only)

Usage:
  python -m src.eval.eval_detection --data-yaml <split.yaml> --split test \
    --weights <best.pt> --imgsz 1152 [--tile-size 0] --out reports/eval_<tag>.json

  # non-default backend (src/serve/backends.py), e.g. iteration-3 RF-DETR:
  python -m src.eval.eval_detection ... --backend rfdetr --weights <ckpt.pth>

  # predictions computed elsewhere (GPU cluster job) -- same matching, sweep,
  # size buckets and .raw.parquet, zero inference here:
  python -m src.eval.eval_detection --data-yaml <split.yaml> --split val \
    --preds-jsonl models/v3/evals/<name>/sdis_val_preds.jsonl --out ...
Ground truth is always read locally from the frozen YOLO label files, so the
offline path is identical to the in-process path apart from where the boxes
came from.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.serve.detector import SmokeDetector  # noqa: E402

SIZE_BUCKETS = [("tiny", 0.0, 0.001), ("small", 0.001, 0.01),
                ("medium", 0.01, 0.05), ("large", 0.05, 1.01)]
CONF_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]


def load_gt(img_path: Path, smoke_ids: set):
    # ultralytics convention: labels path mirrors images path
    lp = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    boxes = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            p = line.split()
            if len(p) >= 5 and int(float(p[0])) in smoke_ids:
                cx, cy, w, h = map(float, p[1:5])
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, w * h])
    return boxes  # normalized xyxy + area_frac


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def sweep_report(rows: list[dict], match_iou: float,
                 conf_grid: list[float] | None = None) -> dict:
    """Confidence sweep + size-bucket recall + negative-image FP rate.

    Split out of main() so the same matching code serves in-process inference
    and predictions computed elsewhere (cluster jobs); rows are
    {"image", "gt": [x1n,y1n,x2n,y2n,area_frac], "preds": [x1n,y1n,x2n,y2n,score]}.
    conf_grid defaults to the frozen CONF_GRID (reported numbers); a finer grid
    is used by matched-false-positive comparisons across models.
    """
    report = {"n_images": len(rows),
              "n_positive_images": sum(1 for r in rows if len(r["gt"])),
              "n_negative_images": sum(1 for r in rows if not len(r["gt"])),
              "match_iou": match_iou, "conf_sweep": []}
    for conf in (CONF_GRID if conf_grid is None else conf_grid):
        tp = fp = fn = 0
        neg_fp_images = 0
        bucket_tp = {b[0]: 0 for b in SIZE_BUCKETS}
        bucket_n = {b[0]: 0 for b in SIZE_BUCKETS}
        for r in rows:
            preds = sorted([p for p in r["preds"] if p[4] >= conf],
                           key=lambda p: -p[4])
            matched_gt = set()
            for p in preds:
                best_j, best_i = -1, 0.0
                for j, g in enumerate(r["gt"]):
                    if j in matched_gt:
                        continue
                    v = iou(p[:4], g[:4])
                    if v > best_i:
                        best_j, best_i = j, v
                if best_i >= match_iou:
                    matched_gt.add(best_j)
                    tp += 1
                else:
                    fp += 1
            fn += len(r["gt"]) - len(matched_gt)
            if not len(r["gt"]) and len(preds):
                neg_fp_images += 1
            for j, g in enumerate(r["gt"]):
                for name, lo, hi in SIZE_BUCKETS:
                    if lo <= g[4] < hi:
                        bucket_n[name] += 1
                        if j in matched_gt:
                            bucket_tp[name] += 1
        n_neg = report["n_negative_images"]
        report["conf_sweep"].append({
            "conf": conf,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "neg_image_fp_rate": round(neg_fp_images / n_neg, 4) if n_neg else None,
            "recall_by_size": {n: (round(bucket_tp[n] / bucket_n[n], 4)
                                   if bucket_n[n] else None) for n in bucket_tp},
            "gt_by_size": bucket_n,
        })
    return report


def load_preds_jsonl(path: Path) -> dict[str, list]:
    """{image_path: [[x1n,y1n,x2n,y2n,score], ...]} from a cluster inference run."""
    preds: dict[str, list] = {}
    n_err = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "error" in rec:
            n_err += 1
            continue
        preds[rec["image"]] = rec.get("preds", [])
    if n_err:
        print(f"warning: {n_err} unreadable images in {path}")
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", default=None,
                    help="required unless --preds-jsonl is given")
    ap.add_argument("--backend", default="yolo",
                    help="src/serve/backends.py backend name (default yolo)")
    ap.add_argument("--imgsz", type=int, default=1152)
    ap.add_argument("--tile-size", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--match-iou", type=float, default=0.4)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--preds-jsonl", default=None,
                    help="use predictions from a cluster inference run instead of "
                         "running the model here (same matching/sweep/parquet)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.weights and not args.preds_jsonl:
        ap.error("--weights is required unless --preds-jsonl is given")

    dy = yaml.safe_load(Path(args.data_yaml).read_text())
    base = Path(dy.get("path", Path(args.data_yaml).parent))
    split_src = dy[args.split]
    split_path = Path(split_src) if str(split_src).startswith("/") else base / split_src
    names = dy["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    smoke_ids = {int(k) for k, v in names.items() if str(v).lower() == "smoke"} or {0}

    if split_path.suffix == ".txt":       # ultralytics list-file split
        imgs = sorted(Path(l.strip()) for l in split_path.read_text().splitlines()
                      if l.strip())
    else:
        imgs = sorted([p for p in split_path.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.max_images:
        rng = np.random.default_rng(0)
        imgs = [imgs[i] for i in sorted(rng.choice(len(imgs), args.max_images, replace=False))]

    rows = []
    if args.preds_jsonl:
        # predictions from a cluster inference run; GT still read locally
        pred_map = load_preds_jsonl(Path(args.preds_jsonl))
        missing = 0
        for ip in imgs:
            key = str(ip)
            if key not in pred_map:
                missing += 1
                continue
            rows.append({"image": key, "gt": load_gt(ip, smoke_ids),
                         "preds": pred_map[key]})
        if missing:
            print(f"warning: {missing}/{len(imgs)} listed images had no predictions")
        print(f"loaded {len(rows)} rows from {args.preds_jsonl}")
    else:
        from src.serve.backends import load_backend  # noqa: PLC0415
        det = (SmokeDetector(args.weights, device=args.device, imgsz=args.imgsz,
                             tile_size=args.tile_size, conf_floor=0.02)
               if args.backend == "yolo" else
               load_backend(args.backend, weights_path=args.weights,
                            device=args.device, imgsz=args.imgsz,
                            tile_size=args.tile_size, conf_floor=0.02))
        for i, ip in enumerate(imgs):
            img = cv2.imread(str(ip))
            if img is None:
                continue
            h, w = img.shape[:2]
            gt = load_gt(ip, smoke_ids)
            dets, _ = det.detect(img)
            preds = [[d.x1 / w, d.y1 / h, d.x2 / w, d.y2 / h, d.score] for d in dets
                     if d.cls_id in smoke_ids or len(det.class_names) == 1]
            rows.append({"image": str(ip), "gt": gt, "preds": preds})
            if (i + 1) % 200 == 0:
                print(f"{i+1}/{len(imgs)}", flush=True)

    report = sweep_report(rows, args.match_iou)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "weights": args.weights, "backend": args.backend,
        "preds_jsonl": args.preds_jsonl,
        "data_yaml": args.data_yaml, "split": args.split,
        "imgsz": args.imgsz, "tile_size": args.tile_size, **report}, indent=2))
    # raw per-image results so threshold work never needs re-inference
    pd.DataFrame(rows).to_parquet(out.with_suffix(".raw.parquet"))
    print(json.dumps(report["conf_sweep"], indent=2)[:2000])
    print("saved:", out)


if __name__ == "__main__":
    main()
