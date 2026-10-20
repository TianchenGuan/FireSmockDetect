"""Tile-pathway confidence-offset tuning on the frozen VAL split (CPU devbox).

Iteration-1 tiling was rejected: tile detections at the shared threshold
exploded false alarms. Hypothesis: tiles need their OWN higher threshold.
This script runs ONE inference pass per sampled val image (full pass + tiles,
both raw, plus the deployment-order class-aware NMS merge at conf_floor) and
then sweeps (base_conf x tile_conf_offset) offline:

  - baseline "no tiling"  = raw full-pass detections filtered at base_conf
  - tiled  (conf, offset) = merged dets post-filtered per source
    (full: score >= conf; tile: score >= conf + offset) — exactly the
    SmokeDetector.filter_by_conf pathway.

Metrics reuse the matching logic of src/eval/eval_detection.py
(greedy score-desc matching at IoU>=0.4, size buckets, neg-image FP rate).

VAL ONLY. Never point this at the test split or FIgLib.

Usage:
  python -m src.eval.tile_tuning --weights models/expB_best.pt \
    --list data/yolo/lists/local/sdis_val.txt --imgsz 1280 --tile-size 1024 \
    --sample 2000 --seed 0 --workers 24 --out reports/v2/tile_tuning_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.eval_detection import SIZE_BUCKETS, load_gt  # noqa: E402

CONF_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
OFFSETS = [0.0, 0.1, 0.2, 0.3]

_W = {}  # per-worker state


def _init_worker(weights, imgsz, tile_size, threads):
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(threads)
    from src.serve.detector import SmokeDetector
    _W["cv2"] = cv2
    _W["det"] = SmokeDetector(weights, device="cpu", imgsz=imgsz,
                              tile_size=tile_size, conf_floor=0.02)


def _infer_one(path: str):
    cv2, det = _W["cv2"], _W["det"]
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    gt = load_gt(Path(path), {0})
    full = det._predict_array(img, "full")
    tiles = []
    if det.tile_size and (w > det.tile_size or h > det.tile_size):
        for x1, y1, x2, y2 in det._tiles(w, h):
            tiles += det._predict_array(img[y1:y2, x1:x2], "tile", offset=(x1, y1))
    merged = det._merge(full + tiles) if tiles else list(full)
    def norm(ds):
        return [[d.x1 / w, d.y1 / h, d.x2 / w, d.y2 / h, d.score, d.source]
                for d in ds]
    return {"image": path, "gt": gt, "full_raw": norm(full), "merged": norm(merged)}


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def sweep_point(rows, key, conf, offset, match_iou=0.4):
    """Metrics for one (conf, offset) config over cached per-image preds."""
    tp = fp = fn = 0
    neg_fp_images = n_neg = 0
    bucket_tp = {b[0]: 0 for b in SIZE_BUCKETS}
    bucket_n = {b[0]: 0 for b in SIZE_BUCKETS}
    for r in rows:
        preds = sorted(
            (p for p in r[key]
             if p[4] >= (conf + offset if p[5] == "tile" else conf)),
            key=lambda p: -p[4])
        matched_gt = set()
        for p in preds:
            best_j, best_i = -1, 0.0
            for j, g in enumerate(r["gt"]):
                if j in matched_gt:
                    continue
                v = _iou(p[:4], g[:4])
                if v > best_i:
                    best_j, best_i = j, v
            if best_i >= match_iou:
                matched_gt.add(best_j)
                tp += 1
            else:
                fp += 1
        fn += len(r["gt"]) - len(matched_gt)
        if not r["gt"]:
            n_neg += 1
            if preds:
                neg_fp_images += 1
        for j, g in enumerate(r["gt"]):
            for name, lo, hi in SIZE_BUCKETS:
                if lo <= g[4] < hi:
                    bucket_n[name] += 1
                    if j in matched_gt:
                        bucket_tp[name] += 1
    return {
        "conf": conf, "tile_conf_offset": offset,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "tiny_recall": (round(bucket_tp["tiny"] / bucket_n["tiny"], 4)
                        if bucket_n["tiny"] else None),
        "neg_image_fp_rate": round(neg_fp_images / n_neg, 4) if n_neg else None,
        "tp": tp, "fp": fp, "fn": fn,
        "recall_by_size": {n: (round(bucket_tp[n] / bucket_n[n], 4)
                               if bucket_n[n] else None) for n in bucket_tp},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/expB_best.pt")
    ap.add_argument("--list", dest="list_file",
                    default="data/yolo/lists/local/sdis_val.txt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads-per-worker", type=int, default=7)
    ap.add_argument("--match-iou", type=float, default=0.4)
    ap.add_argument("--out", default="reports/v2/tile_tuning_val.json")
    args = ap.parse_args()

    if "test" in Path(args.list_file).name or "figlib" in args.list_file.lower():
        raise SystemExit("refusing: tuning is VAL-ONLY")

    imgs = [l.strip() for l in Path(args.list_file).read_text().splitlines()
            if l.strip()]
    if args.sample and args.sample < len(imgs):
        rng = np.random.default_rng(args.seed)
        idx = sorted(rng.choice(len(imgs), args.sample, replace=False))
        imgs = [imgs[i] for i in idx]
    print(f"{len(imgs)} images (seed={args.seed}) | weights={args.weights} "
          f"imgsz={args.imgsz} tile={args.tile_size}", flush=True)

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker,
            initargs=(args.weights, args.imgsz, args.tile_size,
                      args.threads_per_worker)) as ex:
        for i, r in enumerate(ex.map(_infer_one, imgs, chunksize=8)):
            if r is not None:
                rows.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(imgs)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"inference done: {len(rows)} rows in {time.time()-t0:.0f}s", flush=True)

    baseline = [sweep_point(rows, "full_raw", c, 0.0, args.match_iou)
                for c in CONF_GRID]
    tiled = [sweep_point(rows, "merged", c, o, args.match_iou)
             for o in OFFSETS for c in CONF_GRID]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "weights": args.weights, "list": args.list_file, "split": "val",
        "imgsz": args.imgsz, "tile_size": args.tile_size,
        "sample": len(rows), "seed": args.seed, "match_iou": args.match_iou,
        "n_positive_images": sum(1 for r in rows if r["gt"]),
        "n_negative_images": sum(1 for r in rows if not r["gt"]),
        "baseline_no_tiling": baseline, "tiled": tiled,
    }
    out.write_text(json.dumps(report, indent=2))
    import pandas as pd
    # nested lists mix floats with the source tag -> store as JSON strings
    pd.DataFrame({"image": [r["image"] for r in rows],
                  "gt": [json.dumps(r["gt"]) for r in rows],
                  "full_raw": [json.dumps(r["full_raw"]) for r in rows],
                  "merged": [json.dumps(r["merged"]) for r in rows],
                  }).to_parquet(out.with_suffix(".raw.parquet"))
    print("saved:", out, "and", out.with_suffix(".raw.parquet"))

    hdr = f"{'conf':>5} {'off':>4} {'prec':>6} {'rec':>6} {'tiny':>6} {'negFP':>6}"
    print("\n-- no tiling --\n" + hdr)
    for b in baseline:
        print(f"{b['conf']:>5} {'-':>4} {b['precision']:>6} {b['recall']:>6} "
              f"{b['tiny_recall']:>6} {b['neg_image_fp_rate']:>6}")
    print("\n-- tiled (merge then per-source filter) --\n" + hdr)
    for t in tiled:
        print(f"{t['conf']:>5} {t['tile_conf_offset']:>4} {t['precision']:>6} "
              f"{t['recall']:>6} {t['tiny_recall']:>6} {t['neg_image_fp_rate']:>6}")


if __name__ == "__main__":
    main()
