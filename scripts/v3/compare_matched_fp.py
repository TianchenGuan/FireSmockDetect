"""Matched-false-positive comparison across models on one frozen split.

Confidence thresholds are not comparable across architectures (RF-DETR's
sigmoid-scored queries and YOLO's objectness*class scores have different
distributions), so every cross-model claim in reports/v3/rfdetr_eval.md is made
at MATCHED negative-image false-positive rate: each model's curve is swept on a
fine confidence grid with the identical matching code
(src/eval/eval_detection.sweep_report, IoU>=0.4) and then linearly interpolated
to the requested FP levels.

Inputs are the .raw.parquet files written by eval_detection.py, so no inference
runs here.

Usage:
  python scripts/v3/compare_matched_fp.py \
      --model "rfdetrL=reports/v3/eval_rfdetrL_e48_val.raw.parquet" \
      --model "expE_y11m=reports/eval_expE_val.raw.parquet" \
      --model "expB_y11s=reports/eval_expB_val.raw.parquet" \
      --out reports/v3/matched_fp_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.eval_detection import sweep_report  # noqa: E402

FP_TARGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def curve(parquet: Path, match_iou: float, grid: list[float]) -> dict:
    rows = pd.read_parquet(parquet).to_dict("records")
    rep = sweep_report(rows, match_iou, conf_grid=grid)
    return rep


def interp_at_fp(sweep: list[dict], target: float, key) -> float | None:
    """Interpolate a metric at a target negative-image FP rate.

    negFP is monotonically non-increasing in confidence; find the bracketing
    pair and linearly interpolate. Returns None if the target is outside the
    model's achievable FP range on this split.
    """
    pts = [(s["neg_image_fp_rate"], key(s)) for s in sweep
           if s["neg_image_fp_rate"] is not None and key(s) is not None]
    fps = [p[0] for p in pts]
    if target > max(fps) or target < min(fps):
        return None
    for (f1, v1), (f2, v2) in zip(pts, pts[1:]):
        lo, hi = min(f1, f2), max(f1, f2)
        if lo <= target <= hi:
            if f1 == f2:
                return round(float(v1), 4)
            w = (target - f1) / (f2 - f1)
            return round(float(v1 + w * (v2 - v1)), 4)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="label=path.raw.parquet")
    ap.add_argument("--match-iou", type=float, default=0.4)
    ap.add_argument("--grid-step", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    grid = [round(float(x), 4) for x in
            np.arange(args.grid_step, 0.99 + 1e-9, args.grid_step)]
    models = {}
    for spec in args.model:
        label, path = spec.split("=", 1)
        rep = curve(Path(path), args.match_iou, grid)
        models[label] = rep
        print(f"{label}: {rep['n_images']} imgs "
              f"({rep['n_negative_images']} neg), curve pts {len(rep['conf_sweep'])}",
              flush=True)

    table = {}
    for label, rep in models.items():
        sw = rep["conf_sweep"]
        table[label] = {}
        for t in FP_TARGETS:
            table[label][f"fp{t:.2f}"] = {
                "conf": interp_at_fp(sw, t, lambda s: s["conf"]),
                "recall": interp_at_fp(sw, t, lambda s: s["recall"]),
                "precision": interp_at_fp(sw, t, lambda s: s["precision"]),
                "tiny_recall": interp_at_fp(sw, t, lambda s: s["recall_by_size"]["tiny"]),
                "small_recall": interp_at_fp(sw, t, lambda s: s["recall_by_size"]["small"]),
            }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "match_iou": args.match_iou, "fp_targets": FP_TARGETS,
        "models": {k: {"n_images": v["n_images"],
                       "n_negative_images": v["n_negative_images"],
                       "n_positive_images": v["n_positive_images"],
                       "curve": v["conf_sweep"]} for k, v in models.items()},
        "matched_fp": table}, indent=2))

    # markdown-ish console table for the report
    hdr = f"{'FP(neg-img)':>11} | " + " | ".join(
        f"{l:>28}" for l in table)
    print(hdr)
    print(f"{'':>11} | " + " | ".join(f"{'conf  R    tinyR  P':>28}" for _ in table))
    for t in FP_TARGETS:
        cells = []
        for label in table:
            d = table[label][f"fp{t:.2f}"]
            if d["recall"] is None:
                cells.append(f"{'n/a':>28}")
            else:
                cells.append(f"{d['conf']:>5.2f} {d['recall']:>5.3f} "
                             f"{d['tiny_recall']:>5.3f} {d['precision']:>5.3f}    ")
        print(f"{t:>11.2f} | " + " | ".join(cells))
    print("saved:", out)


if __name__ == "__main__":
    main()
