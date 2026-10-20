"""Partial AUC of (tiny-)recall vs negative-image FP from a matched-FP JSON.

This is the single consistent integration used for every pAUC number in
reports/v3/trackBC_results.md: sort curve points by neg-image FP, clip to
[0, fpmax], linearly interpolate the value at fpmax, anchor at (0, 0) when
the model cannot reach FP 0, trapezoid-integrate, normalize by fpmax.
(Values for expE/expB/rfdetrL differ by <=0.02 from the ad-hoc integration
quoted in reports/v3/rfdetr_eval.md; orderings are unchanged.)

Usage:
  python scripts/v3/pauc.py reports/v3/matched_fp_val_trackBC.json [fpmax]
"""
from __future__ import annotations

import json
import sys

import numpy as np


def pauc(curve: list[dict], tiny: bool = False, fpmax: float = 0.5) -> float:
    pts = sorted(((c["neg_image_fp_rate"],
                   c["recall_by_size"]["tiny"] if tiny else c["recall"])
                  for c in curve), key=lambda t: t[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xs2 = [x for x in xs if x <= fpmax]
    ys2 = ys[: len(xs2)]
    if xs2 and xs2[-1] < fpmax:
        hi = [(x, y) for x, y in zip(xs, ys) if x > fpmax]
        if hi:
            x1, y1 = xs2[-1], ys2[-1]
            x2, y2 = hi[0]
            xs2.append(fpmax)
            ys2.append(y1 + (y2 - y1) * (fpmax - x1) / (x2 - x1))
    if xs2[0] > 0:
        xs2 = [0.0] + xs2
        ys2 = [0.0] + ys2
    return float(np.trapezoid(ys2, xs2) / fpmax)


def main():
    path = sys.argv[1]
    fpmax = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    d = json.load(open(path))
    for name, m in d["models"].items():
        print(f"{name:16s} pAUC(R,FP<={fpmax}) {pauc(m['curve'], fpmax=fpmax):.3f}  "
              f"pAUC(tinyR) {pauc(m['curve'], tiny=True, fpmax=fpmax):.3f}")


if __name__ == "__main__":
    main()
