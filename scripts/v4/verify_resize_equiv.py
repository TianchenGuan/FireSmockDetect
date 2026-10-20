"""Check that pre-resizing a frame to its 1280-fitted size on the devbox gives
the SAME frozen-expE features as handing the native frame to the pod extractor.

Rationale: ultralytics LetterBox does exactly one cv2.INTER_LINEAR resize from
native size to the imgsz-fitted size, then pads to a stride multiple. Doing that
same single resize ahead of time makes the pod-side resize a no-op (r==1.0), so
the two paths should agree to JPEG-recompression noise only. This exists so the
~16.5 GB v4 frame upload can be shrunk to ~1.8 GB without hand-waving.

Usage: python scripts/v4/verify_resize_equiv.py [--per-source 3]
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
from src.v3.trackd_features import Extractor  # noqa: E402
from scripts.v4.build_trackd_v4 import STAGE_ROOT, rel_for  # noqa: E402

WEIGHTS = ROOT / "models/trained/expE_mix_y11m_1280_v2/seg05/best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "reports/v4/resize_equiv_check.json"))
    a = ap.parse_args()

    import cv2
    df = pd.read_parquet(ROOT / "data/v4/sequences/sequences_v4.parquet")
    ex = Extractor(str(WEIGHTS), "cpu", 1280)
    rows = []
    for src in ["ausmoke", "boreal", "firesense"]:
        sub = df[df.source == src].iloc[:: max(1, len(df[df.source == src]) //
                                              a.per_source)][: a.per_source]
        for r in sub.itertuples():
            staged = STAGE_ROOT / rel_for(src, r.abs_path)
            if not staged.exists():
                rows.append({"source": src, "frame": Path(r.abs_path).name,
                             "status": "staged_missing"})
                continue
            f_nat, s_nat, _ = ex.frame(cv2.imread(r.abs_path))
            f_pre, s_pre, _ = ex.frame(cv2.imread(str(staged)))
            fn, fp = f_nat.astype(np.float32), f_pre.astype(np.float32)
            denom = float(np.abs(fn).mean()) or 1.0
            rows.append({
                "source": src, "frame": Path(r.abs_path).name,
                "native_shape": list(cv2.imread(r.abs_path).shape[:2]),
                "staged_shape": list(cv2.imread(str(staged)).shape[:2]),
                "score_native": round(float(s_nat), 6),
                "score_prescaled": round(float(s_pre), 6),
                "score_abs_diff": round(abs(float(s_nat) - float(s_pre)), 6),
                "feat_max_abs_diff": round(float(np.abs(fn - fp).max()), 5),
                "feat_rel_mean_abs_diff": round(
                    float(np.abs(fn - fp).mean() / denom), 6),
            })
    ok = [r for r in rows if "score_abs_diff" in r]
    summary = {
        "weights": str(WEIGHTS), "imgsz": 1280, "n_checked": len(ok),
        "max_score_abs_diff": round(max((r["score_abs_diff"] for r in ok),
                                        default=float("nan")), 6),
        "max_feat_rel_mean_abs_diff": round(
            max((r["feat_rel_mean_abs_diff"] for r in ok), default=float("nan")), 6),
        "note": "identical shapes + tiny diffs => the pre-resize is the same "
                "single INTER_LINEAR op the pod letterbox would have done; "
                "residual is JPEG q95 recompression only (passthrough frames "
                "are byte-identical and must show 0.0).",
        "rows": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
