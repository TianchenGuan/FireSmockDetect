"""Score the one-shot DINO TEST sweeps at the PRE-REGISTERED val-selected
thresholds (reports/v4/test/dino_test_thresholds.json): for every run and test
set, recall / tiny-recall / precision / neg-image FP rate at exactly those
confidences, using the identical matching code (sweep_report, IoU>=0.4) on the
.raw.parquet rows written by eval_detection. No selection happens here.

  .venv/bin/python scripts/v4/dino_test_at_thresholds.py \
      --thresholds reports/v4/test/dino_test_thresholds.json \
      --evaldir reports/v4/test --out reports/v4/test/dino_test_at_preregistered_thresholds.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.eval_detection import sweep_report  # noqa: E402

SETS = ("sdis_test", "boreal_test", "forest_test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--evaldir", required=True)
    ap.add_argument("--match-iou", type=float, default=0.4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    thr = json.loads(Path(a.thresholds).read_text())
    out = {"thresholds_source": a.thresholds, "match_iou": a.match_iou, "results": {}}
    for run, t in thr["runs"].items():
        regimes = {"source_only_sdis_val": {k: v["conf"] for k, v in t["sdis_val_source_only"].items()},
                   "target_adapted_boreal_val": {k: v["conf"] for k, v in t["boreal_val_target_adapted"].items()}}
        confs = sorted({c for reg in regimes.values() for c in reg.values() if c is not None})
        out["results"][run] = {}
        for s in SETS:
            pq = Path(a.evaldir) / f"eval_{run}_{s}.raw.parquet"
            if not pq.exists():
                out["results"][run][s] = {"missing": str(pq)}
                continue
            rows = pd.read_parquet(pq).to_dict("records")
            rep = sweep_report(rows, a.match_iou, conf_grid=confs)
            by_conf = {round(c["conf"], 6): c for c in rep["conf_sweep"]}
            res = {"n_images": rep["n_images"], "n_negative_images": rep["n_negative_images"],
                   "n_positive_images": rep["n_positive_images"]}
            for reg, m in regimes.items():
                res[reg] = {}
                for k, c in m.items():
                    if c is None:
                        res[reg][k] = None
                        continue
                    c_ = by_conf[round(c, 6)]
                    res[reg][k] = {"conf": c, "recall": c_["recall"],
                                   "tiny_recall": c_["recall_by_size"]["tiny"],
                                   "small_recall": c_["recall_by_size"]["small"],
                                   "precision": c_["precision"],
                                   "neg_image_fp_rate": c_["neg_image_fp_rate"]}
            out["results"][run][s] = res
            print(f"{run:16s} {s:12s} " + " | ".join(
                f"{k}: R={v['recall']} negFP={v['neg_image_fp_rate']}"
                for k, v in res["source_only_sdis_val"].items() if v))
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("saved:", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
