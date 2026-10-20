"""Score YOLO test parquets at pre-registered VAL-selected thresholds.

Two regimes (PREREGISTRATION_V4.md):
  source-only      : conf at 5/10/20% neg-image FP on pyro-sdis VAL
  target-adapted   : conf at 1/5% neg-image FP on boreal VAL
For each system the val parquets are swept once with the frozen matching code
to obtain the confs; each test parquet is then scored at exactly those confs.
Usage: python scripts/v4/yolo_test_at_thresholds.py \
   --system S1_A_ctrl=v4a_s2 --system S2_E_ET=v4e_s1 ... \
   --out reports/v4/test/yolo_test_at_preregistered_thresholds.json
system spec = <test_name>=<val_ckpt_tag> ; test parquets are
reports/v4/test/eval_<test_name>_{sdistest,borealtest,foresttest}.raw.parquet,
val parquets reports/v4/eval_<val_tag>_{sdisval,borealval}.raw.parquet
(or reports/eval_<val_tag>_val.raw.parquet for iteration-2 baselines).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.eval_detection import sweep_report
from scripts.v4.compare_matched_fp import interp_at_fp

GRID = [round(x, 4) for x in np.arange(0.01, 0.99 + 1e-9, 0.01)]
ap = argparse.ArgumentParser()
ap.add_argument("--system", action="append", required=True)
ap.add_argument("--out", default="reports/v4/test/yolo_test_at_preregistered_thresholds.json")
a = ap.parse_args()

def sweep(p):
    return sweep_report(pd.read_parquet(p).to_dict("records"), 0.4, conf_grid=GRID)

def conf_at(sw, fp):
    return interp_at_fp(sw["conf_sweep"], fp, lambda s: s["conf"])

def score_at(p, conf):
    rows = pd.read_parquet(p).to_dict("records")
    rep = sweep_report(rows, 0.4, conf_grid=[round(conf, 4)])
    s = rep["conf_sweep"][0]
    return {"conf": round(conf, 4), "recall": s["recall"], "tiny_recall": s["recall_by_size"]["tiny"],
            "small_recall": s["recall_by_size"]["small"], "precision": s["precision"],
            "neg_image_fp_rate": s["neg_image_fp_rate"], "n_images": rep["n_images"],
            "n_neg": rep["n_negative_images"], "n_pos": rep["n_positive_images"]}

out = {"thresholds": {}, "results": {}}
for spec in a.system:
    name, vt = spec.split("=", 1)
    sdis_val = Path(f"reports/v4/eval_{vt}_sdisval.raw.parquet")
    if not sdis_val.exists():
        sdis_val = Path(f"reports/eval_{vt}_val.raw.parquet")
    bor_val = Path(f"reports/v4/eval_{vt}_borealval.raw.parquet")
    th = {}
    sw = sweep(sdis_val)
    th["source_only_sdis_val"] = {f"negfp_{int(f*100)}pct": conf_at(sw, f) for f in (0.05, 0.10, 0.20)}
    if bor_val.exists():
        swb = sweep(bor_val)
        th["target_adapted_boreal_val"] = {f"negfp_{int(f*100)}pct": conf_at(swb, f) for f in (0.01, 0.05)}
    else:
        th["target_adapted_boreal_val"] = None
    out["thresholds"][name] = th
    res = {}
    for tset in ["sdistest", "borealtest", "foresttest"]:
        tp = Path(f"reports/v4/test/eval_{name}_{tset}.raw.parquet")
        if not tp.exists():
            res[tset] = "MISSING"; continue
        res[tset] = {}
        for regime, confs in th.items():
            if not confs: continue
            res[tset][regime] = {k: (score_at(tp, c) if c is not None else "n/a(val FP floor)")
                                 for k, c in confs.items()}
    out["results"][name] = res
    print(name, "thresholds:", json.dumps(th))
Path(a.out).write_text(json.dumps(out, indent=1))
print("saved", a.out)
