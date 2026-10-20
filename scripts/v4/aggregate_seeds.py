"""3-seed aggregation of matched-FP (or recall-at-conf) results per arm.

Reads per-seed eval parquets reports/v4/eval_<ckpt>_<valset>.raw.parquet,
recomputes the curve with the frozen matching code, and reports mean/std/
min/max per arm at each FP target (or conf for positives-only splits), plus
the pre-registered selection quantities:
  * S2 rule: 3-seed mean boreal-val recall @1% negFP for C_EGT vs E_ET
  * median-seed rule: seed whose sdis-val recall@10% is the median
Usage: python scripts/v4/aggregate_seeds.py --out reports/v4/seed_summary.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.eval_detection import sweep_report
from scripts.v4.compare_matched_fp import interp_at_fp  # noqa

ARMS = {"A_ctrl_E": ["v4a", "v4a_s1", "v4a_s2"],
        "E_ET": ["v4e", "v4e_s1", "v4e_s2"],
        "C_EGT": ["v4c", "v4c_s1", "v4c_s2"],
        "D_curr": ["v4dada", "v4dada_s1", "v4dada_s2"],
        "B_Gonly": ["v4b"]}
SETS = ["sdisval", "borealval", "forestval"]
FP_T = [0.01, 0.02, 0.05, 0.10, 0.20]
CONF_T = [0.10, 0.20, 0.30, 0.50]
GRID = [round(x, 4) for x in np.arange(0.01, 0.99 + 1e-9, 0.01)]

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="reports/v4/seed_summary.json")
a = ap.parse_args()
out = {"arms": {}, "selection": {}}
for arm, cks in ARMS.items():
    out["arms"][arm] = {}
    for vs in SETS:
        per_seed = {}
        for ck in cks:
            p = Path(f"reports/v4/eval_{ck}_{vs}.raw.parquet")
            if not p.exists():
                continue
            rep = sweep_report(pd.read_parquet(p).to_dict("records"), 0.4, conf_grid=GRID)
            sw = rep["conf_sweep"]
            if rep["n_negative_images"] > 0:
                per_seed[ck] = {f"fp{t:.2f}": {"recall": interp_at_fp(sw, t, lambda s: s["recall"]),
                                              "tiny": interp_at_fp(sw, t, lambda s: s["recall_by_size"]["tiny"]),
                                              "conf": interp_at_fp(sw, t, lambda s: s["conf"])} for t in FP_T}
            else:
                bc = {round(s["conf"], 2): s for s in sw}
                per_seed[ck] = {f"conf{c:.2f}": {"recall": bc[c]["recall"], "tiny": bc[c]["recall_by_size"]["tiny"]}
                                for c in CONF_T if c in bc}
        if not per_seed:
            continue
        keys = next(iter(per_seed.values())).keys()
        agg = {}
        for k in keys:
            for m in ["recall", "tiny"]:
                vals = [v[k][m] for v in per_seed.values() if v[k].get(m) is not None]
                if vals:
                    agg[f"{k}.{m}"] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4),
                                       "min": round(min(vals), 4), "max": round(max(vals), 4), "n": len(vals)}
        out["arms"][arm][vs] = {"per_seed": per_seed, "agg": agg}

def g(arm, vs, key):
    return out["arms"].get(arm, {}).get(vs, {}).get("agg", {}).get(key)
sel = out["selection"]
for arm in ["C_EGT", "E_ET"]:
    sel[f"{arm}.boreal_R@1%.mean"] = g(arm, "borealval", "fp0.01.recall")
c, e = sel.get("C_EGT.boreal_R@1%.mean"), sel.get("E_ET.boreal_R@1%.mean")
if c and e:
    diff = c["mean"] - e["mean"]
    sel["S2_pick"] = "E_ET" if abs(diff) < 0.01 else ("C_EGT" if diff > 0 else "E_ET")
    sel["S2_rule"] = f"higher 3-seed mean boreal R@1%; tie <1pt -> E_ET (diff={diff:+.4f})"
for arm in ["A_ctrl_E", "E_ET", "C_EGT", "D_curr"]:
    ps = out["arms"].get(arm, {}).get("sdisval", {}).get("per_seed", {})
    r = {ck: v["fp0.10"]["recall"] for ck, v in ps.items() if v["fp0.10"]["recall"] is not None}
    if len(r) >= 1:
        med = sorted(r.items(), key=lambda kv: kv[1])[len(r) // 2]
        sel[f"{arm}.median_seed(sdis R@10%)"] = {"ckpt": med[0], "recall": med[1], "n_seeds": len(r)}
Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps(sel, indent=1))
for arm in out["arms"]:
    for vs in out["arms"][arm]:
        ag = out["arms"][arm][vs]["agg"]
        k = "fp0.01.recall" if "fp0.01.recall" in ag else "conf0.20.recall"
        k2 = "fp0.10.recall" if "fp0.10.recall" in ag else "conf0.30.recall"
        f = lambda kk: (f"{kk}={ag[kk]['mean']:.3f}±{ag[kk]['std']:.3f}" if kk in ag else f"{kk}=n/a(FP floor)")
        n = ag.get(k, ag.get(k2, {})).get("n", 0)
        print(f"{arm:9} {vs:10} n={n} {f(k)}  {f(k2)}")
