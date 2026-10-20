"""TRANSFER_RESULTS_V4.csv from saved matched-FP / recall-at-conf JSONs.

Generated from artifacts, never typed. One row per (system, valset, operating
point). Re-run whenever new eval JSONs land (seeds, DINO, D_adapt, test).

Usage: python scripts/v4/make_transfer_csv.py [--glob 'reports/v4/matched_fp_*.json'] \
         --out reports/v4/TRANSFER_RESULTS_V4.csv
"""
import argparse, csv, glob, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default="reports/v4/matched_fp_*.json")
ap.add_argument("--out", default="reports/v4/TRANSFER_RESULTS_V4.csv")
a = ap.parse_args()

rows = []
for f in sorted(glob.glob(a.glob)):
    d = json.load(open(f))
    valset = Path(f).stem.replace("matched_fp_", "")
    for sysname, m in d["models"].items():
        base = dict(system=sysname, split=valset, n_images=m["n_images"],
                    n_pos=m["n_positive_images"], n_neg=m["n_negative_images"],
                    source_json=f)
        if d.get("matched_fp"):
            for op, v in d["matched_fp"][sysname].items():
                rows.append({**base, "operating_point": f"negFP={op[2:]}",
                             "regime": "matched-FP",
                             "conf": v["conf"], "recall": v["recall"],
                             "tiny_recall": v["tiny_recall"],
                             "small_recall": v["small_recall"],
                             "precision": v["precision"]})
        else:  # positives-only split
            for ck in ["conf0.10", "conf0.20", "conf0.30", "conf0.50"]:
                v = d["recall_at_conf"][sysname].get(ck)
                if v:
                    rows.append({**base, "operating_point": ck, "regime": "recall-at-conf",
                                 "conf": float(ck[4:]), "recall": v["recall"],
                                 "tiny_recall": v["tiny_recall"],
                                 "small_recall": v["small_recall"],
                                 "precision": v["precision"]})
cols = ["system", "split", "regime", "operating_point", "conf", "recall",
        "tiny_recall", "small_recall", "precision", "n_images", "n_pos", "n_neg",
        "source_json"]
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows -> {a.out}")
