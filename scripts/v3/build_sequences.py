"""Track D (iteration 3): build REAL chronological camera sequences from
pyro-sdis for temporal-model training/eval.

Rules (frozen):
- TRAIN sequences from TRAIN sites only (brison, serre-de-barre, courmettes,
  valbonne); VAL sequences from VAL sites (cabanelle, marguerite).
  TEST sites are never touched.
- camera identity = {partner}_{site}-{azimuth} (manifest partner + camera_id).
- sort frames by capture datetime; cut a sequence whenever the gap to the
  previous frame exceeds --max-gap-s (default 600 s); never stitch across
  cameras or gaps.
- keep sequences with >= --min-len frames (default 3).
- per-frame label: smoke present = n_boxes > 0 (boxes carried through for
  optional box-conf supervision).

Outputs:
  data/v3/sequences/sequences_{train,val}.json   (one record per sequence)
  data/v3/sequences/audit.json                   (stats used in trackD notes)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SITES = ["brison", "serre-de-barre", "courmettes", "valbonne"]
VAL_SITES = ["cabanelle", "marguerite"]


def label_pattern(flags: list[bool]) -> str:
    """Classify the smoke-presence pattern of a sequence."""
    s = "".join("1" if f else "0" for f in flags)
    if "1" not in s:
        return "all_neg"
    if "0" not in s:
        return "all_pos"
    first1, last1 = s.index("1"), s.rindex("1")
    interior = s[first1:last1 + 1]
    if "0" not in interior:
        # single contiguous positive block
        if first1 > 0 and last1 == len(s) - 1:
            return "appears"          # neg prefix then smoke to the end
        if first1 == 0 and last1 < len(s) - 1:
            return "disappears"       # smoke from start, then gone
        return "appears_disappears"   # neg / pos / neg
    return "flicker"                  # positives interrupted by negatives


def build(split_name: str, df: pd.DataFrame, max_gap_s: float, min_len: int):
    seqs, dropped_short = [], 0
    intervals_kept = []
    for cam, g in df.groupby("cam", sort=True):
        g = g.sort_values(["dt", "image_name"], kind="stable")
        ts = g["dt"].astype("int64").to_numpy() // 10**9
        # split at gaps
        cut = np.where(np.diff(ts) > max_gap_s)[0] + 1
        for chunk in np.split(np.arange(len(g)), cut):
            sub = g.iloc[chunk]
            if len(sub) < min_len:
                dropped_short += 1
                continue
            t = ts[chunk]
            intervals_kept.append(np.diff(t))
            flags = (sub["n_boxes"] > 0).tolist()
            seqs.append({
                "seq_id": f"{split_name}_{cam}_{pd.Timestamp(sub['dt'].iloc[0]).strftime('%Y%m%dT%H%M%S')}",
                "camera": cam,
                "site": sub["site"].iloc[0],
                "split": split_name,
                "n_frames": int(len(sub)),
                "duration_s": int(t[-1] - t[0]),
                "label_pattern": label_pattern(flags),
                "frames": [
                    {
                        "image_name": r.image_name,
                        "image_path": r.image_path,
                        "ts": int(pd.Timestamp(r.dt).timestamp()),
                        "smoke": bool(r.n_boxes > 0),
                        "boxes_xywhn": json.loads(r.boxes),
                    }
                    for r in sub.itertuples()
                ],
            })
    ints = np.concatenate(intervals_kept) if intervals_kept else np.array([])
    return seqs, dropped_short, ints


def hist(vals, edges):
    c, _ = np.histogram(vals, bins=edges)
    return {f"[{edges[i]},{edges[i+1]})": int(c[i]) for i in range(len(c))}


def audit(split_name, seqs, dropped_short, ints, df):
    lens = np.array([s["n_frames"] for s in seqs])
    frames_in_seq = int(lens.sum())
    patterns = Counter(s["label_pattern"] for s in seqs)
    pos_frames = sum(f["smoke"] for s in seqs for f in s["frames"])
    return {
        "split": split_name,
        "sites": sorted(df["site"].unique().tolist()),
        "n_images_in_split": int(len(df)),
        "n_sequences": len(seqs),
        "n_dropped_short_runs": dropped_short,
        "frames_in_sequences": frames_in_seq,
        "frames_coverage_of_split": round(frames_in_seq / len(df), 4),
        "pos_frames_in_sequences": int(pos_frames),
        "neg_frames_in_sequences": int(frames_in_seq - pos_frames),
        "seq_len": {
            "min": int(lens.min()), "median": float(np.median(lens)),
            "p90": float(np.percentile(lens, 90)), "max": int(lens.max()),
            "hist": hist(lens, [3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 10000]),
        },
        "intra_seq_interval_s": {
            "p10": float(np.percentile(ints, 10)),
            "p25": float(np.percentile(ints, 25)),
            "median": float(np.median(ints)),
            "p75": float(np.percentile(ints, 75)),
            "p90": float(np.percentile(ints, 90)),
            "max": float(ints.max()),
            "hist": hist(ints, [0, 5, 15, 45, 90, 180, 300, 601]),
        },
        "label_patterns": dict(patterns),
        "n_cameras": len({s["camera"] for s in seqs}),
        "per_site_sequences": dict(Counter(s["site"] for s in seqs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/pyro_sdis_manifest.parquet"))
    ap.add_argument("--max-gap-s", type=float, default=600.0)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--outdir", default=str(ROOT / "data/v3/sequences"))
    args = ap.parse_args()

    m = pd.read_parquet(args.manifest)
    assert m["name_meta_match"].all() and m["decode_ok"].all()
    m["dt"] = pd.to_datetime(m["datetime"])
    m["cam"] = m["partner"] + "_" + m["camera_id"]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    audits = {"params": {"max_gap_s": args.max_gap_s, "min_len": args.min_len,
                         "train_sites": TRAIN_SITES, "val_sites": VAL_SITES}}
    for split_name, sites in [("train", TRAIN_SITES), ("val", VAL_SITES)]:
        df = m[m["site"].isin(sites)]
        seqs, dropped, ints = build(split_name, df, args.max_gap_s, args.min_len)
        (out / f"sequences_{split_name}.json").write_text(
            json.dumps(seqs, indent=None, separators=(",", ":")))
        audits[split_name] = audit(split_name, seqs, dropped, ints, df)
        print(f"{split_name}: {len(seqs)} sequences, "
              f"{audits[split_name]['frames_in_sequences']} frames "
              f"({audits[split_name]['frames_coverage_of_split']:.1%} of split)")
    (out / "audit.json").write_text(json.dumps(audits, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("per_site_sequences",)}
                      for k, v in audits.items() if k != "params"}, indent=1))


if __name__ == "__main__":
    main()
