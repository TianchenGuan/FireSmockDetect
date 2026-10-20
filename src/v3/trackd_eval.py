"""Track D val comparison: (1) single-frame expE score, (2) deployed 3-of-5
persistence, (3) learned causal temporal head — image-level, at matched FPR
on val negative frames.

s2 (persistence) as a scalar score: the 3rd-largest single-frame score in the
causal last-5 window (sweeping a threshold over this order statistic traces
the exact 3-of-5 ROC); frames with <3 frames of history score 0 (deployed
logic cannot confirm there). Deployed operating point (seq 0.15 3-of-5 OR
single-frame 0.30) reported as a discrete point.

FPR pools: A = all val neg frames; B = "clean" negs (no positive GT frame in
the past-5 window) — pool A penalizes temporal smoothing on smoke decay,
pool B does not (both reported).

Usage:
  python -m src.v3.trackd_eval --features data/v3/features/ \
      --head models/v3/trackd/<run>/best.pt --out reports/v3/trackD_eval_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v3.trackd_temporal import (SeqStore, TemporalHead, WindowDataset,  # noqa: E402
                                    average_precision)
from torch.utils.data import DataLoader  # noqa: E402

WIN = 5
TINY_AREA = 0.01
FPR_TARGETS = [0.005, 0.01, 0.02, 0.05, 0.10]


def persistence_score(scores: np.ndarray, k: int = 3) -> np.ndarray:
    out = np.zeros_like(scores)
    for t in range(len(scores)):
        w = scores[max(0, t - (WIN - 1)): t + 1]
        out[t] = np.sort(w)[-k] if len(w) >= k else 0.0
    return out


def roc_auc(scores, labels):
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    r = np.arange(1, len(scores) + 1, dtype=np.float64)
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks[order] = r
    n_pos, n_neg = labels.sum(), (1 - labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def matched_fpr_threshold(neg_scores: np.ndarray, target: float) -> float:
    """Smallest threshold thr s.t. mean(neg > thr) <= target."""
    k = int(np.floor(target * len(neg_scores)))
    if k <= 0:
        return float(neg_scores.max())
    return float(np.sort(neg_scores)[-k - 1] if k < len(neg_scores) else -np.inf)


def episodes(labels: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of consecutive positive frames -> (start, end_exclusive)."""
    eps, t = [], 0
    while t < len(labels):
        if labels[t] == 1:
            e = t
            while t < len(labels) and labels[t] == 1:
                t += 1
            eps.append((e, t))
        else:
            t += 1
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="dir containing val/")
    ap.add_argument("--head", default=None, help="temporal head best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--xcheck-parquet", default=None,
                    help="reports/eval_expE_val.raw.parquet cross-check")
    args = ap.parse_args()

    store = SeqStore(Path(args.features) / "val")
    per_seq = []
    for s in store.seqs:
        d = {"s1": s["score"], "label": s["smoke"], "ts": s["ts"],
             "tiny": (s["max_gt_area"] > 0) & (s["max_gt_area"] < TINY_AREA)}
        d["s2"] = persistence_score(s["score"])
        clean = np.ones(len(s["smoke"]), dtype=bool)
        for t in range(len(s["smoke"])):
            w = s["smoke"][max(0, t - (WIN - 1)): t + 1]
            clean[t] = (w.sum() == 0)
        d["clean_neg"] = clean & (s["smoke"] == 0)
        per_seq.append(d)

    systems = {"s1_single_frame": np.concatenate([d["s1"] for d in per_seq]),
               "s2_persistence_3of5": np.concatenate([d["s2"] for d in per_seq])}

    if args.head:
        model = TemporalHead()
        ck = torch.load(args.head, map_location="cpu")
        model.load_state_dict(ck["model"])
        model.to(args.device).eval()
        t1_only = bool(ck.get("args", {}).get("t1_only", False))
        if t1_only:
            print("NOTE: head was trained t1-only (ablation control)")
        dl = DataLoader(WindowDataset(store, augment=False, t1_only=t1_only),
                        batch_size=512, num_workers=4, shuffle=False)
        outs = []
        with torch.no_grad():
            for b in dl:
                o = model(b["feat"].to(args.device), b["mask"].to(args.device),
                          b["dt"].to(args.device), b["cur_score"].to(args.device))
                outs.append(torch.sigmoid(o).float().cpu().numpy())
        s3 = np.concatenate(outs)
        # index order of SeqStore == concatenation order of per_seq
        systems["s3_learned_head" + ("_t1only" if t1_only else "")] = s3
        off = 0
        for d in per_seq:
            d["s3"] = s3[off: off + len(d["s1"])]
            off += len(d["s1"])
        assert off == len(s3)

    labels = np.concatenate([d["label"] for d in per_seq]).astype(np.int64)
    tiny = np.concatenate([d["tiny"] for d in per_seq])
    clean_neg = np.concatenate([d["clean_neg"] for d in per_seq])
    neg = labels == 0
    pos = labels == 1
    print(f"val frames {len(labels)}: pos {pos.sum()} neg {neg.sum()} "
          f"(clean neg {clean_neg.sum()}), tiny-GT pos frames {(tiny & pos).sum()}")

    if args.xcheck_parquet:
        import pandas as pd
        r = pd.read_parquet(args.xcheck_parquet)
        pmax = {Path(p).name: (max((x[4] for x in pr), default=0.0))
                for p, pr in zip(r["image"], r["preds"])}
        names = np.concatenate([np.load(f)["names"] for f in
                                sorted((Path(args.features) / "val").glob("*.npz"))])
        s1 = systems["s1_single_frame"]
        diffs = [abs(s1[i] - pmax.get(str(names[i]), np.nan))
                 for i in range(0, len(names), 37)]
        print(f"xcheck vs {args.xcheck_parquet}: n={len(diffs)} "
              f"max|diff|={np.nanmax(diffs):.4f} median={np.nanmedian(diffs):.4f}")

    report = {"n_frames": int(len(labels)), "n_pos": int(pos.sum()),
              "n_neg": int(neg.sum()), "n_clean_neg": int(clean_neg.sum()),
              "n_tiny_pos_frames": int((tiny & pos).sum()),
              "fpr_targets": FPR_TARGETS, "systems": {}}

    for name, sc in systems.items():
        entry = {"pr_auc": round(average_precision(sc, labels.astype(np.float64)), 4),
                 "roc_auc": round(roc_auc(sc, labels), 4),
                 "matched_fpr": {}}
        for pool_name, pool_mask in [("all_neg", neg), ("clean_neg", clean_neg)]:
            for tgt in FPR_TARGETS:
                thr = matched_fpr_threshold(sc[pool_mask], tgt)
                alert = sc > thr
                ent = {
                    "threshold": round(thr, 5),
                    "achieved_fpr": round(float(alert[pool_mask].mean()), 5),
                    "recall_pos_frames": round(float(alert[pos].mean()), 4),
                    "recall_tiny_frames": round(float(alert[tiny & pos].mean()), 4),
                }
                # onset-episode latency at this threshold
                lat, det, n_ep = [], 0, 0
                key = name.split("_")[0]
                for d in per_seq:
                    if key not in d:
                        continue
                    a = d[key] > thr
                    for e, f in episodes(d["label"]):
                        n_ep += 1
                        hit = np.where(a[e:f])[0]
                        if len(hit):
                            det += 1
                            lat.append(int(hit[0]))
                ent["episodes"] = n_ep
                ent["episode_detection_rate"] = round(det / max(n_ep, 1), 4)
                ent["median_onset_latency_frames"] = (
                    float(np.median(lat)) if lat else None)
                entry["matched_fpr"][f"{pool_name}@{tgt}"] = ent
        report["systems"][name] = entry

    # deployed discrete operating point
    s1 = systems["s1_single_frame"]
    s2 = systems["s2_persistence_3of5"]
    for pname, alert in [
            ("deployed_confirmed_only(s2>=0.15)", s2 >= 0.15),
            ("deployed_confirmed_or_provisional(s2>=0.15|s1>=0.30)",
             (s2 >= 0.15) | (s1 >= 0.30))]:
        report.setdefault("deployed_points", {})[pname] = {
            "fpr_all_neg": round(float(alert[neg].mean()), 5),
            "fpr_clean_neg": round(float(alert[clean_neg].mean()), 5),
            "recall_pos_frames": round(float(alert[pos].mean()), 4),
            "recall_tiny_frames": round(float(alert[tiny & pos].mean()), 4),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    # compact table
    print("\n| system | PR-AUC | ROC-AUC | R@FPR1% | R@FPR2% | R@FPR5% | tinyR@FPR5% |")
    print("|---|---|---|---|---|---|---|")
    for name, e in report["systems"].items():
        m = e["matched_fpr"]
        print(f"| {name} | {e['pr_auc']} | {e['roc_auc']} | "
              f"{m['all_neg@0.01']['recall_pos_frames']} | "
              f"{m['all_neg@0.02']['recall_pos_frames']} | "
              f"{m['all_neg@0.05']['recall_pos_frames']} | "
              f"{m['all_neg@0.05']['recall_tiny_frames']} |")
    print(json.dumps(report.get("deployed_points", {}), indent=1))


if __name__ == "__main__":
    main()
