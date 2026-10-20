"""Iteration-4 EARLY-WARNING evaluation: four systems, validation only.

Systems (all read the SAME frozen expE features, so one detector pass serves all):
  s1_expE            single-frame detector baseline (per-frame max det conf)
  s1_<ext>           single-frame score of an external detector (v4 YOLO arm,
                     DINO) joined by image basename -- baseline only, no training
  s2_persist_3of5    the deployed persistence rule as a scalar score: the
                     3rd-largest single-frame score in the causal last-5 window
                     (sweeping it traces the exact 3-of-5 ROC); frames with <3
                     frames of history score 0, because the deployed logic
                     cannot confirm there
  t1_<run>           learned single-frame readout (window = current frame only)
  tmp_<run>          learned causal temporal head (ConvGRU, T<=5, valid at T=1)

Sections:
  A  pyro-sdis val, frame level, matched FPR (v3 protocol, extended)
  B  v4 val replay, per source, with truth_level enforced
  C  false-alert EPISODES per hour on continuous negative footage
  D  YOLO + low-frequency DINO sweep policy

Honesty rules enforced in code, not prose:
  * video-level positive sequences (boreal/firesense) may NEVER contribute
    frame-level truth -- only clip-level detection. Enforced by frame_truth.
  * AuSmoke observes no onset, so no latency/onset number is emitted for it.
  * firesense negatives are flagged contaminated for models trained on v4 G
    corpora; the headline FP pool excludes them.

Usage:
  python -m src.v4.trackd_eval_v4 \
      --features-v3 data/v3/features --features-v4 data/v4/features \
      --seq-val-v3 data/v3/sequences/sequences_val.json \
      --seq-val-replay data/v4/sequences/trackd/sequences_v4_val_replay.json \
      --head tmp_v4_s0=models/v4/trackd/heads/v4_temporal_s0/best.pt ... \
      --ext-single v4a=models/v4/evals/v4evalr2/v4a__sdisval.jsonl \
      --dino-sweep b1v3=reports/v3/eval_B1_val.raw.parquet \
      --out-dir reports/v4 --pred-dir models/v4/trackd/preds
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.v3.trackd_temporal import (SeqStore, TemporalHead,  # noqa: E402
                                    WindowDataset, average_precision)
from src.v3.trackd_eval import (episodes, matched_fpr_threshold,  # noqa: E402
                                persistence_score, roc_auc)
from torch.utils.data import DataLoader  # noqa: E402

WIN = 5
TINY_AREA = 0.01
FPR_TARGETS = [0.005, 0.01, 0.02, 0.05, 0.10]
# deployed operating point (src/serve/config.py)
DEP_SINGLE = 0.30
DEP_SEQ = 0.15
DEP_K = 3
COOLDOWN_S = 60.0
# measured batch-1 latency on 1x L40S, fp32 (models/v3/evals/*/latency_gpu.json)
LAT_YOLO_MS = 11.24
LAT_DINO_MS = 156.43


# --------------------------------------------------------------- scoring util
def score_head(ckpt_path: Path, store: SeqStore, device: str):
    """Return (per-frame sigmoid scores in SeqStore.index order, n_params, t1)."""
    model = TemporalHead()
    ck = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    t1 = bool(ck.get("args", {}).get("t1_only", False))
    dl = DataLoader(WindowDataset(store, augment=False, t1_only=t1),
                    batch_size=512, num_workers=4, shuffle=False)
    outs = []
    with torch.no_grad():
        for b in dl:
            o = model(b["feat"].to(device), b["mask"].to(device),
                      b["dt"].to(device), b["cur_score"].to(device))
            outs.append(torch.sigmoid(o).float().cpu().numpy())
    n_par = sum(p.numel() for p in model.parameters())
    return np.concatenate(outs), n_par, t1


def load_ext_scores(path: Path) -> dict:
    """basename -> max detection confidence, from a preds jsonl or raw parquet."""
    out = {}
    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        for im, pr in zip(df["image"], df["preds"]):
            out[Path(str(im)).name] = float(max((p[4] for p in pr), default=0.0))
        return out
    for line in open(path):
        r = json.loads(line)
        if "preds" not in r:
            continue
        out[Path(r["image"]).name] = float(max((p[4] for p in r["preds"]),
                                               default=0.0))
    return out


def split_by_seq(flat: np.ndarray, lens: list) -> list:
    out, off = [], 0
    for n in lens:
        out.append(flat[off:off + n])
        off += n
    assert off == len(flat), (off, len(flat))
    return out


# --------------------------------------------------------------- episode util
def alert_episodes(alert: np.ndarray, ts: np.ndarray, cooldown_s=COOLDOWN_S):
    """Maximal runs of alert-on frames; runs whose gap is < cooldown are one
    episode (the deployed engine suppresses repeat confirmations for 60 s)."""
    runs = episodes(alert.astype(np.int64))
    if not runs:
        return 0, 0
    merged = 1
    for k in range(1, len(runs)):
        gap = float(ts[runs[k][0]] - ts[runs[k - 1][1] - 1])
        if gap >= cooldown_s:
            merged += 1
    return len(runs), merged


def hours_of(ts_list: list) -> float:
    """Wall-clock hours covered by a set of sequences, from their timestamps.
    A single-frame sequence contributes one median-interval frame."""
    tot = 0.0
    for ts in ts_list:
        if len(ts) < 2:
            tot += 1.0
            continue
        d = np.diff(ts.astype(np.float64))
        d = d[(d > 0) & (d <= 600)]
        step = float(np.median(d)) if len(d) else 1.0
        tot += float(ts[-1] - ts[0]) + step  # + one frame of dwell at the end
    return tot / 3600.0


# Sources whose per-frame wall clock is actually known. AuSmoke ships filename
# offset units and its original fps was never recorded (data/v4/sequences/
# REPORT.md), so any hours-based rate for it would be fabricated.
CADENCE_KNOWN_SOURCES = {"boreal", "firesense", "pyrosdis"}


def poisson_rate_ci(k: int, hours: float, alpha: float = 0.05):
    """Exact (Garwood) 95% CI for an episodes-per-hour rate. The negative-footage
    pools here are ~1.5 h, so a bare point estimate would be misleading."""
    from scipy.stats import chi2
    if hours <= 0:
        return None
    lo = 0.0 if k == 0 else chi2.ppf(alpha / 2, 2 * k) / 2 / hours
    hi = chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / 2 / hours
    return [round(float(lo), 2), round(float(hi), 2)]


def thr_for_n_fp(neg_scores: np.ndarray, k: int) -> float:
    """Threshold whose strict-greater rule admits exactly k negative frames."""
    k = int(max(0, min(k, len(neg_scores))))
    srt = np.sort(neg_scores)
    if k <= 0:
        return float(srt[-1])
    if k >= len(srt):
        return float(-np.inf)
    return float(srt[-k - 1])


# --------------------------------------------------------------- section A
def section_a(store, systems, per_seq_scores, report):
    labels = np.concatenate([s["smoke"] for s in store.seqs]).astype(np.int64)
    tiny = np.concatenate([(s["max_gt_area"] > 0) & (s["max_gt_area"] < TINY_AREA)
                           for s in store.seqs])
    clean = []
    for s in store.seqs:
        c = np.ones(len(s["smoke"]), dtype=bool)
        for t in range(len(s["smoke"])):
            c[t] = s["smoke"][max(0, t - (WIN - 1)): t + 1].sum() == 0
        clean.append(c & (s["smoke"] == 0))
    clean_neg = np.concatenate(clean)
    neg, pos = labels == 0, labels == 1
    lens = [len(s["smoke"]) for s in store.seqs]
    lab_seq = [s["smoke"] for s in store.seqs]

    out = {"n_frames": int(len(labels)), "n_pos": int(pos.sum()),
           "n_neg": int(neg.sum()), "n_clean_neg": int(clean_neg.sum()),
           "n_tiny_pos_frames": int((tiny & pos).sum()),
           "fpr_targets": FPR_TARGETS, "systems": {}}
    thr_table = {}
    for name, sc in systems.items():
        e = {"pr_auc": round(average_precision(sc, labels.astype(np.float64)), 4),
             "roc_auc": round(roc_auc(sc, labels), 4), "matched_fpr": {}}
        seq_sc = per_seq_scores[name]
        for pool_name, pool in [("all_neg", neg), ("clean_neg", clean_neg)]:
            for tgt in FPR_TARGETS:
                thr = matched_fpr_threshold(sc[pool], tgt)
                al = sc > thr
                ent = {"threshold": round(float(thr), 5),
                       "achieved_fpr": round(float(al[pool].mean()), 5),
                       "recall_pos_frames": round(float(al[pos].mean()), 4),
                       "recall_tiny_frames": round(float(al[tiny & pos].mean()), 4)}
                lat, det, nep = [], 0, 0
                for ss, ll in zip(seq_sc, lab_seq):
                    a = ss > thr
                    for st, en in episodes(ll.astype(np.int64)):
                        nep += 1
                        hit = np.where(a[st:en])[0]
                        if len(hit):
                            det += 1
                            lat.append(int(hit[0]))
                ent["episodes"] = nep
                ent["episode_detection_rate"] = round(det / max(nep, 1), 4)
                ent["median_onset_latency_frames"] = (float(np.median(lat))
                                                      if lat else None)
                e["matched_fpr"]["%s@%s" % (pool_name, tgt)] = ent
                if pool_name == "all_neg":
                    thr_table.setdefault(name, {})[tgt] = float(thr)
        out["systems"][name] = e

    s1 = systems["s1_expE"]
    s2 = systems["s2_persist_3of5"]
    out["deployed_points"] = {}
    for pname, al in [
            ("confirmed_only(s2>=%.2f)" % DEP_SEQ, s2 >= DEP_SEQ),
            ("confirmed_or_provisional(s2>=%.2f|s1>=%.2f)" % (DEP_SEQ, DEP_SINGLE),
             (s2 >= DEP_SEQ) | (s1 >= DEP_SINGLE))]:
        out["deployed_points"][pname] = {
            "fpr_all_neg": round(float(al[neg].mean()), 5),
            "fpr_clean_neg": round(float(al[clean_neg].mean()), 5),
            "recall_pos_frames": round(float(al[pos].mean()), 4),
            "recall_tiny_frames": round(float(al[tiny & pos].mean()), 4)}
    report["A_sdis_val_frame_level"] = out
    return thr_table, lens


# --------------------------------------------------------------- section B
def section_b(meta, systems_v4, per_seq_v4, thr_table, report):
    """Per-source val-replay results. thr_table: system -> {fpr_target: thr}
    carried over from pyro-sdis val (the only frame-truth FPR pool we have)."""
    res = {"threshold_source": "pyro-sdis val all_neg matched FPR (section A)",
           "sources": {}}
    groups = {}
    for i, m in enumerate(meta):
        groups.setdefault("%s/%s" % (m["source"], m["role"]), []).append(i)

    for gname, idxs in sorted(groups.items()):
        g = {"sequences": len(idxs),
             "frames": int(sum(meta[i]["n_frames"] for i in idxs)),
             "frame_truth": bool(meta[idxs[0]]["frame_truth"]),
             "truth_level": meta[idxs[0]]["truth_level"],
             "parent_units": sorted({meta[i]["parent_unit"] for i in idxs}),
             "cadence_known": meta[idxs[0]]["source"] in CADENCE_KNOWN_SOURCES,
             "wall_clock_h": (round(hours_of([meta[i]["ts"] for i in idxs]), 4)
                              if meta[idxs[0]]["source"] in CADENCE_KNOWN_SOURCES
                              else None),
             "wall_clock_note": (None if meta[idxs[0]]["source"]
                                 in CADENCE_KNOWN_SOURCES else
                                 "original fps never recorded (order units) -> "
                                 "no per-hour rate is emitted for this source"),
             "notes": sorted({n for i in idxs for n in meta[i]["notes"]}),
             "systems": {}}
        for name in systems_v4:
            per = per_seq_v4[name]
            ent = {}
            for tgt, thr in sorted(thr_table.get(name, {}).items()):
                a = [per[i] > thr for i in idxs]
                nfr = sum(len(x) for x in a)
                nalert = int(sum(int(x.sum()) for x in a))
                clip_det = int(sum(1 for x in a if x.any()))
                raw = merged = 0
                for i, x in zip(idxs, a):
                    r, mg = alert_episodes(x, meta[i]["ts"])
                    raw += r
                    merged += mg
                row = {"threshold": round(float(thr), 5),
                       "frame_alert_rate": round(nalert / max(nfr, 1), 4),
                       "clip_alert_rate": round(clip_det / len(idxs), 4),
                       "clips_alerted": clip_det, "clips": len(idxs),
                       "episodes_raw": raw, "episodes_cooldown_merged": merged}
                if g["wall_clock_h"]:
                    row["episodes_per_hour"] = round(merged / g["wall_clock_h"], 2)
                    row["episodes_per_hour_ci95"] = poisson_rate_ci(
                        merged, g["wall_clock_h"])
                if g["frame_truth"] and g["truth_level"] == "frame_boxes":
                    lab = np.concatenate([meta[i]["smoke"] for i in idxs])
                    sc = np.concatenate([per[i] for i in idxs])
                    row["recall_pos_frames"] = round(
                        float((sc > thr)[lab == 1].mean()), 4) \
                        if (lab == 1).any() else None
                ent["matched@fpr%s" % tgt] = row
            g["systems"][name] = ent
        res["sources"][gname] = g

    # deployed discrete point, per source
    for gname, idxs in sorted(groups.items()):
        s1 = per_seq_v4["s1_expE"]
        s2 = per_seq_v4["s2_persist_3of5"]
        raw = merged = 0
        nalert = nfr = clips = 0
        for i in idxs:
            al = (s2[i] >= DEP_SEQ) | (s1[i] >= DEP_SINGLE)
            r, mg = alert_episodes(al, meta[i]["ts"])
            raw += r
            merged += mg
            nalert += int(al.sum())
            nfr += len(al)
            clips += int(al.any())
        h = res["sources"][gname]["wall_clock_h"] or 0.0
        res["sources"][gname]["deployed_confirmed_or_provisional"] = {
            "frame_alert_rate": round(nalert / max(nfr, 1), 4),
            "clip_alert_rate": round(clips / len(idxs), 4),
            "episodes_raw": raw, "episodes_cooldown_merged": merged,
            "episodes_per_hour": round(merged / h, 2) if h > 0 else None,
            "episodes_per_hour_ci95": poisson_rate_ci(merged, h)}
    report["B_v4_val_replay"] = res


# --------------------------------------------------------------- section C
def section_c(store3, seq3_meta, meta4, per_seq3, per_seq4, thr_table, report):
    """False-alert episodes per hour on continuous negative footage."""
    pools = {}
    # pyro-sdis val sequences with NO positive frame at all -> cleanest pool
    allneg = [i for i, s in enumerate(store3.seqs) if s["smoke"].sum() == 0]
    pools["pyrosdis_val_all_neg_sequences"] = {
        "kind": "v3", "idx": allneg, "contaminated": False,
        "note": "pyro-sdis val sequences whose every frame is negative; real "
                "29 s-cadence wall clock; expE never saw the val sites"}
    # boreal non-smoke clips, excluding the train-site clip
    bor = [i for i, m in enumerate(meta4)
           if m["source"] == "boreal" and m["role"] == "negative_footage"]
    bor_val_site = [i for i in bor if "karkkila" not in meta4[i]["parent_unit"]]
    pools["boreal_val_nonsmoke_heinola_only"] = {
        "kind": "v4", "idx": bor_val_site, "contaminated": False,
        "note": "boreal heinola (VAL burn site) non-smoke clips, 1 fps"}
    pools["boreal_nonsmoke_all_incl_train_site"] = {
        "kind": "v4", "idx": bor, "contaminated": False,
        "note": "adds the karkkila clip, whose background is a TRAIN burn site"}
    fs = [i for i, m in enumerate(meta4)
          if m["source"] == "firesense" and m["role"] == "negative_footage"]
    pools["firesense_negatives_CONTAMINATED"] = {
        "kind": "v4", "idx": fs, "contaminated": True,
        "note": "these 292 frames are background negatives inside v4 YOLO "
                "CORPUS_B_G_v1 / C_EGT_v1 / D_adapt_v1 -> optimistic for any "
                "model trained on those corpora. The temporal heads here read "
                "frozen expE (v3) features and expE never saw them, but the "
                "numbers are still flagged."}

    out = {"cooldown_s": COOLDOWN_S, "pools": {}}
    for pname, p in pools.items():
        ts_list = ([store3.seqs[i]["ts"] for i in p["idx"]] if p["kind"] == "v3"
                   else [meta4[i]["ts"] for i in p["idx"]])
        h = hours_of(ts_list)
        entry = {"sequences": len(p["idx"]),
                 "frames": int(sum(len(t) for t in ts_list)),
                 "wall_clock_h": round(h, 4), "contaminated": p["contaminated"],
                 "note": p["note"], "systems": {}}
        per = per_seq3 if p["kind"] == "v3" else per_seq4
        for name in per:
            ent = {}
            for tgt, thr in sorted(thr_table.get(name, {}).items()):
                raw = merged = nalert = nfr = 0
                for i, ts in zip(p["idx"], ts_list):
                    al = per[name][i] > thr
                    r, mg = alert_episodes(al, ts)
                    raw += r
                    merged += mg
                    nalert += int(al.sum())
                    nfr += len(al)
                ent["matched@fpr%s" % tgt] = {
                    "threshold": round(float(thr), 5),
                    "frame_alert_rate": round(nalert / max(nfr, 1), 4),
                    "episodes_raw": raw, "episodes_cooldown_merged": merged,
                    "episodes_per_hour": round(merged / h, 2) if h > 0 else None,
                    "episodes_per_hour_ci95": poisson_rate_ci(merged, h)}
            entry["systems"][name] = ent
        # deployed point (provisional OR confirmed)
        raw = merged = nalert = nfr = 0
        for i, ts in zip(p["idx"], ts_list):
            al = (per["s2_persist_3of5"][i] >= DEP_SEQ) | \
                 (per["s1_expE"][i] >= DEP_SINGLE)
            r, mg = alert_episodes(al, ts)
            raw += r
            merged += mg
            nalert += int(al.sum())
            nfr += len(al)
        entry["deployed_confirmed_or_provisional"] = {
            "frame_alert_rate": round(nalert / max(nfr, 1), 4),
            "episodes_raw": raw, "episodes_cooldown_merged": merged,
            "episodes_per_hour": round(merged / h, 2) if h > 0 else None,
            "episodes_per_hour_ci95": poisson_rate_ci(merged, h)}
        out["pools"][pname] = entry
    report["C_false_alert_episodes_per_hour"] = out


# --------------------------------------------------------------- section D
def section_d(store3, per_seq3, dino_scores, report, tag="dino",
              ns=(5, 10, 20, 35, 70, 140)):
    """YOLO every frame + an independent DINO pass every Nth frame.

    Combined alert at frame t = (yolo_t > thr_y) OR (t swept AND dino_t > thr_d).
    Both thresholds are matched to the SAME per-frame FPR on val negatives, so
    the comparison is at matched FP by construction. Results are averaged over
    all N sweep phases (phase luck removed)."""
    lens = [len(s["smoke"]) for s in store3.seqs]
    names = [np.load(f)["names"] for f in
             sorted((Path(store3.split_dir).glob("*.npz")))] \
        if hasattr(store3, "split_dir") else None
    yolo = per_seq3["s1_expE"]
    dino = dino_scores
    lab = [s["smoke"] for s in store3.seqs]
    ts = [s["ts"] for s in store3.seqs]
    flat_lab = np.concatenate(lab).astype(np.int64)
    neg = flat_lab == 0
    fy = np.concatenate(yolo)
    fd = np.concatenate(dino)

    out = {"latency_ms_per_frame": {"yolo11m_expE_L40S": LAT_YOLO_MS,
                                    "dinov3L_B1_L40S": LAT_DINO_MS,
                                    "cost_ratio": round(LAT_DINO_MS / LAT_YOLO_MS, 2)},
           "n_for_20pct_added_cost": int(np.ceil(LAT_DINO_MS / LAT_YOLO_MS / 0.20)),
           "fpr_targets": FPR_TARGETS, "results": {}}

    for tgt in FPR_TARGETS:
        thr_y = matched_fpr_threshold(fy[neg], tgt)
        thr_d = matched_fpr_threshold(fd[neg], tgt)
        base_det = base_n = 0
        base_lat, base_fp_frames = [], 0
        for sy, ll in zip(yolo, lab):
            a = sy > thr_y
            for st, en in episodes(ll.astype(np.int64)):
                base_n += 1
                hit = np.where(a[st:en])[0]
                if len(hit):
                    base_det += 1
                    base_lat.append(int(hit[0]))
        base_fp_frames = int((fy > thr_y)[neg].sum())
        rows = {}
        for N in ns:
            det = 0
            lat = []
            fp_frames = 0
            n_phases = N
            for ph in range(n_phases):
                for sy, sd, ll in zip(yolo, dino, lab):
                    sw = (np.arange(len(sy)) % N) == (ph % N)
                    a = (sy > thr_y) | (sw & (sd > thr_d))
                    for st, en in episodes(ll.astype(np.int64)):
                        hit = np.where(a[st:en])[0]
                        if len(hit):
                            det += 1
                            lat.append(int(hit[0]))
                    fp_frames += int(a[ll == 0].sum())
            rows["N=%d" % N] = {
                "added_compute_frac": round(LAT_DINO_MS / LAT_YOLO_MS / N, 4),
                "dino_passes_per_100_frames": round(100.0 / N, 2),
                "episode_detection_rate": round(det / max(base_n * n_phases, 1), 4),
                "delta_episode_detection_rate": round(
                    det / max(base_n * n_phases, 1) - base_det / max(base_n, 1), 4),
                "median_latency_frames": float(np.median(lat)) if lat else None,
                "false_alert_frames_per_phase": round(fp_frames / n_phases, 1),
                "added_false_alert_frames_per_phase": round(
                    fp_frames / n_phases - base_fp_frames, 1),
                "added_false_alert_frac_of_negatives": round(
                    (fp_frames / n_phases - base_fp_frames) / max(int(neg.sum()), 1),
                    5),
            }
            # THE decisive comparison: give YOLO alone the SAME false-alert
            # budget the combined policy actually spends, then compare coverage.
            k_eq = int(round(fp_frames / n_phases))
            thr_y_eq = thr_for_n_fp(fy[neg], k_eq)
            det_eq = 0
            lat_eq = []
            for sy, ll in zip(yolo, lab):
                ae = sy > thr_y_eq
                for st, en in episodes(ll.astype(np.int64)):
                    hit = np.where(ae[st:en])[0]
                    if len(hit):
                        det_eq += 1
                        lat_eq.append(int(hit[0]))
            r = rows["N=%d" % N]
            r["yolo_alone_at_same_fp"] = {
                "false_alert_frames": k_eq,
                "threshold": round(float(thr_y_eq), 5),
                "episode_detection_rate": round(det_eq / max(base_n, 1), 4),
                "median_latency_frames": (float(np.median(lat_eq))
                                          if lat_eq else None)}
            r["combo_minus_yolo_at_same_fp"] = round(
                det / max(base_n * n_phases, 1) - det_eq / max(base_n, 1), 4)
            r["combo_beats_yolo_at_matched_fp"] = bool(
                det / max(base_n * n_phases, 1) > det_eq / max(base_n, 1))
            base_med = float(np.median(base_lat)) if base_lat else None
            r["median_latency_delta_frames_vs_yolo_alone"] = (
                None if (r["median_latency_frames"] is None or base_med is None)
                else round(r["median_latency_frames"] - base_med, 2))
        out["results"]["matched_fpr_%s" % tgt] = {
            "thr_yolo": round(float(thr_y), 5), "thr_dino": round(float(thr_d), 5),
            "yolo_alone": {
                "episodes": base_n,
                "episode_detection_rate": round(base_det / max(base_n, 1), 4),
                "median_latency_frames": (float(np.median(base_lat))
                                          if base_lat else None),
                "false_alert_frames": base_fp_frames,
                "n_negative_frames": int(neg.sum())},
            "sweep": rows}
    # how many DINO passes a typical val sequence would even get
    out["sequence_coverage"] = {
        "n_sequences": len(lens),
        "median_len_frames": float(np.median(lens)),
        "frac_sequences_with_zero_dino_pass": {
            "N=%d" % N: round(float(np.mean([l < N for l in lens])), 4)
            for N in ns}}
    report.setdefault("D_yolo_plus_dino_sweep", {})[tag] = out


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-v3", required=True)
    ap.add_argument("--features-v4", default=None)
    ap.add_argument("--seq-val-replay", default=None)
    ap.add_argument("--head", action="append", default=[],
                    help="name=path/to/best.pt (repeatable)")
    ap.add_argument("--ext-single", action="append", default=[],
                    help="name=preds.jsonl|raw.parquet (repeatable)")
    ap.add_argument("--dino-sweep", action="append", default=[],
                    help="name=preds.jsonl|raw.parquet for section D (repeatable)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-dir", default=None)
    a = ap.parse_args()

    report = {"protocol": {
        "selection": "all thresholds and checkpoints chosen on VALIDATION only",
        "fpr_targets": FPR_TARGETS,
        "deployed_point": {"single_frame_alert_conf": DEP_SINGLE,
                           "seq_frame_conf": DEP_SEQ, "seq_hits_required": DEP_K,
                           "seq_window": WIN, "cooldown_s": COOLDOWN_S},
        "test_material": "untouched (pyro-sdis test sites, FIgLib, lockboxes)"}}

    # ---- pyro-sdis val
    v3val = Path(a.features_v3) / "val"
    store3 = SeqStore(v3val)
    store3.split_dir = v3val
    names3 = np.concatenate([np.load(f, allow_pickle=False)["names"]
                             for f in sorted(v3val.glob("*.npz"))])
    lens3 = [len(s["smoke"]) for s in store3.seqs]

    systems, per_seq3, params = {}, {}, {}
    systems["s1_expE"] = np.concatenate([s["score"] for s in store3.seqs])
    per_seq3["s1_expE"] = [s["score"] for s in store3.seqs]
    systems["s2_persist_3of5"] = np.concatenate(
        [persistence_score(s["score"], DEP_K) for s in store3.seqs])
    per_seq3["s2_persist_3of5"] = [persistence_score(s["score"], DEP_K)
                                   for s in store3.seqs]

    for spec in a.ext_single:
        nm, pth = spec.split("=", 1)
        m = load_ext_scores(Path(pth))
        hit = sum(1 for n in names3 if str(n) in m)
        sc = np.array([m.get(str(n), 0.0) for n in names3], dtype=np.float32)
        systems["s1_" + nm] = sc
        per_seq3["s1_" + nm] = split_by_seq(sc, lens3)
        report.setdefault("ext_single_frame_join", {})[nm] = {
            "path": pth, "matched_frames": int(hit), "total_frames": len(names3)}
        assert hit == len(names3), "%s joined only %d/%d frames" % (nm, hit,
                                                                   len(names3))

    for spec in a.head:
        nm, pth = spec.split("=", 1)
        sc, npar, t1 = score_head(Path(pth), store3, a.device)
        key = ("t1_" if t1 else "tmp_") + nm
        systems[key] = sc
        per_seq3[key] = split_by_seq(sc, lens3)
        params[key] = {"n_params": int(npar), "t1_only": t1, "ckpt": pth}
    report["param_counts"] = params

    thr_table, _ = section_a(store3, systems, per_seq3, report)

    # ---- v4 val replay
    if a.features_v4 and a.seq_val_replay:
        v4dir = Path(a.features_v4) / "val_replay"
        store4 = SeqStore(v4dir)
        store4.split_dir = v4dir
        seqmeta = {s["seq_id"]: s for s in
                   json.loads(Path(a.seq_val_replay).read_text())}
        meta4 = []
        for s in store4.seqs:
            m = seqmeta[s["seq_id"]]
            meta4.append({"seq_id": s["seq_id"], "source": m["source"],
                          "role": m["role"], "frame_truth": m["frame_truth"],
                          "truth_level": m["truth_level"],
                          "parent_unit": m["parent_unit"], "notes": m["notes"],
                          "n_frames": len(s["smoke"]), "ts": s["ts"],
                          "smoke": s["smoke"]})
        lens4 = [len(s["smoke"]) for s in store4.seqs]
        sys4, per4 = {}, {}
        per4["s1_expE"] = [s["score"] for s in store4.seqs]
        per4["s2_persist_3of5"] = [persistence_score(s["score"], DEP_K)
                                   for s in store4.seqs]
        sys4["s1_expE"] = np.concatenate(per4["s1_expE"])
        sys4["s2_persist_3of5"] = np.concatenate(per4["s2_persist_3of5"])
        for spec in a.head:
            nm, pth = spec.split("=", 1)
            sc, npar, t1 = score_head(Path(pth), store4, a.device)
            key = ("t1_" if t1 else "tmp_") + nm
            sys4[key] = sc
            per4[key] = split_by_seq(sc, lens4)
        section_b(meta4, sys4, per4, thr_table, report)
        section_c(store3, None, meta4, per_seq3, per4, thr_table, report)
        if a.pred_dir:
            pd_ = Path(a.pred_dir)
            pd_.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pd_ / "preds_val_replay_v4.npz",
                seq_id=np.array([m["seq_id"] for m in meta4]),
                n_frames=np.array(lens4),
                **{k: np.concatenate(v) for k, v in per4.items()})
    else:
        section_c(store3, None, [], per_seq3, {}, thr_table, report)

    # ---- section D
    for spec in a.dino_sweep:
        nm, pth = spec.split("=", 1)
        m = load_ext_scores(Path(pth))
        hit = sum(1 for n in names3 if str(n) in m)
        assert hit == len(names3), "dino joined only %d/%d" % (hit, len(names3))
        dsc = np.array([m.get(str(n), 0.0) for n in names3], dtype=np.float32)
        report.setdefault("D_dino_sources", {})[nm] = {
            "path": pth, "matched_frames": int(hit)}
        section_d(store3, per_seq3, split_by_seq(dsc, lens3), report, tag=nm)

    if a.pred_dir:
        pd_ = Path(a.pred_dir)
        pd_.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            pd_ / "preds_sdis_val.npz",
            image_name=names3, label=np.concatenate([s["smoke"] for s in store3.seqs]),
            seq_id=np.array([s["seq_id"] for s in store3.seqs]),
            n_frames=np.array(lens3), **systems)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print("wrote", a.out)

    A = report["A_sdis_val_frame_level"]
    print("\n=== A. pyro-sdis val frame level (all_neg pool) ===")
    print("| system | params | PR-AUC | ROC-AUC | R@FPR1% | R@FPR2% | R@FPR5% |")
    print("|---|---|---|---|---|---|---|")
    for n, e in A["systems"].items():
        m = e["matched_fpr"]
        print("| %s | %s | %s | %s | %s | %s | %s |" % (
            n, params.get(n, {}).get("n_params", "-"), e["pr_auc"], e["roc_auc"],
            m["all_neg@0.01"]["recall_pos_frames"],
            m["all_neg@0.02"]["recall_pos_frames"],
            m["all_neg@0.05"]["recall_pos_frames"]))


if __name__ == "__main__":
    main()
