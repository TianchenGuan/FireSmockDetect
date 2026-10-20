"""V5 experiment A — FIXED-CAMERA threshold calibration (no training).

For one detector system (per-image predictions already computed on the 7,335
pyro-sdis VAL images), sweep (conf_single, conf_seq) pairs of the deployed
AlertEngine and report, per pair:

  * R_single  — frame recall at conf_single, EXACTLY src.eval.eval_detection
                matching (IoU>=0.4, greedy by score), i.e. the v1-v4 metric.
  * R_eff     — engine-aware box recall: the AlertEngine is replayed over each
                VAL sequence in time order (fresh engine per sequence; sequences
                are separated by >600 s = stale reset); a frame's boxes count at
                conf_seq when the engine is CONFIRMED on that frame, else at
                conf_single. Reduces to R_single when conf_seq == conf_single.
  * episodes per VALID hour (PROTOCOL_V5 §3) on the negative observation set:
      - observation set = maximal runs of consecutive frames WITHOUT GT smoke
        boxes inside each VAL sequence (embedded negatives), fresh engine per
        run (no positive-frame history -> trailing alerts after real smoke are
        NOT counted as false);
      - an episode starts on an alert state (provisional or confirmed) and a
        later alert frame belongs to the same episode iff ts - last_alert_ts <
        cooldown_s (60 s) — i.e. the device had been out of alert for >=
        cooldown; a diagnostic 'strict' count also treats 'watch' as alert;
      - valid time = sum of inter-frame gaps inside a run that are <= 3x the
        SEQUENCE's median cadence; larger gaps are excluded; runs are never
        concatenated; a 1-frame run has zero valid time;
      - primary rate = episodes in TIMED runs (valid time > 0) / valid hours;
        'all' rate additionally counts episodes from isolated 1-frame runs
        over the same hours (upper bound);
      - fraction of valid time in alert = accepted gaps whose starting frame
        is in an alert state / valid time.

Calibration rule (§3): keep R_single >= anchor (deployed expB at 0.30/0.15,
recomputed here from reports/eval_expB_val.raw.parquet with the same code),
then minimise episodes/h; ties -> higher R_eff, higher R_single, lower confs.
An engine-aware variant additionally requires R_eff >= expB's R_eff.

FIgLib-39 is NOT used for selection here (historical set; checked afterwards
by scripts/v5/figlib_check_fixed.py).

Usage:
  python scripts/v5/calibrate_fixed.py --systems all --workers 64
  python scripts/v5/calibrate_fixed.py --systems expB,v4B4_lora_L
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.eval_detection import iou, sweep_report, SIZE_BUCKETS  # noqa: E402
from src.serve.alerting import AlertEngine  # noqa: E402
from src.serve.config import Settings  # noqa: E402

SEQ_FILE = ROOT / "data/v3/sequences/sequences_val.json"
MATCH_IOU = 0.4
ALERT_STATES = ("provisional", "confirmed")
STRICT_STATES = ("watch", "provisional", "confirmed")

SYSTEMS = {
    "expB": dict(parquet="reports/eval_expB_val.raw.parquet", kind="yolo",
                 weights="models/expB_best.pt", note="deployed default (expB-mix-y11s-1280@seg03)"),
    "expE": dict(parquet="reports/eval_expE_val.raw.parquet", kind="yolo",
                 weights="models/trained/expE_mix_y11m_1280_v2/seg05/best.pt", note="v3 S0"),
    "v4a_s2": dict(parquet="reports/v4/eval_v4a_s2_sdisval.raw.parquet", kind="yolo",
                   weights="models/trained/expV4A_ctrl_s2/seg05/best.pt", note="v4 arm A control, median seed"),
    "v4B4_lora_L": dict(parquet="reports/v4/eval_v4B4_lora_L_sdis_val.raw.parquet", kind="dino",
                        weights="models/v4/dino/v4B4_lora_L (cluster ckpt)", note="DINOv3-L LoRA (post-hoc efficiency candidate)",
                        figlib_scores="models/v4/evals/v4B4_lora_L_figlib/figlib39_preds.jsonl",
                        v4_val_point=(0.93, 0.82)),
    "v4B6_lora_Hplus": dict(parquet="reports/v4/eval_v4B6_lora_Hplus_sdis_val.raw.parquet", kind="dino",
                            weights="models/v4/dino/v4B6_lora_Hplus (cluster ckpt)", note="DINOv3-H+ LoRA (v4 pre-registered S4)",
                            figlib_scores="models/v4/evals/v4B6_lora_Hplus_figlib/figlib39_preds.jsonl",
                            v4_val_point=(0.95, 0.88)),
}
ANCHOR_SYSTEM = "expB"
DEPLOYED = (0.30, 0.15)


# --------------------------------------------------------------------------
def build_grid() -> list[float]:
    g = set(round(float(x), 4) for x in np.arange(0.05, 0.95, 0.01))
    g |= set(round(float(x), 4) for x in np.arange(0.95, 0.99, 0.001))
    g |= set(round(float(x), 4) for x in np.arange(0.99, 1.0, 0.0005))
    g |= {0.30, 0.15, 0.93, 0.82, 0.95, 0.88, 0.9881, 0.973, 0.9901, 0.9801}
    return sorted(g)


def engine_cfg(conf_single: float, conf_seq: float, base: Settings) -> SimpleNamespace:
    return SimpleNamespace(
        single_frame_alert_conf=conf_single, seq_frame_conf=conf_seq,
        seq_window=base.seq_window, seq_hits_required=base.seq_hits_required,
        cooldown_s=base.cooldown_s, stale_after_s=base.stale_after_s,
        scene_change_hamming=base.scene_change_hamming)


def match_count(gt, preds, conf, match_iou=MATCH_IOU) -> int:
    """Greedy matching identical to eval_detection.sweep_report (one conf)."""
    ps = sorted([p for p in preds if p[4] >= conf], key=lambda p: -p[4])
    matched = set()
    for p in ps:
        best_j, best_i = -1, 0.0
        for j, g in enumerate(gt):
            if j in matched:
                continue
            v = iou(p[:4], g[:4])
            if v > best_i:
                best_j, best_i = j, v
        if best_i >= match_iou:
            matched.add(best_j)
    return len(matched)


def load_rows(parquet: Path, conf_floor: float):
    df = pd.read_parquet(parquet)
    rows = {}
    for img, gt, preds in zip(df["image"], df["gt"], df["preds"]):
        gt_l = [list(map(float, g)) for g in gt]
        pr_l = [list(map(float, p)) for p in preds]
        mx = max((p[4] for p in pr_l if p[4] >= conf_floor), default=0.0)
        rows[img] = {"image": img, "gt": gt_l, "preds": pr_l, "max": mx}
    return rows


def load_sequences():
    seqs = json.load(open(SEQ_FILE))
    out = []
    for s in seqs:
        ts = [f["ts"] for f in s["frames"]]
        assert ts == sorted(ts), s["seq_id"]
        gaps = np.diff(ts)
        med = float(np.median(gaps)) if len(gaps) else 0.0
        out.append({"seq_id": s["seq_id"], "camera": s["camera"], "site": s["site"],
                    "median_cadence_s": med, "gap_thr_s": 3.0 * med,
                    "frames": [(f["image_path"], int(f["ts"]), bool(f["smoke"])) for f in s["frames"]]})
    return out


def negative_runs(seqs, rows):
    runs = []
    for s in seqs:
        cur = []
        for img, ts, smoke in s["frames"]:
            has_gt = len(rows[img]["gt"]) > 0
            assert has_gt == smoke, img
            if not has_gt:
                cur.append((img, ts))
            elif cur:
                runs.append((s, cur)); cur = []
        if cur:
            runs.append((s, cur))
    return runs


# ------------------------------- worker ------------------------------------
G = {}  # globals for forked workers


def _init(seqs, runs, rows, grid, M, thr_index, n_gt, base_cfg):
    G.update(seqs=seqs, runs=runs, rows=rows, grid=grid, M=M, thr_index=thr_index,
             n_gt=n_gt, base=base_cfg)


def episodes_in_run(states_ts, cooldown, alert_states):
    n = 0; last = None
    for ts, st in states_ts:
        if st in alert_states:
            if last is None or ts - last >= cooldown:
                n += 1
            last = ts
    return n


def eval_pair(conf_single: float, conf_seq: float) -> dict:
    base = G["base"]; rows = G["rows"]; M = G["M"]; thr_index = G["thr_index"]
    cfg = engine_cfg(conf_single, conf_seq, base)
    # --- engine-aware recall over full sequences
    i_single = thr_index[conf_single]; i_seq = thr_index[conf_seq]
    matched = 0
    for s in G["seqs"]:
        eng = AlertEngine(cfg)
        for img, ts, _ in s["frames"]:
            r = rows[img]
            st = eng.update("d", float(ts), r["max"])["state"]
            if r["gt"]:
                matched += int(M[r["idx"], i_seq if st == "confirmed" else i_single])
    r_eff = matched / G["n_gt"]
    # --- false-alert episodes on negative runs
    ep_timed = ep_all = ep_strict_timed = ep_strict_all = 0
    alert_time = 0.0; valid_time = 0.0
    n_alert_frames = 0; n_alert_frames_timed = 0
    per_cam_ep = Counter(); per_cam_valid = defaultdict(float)
    for s, run in G["runs"]:
        eng = AlertEngine(cfg)
        sts = []
        for img, ts in run:
            st = eng.update("d", float(ts), rows[img]["max"])["state"]
            sts.append((ts, st))
        thr = s["gap_thr_s"]
        vt = 0.0; at = 0.0
        for (t0, st0), (t1, _) in zip(sts[:-1], sts[1:]):
            gap = t1 - t0
            if gap <= thr:
                vt += gap
                if st0 in ALERT_STATES:
                    at += gap
        n_al = sum(1 for _, st in sts if st in ALERT_STATES)
        e = episodes_in_run(sts, cfg.cooldown_s, ALERT_STATES)
        es = episodes_in_run(sts, cfg.cooldown_s, STRICT_STATES)
        ep_all += e; ep_strict_all += es; n_alert_frames += n_al
        if vt > 0:
            ep_timed += e; ep_strict_timed += es; n_alert_frames_timed += n_al
            valid_time += vt; alert_time += at
            per_cam_ep[s["camera"]] += e; per_cam_valid[s["camera"]] += vt
    vh = valid_time / 3600.0
    return {
        "conf_single": conf_single, "conf_seq": conf_seq,
        "R_eff": round(r_eff, 4),
        "episodes_timed": ep_timed, "episodes_all": ep_all,
        "episodes_strict_timed": ep_strict_timed, "episodes_strict_all": ep_strict_all,
        "valid_hours": round(vh, 4),
        "eps_per_h": round(ep_timed / vh, 3) if vh else None,
        "eps_per_h_all": round(ep_all / vh, 3) if vh else None,
        "eps_per_h_strict": round(ep_strict_timed / vh, 3) if vh else None,
        "frac_valid_time_in_alert": round(alert_time / valid_time, 4) if valid_time else None,
        "neg_frame_alert_frac": round(n_alert_frames / len(rows_neg_cache()), 4),
        "per_camera_episodes": dict(per_cam_ep),
    }


_NEG_CACHE = None


def rows_neg_cache():
    global _NEG_CACHE
    if _NEG_CACHE is None:
        _NEG_CACHE = [img for _, run in G["runs"] for img, _ in run]
    return _NEG_CACHE


def worker(conf_single: float) -> list[dict]:
    out = []
    for conf_seq in G["grid"]:
        if conf_seq > conf_single:
            break
        out.append(eval_pair(conf_single, conf_seq))
    return out


# --------------------------------------------------------------------------
def run_system(name: str, spec: dict, anchor: dict, workers: int, out_dir: Path, base_cfg: Settings):
    t0 = time.time()
    parquet = ROOT / spec["parquet"]
    rows = load_rows(parquet, base_cfg.conf_floor)
    seqs = load_sequences()
    runs = negative_runs(seqs, rows)
    grid = build_grid()
    thr_index = {c: i for i, c in enumerate(grid)}
    imgs = list(rows.keys())
    for i, img in enumerate(imgs):
        rows[img]["idx"] = i
    row_list = [rows[i] for i in imgs]
    n_gt = sum(len(r["gt"]) for r in row_list)

    # frame recall by the canonical code path (sweep_report), full grid
    rep = sweep_report(row_list, MATCH_IOU, conf_grid=grid)
    r_single = {c["conf"]: c for c in rep["conf_sweep"]}
    # matched-count matrix (same greedy matching); self-check vs sweep_report
    M = np.zeros((len(row_list), len(grid)), dtype=np.int16)
    for i, r in enumerate(row_list):
        if r["gt"]:
            for j, c in enumerate(grid):
                M[i, j] = match_count(r["gt"], r["preds"], c)
    for j, c in enumerate(grid):
        assert round(M[:, j].sum() / n_gt, 4) == r_single[c]["recall"], (c, M[:, j].sum() / n_gt, r_single[c]["recall"])

    # observation-set bookkeeping
    all_gaps = [t1 - t0 for s in seqs for (_, t0, _), (_, t1, _) in zip(s["frames"][:-1], s["frames"][1:])]
    timed_runs = 0; iso = 0; excl = 0; vt_seq = 0.0; vt_glob = 0.0; neg_timed = 0
    per_cam_valid = defaultdict(float); cams = set()
    for s, run in runs:
        ts = [t for _, t in run]
        gaps = np.diff(ts)
        acc = gaps[gaps <= s["gap_thr_s"]]
        excl += int((gaps > s["gap_thr_s"]).sum())
        vt_seq += float(acc.sum()); vt_glob += float(gaps[gaps <= 3 * 29.0].sum())
        if len(run) == 1:
            iso += 1
        if acc.sum() > 0:
            timed_runs += 1; neg_timed += len(run); per_cam_valid[s["camera"]] += float(acc.sum()) / 3600
            cams.add(s["camera"])
    data_info = {
        "n_frames": len(row_list), "n_positive_frames": rep["n_positive_images"],
        "n_negative_frames": rep["n_negative_images"], "n_gt_boxes": n_gt,
        "n_sequences": len(seqs), "n_cameras": len({s["camera"] for s in seqs}),
        "sites": sorted({s["site"] for s in seqs}),
        "cadence": {"global_median_s": float(np.median(all_gaps)),
                    "global_p10_s": float(np.quantile(all_gaps, 0.1)), "global_p90_s": float(np.quantile(all_gaps, 0.9)),
                    "per_sequence_median_quantiles_s": {q: float(np.quantile([s["median_cadence_s"] for s in seqs], float(q)))
                                                        for q in ["0.1", "0.25", "0.5", "0.75", "0.9"]},
                    "rule": "gap excluded if > 3 x sequence median cadence (PROTOCOL_V5 §3)"},
        "negative_observation_set": {
            "n_runs": len(runs), "n_runs_timed": timed_runs, "n_isolated_single_frame_runs": iso,
            "n_negative_frames_in_timed_runs": neg_timed,
            "n_excluded_gaps": excl,
            "valid_hours": round(vt_seq / 3600, 4),
            "valid_hours_if_global_29s_rule": round(vt_glob / 3600, 4),
            "n_cameras_with_valid_time": len(cams),
            "per_camera_valid_hours": {k: round(v, 4) for k, v in sorted(per_cam_valid.items())},
            "nature": "embedded negatives = no-GT-box frames inside pyro-sdis VAL alert sequences (before/after/between smoke), NOT continuous clean footage",
        },
    }

    # sweep
    with Pool(workers, initializer=_init, initargs=(seqs, runs, rows, grid, M, thr_index, n_gt, base_cfg)) as pool:
        results = [p for chunk in pool.map(worker, grid, chunksize=1) for p in chunk]
    for p in results:
        rs = r_single[p["conf_single"]]
        p["R_single"] = rs["recall"]; p["neg_image_fp_rate_single"] = rs["neg_image_fp_rate"]
        p["precision_single"] = rs["precision"]
        p["recall_by_size_single"] = rs["recall_by_size"]

    def pick(cands, key_rule):
        feas = [p for p in cands if key_rule(p)]
        if not feas:
            return None, 0
        feas.sort(key=lambda p: (p["eps_per_h"], -p["R_eff"], -p["R_single"], p["conf_single"], p["conf_seq"]))
        return feas[0], len(feas)

    chosen, n_feas = pick(results, lambda p: p["R_single"] >= anchor["R_single"] and p["eps_per_h"] is not None)
    chosen_ea, n_feas_ea = pick(results, lambda p: p["R_single"] >= anchor["R_single"] and p["R_eff"] >= anchor["R_eff"] and p["eps_per_h"] is not None)
    by_pair = {(p["conf_single"], p["conf_seq"]): p for p in results}
    deployed_pts = [dict(label="YOLO deployed 0.30/0.15", **by_pair[DEPLOYED])]
    if "v4_val_point" in spec:
        vp = spec["v4_val_point"]
        deployed_pts.append(dict(label=f"v4 sdis-val-selected (expB-matched negFP) {vp[0]}/{vp[1]}", **by_pair[vp]))
    # frontier: per conf_single the min-eps conf_seq
    frontier = []
    for c in grid:
        cand = [p for p in results if p["conf_single"] == c]
        best = min(cand, key=lambda p: (p["eps_per_h"], -p["R_eff"], p["conf_seq"]))
        frontier.append({k: best[k] for k in ("conf_single", "conf_seq", "R_single", "R_eff", "episodes_timed", "episodes_all", "eps_per_h", "eps_per_h_all", "eps_per_h_strict", "frac_valid_time_in_alert", "neg_frame_alert_frac", "neg_image_fp_rate_single")})

    out = {
        "system": name, "kind": spec["kind"], "note": spec["note"], "weights": spec["weights"],
        "preds_parquet": str(spec["parquet"]), "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": "reports/v5/PROTOCOL_V5.md §3 (fixed-camera)", "scenario": "FIXED-CAMERA",
        "engine": {"seq_window": base_cfg.seq_window, "seq_hits_required": base_cfg.seq_hits_required,
                   "cooldown_s": base_cfg.cooldown_s, "stale_after_s": base_cfg.stale_after_s,
                   "conf_floor": base_cfg.conf_floor, "scene_change": "not simulated (no phash; sequences are per camera-azimuth)",
                   "alert_states_counted": list(ALERT_STATES), "strict_variant_states": list(STRICT_STATES),
                   "episode_rule": "new episode iff ts - last_alert_frame_ts >= cooldown_s; fresh engine per sequence (R_eff) / per negative run (FP)"},
        "match_iou": MATCH_IOU, "grid_size": len(grid), "n_pairs": len(results),
        "data": data_info,
        "anchor": anchor,
        "deployed_points": deployed_pts,
        "n_feasible_protocol": n_feas, "chosen_protocol": chosen,
        "n_feasible_engine_aware": n_feas_ea, "chosen_engine_aware": chosen_ea,
        "frontier": frontier,
        "runtime_s": round(time.time() - t0, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=1))
    keys = ["conf_single", "conf_seq", "R_single", "R_eff", "precision_single", "neg_image_fp_rate_single",
            "episodes_timed", "episodes_all", "episodes_strict_timed", "episodes_strict_all", "valid_hours",
            "eps_per_h", "eps_per_h_all", "eps_per_h_strict", "frac_valid_time_in_alert", "neg_frame_alert_frac"]
    with open(out_dir / f"{name}_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for p in sorted(results, key=lambda p: (p["conf_single"], p["conf_seq"])):
            w.writerow(p)
    print(f"[{name}] done in {out['runtime_s']}s | anchor R={anchor['R_single']} | deployed {DEPLOYED}: "
          f"R_single={by_pair[DEPLOYED]['R_single']} eps/h={by_pair[DEPLOYED]['eps_per_h']} | "
          f"chosen={None if not chosen else (chosen['conf_single'], chosen['conf_seq'], chosen['R_single'], chosen['R_eff'], chosen['eps_per_h'])} "
          f"(feasible {n_feas}) | engine-aware={None if not chosen_ea else (chosen_ea['conf_single'], chosen_ea['conf_seq'], chosen_ea['eps_per_h'])}", flush=True)
    return out


def compute_anchor(base_cfg: Settings, workers: int) -> dict:
    """expB frame recall (sweep_report) and R_eff at the deployed 0.30/0.15."""
    spec = SYSTEMS[ANCHOR_SYSTEM]
    rows = load_rows(ROOT / spec["parquet"], base_cfg.conf_floor)
    seqs = load_sequences(); runs = negative_runs(seqs, rows)
    imgs = list(rows.keys())
    for i, img in enumerate(imgs):
        rows[img]["idx"] = i
    row_list = [rows[i] for i in imgs]
    n_gt = sum(len(r["gt"]) for r in row_list)
    grid = [DEPLOYED[1], DEPLOYED[0]]
    rep = sweep_report(row_list, MATCH_IOU, conf_grid=grid)
    M = np.zeros((len(row_list), 2), dtype=np.int16)
    for i, r in enumerate(row_list):
        if r["gt"]:
            for j, c in enumerate(grid):
                M[i, j] = match_count(r["gt"], r["preds"], c)
    _init(seqs, runs, rows, grid, M, {c: i for i, c in enumerate(grid)}, n_gt, base_cfg)
    p = eval_pair(DEPLOYED[0], DEPLOYED[1])
    rs = [c for c in rep["conf_sweep"] if c["conf"] == DEPLOYED[0]][0]
    return {"system": ANCHOR_SYSTEM, "weights": spec["weights"], "conf_single": DEPLOYED[0], "conf_seq": DEPLOYED[1],
            "R_single": rs["recall"], "R_eff": p["R_eff"], "recall_by_size": rs["recall_by_size"],
            "neg_image_fp_rate": rs["neg_image_fp_rate"], "eps_per_h": p["eps_per_h"],
            "definition": "frame recall (IoU>=0.4 greedy, eval_detection.sweep_report) of deployed expB at conf_single=0.30 on pyro-sdis VAL; R_eff = engine-aware recall at 0.30/0.15"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="all")
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 8))
    ap.add_argument("--out-dir", default=str(ROOT / "reports/v5/calib"))
    args = ap.parse_args()
    base_cfg = Settings()
    for k in ("single_frame_alert_conf", "seq_frame_conf", "seq_window", "seq_hits_required", "cooldown_s", "stale_after_s", "conf_floor"):
        print(f"engine base {k} = {getattr(base_cfg, k)}")
    anchor = compute_anchor(base_cfg, args.workers)
    print("ANCHOR:", json.dumps(anchor))
    names = list(SYSTEMS) if args.systems == "all" else args.systems.split(",")
    for n in names:
        run_system(n, SYSTEMS[n], anchor, args.workers, Path(args.out_dir), base_cfg)


if __name__ == "__main__":
    main()
