"""V5 AERIAL threshold-calibration baseline (no training, CPU only).

Scenario: AERIAL/UAV (PROTOCOL_V5.md §1). Calibration material = Boreal
heinola VAL ONLY (PROTOCOL §2: heinola is AERIAL CALIBRATION/VAL, never
training; it is NOT fixed-camera calibration material). Boreal ruokolahti
(aerial FINAL TEST, opened once in v4) is historical and is not read here.

Material
  (i)  boreal VAL frames: data/v4/eval/boreal_val (906 positive / 217 negative
       frames, 917 GT boxes) -- predictions already exist for every system in
       reports/v4/eval_<tag>_borealval.raw.parquet (YOLO, cluster run,
       imgsz 1280, conf 0.02) and reports/v4/eval_<run>_boreal_val.raw.parquet
       (DINO, cluster run, conf floor 0.02). No inference is repeated for them.
  (ii) the 14 heinola NON-SMOKE 30-s clips (video-level label, Boreal
       Subset-B), 1-fps frames under data/v4/sequences/boreal_frames/<clip>/,
       enumerated from data/v4/sequences/sequences_v4.parquet (source=boreal,
       split=val, seq_label=no_smoke) and cross-checked against
       data/v4/manifests/boreal_manifest.parquet (label_type=videolabel,
       has_smoke_label=False, parent_unit=burn_heinola). These frames have no
       saved scores: `score` runs the detectors on CPU and caches
       data/v5/calib_aerial/clip_preds_<system>.parquet.
  (iii) AuSmoke DJI val unit aus_DJI_0792_001 (195 frames, positives only,
       permission PENDING) from reports/v4/eval_<tag>_forestval.raw.parquet --
       recall-at-chosen-conf only, as the only other aerial material.

Rule (PROTOCOL §3, aerial constraint): choose the HIGHEST conf such that
boreal-VAL frame recall >= 0.90 (hold the recall constraint, then minimise
false alerts) -- not "raise the threshold until FP looks good". Recall is
the frozen v1-v4 definition (src/eval/eval_detection.sweep_report: greedy
IoU>=0.4 matching, box-level TP/(TP+FN)).

Episodes (PROTOCOL §3): src/serve/alerting.AlertEngine at 1 fps, one device
per clip (clips are NEVER concatenated), episode starts at any state != none
and ends when the device returns to none for >= cooldown (60 s). Valid time
= sum of covered 1-s frames of the 14 clips (439 s = 7.32 min, ONE burn, ONE
day) -> rates are anecdotal with wide exact-Poisson intervals. Because every
clip is <= 32 s and cooldown is 60 s, a clip can hold at most ONE episode.

Usage
  python scripts/v5/calibrate_aerial.py score --system expB_deployed [--threads 8]
  python scripts/v5/calibrate_aerial.py score --system DINO_B4_LoRA_L --threads 24
  python scripts/v5/calibrate_aerial.py parity --system expB_deployed   # weights <-> parquet check
  python scripts/v5/calibrate_aerial.py calibrate                       # tables + report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data/v5/calib_aerial"
REPORT = ROOT / "reports/v5/CALIBRATION_AERIAL_V5.md"
RESULTS_JSON = OUT_DIR / "calibration_aerial_v5.json"

SEQ_PARQUET = ROOT / "data/v4/sequences/sequences_v4.parquet"
MANIFEST = ROOT / "data/v4/manifests/boreal_manifest.parquet"

RECALL_TARGET = 0.90           # PROTOCOL §3 aerial constraint
SECONDARY_TARGETS = [0.95, 0.90, 0.85, 0.80]
MATCH_IOU = 0.4
CONF_FLOOR = 0.02              # same floor as the saved cluster predictions
CADENCE_S = 1.0                # clip frames are 1 fps
AUSMOKE_UNIT = "DJI_0792_001"

# alert engine settings = deployed service defaults except the two confs
ALERT_DEFAULTS = dict(seq_window=5, seq_hits_required=3, cooldown_s=60.0,
                      stale_after_s=600.0, scene_change_hamming=20)

SYSTEMS = {
    # name: kind, weights, boreal-val preds, forest-val preds, label
    "expB_deployed": dict(kind="yolo", weights="models/expB_best.pt",
                          val="reports/v4/eval_expB_borealval.raw.parquet",
                          forest="reports/v4/eval_expB_forestval.raw.parquet",
                          label="S0b expB y11s (DEPLOYED default)"),
    "expE": dict(kind="yolo", weights="models/trained/expE_mix_y11m_1280_v2/seg05/best.pt",
                 val="reports/v4/eval_expE_borealval.raw.parquet",
                 forest="reports/v4/eval_expE_forestval.raw.parquet",
                 label="S0 expE y11m (v3 candidate)"),
    "A_ctrl_v4a_s2": dict(kind="yolo", weights="models/trained/expV4A_ctrl_s2/seg05/best.pt",
                          val="reports/v4/eval_v4a_s2_borealval.raw.parquet",
                          forest="reports/v4/eval_v4a_s2_forestval.raw.parquet",
                          label="S1 A_ctrl_E (v4a_s2, source-only)"),
    "E_ET_v4e_s1": dict(kind="yolo", weights="models/v4/runs/expV4E_et_s1/seg05/best.pt",
                        val="reports/v4/eval_v4e_s1_borealval.raw.parquet",
                        forest="reports/v4/eval_v4e_s1_forestval.raw.parquet",
                        label="S2 E_ET (v4e_s1, +target aerial)"),
    "C_EGT_v4c_s1": dict(kind="yolo", weights="models/v4/runs/expV4C_egt_s1/seg05/best.pt",
                         val="reports/v4/eval_v4c_s1_borealval.raw.parquet",
                         forest="reports/v4/eval_v4c_s1_forestval.raw.parquet",
                         label="S2c C_EGT (v4c_s1)"),
    "D_curr_v4dada_s1": dict(kind="yolo", weights="models/v4/runs/expV4Dada_s1/seg03/best.pt",
                             val="reports/v4/eval_v4dada_s1_borealval.raw.parquet",
                             forest="reports/v4/eval_v4dada_s1_forestval.raw.parquet",
                             label="S3 D_curr (v4dada_s1, curriculum)"),
    "DINO_B4_LoRA_L": dict(kind="dino", weights="models/v3/runs/v4B4_lora_L/seg05/best.pt",
                           val="reports/v4/eval_v4B4_lora_L_boreal_val.raw.parquet",
                           forest="reports/v4/eval_v4B4_lora_L_forest_val.raw.parquet",
                           label="S4p DINOv3-L LoRA r16 (B4)"),
    "DINO_B6_LoRA_Hplus": dict(kind="dino", weights="models/v3/runs/v4B6_lora_Hplus/seg10/best.pt",
                               val="reports/v4/eval_v4B6_lora_Hplus_boreal_val.raw.parquet",
                               forest="reports/v4/eval_v4B6_lora_Hplus_forest_val.raw.parquet",
                               label="S4 DINOv3-H+ LoRA r16 (B6, v4 pre-reg pick)"),
}
# val-only (no local weights scored on clips) -- listed for the recall side
VAL_ONLY = {
    "DINO_B1_frozen_L": dict(kind="dino", weights=None,
                             val="reports/v4/eval_v4B1_frozen_L_boreal_val.raw.parquet",
                             forest="reports/v4/eval_v4B1_frozen_L_forest_val.raw.parquet",
                             label="S4c DINOv3-L frozen (B1, control) -- clips NOT scored"),
}


# ----------------------------------------------------------------------------
# material
def heinola_nonsmoke_clips() -> pd.DataFrame:
    s = pd.read_parquet(SEQ_PARQUET)
    c = s[(s.source == "boreal") & (s.split == "val") & (s.seq_label == "no_smoke")].copy()
    assert (c.parent_unit == "burn_heinola").all(), c.parent_unit.unique()
    c["clip"] = c.seq_id.str.replace("boreal_", "", regex=False)
    c = c.sort_values(["clip", "ts_or_order"]).reset_index(drop=True)
    # cross-check against the manifest video labels
    m = pd.read_parquet(MANIFEST)
    vl = m[(m.label_type == "videolabel") & (m.parent_unit == "burn_heinola") & (~m.has_smoke_label)]
    man_clips = sorted(Path(p).stem for p in vl.rel_path)
    seq_clips = sorted(c["clip"].unique())
    assert man_clips == seq_clips, (man_clips, seq_clips)
    missing = [p for p in c.abs_path if not Path(p).is_file()]
    assert not missing, missing[:3]
    return c


def load_rows(parquet: str) -> list[dict]:
    df = pd.read_parquet(ROOT / parquet)
    return [{"image": r["image"], "gt": [list(map(float, g)) for g in r["gt"]],
             "preds": [list(map(float, p)) for p in r["preds"]]}
            for r in df.to_dict("records")]


# ----------------------------------------------------------------------------
# scoring (clips)
def _norm_preds_from_xyxy(boxes, scores, w, h):
    return [[float(b[0]) / w, float(b[1]) / h, float(b[2]) / w, float(b[3]) / h, float(s)]
            for b, s in zip(boxes, scores)]


def score_yolo(weights: Path, paths: list[str], threads: int, log):
    import torch
    torch.set_num_threads(threads)
    from ultralytics import YOLO
    model = YOLO(str(weights))
    out = {}
    t0 = time.time()
    B = 8
    for i in range(0, len(paths), B):
        batch = paths[i:i + B]
        # identical settings to cluster/v4/yolo_eval_entry.sh (imgsz 1280, conf 0.02,
        # default NMS iou 0.7 / max_det 300), device cpu instead of cuda:0
        res = model.predict(batch, imgsz=1280, conf=CONF_FLOOR, device="cpu", verbose=False)
        for p, r in zip(batch, res):
            h, w = r.orig_shape
            out[p] = _norm_preds_from_xyxy(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), w, h)
        if (i // B) % 10 == 0:
            log(f"  {i + len(batch)}/{len(paths)}  {time.time() - t0:.0f}s")
    return out


def score_dino(weights: Path, paths: list[str], threads: int, log):
    import cv2
    import torch
    torch.set_num_threads(threads)
    from src.serve.backend_dino3 import Dino3Backend
    # conf_floor 0.02 = cluster dino_infer.py --conf-floor used for the saved parquets;
    # lora_merge is function-preserving (reports/v4/BACKEND_DINO3_LORA.md: 3e-6)
    b = Dino3Backend(str(weights), device="cpu", conf_floor=CONF_FLOOR, lora_merge=True)
    log(f"  backend: {json.dumps(b.info())}")
    out = {}
    t0 = time.time()
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        h, w = img.shape[:2]
        dets, _ = b.detect(img)
        out[p] = [[d.x1 / w, d.y1 / h, d.x2 / w, d.y2 / h, d.score] for d in dets]
        if i % 20 == 0:
            log(f"  {i + 1}/{len(paths)}  {time.time() - t0:.0f}s")
    return out


def cmd_score(args):
    sysd = SYSTEMS[args.system]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pq = OUT_DIR / f"clip_preds_{args.system}.parquet"
    if out_pq.exists() and not args.force:
        print("exists, skip:", out_pq)
        return
    logf = (OUT_DIR / "logs" / f"score_{args.system}.log").open("a")

    def log(msg):
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    clips = heinola_nonsmoke_clips()
    paths = clips.abs_path.tolist()
    log(f"score {args.system} kind={sysd['kind']} weights={sysd['weights']} frames={len(paths)} "
        f"clips={clips["clip"].nunique()} threads={args.threads} device=cpu")
    t0 = time.time()
    fn = score_yolo if sysd["kind"] == "yolo" else score_dino
    preds = fn(ROOT / sysd["weights"], paths, args.threads, log)
    clips["preds"] = [preds[p] for p in paths]
    clips["max_score"] = [max([q[4] for q in preds[p]], default=0.0) for p in paths]
    clips["system"] = args.system
    clips["scored_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    clips["source_tag"] = "historical_replay"   # PROTOCOL §3: never a live/field run
    keep = ["system", "clip", "seq_id", "parent_unit", "frame_idx", "ts_or_order", "abs_path",
            "seq_label", "label_kind", "preds", "max_score", "scored_at_utc", "source_tag"]
    clips[keep].to_parquet(out_pq, index=False)
    log(f"wrote {out_pq}  {len(paths)} frames in {time.time() - t0:.0f}s "
        f"({(time.time() - t0) / len(paths):.2f} s/frame)")


# ----------------------------------------------------------------------------
# parity: local CPU weights vs saved cluster parquet on a few boreal-val frames
def _match_scores(a, b, iou_min=0.8):
    from src.eval.eval_detection import iou
    diffs, unmatched = [], 0
    used = set()
    for p in a:
        best, bj = 0.0, -1
        for j, q in enumerate(b):
            if j in used:
                continue
            v = iou(p[:4], q[:4])
            if v > best:
                best, bj = v, j
        if best >= iou_min:
            used.add(bj); diffs.append(abs(p[4] - b[bj][4]))
        else:
            unmatched += 1
    return diffs, unmatched + (len(b) - len(used))


def cmd_parity(args):
    sysd = SYSTEMS[args.system]
    rows = load_rows(sysd["val"])
    rng = np.random.RandomState(0)
    pos = [r for r in rows if r["gt"]]
    neg = [r for r in rows if not r["gt"]]
    pick = [pos[i] for i in rng.choice(len(pos), args.n // 2, replace=False)] + \
           [neg[i] for i in rng.choice(len(neg), args.n - args.n // 2, replace=False)]
    paths = [r["image"] for r in pick]
    fn = score_yolo if sysd["kind"] == "yolo" else score_dino
    preds = fn(ROOT / sysd["weights"], paths, args.threads, lambda m: None)
    all_d, all_un, n_local, n_saved = [], 0, 0, 0
    for r in pick:
        # compare above 0.05 only: boxes hovering at the 0.02 floor appear/disappear with fp32 kernels
        a = [p for p in preds[r["image"]] if p[4] >= 0.05]
        b = [p for p in r["preds"] if p[4] >= 0.05]
        d, un = _match_scores(a, b)
        all_d += d; all_un += un; n_local += len(a); n_saved += len(b)
    res = {"system": args.system, "n_images": len(pick), "boxes_local_ge0.05": n_local,
           "boxes_saved_ge0.05": n_saved, "matched_boxes": len(all_d),
           "max_abs_dscore": round(max(all_d), 4) if all_d else None,
           "median_abs_dscore": round(float(np.median(all_d)), 4) if all_d else None,
           "unmatched_boxes": all_un}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"parity_{args.system}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


# ----------------------------------------------------------------------------
# calibration
def conf_grid(kind: str) -> list[float]:
    g = list(np.round(np.arange(CONF_FLOOR, 0.99 + 1e-9, 0.001), 4))
    if kind == "dino":   # softmax-scored, saturates above 0.99
        g += list(np.round(np.arange(0.9901, 0.9999 + 1e-9, 0.0001), 4))
    return sorted(set(float(x) for x in g))


def sweep(rows, grid):
    from src.eval.eval_detection import sweep_report
    rep = sweep_report(rows, MATCH_IOU, conf_grid=grid)
    for s in rep["conf_sweep"]:
        s["neg_fp_images"] = int(round(s["neg_image_fp_rate"] * rep["n_negative_images"])) \
            if s["neg_image_fp_rate"] is not None else None
    return rep


def pick_conf(sw: list[dict], target: float):
    """Highest conf on the grid whose recall >= target (hold recall, then minimise FP)."""
    ok = [s for s in sw if s["recall"] is not None and s["recall"] >= target]
    if not ok:
        return None
    return max(ok, key=lambda s: s["conf"])


def poisson_ci(k: int, hours: float):
    """Exact 95% CI for a Poisson rate (events/hour)."""
    try:
        from scipy.stats import chi2
        lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
        hi = chi2.ppf(0.975, 2 * k + 2) / 2
    except Exception:  # fallback: Byar's approximation
        lo = 0.0 if k == 0 else k * (1 - 1 / (9 * k) - 1.96 / (3 * math.sqrt(k))) ** 3
        kk = k + 1
        hi = kk * (1 - 1 / (9 * kk) + 1.96 / (3 * math.sqrt(kk))) ** 3
    return [round(lo / hours, 1), round(hi / hours, 1)]


def run_alerts(clip_df: pd.DataFrame, single_conf: float, seq_conf: float):
    """AlertEngine per clip (one device each; never concatenated). Returns
    protocol episode counts and alert-time fractions."""
    from src.serve.alerting import AlertEngine
    cfg = SimpleNamespace(single_frame_alert_conf=single_conf, seq_frame_conf=seq_conf,
                          **ALERT_DEFAULTS)
    eng = AlertEngine(cfg)
    cooldown = cfg.cooldown_s
    n_frames = 0
    frames_alert = 0        # provisional or confirmed
    frames_not_none = 0     # literal PROTOCOL reading: any state != none (incl. watch)
    episodes_alert = 0
    episodes_not_none = 0
    clips_alert, clips_not_none = [], []
    per_clip = {}
    for clip, g in clip_df.groupby("clip", sort=True):
        g = g.sort_values("ts_or_order")
        states = []
        for _, r in g.iterrows():
            ts = float(r.ts_or_order) * CADENCE_S
            st = eng.update(f"heinola_{clip}", ts, float(r.max_score))
            states.append((ts, st["state"]))
        n_frames += len(states)

        def count_eps(active_fn):
            eps, in_ep, none_since = 0, False, None
            for ts, s in states:
                a = active_fn(s)
                if a:
                    if not in_ep:
                        eps += 1; in_ep = True
                    none_since = None
                else:
                    if in_ep:
                        if none_since is None:
                            none_since = ts
                        elif ts - none_since >= cooldown:
                            in_ep = False; none_since = None
            return eps
        e_alert = count_eps(lambda s: s in ("provisional", "confirmed"))
        e_nn = count_eps(lambda s: s != "none")
        episodes_alert += e_alert; episodes_not_none += e_nn
        fa = sum(1 for _, s in states if s in ("provisional", "confirmed"))
        fn_ = sum(1 for _, s in states if s != "none")
        frames_alert += fa; frames_not_none += fn_
        if e_alert: clips_alert.append(clip)
        if e_nn: clips_not_none.append(clip)
        per_clip[clip] = {"frames": len(states), "frames_alert": fa, "frames_not_none": fn_,
                          "episodes_alert": e_alert, "episodes_not_none": e_nn,
                          "max_score": round(float(g.max_score.max()), 4),
                          "first_alert_s": next((ts for ts, s in states if s in ("provisional", "confirmed")), None),
                          "any_confirmed": any(s == "confirmed" for _, s in states)}
    hours = n_frames * CADENCE_S / 3600.0
    return {"single_frame_alert_conf": single_conf, "seq_frame_conf": seq_conf,
            "n_clips": int(clip_df["clip"].nunique()), "n_frames": n_frames,
            "valid_hours": round(hours, 4), "valid_minutes": round(hours * 60, 2),
            "cadence": "1 fps (1-s frames)",
            "episodes_alert": episodes_alert,
            "episodes_per_valid_hour_alert": round(episodes_alert / hours, 1),
            "episodes_per_valid_hour_alert_ci95": poisson_ci(episodes_alert, hours),
            "clips_alerted": len(clips_alert), "clips_alerted_list": clips_alert,
            "frac_time_in_alert": round(frames_alert / n_frames, 4),
            "episodes_not_none": episodes_not_none,
            "episodes_per_valid_hour_not_none": round(episodes_not_none / hours, 1),
            "episodes_per_valid_hour_not_none_ci95": poisson_ci(episodes_not_none, hours),
            "clips_not_none": len(clips_not_none),
            "frac_time_not_none": round(frames_not_none / n_frames, 4),
            "per_clip": per_clip}


def ausmoke_recall(rows, conf):
    from src.eval.eval_detection import sweep_report
    unit = [r for r in rows if AUSMOKE_UNIT in r["image"]]
    assert unit and all(r["gt"] for r in unit), "AuSmoke unit must be positives-only"
    rep = sweep_report(unit, MATCH_IOU, conf_grid=[conf])
    s = rep["conf_sweep"][0]
    frames_hit = sum(1 for r in unit if any(p[4] >= conf for p in r["preds"]))
    return {"unit": f"aus_{AUSMOKE_UNIT}", "n_frames": len(unit),
            "n_gt_boxes": sum(len(r["gt"]) for r in unit),
            "box_recall": s["recall"], "recall_by_size": s["recall_by_size"],
            "gt_by_size": s["gt_by_size"],
            "frames_with_any_detection": frames_hit,
            "frame_hit_rate": round(frames_hit / len(unit), 4)}


def cmd_calibrate(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips = heinola_nonsmoke_clips()
    clip_frames = int(len(clips)); clip_n = int(clips["clip"].nunique())
    results = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "protocol": "reports/v5/PROTOCOL_V5.md §1-3 (aerial constraint: boreal VAL recall >= 0.90)",
               "scenario": "AERIAL/UAV",
               "recall_target": RECALL_TARGET, "match_iou": MATCH_IOU, "conf_floor": CONF_FLOOR,
               "alert_engine_defaults": ALERT_DEFAULTS,
               "clips": {"n_clips": clip_n, "n_frames": clip_frames,
                         "valid_minutes": round(clip_frames * CADENCE_S / 60, 2),
                         "per_clip_frames": clips.groupby("clip").size().to_dict(),
                         "burns": sorted(clips.parent_unit.unique().tolist()),
                         "source_tag": "historical_replay"},
               "systems": {}}
    all_systems = {**SYSTEMS, **VAL_ONLY}
    for name, sysd in all_systems.items():
        print("==", name, flush=True)
        rows = load_rows(sysd["val"])
        rep = sweep(rows, conf_grid(sysd["kind"]))
        sw = rep["conf_sweep"]
        r_floor = sw[0]
        best = max(sw, key=lambda s: (s["recall"] or 0))
        entry = {"label": sysd["label"], "kind": sysd["kind"], "weights": sysd["weights"],
                 "val_parquet": sysd["val"], "forest_parquet": sysd["forest"],
                 "n_images": rep["n_images"], "n_positive_images": rep["n_positive_images"],
                 "n_negative_images": rep["n_negative_images"],
                 "n_gt_boxes": sum(sw[0]["gt_by_size"].values()),
                 "recall_at_floor": {"conf": r_floor["conf"], "recall": r_floor["recall"],
                                     "neg_fp_images": r_floor["neg_fp_images"],
                                     "neg_image_fp_rate": r_floor["neg_image_fp_rate"]},
                 "max_recall": {"conf": best["conf"], "recall": best["recall"]},
                 "targets": {}}
        clip_pq = OUT_DIR / f"clip_preds_{name}.parquet"
        cdf = pd.read_parquet(clip_pq) if clip_pq.exists() else None
        entry["clips_scored"] = cdf is not None
        if cdf is not None:
            assert len(cdf) == clip_frames and cdf["clip"].nunique() == clip_n
            entry["clip_preds_parquet"] = str(clip_pq.relative_to(ROOT))
            entry["clip_max_score_summary"] = {
                "median": round(float(cdf.max_score.median()), 4),
                "p90": round(float(cdf.max_score.quantile(0.9)), 4),
                "max": round(float(cdf.max_score.max()), 4),
                "frames_with_any_box_ge_floor": int((cdf.max_score >= CONF_FLOOR).sum())}
        frows = load_rows(sysd["forest"])
        for tgt in SECONDARY_TARGETS:
            pt = pick_conf(sw, tgt)
            if pt is None:
                entry["targets"][f"R>={tgt:.2f}"] = {"feasible": False}
                continue
            t = {"feasible": True, "conf": pt["conf"], "val_recall": pt["recall"],
                 "val_precision": pt["precision"], "val_recall_by_size": pt["recall_by_size"],
                 "neg_fp_images": pt["neg_fp_images"], "neg_image_fp_rate": pt["neg_image_fp_rate"],
                 "ausmoke_dji_val_unit": ausmoke_recall(frows, pt["conf"])}
            if cdf is not None:
                t["clips_primary"] = run_alerts(cdf, pt["conf"], pt["conf"])
                t["clips_deployed_ratio"] = run_alerts(cdf, pt["conf"], round(pt["conf"] / 2, 4))
                if tgt != RECALL_TARGET:   # keep JSON small
                    for k in ("clips_primary", "clips_deployed_ratio"):
                        t[k].pop("per_clip", None)
            entry["targets"][f"R>={tgt:.2f}"] = t
        # deployed reference operating point (expB service defaults 0.30 / 0.15)
        if name == "expB_deployed":
            pt = next(s for s in sw if abs(s["conf"] - 0.30) < 1e-6)
            ref = {"conf": 0.30, "seq_conf": 0.15, "val_recall": pt["recall"],
                   "neg_fp_images": pt["neg_fp_images"], "neg_image_fp_rate": pt["neg_image_fp_rate"],
                   "ausmoke_dji_val_unit": ausmoke_recall(frows, 0.30)}
            if cdf is not None:
                ref["clips"] = run_alerts(cdf, 0.30, 0.15)
                ref["clips"].pop("per_clip", None)
            entry["deployed_reference_0.30_0.15"] = ref
        par = OUT_DIR / f"parity_{name}.json"
        if par.exists():
            entry["parity_check"] = json.loads(par.read_text())
        results["systems"][name] = entry
    RESULTS_JSON.write_text(json.dumps(results, indent=1))
    print("saved", RESULTS_JSON)
    write_report(results)
    print("saved", REPORT)


# ----------------------------------------------------------------------------
def fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_report(R):
    S = R["systems"]; C = R["clips"]
    key0 = f"R>={R['recall_target']:.2f}"
    L = []
    L.append("# V5 AERIAL threshold-calibration baseline (Day 1, no training)\n")
    L.append(f"Generated {R['generated_utc']} by `scripts/v5/calibrate_aerial.py` (CPU only; no cluster use). "
             f"Numbers come from `data/v5/calib_aerial/calibration_aerial_v5.json`.\n")
    L.append("**Scope statements (read first).**\n")
    L.append("- Scenario = **AERIAL/UAV** only (PROTOCOL_V5 §1). Nothing here calibrates the fixed-camera service.")
    L.append("- Calibration material = **Boreal heinola VAL only**: 217 negative + 906 positive frames (917 GT boxes) "
             "and the 14 heinola NON-SMOKE 30-s clips. Heinola is AERIAL VAL material (PROTOCOL §2); it is **not** "
             "fixed-camera calibration data and must never enter training. All 1,123 frames and 14 clips come from "
             "**ONE prescribed burn on ONE day (2022-06-28)**; frames and clips are not independent units.")
    L.append("- **Boreal ruokolahti (aerial FINAL TEST) is historical and was not read** by this script.")
    L.append("- The 14 clips carry a **video-level** 'non-smoke' label (Boreal Subset-B); there is no frame-level truth, "
             "and detector silence is not evidence of no-smoke. Any detection on them is counted as a false alert.")
    L.append("- Clip scoring is a **historical replay** (`source_tag=historical_replay`), not a field run.")
    L.append("- AuSmoke `aus_DJI_0792_001` is a forest-holdout **VAL** unit, positives only, licence **PERMISSION PENDING** "
             "(PROTOCOL §2): it is reported as recall-at-conf only, as the only other aerial material on disk.")
    L.append("- Saved predictions for the 1,123 VAL frames are the v4 cluster runs (YOLO imgsz 1280 conf 0.02; DINO conf floor "
             "0.02); the clip frames were scored locally on CPU with the same settings. Weight<->parquet parity was checked "
             "on sampled VAL frames (§6).\n")

    # ---- §0 dynamic headline (computed from the same results dict)
    L.append("## 0. Headline (all numbers from §3; one burn, 7.3 valid minutes — read §8 before quoting)\n")
    scored = {n: e for n, e in S.items() if e["targets"][key0]["feasible"]}
    infeas = [e["label"] for e in S.values() if not e["targets"][key0]["feasible"]]
    L.append(f"- Constraint 'boreal-VAL recall >= {R['recall_target']:.2f}' is **feasible for {len(scored)}/{len(S)} systems**"
             + (f"; infeasible: {', '.join(infeas)}." if infeas else "."))
    for n, e in scored.items():
        t = e["targets"][key0]
        line = (f"- {e['label']}: conf* **{t['conf']:.4f}** -> VAL recall {t['val_recall']:.3f}, neg-frame FP **{t['neg_fp_images']}/217**")
        if "clips_primary" in t:
            cp = t["clips_primary"]
            line += (f", clips alerted **{cp['clips_alerted']}/14**, false episodes **{cp['episodes_per_valid_hour_alert']:.0f}/valid h** "
                     f"[{cp['episodes_per_valid_hour_alert_ci95'][0]:.0f}, {cp['episodes_per_valid_hour_alert_ci95'][1]:.0f}], "
                     f"time in alert {cp['frac_time_in_alert']:.2f}")
        else:
            line += ", clips not scored"
        line += f"; AuSmoke DJI_0792 box recall {t['ausmoke_dji_val_unit']['box_recall']:.3f}."
        L.append(line)
    # clips fired by every scored system
    fired_sets = []
    for e in scored.values():
        t = e["targets"][key0]
        if "clips_primary" in t:
            fired_sets.append({c for c, v in t["clips_primary"]["per_clip"].items() if v["episodes_alert"]})
    if fired_sets:
        common = set.intersection(*fired_sets)
        union = set.union(*fired_sets)
        L.append(f"- Of the 14 'non-smoke' clips, **{len(common)} fire for every scored system** at its conf* "
                 f"({', '.join(sorted(common)) if common else 'none'}) and {len(union)} fire for at least one. The universally-fired clips "
                 "need **human visual review** before they are counted as false alerts (they come from a prescribed-burn day; the label is "
                 "video-level). Detector agreement is NOT evidence of smoke, and detector silence on the others is NOT evidence of no-smoke; "
                 "they are review candidates for the negative-review tool, nothing more.")
    L.append("- The 217 negative FRAMES and the 14 clips disagree in severity: systems with 0-1/217 flagged frames still alert on 5-7/14 clips. "
             "Continuous footage is a harder negative test than curated 'empty' frames; frame-level neg-FP understates aerial false alerts.")
    eb = S.get("expB_deployed")
    if eb and "deployed_reference_0.30_0.15" in eb and eb["targets"][key0]["feasible"]:
        d = eb["deployed_reference_0.30_0.15"]; t = eb["targets"][key0]
        dc = d.get("clips", {}); tc = t.get("clips_primary", {})
        L.append(f"- No single expB threshold satisfies both sides on heinola: the DEPLOYED point 0.30/0.15 gives VAL recall {d['val_recall']:.3f} "
                 f"(constraint {'met' if d['val_recall'] >= R['recall_target'] else 'NOT met'}) and alerts on {dc.get('clips_alerted', '?')}/14 clips; "
                 f"conf* {t['conf']:.3f} holds recall {t['val_recall']:.3f} and alerts on {tc.get('clips_alerted', '?')}/14 clips "
                 f"({t['neg_fp_images']}/217 negative frames flagged).")
    dn = {n: e for n, e in S.items() if e["kind"] == "dino" and e["targets"][key0]["feasible"]}
    if dn:
        confs = [e["targets"][key0]["conf"] for e in dn.values()]
        aus = [e["targets"][key0]["ausmoke_dji_val_unit"]["box_recall"] for e in dn.values()]
        L.append(f"- DINO conf* lies at {min(confs):.3f}-{max(confs):.4f} (saturated tail) and gives AuSmoke DJI_0792 box recall "
                 f"{min(aus):.3f}-{max(aus):.3f}: the heinola-selected DINO operating points do not transfer to a different aerial source "
                 "(same failure mode as the v4 FIgLib replay, reports/v4/dino_adapt_results.md).")
    L.append("- v4 reported 74 false episodes/h for the deployed expE point on heinola non-smoke footage (reports/v4/tables/T20) with its own "
             "episode-merge rule and thresholds; the numbers here use the PROTOCOL_V5 definition and conf* and are not directly comparable.\n")
    L.append("## 1. Material (counts from artifacts)\n")
    any_sys = next(iter(S.values()))
    L.append(f"| material | purpose | n | note |\n|---|---|---|---|")
    L.append(f"| boreal VAL frames (`data/v4/eval/boreal_val`) | recall constraint + neg-image FP | "
             f"{any_sys['n_images']} frames = {any_sys['n_positive_images']} pos / {any_sys['n_negative_images']} neg, "
             f"{any_sys['n_gt_boxes']} GT boxes | heinola, 4096x2160, human boxes |")
    L.append(f"| heinola non-smoke clips (`data/v4/sequences/boreal_frames/heinola_*`) | episode false alerts | "
             f"{C['n_clips']} clips, {C['n_frames']} frames @ 1 fps = **{C['valid_minutes']} min valid** "
             f"({C['n_frames'] / 3600:.4f} h) | video-level label only; ONE burn; clips never concatenated |")
    L.append(f"| AuSmoke `aus_DJI_0792_001` (`data/v4/eval/forest_val`) | aerial recall cross-check only | "
             f"195 frames, positives only | VAL unit; permission PENDING |\n")
    L.append("The 217 negative frames come from 3 heinola flights (DJI_0024: 59, DJI_0028: 111, DJI_0048: 47); flights 0028 and 0048 also "
             "contribute 156 and 15 of the 906 positive frames, i.e. the 'empty' frames are smoke-free moments of flights that also filmed smoke, "
             "not separate no-smoke sorties (counts from data/v4/manifests/boreal_manifest.parquet and the eval parquet image names).")
    L.append("Per-clip frame counts: " + ", ".join(f"{k}={v}" for k, v in sorted(C["per_clip_frames"].items())) + ".\n")
    L.append("## 2. Method\n")
    L.append(f"- **Recall** = frozen v1-v4 definition (`src/eval/eval_detection.sweep_report`): greedy matching, IoU >= {R['match_iou']}, "
             "box-level TP/(TP+FN) over the 917 VAL GT boxes. **neg-image FP** = negative frames with >= 1 box at the conf (k/217).")
    L.append(f"- **Selection rule** (PROTOCOL §3): on a 0.001 grid from {R['conf_floor']} to 0.99 (DINO: +0.0001 steps to 0.9999), take the "
             f"**highest** conf with VAL recall >= **{R['recall_target']:.2f}**; the constraint is held first, FP is minimised second. "
             "Infeasible = recall never reaches the target even at the 0.02 floor.")
    L.append("- **Episodes** (PROTOCOL §3): `src/serve/alerting.AlertEngine`, one device per clip, frames at 1 fps, "
             f"settings = service defaults ({', '.join(f'{k}={v}' for k, v in R['alert_engine_defaults'].items())}) with "
             "`single_frame_alert_conf = seq_frame_conf = conf*` (**primary**: then no 'watch' state exists and an episode = any frame >= conf*). "
             "Episode starts at any state != none, ends after >= 60 s of none. Because every clip is <= 32 s, **a clip holds at most one episode**, "
             "so episodes == clips alerted; the maximum possible rate is 14 / 0.122 h = 114.8/h. Rate denominators are VALID observed hours only "
             "(sum of covered 1-s frames; no gaps bridged; no concatenation). 95% intervals are exact Poisson.")
    L.append("- **Variant** (secondary): `seq_frame_conf = conf*/2` as in the deployed 0.30/0.15 ratio, counting the literal 'any state != none' "
             "(includes 'watch'); reported in §5.")
    L.append("- **Fraction of valid time in alert** = frames in provisional/confirmed state / valid frames.\n")

    L.append(f"## 3. Main result — conf at VAL recall >= {R['recall_target']:.2f} and what it costs on heinola non-smoke footage\n")
    L.append("| system | conf* | VAL recall (box) | neg-img FP k/217 | clips alerted /14 | **false episodes / valid h** [95% CI] | frac time in alert | AuSmoke DJI_0792 box recall (frames w/ any box, not matched) — VAL unit, PENDING |")
    L.append("|---|---|---|---|---|---|---|---|")
    key = f"R>={R['recall_target']:.2f}"
    for name, e in S.items():
        t = e["targets"][key]
        if not t["feasible"]:
            L.append(f"| {e['label']} | **infeasible** (max recall {e['max_recall']['recall']:.3f} @ conf {e['max_recall']['conf']:.3f}; "
                     f"recall at 0.02 floor {e['recall_at_floor']['recall']:.3f}, neg FP {e['recall_at_floor']['neg_fp_images']}/217) | - | - | - | - | - | - |")
            continue
        au = t["ausmoke_dji_val_unit"]
        if "clips_primary" in t:
            cp = t["clips_primary"]
            ep = f"**{cp['episodes_per_valid_hour_alert']:.1f}** [{cp['episodes_per_valid_hour_alert_ci95'][0]}, {cp['episodes_per_valid_hour_alert_ci95'][1]}]"
            ca = f"{cp['clips_alerted']}"; ft = f"{cp['frac_time_in_alert']:.3f}"
        else:
            ep = "clips not scored"; ca = "-"; ft = "-"
        L.append(f"| {e['label']} | {t['conf']:.4f} | {t['val_recall']:.3f} | {t['neg_fp_images']}/217 ({t['neg_image_fp_rate']:.3f}) | {ca} | {ep} | {ft} | "
                 f"{au['box_recall']:.3f} ({au['frames_with_any_detection']}/{au['n_frames']} frames w/ any box) |")
    L.append("")
    L.append(f"Valid time = {C['valid_minutes']} min ({C['n_frames']} one-second frames, {C['n_clips']} clips, 1 fps, ONE burn). "
             "With this little footage the rates are **anecdotal**: even 0 episodes only bounds the rate below ~30/h, and 14/14 clips alerted is the ceiling. "
             "They are reported because PROTOCOL §3 asks for episodes per valid hour, not because 7 minutes can establish a deployable rate.\n")

    # deployed reference
    e = S.get("expB_deployed")
    if e and "deployed_reference_0.30_0.15" in e:
        d = e["deployed_reference_0.30_0.15"]
        L.append("## 4. Deployed reference point (expB at the service defaults 0.30 / 0.15) — NOT a calibrated point\n")
        row = (f"VAL recall {d['val_recall']:.3f} (constraint {R['recall_target']:.2f} **{'met' if d['val_recall'] >= R['recall_target'] else 'NOT met'}**), "
               f"neg-img FP {d['neg_fp_images']}/217")
        if "clips" in d:
            c = d["clips"]
            row += (f"; clips alerted {c['clips_alerted']}/14, false episodes {c['episodes_per_valid_hour_alert']:.1f}/valid h "
                    f"[{c['episodes_per_valid_hour_alert_ci95'][0]}, {c['episodes_per_valid_hour_alert_ci95'][1]}] (provisional/confirmed), "
                    f"frac time in alert {c['frac_time_in_alert']:.3f}; literal any-state!=none {c['episodes_per_valid_hour_not_none']:.1f}/h, "
                    f"frac {c['frac_time_not_none']:.3f}")
        row += f"; AuSmoke DJI_0792 recall {d['ausmoke_dji_val_unit']['box_recall']:.3f}."
        L.append(row + "\n")

    L.append("## 5. Recall/false-alert trade-off (same rule at other recall targets; primary alert config)\n")
    L.append("Shows how much false-alert reduction the recall constraint buys or forgoes. Episodes are provisional/confirmed; "
             "the `!=none` column is the literal reading with `seq_frame_conf = conf*/2` (watch state counted).\n")
    L.append("| system | target | conf | VAL recall | neg FP k/217 | clips /14 | episodes/valid h [CI] | frac alert | variant seq=conf*/2: clips!=none /14, ep/h, frac!=none |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, e in S.items():
        for tgt in SECONDARY_TARGETS:
            t = e["targets"][f"R>={tgt:.2f}"]
            if not t["feasible"]:
                L.append(f"| {e['label']} | >={tgt:.2f} | infeasible | - | - | - | - | - | - |")
                continue
            if "clips_primary" in t:
                cp, cv = t["clips_primary"], t["clips_deployed_ratio"]
                cl = (f"{cp['clips_alerted']} | {cp['episodes_per_valid_hour_alert']:.1f} [{cp['episodes_per_valid_hour_alert_ci95'][0]}, "
                      f"{cp['episodes_per_valid_hour_alert_ci95'][1]}] | {cp['frac_time_in_alert']:.3f} | "
                      f"{cv['clips_not_none']}, {cv['episodes_per_valid_hour_not_none']:.1f}/h, {cv['frac_time_not_none']:.3f}")
            else:
                cl = "- | not scored | - | -"
            L.append(f"| {e['label']} | >={tgt:.2f} | {t['conf']:.4f} | {t['val_recall']:.3f} | {t['neg_fp_images']}/217 | {cl} |")
    L.append("")

    L.append("## 6. Weight <-> saved-prediction parity (local CPU weights vs v4 cluster parquet, sampled VAL frames)\n")
    L.append("| system | n imgs | boxes local/saved (>=0.05) | matched | max |d score| | median |d score| | unmatched |")
    L.append("|---|---|---|---|---|---|---|")
    for name, e in S.items():
        p = e.get("parity_check")
        if not p:
            L.append(f"| {e['label']} | not run | | | | | |"); continue
        L.append(f"| {e['label']} | {p['n_images']} | {p['boxes_local_ge0.05']}/{p['boxes_saved_ge0.05']} | {p['matched_boxes']} | "
                 f"{fmt(p['max_abs_dscore'], 4)} | {fmt(p['median_abs_dscore'], 4)} | {p['unmatched_boxes']} |")
    L.append("\nSmall score differences are CPU-fp32 vs GPU kernels; a mismatch in box count or a large score gap would mean the "
             "weights file is not the checkpoint behind the parquet.\n")

    L.append("## 7. Per-clip detail at conf* (which non-smoke clips fire)\n")
    L.append("| system | conf* | clips alerted (first alert s, max score) |")
    L.append("|---|---|---|")
    for name, e in S.items():
        t = e["targets"][key]
        if not t.get("feasible") or "clips_primary" not in t:
            continue
        pc = t["clips_primary"]["per_clip"]
        fired = [f"{c} (t={v['first_alert_s']:.0f}s, max {v['max_score']:.3f}{', confirmed' if v['any_confirmed'] else ''})"
                 for c, v in sorted(pc.items()) if v["episodes_alert"]]
        L.append(f"| {e['label']} | {t['conf']:.4f} | {'; '.join(fired) if fired else 'none'} |")
    L.append("")
    L.append("Clip max-score summary (any box >= 0.02 floor): " + "; ".join(
        f"{e['label'].split(' ')[0]} {e['label'].split(' ')[1]}: median {e['clip_max_score_summary']['median']:.3f}, p90 {e['clip_max_score_summary']['p90']:.3f}, "
        f"max {e['clip_max_score_summary']['max']:.3f}, frames with any box {e['clip_max_score_summary']['frames_with_any_box_ge_floor']}/{C['n_frames']}"
        for e in S.values() if e.get("clip_max_score_summary")) + ".\n")

    L.append("## 8. Limitations (state, do not soften)\n")
    L.append("- ONE burn, ONE day, ONE aircraft type: the 217 negatives, the 14 clips and the 906 positives share background, lighting and smoke "
             "character. A conf* chosen here is a heinola operating point, not an aerial operating point; the v4 heinola->ruokolahti transfer "
             "already showed val gaps shrinking on the test burn (reports/v4/AERIAL_EVAL_V4.md).")
    L.append("- 7.3 valid minutes of non-smoke footage cannot establish an episode rate; PROTOCOL §6 requires >= 2 h / >= 3 independent units per scenario. "
             "The aerial scenario currently has 0 h of independent no-smoke UAV footage beyond this burn.")
    L.append("- Clip labels are video-level. If a clip contained faint smoke that the labeller did not count, an 'episode' here would be a true alert; "
             "conversely detector silence on these clips says nothing about their content.")
    L.append("- The 217 negative frames (Subset-A 'Empty-Images') and the 14 clips (Subset-B videos) come from the same flights/day; they are not two "
             "independent negative sources, and the frames' FP and the clips' episodes are correlated readings of the same footage.")
    L.append("- Recall is box-level over 917 GT boxes dominated by large plumes (902 'large'); 'recall >= 0.90' says nothing about faint/early smoke.")
    L.append("- DINO scores saturate (>= 0.99 operating points): conf* for B4/B6 sits on a 0.0001 grid in the saturated tail and is fragile; "
             "see reports/v4/dino_adapt_results.md for the threshold-transfer failure on FIgLib.")
    L.append("- Training exposure differs by system: expB/expE (v1-v3 recipes) saw no Boreal material at all; the v4 T-arms (E_ET, C_EGT, D_curr) and the "
             "DINO LoRA runs (E_ET recipe) trained on the OTHER Boreal burns (evo, karkkila, incl. their 37 verified-empty frames). The heinola background "
             "is unseen by every system, but Boreal smoke/background character is not new to the T-arms or DINO — their lower false-alert counts here are "
             "partly same-dataset familiarity, not proof of UAV generalisation (the v4 ruokolahti test showed the heinola val gap shrinking).")
    L.append("- AuSmoke permission is PENDING; its number is a cross-check on existing v4 predictions, not a licence to use the data further.\n")
    L.append("## 9. Artifacts\n")
    L.append("- `data/v5/calib_aerial/calibration_aerial_v5.json` — every number above, full conf curves not stored (recompute via `calibrate`).")
    L.append("- `data/v5/calib_aerial/clip_preds_<system>.parquet` — per-frame boxes/scores for the 439 clip frames (`source_tag=historical_replay`).")
    L.append("- `data/v5/calib_aerial/parity_<system>.json`, `data/v5/calib_aerial/logs/score_<system>.log`.")
    L.append("- Inputs (read only): `reports/v4/eval_*_borealval.raw.parquet`, `reports/v4/eval_*_boreal_val.raw.parquet`, `reports/v4/eval_*_forestval.raw.parquet`, "
             "`data/v4/sequences/sequences_v4.parquet`, `data/v4/manifests/boreal_manifest.parquet`.")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("score"); a.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    a.add_argument("--threads", type=int, default=8); a.add_argument("--force", action="store_true")
    a.set_defaults(fn=cmd_score)
    p = sub.add_parser("parity"); p.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    p.add_argument("--threads", type=int, default=8); p.add_argument("--n", type=int, default=8)
    p.set_defaults(fn=cmd_parity)
    c = sub.add_parser("calibrate"); c.set_defaults(fn=cmd_calibrate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
