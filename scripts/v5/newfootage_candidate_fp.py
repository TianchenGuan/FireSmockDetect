"""V5 Day-1/2 — candidate-FP triage of NEW, UNVERIFIED no-smoke footage (MONIMET MC124, BAMBI flight_98).

Everything here is "candidate FP on UNVERIFIED footage: detector output is not evidence of smoke or of
its absence" (PROTOCOL_V5 §6). CPU-only except where DINO preds come from the cluster job
(cluster/v3/submit_dino_eval.py --frames-tar, same route as the v4 FIgLib replay).

Subcommands
  lists        build ordered frame lists + per-frame unit/timestamp tables
               -> data/v5/frames/lists/<set>.txt, data/v5/newfootage/frames_<set>.csv, units_<set>.csv
  score-yolo   --system S --set SET --shard i/N --threads T   (ultralytics predict imgsz 1280 conf 0.02, CPU)
               -> models/v5/evals/_parts/<S>__<SET>.part<i>.jsonl
  merge        --system S --set SET   -> models/v5/evals/<S>__<SET>.jsonl  ({"image","w","h","preds":[[x1n,y1n,x2n,y2n,score]..],
               "source_tag":"historical_replay",...})
  score-dino   --system S --set SET [--list file]  CPU fallback via src/serve/backend_dino3 (slow: 4-7 s/frame)
  metrics      --set SET   per system x unit: frame alert rate at the Day-1 operating point, episodes per VALID hour
               (src/serve/alerting.AlertEngine replayed per unit), fraction of valid time in alert
               -> data/v5/newfootage/metrics_<SET>_units.csv, metrics_<SET>_summary.json
  review-lists --set SET   frame lists for the human review package (negative_review.py sheets) + per-frame verdict template
               -> data/v5/review/newfootage_day1/...

Operating points (Day-1 calibrations; reports/v5/CALIBRATION_FIXED_V5.md §5, CALIBRATION_AERIAL_V5.md §3) are in SYSTEMS.
Cadence rule (PROTOCOL §3): valid observed time = covered intervals only; a gap > 3x median cadence is excluded.
MONIMET is 30-min SAMPLED: every frame is a 1-s snapshot; "episodes per valid hour" is a SAMPLED rate, not continuous.
With the service engine (stale_after_s = 600 s < 1800 s cadence) the K-of-N path never fires on MONIMET, so an episode
there is exactly one alert frame (cooldown 60 s << cadence) -> MONIMET measures frame-level FP, not episode behaviour.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/local/home/tianrgua/cvproject")
sys.path.insert(0, str(ROOT))
from scripts.v5 import review_common as rc  # noqa: E402
from src.serve.alerting import AlertEngine  # noqa: E402
from src.serve.config import Settings  # noqa: E402

FRAMES = ROOT / "data/v5/frames"
LISTS = FRAMES / "lists"
NF = ROOT / "data/v5/newfootage"
EVALS = ROOT / "models/v5/evals"
PARTS = EVALS / "_parts"
REVIEW = ROOT / "data/v5/review/newfootage_day1"
CONF_FLOOR_INFER = 0.02           # cluster/v4 yolo_eval_entry.sh + dino_infer.py floor
ALERT_STATES = ("provisional", "confirmed")
SOURCE_TAG = "historical_replay"  # PROTOCOL §3: never a live/field run

SETS = {
    "monimet_mc124": dict(scenario="fixed", dir=FRAMES / "monimet_mc124", cadence_label="30 min sampled",
                          cadence_hint_s=1800.0, unit_rule="camera-day (MC124 + local date in file name)"),
    "bambi_flight98": dict(scenario="aerial", dir=FRAMES / "bambi_flight98/frames", cadence_label="1 fps",
                           cadence_hint_s=1.0, unit_rule="one flight segment = one unit"),
}

# system -> kind, weights, {scenario: (conf_single, conf_seq)} ; label
SYSTEMS = {
    "expB":   dict(kind="yolo", weights="models/expB_best.pt", label="expB (deployed default, YOLO11s@1280)",
                   fixed=(0.30, 0.23), aerial=(0.1070, 0.1070)),
    "expE":   dict(kind="yolo", weights="models/trained/expE_mix_y11m_1280_v2/seg05/best.pt", label="expE (v3 S0, YOLO11m@1280)",
                   fixed=(0.32, 0.20)),
    "E_ET":   dict(kind="yolo", weights="models/v4/runs/expV4E_et_s1/seg05/best.pt", label="E_ET (v4e_s1, +target aerial)",
                   aerial=(0.2160, 0.2160)),
    "D_curr": dict(kind="yolo", weights="models/v4/runs/expV4Dada_s1/seg03/best.pt", label="D_curr (v4dada_s1, curriculum)",
                   aerial=(0.2120, 0.2120)),
    "B4":     dict(kind="dino", weights="models/v3/runs/v4B4_lora_L/seg05/best.pt", label="DINOv3-L LoRA r16 (B4)",
                   fixed=(0.91, 0.89), aerial=(0.9890, 0.9890)),
    "B6":     dict(kind="dino", weights="models/v3/runs/v4B6_lora_Hplus/seg10/best.pt", label="DINOv3-H+ LoRA r16 (B6)",
                   fixed=(0.94, 0.88), aerial=(0.9934, 0.9934)),
}
# the service floor applied before the engine in each calibration (fixed: Settings.conf_floor 0.05; aerial: 0.02 floor)
SCENARIO_FLOOR = {"fixed": 0.05, "aerial": 0.02}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def systems_for(scenario):
    return [s for s, d in SYSTEMS.items() if scenario in d]


# ----------------------------------------------------------------------------- lists
def cmd_lists(args):
    LISTS.mkdir(parents=True, exist_ok=True); NF.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, S in SETS.items():
        paths = sorted(str(p) for p in S["dir"].glob("*.jpg"))
        rows = []
        for p in paths:
            t, kind = rc.parse_ts_from_name(os.path.basename(p))
            assert t is not None, p
            if name == "monimet_mc124":
                d = dt.datetime.fromtimestamp(t, dt.timezone.utc)   # wall-clock as written (UTC+2 local)
                unit = f"MC124_{d:%Y-%m-%d}"
                ts_abs = d.strftime("%Y-%m-%dT%H:%M:%S+02:00")
            else:
                unit = "bambi_flight98_point1"; ts_abs = ""
            rows.append(dict(image=p, unit=unit, ts_s=t, ts_kind=kind, ts_as_written=ts_abs))
        rows.sort(key=lambda r: (r["ts_s"], r["image"]))
        for i, r in enumerate(rows):
            r["idx"] = i
        (LISTS / f"{name}.txt").write_text("".join(r["image"] + "\n" for r in rows))
        rc.write_csv(NF / f"frames_{name}.csv", rows, ["idx", "image", "unit", "ts_s", "ts_kind", "ts_as_written"])
        # units: valid observed time per unit (PROTOCOL §3, 3x median cadence, cadence hint = set cadence)
        by_unit = defaultdict(list)
        for r in rows:
            by_unit[r["unit"]].append(r["ts_s"])
        urows = []
        for u, ts in sorted(by_unit.items()):
            v = rc.valid_observed_seconds(ts, gap_factor=3.0, cadence_hint=S["cadence_hint_s"])
            urows.append(dict(unit=u, n_frames=len(ts), first_ts=ts[0], last_ts=ts[-1], span_s=v["span_s"],
                              cadence_s=v["cadence_s"], gap_limit_s=v.get("gap_limit_s", 3 * S["cadence_hint_s"]),
                              valid_s=v["valid_s"], n_gaps_excluded=v["n_gaps_excluded"], excluded_s=v["excluded_s"]))
        rc.write_csv(NF / f"units_{name}.csv", urows, list(urows[0]))
        all_ts = [r["ts_s"] for r in rows]
        diffs = [b - a for a, b in zip(all_ts[:-1], all_ts[1:])]
        summary[name] = dict(n_frames=len(rows), n_units=len(urows), cadence_label=S["cadence_label"],
                             median_gap_s=statistics.median(diffs) if diffs else None,
                             valid_hours=round(sum(u["valid_s"] for u in urows) / 3600, 4),
                             valid_rule="frame covers its following gap if <= 3x cadence else one cadence; last frame one cadence (review_common.valid_observed_seconds)",
                             unit_rule=S["unit_rule"], first=rows[0]["ts_as_written"] or rows[0]["image"],
                             last=rows[-1]["ts_as_written"] or rows[-1]["image"],
                             list=str(LISTS / f"{name}.txt"), built_utc=utc_now())
        print(f"[lists] {name}: {len(rows)} frames, {len(urows)} units, valid {summary[name]['valid_hours']} h "
              f"({S['cadence_label']}) -> {LISTS / (name + '.txt')}")
    rc.write_json(NF / "cadence_summary.json", summary)


# ----------------------------------------------------------------------------- scoring
def _list_paths(name, list_file=None):
    f = Path(list_file) if list_file else LISTS / f"{name}.txt"
    return [l.strip() for l in open(f) if l.strip()]


def _shard(paths, spec):
    i, n = (int(x) for x in spec.split("/"))
    return paths[i::n], i, n


def _write_part(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def cmd_score_yolo(args):
    import torch
    torch.set_num_threads(args.threads)
    from ultralytics import YOLO
    sysd = SYSTEMS[args.system]; assert sysd["kind"] == "yolo"
    paths, i, n = _shard(_list_paths(args.set, args.list), args.shard)
    out = PARTS / f"{args.system}__{args.set}.part{i}of{n}.jsonl"
    if out.exists() and not args.force:
        print("exists, skip", out); return
    model = YOLO(str(ROOT / sysd["weights"]))
    recs = []; t0 = time.time(); B = 8
    for k in range(0, len(paths), B):
        batch = paths[k:k + B]
        # identical to cluster/v4/yolo_eval_entry.sh and scripts/v5/calibrate_aerial.score_yolo (imgsz 1280, conf 0.02,
        # default NMS iou 0.7 / max_det 300), device cpu
        res = model.predict(batch, imgsz=1280, conf=CONF_FLOOR_INFER, device="cpu", verbose=False)
        for p, r in zip(batch, res):
            h, w = r.orig_shape
            preds = [[round(float(b[0]) / w, 6), round(float(b[1]) / h, 6), round(float(b[2]) / w, 6),
                      round(float(b[3]) / h, 6), round(float(s), 6)]
                     for b, s in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist())]
            recs.append(dict(image=p, w=int(w), h=int(h), preds=preds))
        if (k // B) % 20 == 0:
            print(f"[{args.system} {args.set} shard {i}/{n}] {k + len(batch)}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    _write_part(out, recs)
    print(f"[{args.system} {args.set} shard {i}/{n}] wrote {out} {len(recs)} frames in {time.time() - t0:.0f}s "
          f"({(time.time() - t0) / max(1, len(recs)):.2f} s/frame)")


def cmd_score_dino(args):
    import cv2
    import torch
    torch.set_num_threads(args.threads)
    from src.serve.backend_dino3 import Dino3Backend
    sysd = SYSTEMS[args.system]; assert sysd["kind"] == "dino"
    paths, i, n = _shard(_list_paths(args.set, args.list), args.shard)
    out = PARTS / f"{args.system}__{args.set}.part{i}of{n}.jsonl"
    if out.exists() and not args.force:
        print("exists, skip", out); return
    # lora_merge is function-preserving (reports/v4/BACKEND_DINO3_LORA.md); conf_floor 0.02 as dino_infer.py
    b = Dino3Backend(str(ROOT / sysd["weights"]), device="cpu", conf_floor=CONF_FLOOR_INFER, lora_merge=True)
    print("backend", json.dumps(b.info())[:300], flush=True)
    recs = []; t0 = time.time()
    for k, p in enumerate(paths):
        img = cv2.imread(p); h, w = img.shape[:2]
        dets, _ = b.detect(img)
        recs.append(dict(image=p, w=int(w), h=int(h),
                         preds=[[round(d.x1 / w, 6), round(d.y1 / h, 6), round(d.x2 / w, 6), round(d.y2 / h, 6),
                                 round(float(d.score), 6)] for d in dets]))
        if k % 10 == 0:
            print(f"[{args.system} {args.set} shard {i}/{n}] {k + 1}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    _write_part(out, recs)
    print(f"[{args.system} {args.set} shard {i}/{n}] wrote {out} {len(recs)} in {time.time() - t0:.0f}s")


def cmd_merge(args):
    """Merge shard parts (or a cluster preds jsonl via --from) into models/v5/evals/<S>__<SET>.jsonl in list order."""
    sysd = SYSTEMS[args.system]
    paths = _list_paths(args.set)
    recs = {}
    srcs = [Path(args.from_file)] if args.from_file else sorted(PARTS.glob(f"{args.system}__{args.set}.part*.jsonl"))
    for src in srcs:
        for r in rc._open_maybe_gz(str(src)):
            r = r.strip()
            if not r:
                continue
            o = json.loads(r)
            recs[o["image"]] = o
    missing = [p for p in paths if p not in recs]
    out = EVALS / f"{args.system}__{args.set}.jsonl"
    EVALS.mkdir(parents=True, exist_ok=True)
    meta = dict(system=args.system, label=sysd["label"], weights=sysd["weights"], set=args.set, source_tag=SOURCE_TAG,
                infer=("cluster L40S dino_infer.py 1280x736 --lora 16" if sysd["kind"] == "dino" and args.from_file
                       else "CPU"), conf_floor=CONF_FLOOR_INFER, n_frames=len(paths) - len(missing), n_missing=len(missing),
                merged_utc=utc_now(), sources=[str(s) for s in srcs])
    with open(out, "w") as f:
        for p in paths:
            if p in recs:
                o = recs[p]
                f.write(json.dumps(dict(image=p, w=o.get("w"), h=o.get("h"), preds=o.get("preds") or [],
                                        source_tag=SOURCE_TAG, system=args.system)) + "\n")
    rc.write_json(EVALS / f"{args.system}__{args.set}.meta.json", meta)
    print(f"[merge] {out}: {len(paths) - len(missing)} frames, missing {len(missing)}, from {len(srcs)} source(s)")
    if missing:
        print("  first missing:", missing[:3])


# ----------------------------------------------------------------------------- metrics
def engine_cfg(conf_single, conf_seq, base):
    return SimpleNamespace(single_frame_alert_conf=conf_single, seq_frame_conf=conf_seq, seq_window=base.seq_window,
                           seq_hits_required=base.seq_hits_required, cooldown_s=base.cooldown_s,
                           stale_after_s=base.stale_after_s, scene_change_hamming=base.scene_change_hamming)


def episodes_in_run(states_ts, cooldown):
    """scripts/v5/calibrate_fixed.episodes_in_run: new episode iff ts - last_alert_ts >= cooldown."""
    n = 0; last = None
    for ts, st in states_ts:
        if st in ALERT_STATES:
            if last is None or ts - last >= cooldown:
                n += 1
            last = ts
    return n


def cmd_metrics(args):
    S = SETS[args.set]; scenario = S["scenario"]; floor = SCENARIO_FLOOR[scenario]
    base = Settings()
    frames = rc.read_csv(NF / f"frames_{args.set}.csv")
    units = rc.read_csv(NF / f"units_{args.set}.csv")
    unit_valid = {u["unit"]: float(u["valid_s"]) for u in units}
    by_unit = defaultdict(list)
    for r in frames:
        by_unit[r["unit"]].append((float(r["ts_s"]), r["image"]))
    cad = S["cadence_hint_s"]; gap_lim = 3 * cad
    out_rows = []; summary = dict(set=args.set, scenario=scenario, cadence=S["cadence_label"], engine=dict(
        seq_window=base.seq_window, seq_hits_required=base.seq_hits_required, cooldown_s=base.cooldown_s,
        stale_after_s=base.stale_after_s, floor_before_engine=floor,
        note=("MONIMET: cadence 1800 s > stale_after_s 600 s -> history resets every frame; only the single-frame "
              "provisional path can fire; episode == alert frame; rate is a 30-min SAMPLED rate" if scenario == "fixed"
              else "1 fps, one device per unit, fresh engine")), systems={}, computed_utc=utc_now(),
        caveat="candidate FP on UNVERIFIED footage: detector output is not evidence of smoke or of its absence")
    frame_scores = {}   # system -> {image: max}
    for system in systems_for(scenario):
        pj = EVALS / f"{system}__{args.set}.jsonl"
        if not pj.exists():
            print(f"[metrics] {system}: no preds at {pj} -> skipped"); summary["systems"][system] = dict(status="NOT SCORED"); continue
        preds = rc.load_preds(str(pj))
        cs, cq = SYSTEMS[system][scenario]
        cfg = engine_cfg(cs, cq, base)
        mx = {}
        for img, pr in preds.items():
            mx[img] = max((p[4] for p in pr if p[4] >= floor), default=0.0)
        frame_scores[system] = mx
        tot = dict(frames=0, scored=0, alert_frames=0, single_frames=0, confirmed_frames=0, episodes=0, valid_s=0.0,
                   alert_s=0.0, units=0, units_with_alert=0, units_timed=0)
        for u, lst in sorted(by_unit.items()):
            lst.sort()
            eng = AlertEngine(cfg); sts = []; n_scored = 0
            for ts, img in lst:
                if img not in mx:
                    continue
                n_scored += 1
                st = eng.update(u, ts, mx[img])["state"]
                sts.append((ts, st, mx[img]))
            n_alert = sum(1 for _, st, _ in sts if st in ALERT_STATES)
            n_conf = sum(1 for _, st, _ in sts if st == "confirmed")
            n_single = sum(1 for _, _, m in sts if m >= cs)
            ep = episodes_in_run([(t, s) for t, s, _ in sts], cfg.cooldown_s)
            # time in alert: an alert frame covers its following gap if <= 3x cadence else one cadence (same convention as valid_s)
            at = 0.0
            for j, (t, st, _) in enumerate(sts):
                if st in ALERT_STATES:
                    gap = (sts[j + 1][0] - t) if j + 1 < len(sts) else cad
                    at += gap if gap <= gap_lim else cad
            vs = unit_valid[u]
            out_rows.append(dict(system=system, unit=u, n_frames=len(lst), n_scored=n_scored, conf_single=cs, conf_seq=cq,
                                 alert_frames=n_alert, confirmed_frames=n_conf, frame_alert_rate=round(n_alert / max(1, n_scored), 4),
                                 episodes=ep, valid_h=round(vs / 3600, 4),
                                 episodes_per_valid_h=round(ep / (vs / 3600), 3) if vs > 0 else "",
                                 frac_time_in_alert=round(at / vs, 4) if vs > 0 else "",
                                 max_score=round(max((m for _, _, m in sts), default=0.0), 4),
                                 caveat="candidate FP on UNVERIFIED footage"))
            tot["frames"] += len(lst); tot["scored"] += n_scored; tot["alert_frames"] += n_alert; tot["single_frames"] += n_single
            tot["confirmed_frames"] += n_conf; tot["episodes"] += ep; tot["valid_s"] += vs; tot["alert_s"] += at
            tot["units"] += 1; tot["units_with_alert"] += int(n_alert > 0); tot["units_timed"] += int(vs > 0)
        vh = tot["valid_s"] / 3600
        summary["systems"][system] = dict(
            label=SYSTEMS[system]["label"], conf_single=cs, conf_seq=cq, preds=str(pj),
            frames=tot["frames"], scored=tot["scored"], alert_frames=tot["alert_frames"],
            frame_alert_rate=round(tot["alert_frames"] / max(1, tot["scored"]), 4),
            frames_ge_conf_single=tot["single_frames"], confirmed_frames=tot["confirmed_frames"],
            episodes=tot["episodes"], valid_h=round(vh, 4),
            episodes_per_valid_h=round(tot["episodes"] / vh, 3) if vh > 0 else None,
            frac_time_in_alert=round(tot["alert_s"] / tot["valid_s"], 4) if tot["valid_s"] > 0 else None,
            units=tot["units"], units_with_alert=tot["units_with_alert"],
            rate_kind=("SAMPLED (30-min cadence; episode == alert frame)" if scenario == "fixed" else "1 fps continuous"),
            caveat=summary["caveat"])
        print(f"[metrics] {args.set} {system} @ {cs}/{cq}: alert frames {tot['alert_frames']}/{tot['scored']} "
              f"({100 * tot['alert_frames'] / max(1, tot['scored']):.2f} %), episodes {tot['episodes']} / {vh:.3f} valid h "
              f"= {tot['episodes'] / vh if vh else float('nan'):.2f}/h, time-in-alert {tot['alert_s'] / max(1e-9, tot['valid_s']):.4f}, "
              f"units with >=1 alert {tot['units_with_alert']}/{tot['units']}  [candidate FP, UNVERIFIED]")
    # per-frame table (all systems) for the review export
    cols = ["idx", "image", "unit", "ts_s", "ts_as_written"] + [f"score_{s}" for s in frame_scores] + \
           [f"alert_{s}" for s in frame_scores] + ["n_systems_alert"]
    frows = []
    for r in frames:
        row = dict(idx=r["idx"], image=r["image"], unit=r["unit"], ts_s=r["ts_s"], ts_as_written=r["ts_as_written"])
        n = 0
        for s, mx in frame_scores.items():
            m = mx.get(r["image"])
            row[f"score_{s}"] = "" if m is None else round(m, 4)
            a = int(m is not None and m >= SYSTEMS[s][scenario][0])
            row[f"alert_{s}"] = a; n += a
        row["n_systems_alert"] = n
        frows.append(row)
    rc.write_csv(NF / f"frame_scores_{args.set}.csv", frows, cols)
    rc.write_csv(NF / f"metrics_{args.set}_units.csv", out_rows, list(out_rows[0]) if out_rows else ["system"])
    rc.write_json(NF / f"metrics_{args.set}_summary.json", summary)
    print(f"[metrics] wrote {NF / f'metrics_{args.set}_summary.json'}")


# ----------------------------------------------------------------------------- review lists
def cmd_review_lists(args):
    """Select frames for the human: every frame that alerts for >=1 system at its operating point (candidate FPs),
    plus the top-K highest-scoring non-alerting frames per unit (spot check), in time order; grouped into review units
    (MONIMET: per month; BAMBI: whole segment). Writes .txt lists + a negative_review.py manifest + per-frame verdict template."""
    S = SETS[args.set]; scenario = S["scenario"]
    frows = rc.read_csv(NF / f"frame_scores_{args.set}.csv")
    systems = [c[len("score_"):] for c in frows[0] if c.startswith("score_")]
    REVIEW.mkdir(parents=True, exist_ok=True); ldir = REVIEW / "_lists"; ldir.mkdir(exist_ok=True)
    sel = {}
    by_unit = defaultdict(list)
    for r in frows:
        by_unit[r["unit"]].append(r)
    for u, lst in by_unit.items():
        for r in lst:
            if int(r["n_systems_alert"]) > 0:
                sel[r["image"]] = (r, "alert_any")
        # top-K non-alerting by max normalised rank: use the max over YOLO scores (comparable scale), DINO as tie-break
        def key(r):
            ys = [float(r[f"score_{s}"] or 0) for s in systems if SYSTEMS[s]["kind"] == "yolo"]
            ds = [float(r[f"score_{s}"] or 0) for s in systems if SYSTEMS[s]["kind"] == "dino"]
            return (max(ys, default=0.0), max(ds, default=0.0))
        rest = sorted((r for r in lst if r["image"] not in sel), key=key, reverse=True)
        for r in rest[:args.top_k_per_unit]:
            sel[r["image"]] = (r, "top_nonalert")
    # review units
    def review_unit(r):
        if args.set == "monimet_mc124":
            return "monimet_mc124_" + r["ts_as_written"][:7].replace("-", "")   # YYYYMM
        return "bambi_flight98_point1"
    if args.set == "bambi_flight98" and args.all_bambi:
        for r in frows:
            if r["image"] not in sel:
                sel[r["image"]] = (r, "all_frames")
    groups = defaultdict(list)
    for img, (r, why) in sel.items():
        groups[review_unit(r)].append((float(r["ts_s"]), r, why))
    man_rows = []; verdict_rows = []
    lic = {"monimet_mc124": "CC BY 4.0 (Zenodo 10.5281/zenodo.7500295)", "bambi_flight98": "CC BY 4.0 (Zenodo 10.5281/zenodo.19177615)"}[args.set]
    for ru, lst in sorted(groups.items()):
        lst.sort(key=lambda x: (x[0], x[1]["image"]))
        lp = ldir / f"{ru}.txt"
        lp.write_text("".join(r["image"] + "\n" for _, r, _ in lst))
        man_rows.append(dict(unit=ru, path=str(lp), scenario=("fixed_camera" if scenario == "fixed" else "aerial"),
                             source=args.set, independent_unit=("MC124_2020" if args.set == "monimet_mc124" else "bambi_flight98"),
                             license=lic, cadence_s=("" if args.set == "monimet_mc124" else "1"), sample_s="",
                             contamination="none (no model has trained on this source)",
                             notes=f"UNVERIFIED no-smoke candidate; {sum(1 for x in lst if x[2] == 'alert_any')} frames alert for >=1 system at the Day-1 operating point, "
                                   f"{sum(1 for x in lst if x[2] != 'alert_any')} spot-check frames; selected subset, not the whole unit"))
        for k, (ts, r, why) in enumerate(lst):
            verdict_rows.append(dict(review_unit=ru, idx_in_unit=k, image=r["image"], camera_day_or_unit=r["unit"],
                                     ts_as_written=r["ts_as_written"] or f"t=+{int(float(r['ts_s']))}s", why=why,
                                     n_systems_alert=r["n_systems_alert"],
                                     systems_alerting=";".join(s for s in systems if r[f"alert_{s}"] == "1"),
                                     **{f"score_{s}": r[f"score_{s}"] for s in systems},
                                     verdict_smoke_yes_no_uncertain="", confounder_if_no="", reviewer="", reviewed_utc=""))
    mp = REVIEW / f"_inputs_manifest_{args.set}.csv"
    rc.write_csv(mp, man_rows, list(man_rows[0]))
    vp = REVIEW / f"frame_verdicts_TEMPLATE_{args.set}.csv"
    rc.write_csv(vp, verdict_rows, list(verdict_rows[0]))
    print(f"[review-lists] {args.set}: {len(sel)} frames selected ({sum(1 for v in sel.values() if v[1] == 'alert_any')} alert-any, "
          f"{sum(1 for v in sel.values() if v[1] == 'top_nonalert')} spot-check, {sum(1 for v in sel.values() if v[1] == 'all_frames')} all) "
          f"in {len(groups)} review units -> {mp}, {vp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("lists").set_defaults(fn=cmd_lists)
    for nm, fn in (("score-yolo", cmd_score_yolo), ("score-dino", cmd_score_dino)):
        s = sp.add_parser(nm); s.set_defaults(fn=fn)
        s.add_argument("--system", required=True, choices=list(SYSTEMS)); s.add_argument("--set", required=True, choices=list(SETS))
        s.add_argument("--shard", default="0/1"); s.add_argument("--threads", type=int, default=8)
        s.add_argument("--list", default=None, help="alternative frame list (subset)"); s.add_argument("--force", action="store_true")
    s = sp.add_parser("merge"); s.set_defaults(fn=cmd_merge)
    s.add_argument("--system", required=True, choices=list(SYSTEMS)); s.add_argument("--set", required=True, choices=list(SETS))
    s.add_argument("--from", dest="from_file", default=None, help="cluster preds jsonl[.gz] instead of local parts")
    s = sp.add_parser("metrics"); s.set_defaults(fn=cmd_metrics); s.add_argument("--set", required=True, choices=list(SETS))
    s = sp.add_parser("review-lists"); s.set_defaults(fn=cmd_review_lists); s.add_argument("--set", required=True, choices=list(SETS))
    s.add_argument("--top-k-per-unit", type=int, default=2); s.add_argument("--all-bambi", action="store_true")
    args = ap.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
