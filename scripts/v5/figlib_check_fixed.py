"""FIgLib-39 constraint CHECK (PROTOCOL_V5 §3: >= 30/39 events detected) for the
operating points chosen by scripts/v5/calibrate_fixed.py.

FIgLib-39 is a HISTORICAL set (open since iteration 1; HPWREN fixed cameras,
60 s cadence, no boxes). It is NOT used for selection — the points are chosen
on pyro-sdis VAL only; this script only checks them and records sensitivity.

Two kinds of numbers, kept apart:
  official   — YOLO: src/eval/replay_eval.py re-run with --weights at imgsz 1280
               (launched separately, outputs reports/v5/calib/figlib/replay_<sys>_chosen_*.json);
               DINO: replay_eval --scores-jsonl on the saved v4 frame scores
               (models/v4/evals/*_figlib/figlib39_preds.jsonl) — the only form available.
  diagnostic — replay_eval --scores-jsonl on per-frame max scores extracted from
               the saved 1280-px deployed-point replays (reports/v5/calib/figlib_scores/,
               scores rounded to 4 dp). Used for sensitivity rows and as a
               cross-check of the official runs; never as the constraint value.

Usage: python scripts/v5/figlib_check_fixed.py [--diag] [--out reports/v5/calib/figlib_check.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
MANIFEST = ROOT / "data/manifests/figlib_subset_manifest.parquet"
CALIB = ROOT / "reports/v5/calib"
FIG = CALIB / "figlib"
MIN_DETECTED = 30

SYSTEMS = {
    "expB": dict(kind="yolo", deployed_ref=ROOT / "reports/replay_expB_figlib.json",
                 diag_scores=CALIB / "figlib_scores/expB_figlib39_maxsmoke_from_replay1280.jsonl"),
    "expE": dict(kind="yolo", deployed_ref=ROOT / "reports/replay_expE_figlib.json",
                 diag_scores=CALIB / "figlib_scores/expE_figlib39_maxsmoke_from_replay1280.jsonl"),
    "v4a_s2": dict(kind="yolo", deployed_ref=ROOT / "reports/v4/replay/replay_v4a_s2_figlib39.json",
                   diag_scores=CALIB / "figlib_scores/v4a_s2_figlib39_maxsmoke_from_replay1280.jsonl"),
    "v4B4_lora_L": dict(kind="dino", scores=ROOT / "models/v4/evals/v4B4_lora_L_figlib/figlib39_preds.jsonl",
                        v4_refs={"v4 v3-rule 0.9881/0.973": ROOT / "reports/v4/replay_v4B4_lora_L_figlib39.json",
                                 "v4 expB-matched-negFP 0.93/0.82": FIG / "replay_v4B4_lora_L_v4valpt_0.93_0.82.json"}),
    "v4B6_lora_Hplus": dict(kind="dino", scores=ROOT / "models/v4/evals/v4B6_lora_Hplus_figlib/figlib39_preds.jsonl",
                            v4_refs={"v4 v3-rule 0.9901/0.9801": ROOT / "reports/v4/replay_v4B6_lora_Hplus_figlib39.json",
                                     "v4 expB-matched-negFP 0.95/0.88": FIG / "replay_v4B6_lora_Hplus_v4valpt_0.95_0.88.json"}),
}


def summ(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.load(open(path)); s = d["summary"]
    # events detected within N min of first visible smoke (misses stay in denominator)
    within = {}
    for n in (1, 3, 5):
        within[f"detected_within_{n}min"] = sum(1 for e in d["events"].values()
                                                if e["detected"] and e["delay_from_first_visible_s"] is not None
                                                and e["delay_from_first_visible_s"] <= 60 * n)
    return {"file": str(path.relative_to(ROOT)), "n_events": s["n_events"], "n_detected": s["n_detected"],
            "n_missed": s["n_missed"], "n_pre_visible_alerts": s["n_pre_visible_alerts"],
            "median_delay_s": s["median_delay_s_among_detected"], "mean_delay_s": (round(s["mean_delay_s_among_detected"], 1)
                                                                                 if s["mean_delay_s_among_detected"] is not None else None),
            "conf_single": s["config"]["single_frame_alert_conf"], "conf_seq": s["config"]["seq_frame_conf"],
            "imgsz": s["config"]["imgsz"], "weights": s["config"]["weights"], "scores_jsonl": s["config"].get("scores_jsonl"),
            "n_frames_without_score": s["config"].get("n_frames_without_score", 0), **within}


def run_scores(scores: Path, cs: float, cq: float, out: Path) -> dict:
    if not out.exists():
        cmd = [PY, "-m", "src.eval.replay_eval", "--manifest", str(MANIFEST), "--scores-jsonl", str(scores),
               "--imgsz", "1280", "--conf-single", f"{cs}", "--conf-seq", f"{cq}", "--out", str(out)]
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    return summ(out)


def fmt(cs, cq):
    return f"{cs:.4g}_{cq:.4g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true", help="also run saved-score sensitivity rows")
    ap.add_argument("--out", default=str(CALIB / "figlib_check.json"))
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    report = {"set": "FIgLib-39 (HISTORICAL; HPWREN fixed cameras; 39 events / 3,056 frames; open since v1)",
              "constraint": f"detected >= {MIN_DETECTED}/39 at the chosen operating point (PROTOCOL_V5 §3)",
              "systems": {}}
    for name, spec in SYSTEMS.items():
        cal = json.load(open(CALIB / f"{name}.json"))
        ch = cal["chosen_protocol"]; ea = cal["chosen_engine_aware"]
        cs, cq = ch["conf_single"], ch["conf_seq"]
        entry = {"kind": spec["kind"], "chosen_point": [cs, cq], "engine_aware_point": [ea["conf_single"], ea["conf_seq"]],
                 "deployed_reference": {}, "official": None, "diagnostic": {}}
        if spec["kind"] == "yolo":
            entry["deployed_reference"]["YOLO deployed 0.30/0.15 (v3/v4 replay, weights, imgsz 1280)"] = summ(spec["deployed_ref"])
            cands = sorted(FIG.glob(f"replay_{name}_chosen_*.json"))
            entry["official"] = summ(cands[0]) if cands else None
            entry["official_kind"] = "replay_eval --weights (re-inference, CPU, imgsz 1280)"
            if entry["official"] and (abs(entry["official"]["conf_single"] - cs) > 1e-9 or abs(entry["official"]["conf_seq"] - cq) > 1e-9):
                entry["official_warning"] = "official replay confs differ from calib chosen point"
            scores = spec["diag_scores"]
        else:
            for lbl, p in spec["v4_refs"].items():
                entry["deployed_reference"][lbl] = summ(p)
            entry["official"] = run_scores(spec["scores"], cs, cq, FIG / f"replay_{name}_chosen_{cs}_{cq}.json")
            entry["official_kind"] = "replay_eval --scores-jsonl on saved v4 cluster frame scores (1280x736 letterbox)"
            scores = spec["scores"]
        if entry["official"]:
            entry["constraint_pass"] = entry["official"]["n_detected"] >= MIN_DETECTED
        if args.diag and scores.exists():
            pts = {"chosen (saved-score cross-check)": (cs, cq), "engine_aware": (ea["conf_single"], ea["conf_seq"])}
            fr = {f["conf_single"]: f for f in cal["frontier"]}
            targets = ([0.25, 0.3, 0.35, 0.4, 0.5] if spec["kind"] == "yolo" else [0.85, 0.9, 0.93, 0.95, 0.97])
            for t in targets:
                if t in fr:
                    pts[f"frontier conf_single={t} (R_single={fr[t]['R_single']}, eps/h={fr[t]['eps_per_h']})"] = (t, fr[t]["conf_seq"])
            for lbl, (a, b) in pts.items():
                entry["diagnostic"][lbl] = run_scores(scores, a, b, FIG / f"diag_{name}_{fmt(a, b)}.json")
            entry["diagnostic_note"] = ("DIAGNOSTIC from saved per-frame scores (4-dp rounded for YOLO); sensitivity only; "
                                        "not used for selection (FIgLib is banned from tuning/selection, §2)")
        report["systems"][name] = entry
        cal["figlib_check"] = entry
        (CALIB / f"{name}.json").write_text(json.dumps(cal, indent=1))
        off = entry["official"]
        print(f"{name}: chosen {cs}/{cq} official={'PENDING' if not off else f'{off['n_detected']}/39 (pre-vis {off['n_pre_visible_alerts']}, med {off['median_delay_s']}s)'} "
              f"pass={entry.get('constraint_pass')} | diag rows {len(entry['diagnostic'])}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
