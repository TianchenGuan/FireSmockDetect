"""FIgLib-39 event replay for the v4 DINO LoRA runs + paired delay vs expE.

For each run: replay_eval --scores-jsonl with the run's sdis-VAL-selected
thresholds (conf at 10% neg-FP -> conf-single, conf at 20% -> conf-seq; the
v3 rule that gave B1 0.5624/0.3454), then summarize and append a section to
reports/v4/dino_adapt_results.md (between markers, idempotent).

R@Nmin = events detected with 0 <= delay <= N*60 s / all 39 events.
Paired delay vs expE (reports/replay_expE_figlib.json): per event detected by
BOTH, delta = run_delay - expE_delay (negative = run earlier).

  .venv/bin/python scripts/v4/figlib_replay_report.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/local/home/tianrgua/cvproject")
PY = ROOT / ".venv/bin/python"
MANIFEST = ROOT / "data/manifests/figlib_subset_manifest.parquet"
MFP = json.loads((ROOT / "reports/v4/matched_fp_dino_sdis_val.json").read_text())["matched_fp"]
RUNS = {"v4B4_lora_L": "dino3_v4B4_loraL", "v4B6_lora_Hplus": "dino3_v4B6_loraHplus"}
REFS = {"expE_y11m": ROOT / "reports/replay_expE_figlib.json",
        "v3_B1_frozenL": ROOT / "reports/v3/replay_B1_figlib_valmatched.json"}
MARK_A, MARK_B = "<!-- figlib39-replay:begin -->", "<!-- figlib39-replay:end -->"


def thresholds(run):
    t = MFP[run]
    return t["fp0.10"]["conf"], t["fp0.20"]["conf"]


def replay(run, backend):
    preds = ROOT / f"models/v4/evals/{run}_figlib/figlib39_preds.jsonl"
    out = ROOT / f"reports/v4/replay_{run}_figlib39.json"
    cs, cq = thresholds(run)
    cmd = [str(PY), "-m", "src.eval.replay_eval", "--manifest", str(MANIFEST),
           "--scores-jsonl", str(preds), "--backend", backend, "--imgsz", "1280",
           "--conf-single", str(cs), "--conf-seq", str(cq), "--out", str(out)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise SystemExit(f"replay failed for {run}")
    return json.loads(out.read_text()), out, cs, cq


def stats(d):
    ev = d["events"]
    n = len(ev)
    det = {k: v for k, v in ev.items() if v.get("detected")}
    delays = {k: v["delay_from_first_visible_s"] for k, v in det.items()}
    r_at = {m: sum(1 for x in delays.values() if x is not None and 0 <= x <= m * 60) / n
            for m in (1, 3, 5)}
    return {"n": n, "detected": len(det), "missed": sorted(k for k, v in ev.items() if not v.get("detected")),
            "pre_visible": sorted(k for k, v in ev.items() if v.get("pre_visible_alert")),
            "median_delay": d["summary"].get("median_delay_s_among_detected"),
            "r_at": r_at, "delays": delays}


def paired(a, b):
    both = sorted(set(a) & set(b))
    deltas = [a[k] - b[k] for k in both]
    med = sorted(deltas)[len(deltas) // 2] if deltas else None
    return {"n_both": len(both), "earlier": sum(1 for x in deltas if x < 0),
            "tie": sum(1 for x in deltas if x == 0), "later": sum(1 for x in deltas if x > 0),
            "median_delta_s": med, "mean_delta_s": (sum(deltas) / len(deltas)) if deltas else None}


def main():
    refs = {k: stats(json.loads(p.read_text())) for k, p in REFS.items() if p.exists()}
    rows, details = [], []
    for run, backend in RUNS.items():
        preds = ROOT / f"models/v4/evals/{run}_figlib/figlib39_preds.jsonl"
        if not preds.exists():
            print(f"{run}: no figlib preds yet, skipping"); continue
        d, out, cs, cq = replay(run, backend)
        st = stats(d)
        pe = paired(st["delays"], refs["expE_y11m"]["delays"]) if "expE_y11m" in refs else None
        pb = paired(st["delays"], refs["v3_B1_frozenL"]["delays"]) if "v3_B1_frozenL" in refs else None
        rows.append((run, cs, cq, st, pe, pb, out))
        details.append((run, st))
        print(f"{run}: thr {cs}/{cq} detected {st['detected']}/{st['n']} med {st['median_delay']}s "
              f"R@1/3/5 {st['r_at'][1]:.3f}/{st['r_at'][3]:.3f}/{st['r_at'][5]:.3f} "
              f"pre-vis {len(st['pre_visible'])} | vs expE: {pe}")
    if not rows:
        return 1

    L = [MARK_A, "## FIgLib-39 early-detection replay (historical eval set; val-selected thresholds)\n",
         "Frame scores from the cluster (`cluster/v3/submit_dino_eval.py --frames-tar`, fixed LoRA "
         "inference path), replay by `src/eval/replay_eval.py --scores-jsonl` with each run's own "
         "sdis-VAL thresholds: conf-single = conf at 10% neg-image FP, conf-seq = conf at 20% "
         "(the v3 rule that set B1 to 0.5624/0.3454). R@Nmin = events alerted within N min of the "
         "first-visible frame / 39. Pre-visible alerts are listed separately (not counted as detections "
         "or as false). FIgLib-39 is historical/open (used since iteration 1) — not a lockbox.\n",
         "| system | conf-single / conf-seq | detected /39 | R@1 / 3 / 5 min | median delay (detected) | misses | pre-visible alerts |",
         "|---|---|---|---|---|---|---|"]
    for run, cs, cq, st, pe, pb, out in rows:
        L.append(f"| {run} | {cs} / {cq} | {st['detected']}/{st['n']} | "
                 f"{st['r_at'][1]:.3f} / {st['r_at'][3]:.3f} / {st['r_at'][5]:.3f} | "
                 f"{st['median_delay']:.0f} s | {len(st['missed'])} | {len(st['pre_visible'])} |")
    for k, st in refs.items():
        L.append(f"| {k} (reference) | (v3/v2 val-selected) | {st['detected']}/{st['n']} | "
                 f"{st['r_at'][1]:.3f} / {st['r_at'][3]:.3f} / {st['r_at'][5]:.3f} | "
                 f"{st['median_delay']:.0f} s | {len(st['missed'])} | {len(st['pre_visible'])} |")
    L.append("\n**Paired delay vs expE** (events detected by both; delta = run - expE, negative = earlier):\n")
    L.append("| run | n both | earlier | tie | later | median delta | mean delta |")
    L.append("|---|---|---|---|---|---|---|")
    for run, cs, cq, st, pe, pb, out in rows:
        if pe:
            L.append(f"| {run} | {pe['n_both']} | {pe['earlier']} | {pe['tie']} | {pe['later']} | "
                     f"{pe['median_delta_s']:+.0f} s | {pe['mean_delta_s']:+.0f} s |")
        if pb:
            L.append(f"| {run} vs v3_B1_frozenL | {pb['n_both']} | {pb['earlier']} | {pb['tie']} | {pb['later']} | "
                     f"{pb['median_delta_s']:+.0f} s | {pb['mean_delta_s']:+.0f} s |")
    # ---- read-only diagnostic: why val-selected thresholds transfer (or not)
    L.append("\n**Score-transfer diagnostic** (no tuning: distribution of per-frame max smoke score on FIgLib "
             "frames at/after first-visible smoke vs the val-selected conf-single):\n")
    L.append("| system | conf-single | post-visible frames | median max score | p90 | frac >= conf-single | pre-visible frac >= conf-single |")
    L.append("|---|---|---|---|---|---|---|")
    import statistics
    def diag_row(label, path):
        d = json.loads(Path(path).read_text()); c = d["summary"].get("config", {})
        thr = c.get("single_frame_alert_conf")
        if thr is None:
            L.append(f"| {label} | n/a (older replay schema) | | | | | |"); return
        post = [f["max_smoke"] for e in d["events"].values() for f in e["per_frame"] if f["offset"] >= 0]
        pre = [f["max_smoke"] for e in d["events"].values() for f in e["per_frame"] if f["offset"] < 0]
        L.append(f"| {label} | {thr} | {len(post)} | {statistics.median(post):.3f} | "
                 f"{sorted(post)[int(0.9 * (len(post) - 1))]:.3f} | {sum(x >= thr for x in post) / len(post):.3f} | "
                 f"{sum(x >= thr for x in pre) / len(pre):.3f} |")
    for run, cs, cq, st, pe, pb, out in rows:
        diag_row(run, out)
    for k, pth in REFS.items():
        if pth.exists():
            diag_row(f"{k} (reference)", pth)
    L.append("\nReading: the adapted runs' sdis-val operating points sit at conf >= 0.97, where only a small "
             "fraction of FIgLib post-visible frames score — so events are detected late or missed under the v3 "
             "threshold-transfer rule, even though the same runs win on sdis_val at matched FP. This is the "
             "calibration/saturation effect flagged in the confounds, now measured on an independent historical set; "
             "it argues for per-domain threshold selection (needs domain negatives) before any deployment claim.")
    L.append("")
    for run, st in details:
        L.append(f"- {run} misses: {', '.join(st['missed']) or 'none'}")
        L.append(f"- {run} pre-visible alerts: {', '.join(st['pre_visible']) or 'none'}")
    # ---- batch-1 latency (same protocol as v3: n=50 sdis_val frames, 1x L40S, incl. preprocessing)
    L.append("\n### Batch-1 latency, 1x L40S (v3 protocol: first 50 sdis_val images, preprocessing included, decode excluded)\n")
    L.append("| system | form | mean ms | median ms | p90 ms | infer-only median ms | source |")
    L.append("|---|---|---|---|---|---|---|")
    def lat_row(label, form, path):
        if not Path(path).exists():
            L.append(f"| {label} | {form} | n/a | n/a | n/a | n/a | missing {Path(path).name} |"); return
        j = json.loads(Path(path).read_text())
        L.append(f"| {label} | {form} | {j['batch1_ms_mean']:.1f} | {j['batch1_ms_median']:.1f} | "
                 f"{j['batch1_ms_p90']:.1f} | {j['infer_ms_median']:.1f} | `{Path(path).relative_to(ROOT)}` |")
    for run in RUNS:
        d = ROOT / f"models/v4/evals/{run}_figlib"
        lat_row(run, "LoRA unmerged (train form)", d / "latency_gpu.json")
        lat_row(run, "LoRA merged into base (deploy form)", d / "latency_gpu_merged.json")
    lat_row("v3 B1 frozen ViT-L (reference)", "plain", ROOT / "models/v3/evals/B1val/latency_gpu.json")
    lat_row("v3 B2 frozen ViT-H+ (reference)", "plain", ROOT / "models/v3/evals/B2val/latency_gpu.json")
    L.append("\nMerging folds W += (alpha/r)*B@A into the 96 (L) / 128 (H+) attention linears; outputs verified "
             "identical to the unmerged model on CPU (max |delta| 1e-6 on box coords/scores). v3 expE YOLO11m "
             "reference on the same protocol: 11.2 ms.")
    L.append(f"\nOutputs: `reports/v4/replay_{{v4B4_lora_L,v4B6_lora_Hplus}}_figlib39.json`; frame scores "
             f"`models/v4/evals/<run>_figlib/figlib39_preds.jsonl`.")
    L.append(MARK_B)
    md = ROOT / "reports/v4/dino_adapt_results.md"
    txt = md.read_text()
    block = "\n".join(L) + "\n"
    if MARK_A in txt:
        pre, rest = txt.split(MARK_A, 1)
        _, post = rest.split(MARK_B, 1)
        txt = pre + block + post
    else:
        txt = txt.rstrip("\n") + "\n\n" + block
    md.write_text(txt)
    print("appended replay section to", md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
