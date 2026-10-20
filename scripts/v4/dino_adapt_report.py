"""Assemble reports/v4/dino_adapt_results.md for the iteration-4 DINOv3
visual-adaptation track (v4B1 frozen / v4B4 LoRA-L / v4B5 full-L / v4B6 LoRA-H+).

Inputs (all produced earlier in the track; missing pieces are reported, never
fabricated):
  reports/v4/eval_<run>_<set>.raw.parquet     per-run per-valset sweeps (eval_detection)
  reports/v4/eval_v4{a,e,c}_<set>.raw.parquet  YOLO arm baselines (A ctrl-E, E E+T, C EGT)
  reports/eval_expE_val.raw.parquet            iteration-2 YOLO11m expE (sdis-val only)
  models/v4/dino_run_summary.json              segments / node-hours / steps / evals
  models/v4/param_counts_from_ckpt.json        trainable counts by group (from ckpts)
  models/v4/backbone_totals.json               HF backbone totals (meta-device count)

Steps: run scripts/v4/compare_matched_fp.py per valset over every model that
has a parquet (fine top grid: softmax-scored detectors saturate >0.99), then
for forest_val (0 negatives -> matched-FP undefined) report recall at the
confidence each model needs for the sdis-val FP targets ("sdis-anchored").

  .venv/bin/python scripts/v4/dino_adapt_report.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/local/home/tianrgua/cvproject")
PY = ROOT / ".venv/bin/python"
REP = ROOT / "reports/v4"
FP_TARGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
N_TRAIN = 51420

RUNS = {
    "v4B1_frozen_L": dict(label="B1 frozen ViT-L", backbone="dinov3-vitl16", mode="frozen (head+neck only)",
                          bs=4, lr_head="1e-4", lr_bb="-", ckpt="no", extra="(none)"),
    "v4B4_lora_L": dict(label="B4 LoRA-r16 ViT-L", backbone="dinov3-vitl16", mode="LoRA r16 a32 on q/k/v/o",
                        bs=4, lr_head="1e-4", lr_bb="1e-4", ckpt="yes", extra="--lora 16 --backbone-lr 1e-4 --grad-ckpt"),
    "v4B5_full_L": dict(label="B5 full-unfreeze ViT-L", backbone="dinov3-vitl16", mode="all blocks trainable",
                        bs=2, lr_head="1e-4", lr_bb="1e-5", ckpt="yes", extra="--unfreeze-all --grad-ckpt --backbone-lr 1e-5"),
    "v4B6_lora_Hplus": dict(label="B6 LoRA-r16 ViT-H+", backbone="dinov3-vith16plus", mode="LoRA r16 a32 on q/k/v/o",
                            bs=4, lr_head="1e-4", lr_bb="1e-4", ckpt="yes", extra="--lora 16 --backbone-lr 1e-4 --grad-ckpt"),
}
SETS = {"sdis_val": "sdisval", "boreal_val": "borealval", "forest_val": "forestval"}
BASELINES = {  # label -> (parquet pattern with {s} = yolo-arm suffix, sets)
    "YOLO_A_ctrlE": ("reports/v4/eval_v4a_{s}.raw.parquet", ("sdis_val", "boreal_val", "forest_val")),
    "YOLO_E_ET": ("reports/v4/eval_v4e_{s}.raw.parquet", ("sdis_val", "boreal_val", "forest_val")),
    "YOLO_C_EGT": ("reports/v4/eval_v4c_{s}.raw.parquet", ("sdis_val", "boreal_val", "forest_val")),
    "expE_y11m_iter2": ("reports/eval_expE_val.raw.parquet", ("sdis_val",)),
    "v3_B1_frozenL_Eonly12ep": ("reports/v3/eval_B1_val.raw.parquet", ("sdis_val",)),
    "v3_B3_top4L_Eonly12ep": ("reports/v3/eval_B3_val.raw.parquet", ("sdis_val",)),
}


def fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def na_cell(curve, target):
    """Explain an out-of-range FP target: below the model's FP floor (conf
    floor 0.02 still yields more FP than target) vs above its FP ceiling (the
    model never exceeds the target FP even at conf floor -> recall is already
    maximal there)."""
    fps = [(c["neg_image_fp_rate"], c["recall"]) for c in curve
           if c["neg_image_fp_rate"] is not None and c["recall"] is not None]
    if not fps:
        return "n/a"
    lo, hi = min(fps), max(fps)
    if target > hi[0]:
        return f"FP ceiling {hi[0]:.1%} -> R>={hi[1]:.3f}"
    if target < lo[0]:
        return f"n/a (FP floor {lo[0]:.1%})"
    return "n/a"


def interp_curve(curve, conf):
    """recall / tiny_recall at an arbitrary conf from a fine sweep (linear)."""
    pts = sorted((c["conf"], c["recall"], c["recall_by_size"]["tiny"]) for c in curve
                 if c["recall"] is not None)
    if not pts or conf is None:
        return None, None
    if conf <= pts[0][0]:
        return pts[0][1], pts[0][2]
    if conf >= pts[-1][0]:
        return pts[-1][1], pts[-1][2]
    for (c1, r1, t1), (c2, r2, t2) in zip(pts, pts[1:]):
        if c1 <= conf <= c2:
            w = 0 if c2 == c1 else (conf - c1) / (c2 - c1)
            return r1 + w * (r2 - r1), (None if t1 is None or t2 is None else t1 + w * (t2 - t1))
    return None, None


def main():
    summary = json.loads((ROOT / "models/v4/dino_run_summary.json").read_text())
    pc = json.loads((ROOT / "models/v4/param_counts_from_ckpt.json").read_text())
    bt = json.loads((ROOT / "models/v4/backbone_totals.json").read_text()) \
        if (ROOT / "models/v4/backbone_totals.json").exists() else {}

    # ---- matched-FP per valset --------------------------------------------
    mfp = {}
    for vs, ysuf in SETS.items():
        specs = []
        for run in RUNS:
            p = REP / f"eval_{run}_{vs}.raw.parquet"
            if p.exists():
                specs.append(f"{run}={p}")
        for lab, (pat, sets) in BASELINES.items():
            if vs in sets:
                p = ROOT / pat.format(s=ysuf)
                if p.exists():
                    specs.append(f"{lab}={p}")
        if not specs:
            continue
        out = REP / f"matched_fp_dino_{vs}.json"
        cmd = [str(PY), str(ROOT / "scripts/v4/compare_matched_fp.py"), "--top-fine",
               "--fp-targets", ",".join(str(t) for t in FP_TARGETS), "--out", str(out)]
        for s in specs:
            cmd += ["--model", s]
        print(f"[{vs}] {len(specs)} models -> {out.name}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit(f"compare_matched_fp failed for {vs}")
        mfp[vs] = json.loads(out.read_text())

    # ---- markdown ------------------------------------------------------------
    L = []
    L.append("# Iteration 4 — DINOv3 visual adaptation on the v4 E+T recipe (VAL results)\n")
    L.append(f"_generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')} by "
             f"scripts/v4/dino_adapt_report.py — validation only; no test/lockbox data touched._\n")
    L.append("## Headline (val only; see confounds below)\n")
    L.append("- **Adapting the backbone helps on sdis_val at every matched-FP level.** At 5% neg-image FP: "
             "B6 LoRA-H+ R 0.227 / tiny 0.144, B4 LoRA-L 0.210 / 0.144, B5 full-L 0.209 / 0.131 vs frozen B1 "
             "0.175 / 0.104 and the best YOLO arm (A ctrl-E) 0.204 / 0.112. At 1-2% FP the LoRA runs lead all "
             "YOLO arms and the v3 E-only DINO runs (B6 0.097 / B4 0.085 vs YOLO-A 0.053, expE 0.059 at 1%). "
             "B5 cannot reach 1% FP on sdis_val (floor 1.4%).")
    L.append("- **H+ LoRA (B6) is the best single config on sdis_val** (also best COCO mAP 0.422) but costs 2.1x "
             "the node-hours of B4 (8.0 vs 3.8) and ~2.4x its inference latency class (v3: 371 vs 156 ms b1 L40S). "
             "B4 LoRA-L gets most of the gain for +3.1M trainable params and the same wall time as a frozen run "
             "would with ckpt (3.8 node-h vs 1.7 for B1).")
    L.append("- **Full unfreeze (B5) does not beat LoRA** despite 100x the trainable params and 2.2x the node-hours; "
             "its bs-2/16 schedule and end-only checkpoint are confounds (below), but there is no evidence it is worth it.")
    L.append("- **boreal_val (aerial burn, large plumes)**: every DINO run sits at R 0.96-0.99 by 1-5% FP; YOLO arms "
             "trained with T data (E/C) reach 0.93-0.95, the E-only YOLO control 0.64-0.83. Differences among DINO "
             "runs here are within a handful of images (217 negatives, 906 positives).")
    L.append("- **forest_val (no negatives)**: at a fixed conf 0.50 adapted runs recall far more (B6 0.553, B4 0.519, "
             "B5 0.489 vs B1 0.383, YOLO 0.07-0.12), but under sdis-anchored thresholds the frozen B1 transfers best "
             "because adapted scores saturate near 1.0 — a calibration effect, not a detection-quality ranking. "
             "Forest false-alarm behavior is unmeasured (no forest negatives exist in the package).")
    L.append("- **Incidents**: no training OOM/retries at the planned batch sizes. One EVAL incident: B4's first val "
             "eval job crashed in every shard (`ckpt mismatch ... unexpected lora_A/B`) because `dino_infer.py` built "
             "a plain backbone; fixed by adding `--lora R` injection (CPU-verified against B4/B5 checkpoints), "
             "an empty-preds hard-fail in the driver, and re-running (bogus empty sweeps were deleted; the failed "
             "eval dir is kept as `models/v3/evals/v4B4_lora_L_val.FAILED_lora_keys`). Cost: +0.2 node-h.\n")
    L.append("## Setup\n")
    L.append(f"- Train corpus: frozen `data/v4/yolo/E_ET_v1` -> `data/v4/coco/instances_train_eet.json` "
             f"({N_TRAIN:,} images: E 38,454 / T 12,966; 48,480 boxes; 0 label files missing; "
             f"7 E boxes dropped <1px after clamping). file_name = absolute pod paths.")
    L.append("- Val sets: `sdis_val` (7,335 img / 1,158 neg; v3 json unchanged), `boreal_val` = burn_heinola "
             "(1,123 img / 217 explicit-empty neg / 917 boxes, **902 of 917 GT boxes are 'large' >5% area**), "
             "`forest_val` = AuSmoke forest-holdout VAL units (1,471 img / **0 negatives** / 1,975 mask-derived boxes). "
             "Package: `data/v4/stage/v4_evalval.tar` staged once at `/scratch/.../smoke/v4data/v4_evalval/`.")
    L.append("- Common: 1280x736 letterbox, 10 epochs planned, warmup 500, wd 1e-4, eval-max 1500, "
             "8x L40S per run, segmented 1h jobs via `cluster/v3/orchestrate_dino.py`, image smoke-dino3:v1.")
    L.append("- Inference: `cluster/v3/dino_infer.py` (multi-root `--root-map` added), best.pt (COCO-mAP-selected on the "
             "1500-img sdis-val subset), conf floor 0.02; sweeps by `src/eval/eval_detection.py` (IoU>=0.4) and "
             "`scripts/v4/compare_matched_fp.py --top-fine`.\n")

    L.append("## Runs\n")
    L.append("| run | backbone | adaptation | bs/GPU (eff.) | lr head / backbone | grad-ckpt | extra args |")
    L.append("|---|---|---|---|---|---|---|")
    for run, c in RUNS.items():
        L.append(f"| {run} | {c['backbone']} | {c['mode']} | {c['bs']} ({c['bs']*8}) | "
                 f"{c['lr_head']} / {c['lr_bb']} | {c['ckpt']} | `{c['extra']}` |")

    L.append("\n## Parameter counts (from checkpoints + HF config meta-count)\n")
    L.append("| run | backbone trainable / total | neck | head | ALL trainable |")
    L.append("|---|---|---|---|---|")
    for run, c in RUNS.items():
        d = pc.get(run, {})
        g = d.get("by_group", {})
        tot = bt.get(c["backbone"], {}).get("total")
        L.append(f"| {run} | {g.get('backbone', 0):,} / {tot:,} | {g.get('neck', 0):,} | "
                 f"{g.get('head', 0):,} | {d.get('ckpt_trainable_total', 0):,} |"
                 if tot else
                 f"| {run} | {g.get('backbone', 0):,} / (total n/a) | {g.get('neck', 0):,} | "
                 f"{g.get('head', 0):,} | {d.get('ckpt_trainable_total', 0):,} |")
    L.append("\nLoRA r16 = layers x 4 proj x 16 x (d+d): ViT-L 24x4x16x2048 = 3,145,728; "
             "ViT-H+ 32x4x16x2560 = 5,242,880. Full unfreeze leaves only the unused 1,024-param mask_token frozen.\n")

    L.append("## Training outcome / cost\n")
    L.append("| run | status | segments | node-hours (8 L40S) | steps | epochs achieved (exposures) | best COCO mAP (1500-img sdis-val) @step | OOM / retries |")
    L.append("|---|---|---|---|---|---|---|---|")
    for run, c in RUNS.items():
        s = summary.get(run, {})
        step = s.get("last_step") or 0
        expo = step * c["bs"] * 8
        ep = expo / N_TRAIN
        evs = s.get("evals", [])
        best = max(evs, key=lambda e: e.get("mAP", -1)) if evs else None
        best_s = f"{best['mAP']:.3f} @{best['step']}" if best else "n/a (no epoch-boundary eval)"
        status = "complete" if s.get("complete") else f"INCOMPLETE ({step}/{s.get('total_steps')})"
        L.append(f"| {run} | {status} | {s.get('n_segments_submitted')} | {s.get('node_hours')} | "
                 f"{step}/{s.get('total_steps')} | {ep:.2f} ({expo:,}) | {best_s} | "
                 f"{'none' if not s.get('oom_segments') else s['oom_segments']} |")
    L.append("\nAll four first segments exited rc=0 at the planned batch sizes — no OOM, no retries, "
             "no fallbacks used. Node-hours = sum of orchestrator-observed segment wall times "
             "(includes pod start, code/json fetch, eval passes).\n")

    L.append("## Val matched-FP results (recall / tiny-recall at neg-image FP rate)\n")
    L.append("Cell = `R / tinyR (c=conf)` interpolated on each model's own confidence curve at the FP target. "
             "`FP ceiling X% -> R>=r`: even at the 0.02 conf floor the model produces only X% neg-image FP, so its "
             "recall at any higher FP target is at least r (already maximal). `n/a (FP floor X%)`: the model cannot "
             "get below X% FP on this split.\n")
    for vs in ("sdis_val", "boreal_val"):
        if vs not in mfp:
            L.append(f"### {vs}\n\n_no results yet_\n")
            continue
        m = mfp[vs]
        if not m.get("matched_fp"):
            L.append(f"### {vs}\n\n_matched-FP undefined (no negatives)_\n")
            continue
        labels = list(m["matched_fp"].keys())
        nneg = {k: v["n_negative_images"] for k, v in m["models"].items()}
        L.append(f"### {vs}  (negatives: {nneg[labels[0]]})\n")
        L.append("| FP(neg-img) | " + " | ".join(labels) + " |")
        L.append("|---|" + "---|" * len(labels))
        for t in FP_TARGETS:
            cells = []
            for lab in labels:
                d = m["matched_fp"][lab][f"fp{t:.2f}"]
                cells.append(na_cell(m["models"][lab]["curve"], t) if d["recall"] is None else
                             f"{fmt(d['recall'])} / {fmt(d['tiny_recall'])} (c={d['conf']:.3f})")
            L.append(f"| {t:.0%} | " + " | ".join(cells) + " |")
        L.append("")
        if vs == "boreal_val":
            L.append("boreal_val has 1 tiny + 5 small GT boxes in total — tiny-recall there is noise; "
                     "read only overall recall. With 217 negatives, 1% FP = 2 images.\n")

    # forest_val: sdis-anchored thresholds
    L.append("### forest_val  (0 negatives -> matched-FP undefined; sdis-anchored thresholds)\n")
    if "forest_val" in mfp and "sdis_val" in mfp:
        fm, sm = mfp["forest_val"], mfp["sdis_val"]
        # forest_val has no negatives -> compare_matched_fp leaves matched_fp
        # null/empty; only its per-model curves are usable here.
        labels = [l for l in fm["models"] if l in (sm.get("matched_fp") or {})]
        L.append("Threshold = the confidence at which the SAME model sits at the given neg-image FP rate on "
                 "sdis_val; recall / tiny-recall on forest_val evaluated at that threshold. "
                 "Last row: fixed conf 0.50 for reference.\n")
        L.append("| sdis-FP anchor | " + " | ".join(labels) + " |")
        L.append("|---|" + "---|" * len(labels))
        for t in FP_TARGETS:
            cells = []
            for lab in labels:
                conf = sm["matched_fp"][lab][f"fp{t:.2f}"]["conf"]
                r, tr = interp_curve(fm["models"][lab]["curve"], conf)
                cells.append("n/a" if r is None else f"{fmt(r)} / {fmt(tr)} (c={conf:.3f})")
            L.append(f"| {t:.0%} | " + " | ".join(cells) + " |")
        cells = []
        for lab in labels:
            r, tr = interp_curve(fm["models"][lab]["curve"], 0.5)
            cells.append("n/a" if r is None else f"{fmt(r)} / {fmt(tr)}")
        L.append("| conf=0.50 | " + " | ".join(cells) + " |")
        L.append("")
    else:
        L.append("_no results yet_\n")

    L.append("## Confounds and caveats (read before comparing)\n")
    L.append("- **Batch size**: B5 (full unfreeze) trains at bs 2/GPU (eff. 16) vs bs 4/GPU (eff. 32) for B1/B4/B6 -> "
             "2x the optimizer steps for the same 10 epochs (32,140 vs 16,070), different warmup fraction, "
             "different effective noise. B5-vs-others is confounded by batch size and LR schedule granularity.")
    L.append("- **Backbone LR**: LoRA runs use 1e-4 on the adapters, B5 uses 1e-5 on all weights; not a like-for-like "
             "LR sweep — each is the notes' recommended default for its mode.")
    L.append("- **Checkpoint selection**: best.pt is picked by COCO mAP on a 1,500-image sdis-val subset at epoch "
             "boundaries. B5's dataloader pass (3,214 steps at bs 2) never completes inside a ~50-min segment, "
             "so B5 evaluates ONLY at the end -> its best.pt == final weights, while B1/B4/B6 got several "
             "epoch-boundary evals and may benefit from selection. Selecting on val also mildly optimistic-biases "
             "every sdis-val number here (same as v3).")
    L.append("- **Segment resume**: on resume the sampler restarts the current epoch's permutation from the start "
             "(v3 behavior), so per-image exposure is uneven across the corpus; epochs above are step-derived.")
    L.append("- **Backbone size (B6 vs B4)**: H+ also changes the neck input dim (1280 vs 1024 -> 10.6M vs 7.8M "
             "neck params) and LoRA param count (5.2M vs 3.1M); ~2x slower per step.")
    L.append("- **YOLO baselines**: different input pipeline (1280 square vs 1280x736 letterbox), different training "
             "length/recipe; compare only at matched neg-image FP, never by confidence.")
    L.append("- **boreal_val** is a single burn (heinola) with 97% large boxes; **forest_val** has no negatives, so "
             "its recall is only meaningful at externally anchored thresholds (sdis-anchored above) and says "
             "nothing about false alarms on Australian forest backgrounds.")
    L.append("- **Score saturation / threshold transfer**: adapted backbones (LoRA, full) push Faster R-CNN softmax "
             "scores toward 1.0, so the confidence that gives 1-5% neg-FP on sdis_val is >=0.99 for B4/B6 vs ~0.97 "
             "for frozen B1 and ~0.8 for YOLO. The sdis-anchored forest_val rows therefore penalize saturated models "
             "(a single threshold transfers worse across domains) even where fixed-threshold recall is higher — "
             "compare the conf=0.50 row and the boreal_val table before concluding on forest generalization; a "
             "deployment would calibrate per domain (or use a per-domain FP target), which needs forest NEGATIVES "
             "that this val package does not have.")
    L.append("- expE_y11m_iter2 was trained on E only (iteration 2); arms A/E/C are the iteration-4 YOLO arms. "
             "v3_B1/v3_B3 are the iteration-3 DINOv3 ViT-L runs (E-only corpus, 12 epochs, bs 4 / bs 2) — the "
             "v4B1-vs-v3_B1 delta isolates the E+T corpus effect for the frozen recipe (10 vs 12 epochs confound).\n")

    L.append("## Artifacts\n")
    L.append("- preds: `models/v4/evals/<run>/{sdis_val,boreal_val,forest_val}_preds.jsonl`")
    L.append("- sweeps: `reports/v4/eval_<run>_<set>.json` (+ `.raw.parquet`)")
    L.append("- matched-FP: `reports/v4/matched_fp_dino_{sdis_val,boreal_val,forest_val}.json`")
    L.append("- run/cost summary: `models/v4/dino_run_summary.json`; param counts: `models/v4/param_counts_from_ckpt.json`, "
             "`models/v4/backbone_totals.json`")
    L.append("- checkpoints: `models/v3/runs/<run>/seg*/{best,last}.pt` (local) and "
             "`/scratch/scratch/tianrgua-sandbox/smoke/runs_v3/<run>/` (cluster)")
    L.append("- logs: `logs/v4/dino_<run>.log`, `logs/v4/eval_<run>.log`; data builders `scripts/v4/yolo_to_coco_eet.py`, "
             "`scripts/v4/build_evalval.py`; eval driver `scripts/v4/eval_dino_run.sh`")
    out = REP / "dino_adapt_results.md"
    out.write_text("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
