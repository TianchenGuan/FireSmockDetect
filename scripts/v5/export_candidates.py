#!/usr/bin/env python
"""export_candidates.py — rank sequences for HUMAN review using already-saved detector
outputs (no inference), and write a self-contained review package.

Three signals, computed from saved scores only (PROTOCOL_V5: scores are assistive):
  * disagreement  — at matched negative-frame FP operating points (--fp-target, default
                    10% of label-negative frames per model) the models do not agree.
  * high-FP       — label-NEGATIVE frames on which >=1 model alerts (either a true false
                    alert = hard negative, or a label miss = smoke the annotator did not box).
  * onset         — sequences whose first positive label is preceded by >= N label-clear
                    frames (--min-clear-frames); also flags when models fire BEFORE the
                    first positive label (label late? or FP?).

Package layout (<out>/):
  candidates.csv   one row per selected sequence (kind=onset|high_fp, rank, metrics)
  frames.csv       one row per exported frame (label, per-model score/alert, disagreement)
  thresholds.json  per-model conf threshold at --fp-target and achieved neg-frame FPR
  frames/<seq_id>/<idx>__<name>.jpg   copied frames (window around the region of interest)
  sheets/<seq_id>.jpg                  contact sheet (idx / time / L+/L- / scores, * = alert)
  README.md        what to do, caveats
--blind drops scores/labels from sheets and frames.csv (for packages that must be
labelled without seeing candidate outputs). TEST-split sequences are refused.

Example (pyro-sdis VAL, 3 saved score sets):
  python scripts/v5/export_candidates.py --sequences data/v3/sequences/sequences_val.json \
     --scores yoloA_v4a_s2=reports/v4/eval_v4a_s2_sdisval.raw.parquet \
     --scores yoloE_v4e_s1=reports/v4/eval_v4e_s1_sdisval.raw.parquet \
     --scores dinoB4=reports/v4/eval_v4B4_lora_L_sdis_val.raw.parquet \
     --out data/v5/candidates/sdisval_day1 --n-onset 10 --n-fp 10 --min-clear-frames 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_common as rc  # noqa: E402


def load_sequences(path: str):
    seqs = json.load(open(path))
    bad = [s["seq_id"] for s in seqs if str(s.get("split", "")).lower() == "test"]
    if bad:
        raise SystemExit(f"refusing: {len(bad)} TEST-split sequences in {path} (final-test material)")
    out = []
    for s in sorted(seqs, key=lambda s: s["seq_id"]):
        fr = s["frames"]
        ts0 = fr[0]["ts"]
        frames = [dict(idx=i, path=f["image_path"], name=Path(f["image_path"]).name, ts=f["ts"], rel_s=f["ts"] - ts0,
                       label=bool(f["smoke"]), n_boxes=len(f.get("boxes_xywhn") or [])) for i, f in enumerate(fr)]
        out.append(dict(seq_id=s["seq_id"], site=s.get("site"), camera=s.get("camera"), split=s.get("split"),
                        label_pattern=s.get("label_pattern"), frames=frames))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--scores", action="append", required=True, help="name=path (raw.parquet or jsonl[.gz]); repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-onset", type=int, default=10)
    ap.add_argument("--n-fp", type=int, default=10)
    ap.add_argument("--min-clear-frames", type=int, default=3)
    ap.add_argument("--fp-target", type=float, default=0.10, help="matched neg-frame FP rate per model for the alert operating point")
    ap.add_argument("--max-lead-frames", type=int, default=24, help="onset export: keep at most this many trailing clear frames")
    ap.add_argument("--post-frames", type=int, default=12, help="onset export: label-positive frames after first label")
    ap.add_argument("--fp-context", type=int, default=2, help="high-FP export: neighbours around each alerting negative frame")
    ap.add_argument("--max-frames-per-seq", type=int, default=36)
    ap.add_argument("--thumb-w", type=int, default=320)
    ap.add_argument("--blind", action="store_true", help="omit scores and labels from sheets/frames.csv")
    a = ap.parse_args()

    seqs = load_sequences(a.sequences)
    models = []
    pmaps = {}
    for spec in a.scores:
        name, path = spec.split("=", 1)
        pmaps[name] = rc.load_preds(path)
        models.append(dict(name=name, path=path))
    names = [m["name"] for m in models]

    # ---- per-frame scores ----
    all_neg_scores = {n: [] for n in names}
    n_missing = {n: 0 for n in names}
    for s in seqs:
        for f in s["frames"]:
            f["score"] = {}
            for n in names:
                v = rc.lookup_score(pmaps[n], f["path"])
                f["score"][n] = v
                if v is None:
                    n_missing[n] += 1
                elif not f["label"]:
                    all_neg_scores[n].append(v)
    thr = {}
    for n in names:
        negs = np.asarray(all_neg_scores[n], dtype=float)
        if len(negs) == 0:
            raise SystemExit(f"{n}: no label-negative frames with scores — cannot set a matched-FP operating point")
        t = float(np.quantile(negs, 1.0 - a.fp_target, method="higher"))
        achieved = float((negs > t).mean())
        thr[n] = dict(conf_threshold=t, achieved_neg_frame_fpr=achieved, n_neg_frames=int(len(negs)),
                      n_frames_missing_score=n_missing[n], rule=f"alert if score > quantile_{1 - a.fp_target:.2f}(label-negative frame scores)")
    neg_sorted = {n: np.sort(np.asarray(all_neg_scores[n], dtype=float)) for n in names}

    def neg_pct(n, v):  # fraction of label-negative frames scoring strictly below v (0..1, higher = more unusual)
        arr = neg_sorted[n]
        return float(np.searchsorted(arr, v, side="left") / len(arr))

    # ---- per-sequence metrics ----
    rows = []
    for s in seqs:
        fr = s["frames"]
        for f in fr:
            f["alert"] = {n: (f["score"][n] is not None and f["score"][n] > thr[n]["conf_threshold"]) for n in names}
            avail = [n for n in names if f["score"][n] is not None]
            na = sum(f["alert"][n] for n in avail)
            f["n_alert"] = na
            f["disagree"] = 0 < na < len(avail)
            f["neg_pct_mean"] = float(np.mean([neg_pct(n, f["score"][n]) for n in avail])) if avail else 0.0
        obs = rc.valid_observed_seconds([f["rel_s"] for f in fr])
        pos_idx = [f["idx"] for f in fr if f["label"]]
        first_pos = pos_idx[0] if pos_idx else None
        lead = first_pos if first_pos is not None else None  # frames before first positive are all negative by construction
        r = dict(seq_id=s["seq_id"], site=s["site"], camera=s["camera"], split=s["split"], label_pattern=s["label_pattern"],
                 n_frames=len(fr), n_pos=len(pos_idx), n_neg=len(fr) - len(pos_idx), valid_observed_s=obs["valid_s"],
                 cadence_s=obs["cadence_s"], n_gaps_excluded=obs["n_gaps_excluded"],
                 first_pos_idx=first_pos, first_pos_rel_s=(fr[first_pos]["rel_s"] if first_pos is not None else None),
                 first_pos_abs_utc=(rc.fmt_abs(fr[first_pos]["ts"]) if first_pos is not None else ""),
                 n_leading_clear=lead, lead_span_s=(fr[first_pos]["rel_s"] - fr[0]["rel_s"] if first_pos else None),
                 n_disagree_frames=sum(f["disagree"] for f in fr), n_disagree_neg_frames=sum(f["disagree"] for f in fr if not f["label"]),
                 n_models_alerting_neg=0, n_alert_neg_frames_any=sum(1 for f in fr if not f["label"] and f["n_alert"] > 0),
                 max_neg_pct=max([f["neg_pct_mean"] for f in fr if not f["label"]] + [0.0]),
                 models_fire_before_first_label=False)
        for n in names:
            negf = [f for f in fr if not f["label"]]
            alerts_neg = [f for f in negf if f["alert"][n]]
            r[f"{n}__thr"] = round(thr[n]["conf_threshold"], 4)
            r[f"{n}__n_alert_neg"] = len(alerts_neg)
            r[f"{n}__max_neg_score"] = round(max([f["score"][n] or 0.0 for f in negf] + [0.0]), 4)
            r[f"{n}__first_alert_idx"] = next((f["idx"] for f in fr if f["alert"][n]), None)
            pre = [f for f in fr[:first_pos] if f["alert"][n]] if first_pos else []
            r[f"{n}__pre_label_alerts"] = len(pre)
            r[f"{n}__pre_label_first_alert_idx"] = pre[0]["idx"] if pre else None
            if alerts_neg:
                r["n_models_alerting_neg"] += 1
            if pre:
                r["models_fire_before_first_label"] = True
        rows.append(r)
    by_id = {s["seq_id"]: s for s in seqs}

    # ---- selection ----
    onset_pool = [r for r in rows if r["first_pos_idx"] is not None and (r["n_leading_clear"] or 0) >= a.min_clear_frames]
    onset_pool.sort(key=lambda r: (-r["n_leading_clear"], -(r["lead_span_s"] or 0), r["seq_id"]))
    onset_sel = onset_pool[:a.n_onset]
    taken = {r["seq_id"] for r in onset_sel}
    fp_pool = [r for r in rows if r["n_alert_neg_frames_any"] > 0 and r["seq_id"] not in taken]
    fp_pool.sort(key=lambda r: (-r["n_models_alerting_neg"], -r["n_alert_neg_frames_any"], -r["max_neg_pct"], r["seq_id"]))
    fp_sel = fp_pool[:a.n_fp]

    # ---- export ----
    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    cand_rows, frame_rows = [], []

    def export(r, kind, rank):
        s = by_id[r["seq_id"]]
        fr = s["frames"]
        if kind == "onset":
            fp = r["first_pos_idx"]
            lead_ids = list(range(max(0, fp - a.max_lead_frames), fp))
            post_ids = [f["idx"] for f in fr[fp:] if f["label"]][:a.post_frames]
            ids = lead_ids + post_ids
            reason = (f"{r['n_leading_clear']} label-clear frames ({r['lead_span_s']:.0f}s) before first positive label at #{fp}; "
                      + ("models fire BEFORE first label (label late? or FP) — check" if r["models_fire_before_first_label"] else "no model alert before first label"))
        else:
            hot = sorted([f for f in fr if not f["label"] and f["n_alert"] > 0], key=lambda f: (-f["neg_pct_mean"], f["idx"]))
            ids = set()
            for f in hot:
                for j in range(f["idx"] - a.fp_context, f["idx"] + a.fp_context + 1):
                    if 0 <= j < len(fr):
                        ids.add(j)
                if len(ids) >= a.max_frames_per_seq:
                    break
            ids = sorted(ids)
            reason = (f"{r['n_alert_neg_frames_any']} label-negative frames alerted by {r['n_models_alerting_neg']}/{len(names)} models at matched FP "
                      f"{a.fp_target:.0%}; {r['n_disagree_neg_frames']} disagreement frames. Decide: true false alert (hard negative) or missed smoke (label error)")
        ids = ids[:a.max_frames_per_seq]
        sd = out / "frames" / r["seq_id"]
        sd.mkdir(parents=True, exist_ok=True)
        tiles = []
        for i in ids:
            f = fr[i]
            dst = sd / f"{i:04d}__{f['name']}"
            if not dst.exists():
                shutil.copyfile(f["path"], dst)
            im = rc.fit_width(rc.FrameSource.from_paths([f["path"]]).load(0, max_side=a.thumb_w), a.thumb_w)
            l1 = f"#{i:04d} {rc.fmt_rel(f['rel_s'])}  {rc.fmt_abs(f['ts'])}"
            lines = [l1]
            if not a.blind:
                l2 = ("L+" if f["label"] else "L-") + ("  DISAGREE" if f["disagree"] else "")
                sc = []
                for n in names:
                    v = f["score"][n]
                    sc.append(f"{n[:7]}={'na' if v is None else format(v, '.2f')}{'*' if f['alert'][n] else ''}")
                # wrap scores so nothing is cut off at the tile width (~44 chars at size 12 / 320 px)
                line, lines2 = "", []
                for tok in sc:
                    if line and len(line) + 1 + len(tok) > 44:
                        lines2.append(line)
                        line = tok
                    else:
                        line = (line + " " + tok).strip()
                if line:
                    lines2.append(line)
                lines += [l2] + lines2
            tiles.append(rc.burn_text(im, lines, size=12))
            fo = dict(seq_id=r["seq_id"], kind=kind, idx=i, rel_s=round(f["rel_s"], 1), abs_utc=rc.fmt_abs(f["ts"]), name=f["name"],
                      exported_path=str(dst), src_path=f["path"])
            if not a.blind:
                fo["label_smoke"] = int(f["label"])
                fo["n_gt_boxes"] = f["n_boxes"]
                for n in names:
                    fo[f"score_{n}"] = "" if f["score"][n] is None else round(f["score"][n], 4)
                    fo[f"alert_{n}"] = int(f["alert"][n])
                fo["disagreement"] = int(f["disagree"])
            frame_rows.append(fo)
        title = f"{kind.upper()} rank {rank}  {r['seq_id']}  {len(ids)}/{r['n_frames']} frames  " + ("BLIND" if a.blind else f"scores assistive; * = alert @{a.fp_target:.0%} negFP; L+/L- = dataset label")
        sheet = rc.contact_sheet(tiles, cols=6, thumb_w=a.thumb_w, title=title)
        sheet.save(out / "sheets" / f"{kind}_{rank:02d}_{r['seq_id']}.jpg", quality=86, optimize=True)
        c = dict(kind=kind, rank=rank, reason=reason, frames_exported=len(ids), sheet=f"sheets/{kind}_{rank:02d}_{r['seq_id']}.jpg")
        c.update(r)
        cand_rows.append(c)

    for k, r in enumerate(onset_sel, 1):
        export(r, "onset", k)
    for k, r in enumerate(fp_sel, 1):
        export(r, "high_fp", k)

    base_cols = ["kind", "rank", "seq_id", "site", "camera", "split", "label_pattern", "n_frames", "n_pos", "n_neg", "valid_observed_s",
                 "cadence_s", "n_gaps_excluded", "first_pos_idx", "first_pos_rel_s", "first_pos_abs_utc", "n_leading_clear", "lead_span_s",
                 "models_fire_before_first_label", "n_models_alerting_neg", "n_alert_neg_frames_any", "n_disagree_frames",
                 "n_disagree_neg_frames", "max_neg_pct", "frames_exported", "sheet", "reason"]
    model_cols = [f"{n}__{k}" for n in names for k in ("thr", "n_alert_neg", "max_neg_score", "first_alert_idx", "pre_label_alerts", "pre_label_first_alert_idx")]
    rc.write_csv(out / "candidates.csv", cand_rows, base_cols + model_cols)
    fcols = ["seq_id", "kind", "idx", "rel_s", "abs_utc", "name", "exported_path", "src_path"]
    if not a.blind:
        fcols += ["label_smoke", "n_gt_boxes"] + [f"score_{n}" for n in names] + [f"alert_{n}" for n in names] + ["disagreement"]
    rc.write_csv(out / "frames.csv", frame_rows, fcols)
    # full ranking tables for transparency
    rc.write_csv(out / "all_sequences_metrics.csv", rows, [c for c in base_cols if c not in ("kind", "rank", "frames_exported", "sheet", "reason")] + model_cols)
    rc.write_json(out / "thresholds.json", dict(fp_target=a.fp_target, models=thr, sequences_file=a.sequences,
                                                n_sequences=len(seqs), n_frames=sum(r["n_frames"] for r in rows),
                                                n_label_neg_frames=sum(r["n_neg"] for r in rows),
                                                onset_pool=len(onset_pool), fp_pool=len(fp_pool) + len([r for r in onset_sel if r["n_alert_neg_frames_any"] > 0]),
                                                generated_utc=rc.utc_now()))
    readme = f"""# Review package: {out.name}

Source sequences: `{a.sequences}` ({len(seqs)} sequences, {sum(r['n_frames'] for r in rows)} frames, split(s) {sorted(set(s['split'] for s in seqs))}).
Saved detector outputs used (NO inference was run): {', '.join(f"{m['name']} <- {m['path']}" for m in models)}.
Operating point per model: alert if score > the {1 - a.fp_target:.2f}-quantile of label-negative frame scores (matched {a.fp_target:.0%} neg-frame FP; see thresholds.json).

## What is in here
* `candidates.csv` — {len(onset_sel)} ONSET candidates (>= {a.min_clear_frames} label-clear frames before the first positive label; ranked by
  number of leading clear frames, then lead span) and {len(fp_sel)} HIGH-FP candidates (label-negative frames alerted by >= 1 model;
  ranked by #models alerting, #alerted negative frames, score percentile). Lists are disjoint.
* `frames.csv` — every exported frame with time, dataset label, per-model score/alert and the disagreement flag{' (BLIND: omitted)' if a.blind else ''}.
* `frames/<seq_id>/` — copied frames; `sheets/` — one contact sheet per candidate.
* `all_sequences_metrics.csv` — the same metrics for ALL sequences (so the ranking is auditable).

## How to review
* ONSET candidates: open the sheet, find the FIRST frame where smoke is clearly visible and the LAST frame that is clearly smoke-free,
  then record them with `scripts/v5/onset_annotate.py annotate` (keep already-smoke-at-start / after-camera-turn / true onset apart).
  If models fire before the first positive label, check whether the label is late (faint smoke) — do NOT assume the model is right.
* HIGH-FP candidates: decide per alerted frame whether it is a true false alert (hard negative, keep as negative) or missed smoke
  (label error -> fix the label / treat as positive). Record in the verdict CSV of the review workspace.
* Model scores are assistive only. Detector silence is not evidence of no smoke.

## Limits
* pyro-sdis VAL frames are sampled at ~29 s cadence; durations use PROTOCOL_V5 §3 valid observed time (gaps > 3x median excluded).
* Thresholds are matched on label-negative frames of THIS set; they are review aids, not deployment thresholds.
* Nothing from any TEST split is read by this tool.
"""
    (out / "README.md").write_text(readme)
    print(f"[export] {len(seqs)} sequences; label-neg frames {sum(r['n_neg'] for r in rows)}; onset pool {len(onset_pool)} -> {len(onset_sel)} exported; "
          f"high-FP pool {len(fp_pool)} -> {len(fp_sel)} exported; frames copied {len(frame_rows)} -> {out}")
    for n in names:
        print(f"  {n}: thr={thr[n]['conf_threshold']:.4f} achieved negFPR={thr[n]['achieved_neg_frame_fpr']:.3f} missing={n_missing[n]}")
    for c in cand_rows:
        print(f"  {c['kind']:<8} r{c['rank']:02d} {c['seq_id']:<48} pat={c['label_pattern']:<18} lead={c['n_leading_clear']} "
              f"alertneg={c['n_alert_neg_frames_any']} models={c['n_models_alerting_neg']} dis={c['n_disagree_frames']} pre={c['models_fire_before_first_label']}")


if __name__ == "__main__":
    main()
