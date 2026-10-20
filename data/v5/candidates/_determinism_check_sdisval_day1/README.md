# Review package: _determinism_check_sdisval_day1

Source sequences: `data/v3/sequences/sequences_val.json` (266 sequences, 7335 frames, split(s) ['val']).
Saved detector outputs used (NO inference was run): yoloA_v4a_s2 <- reports/v4/eval_v4a_s2_sdisval.raw.parquet, yoloE_v4e_s1 <- reports/v4/eval_v4e_s1_sdisval.raw.parquet, dinoB4 <- reports/v4/eval_v4B4_lora_L_sdis_val.raw.parquet.
Operating point per model: alert if score > the 0.90-quantile of label-negative frame scores (matched 10% neg-frame FP; see thresholds.json).

## What is in here
* `candidates.csv` — 10 ONSET candidates (>= 3 label-clear frames before the first positive label; ranked by
  number of leading clear frames, then lead span) and 10 HIGH-FP candidates (label-negative frames alerted by >= 1 model;
  ranked by #models alerting, #alerted negative frames, score percentile). Lists are disjoint.
* `frames.csv` — every exported frame with time, dataset label, per-model score/alert and the disagreement flag.
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
