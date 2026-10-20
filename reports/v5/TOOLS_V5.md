# V5 Day-1 review tooling — negative review, onset annotation, candidate export

Written 2026-10-02 (V5 prep, Day 1). All three tools are CPU-only, deterministic (sorted inputs,
fixed tie-breaks, no RNG; a re-run of the candidate export produced byte-identical CSVs and sheets),
run from the repo venv, and never read TEST-split material (`split == test` is refused).
Shared helpers live in `scripts/v5/review_common.py` (frame sources: directory / video / path list;
timestamp parsing from ISO names, `_tNNNN` 1-fps indices, `frameN` stills; PROTOCOL_V5 §3 valid
observed time with the 3x-median-cadence gap rule; saved-score loaders for `*.raw.parquet` and
`*.jsonl[.gz]`; contact-sheet rendering with burned-in labels).

Protocol anchors: PROTOCOL_V5 §3 (valid observed time; onset interval [last_clear_no_smoke,
first_clear_smoke]; scores assistive), §6 (negatives: >=2 h / >=3 independent units / human-verified;
onsets: >=10 independent human-reviewed events, final-test onsets labelled blind).

---------------------------------------------------------------------------------------------------
## 1. `scripts/v5/negative_review.py` — human no-smoke verification

### Commands run
```bash
source .venv/bin/activate
# 14 heinola "Has Smoke=false" clips (1-fps frame dirs from v4) + 2 FIRESENSE smoke_neg VIDEOS (.avi, sampled 1 s)
python scripts/v5/negative_review.py sheets --manifest data/v5/review/_inputs/negatives_day1.csv --out data/v5/review
# extra: the 217 Boreal Subset-A heinola "Empty-Images" stills (the v4/v5 aerial negFP denominator), one unit per flight
python scripts/v5/negative_review.py sheets --manifest data/v5/review/_inputs/negatives_day1_subsetA_empty.csv --out data/v5/review
# hours verified no-smoke by scenario (only verdict == no_smoke counts)
python scripts/v5/negative_review.py summarise --review-dir data/v5/review --md reports/v5/negative_review_summary_blank.md --json reports/v5/negative_review_summary_blank.json
python scripts/v5/negative_review.py summarise --review-dir data/v5/review --verdicts data/v5/review/verdicts.EXAMPLE_ONLY.csv --md reports/v5/negative_review_summary_EXAMPLE_ONLY.md
```
Manifest columns: `unit, path, scenario, source, independent_unit, license, cadence_s, sample_s, contamination, notes`
(`path` = frame directory | video file | .txt list of frame paths). Runtime: 16 units (448 4K JPEGs + 160 s of AVI) in 21 s on CPU.

### Outputs (data/v5/review/)
* `<unit>/sheet_NNN.jpg` — 4x6 contact sheets, 480-px thumbnails, each tile burned with `unit #idx t=+m:ss` and
  file name (or absolute UTC when the name carries an ISO timestamp); sheet title tells the reviewer to check
  full-res frames for faint plumes. `<unit>/frames_index.csv` maps idx -> sheet/row/col/time/full-res path (or video+frame no).
* `units.csv` — per-unit metadata incl. §3 valid observed seconds, cadence, gap exclusions, licence, contamination flag.
* `verdicts.csv` — the TEMPLATE the human edits (columns exactly: `unit, reviewer, verdict{no_smoke,smoke,uncertain},
  first_smoke_frame_or_blank, notes, reviewed_utc`). Re-running `sheets` appends rows for new units and never overwrites filled rows.
* `screening.agent_provisional.csv` — LLM contact-sheet screening to prioritise the human (NOT a verdict; `summarise` ignores it).
* `README.md`, `_inputs/` (manifests), `_runs.jsonl` (run log).

### Sample output — `summarise` on the real (still blank) verdict file
```
| scenario | units | no_smoke | smoke | uncertain | unreviewed | candidate h | VERIFIED no-smoke h | of which contaminated h | independent units verified/all | cadence_s | acceptance |
| aerial            | 17 | 0 | 0 | 0 | 17 | 0.195 | **0.000** | 0.000 | 0/1 | 0.96,1.0 | NOT MET |
| fixed_camera_like |  2 | 0 | 0 | 0 |  2 | 0.044 | **0.000** | 0.000 | 0/2 | 1.0      | NOT MET |
```
Candidate footage on disk is tiny: aerial 0.195 h (14 heinola clips = 7.3 min; 217 Subset-A stills = 4.4 min of
non-contiguous ~0.96-s frames, gap rule applied) and 0.044 h fixed-camera-like (2 FIRESENSE videos, both flagged
"already in v4 train corpora"). Even if every unit were verified no-smoke, neither scenario would reach the §6
2 h / 3 independent units bar (all aerial material is ONE burn, heinola 2022-06-28).
With the synthetic `verdicts.EXAMPLE_ONLY.csv` (13 no_smoke, 1 uncertain, 1 smoke) the tool reports aerial 0.113 h
verified, fixed_camera_like 0.032 h (all of it contaminated) — demonstrating the arithmetic only.

### Finding from the screening (agent, 480-px thumbnails — needs human confirmation)
The Boreal Subset-B video label `Has Smoke=false` is NOT a reliable no-smoke label:
* `heinola_13` — grey-brown plume at the treeline in frames #0000-#0009, camera pans away by #0010.
* `heinola_53` — large white plume in every screened frame.
* `heinola_62` — active flames on the ground plus dense smoke in every frame.
* `heinola_103` — faint grey haze above the horizon (#0000-#0007), uncertain.
Frame-dir <-> mp4 mapping was verified (frame 0 of each mp4 vs the extracted `_t0000.jpg`: MAD 3-13 grey levels
vs 47-79 against another clip), and the raw CSV lines really say `false`. So 3 of the 14 clips the v4 replay used as
"negative footage" contain smoke; the "no plume seen" clips are 21, 32, 37, 43, 49, 74, 76, 104, 116, 124 (first
sheet only for most). Among the 217 Subset-A `Empty-Images` stills (described in the dataset card as reviewed
smoke-free): `heinola_DJI_0028_frame14` shows a distinct plume and frame39-41 a faint one; DJI_0024 (59) and
DJI_0048 (47) look clean. Consequence: the boreal-val negFP denominator used in v4 (and planned for v5 aerial
calibration) contains >=1 labelled-negative smoke frame; the v4 "1% negFP ~ 2 of 217 negatives" points sit at
that granularity. FIRESENSE `testneg02.808` is a car park with people (small white puffs near them at #0019-#0021:
breath/cigarette? designed hard negative) -> uncertain; `testneg04.810` shows no plume (sheets 1 and 3 screened).

---------------------------------------------------------------------------------------------------
## 2. `scripts/v5/onset_annotate.py` — interval annotation of first clear smoke

Design: `render` works WITHOUT any detector output (default) — required for final-test labelling; `--preds name=path`
adds an assistive score curve and per-tile scores, and whether scores/labels were shown is written into
`render_meta.json` and copied into every `annotation.json` (`scores_shown_during_annotation`, `dataset_labels_shown`).
`--crop x0,y0,x1,y1` zooms every tile into a region (pyro-sdis plumes are ~1-3 % of frame width; full-frame 320-px
tiles cannot place an onset). `annotate` validates and writes one JSON per event; the three cases the user insists on
are explicit via `onset_class`:
`true_first_visible | visible_after_camera_turn | already_smoke_at_start | no_smoke_observed | onset_not_observed_other`.
Only `true_first_visible` + `review_status=human` counts toward §6 acceptance.

### Commands run
```bash
# blind renders (no scores, no dataset labels)
python scripts/v5/onset_annotate.py render --event-id sdisval_marguerite-282_20240313T161635 --seq-json data/v3/sequences/sequences_val.json --seq-id val_sdis-07_marguerite-282_20240313T161635
python scripts/v5/onset_annotate.py render --event-id sdisval_marguerite-339_20240206T155810 --seq-json data/v3/sequences/sequences_val.json --seq-id val_sdis-07_marguerite-339_20240206T155810
python scripts/v5/onset_annotate.py render --event-id boreal_train_evo_1 --frames-dir data/v4/sequences/boreal_frames/evo_1
# zoomed re-renders (crop chosen from where the dataset boxes sit; recorded in render_meta.json)
python scripts/v5/onset_annotate.py render --event-id sdisval_marguerite-282_20240313T161635__crop --seq-json data/v3/sequences/sequences_val.json --seq-id val_sdis-07_marguerite-282_20240313T161635 --crop 0.78,0.36,1.0,0.56
python scripts/v5/onset_annotate.py render --event-id sdisval_marguerite-339_20240206T155810__crop --seq-json data/v3/sequences/sequences_val.json --seq-id val_sdis-07_marguerite-339_20240206T155810 --crop 0.17,0.44,0.39,0.66
# assistive variant (scores + labels) — demonstrates the optional curve; NOT used for the blind annotations
python scripts/v5/onset_annotate.py render --event-id sdisval_marguerite-282_20240313T161635__WITHSCORES --seq-json data/v3/sequences/sequences_val.json --seq-id val_sdis-07_marguerite-282_20240313T161635 \
    --preds yoloA_v4a_s2=reports/v4/eval_v4a_s2_sdisval.raw.parquet --preds dinoB4=reports/v4/eval_v4B4_lora_L_sdis_val.raw.parquet --show-labels
# annotations (agent-provisional; a human re-runs the same command with --review-status human --force)
python scripts/v5/onset_annotate.py annotate --event-id boreal_train_evo_1 --starts-with-smoke --annotator "..." --review-status agent_provisional --notes "..."
python scripts/v5/onset_annotate.py annotate --event-id sdisval_marguerite-282_20240313T161635__crop --last-clear-no-smoke-idx 11 --first-clear-smoke-idx 19 --annotator "..." --review-status agent_provisional --notes "..."
python scripts/v5/onset_annotate.py annotate --event-id sdisval_marguerite-339_20240206T155810__crop --starts-with-smoke --first-clear-smoke-idx 9 --annotator "..." --review-status agent_provisional --notes "..."
python scripts/v5/onset_annotate.py summary --out data/v5/onset --md reports/v5/onset_annotations_day1.md
```
Other modes: `--no-smoke-observed`; `--onset-not-observed --notes "<why>"` (gap / clip ends / occlusion);
`--camera-turn-before-smoke` with both indices -> `visible_after_camera_turn` (`onset_observed=false`,
`usable_for_early_warning_eval=false`). Inputs: `--frames-dir`, `--seq-json + --seq-id`, `--list file.txt`, `--video`.

### Outputs (data/v5/onset/<event_id>/)
`strip.jpg` (12 evenly spaced keyframes), `sheet_NNN.jpg` (all frames, 6 columns), `frames_index.csv`
(idx, rel_s, abs_utc, path[, dataset label][, score_<model>]), `render_meta.json`, `scores.png` (only with `--preds`),
`annotation.json`. Annotation record fields: `event_id, sequence_ref, site, camera, split, n_frames, cadence_s,
valid_observed_s, onset_class, onset_observed, starts_with_smoke, camera_turn_before_smoke, last_clear_no_smoke_idx/_ts,
first_clear_smoke_idx/_ts, uncertain_span{start_idx,end_idx,n_frames,duration_s}, bracket_s,
usable_for_early_warning_eval, scores_shown_during_annotation, score_models, dataset_labels_shown, annotator,
review_status{human,agent_provisional}, notes, annotated_utc`.

### Sample output — `summary`
```
| event_id | class | observed | last_clear idx/t | first_clear idx/t | bracket s | cam turn | scores shown | status |
| boreal_train_evo_1                           | already_smoke_at_start | False |            | 0/t=+0:00  |       | False | False | agent_provisional |
| sdisval_marguerite-282_20240313T161635__crop | true_first_visible     | True  | 11/t=+5:20 | 19/t=+7:29 | 129.0 | False | False | agent_provisional |
| sdisval_marguerite-339_20240206T155810__crop | already_smoke_at_start | False |            | 9/t=+4:30  |       | False | False | agent_provisional |
classes: {'already_smoke_at_start': 2, 'true_first_visible': 1}
human-reviewed true_first_visible events: 0 — PROTOCOL_V5 §6 acceptance (>=10 independent, human-reviewed): NOT MET
```
Notes on the three provisional annotations (all need a human): evo_1 has a dense plume in frame #0000 (Boreal
video label smoke=true) -> onset not observed in the 30-s clip, as expected for Subset-B clips. marguerite-282:
frames 0-11 clearly clear, #0012-#0018 possible faint wisp (uncertain span, 129 s), first CLEAR puff at #0019 — the
dataset's first positive label is also #0019, while the saved YOLO-A / DINO-B4 scores (seen only in the separate
`__WITHSCORES` render) rise from #0012-#0015, i.e. inside the uncertain span. marguerite-339: a faint grey smudge is
on the horizon already at #0000 and is a clear plume by #0009, so it is recorded as already-smoke-at-start although
the dataset labels frames 0-7 negative; a human may flip it to `true_first_visible` with last_clear=3. (In the candidate package the saved yoloA_v4a_s2 scores alert on the label-negative frames #0002-#0005 of this same sequence — consistent with smoke being present before the first positive label.)
Dataset label flicker is therefore not the only label issue: label-late / label-early cases exist in pyro-sdis VAL.

---------------------------------------------------------------------------------------------------
## 3. `scripts/v5/export_candidates.py` — rank sequences for review from SAVED scores

### Command run (no inference, ~6 s CPU)
```bash
python scripts/v5/export_candidates.py --sequences data/v3/sequences/sequences_val.json \
  --scores yoloA_v4a_s2=reports/v4/eval_v4a_s2_sdisval.raw.parquet \
  --scores yoloE_v4e_s1=reports/v4/eval_v4e_s1_sdisval.raw.parquet \
  --scores dinoB4=reports/v4/eval_v4B4_lora_L_sdis_val.raw.parquet \
  --out data/v5/candidates/sdisval_day1 --n-onset 10 --n-fp 10 --min-clear-frames 3 --fp-target 0.10
```
Operating point per model = alert if score > 0.90-quantile of label-negative frame scores (matched 10 % neg-frame FP):
yoloA_v4a_s2 0.6740, yoloE_v4e_s1 0.6615, dinoB4 0.9886 (achieved negFPR 0.099 each; 0 frames missing scores).
Flags: per-frame DISAGREEMENT (some but not all models alert), HIGH-FP (label-negative frame with >=1 alert),
ONSET candidate (first positive label preceded by >= N label-clear frames; plus `models_fire_before_first_label`).
`--blind` drops scores/labels from sheets and `frames.csv` (verified: header reduces to
`seq_id,kind,idx,rel_s,abs_utc,name,exported_path,src_path`). TEST-split sequences are refused.

### Sample output
```
[export] 266 sequences; label-neg frames 1158; onset pool 13 -> 10 exported; high-FP pool 45 -> 10 exported; frames copied 379 -> data/v5/candidates/sdisval_day1
  onset    r01 val_sdis-07_marguerite-282_20240313T161635  pat=appears             lead=19 alertneg=0  models=0 dis=3  pre=False
  onset    r02 val_force-06_cabanelle-327_20240223T084749  pat=flicker             lead=16 alertneg=11 models=3 dis=26 pre=True
  onset    r03 val_force-06_cabanelle-46_20240223T101010   pat=appears_disappears  lead=15 alertneg=9  models=3 dis=7  pre=True
  ...
  high_fp  r01 val_force-06_cabanelle-327_20240403T080031  pat=flicker             lead=2  alertneg=25 models=3 dis=22 pre=True
  high_fp  r02 val_force-06_cabanelle-125_20240410T081837  pat=flicker             lead=0  alertneg=15 models=3 dis=26 pre=False
```
Package (`data/v5/candidates/sdisval_day1/`, 50 MB): `candidates.csv` (20 rows, per-model thresholds / alert counts /
first alert idx / pre-label alerts), `frames.csv` (379 exported frames with label, per-model score + alert, disagreement),
`all_sequences_metrics.csv` (all 266 sequences, so the ranking is auditable), `thresholds.json`, `frames/<seq_id>/`,
`sheets/` (20 contact sheets; `*` marks an alert, `L+/L-` the dataset label), `README.md` (review instructions).
Observations: 7 of the 10 onset candidates have at least one model alerting BEFORE the first positive label
(label-late or FP — to be decided by a human); all 10 high-FP candidates are `flicker` sequences from cabanelle /
marguerite where all three models alert on label-negative frames (cloud-over-ridge scenes in the sheets).
Determinism check: a second run into `data/v5/candidates/_determinism_check_sdisval_day1/` gave byte-identical
`candidates.csv`, `all_sequences_metrics.csv`, `frames.csv` (modulo out-dir) and all 20 sheets.
`_blind_smoketest/` holds the 2+2 `--blind` run.

---------------------------------------------------------------------------------------------------
## 4. Limitations and blockers
* No human verdict or human onset annotation exists yet; everything marked `agent_provisional` / `screening` is a
  prioritisation aid, not verification. The protocol's §6 acceptance is NOT MET for either scenario and cannot be met
  from material on disk (0.24 h candidate footage, 1 aerial burn).
* Boreal Subset-B `Has Smoke=false` labels are wrong for at least heinola_13/53/62 (and `Empty-Images`
  heinola_DJI_0028_frame14 contains smoke) -> the aerial calibration baseline must drop/flag these before counting
  episodes/h, and v4 replay-negative numbers on boreal clips are contaminated by real smoke. Human confirmation needed.
* Saved sdis-VAL scores for the deployed expB / expE were not found among reports/v4 artifacts (only v4* arms and
  DINO B1/B4/B5/B6); the export proof therefore used v4a_s2 / v4e_s1 / B4. The tool accepts any `*.raw.parquet` or
  `*.jsonl[.gz]` with `{image, preds}` once the calibration agent writes expB preds.
* Thumbnail resolution (480 px review, 320 px crops) is a triage aid; faint plumes need the full-res frame paths in
  `frames_index.csv`. Video inputs are sampled at 1 s (`--sample-s`), so sub-second events are not shown.
* Agent screening covered the FIRST sheet of most heinola clips (frames 0-23 of ~30); `heinola_13` both sheets;
  FIRESENSE testneg04 sheets 1 and 3 only; all Subset-A sheets.
* No cluster was used; nothing under reports/v4, STATUS.md, TEST splits, ruokolahti, AuSmoke or lockboxes was touched.
