# newfootage_day1 — human review package for the NEW, UNVERIFIED no-smoke candidates (V5 Day-1/2)

Everything in this directory is **candidate FP on UNVERIFIED footage: detector output is not evidence of smoke
or of its absence**. No verdict exists until a human fills the CSVs below. No model has trained on this material.

## What is here
| set | review units | frames on sheets | sheets | selection rule |
|---|---|---|---|---|
| MONIMET MC124 (fixed camera, 30-min cadence) | 11 (`monimet_mc124_YYYYMM/`) | 881 of 6,375 | 42 | all 516 frames that alert for >=1 of expB/expE/B4/B6 at the Day-1 operating point + the 2 highest-scoring NON-alerting frames of every camera-day (365; spot check) |
| BAMBI flight_98 point1 (aerial nadir, 1 fps) | 1 (`bambi_flight98_point1/`) | 96 of 96 | 4 | whole 95.9-s segment (5 frames alert for >=1 of expB/E_ET/D_curr; B4/B6 output no box) |

* `<unit>/sheet_NNN.jpg` — 4x6 contact sheets (480-px tiles, burned-in `unit #idx t=` and the file name /
  wall-clock as written in the MONIMET name, which is UTC+2). `<unit>/frames_index.csv` maps idx -> sheet/row/col/full-res path.
* `frame_verdicts_TEMPLATE_<set>.csv` — **per-FRAME** template (one row per sheet frame): `why` (`alert_any` |
  `top_nonalert` | `all_frames`), `systems_alerting`, per-system scores, and the blank columns
  `verdict_smoke_yes_no_uncertain, confounder_if_no, reviewer, reviewed_utc`. This is the file that turns candidate FPs into
  verified FPs (or into real smoke).
* `verdicts.csv` — the per-UNIT template written by `scripts/v5/negative_review.py` (`verdict in {no_smoke,smoke,uncertain}`).
  NOTE: a unit verdict here covers only the frames on the sheets (a SUBSET of the unit); it does not verify the camera-month.
  `units.csv` `valid_observed_s` likewise refers to the selected subset — the authoritative valid hours are in
  `data/v5/newfootage/units_monimet_mc124.csv` / `cadence_summary.json`.
* `_inputs_manifest_<set>.csv`, `_lists/<unit>.txt` — inputs (re-runnable: `python scripts/v5/negative_review.py sheets --manifest ... --out data/v5/review/newfootage_day1`).
* Scores come from `models/v5/evals/<system>__<set>.jsonl` (source_tag = historical_replay); per-frame table with all systems:
  `data/v5/newfootage/frame_scores_<set>.csv`.

## How to review (priority order)
1. The 19 MONIMET frames that alert for >=3 systems (listed in reports/v5/NEWFOOTAGE_CANDIDATE_FP_V5.md §5). Agent screening of
   two of them at full resolution (2020-04-21 17:01, 2020-04-20 17:01) shows a white plume / thin column above the far-left farm
   buildings — i.e. likely REAL smoke (field or slash burning) inside the "no-smoke" candidate. Human must confirm; if smoke,
   the frame leaves the negative pool (and becomes onset-candidate material), it is not an FP.
2. The remaining `alert_any` frames (candidate FPs; expected confounders: low sun / lens flare, overcast gloom, snow, dusk).
3. The `top_nonalert` spot-check frames (do the detectors miss anything obvious?).
4. BAMBI: watch all 4 sheets; the last 6 s (frames 90-95) are over houses/gardens, not forest — the 3 YOLO alerts fall there.
Fill BOTH templates; faint plumes need the full-res JPG (`path` column). Full verification of MONIMET as a negative pool means
all 6,375 frames (266 sheets), not only this package: silence of the detectors is not evidence.
