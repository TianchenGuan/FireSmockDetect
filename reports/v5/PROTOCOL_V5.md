# V5 Protocol — frozen 2026-10-02 00:40 UTC (prep phase, Day 1)

Owner session: cvproject-15. Any change to this file after freeze is appended
as a dated amendment; nothing above the amendment line is edited.

## 0. Two questions (and only two)
Q1. With matched no-smoke footage added, does the detector keep smoke recall
    while cutting false-alert EPISODES significantly?
Q2. With real onset sequences added, does it fire the first VALID warning
    earlier at the same false-alert budget?
Primary target (pick one per deployment scenario, before training): episode
false alerts ↓ ≥50% at non-inferior recall/delay. Early-recall +10pt @3 min is
SECONDARY and reported with its interval (tiny event counts).

## 1. Deployment scenarios — calibrated and reported SEPARATELY
- FIXED-CAMERA (pyro-sdis sites; cadence ~29 s; HPWREN/FIgLib historical).
- AERIAL/UAV (Boreal burns, DJI; AuSmoke DJI units).
No shared alert threshold is required across scenarios.

## 2. Data use — per purpose, inherited forever
| material | purpose | rule |
|---|---|---|
| pyro-sdis train sites / D-Fire / E corpus | TRAIN | unchanged (splits_frozen.json) |
| pyro-sdis VAL sites (cabanelle, marguerite) | CALIBRATION + model selection | embedded negatives → fixed-camera episodes/h |
| pyro-sdis TEST sites | FINAL TEST (used once per iteration) | v5 final only |
| Boreal evo + karkkila | TRAIN (+ the 1 karkkila non-smoke clip = train negative) | |
| Boreal heinola (incl. 14 non-smoke clips, 217 neg frames) | AERIAL CALIBRATION/VAL | **never training**; raw video of this burn, if obtained, inherits VAL |
| Boreal ruokolahti | AERIAL FINAL TEST — opened once in v4, now HISTORICAL | not reused as a fresh final set |
| AuSmoke train / val / test groups | as forest_holdout_v4.json | **permission status: PENDING** — no new training use until the licence is confirmed in writing; existing v4 use is disclosed, not extended |
| MultiNatSmoke non-AuSmoke "usable" (G) | permission PENDING | same |
| FIRESENSE smoke_neg (9 videos) | fixed-camera-like negative REPLAY only (already in v4 train corpora → flagged optimistic for v4-trained models) | |
| FIgLib-39 / lockbox v2–v4 | HISTORICAL comparison only | banned from training/tuning/selection |
| MmodalFire | AUXILIARY TRAIN candidate after human visual labelling; never a forest/UAV eval | |
| any newly acquired footage | assigned by WHOLE site/flight/date at ingest, recorded in data/v5/manifests before any model sees it | |

## 3. Metrics (definitions fixed now)
- Frame-level: unified matching IoU≥0.4, size buckets, neg-image FP — as v1–v4.
- EPISODE false alerts per hour (both scenarios): an episode starts at any
  alert state ≠ none (provisional warnings COUNT) and ends when the device
  returns to none for ≥ cooldown; rate = episodes / valid observed hours.
  VALID observed time = sum of covered intervals only: a gap > 3× the
  sequence's median cadence is excluded, independent clips are never
  concatenated, and each result states the cadence (e.g. "29 s sampled",
  "1 fps") and total valid hours. Also report fraction of valid time spent
  in an alert state.
- Early warning: events detected within 1/3/5 min of FIRST CLEAR SMOKE
  (annotated interval [last_clear_no_smoke, first_clear_smoke]); misses stay
  in the denominator; pre-visible alerts reported separately, never counted
  as wins; paired delay deltas on jointly detected events.
- Calibration objective: choose thresholds that HOLD the pre-agreed recall /
  delay constraint (below) and then minimise episodes/h — not "raise the
  threshold until FP looks good".
  * fixed-camera constraint: frame recall on sdis VAL ≥ deployed expB's
    recall at its current 0.30/0.15 operating point (reported value is the
    anchor), FIgLib-39 detected ≥ 30/39.
  * aerial constraint: boreal VAL frame recall ≥ 0.90 at the chosen conf.
- Shadow/bypass runs: every record is tagged source=historical_replay or
  source=live_upload; replay results are never described as field runs.

## 4. Experiments (training phase, only after data acceptance)
A re-calibrate only · B continue-training control (matched budget) ·
C + matched negatives (scenario-balanced sampling) · D C + reviewed onset /
faint-smoke supervision. Fixed init (YOLO11m COCO), 1280, same optimiser;
best arm + its control ×3 seeds. DINOv3-L LoRA gets the best YOLO recipe
only (no 7B, no full matrix). Temporal: single-frame vs learned single-frame
readout vs causal module, windows defined in SECONDS not frames; first-frame
output mandatory; history reset on gap/scene change. Opens only if onset
truth passes acceptance (≥10 independent events with human-reviewed
intervals); otherwise that conclusion is DEFERRED, not faked.

## 5. Compute
Prep: ≤1 g6e node (8 L40S) at any time, CPU first (cluster/v5/BUDGET_V5.json,
registry via cluster/v5/registry.sh). Training window: separate budget file,
≤128 GPUs / ≤3,072 GPU-h, one coordinator session (ListAgents check before
one-shot evals).

## 6. Acceptance for "data ready"
Negatives: per scenario ≥2 h valid, ≥3 independent units, human-verified
no-smoke (detector silence is NOT evidence), licence recorded. Onsets: ≥10
independent events with reviewed intervals, final-test onsets labelled
without seeing candidate-model outputs.

---
## Amendment A1 — 2026-10-02 02:05 UTC (licence correction)
HPWREN FIgLib material (incl. lockboxes v2–v4 and the FIgLib-39 replay set)
is published under **CC BY-NC-ND 4.0** per HPWREN/ALERTCalifornia terms
(reports/v5/ACQUISITION_LEADS_V5.md), not the "credit-reference" status
recorded in v1–v4 cards. Consequence: non-commercial, no derivative
redistribution; our use (internal research evaluation, no redistribution)
is within NC terms; any fine-tuned model or crop set DERIVED from FIgLib
frames may not be redistributed. FIgLib remains eval-only (unchanged).
Permission column for FIgLib in DATA_INVENTORY_V5 = "CC BY-NC-ND 4.0 —
eval-only, no derivative release".
