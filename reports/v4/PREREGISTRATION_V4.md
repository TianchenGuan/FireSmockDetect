# Iteration 4 — pre-registered final evaluation plan

Frozen 2026-09-14 ~19:00 UTC, BEFORE any test-set prediction exists and
before the 3-seed repeats finish. The sealed test package is built by
scripts/v4/build_evaltest.py (reports/v4/LOCKBOX_V4_FROZEN.md); it is
evaluated ONCE, at iteration close, with exactly the systems and thresholds
selected by the rules below. No re-runs, no post-hoc threshold changes.

## Test sets (evaluated once each)
1. pyro-sdis TEST (frozen since iteration 1; sites ferion + croix-augas) —
   fixed-camera source domain, matched-FP tables at 1/2/5/10/20% neg-image FP.
2. boreal TEST = burn_ruokolahti — genuine UAV aerial, ONE independent burn.
3. AuSmoke forest TEST = 9 held-out units / 4 independent events (3 ground
   burns + 1 DJI aerial flight; the flight's staged TEST_RUN video is reported
   separately). Positives-only -> recall-at-threshold, no matched-FP.
4. FIgLib-39 event replay — HISTORICAL (used in v1-v3), reported as a clearly
   labelled historical comparison only. Lockbox v2/v3 are NOT re-opened.

## Systems (fixed by validation rules, not by test)
- S0  expE (iteration-2 YOLO11m, deployed candidate) and expB (deployed
      default) — deployment baselines, thresholds 0.30/0.15 as deployed.
- S1  Arm A (E only, existing-data control) — 3-seed mean; report the seed
      whose sdis-val matched-FP recall@10% is the median of the three.
- S2  Best v4 data arm = whichever of {C_EGT, E_ET} has the higher 3-seed
      mean of boreal-val recall@1%FP; tie (<1pt) -> E_ET (simpler, no G).
      Same median-seed rule. The other arm is reported alongside as the
      matched control (this IS the G-value comparison).
- S3  Arm B (G-only, source-only) and D_adapt (curriculum) — seed 0 only,
      reported for completeness; not candidates.
- S4  Best DINOv3 adaptation run = highest sdis-val matched-FP recall@5%
      among {B1 frozen-L, B4 LoRA-L, B5 full-L, B6 LoRA-H+} (reports/v4/
      dino_adapt_results.md), if the run completed. Frozen-L is the control.

## Thresholds (two regimes, both reported)
- SOURCE-ONLY regime: confidence chosen on pyro-sdis VAL at 5% and 10%
  neg-image FP for each system; applied unchanged to boreal/forest test.
- TARGET-ADAPTED regime: confidence chosen on boreal VAL (1% and 5% neg-FP)
  for boreal test, and on AuSmoke forest VAL (recall-maximising conf subject
  to <=5% neg-FP on boreal val, since forest val has no negatives) for
  forest test. Using target val for thresholds is itself adaptation and is
  labelled as such.
- Alert policy for FIgLib replay: deployed 0.30/0.15 3-of-5 for all YOLO
  systems (comparable to v1-v3); no per-system tuning.

## Primary outcomes
Tiny/faint recall at matched FP (sdis); aerial recall at 1%/5% FP with the
217-negative denominator caveat; forest recall-at-threshold per event;
FIgLib R@1/3/5 min with misses and paired delay deltas; batch-1 latency on
L40S for any system proposed as backend. Denominators and unit counts stated
in every table.

## What would change the deployed default
Only a system that (a) beats expE on sdis TEST matched-FP recall@10% or ties
within 1pt AND beats it by >=10pt on boreal TEST recall@1%FP, (b) does not
lose FIgLib events vs expE, (c) passes latency/integration checks. Otherwise
the new model is registered as an optional backend and the default stays.
