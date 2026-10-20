# V5b pre-registered evaluation plan — frozen 2026-10-02 09:30 UTC
(before any arm B/C/D checkpoint exists; before any holdout prediction exists)

## Systems
S0 expB (deployed default) · S0' expE · S1 arm A control (expV5bA_ctrl_s3 — seed-0
replicate, labelled) · S2 arm B (+new labelled data, source-balanced) · S3 arm C
(B + conservative ignore regions) · S4 arm D (B + filtered teacher pseudo-boxes)
· S5 ACCURATE ensemble = recipe in models/v5b/fusion/accurate_recipe.json
(A+B4+B6+B1 calibrated WBF, fit site-grouped OOF on sdis VAL only) · S6 B200
high-res DINOv3-L LoRA (if it trains) · SEG branch reported on masks only.
Best-of-{B,C,D} by the rule below and its matched control get seeds 1,2 if the
window allows; otherwise single-seed results are labelled as such.

## Selection rules (validation only)
- Student arms: 3-arm comparison on pyro-sdis VAL matched-FP recall @5%/10%
  (primary) and FireMan VAL video (secondary, image-level + box); "best" =
  highest mean of sdis R@10% and FireMan-val R@10%; ties → simpler arm (B<C<D).
- Thresholds per system: sdis-VAL conf at 5/10/20% neg-image FP (fixed-camera
  regime) and, for aerial sets, FireMan-VAL conf at the same FP levels
  (aerial regime; FireMan val has 1,154 explicit negatives in one video).
- Ensemble: recipe frozen as shipped; no refit on any holdout.

## Final holdouts (evaluated ONCE at close)
H1 FireMan burn C (author CVAT labels, GPS-confirmed independent burn, 640×512
   RGB, ~38.9k frames → evaluated on every 10th frame to limit correlation;
   image-level smoke presence + box recall @ IoU≥0.4 at the pre-registered
   confs; explicit negatives give image-level FP). NEVER used for pseudo-labels,
   mining, fusion fitting, thresholds or selection.
H2 pyro-sdis TEST (frozen since v1; opened once at v4 close → disclosed as a
   second opening; fixed-camera matched-FP tables).
H3 Sheffield rec_0008 (same burn as its train recordings → reported as
   within-incident generalisation only, clearly labelled).
H4 historical, reported for continuity only: Boreal ruokolahti (v4 lockbox,
   opened), FIgLib-39 causal replay with signed delays (pre-visible alerts
   separate), AuSmoke forest test (permission pending → numbers shown but not
   used for decisions).
No teacher prediction replaces any evaluation label.

## Metrics
Matched-FP recall/tiny-recall (1/2/5/10/20% neg-image FP where ≥100 negatives
exist); image-level precision/recall; per-source and aerial-vs-fixed tables;
author-label vs noisy-label diagnostics kept separate (FireMan boxes are
interpolated region boxes → box IoU metrics labelled weak); latency (batch-1
L40S, CPU) and throughput; single vs ensemble vs distilled student.

## Deployment decision rule
Replace FAST default only if a candidate beats expB on H2 matched-FP R@10% by
≥2pt AND does not lose on H1 image-level recall at matched FP AND passes
integration tests; ACCURATE mode ships only if the ensemble beats the best
single model on H1+H2 at matched FP by ≥2pt — otherwise single model, reported.

---
Amendment (09:35 UTC, before any holdout prediction): the FireMan burn-C holdout
has 0 explicit-negative RGB frames (19,459 frames / 5 videos, all smoke-positive
per author tags) → H1 reports RECALL ONLY (image-level and box) at the
pre-registered confs; aerial false-positive rates come from FireMan VAL
(1,154 author negatives, one video) and Boreal heinola VAL (217 + 14 clips),
both labelled validation-side. No holdout-based FP claim will be made.

Clarification (2026-10-02T11:29:16Z): H1 "~38.9k frames" counted burn-C rows incl. thermal; the RGB
holdout is 19,459 frames → 1,949 at stride 10 (LOCKBOX_V5B_FROZEN.md). Rule unchanged.
