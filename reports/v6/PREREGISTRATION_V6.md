# V6 SMOKE-EDGE-SYNTH — Pre-registration (FROZEN)

Frozen 2026-10-20T20:10Z, before any v6 training or generation result was seen.
Window: 2026-10-20T19:50:37Z → 2026-10-21T19:50:37Z.
Changes after this point go into AMENDMENTS at the bottom with timestamps and
reasons; nothing above may be edited.

## Question

Can automatically constructed synthetic smoke data (RGBA-asset compositing
and/or masked inpainting), mixed ~50/50 with the existing real recipe, improve
a compact YOLO11n smoke detector on REAL author-labeled evaluation data at
matched negative-image false-positive rates — and does detector distillation
add anything on top? The answer may be NO; a negative result is a valid
outcome and will be reported as such.

## Data roles (inherited, unchanged)

- REAL TRAIN: the existing recipe list (`expB_train.txt`: pyro-sdis train +
  D-Fire train, baked in image). Backgrounds for composition come ONLY from
  TRAIN-role images with explicit no-smoke evidence (per
  `data/v6/background_pool_v1.json`). MONIMET, FIgLib/HPWREN-derived, all
  val/test-role images and Boreal validation flights are EXCLUDED from
  generation conditioning and composition.
- VAL (decisions, thresholds, generator/mixture selection): `sdis_val`
  (fixed-camera) primary; boreal val (aerial) secondary.
- FINAL TEST (frozen, one shot at the end, no tuning): the v5b frozen sets —
  pyro-sdis TEST (H2), FireMan holdout s10, Sheffield holdout, boreal test,
  forest test. These are HISTORICAL benchmarks (used once in v5b); this is
  disclosed in all reporting. If the inventory finds an eligible never-used
  author-labeled holdout, it is added as H-NEW before any v6 model is
  evaluated on it; if none exists, that is disclosed as a limitation.
- Synthetic images are TRAINING data only. Synthetic metrics are never
  headline results. Test labels are never replaced by model predictions.

## Student, teacher, matrix

Student: YOLO11n (init `yolo11n.pt` COCO-pretrained, ~2.6 M params).
Teacher for arm E: chosen from existing checkpoints by sdis VAL matched-FP
evidence (candidates: v5b arm-A YOLO11m seeds, expE y11m, expB y11s),
recorded in `data/v6/INVENTORY_V6.md` BEFORE arm E launches.

All arms: imgsz 1280, batch 48 (8×L40S, one node), optimizer/augmentation =
ultralytics defaults of the existing recipe (same train_yolo.py), seed 0 for
the matrix. Total optimizer steps matched to A's 60 epochs × N_R images:

| Arm | Corpus (per epoch) | Epochs | Real:synth exposure |
|---|---|---|---|
| A  | R (= expB_train, N_R imgs) | 60 | 100:0 |
| A-long | R | 120 | 100:0 (extra-compute control) |
| B  | R + P (procedural copy-paste, N_P=N_R) | 30 | 50:50 |
| C  | R + S (generative, N_S=N_R) | 30 | 50:50 |
| D  | R + S_q (N=N_R/3) | 45 | 75:25 |
| E  | C's corpus + detector KD losses | 30 | 50:50 |

Synthetic lists reach N_R by multiplicity over the accepted pool if the pool
is smaller; unique counts are reported separately from exposures. Smoke /
no-smoke balance inside each synthetic block mirrors R's pos:neg ratio.
Known deviation (accepted): warmup_epochs=3 covers different step counts
across arms; LR schedule is epoch-fractional so its shape over total steps
matches. Multiple seeds (≥3) are run for the best arm and its matched
control; claims cite the seed spread.

## Synthetic data rules

- Route 1 (RGBA assets composited on eligible real train backgrounds):
  alpha + transform metadata kept; boxes/soft masks derived from visible
  transformed support; visibility checked AFTER 1280 resize; invisible
  support → ignore, not positive; one plume-level box per connected plume,
  no boxes from disconnected translucent pixels.
- Route 2 (masked inpainting): the request mask is NOT ground truth; labels
  derive from the actual output (difference support ∩ requested region +
  model-assisted localization), marked model-assisted/weak.
- Controls in every generator/background family: synthetic faint smoke,
  synthetic visible smoke, synthetic NO-smoke edits and confounders (fog,
  cloud, dust, steam), plus matched no-smoke controls passed through the
  SAME compositing/compression pipeline. "Synthetic" ≠ "positive".
- No generation conditioned on val/test images or their derivatives.
- All review is AUTOMATED/MODEL-ASSISTED and recorded as such; acceptance
  filters must not require the old YOLO to already detect the smoke (that
  would delete the faint cases); difficulty strata + rejection reasons logged.
- Pilot corpus target 5,000–10,000 accepted composites (unique backgrounds,
  unique assets, composites counted separately), versioned manifest
  `data/v6/manifests/CORPUS_V6_*.json` frozen before each training launch.

## Metrics and decision rules (frozen)

- Primary: pyro-sdis VAL recall + tiny-recall at matched negative-image FP
  1/2/5/10/20% (IoU≥0.4, `src/eval/eval_detection.sweep_report` via
  `scripts/v4/compare_matched_fp.py`). Headline operating points: 5% and 10%.
- Secondary: boreal val (aerial) matched-FP recall; synthetic val slices are
  diagnostics only.
- Arm selection: best VAL recall@10% subject to not losing >1pt @5% vs A.
  Seeds: best arm + its control ×3 seeds; report mean ± spread and paired
  deltas.
- Final test: the frozen sets, once, after thresholds/choices are frozen on
  val. Per-source and aerial-vs-fixed reported separately.
- Success criterion for "synthesis helps": paired VAL gain ≥ +2pt recall@10%
  over A (and ≥ A-long, the extra-compute control is B/C/D already matched on
  steps so A-long guards the "more unique exposures" confound) confirmed on
  TEST with the same sign. Anything less → reported as no-effect/negative.
- Distillation value: E vs C under identical corpus/steps.

## Edge deliverable (frozen targets)

Export the selected student (and arm A control) to ONNX; verify FP parity
(same preds on ≥200 val images within tolerance); convert to LiteRT/TFLite
with INT8 PTQ calibrated on TRAIN/VAL only; report file size, operator
fallbacks, measured x86-CPU latency (clearly labeled as proxy), accuracy
delta INT8 vs FP on val + final test. Real phone/drone latency: UNMEASURED
(no device attached) — a benchmark harness ships instead. ≤5 MB artifact is
a target, not a promise. Server default (`expB`) stays unchanged.

## Budgets

128 GPU ceiling (≤15 L40S nodes + ≤1 B200 node), registry
`cluster/v6/registry.sh`; Bedrock ≤ USD 2,000 total / USD 200 generator
comparison, ledger `scripts/v6/bedrock_ledger.py` (reserve-before-submit).
No new instance provisioning, no new model agreements/subscriptions, no
IAM/network changes.

## AMENDMENTS

### A1 — 2026-10-20T20:55Z (after recon, BEFORE any v6 training/generation result)

1. **Teacher (rule applied)**: `expB_y11s` (`models/expB_best.pt`), sdis VAL
   matched-FP R@10% 0.329 / R@5% 0.203 — best YOLO on the frozen val
   (beats v5bA_ctrl_s3 0.302, v4A_ctrl_s2 0.304, expE 0.309). DINO B6
   (0.340) is head-incompatible with response KD and is not a YOLO teacher.
2. **Arm C/D generator**: GLM-Image (zai-org, MIT-declared; Apache-2.0
   tokenizer) — unencumbered outputs/students. Qwen-Image-2.1 is EXCLUDED
   from the training pipeline (Qwen Research License: non-commercial +
   "Built with Qwen" attribution would encumber released student weights).
3. **Bedrock-generated images** (Stability SD3.5-L/Core/Ultra/Inpaint): AWS
   Service Terms §50.5 ("express purpose of training an AI model") is
   UNCLEAR despite the Stability Bedrock EULA granting output ownership with
   no training restriction → Bedrock outputs are used ONLY for the
   generator comparison (quality/acceptance/cost measurement) and demo
   illustration, marked `permission_unclear_aws50.5`, and are NOT placed in
   any training corpus unless that reading is resolved in writing.
4. **H-NEW fresh holdout**: FireSpot-CNX-r1 (3,996 imgs / 2,817 pos /
   1,179 author-negative; never used in any v1–v5b training or evaluation).
   Licence UNSPECIFIED (permission pending) → EVAL-ONLY with disclosure;
   never training. Caveat recorded: 2,862 MNS re-encodes of FireSpot frames
   sat in v4 G-lineage corpora → H-NEW comparisons involving v4 G-lineage
   checkpoints (v4C_egt, v4Dada) carry a contamination flag; v6 students and
   arm A/A-long are clean.
5. **Corpus constants** (from inventory): N_R = 38,454 (27,197 pos /
   11,257 neg). Synthetic blocks mirror the 70.7/29.3 pos:neg ratio.
   Background pool: 12,890 train-role author-negative images
   (`data/v6/background_pool_v1.json`).
