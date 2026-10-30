# SYNTH_DATA_CARD_V6 — construction & annotation provenance

Living document; sections are appended as corpora freeze. All review in this
card is AUTOMATED or MODEL-ASSISTED (Claude Haiku 4.5 on Bedrock) — there is
NO human review of synthetic images in V6, and nothing here is presented as
human ground truth.

## Shared construction pipeline (arms B/C/D)

- Backgrounds: `data/v6/background_pool_v1.json` — 12,890 TRAIN-role images
  with author no-smoke evidence (pyro-sdis 3,424 author-exhaustive empties;
  D-Fire 7,833 author "none"-category; boreal karkkila 37; FireMan burn-A
  1,572; Sheffield rec0005 24). Excluded: MONIMET (real-plume contamination),
  FIgLib/HPWREN (CC BY-NC-ND), every val/test-role image, Boreal validation
  flights. An unlabeled background is never assumed negative.
- Placement: plume bases sampled from the EMPIRICAL distribution of real
  labelled plume bases (per-camera for pyro-sdis, global for D-Fire;
  uniform-terrain for aerial sources). Confounders go to the sky band or
  terrain level. (v1 of the corpus used uniform placement; it produced
  sky-floating plumes and was discarded before any training.)
- Degradation: blur σ∈[0,1.2], sensor noise ≤0.012, JPEG q∈[62,95] —
  IDENTICAL pipeline for positives, matched controls, confounders and plain
  negatives (same sampled params for each pos/ctrl pair), so classes are not
  separable by processing artifacts.
- Labels: boxes derive from the visible transformed alpha support (threshold
  0.03) AFTER resizing to the student input (1280 long side); one plume-level
  union box per asset; min support 25 px and min contrast 1.5/255, else the
  composite is kept as an unlabeled negative (`synth_invisible_ignore`).
  Alpha masks are construction metadata, saved under `masks/`, and are NOT
  human-perceived segmentation.
- Family traceability: every record carries bg path, asset family_id, full
  transform + degrade params (`manifest.jsonl`).

## CORPUS_B_v1 (arm B — procedural/copy-paste, NO generative model)

- 38,454 images = 27,197 positives + 11,257 negatives
  (5,628 matched controls / 3,377 confounder composites / 2,252 plain
  degraded backgrounds); faint stratum 45% of positives.
- Unique: 11,912 backgrounds, 3,000 assets (1,200 copy-paste crops from REAL
  train positives ≥24 px, feathered rounded-rect matte; 1,200 parametric
  perlin plumes; 600 parametric fog/cloud confounders).
- Errors: 0; invisible-after-retry: 0.
- Model-assisted review (300 stratified, Claude Haiku 4.5,
  `data/v6/review/synthB_sample.jsonl`):
  | stratum | n | smoke seen | plausible | med quality | top artifacts |
  |---|---|---|---|---|---|
  | faint positives | 120 | 23% | 44% | 1/5 | global_scene_change 20%, halo 8% |
  | visible positives | 80 | 38% | 44% | 1/5 | halo 26%, hard_cutout_edge 25% |
  | confounders | 50 | 2% | 96% | — | — |
  | matched controls | 30 | 3% | 100% | 5/5 | — |
  | plain negatives | 20 | 5% | 100% | — | — |
  Reading: negatives are clean; arm-B positives are frequently detectable as
  pasted (halos, rectangle edges) and the faint stratum is often below VLM
  visibility — BY DESIGN, arm B is the crude-synthesis control. The
  `text_or_watermark` flags are the real cameras' timestamp overlays, not a
  generation artifact.

## Generator comparison (frozen scene specs, `reports/v6/GENCMP_V6.md`)

Bedrock SD3.5-Large 85% asset acceptance ($0.094/accepted, 5.8 s);
Stable Image Core 17% (frame-filling smoke); Stable Image Inpaint 20/20
positives localized with 0.0000 median outside-mask change ($0.07, 17.6 s),
10/10 negative edits clean. Per prereg A1 (AWS Service Terms §50.5), Bedrock
outputs are used ONLY for this comparison and demo illustration — never
training. Stability outputs may carry invisible provenance watermarks; they
are not removed (EULA 50.12.6).

## CORPUS_C/D (arm C/D — GLM-Image black-bg assets) — SUPERSEDED

The black-background asset route was paused by prereg A3 after visual QC
(studio styling, source props, no native alpha). Its accepted pools
(`C_glm` 1,903 / `C_glm_b200` 257) are preserved as diagnostics and were
never trained on. Replaced by CORPUS_REP_v1 below.

## CORPUS_REP_v1 (arm Crep — GLM-Image i2i edits of real train backgrounds)

- Pipeline: `reports/v6/repair/SYNTH_QA_REPAIR.md` (prereg A3). GLM-Image
  i2i conditioned on eligible TRAIN-role author-negative backgrounds
  (luma≥0.25; dfire restricted to AoF fixed-camera frames after
  non-wildland web images were found in the pool), prompt bank
  v6-repair-1c (positive-only wording — GLM renders negated concepts),
  E01 positives + E02 same-background style-matched controls.
- 2,104 images = 1,351 positives + 753 controls, from 4,829 generated
  (acceptance 28%/48% under blind review rule v3; 968 uncertain EXCLUDED
  from supervision, never auto-negative). 1,891 unique backgrounds;
  same-site repetition recorded in `SCALE_RENDER_MANIFEST.json`.
- Labels: MODEL-ASSISTED weak boxes from the blind full-frame review at
  detector scale (reviewer blinded to class/location; 'uncertain' allowed;
  2–4% hallucination floor measured on controls). GLM i2i re-renders the
  whole frame (structural diff spans the image), so boxes come from the
  blind review alone; spot-check: ~3/12 boxes offset/partial — disclosed.
- Review spend ledgered (~$26 Haiku 4.5 total for pilot+scale).

## CORPUS_B_v2 revisions (visual QC before training; v1 and v2.0 discarded untrained)

Spot-checks (images reviewed by the coordinating model, recorded here) found
and fixed, in order: dark-scene additive "glow blocks" (scene-brightness
floor 0.25→0.06); square plume silhouettes at high density (density
0.35–0.8, border taper); checkerboard texture from unsmoothed value-noise
octaves and binary occlusion (per-octave Gaussian smoothing; soft occlusion
≤0.12); plumes based in cloud decks on overlook-style D-Fire backgrounds
(D-Fire anchor pool floored at y_base≥0.4); copy-paste crops showing ghost
rectangles and non-smoke content incl. a flame glow and a white billboard
(copy-paste source REMOVED — arm B is parametric-only).
Known remaining limitations: D-Fire's web-scraped backgrounds include
lookout-webcam frames (HPWREN-style) — background usage follows D-Fire's own
release, consistent with every iteration since v1; faint-stratum positives
sit near the human-visibility floor by design.

### CORPUS_B_v2 (final, parametric-only) — model-assisted review, same 300-sample protocol

| stratum | n | smoke seen | plausible | med quality | notes vs v1 |
|---|---|---|---|---|---|
| faint positives | 120 | 12% | 47% | 1/5 | no more glow; fainter => less VLM-visible |
| visible positives | 80 | 33% | 53% | 1/5 | plausibility +9pt; halo/edge flags persist |
| confounders | 50 | 2% | 96% | 4/5 | quality 1→4 |
| matched controls | 30 | 0% | 100% | 5/5 | fully clean |
| plain negatives | 20 | 0% | 100% | 5/5 | fully clean |

Reading: procedural smoke has a realism ceiling (~0.5 plausibility, low
visibility at faint opacities). That is arm B's intended role as the crude
control; the generative arms (C/D) are reviewed with the identical rubric so
the generator-realism axis is measured, not assumed. One D-Fire "none"
background was observed to contain real smoke-like haze — author-label noise
shared with the real baseline (same images train arm A), recorded as a
limitation, not corrected post hoc.
