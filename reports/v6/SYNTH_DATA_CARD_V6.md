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

## CORPUS_C/D (arm C/D — GLM-Image) — pending generation

Generator: zai-org/GLM-Image (MIT-declared; Apache-2.0 tokenizer), run on
project GPUs from locally staged weights. Same compositor, same controls.
(appended when frozen)
