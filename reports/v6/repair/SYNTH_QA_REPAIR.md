# SYNTH_QA_REPAIR — V6 synthetic-data repair record

Executes `CC_SMOKE_SYNTH_REPAIR.md` (user brief, 2026-10-24) under prereg
amendment A3. All review here is AUTOMATED / MODEL-ASSISTED; nothing is human
ground truth. Statuses: **[OBSERVED]** verified fact, **[HYPOTHESIS]**
plausible but untested, **[DONE]** completed work, **[PENDING]** queued test.

## 0. Pause state [DONE]

- No new bulk generation; arms C/D/E NOT launched on the black-bg GLM pool.
- Preserved unmodified: `data/v6/assets/C_glm` (1,903 accepted / 1,577
  rejected: colored_object 1,393, near_dup 184), `C_glm_b200` (257),
  `data/v6/gen/*` raws, `synthB2` corpus, expV6B2_s0 checkpoints/eval
  (labeled crude-synthesis control). Real-only baselines (A, A-long) stand.
- Repair pilot generation job launched 2026-10-24 ~18:55Z (1 L40S node,
  registry job `glm_repair_pilot_rp1`; shared window/budget, no reset).

## 1. Lineage audit of the sample-zip claims

| claim from samples | verdict |
|---|---|
| arm-B trained on GLM assets? | **NO [OBSERVED]** — `synthB2/manifest.jsonl` (38,454 records): 0 contain "glm"; positives are 27,197 parametric (`assets/B/param`) + 3,377 parametric confounders. Arm-B's negative result is attributable to procedural compositing only. |
| `marguerite` leakage into training? | **NO [OBSERVED]** — 0 hits in `expB_train.txt` and in the background pool; 4,634 in `sdis_val`. The file sits in the HF dataset's `images/train/` DIRECTORY but its canonical ROLE is val. The zip folder name `real_pyro_sdis_train` was a packaging error (sampled by directory, not role list). Same lesson both directions: directory ≠ role. |
| hidden prompt rewriter adding studio style? | **NO [OBSERVED]** — installed `GlmImagePipeline` (diffusers 0.40) applies only a chat template + AR prior-token generation; no LLM prompt expansion. The studio/incense styling came from our own frozen "isolated on a pure black background" asset specs (`scene_specs.py`) + model prior. |
| negative prompts effective? | **NO [OBSERVED]** — `GlmImagePipeline.__call__` has NO `negative_prompt` parameter; `encode_prompt` hardcodes the CFG negative to `""`. The worker passes kwargs by signature introspection, so our NEGATIVE_PROMPT was **silently dropped for all 3,480 GLM generations**. Exclusions must live in the positive prompt prose (repair bank does this). |
| accepted PNGs are RGB without alpha | **CONFIRMED BY DESIGN [OBSERVED]** — no native matte exists anywhere in the chain. The compositor derives a pseudo-alpha from luminance (`clip((luma−p8)·1.8)`) in "luma additive" mode. This is exactly the flagged "thresholding RGB brightness into an alleged physical alpha". Panel `chain_glmacc_glm_1ae38d532c1a7756.png` shows its failure modes: dense cap saturates to α=1 (loses interior translucency), black-bg noise leaks in as speckle, floor-glow bands get nonzero α. |

## 2. New defects found by chain recovery (panels in `panels/`)

- **[OBSERVED] 14.7% of arm-B "positives" were trained as pure background.**
  3,992 / 27,197 `type=pos` records have `boxes_norm=[]` (visibility gate
  failed post-resize) and an EMPTY label file. YOLO has no "ignore" class:
  an empty label file is hard background supervision. Many of these frames
  still contain large smoke-textured support (e.g. `p024945`: 48,788 px
  support, manifest `visible: true`, contrast 0.65/255 < 1.5 gate) — i.e.
  ~10% of the corpus actively taught "faint smoke texture = background".
  The brief's rule ("regenerate or mark uncertain, never auto-convert to
  negative") was violated by the v2 pipeline.
  **[HYPOTHESIS]** this contradictory supervision is a major contributor to
  arm-B's recall loss at every matched-FP point (A 0.310 vs B 0.272 @10%).
  Kept as diagnostic; arm B will not be relaunched.
- **[OBSERVED] Manifest semantics inconsistent**: `visible: true` with empty
  `boxes_norm` (flag computed pre-gate). Repair corpora must write one
  unambiguous status field (`labeled_pos | uncertain_excluded | control`).
- **[OBSERVED] Parametric assets lack plume morphology**: sampled assets
  (pp01179, pp00945) are isotropic multi-octave blob clusters — no base,
  no widening, no wind shear. Composited into hazy/overcast sky they are
  indistinguishable from cloud texture (panel `chain_p001179.png`).
- **[OBSERVED] Night/storm web photos used as compositing backgrounds**
  (`WEB01758.jpg`, a watermarked nighttime storm photo): technically
  author-"none" D-Fire frames, but smoke composited into night scenes sits
  at/below the visibility floor. Background pool has no day/night filter.
- **[OBSERVED] Accepted-filter blind spot**: the `colored_object` filter
  crops bottom props but passes mushroom-cloud/explosion-like morphology,
  purple tint residue after 75% desaturation, and floor-glow bands
  (`glm_1ae38d532c1a7756`). Morphology was never a filter criterion —
  the new stage-2 crop review adds it.

## 3. Repair pilot (prereg A3) [DONE — generation in flight]

- Prompt bank `scripts/v6/repair/prompt_bank_v2.yaml` (v6-repair-1,
  from the brief; UNTESTED templates). Rendered queue
  `data/v6/repair/q_pilot.jsonl` — 360 items, every record carries the EXACT
  prompt string sent to the pipeline (no rewriter exists in the chain):
  - Route B full-scene t2i 1024×576: P01–P06 ×24 = 144 smoke
    (visibility 12 clearly / 8 small / 4 faint per template);
    N01–N03 ×24 = 72 nonsmoke (clean / fog / road-dust confounders).
  - Route A i2i (GLM native image conditioning, verified in pipeline
    source): E01 ×96 smoke (48 bgs × 2 seeds) + E02 ×48 matched controls
    on the SAME backgrounds. Backgrounds: 48 stratified TRAIN-role images
    (20 sdis / 12 dfire / 8 fireman / 4 boreal-karkkila / 4 sheffield),
    all verified present in `expB_train.txt` and absent from every
    val/test list. i2i caveat recorded: no hard mask — full-image
    conditioned regeneration; background preservation is CHECKED, not
    assumed (diff metrics below).
  - Exclusion list appended as prose to every prompt (no native negative
    support); factor variation = position × wind × lighting × forest type,
    not synonyms/seeds.
- **Labels**: requested positions are constraints, never labels. Weak labels
  come from blind stage-1 boxes (∪ i2i diff support), marked model-assisted.
### Pilot iteration log

- **v1 t2i (216/216 generated, 39 min, 1 node) — FAILED [OBSERVED]**:
  random contact-sheet inspection (2 per template, all 9 templates) shows
  saturated magenta/pink color casts and large flames in nearly every frame,
  INCLUDING the N01 clean-forest negatives; two frames render a literal CCTV
  camera object (prompt said "monitoring-camera frame"). Diagnosis
  **[HYPOTHESIS→supported]**: with no negative-prompt channel, the appended
  prose exclusion list ("Avoid: ... magenta stage lighting ... explosive
  eruptions ...") acts as POSITIVE conditioning — GLM renders the negated
  concepts. Consistent with the earlier black-bg purple-tint artifacts.
- **v1 i2i**: first attempt failed on an API mismatch (pipeline requires a
  LIST of condition images); retry in flight with the same v1 prompts —
  kept as the avoid-clause arm of the comparison.
- **v1 i2i (128/144 after list-fix retry; 16 stochastic AR-prior token-count
  failures across 13 different bgs — not bg-specific) [OBSERVED]**:
  scene COMPOSITION is well preserved (vehicles, terrain layout survive),
  so GLM i2i is usable in principle. But the avoid-clause injection hits
  i2i too (magenta plumes, flames despite "Do not add flames"); E02
  controls are NOT photometrically preserved — dusk/sunset backgrounds get
  re-rendered as dramatic fire-colored skies, one control rendered
  watermark-like text. Daylight backgrounds behave markedly better than
  dusk ones. Implication: i2i controls are generator-artifact-matched
  RECONSTRUCTIONS, not pixel-preserving edits; the blind review must (and
  does) reject any control with smoke/fire appearance.
- **v1b (controlled: avoid-clause removed, same seeds/factors) — 346/360,
  visual verdict [OBSERVED]**: large improvement — recognizable wildland
  scenes, several usable white-gray localized plumes (P04/P05/P06); i2i
  preserves composition on daylight backgrounds. Remaining failures:
  (a) both sampled N01 "clean forest" negatives still contain FIRE LINES —
  the template's own "there is no ... smoke, flame, or fire" negation is
  rendered positively; (b) literal CCTV cameras rendered in frame (both N03
  samples) from "monitoring view/camera" nouns; (c) persistent pink/salmon
  cast (weaker than v1); (d) i2i on dusk/sunset bgs still drifts to dramatic
  fire-colored skies; (e) BACKGROUND POOL contamination found: an indoor
  red-lit bar interior and night storm-chase road photos are in the pool
  (author-"none" D-Fire web images; the pool never had a scene-type filter).
- **v1c (bank `prompt_bank_v6_repair_1c.yaml`, changelog inside) [IN
  FLIGHT]**: all negations removed (negatives are purely positive scene
  descriptions that never mention smoke/fire), camera-hardware nouns
  replaced by viewpoint phrasing, smoke color pinned to "white and pale
  gray" + neutral white balance, E02 rewritten as faithful positive
  reproduction, i2i meta-sentences no longer sent to the model. Background
  list v1c: mean-luma ≥ 0.25 filter + visual exclusion of 4 confirmed
  non-wildland pool entries (B&W industrial night scene, cruise ship,
  urban sunset street, near-black dusk), replacements re-inspected.

### Pilot blind-review results [DONE] (695 images, Haiku 4.5, $1.37 ledgered)

Decision evolution (all recorded): v1 rules in `blind_review.py`; v2 = i2i
diff recomputed at 1/8 scale after the full-res diff flagged 98–100% of i2i
as restyle (GLM re-renders every pixel with film grain — the metric measured
grain, not restyle); v3 = `decide_v3.py` calibration: global_restyle no
longer rejects POSITIVES (46/48 restyle-killed v1c positives pass blind
visibility + morphology + realism≥3 — a re-rendered scene with valid smoke is
a valid training image; flag kept as `i2i_rerender`, and the pos/ctrl "pair"
claim is downgraded to style-matched same-background). Uniform realism≥3
gate added for every class.

Acceptance under rule v3 (accepted / reviewed):

| group | v1b | v1c |
|---|---|---|
| t2i smoke | 15% (21/144) | 16% (23/144) |
| t2i nonsmoke | 28% (20/72) | 19% (14/72) |
| i2i smoke | 38% (33/87) | **53% (46/86)** |
| i2i ctrl | 40% (17/43) | **57% (27/47)** |

Key v1b→v1c deltas: t2i negative contamination (blind smoke=yes on
"clean" scenes) 56%→4%; i2i ctrl contamination 21%→9%; literal CCTV
cameras eliminated; top rejection for positives remains pink/salmon
`colored_stage_light`/`neon_tint` (GLM prior; prompt-resistant).
Blind-visibility by intended tier (v1c pos): clearly 69% yes / small 64% /
faint 73% — prompt-requested visibility does NOT control actual difficulty;
strata for training must be assigned from review output, never from the
request. Reviewer hallucination floor: 2–4% smoke=yes on clean controls.
Dashboards: `dashboard_b/`, `dashboard_c/` (seeded-random + best/median/
worst/rejected; not hand-picked).

**Route decision [DONE]**: scale Route A (i2i on real train backgrounds,
bank v1c) ONLY — 53%/57% acceptance vs 16%/19% for t2i. t2i outputs are
retained as diagnostics, excluded from training.

### Scale generation [IN FLIGHT]

5,282 i2i items (3,782 E01 positives = 1,891 bgs × 2 seeds, visibility
50/35/15; 1,500 E02 controls), 6 L40S nodes, registry `glm_repair_scale`.
Eligible backgrounds: luma≥0.25 + dfire restricted to AoF fixed-camera
frames (WEB*/PublicDataset* excluded after non-wildland finds). Unique-bg
1,891; same-site repetition (sdis cameras, fireman burn-A frames, one AoF
site) recorded in `SCALE_RENDER_MANIFEST.json` — repeats are not claimed
as independent scenes. Accepted pilot-c i2i items join the corpus (same
recipe, same review).

- **Scale blind review [DONE]** (4,829 reviewed; interrupted once by Midway
  expiry at 90% and resumed; orphaned ledger reservations settled at the
  measured pilot rate, noted in the ledger). Full-set rule-v3 outcomes:
  smoke accept 1,336 / uncertain 968 / invisible 510 / morphology 224 /
  artifact 213 / low-realism 193; ctrl accept 726 / low-realism 328 /
  smoke-in-ctrl 331.
- **CORPUS_REP_v1 FROZEN [DONE]**: 2,104 images = 1,351 weak-labeled
  positives + 753 reviewed smoke-free controls (scale + 73 pilot-c
  accepted). Box resolution: for re-rendered scenes (`i2i_rerender`, ~all)
  the structural diff box spans the frame and carries no localization
  signal, so the blind stage-1 box alone is the label source
  (`blind_only_rerender`); union-with-diff only for localized edits.
  First build without this rule produced 2 positives (union≈full-frame →
  area gate); fixed BEFORE any training. Visual spot-check of 12 random
  accepted positives WITH boxes: ~9/12 well-placed, ~3/12 offset or
  covering only the plume core — weak-label noise disclosed; one pink
  fire-tinted plume slipped past the artifact gate (filter imperfect,
  recorded). Synthetic block = 5.2% of arm-Crep images.
- **Arm Crep training [IN FLIGHT]**: expB_train (38,454) + synthREP
  (2,104, unique, no multiplicity), 57 epochs ≈ arm A's total steps
  (60×38,454, Δ<1%), YOLO11n COCO init, imgsz 1280, batch 48, seed 0
  (job v6crep0, exp expV6Crep_s0). Window extension 4 recorded
  (Midway-expiry stall 10-25→10-27).
- **[PENDING]** sdis_val + boreal val matched-FP eval of Crep vs A/A-long/B2;
  report per the frozen success criterion (+2pt recall@10% over A and
  ≥A-long, else negative result).

- **blind review protocol** (`scripts/v6/repair/blind_review.py`): stage 1
  full frame at 1280 detector scale, reviewer told NOTHING about class or
  location, 'uncertain' allowed, negatives/controls reviewed identically
  (hallucination denominator); stage 2 crop morphology/artifact review;
  i2i adds local diff metrics (change_frac, p99, global_restyle) as
  construction checks only. Decision rules in the script docstring; a
  mean-pixel-delta is no longer sufficient for visibility.
- **[PENDING] dashboard**: random + best/median/worst/rejected per route,
  with bg / output / detector-size / overlay panes.
- **[PENDING] scale gate**: only if the pilot passes → frozen 2,000–5,000
  corpus, matched-steps comparison vs real-only on sdis_val (metrics and
  decision rules unchanged from the frozen prereg).

## 4. Standing constraints

Bedrock outputs stay excluded from training (AWS §50.5, prereg A1.3).
GLM-Image (MIT) remains the only generator feeding training corpora.
Shared budget/window continue (no reset); Bedrock review spend goes through
the shared ledger. FireSpot stays eval-only.
