# V6 synthetic smoke repair — execution brief

(Received 2026-10-24 from the user, verbatim. Execution record: `SYNTH_QA_REPAIR.md`.)

Continue the current project. This is a targeted dataset repair, not a new giant model sweep.

## Evidence and immediate action

The uploaded sample package contains 30 images, two per folder, and a README. It does NOT contain the generation prompts, opacity masks, training boxes, full corpus manifest, or complete training metrics. Do not extrapolate a corpus-wide failure rate from these selected samples.

The README distinguishes three different pipelines:
- `real_*`: original dataset images;
- `synth_armB_*`: procedural composites on real training backgrounds;
- `gen_glm_*`: GLM-Image assets;
- `gen_bedrock_*`: comparison outputs excluded from training.

Therefore do not blame arm-B's training performance on GLM assets unless the actual arm-B manifest proves those assets were used. Trace data lineage first.

The inspected GLM examples have studio/colored-light styling, source objects or floor artifacts, and narrow incense-like stems. Two samples labeled `accepted` still have problematic shapes or surface cues. The two supplied accepted PNGs are RGB, not RGBA; a valid separate opacity map MAY exist elsewhere, so inspect the actual pipeline rather than assuming either its existence or absence.

The procedural positive samples are difficult to localize visually in the full image. These are suspicious positive labels, not proof that all synthetic examples are unusable. Strong real aerial smoke is plainly visible in several original-source samples, so do not discard real data because it looks less polished.

Pause new bulk generation and new large training jobs that consume the suspected synthetic corpus. Preserve artifacts and checkpoints; keep independent real-only baseline work running. Do not delete the old corpus or overwrite its version. Already expensive jobs can be checkpointed or left as explicitly labeled flawed-corpus controls; do not keep expanding them blindly.

## 1. Recover and inspect the actual generation chain

For each selected sample recover the raw prompt, any prompt-rewriter output, seed, generator/model revision, background/source ID and split, mask/alpha, transformations, final labels, and arm membership. Inspect candidate input, raw asset, extracted matte, composite, final encoded image, and the EXACT detector-preprocessed image.

Test RGB/BGR ordering, straight versus premultiplied alpha, double opacity multiplication, gamma/color handling, downsampling, JPEG encoding, bbox transforms, cropping, and label clipping. A black RGB background is not a native transparency channel. Do not threshold RGB brightness into an alleged physical alpha without recognizing the ambiguity and validating the result.

Check whether automatic prompt expansion adds cinematic/studio/product-photography language. Log the rewritten prompt and compare rewriting on/off; do not assume the rewrite is harmless.

Do not use a fixed mean pixel delta such as 5/255 as the sole visibility gate. A detectable pixel change can be noise or an editing artifact rather than recognizable smoke.

Verify source-split membership from canonical manifests, not folders: the sample folder called `real_pyro_sdis_train` includes a marguerite filename, a historically held-out site. This could be only a misleading sample-folder name. Audit lineage; do not claim leakage without confirming actual training membership.

## 2. Run a structured prompt pilot

Use the accompanying `smoke_prompt_bank_v2.yaml` as UNTESTED starting templates. Adapt to the supported model API rather than assuming every model supports negative prompts, masks, RGBA, or exact spatial instructions.

Prioritize these routes:
A. A permitted local image-editing model editing approved REAL TRAINING backgrounds.
B. A permitted local model generating complete, ordinary forest/UAV monitoring scenes.
C. RGBA assets only if the system provides a genuinely usable alpha/matte and the plume morphology passes inspection.

GLM-Image supports image-to-image generation; inspect its installed implementation. Text-only spatial instructions are not a hard mask. Local-edit instructions do not guarantee background preservation. Validate actual changes and actual smoke extent.

Stop using 'smoke on black background' as the default main prompt. Aim for ordinary monitoring views, localized vegetation-smoke sources, irregular widening plumes, coherent wind direction, recognizable interiors, and translucent edges. Reject neon-colored studio artifacts and source props. Natural gray/blue-gray/brown variation is allowed; do not turn one color into the definition of smoke.

Vary actual factors: ground versus oblique aerial view, forest type, camera direction, smoke scale, contrast, wind, partial occlusion and ordinary lighting. Do not call synonyms or seed changes alone meaningful scene diversity. Avoid concentrating the first pilot on sunset, near-black backgrounds, or almost invisible smoke.

Generate a 300–500-image pilot, including smoke-free and confounder controls. Use a proposed positive difficulty mix of 50% clearly visible, 35% small but discernible, and 15% faint but discernible; this is a starting distribution, not a known optimum. Exclude effectively invisible examples from hard-positive supervision; regenerate or mark uncertain rather than automatically converting them to negatives.

Full-scene generation and editing create different label-quality risks. Requested coordinates/masks are NOT output labels. Derive localization from actual outputs, mark automated masks as weak labels, and exclude doubtful localizations from hard box/mask loss.

## 3. Make semantic review automatic but difficult to game

Run an image-capable reviewer on the final full frame at the actual detector input scale WITHOUT showing the desired location or calling it smoke-positive. Ask whether smoke is visible and where it is. Allow 'uncertain'. Evaluate negatives identically to measure reviewer hallucination.

Then run a local crop review for morphology and artifacts. Pair differences and known alpha can aid construction checks but must not be the only evidence of smoke. Do not rely exclusively on the current YOLO, because it may reject the novel hard examples we need.

Score visibility, plausible localized origin, shape, scene fit, unwanted edits, labeling reliability, and diversity separately. Preserve reviewer model/prompt/version and reasons. Automated review is not human ground truth.

Export a manageable dashboard with raw assets, full composites, detector-sized inputs, and overlays; include seeded-random examples plus best, median, worst and rejected outputs. No hand-picked-only showcase and no thousands-frame manual-review prerequisite.

## 4. Verify utility before scaling

After the repaired pilot passes the automatic checks, build a frozen 2,000–5,000-image training subset. Compare real-only and repaired-synthetic training with the SAME student initialization, optimizer steps, resolution and evaluation. Retain the old flawed-corpus result as a diagnostic, not a result to hide.

Start with one approved generator/route and a simple mixture. A 50/50 real/synthetic arm is allowed, not an acceptance requirement; include a lower-synthetic arm if feasible. Balance positive/negative examples and processing artifacts independently of the real/synthetic split.

Use real validation to choose the recipe. All test images, masks, crops and backgrounds remain excluded from generation, prompt references and selection. No editing of final test data. Do not claim that attractive synthetic images prove better real-world detection or that static small smoke proves earlier wildfire warning.

Expand only if the repaired recipe has useful real-validation behavior. Prompt variants, assets and backgrounds must remain traceable; repeated variants of one background are not independent events.

## 5. Constraints and deliverables

Retain the existing exclusion of the packaged Bedrock outputs until their applicable training-use permissions are affirmatively established. Do not bypass the exclusion merely because the pictures look better. Use an already approved local route while resolving any restrictions. No new emails or human-labeling campaign are required.

Use the existing authorized compute/account configuration and SHARED remaining budget. This repair does not reset the previous GPU, paid-inference, or wall-clock authorization. Use one coordinator and bounded retries; do not launch a full fleet for a pilot.

Deliver `SYNTH_QA_REPAIR.md`, a versioned prompt bank with actual rendered prompts, lineage/mask manifests, the pilot dashboard, quality counts with denominators, and a matched real-validation training comparison. Distinguish observations, hypotheses, completed work, and pending tests.

Begin with the pipeline checks and a small visible-smoke prompt pilot, not another large training launch on the current questionable positives.
