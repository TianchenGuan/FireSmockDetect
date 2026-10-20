# Synthetic-smoke pilot (Track F, iteration 2) — Qwen-Image-Edit-2511

Date: 2026-09-11. Model: Qwen/Qwen-Image-Edit-2511 @
`6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9` (Apache-2.0), pipeline
`QwenImageEditPlusPipeline`, diffusers 0.40.0, bf16 + per-GPU
`enable_model_cpu_offload` (57.7 GB weights > 48 GB L40S).

## What was generated

450 images = 150 TRAIN-split pyro-sdis negative backgrounds (sites brison 49 /
courmettes 40 / valbonne 40 / serre-de-barre 21; 20 cameras; n_boxes==0;
val/test/FIgLib/lockbox NEVER touched) x 3 matched jobs each:

| kind | n | prompt intent |
|---|---|---|
| smoke | 150 | faint tiny/small low-contrast plume at a requested region |
| control | 150 | no-change (75) / minor brightness (75) — same pipeline artifacts, no smoke |
| confounder | 150 | fog bank (50) / dust haze (50) / low cloud (50) at the same region, no smoke |

Spec (exact prompts, seeds 42000-42449, region boxes, params: 30 steps —
deliberate deviation from the model-card example 40 for the 45-min segment
budget — true_cfg 4.0, negative_prompt " "): `data/v2/synth/spec.jsonl`.
Outputs 1376x768 PNG (pipeline ~1MP resize of 1280x720) + per-image
provenance JSON in `data/v2/synth/raw/`. **Every output is marked
verified=false / label_status=unverified-synthetic: the requested
region/prompt is a generation REQUEST, not a verified label — and screening
(below) shows the requests were largely NOT honored.**

## Throughput / cost (green g6e, 8x L40S per node, 1 node per job)

- 5 segments (jobs tianrgua-synthseg1..5), all SUCCEEDED:
  43+96+115+115+81 = 450/450 images. Node wall time 27+39+44+44+37 =
  191 min => **~25.5 GPU-hours total, ~3.4 GPU-min/image, 2.36 img/min/node
  overall**.
- Steady-state **~123 s/image/GPU** (30 steps, CFG on, 1376x768);
  segment mean 179-286 s/img because the FIRST image per worker pays
  **~1000 s** faulting the 57.7 GB mmap through the page cache (cold node).
  Weight staging job itself: 3 min (S3 -> /scratch, byte-verified).
- Extrapolation: 1-2k images ≈ 57-114 GPU-h (~7-14 node-hours) at this
  config. Feasible, but see recommendation.

## Screening result (AUTOMATED / MODEL-ASSISTED — NOT human)

Method: an AI agent (Claude) read the contact sheets
(`contact_sheets/smokezoom_00..03.png` = all 150 smoke jobs as 3x-region
crops; `triplet_00/06/09/12.png` = 48/150 rows for controls+confounders,
all cameras represented) plus a quantitative diff-vs-matched-control
locality metric (`edit_locality.json`). Judgments were made at thumbnail
resolution; no per-pixel human verification. Full per-image labels:
`screening.json`; quarantine: `quarantine.txt`.

Smoke jobs (150):

| category | n | % |
|---|---|---|
| plausible plume, at least near the requested region | 31 | 21% |
| plausible plume but oversized and/or far off region | 48 | 32% |
| weak / uncertain at screening resolution | 28 | 19% |
| no plume visible / global haze wash only | 37 | 25% |
| visible FLAMES (never requested) | 6 | 4% |

Key findings:
1. **Realism is good, adherence is bad.** ~53% of smoke jobs produce a
   photorealistic wildfire plume, but almost all look like established
   fires — the "extremely faint tiny wisp" instruction is honored in
   perhaps 8-12% of cases. We asked for the hard positives we lack;
   we mostly got the easy positives we already have.
2. **Region prompts are ignored.** Median fraction of edit mass inside the
   2x-dilated requested box is 0.04 (i.e. chance-level); median changed-area
   vs matched control is 19.5% of the frame (smoke) / 36.5% (confounder).
   The plume typically lands at a terrain feature near, not in, the box.
   => **requested boxes are unusable as detection labels.** Any training use
   needs post-hoc box derivation (diff-vs-control segmentation and/or
   detector/VLM localization) + verification.
3. **Failure modes:** 25% produce nothing or a global haze wash; 4% add
   visible flames; occasional scene corruption (repetitive "stamped" forest
   texture: bg016/bg060/bg132) and global tone/exposure drift.
4. **Controls work as designed but reveal a fingerprint risk:** no-change
   controls still differ from their source (mean |diff| 6-10/255, p99
   50-119) — the whole synthetic set carries a VAE/re-render fingerprint a
   detector could shortcut on. Synth positives must only ever be paired
   with synth negatives (controls/confounders) in training mixes.
5. **Confounders:** fog = mostly plausible, often nicely localized (good
   hard negatives); dust = near-always a GLOBAL sepia wash (usable only as
   global-haze negatives); cloud = pasted-looking cumulus, occasionally
   smoke-like (bg005 quarantined).

Quarantine (uncertain or unusable-as-labeled; 72 files listed in
`quarantine.txt`): all 6 flames + 37 no-plume + 28 weak/uncertain smoke
jobs + 1 smoke-like cloud confounder. The remaining 79 smoke images are
usable as *image-level* "synthetic smoke somewhere" positives only after
box re-derivation; none are usable as-labeled.

## Go / no-go recommendation

**NO-GO on expanding to 1-2k with the current text-prompt protocol, and
NO-GO on training the YOLO ablation from this batch as labeled.**
Reasons: (a) no usable boxes (region adherence ~0); (b) wrong difficulty
distribution (prominent plumes, not the faint/tiny regime the detector
actually misses); (c) 25-30% label noise from empty/global-haze generations;
(d) 4% flame contamination; (e) re-render fingerprint risk.

**Conditional GO for a revised protocol worth one more (small, ~100-image)
pilot before any scale-up**, changing what failed rather than hoping volume
fixes it:
1. Localized conditioning instead of percent-coordinate text: inpainting-
   style masked edit (crop-edit-composite around the requested region, or a
   region mask if a supported pipeline variant is available) so the box IS
   the label by construction, and composite artifacts stay local.
2. Faintness control: generate at plume-free strength then alpha-blend /
   early-stop denoise, or prompt for distance ("smoke on the horizon many
   kilometers away") — verify against the same zoom-sheet screen.
3. Keep the matched control/confounder design (it worked) and keep fog;
   fix dust prompts ("localized dust cloud behind a vehicle on a dirt
   road"), keep clouds only with a smoke-likeness screen.
4. Automated post-hoc verification loop before any training use: diff-vs-
   control plume segmentation -> derived box -> screen with the SFT-9B VLM
   or expB YOLO (agreement = keep, disagreement = quarantine), then a HUMAN
   spot-check of >=10% — this pilot's screening was model-assisted only.

Per instruction, NO YOLO retraining with synthetic data was launched; that
decision awaits review of this report.

## Artifacts

- container: `000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-diffusion:v1`
  digest `sha256:c35e10b30927de16495cee94bfa4c3e3b6d865d0ee7ecacf45d23a6ceb315edd`
  (cluster/v2/Dockerfile.diffusion; CPU class-load verified from config, no weights)
- weights: HF snapshot byte-verified (35 files / 57,720,463,453 B) ->
  `s3://smoke-detect-data-000000000001-use1/weights/qwen-image-edit-2511/` ->
  `/scratch/scratch/tianrgua-sandbox/models/qwen-image-edit-2511/`
  (job tianrgua-wsqwenimageedit2-29ue, ok=true, 0 mismatches)
- generation: cluster/v2/{gen_synth_pod.py,synth_gen_entry.sh,submit_synth_gen.py};
  jobs tianrgua-synthseg{1..5}-* (all SUCCEEDED); per-segment tars under
  data/v2/synth/segments/
- selection/spec: scripts/build_synth_spec_v2.py -> spec.jsonl, SPEC_META.json,
  backgrounds.txt(+.tar)
- post-processing: scripts/postprocess_synth_v2.py -> raw/, contact_sheets/,
  generation_stats.json, edit_locality.json, screening.json, quarantine.txt
