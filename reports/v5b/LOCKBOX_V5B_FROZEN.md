# LOCKBOX V5B -- FROZEN (final holdouts H1 + H3, iteration 5b)

- Frozen (UTC): **2026-10-02T09:21:23+00:00**
- Built by: `scripts/v5b/build_eval_v5b.py` (DATA PREP ONLY -- no model run, no prediction, no metric).
- Manifest: `/local/home/tianrgua/cvproject/data/v5b/eval/EVAL_V5B_MANIFEST.json`
- **Evaluated once at V5b close; never used for pseudo-labels, mining, fusion fitting, threshold tuning, or model selection.**
- Pre-registration: `reports/v5b/PREREGISTRATION_V5B.md` (frozen 2026-10-02 09:30 UTC + 09:35 amendment).
- Timeline note: build stamp above is the devbox system clock; `PREREGISTRATION_V5B.md` file mtime = 2026-10-02T08:48:43+00:00 (same clock) while its text states 09:30 / 09:35 UTC -> the pre-registration file predates this build; the stated clock times in that file are not those of the system clock.

## What is in the package

| set | role | units | frames (stride) | positives | negatives | boxes | resolution |
|---|---|---|---|---|---|---|---|
| fireman_val_s5 | VALIDATION (aerial thresholds / selection) -- NOT sealed | fireman_burnD_overcast_clearcut_edge (1 video) | 1706 (every 5th of 8530 exported) | 1475 | 231 | 1475 | 958x768: 1706 |
| fireman_holdout_s10 | SEALED FINAL HOLDOUT H1 | fireman_burnC_taivalkoski_2023-08-26 (5 videos) | 1949 (every 10th of 19458 exported) | 1949 | 0 | 1949 | 640x512: 1949 |
| sheffield_holdout | FINAL HOLDOUT H3 (within-incident only) | sheffield_uav_session_2022-07-21 (rec_0008) (1 video) | 327 (all 327 exported) | 327 | 0 | 783 | 1920x1080: 327 |

### Caveats (read before interpreting any number)

- **fireman_holdout_s10 (H1) = ONE burn** (burn C, Taivalkoski 2023-08-26 -- the only GPS/time-confirmed FireMan burn, via the DJI overlay on its paired thermal videos). 5 RGB videos, 640x512 rectified RGB at thermal geometry (low resolution). Frames are every 10th frame of each video to limit within-video correlation; the effective sample size is still 1 event.
- H1 has **0 explicit negatives** (all 19,459 burn-C RGB frames are author smoke-positive) -> per the 09:35 amendment H1 reports **RECALL ONLY** (image-level + box recall @ IoU>=0.4 at the pre-registered confs). No holdout-based false-positive claim will be made. Author boxes are interpolated/projected region boxes -> box IoU metrics are labelled weak.
- H1 source: 19,459 burn-C frames; the YOLO export holds 19,458 (rgb/val/video_002 frame_000706 (presence_no_but_boxes) skipped by the YOLO export; 706 % 10 != 0 so the stride-10 selection is unaffected).
- Per-video H1 frames: rgb_test_video_002 412, rgb_train_video_009 301, rgb_train_video_010 545, rgb_train_video_011 585, rgb_val_video_002 106.
- **sheffield_holdout (H3) = within-incident only**: rec_0008 is the same prescribed burn, same site, same 12-minute UAV session as the train/val recordings rec_0003-0007 (unknown provenance, Roboflow re-deposit). It must never be reported as an independent site. 0 negatives -> recall-only.
- **H3 CONTAMINATION: CONTAMINATED for arms B/C/D.** all 327 rec_0008 frames are in the training lists of arms B/C/D (the Sheffield data agent's B_v1 corpus used ALL 1,327 exported Sheffield frames incl. val rec_0007 and holdout rec_0008); byte-identical (sha256) and same raw files (realpath). Lists: `data/v5b/yolo/B_v1/lists/local/train.txt`, `data/v5b/yolo/B_v1/lists/local/train_sheffield.txt`, `data/v5b/yolo/C_v1/lists/local/train.txt`, `data/v5b/yolo/C_v1/lists/local/train_sheffield.txt`, `data/v5b/yolo/D_v1/lists/local/train.txt`, `data/v5b/yolo/D_v1/lists/local/train_sheffield.txt`. Still valid for: A (expV5bA_ctrl_s3; trained on expB_train.txt only); expB; expE; DINOv3-L LoRA (B200, run v5bB200_lora_L_1920) -- trains on data/v5b/coco/instances_train_eb.json = E (38,454 pyro-sdis + D-Fire) + Boreal train (2,062); 0 Sheffield and 0 FireMan frames (verified from the json). H3 numbers for B/C/D (and the ACCURATE ensemble, which contains B) must be labelled CONTAMINATED (train/test overlap) and must not enter any comparison or decision; even for uncontaminated arms H3 is within-incident generalisation only (rec_0008 = same burn, site and 12-minute session as rec_0003-0007).
- **fireman_val_s5 is NOT sealed.** It is the aerial VALIDATION video (burn D, video_015, overcast clearcut edge, 958x768) used for the pre-registered aerial thresholds and the secondary selection criterion. Every 5th frame of the whole video (frame_idx % 5 == 0), so the 231 selected explicit negatives (of 1,154 in the video) keep the same cadence as the positives. One video -> one unit; its 'negatives' are one pre-/off-burn flight segment, not independent samples.
- H2 (pyro-sdis TEST, `data/yolo/lists/local/sdis_test.txt`) is NOT part of this package; it stays frozen as in v1-v4 (opened once at v4 close -> second opening to be disclosed).
- Burn identification for FireMan units A/B/D is agent visual judgement; only burn C (H1) is metadata-confirmed (`data/v5b/manifests/fireman_video_units.json`).

## Overlap check against every training list

- Reference: 20 lists (`data/v5b/yolo/*/lists/local/*.txt`, `data/yolo/lists/local/*.txt`), 387292 lines, 66035 unique images; sha256 of bytes computed for ALL of them.
- `fireman_val_s5`: path_overlap=0, sha256_overlap=0 (clean)
- `fireman_holdout_s10`: path_overlap=0, sha256_overlap=0 (clean)
- `sheffield_holdout`: path_overlap=327, sha256_overlap=327 -- hits only in ['data/v5b/yolo/B_v1/lists/local/train.txt', 'data/v5b/yolo/B_v1/lists/local/train_sheffield.txt', 'data/v5b/yolo/C_v1/lists/local/train.txt', 'data/v5b/yolo/C_v1/lists/local/train_sheffield.txt', 'data/v5b/yolo/D_v1/lists/local/train.txt', 'data/v5b/yolo/D_v1/lists/local/train_sheffield.txt'] (= the H3 contamination above)

## Seal (sha256)

- tar `/local/home/tianrgua/cvproject/data/v5b/stage/eval_v5b.tar` (466,862,080 bytes, 7994 members): `a154656669bd00ae3bdfa9c15fc69f0ea16d20545524bc547d80ce45218ae7eb`
  - pod extraction dir: `/scratch/scratch/tianrgua-sandbox/smoke/v5bdata/` -> `/scratch/scratch/tianrgua-sandbox/smoke/v5bdata/eval_v5b/<set>/{images,labels,lists,SHA256SUMS}`, `/scratch/scratch/tianrgua-sandbox/smoke/v5bdata/eval_v5b/yamls/<set>.yaml` (pod paths)
- `/local/home/tianrgua/cvproject/data/v5b/eval/fireman_val_s5/lists/local/fireman_val_s5.txt` (1706): `aa36f0e323d134f910209c5138f9cc6879bd289fbe42c020e7669fdc4903bd74`
- `/local/home/tianrgua/cvproject/data/v5b/eval/fireman_val_s5/lists/pod/fireman_val_s5.txt` (1706): `3dd406a6c989ab2c1dfbd0f1b5e92d5bd981fb08611a40d821d7a9f40c420d1f`
- `/local/home/tianrgua/cvproject/data/v5b/eval/fireman_val_s5/SHA256SUMS` (per-image sha256): `891fff3bb34b018ccfbb4ff1d6805da1f4db953d362b0087d41fe0f78240e6b0`
- `/local/home/tianrgua/cvproject/data/v5b/eval/yamls/fireman_val_s5.yaml`: `085178193d6583d192d3584415199f8875c7ccb4ab0413b2c8785b305974846d`
- `/local/home/tianrgua/cvproject/data/v5b/eval/fireman_holdout_s10/lists/local/fireman_holdout_s10.txt` (1949): `e5113180dd96370907c44c81914a54e819e1cc4498828538dc1be70b584ec6bb`
- `/local/home/tianrgua/cvproject/data/v5b/eval/fireman_holdout_s10/lists/pod/fireman_holdout_s10.txt` (1949): `38a4e780447c0a725cb12a848a5c0d454388fe26194b20e1b8189f5f0bb7f83b`
- `/local/home/tianrgua/cvproject/data/v5b/eval/fireman_holdout_s10/SHA256SUMS` (per-image sha256): `171f7c73dcbb90cece67400e3bdca78ba685be0653a652f11163e14fd2c6c71d`
- `/local/home/tianrgua/cvproject/data/v5b/eval/yamls/fireman_holdout_s10.yaml`: `3bdb66a01b0bd8ca35995bd705b7c9dd2a89cbfe89cfa55ae8705c14e613d092`
- `/local/home/tianrgua/cvproject/data/v5b/eval/sheffield_holdout/lists/local/sheffield_holdout.txt` (327): `b910f3103bd474c3b2812e33246b804cbd0a42634830a82be1e5a82dc13e5e09`
- `/local/home/tianrgua/cvproject/data/v5b/eval/sheffield_holdout/lists/pod/sheffield_holdout.txt` (327): `ebc35e54c97d5640c062347f9755216b2b644c49d130a12f06dc2735790b2624`
- `/local/home/tianrgua/cvproject/data/v5b/eval/sheffield_holdout/SHA256SUMS` (per-image sha256): `937ed63f7f4e4456ecef1c89dd332e47721a273b55ccff1ffa34eb6b7a4bd15d`
- `/local/home/tianrgua/cvproject/data/v5b/eval/yamls/sheffield_holdout.yaml`: `2abbaaf23fbff487afe299e43d1e34e7bad55eae923c5b0d46beb339704b0c47`

## Rules

1. Evaluated once at V5b close; never used for pseudo-labels, mining, fusion fitting, threshold tuning, or model selection.
2. fireman_holdout_s10 and sheffield_holdout: no script other than the one-time V5b close evaluation may read the images listed above (this builder only symlinked them, copied label text, hashed bytes and read JPEG headers). Any per-image output of that evaluation is HISTORICAL afterwards.
3. If any list/tar sha256 above changes, the seal is broken and the set must be re-frozen and re-declared before use.
4. H3 results for arms B/C/D and for any ensemble containing them are CONTAMINATED and are reported only as such; H3 is within-incident generalisation for every system.
5. H1 is recall-only (0 negatives); aerial FP rates come from FireMan VAL (fireman_val_s5, 1 video) and Boreal heinola VAL, both validation-side.
6. Layout mirrors `v4_evaltest` (images/ labels/ lists/{local,pod}/<set>.txt, SHA256SUMS, yaml with val: and test:); consumers pass the list or yaml path explicitly.

---
Clarification (2026-10-02T11:29:16Z, no content change): "the ACCURATE ensemble contains B" above
refers to v5b student arm B (trained on all Sheffield frames). The shipped ACCURATE recipe
(models/v5b/fusion/accurate_recipe.json: yolo_A + dino B4/B6/B1 — v4 DINO "B-series") contains
NO Sheffield frame in any training list → S5 is H3-clean; only systems containing a v5b arm
B/C/D/D2/D3 student are H3-contaminated.
