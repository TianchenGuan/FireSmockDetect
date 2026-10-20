# V5 AERIAL threshold-calibration baseline (Day 1, no training)

Generated 2026-10-02T01:37:59+00:00 by `scripts/v5/calibrate_aerial.py` (CPU only; no cluster use). Numbers come from `data/v5/calib_aerial/calibration_aerial_v5.json`.

**Scope statements (read first).**

- Scenario = **AERIAL/UAV** only (PROTOCOL_V5 §1). Nothing here calibrates the fixed-camera service.
- Calibration material = **Boreal heinola VAL only**: 217 negative + 906 positive frames (917 GT boxes) and the 14 heinola NON-SMOKE 30-s clips. Heinola is AERIAL VAL material (PROTOCOL §2); it is **not** fixed-camera calibration data and must never enter training. All 1,123 frames and 14 clips come from **ONE prescribed burn on ONE day (2022-06-28)**; frames and clips are not independent units.
- **Boreal ruokolahti (aerial FINAL TEST) is historical and was not read** by this script.
- The 14 clips carry a **video-level** 'non-smoke' label (Boreal Subset-B); there is no frame-level truth, and detector silence is not evidence of no-smoke. Any detection on them is counted as a false alert.
- Clip scoring is a **historical replay** (`source_tag=historical_replay`), not a field run.
- AuSmoke `aus_DJI_0792_001` is a forest-holdout **VAL** unit, positives only, licence **PERMISSION PENDING** (PROTOCOL §2): it is reported as recall-at-conf only, as the only other aerial material on disk.
- Saved predictions for the 1,123 VAL frames are the v4 cluster runs (YOLO imgsz 1280 conf 0.02; DINO conf floor 0.02); the clip frames were scored locally on CPU with the same settings. Weight<->parquet parity was checked on sampled VAL frames (§6).

## 0. Headline (all numbers from §3; one burn, 7.3 valid minutes — read §8 before quoting)

- Constraint 'boreal-VAL recall >= 0.90' is **feasible for 9/9 systems**.
- S0b expB y11s (DEPLOYED default): conf* **0.1070** -> VAL recall 0.901, neg-frame FP **73/217**, clips alerted **11/14**, false episodes **90/valid h** [45, 161], time in alert 0.51; AuSmoke DJI_0792 box recall 0.306.
- S0 expE y11m (v3 candidate): conf* **0.0760** -> VAL recall 0.901, neg-frame FP **82/217**, clips alerted **13/14**, false episodes **107/valid h** [57, 182], time in alert 0.54; AuSmoke DJI_0792 box recall 0.505.
- S1 A_ctrl_E (v4a_s2, source-only): conf* **0.0750** -> VAL recall 0.902, neg-frame FP **83/217**, clips alerted **10/14**, false episodes **82/valid h** [39, 151], time in alert 0.51; AuSmoke DJI_0792 box recall 0.384.
- S2 E_ET (v4e_s1, +target aerial): conf* **0.2160** -> VAL recall 0.901, neg-frame FP **0/217**, clips alerted **6/14**, false episodes **49/valid h** [18, 107], time in alert 0.19; AuSmoke DJI_0792 box recall 0.578.
- S2c C_EGT (v4c_s1): conf* **0.1600** -> VAL recall 0.901, neg-frame FP **5/217**, clips alerted **5/14**, false episodes **41/valid h** [13, 96], time in alert 0.20; AuSmoke DJI_0792 box recall 0.587.
- S3 D_curr (v4dada_s1, curriculum): conf* **0.2120** -> VAL recall 0.901, neg-frame FP **1/217**, clips alerted **7/14**, false episodes **57/valid h** [23, 118], time in alert 0.20; AuSmoke DJI_0792 box recall 0.466.
- S4p DINOv3-L LoRA r16 (B4): conf* **0.9890** -> VAL recall 0.911, neg-frame FP **0/217**, clips alerted **4/14**, false episodes **33/valid h** [9, 84], time in alert 0.17; AuSmoke DJI_0792 box recall 0.000.
- S4 DINOv3-H+ LoRA r16 (B6, v4 pre-reg pick): conf* **0.9934** -> VAL recall 0.903, neg-frame FP **0/217**, clips alerted **4/14**, false episodes **33/valid h** [9, 84], time in alert 0.16; AuSmoke DJI_0792 box recall 0.000.
- S4c DINOv3-L frozen (B1, control) -- clips NOT scored: conf* **0.9510** -> VAL recall 0.904, neg-frame FP **0/217**, clips not scored; AuSmoke DJI_0792 box recall 0.005.
- Of the 14 'non-smoke' clips, **3 fire for every scored system** at its conf* (heinola_13, heinola_53, heinola_62) and 13 fire for at least one. The universally-fired clips need **human visual review** before they are counted as false alerts (they come from a prescribed-burn day; the label is video-level). Detector agreement is NOT evidence of smoke, and detector silence on the others is NOT evidence of no-smoke; they are review candidates for the negative-review tool, nothing more.
- The 217 negative FRAMES and the 14 clips disagree in severity: systems with 0-1/217 flagged frames still alert on 5-7/14 clips. Continuous footage is a harder negative test than curated 'empty' frames; frame-level neg-FP understates aerial false alerts.
- No single expB threshold satisfies both sides on heinola: the DEPLOYED point 0.30/0.15 gives VAL recall 0.785 (constraint NOT met) and alerts on 11/14 clips; conf* 0.107 holds recall 0.901 and alerts on 11/14 clips (73/217 negative frames flagged).
- DINO conf* lies at 0.951-0.9934 (saturated tail) and gives AuSmoke DJI_0792 box recall 0.000-0.005: the heinola-selected DINO operating points do not transfer to a different aerial source (same failure mode as the v4 FIgLib replay, reports/v4/dino_adapt_results.md).
- v4 reported 74 false episodes/h for the deployed expE point on heinola non-smoke footage (reports/v4/tables/T20) with its own episode-merge rule and thresholds; the numbers here use the PROTOCOL_V5 definition and conf* and are not directly comparable.

## 1. Material (counts from artifacts)

| material | purpose | n | note |
|---|---|---|---|
| boreal VAL frames (`data/v4/eval/boreal_val`) | recall constraint + neg-image FP | 1123 frames = 906 pos / 217 neg, 917 GT boxes | heinola, 4096x2160, human boxes |
| heinola non-smoke clips (`data/v4/sequences/boreal_frames/heinola_*`) | episode false alerts | 14 clips, 439 frames @ 1 fps = **7.32 min valid** (0.1219 h) | video-level label only; ONE burn; clips never concatenated |
| AuSmoke `aus_DJI_0792_001` (`data/v4/eval/forest_val`) | aerial recall cross-check only | 195 frames, positives only | VAL unit; permission PENDING |

The 217 negative frames come from 3 heinola flights (DJI_0024: 59, DJI_0028: 111, DJI_0048: 47); flights 0028 and 0048 also contribute 156 and 15 of the 906 positive frames, i.e. the 'empty' frames are smoke-free moments of flights that also filmed smoke, not separate no-smoke sorties (counts from data/v4/manifests/boreal_manifest.parquet and the eval parquet image names).
Per-clip frame counts: heinola_103=30, heinola_104=32, heinola_116=32, heinola_124=32, heinola_13=32, heinola_21=30, heinola_32=30, heinola_37=30, heinola_43=31, heinola_49=32, heinola_53=32, heinola_62=32, heinola_74=32, heinola_76=32.

## 2. Method

- **Recall** = frozen v1-v4 definition (`src/eval/eval_detection.sweep_report`): greedy matching, IoU >= 0.4, box-level TP/(TP+FN) over the 917 VAL GT boxes. **neg-image FP** = negative frames with >= 1 box at the conf (k/217).
- **Selection rule** (PROTOCOL §3): on a 0.001 grid from 0.02 to 0.99 (DINO: +0.0001 steps to 0.9999), take the **highest** conf with VAL recall >= **0.90**; the constraint is held first, FP is minimised second. Infeasible = recall never reaches the target even at the 0.02 floor.
- **Episodes** (PROTOCOL §3): `src/serve/alerting.AlertEngine`, one device per clip, frames at 1 fps, settings = service defaults (seq_window=5, seq_hits_required=3, cooldown_s=60.0, stale_after_s=600.0, scene_change_hamming=20) with `single_frame_alert_conf = seq_frame_conf = conf*` (**primary**: then no 'watch' state exists and an episode = any frame >= conf*). Episode starts at any state != none, ends after >= 60 s of none. Because every clip is <= 32 s, **a clip holds at most one episode**, so episodes == clips alerted; the maximum possible rate is 14 / 0.122 h = 114.8/h. Rate denominators are VALID observed hours only (sum of covered 1-s frames; no gaps bridged; no concatenation). 95% intervals are exact Poisson.
- **Variant** (secondary): `seq_frame_conf = conf*/2` as in the deployed 0.30/0.15 ratio, counting the literal 'any state != none' (includes 'watch'); reported in §5.
- **Fraction of valid time in alert** = frames in provisional/confirmed state / valid frames.

## 3. Main result — conf at VAL recall >= 0.90 and what it costs on heinola non-smoke footage

| system | conf* | VAL recall (box) | neg-img FP k/217 | clips alerted /14 | **false episodes / valid h** [95% CI] | frac time in alert | AuSmoke DJI_0792 box recall (frames w/ any box, not matched) — VAL unit, PENDING |
|---|---|---|---|---|---|---|---|
| S0b expB y11s (DEPLOYED default) | 0.1070 | 0.901 | 73/217 (0.336) | 11 | **90.2** [45.0, 161.4] | 0.508 | 0.306 (170/195 frames w/ any box) |
| S0 expE y11m (v3 candidate) | 0.0760 | 0.901 | 82/217 (0.378) | 13 | **106.6** [56.8, 182.3] | 0.544 | 0.505 (192/195 frames w/ any box) |
| S1 A_ctrl_E (v4a_s2, source-only) | 0.0750 | 0.902 | 83/217 (0.383) | 10 | **82.0** [39.3, 150.8] | 0.508 | 0.384 (183/195 frames w/ any box) |
| S2 E_ET (v4e_s1, +target aerial) | 0.2160 | 0.901 | 0/217 (0.000) | 6 | **49.2** [18.1, 107.1] | 0.187 | 0.578 (166/195 frames w/ any box) |
| S2c C_EGT (v4c_s1) | 0.1600 | 0.901 | 5/217 (0.023) | 5 | **41.0** [13.3, 95.7] | 0.203 | 0.587 (192/195 frames w/ any box) |
| S3 D_curr (v4dada_s1, curriculum) | 0.2120 | 0.901 | 1/217 (0.005) | 7 | **57.4** [23.1, 118.3] | 0.196 | 0.466 (148/195 frames w/ any box) |
| S4p DINOv3-L LoRA r16 (B4) | 0.9890 | 0.911 | 0/217 (0.000) | 4 | **32.8** [8.9, 84.0] | 0.173 | 0.000 (0/195 frames w/ any box) |
| S4 DINOv3-H+ LoRA r16 (B6, v4 pre-reg pick) | 0.9934 | 0.903 | 0/217 (0.000) | 4 | **32.8** [8.9, 84.0] | 0.164 | 0.000 (0/195 frames w/ any box) |
| S4c DINOv3-L frozen (B1, control) -- clips NOT scored | 0.9510 | 0.904 | 0/217 (0.000) | - | clips not scored | - | 0.005 (1/195 frames w/ any box) |

Valid time = 7.32 min (439 one-second frames, 14 clips, 1 fps, ONE burn). With this little footage the rates are **anecdotal**: even 0 episodes only bounds the rate below ~30/h, and 14/14 clips alerted is the ceiling. They are reported because PROTOCOL §3 asks for episodes per valid hour, not because 7 minutes can establish a deployable rate.

## 4. Deployed reference point (expB at the service defaults 0.30 / 0.15) — NOT a calibrated point

VAL recall 0.785 (constraint 0.90 **NOT met**), neg-img FP 51/217; clips alerted 11/14, false episodes 90.2/valid h [45.0, 161.4] (provisional/confirmed), frac time in alert 0.476; literal any-state!=none 90.2/h, frac 0.497; AuSmoke DJI_0792 recall 0.131.

## 5. Recall/false-alert trade-off (same rule at other recall targets; primary alert config)

Shows how much false-alert reduction the recall constraint buys or forgoes. Episodes are provisional/confirmed; the `!=none` column is the literal reading with `seq_frame_conf = conf*/2` (watch state counted).

| system | target | conf | VAL recall | neg FP k/217 | clips /14 | episodes/valid h [CI] | frac alert | variant seq=conf*/2: clips!=none /14, ep/h, frac!=none |
|---|---|---|---|---|---|---|---|---|
| S0b expB y11s (DEPLOYED default) | >=0.95 | 0.0400 | 0.951 | 79/217 | 12 | 98.4 [50.8, 171.9] | 0.558 | 12, 98.4/h, 0.588 |
| S0b expB y11s (DEPLOYED default) | >=0.90 | 0.1070 | 0.901 | 73/217 | 11 | 90.2 [45.0, 161.4] | 0.508 | 11, 90.2/h, 0.531 |
| S0b expB y11s (DEPLOYED default) | >=0.85 | 0.1870 | 0.851 | 70/217 | 11 | 90.2 [45.0, 161.4] | 0.487 | 11, 90.2/h, 0.512 |
| S0b expB y11s (DEPLOYED default) | >=0.80 | 0.2690 | 0.800 | 62/217 | 11 | 90.2 [45.0, 161.4] | 0.444 | 11, 90.2/h, 0.499 |
| S0 expE y11m (v3 candidate) | >=0.95 | 0.0370 | 0.952 | 101/217 | 13 | 106.6 [56.8, 182.3] | 0.665 | 13, 106.6/h, 0.738 |
| S0 expE y11m (v3 candidate) | >=0.90 | 0.0760 | 0.901 | 82/217 | 13 | 106.6 [56.8, 182.3] | 0.544 | 13, 106.6/h, 0.665 |
| S0 expE y11m (v3 candidate) | >=0.85 | 0.0980 | 0.851 | 63/217 | 13 | 106.6 [56.8, 182.3] | 0.490 | 13, 106.6/h, 0.626 |
| S0 expE y11m (v3 candidate) | >=0.80 | 0.1620 | 0.800 | 38/217 | 10 | 82.0 [39.3, 150.8] | 0.353 | 13, 106.6/h, 0.533 |
| S1 A_ctrl_E (v4a_s2, source-only) | >=0.95 | 0.0320 | 0.951 | 95/217 | 12 | 98.4 [50.8, 171.9] | 0.572 | 13, 106.6/h, 0.613 |
| S1 A_ctrl_E (v4a_s2, source-only) | >=0.90 | 0.0750 | 0.902 | 83/217 | 10 | 82.0 [39.3, 150.8] | 0.508 | 12, 98.4/h, 0.558 |
| S1 A_ctrl_E (v4a_s2, source-only) | >=0.85 | 0.1540 | 0.851 | 56/217 | 9 | 73.8 [33.7, 140.1] | 0.421 | 10, 82.0/h, 0.506 |
| S1 A_ctrl_E (v4a_s2, source-only) | >=0.80 | 0.2310 | 0.800 | 19/217 | 8 | 65.6 [28.3, 129.3] | 0.278 | 10, 82.0/h, 0.465 |
| S2 E_ET (v4e_s1, +target aerial) | >=0.95 | 0.0630 | 0.953 | 2/217 | 7 | 57.4 [23.1, 118.3] | 0.214 | 8, 65.6/h, 0.248 |
| S2 E_ET (v4e_s1, +target aerial) | >=0.90 | 0.2160 | 0.901 | 0/217 | 6 | 49.2 [18.1, 107.1] | 0.187 | 6, 49.2/h, 0.201 |
| S2 E_ET (v4e_s1, +target aerial) | >=0.85 | 0.2890 | 0.851 | 0/217 | 5 | 41.0 [13.3, 95.7] | 0.178 | 6, 49.2/h, 0.196 |
| S2 E_ET (v4e_s1, +target aerial) | >=0.80 | 0.3760 | 0.800 | 0/217 | 4 | 32.8 [8.9, 84.0] | 0.148 | 6, 49.2/h, 0.187 |
| S2c C_EGT (v4c_s1) | >=0.95 | 0.0220 | 0.951 | 41/217 | 10 | 82.0 [39.3, 150.8] | 0.335 | 10, 82.0/h, 0.346 |
| S2c C_EGT (v4c_s1) | >=0.90 | 0.1600 | 0.901 | 5/217 | 5 | 41.0 [13.3, 95.7] | 0.203 | 6, 49.2/h, 0.223 |
| S2c C_EGT (v4c_s1) | >=0.85 | 0.2430 | 0.852 | 4/217 | 4 | 32.8 [8.9, 84.0] | 0.194 | 5, 41.0/h, 0.207 |
| S2c C_EGT (v4c_s1) | >=0.80 | 0.3620 | 0.800 | 2/217 | 4 | 32.8 [8.9, 84.0] | 0.184 | 5, 41.0/h, 0.201 |
| S3 D_curr (v4dada_s1, curriculum) | >=0.95 | 0.0490 | 0.951 | 67/217 | 12 | 98.4 [50.8, 171.9] | 0.544 | 13, 106.6/h, 0.620 |
| S3 D_curr (v4dada_s1, curriculum) | >=0.90 | 0.2120 | 0.901 | 1/217 | 7 | 57.4 [23.1, 118.3] | 0.196 | 11, 90.2/h, 0.314 |
| S3 D_curr (v4dada_s1, curriculum) | >=0.85 | 0.3400 | 0.851 | 0/217 | 4 | 32.8 [8.9, 84.0] | 0.175 | 9, 73.8/h, 0.207 |
| S3 D_curr (v4dada_s1, curriculum) | >=0.80 | 0.4560 | 0.801 | 0/217 | 3 | 24.6 [5.1, 71.9] | 0.166 | 7, 57.4/h, 0.191 |
| S4p DINOv3-L LoRA r16 (B4) | >=0.95 | 0.9780 | 0.952 | 0/217 | 4 | 32.8 [8.9, 84.0] | 0.184 | 7, 57.4/h, 0.257 |
| S4p DINOv3-L LoRA r16 (B4) | >=0.90 | 0.9890 | 0.911 | 0/217 | 4 | 32.8 [8.9, 84.0] | 0.173 | 7, 57.4/h, 0.255 |
| S4p DINOv3-L LoRA r16 (B4) | >=0.85 | 0.9919 | 0.854 | 0/217 | 4 | 32.8 [8.9, 84.0] | 0.166 | 7, 57.4/h, 0.255 |
| S4p DINOv3-L LoRA r16 (B4) | >=0.80 | 0.9939 | 0.801 | 0/217 | 4 | 32.8 [8.9, 84.0] | 0.162 | 7, 57.4/h, 0.255 |
| S4 DINOv3-H+ LoRA r16 (B6, v4 pre-reg pick) | >=0.95 | 0.9830 | 0.951 | 0/217 | 5 | 41.0 [13.3, 95.7] | 0.196 | 6, 49.2/h, 0.248 |
| S4 DINOv3-H+ LoRA r16 (B6, v4 pre-reg pick) | >=0.90 | 0.9934 | 0.903 | 0/217 | 4 | 32.8 [8.9, 84.0] | 0.164 | 6, 49.2/h, 0.246 |
| S4 DINOv3-H+ LoRA r16 (B6, v4 pre-reg pick) | >=0.85 | 0.9953 | 0.851 | 0/217 | 4 | 32.8 [8.9, 84.0] | 0.162 | 6, 49.2/h, 0.246 |
| S4 DINOv3-H+ LoRA r16 (B6, v4 pre-reg pick) | >=0.80 | 0.9964 | 0.804 | 0/217 | 3 | 24.6 [5.1, 71.9] | 0.160 | 6, 49.2/h, 0.246 |
| S4c DINOv3-L frozen (B1, control) -- clips NOT scored | >=0.95 | 0.9100 | 0.951 | 0/217 | - | not scored | - | - |
| S4c DINOv3-L frozen (B1, control) -- clips NOT scored | >=0.90 | 0.9510 | 0.904 | 0/217 | - | not scored | - | - |
| S4c DINOv3-L frozen (B1, control) -- clips NOT scored | >=0.85 | 0.9600 | 0.853 | 0/217 | - | not scored | - | - |
| S4c DINOv3-L frozen (B1, control) -- clips NOT scored | >=0.80 | 0.9650 | 0.810 | 0/217 | - | not scored | - | - |

## 6. Weight <-> saved-prediction parity (local CPU weights vs v4 cluster parquet, sampled VAL frames)

| system | n imgs | boxes local/saved (>=0.05) | matched | max |d score| | median |d score| | unmatched |
|---|---|---|---|---|---|---|
| S0b expB y11s (DEPLOYED default) | 8 | 5/5 | 5 | 0.0001 | 0.0001 | 0 |
| S0 expE y11m (v3 candidate) | 8 | 6/6 | 6 | 0.0002 | 0.0001 | 0 |
| S1 A_ctrl_E (v4a_s2, source-only) | 8 | 5/5 | 5 | 0.0001 | 0.0000 | 0 |
| S2 E_ET (v4e_s1, +target aerial) | 8 | 5/5 | 5 | 0.0000 | 0.0000 | 0 |
| S2c C_EGT (v4c_s1) | 8 | 6/6 | 6 | 0.0000 | 0.0000 | 0 |
| S3 D_curr (v4dada_s1, curriculum) | 8 | 6/6 | 6 | 0.0001 | 0.0001 | 0 |
| S4p DINOv3-L LoRA r16 (B4) | 8 | 4/4 | 4 | 0.0000 | 0.0000 | 0 |
| S4 DINOv3-H+ LoRA r16 (B6, v4 pre-reg pick) | 8 | 4/4 | 4 | 0.0000 | 0.0000 | 0 |
| S4c DINOv3-L frozen (B1, control) -- clips NOT scored | not run | | | | | |

Small score differences are CPU-fp32 vs GPU kernels; a mismatch in box count or a large score gap would mean the weights file is not the checkpoint behind the parquet.

## 7. Per-clip detail at conf* (which non-smoke clips fire)

| system | conf* | clips alerted (first alert s, max score) |
|---|---|---|
| S0b expB y11s (DEPLOYED default) | 0.1070 | heinola_103 (t=11s, max 0.308); heinola_104 (t=0s, max 0.664, confirmed); heinola_116 (t=26s, max 0.458, confirmed); heinola_13 (t=0s, max 0.698, confirmed); heinola_21 (t=0s, max 0.386, confirmed); heinola_37 (t=0s, max 0.340, confirmed); heinola_43 (t=23s, max 0.337, confirmed); heinola_49 (t=0s, max 0.341, confirmed); heinola_53 (t=0s, max 0.798, confirmed); heinola_62 (t=4s, max 0.517, confirmed); heinola_74 (t=0s, max 0.514, confirmed) |
| S0 expE y11m (v3 candidate) | 0.0760 | heinola_103 (t=0s, max 0.455, confirmed); heinola_104 (t=0s, max 0.677, confirmed); heinola_116 (t=0s, max 0.140); heinola_124 (t=0s, max 0.101); heinola_13 (t=0s, max 0.554, confirmed); heinola_21 (t=0s, max 0.285, confirmed); heinola_37 (t=0s, max 0.251, confirmed); heinola_43 (t=16s, max 0.212, confirmed); heinola_49 (t=0s, max 0.150, confirmed); heinola_53 (t=0s, max 0.759, confirmed); heinola_62 (t=0s, max 0.567, confirmed); heinola_74 (t=0s, max 0.419, confirmed); heinola_76 (t=12s, max 0.220, confirmed) |
| S1 A_ctrl_E (v4a_s2, source-only) | 0.0750 | heinola_103 (t=0s, max 0.450, confirmed); heinola_104 (t=0s, max 0.720, confirmed); heinola_13 (t=0s, max 0.611, confirmed); heinola_21 (t=0s, max 0.306, confirmed); heinola_37 (t=0s, max 0.267, confirmed); heinola_49 (t=0s, max 0.210, confirmed); heinola_53 (t=0s, max 0.768, confirmed); heinola_62 (t=0s, max 0.605, confirmed); heinola_74 (t=1s, max 0.258, confirmed); heinola_76 (t=14s, max 0.136, confirmed) |
| S2 E_ET (v4e_s1, +target aerial) | 0.2160 | heinola_104 (t=22s, max 0.271); heinola_116 (t=0s, max 0.556, confirmed); heinola_124 (t=0s, max 0.351); heinola_13 (t=0s, max 0.584, confirmed); heinola_53 (t=0s, max 0.827, confirmed); heinola_62 (t=0s, max 0.700, confirmed) |
| S2c C_EGT (v4c_s1) | 0.1600 | heinola_104 (t=21s, max 0.661, confirmed); heinola_124 (t=25s, max 0.207, confirmed); heinola_13 (t=0s, max 0.705, confirmed); heinola_53 (t=0s, max 0.817, confirmed); heinola_62 (t=0s, max 0.787, confirmed) |
| S3 D_curr (v4dada_s1, curriculum) | 0.2120 | heinola_104 (t=26s, max 0.450, confirmed); heinola_116 (t=20s, max 0.261); heinola_13 (t=0s, max 0.757, confirmed); heinola_53 (t=0s, max 0.839, confirmed); heinola_62 (t=0s, max 0.906, confirmed); heinola_74 (t=20s, max 0.298); heinola_76 (t=18s, max 0.247) |
| S4p DINOv3-L LoRA r16 (B4) | 0.9890 | heinola_104 (t=26s, max 0.995, confirmed); heinola_13 (t=0s, max 0.997, confirmed); heinola_53 (t=0s, max 1.000, confirmed); heinola_62 (t=0s, max 1.000, confirmed) |
| S4 DINOv3-H+ LoRA r16 (B6, v4 pre-reg pick) | 0.9934 | heinola_103 (t=2s, max 0.996); heinola_13 (t=0s, max 0.999, confirmed); heinola_53 (t=0s, max 0.999, confirmed); heinola_62 (t=0s, max 0.999, confirmed) |

Clip max-score summary (any box >= 0.02 floor): S0b expB: median 0.072, p90 0.501, max 0.798, frames with any box 247/439; S0 expE: median 0.078, p90 0.520, max 0.759, frames with any box 302/439; S1 A_ctrl_E: median 0.056, p90 0.548, max 0.768, frames with any box 256/439; S2 E_ET: median 0.000, p90 0.541, max 0.827, frames with any box 115/439; S2c C_EGT: median 0.000, p90 0.726, max 0.817, frames with any box 147/439; S3 D_curr: median 0.045, p90 0.774, max 0.906, frames with any box 266/439; S4p DINOv3-L: median 0.000, p90 0.999, max 1.000, frames with any box 130/439; S4 DINOv3-H+: median 0.000, p90 0.999, max 0.999, frames with any box 129/439.

## 8. Limitations (state, do not soften)

- ONE burn, ONE day, ONE aircraft type: the 217 negatives, the 14 clips and the 906 positives share background, lighting and smoke character. A conf* chosen here is a heinola operating point, not an aerial operating point; the v4 heinola->ruokolahti transfer already showed val gaps shrinking on the test burn (reports/v4/AERIAL_EVAL_V4.md).
- 7.3 valid minutes of non-smoke footage cannot establish an episode rate; PROTOCOL §6 requires >= 2 h / >= 3 independent units per scenario. The aerial scenario currently has 0 h of independent no-smoke UAV footage beyond this burn.
- Clip labels are video-level. If a clip contained faint smoke that the labeller did not count, an 'episode' here would be a true alert; conversely detector silence on these clips says nothing about their content.
- The 217 negative frames (Subset-A 'Empty-Images') and the 14 clips (Subset-B videos) come from the same flights/day; they are not two independent negative sources, and the frames' FP and the clips' episodes are correlated readings of the same footage.
- Recall is box-level over 917 GT boxes dominated by large plumes (902 'large'); 'recall >= 0.90' says nothing about faint/early smoke.
- DINO scores saturate (>= 0.99 operating points): conf* for B4/B6 sits on a 0.0001 grid in the saturated tail and is fragile; see reports/v4/dino_adapt_results.md for the threshold-transfer failure on FIgLib.
- Training exposure differs by system: expB/expE (v1-v3 recipes) saw no Boreal material at all; the v4 T-arms (E_ET, C_EGT, D_curr) and the DINO LoRA runs (E_ET recipe) trained on the OTHER Boreal burns (evo, karkkila, incl. their 37 verified-empty frames). The heinola background is unseen by every system, but Boreal smoke/background character is not new to the T-arms or DINO — their lower false-alert counts here are partly same-dataset familiarity, not proof of UAV generalisation (the v4 ruokolahti test showed the heinola val gap shrinking).
- AuSmoke permission is PENDING; its number is a cross-check on existing v4 predictions, not a licence to use the data further.

## 9. Artifacts

- `data/v5/calib_aerial/calibration_aerial_v5.json` — every number above, full conf curves not stored (recompute via `calibrate`).
- `data/v5/calib_aerial/clip_preds_<system>.parquet` — per-frame boxes/scores for the 439 clip frames (`source_tag=historical_replay`).
- `data/v5/calib_aerial/parity_<system>.json`, `data/v5/calib_aerial/logs/score_<system>.log`.
- Inputs (read only): `reports/v4/eval_*_borealval.raw.parquet`, `reports/v4/eval_*_boreal_val.raw.parquet`, `reports/v4/eval_*_forestval.raw.parquet`, `data/v4/sequences/sequences_v4.parquet`, `data/v4/manifests/boreal_manifest.parquet`.
