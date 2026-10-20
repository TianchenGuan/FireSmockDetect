_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T27. FINAL TEST — AuSmoke FOREST TEST (new region, ACT Australia; 9 units / 4 independent events incl. ONE DJI aerial flight)

Denominators: n_images=2,871 (positives 2,871, negatives 0) — positives-only, so false alarms are UNMEASURABLE on this domain and matched-FP is undefined; recall-at-threshold only. Independent events: g_20241021, g_20240326_fb769, g_20240517_fb884 (ground burns) and g_dji_20230922 (one aerial flight, 3 videos, one of them a staged test-run reported separately). Never pool the 9 units as 9 events; the EVENT rows are frame-weighted over a group's units.

**(a) whole set — recall / tiny-recall at the pre-registered thresholds**

| system | SOURCE-ONLY sdis-val conf@5% | SOURCE-ONLY sdis-val conf@10% | SOURCE-ONLY sdis-val conf@20% | TARGET-ADAPTED boreal-val conf@1% | TARGET-ADAPTED boreal-val conf@5% |
|---|---|---|---|---|---|
| S0 expE (iteration-2 YOLO11m candidate) | 0.009 / 0.023 (c=0.7210) | 0.033 / 0.048 (c=0.6546) | 0.076 / 0.052 (c=0.5354) | 0.092 / 0.058 (c=0.4983) | 0.174 / 0.095 (c=0.2615) |
| S0b expB (deployed default, YOLO11s@1280) | 0.001 / 0.000 (c=0.7231) | 0.021 / 0.001 (c=0.6503) | 0.084 / 0.029 (c=0.5323) | 0.101 / 0.047 (c=0.4983) | 0.166 / 0.075 (c=0.3477) |
| S1 A_ctrl_E, median seed | 0.006 / 0.012 (c=0.7283) | 0.030 / 0.047 (c=0.6727) | 0.077 / 0.056 (c=0.5392) | 0.092 / 0.065 (c=0.4883) | 0.171 / 0.093 (c=0.2558) |
| S2 E_ET, median seed | 0.039 / 0.001 (c=0.7154) | 0.080 / 0.009 (c=0.6603) | 0.156 / 0.072 (c=0.5534) | 0.462 / 0.268 (c=0.0483) | n/a(val FP floor) |
| S2c C_EGT, median seed | 0.032 / 0.000 (c=0.7344) | 0.067 / 0.009 (c=0.6768) | 0.153 / 0.105 (c=0.5767) | 0.266 / 0.183 (c=0.2983) | 0.355 / 0.237 (c=0.0705) |
| S3 B_Gonly, seed 0 | 0.000 / 0.000 (c=0.4231) | 0.000 / 0.000 (c=0.3380) | 0.017 / 0.008 (c=0.1885) | 0.000 / 0.000 (c=0.4583) | 0.000 / 0.000 (c=0.3558) |
| S3 D_curr, median seed | 0.025 / 0.001 (c=0.7292) | 0.070 / 0.006 (c=0.6530) | 0.183 / 0.095 (c=0.5258) | 0.351 / 0.209 (c=0.1683) | 0.370 / 0.215 (c=0.1303) |
| S4 v4B6_lora_Hplus (DINOv3 LoRA-H+) | 0.145 / 0.068 (c=0.9939) | 0.210 / 0.101 (c=0.9901) | 0.280 / 0.131 (c=0.9801) | 0.416 / 0.180 (c=0.8883) | 0.508 / 0.222 (c=0.5115) |
| S4c v4B1_frozen_L (DINOv3 frozen-L) | 0.139 / 0.025 (c=0.9677) | 0.214 / 0.066 (c=0.9407) | 0.307 / 0.119 (c=0.8680) | 0.447 / 0.185 (c=0.4983) | 0.571 / 0.246 (c=0.0558) |
| S4p v4B4_lora_L (DINOv3 LoRA-L) | 0.109 / 0.038 (c=0.9934) | 0.180 / 0.079 (c=0.9881) | 0.281 / 0.139 (c=0.9730) | 0.440 / 0.207 (c=0.8683) | 0.570 / 0.252 (c=0.1458) |

**(b) per unit and per independent EVENT at the TARGET-ADAPTED boreal-val-1% confidence** (conf per system: S1_A_ctrl 0.4883, S2_E_ET 0.0483, S3_D_curr 0.1683, S0_expE 0.4983, S4_B6_loraH 0.8883)

| event (group) | unit | frames | S1_A_ctrl | S2_E_ET | S3_D_curr | S0_expE | S4_B6_loraH |
|---|---|---|---|---|---|---|---|
| g_20240326_fb769 | aus_27032024_S1020005 | 167 | 0.057 | 0.500 | 0.366 | 0.062 | 0.598 |
| g_20240326_fb769 | aus_FB769_Bonshaw_March_26_202403251112 | 530 | 0.372 | 0.812 | 0.673 | 0.316 | 0.650 |
| g_20240326_fb769 | aus_FB769_Bonshaw_March_26_202403261044_Panasonic | 566 | 0.043 | 0.723 | 0.500 | 0.045 | 0.574 |
| **g_20240326_fb769 — EVENT (frame-weighted, 3 units)** |  | 1263 | **0.183** | **0.731** | **0.555** | **0.161** | **0.609** |
| g_20240517_fb884 | aus_20240517_FB884_Sony4K_1300_MAH00006 | 404 | 0.002 | 0.038 | 0.014 | 0.000 | 0.072 |
| **g_20240517_fb884 — EVENT (frame-weighted, 1 unit)** |  | 404 | **0.002** | **0.038** | **0.014** | **0.000** | **0.072** |
| g_20241021 | aus_20241021_S1050004 | 670 | 0.003 | 0.579 | 0.387 | 0.004 | 0.554 |
| g_20241021 | aus_2024_10_21_06_1fps | 276 | 0.004 | 0.118 | 0.100 | 0.003 | 0.068 |
| **g_20241021 — EVENT (frame-weighted, 2 units)** |  | 946 | **0.003** | **0.445** | **0.303** | **0.004** | **0.412** |
| g_dji_20230922 (aerial flight) | aus_DJI_20230922112303_0008_D | 25 | 0.407 | 0.667 | 0.667 | 0.222 | 0.556 |
| g_dji_20230922 (aerial flight) | aus_DJI_20230922112554_0009_D | 143 | 0.685 | 1.000 | 1.000 | 0.944 | 1.000 |
| g_dji_20230922 (aerial flight) | aus_DJI_20230922115131_0004_D_TEST_RUN_SMOKE **[staged TEST_RUN]** | 90 | 0.086 | 0.796 | 0.806 | 0.129 | 0.946 |
| **g_dji_20230922 — EVENT (frame-weighted, 3 units)** |  | 258 | **0.449** | **0.896** | **0.900** | **0.590** | **0.938** |
| g_dji_20230922 — EVENT excl. staged TEST_RUN |  | 168 | 0.644 | 0.950 | 0.950 | 0.837 | 0.934 |

Sources:
- `reports/v4/test/yolo_test_at_preregistered_thresholds.json`
- `reports/v4/test/dino_test_at_preregistered_thresholds.json`
- `reports/v4/test/forest_test_per_unit.json`
- `data/v4/manifests/forest_holdout_v4.json (groups)`
- `data/v4/eval/EVALTEST_MANIFEST.json (aerial_event, staged_test_run)`
