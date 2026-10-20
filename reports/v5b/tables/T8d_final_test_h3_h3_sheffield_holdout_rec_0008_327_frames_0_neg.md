_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

### T8d. FINAL TEST H3 - H3 sheffield_holdout rec_0008 (327 frames, 0 negatives -> recall only; WITHIN-INCIDENT; CONTAMINATED for arms whose corpus holds Sheffield)

One-shot evaluation ran at 2026-10-03T03:20:07Z
2026-10-03T03:48:01Z
2026-10-03T03:49:22Z
2026-10-03T03:49:43Z
2026-10-03T03:50:03Z (reports/v5b/test/.EVALUATED_ONCE). Rows marked CONTAMINATED must not enter any comparison or decision (train/test overlap: all 327 frames are in the B/C/D train lists). Even clean rows are within-incident generalisation (same burn, site, 12-minute session as rec_0003-0007). Author boxes are fragmented sub-boxes -> box recall at IoU 0.4 is convention-limited (sheffield_uav_CARD.md section 7: 91% of author boxes are >=50% inside a predicted box while IoU>=0.4 recall is 0.085 for expE).

| system | R @ sdis-val 5% | R @ sdis-val 10% | R @ sdis-val 20% | R @ fireman-val 5% | R @ fireman-val 10% | denominators |
|---|---|---|---|---|---|---|
| S0_expE | c=0.721 R 0.008 (tiny n/a) | c=0.655 R 0.017 (tiny n/a) | c=0.535 R 0.073 (tiny n/a) | c=0.177 R 0.229 (tiny n/a) | c=0.070 R 0.270 (tiny n/a) | 327 imgs / 0 neg |
| S0b_expB | c=0.723 R 0.000 (tiny n/a) | c=0.650 R 0.006 (tiny n/a) | c=0.532 R 0.051 (tiny n/a) | c=0.144 R 0.206 (tiny n/a) | c=0.063 R 0.243 (tiny n/a) | 327 imgs / 0 neg |
| S1_A | c=0.729 R 0.018 (tiny n/a) | c=0.673 R 0.023 (tiny n/a) | c=0.534 R 0.070 (tiny n/a) | c=0.091 R 0.264 (tiny n/a) | c=0.030 R 0.299 (tiny n/a) | 327 imgs / 0 neg |
| S1_A_s1 | c=0.727 R 0.000 (tiny n/a) | c=0.666 R 0.013 (tiny n/a) | c=0.542 R 0.050 (tiny n/a) | c=0.236 R 0.186 (tiny n/a) | c=0.165 R 0.230 (tiny n/a) | 327 imgs / 0 neg |
| S1_A_s2 | c=0.717 R 0.005 (tiny n/a) | c=0.646 R 0.024 (tiny n/a) | c=0.500 R 0.096 (tiny n/a) | c=0.282 R 0.174 (tiny n/a) | c=0.179 R 0.226 (tiny n/a) | 327 imgs / 0 neg |
| S2_B **CONTAMINATED** | c=0.740 R 0.425 (tiny n/a) | c=0.672 R 0.504 (tiny n/a) | c=0.557 R 0.644 (tiny n/a) | c=0.132 R 0.939 (tiny n/a) | c=0.076 R 0.955 (tiny n/a) | 327 imgs / 0 neg |
| S2_B_s1 **CONTAMINATED** | c=0.728 R 0.374 (tiny n/a) | c=0.659 R 0.476 (tiny n/a) | c=0.528 R 0.665 (tiny n/a) | n/a (no VAL conf) | n/a (no VAL conf) | 327 imgs / 0 neg |
| S2_B_s2 **CONTAMINATED** | c=0.728 R 0.428 (tiny n/a) | c=0.656 R 0.511 (tiny n/a) | c=0.559 R 0.631 (tiny n/a) | c=0.194 R 0.877 (tiny n/a) | c=0.044 R 0.968 (tiny n/a) | 327 imgs / 0 neg |
| S3_C **CONTAMINATED** | c=0.745 R 0.372 (tiny n/a) | c=0.691 R 0.435 (tiny n/a) | c=0.585 R 0.598 (tiny n/a) | c=0.072 R 0.960 (tiny n/a) | c=0.045 R 0.973 (tiny n/a) | 327 imgs / 0 neg |
| S4_D **CONTAMINATED** | c=0.758 R 0.319 (tiny n/a) | c=0.695 R 0.457 (tiny n/a) | c=0.578 R 0.579 (tiny n/a) | c=0.082 R 0.949 (tiny n/a) | c=0.036 R 0.975 (tiny n/a) | 327 imgs / 0 neg |
| S4b_D2 **CONTAMINATED** | c=0.765 R 0.313 (tiny n/a) | c=0.703 R 0.410 (tiny n/a) | c=0.572 R 0.603 (tiny n/a) | c=0.051 R 0.962 (tiny n/a) | c=0.024 R 0.980 (tiny n/a) | 327 imgs / 0 neg |
| S4c_D3 **CONTAMINATED** | c=0.723 R 0.252 (tiny n/a) | c=0.679 R 0.324 (tiny n/a) | c=0.564 R 0.470 (tiny n/a) | c=0.262 R 0.792 (tiny n/a) | c=0.199 R 0.838 (tiny n/a) | 327 imgs / 0 neg |
| S5_ensemble_v4a_s2 | c=0.982 R 0.199 (tiny n/a) | c=0.962 R 0.212 (tiny n/a) | c=0.905 R 0.224 (tiny n/a) | n/a (no VAL conf) | n/a (no VAL conf) | 327 imgs / 0 neg |
| v4B1_frozen_L | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 327 imgs / 0 neg |
| v4B4_lora_L | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 327 imgs / 0 neg |
| v4B6_lora_Hplus | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 327 imgs / 0 neg |
| v4a_s2 | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 327 imgs / 0 neg |

Sources:
- `reports/v5b/test/eval_S0_expE_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S0b_expB_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S1_A_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S1_A_s1_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S1_A_s2_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S2_B_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S2_B_s1_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S2_B_s2_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S3_C_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S4_D_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S4b_D2_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S4c_D3_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S5_ensemble_v4a_s2_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_v4B1_frozen_L_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_v4B4_lora_L_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_v4B6_lora_Hplus_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_v4a_s2_sheffieldholdout.raw.parquet`
