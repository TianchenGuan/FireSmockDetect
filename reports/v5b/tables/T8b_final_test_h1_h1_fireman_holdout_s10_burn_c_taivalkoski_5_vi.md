_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

### T8b. FINAL TEST H1 - H1 fireman_holdout_s10 (burn C Taivalkoski, 5 videos, 1,949 frames, 0 negatives -> RECALL ONLY): recall at the pre-registered VAL confidences

One-shot evaluation ran at 2026-10-03T03:20:07Z
2026-10-03T03:48:01Z
2026-10-03T03:49:22Z
2026-10-03T03:49:43Z
2026-10-03T03:50:03Z (reports/v5b/test/.EVALUATED_ONCE). H1 has 0 explicit negatives (amendment 09:35 UTC) -> RECALL ONLY (image-level + box recall @ IoU>=0.4); no holdout FP claim. Author boxes are interpolated region boxes -> box IoU metrics are weak; one burn, 640x512 rectified RGB, every 10th frame. Exact recall at each conf is recomputed from the test parquet with the frozen matching code.

| system | R @ sdis-val 5% | R @ sdis-val 10% | R @ sdis-val 20% | R @ fireman-val 5% | R @ fireman-val 10% | denominators |
|---|---|---|---|---|---|---|
| S0_expE | c=0.721 R 0.464 (tiny n/a) | c=0.655 R 0.663 (tiny n/a) | c=0.535 R 0.888 (tiny n/a) | c=0.177 R 0.995 (tiny n/a) | c=0.070 R 1.000 (tiny n/a) | 1,949 imgs / 0 neg |
| S0b_expB | c=0.723 R 0.388 (tiny n/a) | c=0.650 R 0.590 (tiny n/a) | c=0.532 R 0.831 (tiny n/a) | c=0.144 R 0.992 (tiny n/a) | c=0.063 R 0.999 (tiny n/a) | 1,949 imgs / 0 neg |
| S1_A | c=0.729 R 0.612 (tiny n/a) | c=0.673 R 0.736 (tiny n/a) | c=0.534 R 0.907 (tiny n/a) | c=0.091 R 0.998 (tiny n/a) | c=0.030 R 1.000 (tiny n/a) | 1,949 imgs / 0 neg |
| S1_A_s1 | c=0.727 R 0.201 (tiny n/a) | c=0.666 R 0.438 (tiny n/a) | c=0.542 R 0.761 (tiny n/a) | c=0.236 R 0.985 (tiny n/a) | c=0.165 R 0.996 (tiny n/a) | 1,949 imgs / 0 neg |
| S1_A_s2 | c=0.717 R 0.444 (tiny n/a) | c=0.646 R 0.630 (tiny n/a) | c=0.500 R 0.908 (tiny n/a) | c=0.282 R 0.994 (tiny n/a) | c=0.179 R 0.999 (tiny n/a) | 1,949 imgs / 0 neg |
| S2_B | c=0.740 R 0.522 (tiny n/a) | c=0.672 R 0.718 (tiny n/a) | c=0.557 R 0.900 (tiny n/a) | c=0.132 R 0.991 (tiny n/a) | c=0.076 R 0.999 (tiny n/a) | 1,949 imgs / 0 neg |
| S2_B_s1 | c=0.728 R 0.426 (tiny n/a) | c=0.659 R 0.650 (tiny n/a) | c=0.528 R 0.912 (tiny n/a) | n/a (no VAL conf) | n/a (no VAL conf) | 1,949 imgs / 0 neg |
| S2_B_s2 | c=0.728 R 0.402 (tiny n/a) | c=0.656 R 0.680 (tiny n/a) | c=0.559 R 0.876 (tiny n/a) | c=0.194 R 0.986 (tiny n/a) | c=0.044 R 0.999 (tiny n/a) | 1,949 imgs / 0 neg |
| S3_C | c=0.745 R 0.709 (tiny n/a) | c=0.691 R 0.856 (tiny n/a) | c=0.585 R 0.950 (tiny n/a) | c=0.072 R 1.000 (tiny n/a) | c=0.045 R 1.000 (tiny n/a) | 1,949 imgs / 0 neg |
| S4_D | c=0.758 R 0.153 (tiny n/a) | c=0.695 R 0.438 (tiny n/a) | c=0.578 R 0.799 (tiny n/a) | c=0.082 R 0.997 (tiny n/a) | c=0.036 R 0.999 (tiny n/a) | 1,949 imgs / 0 neg |
| S4b_D2 | c=0.765 R 0.158 (tiny n/a) | c=0.703 R 0.480 (tiny n/a) | c=0.572 R 0.837 (tiny n/a) | c=0.051 R 0.996 (tiny n/a) | c=0.024 R 0.996 (tiny n/a) | 1,949 imgs / 0 neg |
| S4c_D3 | c=0.723 R 0.614 (tiny n/a) | c=0.679 R 0.725 (tiny n/a) | c=0.564 R 0.889 (tiny n/a) | c=0.262 R 0.975 (tiny n/a) | c=0.199 R 0.986 (tiny n/a) | 1,949 imgs / 0 neg |
| S5_ensemble_v4a_s2 | c=0.982 R 0.981 (tiny n/a) | c=0.962 R 0.983 (tiny n/a) | c=0.905 R 0.983 (tiny n/a) | n/a (no VAL conf) | n/a (no VAL conf) | 1,949 imgs / 0 neg |
| v4B1_frozen_L | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 1,949 imgs / 0 neg |
| v4B4_lora_L | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 1,949 imgs / 0 neg |
| v4B6_lora_Hplus | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 1,949 imgs / 0 neg |
| v4a_s2 | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 1,949 imgs / 0 neg |

Sources:
- `reports/v5b/test/eval_S0_expE_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S0b_expB_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S1_A_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S1_A_s1_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S1_A_s2_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S2_B_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S2_B_s1_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S2_B_s2_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S3_C_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S4_D_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S4b_D2_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S4c_D3_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S5_ensemble_v4a_s2_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_v4B1_frozen_L_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_v4B4_lora_L_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_v4B6_lora_Hplus_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_v4a_s2_firemanholdout.raw.parquet`
