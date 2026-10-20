_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

### T8c. FINAL TEST H2 - H2 pyro-sdis TEST (ferion + croix-augas, 4,124 imgs / 917 negatives; second opening after v4): matched-FP recall / tiny / precision

One-shot evaluation ran at 2026-10-03T03:20:07Z
2026-10-03T03:48:01Z
2026-10-03T03:49:22Z
2026-10-03T03:49:43Z
2026-10-03T03:50:03Z (reports/v5b/test/.EVALUATED_ONCE). Deployment rule (PREREGISTRATION_V5B.md): replace the FAST default only if a candidate beats expB on H2 R@10% by >= 2 pt AND does not lose on H1 image-level recall AND passes integration tests. v4 comparison points: reports/v4/tables/T25_*.md (same split, same protocol).

| system | 1% FP | 2% FP | 5% FP | 10% FP | 20% FP | denominators |
|---|---|---|---|---|---|---|
| S0_expE | c=0.766 R 0.107 / tiny 0.034 / P 0.968 | c=0.727 R 0.172 / tiny 0.081 / P 0.957 | c=0.671 R 0.256 / tiny 0.154 / P 0.924 | c=0.599 R 0.350 / tiny 0.257 / P 0.895 | c=0.457 R 0.506 / tiny 0.406 / P 0.835 | 4,124 imgs / 917 neg |
| S0b_expB | c=0.759 R 0.118 / tiny 0.026 / P 0.963 | c=0.721 R 0.171 / tiny 0.052 / P 0.951 | c=0.655 R 0.258 / tiny 0.115 / P 0.923 | c=0.577 R 0.354 / tiny 0.206 / P 0.887 | c=0.409 R 0.501 / tiny 0.364 / P 0.819 | 4,124 imgs / 917 neg |
| S1_A | c=0.773 R 0.108 / tiny 0.035 / P 0.970 | c=0.723 R 0.183 / tiny 0.092 / P 0.963 | c=0.644 R 0.287 / tiny 0.184 / P 0.928 | c=0.555 R 0.388 / tiny 0.283 / P 0.893 | c=0.391 R 0.553 / tiny 0.435 / P 0.833 | 4,124 imgs / 917 neg |
| S1_A_s1 | c=0.778 R 0.091 / tiny 0.020 / P 0.966 | c=0.756 R 0.124 / tiny 0.040 / P 0.942 | c=0.675 R 0.261 / tiny 0.149 / P 0.927 | c=0.591 R 0.373 / tiny 0.253 / P 0.893 | c=0.452 R 0.521 / tiny 0.426 / P 0.837 | 4,124 imgs / 917 neg |
| S1_A_s2 | c=0.782 R 0.102 / tiny 0.026 / P 0.964 | c=0.736 R 0.172 / tiny 0.065 / P 0.952 | c=0.663 R 0.280 / tiny 0.155 / P 0.928 | c=0.571 R 0.383 / tiny 0.244 / P 0.890 | c=0.426 R 0.511 / tiny 0.384 / P 0.833 | 4,124 imgs / 917 neg |
| S2_B | c=0.776 R 0.089 / tiny 0.019 / P 0.960 | c=0.739 R 0.141 / tiny 0.051 / P 0.951 | c=0.676 R 0.221 / tiny 0.116 / P 0.915 | c=0.602 R 0.309 / tiny 0.199 / P 0.879 | c=0.474 R 0.442 / tiny 0.325 / P 0.821 | 4,124 imgs / 917 neg |
| S2_B_s1 | c=0.779 R 0.085 / tiny 0.017 / P 0.955 | c=0.739 R 0.145 / tiny 0.053 / P 0.951 | c=0.662 R 0.242 / tiny 0.126 / P 0.916 | c=0.567 R 0.349 / tiny 0.228 / P 0.875 | c=0.428 R 0.470 / tiny 0.349 / P 0.805 | 4,124 imgs / 917 neg |
| S2_B_s2 | c=0.766 R 0.091 / tiny 0.020 / P 0.955 | c=0.729 R 0.140 / tiny 0.045 / P 0.947 | c=0.654 R 0.229 / tiny 0.110 / P 0.912 | c=0.551 R 0.332 / tiny 0.213 / P 0.872 | c=0.388 R 0.471 / tiny 0.359 / P 0.790 | 4,124 imgs / 917 neg |
| S3_C | c=0.782 R 0.085 / tiny 0.020 / P 0.959 | c=0.742 R 0.144 / tiny 0.062 / P 0.943 | c=0.682 R 0.221 / tiny 0.135 / P 0.912 | c=0.596 R 0.318 / tiny 0.214 / P 0.875 | c=0.457 R 0.461 / tiny 0.365 / P 0.820 | 4,124 imgs / 917 neg |
| S4_D | c=0.786 R 0.089 / tiny 0.022 / P 0.962 | c=0.746 R 0.149 / tiny 0.070 / P 0.947 | c=0.682 R 0.228 / tiny 0.144 / P 0.919 | c=0.589 R 0.342 / tiny 0.249 / P 0.877 | c=0.457 R 0.462 / tiny 0.356 / P 0.823 | 4,124 imgs / 917 neg |
| S4b_D2 | c=0.776 R 0.116 / tiny 0.038 / P 0.962 | c=0.749 R 0.153 / tiny 0.067 / P 0.949 | c=0.669 R 0.255 / tiny 0.151 / P 0.920 | c=0.582 R 0.348 / tiny 0.245 / P 0.880 | c=0.429 R 0.482 / tiny 0.374 / P 0.809 | 4,124 imgs / 917 neg |
| S4c_D3 | c=0.744 R 0.096 / tiny 0.015 / P 0.954 | c=0.722 R 0.129 / tiny 0.037 / P 0.941 | c=0.628 R 0.260 / tiny 0.143 / P 0.908 | c=0.550 R 0.358 / tiny 0.250 / P 0.876 | c=0.451 R 0.466 / tiny 0.364 / P 0.802 | 4,124 imgs / 917 neg |
| S5_ensemble_v4a_s2 | n/a | n/a | c=0.969 R 0.234 / tiny 0.102 / P 0.914 | c=0.935 R 0.366 / tiny 0.202 / P 0.880 | c=0.854 R 0.552 / tiny 0.367 / P 0.819 | 4,124 imgs / 917 neg |
| v4B1_frozen_L | PENDING | PENDING | PENDING | PENDING | PENDING | - |
| v4B4_lora_L | PENDING | PENDING | PENDING | PENDING | PENDING | - |
| v4B6_lora_Hplus | PENDING | PENDING | PENDING | PENDING | PENDING | - |
| v4a_s2 | c=0.759 R 0.130 / tiny 0.054 / P 0.970 | c=0.732 R 0.170 / tiny 0.084 / P 0.959 | c=0.660 R 0.273 / tiny 0.175 / P 0.927 | c=0.564 R 0.385 / tiny 0.287 / P 0.890 | c=0.409 R 0.549 / tiny 0.441 / P 0.834 | 4,124 imgs / 917 neg |

Sources:
- `reports/v5b/test/eval_S0_expE_sdistest.raw.parquet`
- `reports/v5b/test/eval_S0b_expB_sdistest.raw.parquet`
- `reports/v5b/test/eval_S1_A_sdistest.raw.parquet`
- `reports/v5b/test/eval_S1_A_s1_sdistest.raw.parquet`
- `reports/v5b/test/eval_S1_A_s2_sdistest.raw.parquet`
- `reports/v5b/test/eval_S2_B_sdistest.raw.parquet`
- `reports/v5b/test/eval_S2_B_s1_sdistest.raw.parquet`
- `reports/v5b/test/eval_S2_B_s2_sdistest.raw.parquet`
- `reports/v5b/test/eval_S3_C_sdistest.raw.parquet`
- `reports/v5b/test/eval_S4_D_sdistest.raw.parquet`
- `reports/v5b/test/eval_S4b_D2_sdistest.raw.parquet`
- `reports/v5b/test/eval_S4c_D3_sdistest.raw.parquet`
- `reports/v5b/test/eval_S5_ensemble_v4a_s2_sdistest.raw.parquet`
- `reports/v5b/test/eval_v4a_s2_sdistest.raw.parquet`
