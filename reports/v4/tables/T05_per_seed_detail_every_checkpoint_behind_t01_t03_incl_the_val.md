_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T05. Per-seed detail (every checkpoint behind T01-T03) incl. the val-selected confidences used as thresholds

conf@X% = the checkpoint's own confidence at X% neg-image FP on that val set (these are the SOURCE-ONLY regime thresholds from sdis_val and the TARGET-ADAPTED thresholds from boreal_val in PREREGISTRATION_V4.md). Denominators: sdis n_images=7,335 (positives 6,177, negatives 1,158); boreal n_images=1,123 (positives 906, negatives 217); forest n_images=1,471 (positives 1,471, negatives 0). 'n/a' = FP target not reachable.

| arm | ckpt | sdis R@1% | sdis R@10% | sdis tiny@10% | sdis conf@5% | sdis conf@10% | boreal R@1% | boreal R@5% | boreal conf@1% | forest R@0.20 | forest R@0.30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_ctrl_E | v4a | 0.053 | 0.309 | 0.201 | 0.717 | 0.661 | 0.640 | 0.783 | 0.408 | 0.268 | 0.194 |
| A_ctrl_E | v4a_s1 | 0.056 | 0.302 | 0.191 | 0.729 | 0.673 | 0.562 | 0.800 | 0.498 | 0.234 | 0.156 |
| A_ctrl_E | v4a_s2 | 0.056 | 0.304 | 0.196 | 0.728 | 0.673 | 0.584 | 0.795 | 0.488 | 0.237 | 0.158 |
| E_ET | v4e | 0.046 | 0.247 | 0.132 | 0.746 | 0.691 | 0.931 | n/a | 0.068 | 0.266 | 0.218 |
| E_ET | v4e_s1 | 0.040 | 0.291 | 0.172 | 0.715 | 0.660 | 0.955 | n/a | 0.048 | 0.297 | 0.241 |
| E_ET | v4e_s2 | 0.045 | 0.293 | 0.161 | 0.728 | 0.673 | 0.828 | 0.846 | 0.334 | 0.267 | 0.205 |
| C_EGT | v4c | 0.041 | 0.266 | 0.164 | 0.735 | 0.684 | 0.934 | 0.936 | 0.084 | 0.306 | 0.233 |
| C_EGT | v4c_s1 | 0.045 | 0.262 | 0.151 | 0.734 | 0.677 | 0.828 | 0.936 | 0.298 | 0.224 | 0.155 |
| C_EGT | v4c_s2 | 0.044 | 0.262 | 0.167 | 0.737 | 0.686 | 0.940 | n/a | 0.048 | 0.305 | 0.223 |
| D_curr | v4dada | 0.082 | 0.300 | 0.192 | 0.753 | 0.669 | 0.909 | 0.939 | 0.128 | 0.233 | 0.171 |
| D_curr | v4dada_s1 | 0.055 | 0.318 | 0.180 | 0.729 | 0.653 | 0.914 | 0.924 | 0.168 | 0.234 | 0.169 |
| D_curr | v4dada_s2 | 0.042 | 0.345 | 0.229 | 0.698 | 0.625 | 0.864 | 0.881 | 0.338 | 0.267 | 0.191 |
| B_Gonly | v4b | 0.021 | 0.092 | 0.007 | 0.423 | 0.338 | 0.194 | 0.466 | 0.458 | 0.001 | 0.000 |

Sources:
- `reports/v4/seed_summary.json`
