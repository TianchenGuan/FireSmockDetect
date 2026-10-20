_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T29. AERIAL: boreal VAL (heinola) -> boreal TEST (ruokolahti), exact TEST checkpoints

VAL columns = recall at matched neg-image FP on heinola (n_images=1,123: 906 pos / 217 neg; 1% = ~2 negatives); TEST columns = recall at the pre-registered thresholds on ruokolahti (n_images=1,764: 1,762 pos / 2 neg; FP unmeasurable). Both are ONE burn each. GT box sizes — val: {'tiny': 1, 'small': 5, 'medium': 9, 'large': 902}; test: {'tiny': 0, 'small': 39, 'medium': 52, 'large': 1793} (eval JSON gt_by_size; 'large' = >5% of image area): both are large-plume sets. The delta column compares test recall at the boreal-val-1% conf with val recall at 1% FP (same threshold, different burn). expE/expB val recall recomputed from their boreal-val parquet with the frozen matching code (v4evalr7).

| system | VAL R@1% FP | VAL R@5% FP | TEST R @ sdis-val-10% conf | TEST R @ boreal-val-1% conf | TEST(ta1) - VAL(1%) | val source |
|---|---|---|---|---|---|---|
| S0 expE | 0.553 | 0.751 | 0.877 (c=0.6546; neg flagged 0/2) | 0.920 (c=0.4983; neg flagged 0/2) | +36.7 pt | reports/v4/eval_expE_borealval.raw.parquet |
| S0b expB (deployed) | 0.663 | 0.762 | 0.886 (c=0.6503; neg flagged 0/2) | 0.921 (c=0.4983; neg flagged 0/2) | +25.8 pt | reports/v4/eval_expB_borealval.raw.parquet |
| S1 A_ctrl (v4a_s2) | 0.584 | 0.795 | 0.906 (c=0.6727; neg flagged 0/2) | 0.924 (c=0.4883; neg flagged 0/2) | +34.0 pt | seed_summary.json |
| S2 E_ET (v4e_s1) | 0.955 | n/a (FP ceiling) | 0.922 (c=0.6603; neg flagged 0/2) | 0.934 (c=0.0483; neg flagged 0/2) | -2.1 pt | seed_summary.json |
| S2c C_EGT (v4c_s1) | 0.828 | 0.936 | 0.921 (c=0.6768; neg flagged 0/2) | 0.932 (c=0.2983; neg flagged 0/2) | +10.4 pt | seed_summary.json |
| S3 B_Gonly (v4b) | 0.194 | 0.466 | 0.865 (c=0.3380; neg flagged 0/2) | 0.762 (c=0.4583; neg flagged 0/2) | +56.8 pt | seed_summary.json |
| S3 D_curr (v4dada_s1, post-hoc) | 0.914 | 0.924 | 0.923 (c=0.6530; neg flagged 0/2) | 0.931 (c=0.1683; neg flagged 0/2) | +1.7 pt | seed_summary.json |
| S4 B6 LoRA-H+ | 0.964 | 0.972 | 0.929 (c=0.9901; neg flagged 0/2) | 0.934 (c=0.8883; neg flagged 0/2) | -3.0 pt | matched_fp_dino_boreal_val.json |
| S4c B1 frozen-L | 0.974 | 0.982 | 0.929 (c=0.9407; neg flagged 0/2) | 0.934 (c=0.4983; neg flagged 0/2) | -4.0 pt | matched_fp_dino_boreal_val.json |
| S4p B4 LoRA-L | 0.973 | 0.983 | 0.932 (c=0.9881; neg flagged 0/2) | 0.934 (c=0.8683; neg flagged 0/2) | -3.8 pt | matched_fp_dino_boreal_val.json |

Sources:
- `reports/v4/seed_summary.json`
- `reports/v4/matched_fp_dino_boreal_val.json`
- `reports/v4/eval_exp{E,B}_borealval.raw.parquet`
- `reports/v4/test/yolo_test_at_preregistered_thresholds.json`
- `reports/v4/test/dino_test_at_preregistered_thresholds.json`
- `reports/v4/eval_v4a_s2_borealval.json, reports/v4/test/eval_S1_A_ctrl_borealtest.json (gt_by_size)`
