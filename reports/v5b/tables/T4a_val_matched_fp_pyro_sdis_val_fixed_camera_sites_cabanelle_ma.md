_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

### T4a. VAL matched-FP: pyro-sdis VAL (fixed camera; sites cabanelle + marguerite; 7,335 imgs / 1,158 negatives; 1% FP = ~12 negatives)

Recall / tiny-recall (<0.1% area GT) / precision at matched negative-image FP, interpolated on each system's own confidence sweep (src.eval.eval_detection.sweep_report, IoU>=0.4, grid 0.01..0.99 step 0.01; +0.991..0.999 for DINO-family tokens, as scripts/v4/compare_matched_fp.py --top-fine). 'tiny n/a' = the set has no tiny GT boxes (FireMan/Boreal author boxes are large plume regions). The eval JSONs record `imgsz: 1152`, which is eval_detection's CLI default when fed --preds-jsonl; the pod inference ran at IMGSZ=1280 (cluster/v4/yolo_eval_entry.sh default). FireMan VAL recall saturates for v1-era models -> it is NOT a faint-smoke benchmark (STATUS 10:57 UTC).

| system | 1% FP | 2% FP | 5% FP | 10% FP | 20% FP | denominators |
|---|---|---|---|---|---|---|
| S0_expE | c=0.799 R 0.059 / tiny 0.013 / P 0.961 | c=0.775 R 0.092 / tiny 0.026 / P 0.946 | c=0.721 R 0.190 / tiny 0.097 / P 0.932 | c=0.655 R 0.309 / tiny 0.201 / P 0.915 | c=0.535 R 0.479 / tiny 0.364 / P 0.888 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S0b_expB | c=0.808 R 0.058 / tiny 0.011 / P 0.961 | c=0.780 R 0.098 / tiny 0.030 / P 0.954 | c=0.723 R 0.204 / tiny 0.104 / P 0.940 | c=0.650 R 0.329 / tiny 0.203 / P 0.925 | c=0.532 R 0.485 / tiny 0.359 / P 0.887 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S1_A | c=0.812 R 0.056 / tiny 0.012 / P 0.954 | c=0.780 R 0.108 / tiny 0.041 / P 0.950 | c=0.729 R 0.202 / tiny 0.109 / P 0.938 | c=0.673 R 0.302 / tiny 0.191 / P 0.921 | c=0.534 R 0.499 / tiny 0.374 / P 0.892 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S1_A_s1 | c=0.807 R 0.043 / tiny 0.007 / P 0.957 | c=0.785 R 0.077 / tiny 0.021 / P 0.948 | c=0.727 R 0.184 / tiny 0.089 / P 0.938 | c=0.666 R 0.287 / tiny 0.167 / P 0.916 | c=0.542 R 0.458 / tiny 0.331 / P 0.887 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S1_A_s2 | c=0.809 R 0.050 / tiny 0.011 / P 0.963 | c=0.789 R 0.080 / tiny 0.025 / P 0.953 | c=0.717 R 0.204 / tiny 0.100 / P 0.940 | c=0.646 R 0.313 / tiny 0.179 / P 0.924 | c=0.500 R 0.493 / tiny 0.338 / P 0.891 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S2_B | c=0.814 R 0.056 / tiny 0.011 / P 0.965 | c=0.785 R 0.103 / tiny 0.027 / P 0.957 | c=0.740 R 0.183 / tiny 0.075 / P 0.941 | c=0.672 R 0.302 / tiny 0.174 / P 0.916 | c=0.557 R 0.457 / tiny 0.335 / P 0.881 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S2_B_s1 | c=0.813 R 0.037 / tiny 0.005 / P 0.946 | c=0.784 R 0.082 / tiny 0.022 / P 0.948 | c=0.728 R 0.180 / tiny 0.085 / P 0.933 | c=0.659 R 0.292 / tiny 0.173 / P 0.908 | c=0.528 R 0.449 / tiny 0.325 / P 0.871 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S2_B_s2 | c=0.816 R 0.042 / tiny 0.006 / P 0.950 | c=0.784 R 0.093 / tiny 0.025 / P 0.950 | c=0.728 R 0.196 / tiny 0.089 / P 0.941 | c=0.656 R 0.328 / tiny 0.204 / P 0.916 | c=0.559 R 0.465 / tiny 0.329 / P 0.882 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S3_C | c=0.819 R 0.049 / tiny 0.009 / P 0.957 | c=0.796 R 0.089 / tiny 0.027 / P 0.952 | c=0.745 R 0.183 / tiny 0.087 / P 0.934 | c=0.691 R 0.276 / tiny 0.172 / P 0.913 | c=0.585 R 0.433 / tiny 0.313 / P 0.880 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S4_D | c=0.827 R 0.043 / tiny 0.008 / P 0.959 | c=0.804 R 0.082 / tiny 0.023 / P 0.949 | c=0.758 R 0.170 / tiny 0.084 / P 0.932 | c=0.695 R 0.287 / tiny 0.175 / P 0.919 | c=0.578 R 0.454 / tiny 0.336 / P 0.884 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S4b_D2 | c=0.831 R 0.044 / tiny 0.011 / P 0.956 | c=0.808 R 0.083 / tiny 0.033 / P 0.946 | c=0.765 R 0.158 / tiny 0.074 / P 0.927 | c=0.703 R 0.274 / tiny 0.167 / P 0.911 | c=0.572 R 0.444 / tiny 0.325 / P 0.881 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S4c_D3 | c=0.788 R 0.055 / tiny 0.006 / P 0.962 | c=0.774 R 0.074 / tiny 0.013 / P 0.948 | c=0.723 R 0.164 / tiny 0.052 / P 0.934 | c=0.679 R 0.239 / tiny 0.113 / P 0.909 | c=0.564 R 0.419 / tiny 0.276 / P 0.879 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S5_ensemble_S1_A | n/a (FP floor 3.4%) | n/a (FP floor 3.4%) | c=0.982 R 0.237 / tiny 0.132 / P 0.941 | c=0.962 R 0.344 / tiny 0.225 / P 0.921 | c=0.904 R 0.533 / tiny 0.409 / P 0.892 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S5_ensemble_v4a_s2 | n/a (FP floor 3.3%) | n/a (FP floor 3.3%) | c=0.982 R 0.232 / tiny 0.131 / P 0.940 | c=0.962 R 0.343 / tiny 0.226 / P 0.920 | c=0.905 R 0.533 / tiny 0.407 / P 0.891 | 7,335 imgs / 1,158 neg; tiny GT 2277 |

Sources:
- `reports/v5b/val/eval_S0_expE_sdisval.raw.parquet`
- `reports/v5b/val/eval_S0b_expB_sdisval.raw.parquet`
- `reports/v5b/val/eval_S1_A_sdisval.raw.parquet`
- `reports/v5b/val/eval_S1_A_s1_sdisval.raw.parquet`
- `reports/v5b/val/eval_S1_A_s2_sdisval.raw.parquet`
- `reports/v5b/val/eval_S2_B_sdisval.raw.parquet`
- `reports/v5b/val/eval_S2_B_s1_sdisval.raw.parquet`
- `reports/v5b/val/eval_S2_B_s2_sdisval.raw.parquet`
- `reports/v5b/val/eval_S3_C_sdisval.raw.parquet`
- `reports/v5b/val/eval_S4_D_sdisval.raw.parquet`
- `reports/v5b/val/eval_S4b_D2_sdisval.raw.parquet`
- `reports/v5b/val/eval_S4c_D3_sdisval.raw.parquet`
- `reports/v5b/val/eval_S5_ensemble_S1_A_sdisval.raw.parquet`
- `reports/v5b/val/eval_S5_ensemble_v4a_s2_sdisval.raw.parquet`
