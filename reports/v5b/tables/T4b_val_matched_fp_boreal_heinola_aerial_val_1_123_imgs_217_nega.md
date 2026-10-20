_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

### T4b. VAL matched-FP: Boreal heinola aerial VAL (1,123 imgs / 217 negatives; 1% FP = ~2 negatives -> 1-2% rows are weakly supported)

Recall / tiny-recall (<0.1% area GT) / precision at matched negative-image FP, interpolated on each system's own confidence sweep (src.eval.eval_detection.sweep_report, IoU>=0.4, grid 0.01..0.99 step 0.01; +0.991..0.999 for DINO-family tokens, as scripts/v4/compare_matched_fp.py --top-fine). 'tiny n/a' = the set has no tiny GT boxes (FireMan/Boreal author boxes are large plume regions). The eval JSONs record `imgsz: 1152`, which is eval_detection's CLI default when fed --preds-jsonl; the pod inference ran at IMGSZ=1280 (cluster/v4/yolo_eval_entry.sh default). FireMan VAL recall saturates for v1-era models -> it is NOT a faint-smoke benchmark (STATUS 10:57 UTC).

| system | 1% FP | 2% FP | 5% FP | 10% FP | 20% FP | denominators |
|---|---|---|---|---|---|---|
| S0_expE | c=0.498 R 0.553 / tiny 0.000 / P 0.901 | c=0.356 R 0.698 / tiny 0.000 / P 0.814 | c=0.262 R 0.751 / tiny 0.000 / P 0.743 | c=0.223 R 0.779 / tiny 0.000 / P 0.711 | c=0.145 R 0.810 / tiny 0.000 / P 0.618 | 1,123 imgs / 217 neg; tiny GT 1 |
| S0b_expB | c=0.498 R 0.663 / tiny 0.000 / P 0.925 | c=0.436 R 0.709 / tiny 0.000 / P 0.915 | c=0.348 R 0.762 / tiny 0.000 / P 0.864 | c=0.333 R 0.771 / tiny 0.000 / P 0.848 | c=0.312 R 0.782 / tiny 0.000 / P 0.813 | 1,123 imgs / 217 neg; tiny GT 1 |
| S1_A | c=0.498 R 0.562 / tiny 0.000 / P 0.901 | c=0.346 R 0.743 / tiny 0.000 / P 0.786 | c=0.255 R 0.800 / tiny 0.000 / P 0.717 | c=0.226 R 0.814 / tiny 0.000 / P 0.684 | c=0.179 R 0.837 / tiny 0.000 / P 0.639 | 1,123 imgs / 217 neg; tiny GT 1 |
| S1_A_s1 | c=0.439 R 0.582 / tiny 0.000 / P 0.854 | c=0.416 R 0.598 / tiny 0.000 / P 0.835 | c=0.382 R 0.626 / tiny 0.000 / P 0.801 | c=0.319 R 0.673 / tiny 0.000 / P 0.760 | c=0.259 R 0.724 / tiny 0.000 / P 0.709 | 1,123 imgs / 217 neg; tiny GT 1 |
| S1_A_s2 | c=0.438 R 0.682 / tiny 0.000 / P 0.855 | c=0.387 R 0.717 / tiny 0.000 / P 0.819 | c=0.331 R 0.748 / tiny 0.000 / P 0.781 | c=0.308 R 0.757 / tiny 0.000 / P 0.760 | c=0.259 R 0.783 / tiny 0.000 / P 0.711 | 1,123 imgs / 217 neg; tiny GT 1 |
| S2_B | c=0.308 R 0.852 / tiny 0.000 / P 0.866 | c=0.157 R 0.901 / tiny 0.000 / P 0.784 | c=0.021 R 0.955 / tiny 0.000 / P 0.583 | n/a | n/a | 1,123 imgs / 217 neg; tiny GT 1 |
| S2_B_s1 | c=0.408 R 0.828 / tiny 0.000 / P 0.917 | c=0.117 R 0.917 / tiny 0.000 / P 0.803 | c=0.031 R 0.937 / tiny 0.000 / P 0.668 | n/a | n/a | 1,123 imgs / 217 neg; tiny GT 1 |
| S2_B_s2 | c=0.218 R 0.850 / tiny 0.000 / P 0.896 | c=0.178 R 0.863 / tiny 0.000 / P 0.873 | c=0.049 R 0.925 / tiny 0.000 / P 0.711 | c=0.039 R 0.930 / tiny 0.000 / P 0.677 | c=0.024 R 0.939 / tiny 0.000 / P 0.618 | 1,123 imgs / 217 neg; tiny GT 1 |
| S3_C | c=0.318 R 0.861 / tiny 0.000 / P 0.910 | c=0.197 R 0.891 / tiny 0.000 / P 0.859 | c=0.028 R 0.946 / tiny 0.000 / P 0.626 | c=0.021 R 0.948 / tiny 0.000 / P 0.595 | n/a | 1,123 imgs / 217 neg; tiny GT 1 |
| S4_D | c=0.308 R 0.822 / tiny 0.000 / P 0.909 | c=0.146 R 0.868 / tiny 0.000 / P 0.859 | n/a | n/a | n/a | 1,123 imgs / 217 neg; tiny GT 1 |
| S4b_D2 | c=0.328 R 0.881 / tiny 0.000 / P 0.944 | c=0.117 R 0.943 / tiny 0.000 / P 0.809 | n/a | n/a | n/a | 1,123 imgs / 217 neg; tiny GT 1 |
| S4c_D3 | c=0.388 R 0.821 / tiny 0.000 / P 0.928 | c=0.236 R 0.865 / tiny 0.000 / P 0.895 | n/a | n/a | n/a | 1,123 imgs / 217 neg; tiny GT 1 |
| S5_ensemble_S1_A | PENDING | PENDING | PENDING | PENDING | PENDING | - |
| S5_ensemble_v4a_s2 | PENDING | PENDING | PENDING | PENDING | PENDING | - |

Sources:
- `reports/v5b/val/eval_S0_expE_borealval.raw.parquet`
- `reports/v5b/val/eval_S0b_expB_borealval.raw.parquet`
- `reports/v5b/val/eval_S1_A_borealval.raw.parquet`
- `reports/v5b/val/eval_S1_A_s1_borealval.raw.parquet`
- `reports/v5b/val/eval_S1_A_s2_borealval.raw.parquet`
- `reports/v5b/val/eval_S2_B_borealval.raw.parquet`
- `reports/v5b/val/eval_S2_B_s1_borealval.raw.parquet`
- `reports/v5b/val/eval_S2_B_s2_borealval.raw.parquet`
- `reports/v5b/val/eval_S3_C_borealval.raw.parquet`
- `reports/v5b/val/eval_S4_D_borealval.raw.parquet`
- `reports/v5b/val/eval_S4b_D2_borealval.raw.parquet`
- `reports/v5b/val/eval_S4c_D3_borealval.raw.parquet`
