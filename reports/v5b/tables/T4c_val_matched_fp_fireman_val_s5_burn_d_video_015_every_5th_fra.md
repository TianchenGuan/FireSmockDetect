_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

### T4c. VAL matched-FP: FireMan val_s5 (burn D video_015, every 5th frame; 1,706 imgs / 231 author negatives; one video = one unit; author boxes are large region boxes)

Recall / tiny-recall (<0.1% area GT) / precision at matched negative-image FP, interpolated on each system's own confidence sweep (src.eval.eval_detection.sweep_report, IoU>=0.4, grid 0.01..0.99 step 0.01; +0.991..0.999 for DINO-family tokens, as scripts/v4/compare_matched_fp.py --top-fine). 'tiny n/a' = the set has no tiny GT boxes (FireMan/Boreal author boxes are large plume regions). The eval JSONs record `imgsz: 1152`, which is eval_detection's CLI default when fed --preds-jsonl; the pod inference ran at IMGSZ=1280 (cluster/v4/yolo_eval_entry.sh default). FireMan VAL recall saturates for v1-era models -> it is NOT a faint-smoke benchmark (STATUS 10:57 UTC).

| system | 1% FP | 2% FP | 5% FP | 10% FP | 20% FP | denominators |
|---|---|---|---|---|---|---|
| S0_expE | c=0.256 R 0.963 / tiny n/a / P 0.523 | c=0.208 R 0.971 / tiny n/a / P 0.503 | c=0.177 R 0.981 / tiny n/a / P 0.489 | c=0.070 R 0.993 / tiny n/a / P 0.439 | n/a | 1,706 imgs / 231 neg; tiny GT 0 |
| S0b_expB | c=0.287 R 0.994 / tiny n/a / P 0.360 | c=0.214 R 0.996 / tiny n/a / P 0.346 | c=0.144 R 0.997 / tiny n/a / P 0.334 | c=0.063 R 1.000 / tiny n/a / P 0.315 | c=0.022 R 1.000 / tiny n/a / P 0.295 | 1,706 imgs / 231 neg; tiny GT 0 |
| S1_A | c=0.197 R 0.971 / tiny n/a / P 0.489 | c=0.147 R 0.984 / tiny n/a / P 0.474 | c=0.091 R 0.997 / tiny n/a / P 0.450 | c=0.030 R 1.000 / tiny n/a / P 0.405 | n/a | 1,706 imgs / 231 neg; tiny GT 0 |
| S1_A_s1 | c=0.277 R 0.960 / tiny n/a / P 0.527 | c=0.262 R 0.965 / tiny n/a / P 0.523 | c=0.236 R 0.972 / tiny n/a / P 0.510 | c=0.165 R 0.983 / tiny n/a / P 0.475 | c=0.040 R 1.000 / tiny n/a / P 0.389 | 1,706 imgs / 231 neg; tiny GT 0 |
| S1_A_s2 | c=0.427 R 0.923 / tiny n/a / P 0.750 | c=0.404 R 0.929 / tiny n/a / P 0.728 | c=0.282 R 0.960 / tiny n/a / P 0.638 | c=0.179 R 0.977 / tiny n/a / P 0.582 | c=0.043 R 0.996 / tiny n/a / P 0.495 | 1,706 imgs / 231 neg; tiny GT 0 |
| S2_B | c=0.178 R 0.992 / tiny n/a / P 0.546 | c=0.168 R 0.993 / tiny n/a / P 0.544 | c=0.132 R 0.995 / tiny n/a / P 0.538 | c=0.076 R 0.997 / tiny n/a / P 0.521 | c=0.026 R 0.997 / tiny n/a / P 0.485 | 1,706 imgs / 231 neg; tiny GT 0 |
| S2_B_s1 | c=0.053 R 0.983 / tiny n/a / P 0.590 | c=0.029 R 0.988 / tiny n/a / P 0.575 | n/a | n/a | n/a | 1,706 imgs / 231 neg; tiny GT 0 |
| S2_B_s2 | c=0.437 R 0.968 / tiny n/a / P 0.704 | c=0.374 R 0.980 / tiny n/a / P 0.685 | c=0.194 R 0.993 / tiny n/a / P 0.636 | c=0.044 R 0.997 / tiny n/a / P 0.576 | n/a | 1,706 imgs / 231 neg; tiny GT 0 |
| S3_C | c=0.197 R 0.994 / tiny n/a / P 0.588 | c=0.182 R 0.995 / tiny n/a / P 0.585 | c=0.072 R 0.998 / tiny n/a / P 0.563 | c=0.045 R 0.998 / tiny n/a / P 0.549 | c=0.021 R 1.000 / tiny n/a / P 0.524 | 1,706 imgs / 231 neg; tiny GT 0 |
| S4_D | c=0.193 R 0.992 / tiny n/a / P 0.606 | c=0.114 R 0.995 / tiny n/a / P 0.593 | c=0.082 R 0.996 / tiny n/a / P 0.583 | c=0.036 R 0.998 / tiny n/a / P 0.558 | n/a | 1,706 imgs / 231 neg; tiny GT 0 |
| S4b_D2 | c=0.227 R 0.970 / tiny n/a / P 0.630 | c=0.114 R 0.988 / tiny n/a / P 0.600 | c=0.051 R 0.994 / tiny n/a / P 0.572 | c=0.024 R 0.997 / tiny n/a / P 0.548 | n/a | 1,706 imgs / 231 neg; tiny GT 0 |
| S4c_D3 | c=0.319 R 0.990 / tiny n/a / P 0.621 | c=0.284 R 0.992 / tiny n/a / P 0.611 | c=0.262 R 0.993 / tiny n/a / P 0.606 | c=0.199 R 0.994 / tiny n/a / P 0.587 | c=0.037 R 0.999 / tiny n/a / P 0.502 | 1,706 imgs / 231 neg; tiny GT 0 |
| S5_ensemble_S1_A | PENDING | PENDING | PENDING | PENDING | PENDING | - |
| S5_ensemble_v4a_s2 | PENDING | PENDING | PENDING | PENDING | PENDING | - |

Sources:
- `reports/v5b/val/eval_S0_expE_firemanval.raw.parquet`
- `reports/v5b/val/eval_S0b_expB_firemanval.raw.parquet`
- `reports/v5b/val/eval_S1_A_firemanval.raw.parquet`
- `reports/v5b/val/eval_S1_A_s1_firemanval.raw.parquet`
- `reports/v5b/val/eval_S1_A_s2_firemanval.raw.parquet`
- `reports/v5b/val/eval_S2_B_firemanval.raw.parquet`
- `reports/v5b/val/eval_S2_B_s1_firemanval.raw.parquet`
- `reports/v5b/val/eval_S2_B_s2_firemanval.raw.parquet`
- `reports/v5b/val/eval_S3_C_firemanval.raw.parquet`
- `reports/v5b/val/eval_S4_D_firemanval.raw.parquet`
- `reports/v5b/val/eval_S4b_D2_firemanval.raw.parquet`
- `reports/v5b/val/eval_S4c_D3_firemanval.raw.parquet`
