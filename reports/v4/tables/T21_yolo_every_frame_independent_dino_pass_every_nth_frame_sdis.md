_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T21. YOLO every frame + independent DINO pass every Nth frame (sdis VAL sequences)

Episode detection over the 469 sdis_val positive episodes at thresholds matched on frame FPR (1,158 negative frames); 'vs YOLO@same FP' = combo minus YOLO-alone re-thresholded to the combo's realised false-alert frames.

**DINO source b1_v3** (`reports/v3/eval_B1_val.raw.parquet`). Batch-1 L40S: YOLO11m expE 11.24 ms, DINOv3-L 156.43 ms -> one DINO pass = 13.92x one YOLO pass; N for <=20% added compute = 70. Val sequences 266, median length 15.5 frames; fraction of sequences with ZERO DINO pass: N=5 0.026, N=10 0.308, N=20 0.587, N=35 0.771, N=70 0.913, N=140 0.985

| frame-FPR target | YOLO alone: episode detection | YOLO + DINO every 20th frame | YOLO + DINO every 70th frame (<=20% compute) |
|---|---|---|---|
| FPR 0.005 | 0.1322 (469 ep, 5 FP fr / 1158) | 0.1478 (+0.0156 vs YOLO@same FP; +70% compute; beats) | 0.1375 (+0.0053 vs YOLO@same FP; +20% compute; beats) |
| FPR 0.01 | 0.1748 (469 ep, 11 FP fr / 1158) | 0.1978 (+0.0165 vs YOLO@same FP; +70% compute; beats) | 0.1828 (+0.0080 vs YOLO@same FP; +20% compute; beats) |
| FPR 0.02 | 0.2814 (469 ep, 23 FP fr / 1158) | 0.3045 (+0.0209 vs YOLO@same FP; +70% compute; beats) | 0.2898 (+0.0083 vs YOLO@same FP; +20% compute; beats) |
| FPR 0.05 | 0.4606 (469 ep, 57 FP fr / 1158) | 0.4760 (+0.0091 vs YOLO@same FP; +70% compute; beats) | 0.4651 (+0.0045 vs YOLO@same FP; +20% compute; beats) |
| FPR 0.1 | 0.6141 (469 ep, 115 FP fr / 1158) | 0.6257 (+0.0116 vs YOLO@same FP; +70% compute; beats) | 0.6175 (+0.0035 vs YOLO@same FP; +20% compute; beats) |

**DINO source b1_v4frozenL** (`reports/v4/eval_v4B1_frozen_L_sdis_val.raw.parquet`). Batch-1 L40S: YOLO11m expE 11.24 ms, DINOv3-L 156.43 ms -> one DINO pass = 13.92x one YOLO pass; N for <=20% added compute = 70. Val sequences 266, median length 15.5 frames; fraction of sequences with ZERO DINO pass: N=5 0.026, N=10 0.308, N=20 0.587, N=35 0.771, N=70 0.913, N=140 0.985

| frame-FPR target | YOLO alone: episode detection | YOLO + DINO every 20th frame | YOLO + DINO every 70th frame (<=20% compute) |
|---|---|---|---|
| FPR 0.005 | 0.1322 (469 ep, 5 FP fr / 1158) | 0.1447 (+0.0125 vs YOLO@same FP; +70% compute; beats) | 0.1362 (+0.0041 vs YOLO@same FP; +20% compute; beats) |
| FPR 0.01 | 0.1748 (469 ep, 11 FP fr / 1158) | 0.1902 (+0.0090 vs YOLO@same FP; +70% compute; beats) | 0.1801 (+0.0053 vs YOLO@same FP; +20% compute; beats) |
| FPR 0.02 | 0.2814 (469 ep, 23 FP fr / 1158) | 0.3004 (+0.0168 vs YOLO@same FP; +70% compute; beats) | 0.2884 (+0.0070 vs YOLO@same FP; +20% compute; beats) |
| FPR 0.05 | 0.4606 (469 ep, 57 FP fr / 1158) | 0.4730 (+0.0061 vs YOLO@same FP; +70% compute; beats) | 0.4645 (+0.0040 vs YOLO@same FP; +20% compute; beats) |
| FPR 0.1 | 0.6141 (469 ep, 115 FP fr / 1158) | 0.6232 (+0.0092 vs YOLO@same FP; +70% compute; beats) | 0.6167 (+0.0026 vs YOLO@same FP; +20% compute; beats) |

Sources:
- `reports/v4/trackd_v4_eval.json (D_yolo_plus_dino_sweep, D_dino_sources)`
