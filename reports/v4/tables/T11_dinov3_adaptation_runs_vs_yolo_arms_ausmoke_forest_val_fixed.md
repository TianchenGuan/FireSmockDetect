_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T11. DINOv3 adaptation runs vs YOLO arms, AuSmoke FOREST VAL: fixed confs and both threshold regimes

forest_val is positives-only (n_images=1,471, negatives 0), so matched-FP is undefined. (a) recall / tiny-recall at fixed confidences (comparable only within one score family). (b) the two pre-registered threshold regimes: SOURCE-ONLY = each model's own sdis-val confidence at 5%/10% neg-FP applied unchanged; TARGET-ADAPTED = its boreal-val confidence at 1%/5% neg-FP (217 boreal negatives). Regime rows are computed at the exact confidence from the saved predictions with the frozen matching code where the .raw.parquet exists. Under SOURCE-ONLY thresholds the saturated (adapted) models transfer WORSE than frozen B1 even where their fixed-conf recall is higher: this is a calibration/threshold-transfer effect, not a detection-quality ranking.

**(a) fixed confidence**

| threshold | v4B1_frozen_L | v4B4_lora_L | v4B5_full_L | v4B6_lora_Hplus | YOLO_A_ctrlE | YOLO_E_ET | YOLO_C_EGT |
|---|---|---|---|---|---|---|---|
| conf=0.10 | 0.531 / 0.237 | 0.601 / 0.308 | 0.548 / 0.283 | 0.635 / 0.301 | 0.357 / 0.140 | 0.328 / 0.270 | 0.398 / 0.222 |
| conf=0.20 | 0.478 / 0.211 | 0.579 / 0.289 | 0.522 / 0.283 | 0.608 / 0.272 | 0.268 / 0.103 | 0.266 / 0.237 | 0.306 / 0.163 |
| conf=0.30 | 0.441 / 0.182 | 0.558 / 0.283 | 0.505 / 0.282 | 0.590 / 0.272 | 0.194 / 0.069 | 0.218 / 0.218 | 0.233 / 0.138 |
| conf=0.50 | 0.383 / 0.147 | 0.519 / 0.266 | 0.489 / 0.280 | 0.553 / 0.261 | 0.066 / 0.002 | 0.124 / 0.140 | 0.105 / 0.077 |

**(b) val-selected thresholds (pre-registered regimes)**

| regime | v4B1_frozen_L | v4B4_lora_L | v4B5_full_L | v4B6_lora_Hplus | YOLO_A_ctrlE | YOLO_E_ET | YOLO_C_EGT |
|---|---|---|---|---|---|---|---|
| SOURCE-ONLY: sdis-val conf @5% negFP | 0.098 / 0.000 (c=0.968) | 0.034 / 0.000 (c=0.993) | 0.060 / 0.002 (c=0.998) | 0.047 / 0.000 (c=0.994) | 0.013 / 0.000 (c=0.717) | 0.009 / 0.000 (c=0.746) | 0.008 / 0.000 (c=0.735) |
| SOURCE-ONLY: sdis-val conf @10% negFP | 0.147 / 0.006 (c=0.941) | 0.084 / 0.000 (c=0.988) | 0.094 / 0.015 (c=0.996) | 0.091 / 0.008 (c=0.990) | 0.020 / 0.000 (c=0.661) | 0.016 / 0.000 (c=0.691) | 0.015 / 0.000 (c=0.684) |
| TARGET-ADAPTED: boreal-val conf @1% negFP | 0.384 / 0.149 (c=0.498) | 0.399 / 0.190 (c=0.868) | 0.483 / 0.278 (c=0.568) | 0.418 / 0.228 (c=0.888) | 0.122 / 0.038 (c=0.408) | 0.354 / 0.291 (c=0.068) | 0.415 / 0.239 (c=0.084) |
| TARGET-ADAPTED: boreal-val conf @5% negFP | 0.565 / 0.264 (c=0.056) | 0.586 / 0.293 (c=0.146) | 0.587 / 0.301 (c=0.031) | 0.550 / 0.261 (c=0.511) | 0.226 / 0.088 (c=0.257) | n/a (no val threshold) | 0.458 / 0.268 (c=0.052) |

Sources:
- `reports/v4/matched_fp_dino_forest_val.json`
- `reports/v4/matched_fp_dino_sdis_val.json`
- `reports/v4/matched_fp_dino_boreal_val.json`
- `reports/v4/eval_<run>_forest_val.raw.parquet`
