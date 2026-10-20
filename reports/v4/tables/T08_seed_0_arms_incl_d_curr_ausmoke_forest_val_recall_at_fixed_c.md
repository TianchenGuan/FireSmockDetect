_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T08. Seed-0 arms incl. D_curr, AuSmoke FOREST VAL: recall at fixed confs and under both threshold regimes

forest_val is positives-only (n_images=1,471, negatives 0), so matched-FP is undefined. (a) recall / tiny-recall at fixed confidences (comparable only within one score family). (b) the two pre-registered threshold regimes: SOURCE-ONLY = each model's own sdis-val confidence at 5%/10% neg-FP applied unchanged; TARGET-ADAPTED = its boreal-val confidence at 1%/5% neg-FP (217 boreal negatives). Regime rows are computed at the exact confidence from the saved predictions with the frozen matching code where the .raw.parquet exists. 4 units / ~3 independent events.

**(a) fixed confidence**

| threshold | A_ctrl_E | E_ET | C_EGT | D_curr | B_Gonly |
|---|---|---|---|---|---|
| conf=0.10 | 0.357 / 0.140 | 0.328 / 0.270 | 0.398 / 0.222 | 0.315 / 0.226 | 0.007 / 0.000 |
| conf=0.20 | 0.268 / 0.103 | 0.266 / 0.237 | 0.306 / 0.163 | 0.233 / 0.155 | 0.001 / 0.000 |
| conf=0.30 | 0.194 / 0.069 | 0.218 / 0.218 | 0.233 / 0.138 | 0.171 / 0.119 | 0.000 / 0.000 |
| conf=0.50 | 0.066 / 0.002 | 0.124 / 0.140 | 0.105 / 0.077 | 0.062 / 0.017 | 0.000 / 0.000 |

**(b) val-selected thresholds (pre-registered regimes)**

| regime | A_ctrl_E | E_ET | C_EGT | D_curr | B_Gonly |
|---|---|---|---|---|---|
| SOURCE-ONLY: sdis-val conf @5% negFP | 0.013 / 0.000 (c=0.717) | 0.009 / 0.000 (c=0.746) | 0.008 / 0.000 (c=0.735) | 0.005 / 0.000 (c=0.753) | 0.000 / 0.000 (c=0.423) |
| SOURCE-ONLY: sdis-val conf @10% negFP | 0.020 / 0.000 (c=0.661) | 0.016 / 0.000 (c=0.691) | 0.015 / 0.000 (c=0.684) | 0.014 / 0.000 (c=0.669) | 0.000 / 0.000 (c=0.338) |
| TARGET-ADAPTED: boreal-val conf @1% negFP | 0.122 / 0.038 (c=0.408) | 0.354 / 0.291 (c=0.068) | 0.415 / 0.239 (c=0.084) | 0.288 / 0.205 (c=0.128) | 0.000 / 0.000 (c=0.458) |
| TARGET-ADAPTED: boreal-val conf @5% negFP | 0.226 / 0.088 (c=0.257) | n/a (no val threshold) | 0.458 / 0.268 (c=0.052) | 0.354 / 0.257 (c=0.059) | 0.000 / 0.000 (c=0.356) |

Sources:
- `reports/v4/matched_fp_forestval.json`
- `reports/v4/matched_fp_sdisval.json (SOURCE-ONLY confs)`
- `reports/v4/matched_fp_borealval.json (TARGET-ADAPTED confs)`
- `reports/v4/eval_<ckpt>_forestval.raw.parquet (exact recall at conf)`
