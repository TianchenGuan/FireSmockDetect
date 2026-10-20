_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T07. Seed-0 arms incl. D_curr, boreal AERIAL VAL (burn_heinola): recall at matched neg-image FP

Cell = recall / tiny-recall (conf at which this model sits at the FP target), interpolated on each model's own curve with the frozen matching code (IoU>=0.4). Denominators: n_images=1,123 (positives 906, negatives 217); 1% FP = ~2 negative images. ONE independent burn. Tiny recall is structurally 0 on this set (1 tiny + 5 small GT boxes). 'FP ceiling' = the model never exceeds that neg-FP even at its lowest confidence.

| neg-image FP target | A_ctrl_E | E_ET | C_EGT | D_curr | B_Gonly |
|---|---|---|---|---|---|
| 1% | 0.640 / 0.000 (c=0.408) | 0.931 / 0.000 (c=0.068) | 0.934 / 0.000 (c=0.084) | 0.909 / 0.000 (c=0.128) | 0.194 / 0.000 (c=0.458) |
| 2% | 0.674 / 0.000 (c=0.377) | FP ceiling 1.4% -> R>=0.954 | 0.935 / 0.000 (c=0.073) | 0.927 / 0.000 (c=0.076) | 0.215 / 0.000 (c=0.443) |
| 5% | 0.783 / 0.000 (c=0.257) | FP ceiling 1.4% -> R>=0.954 | 0.936 / 0.000 (c=0.052) | 0.939 / 0.000 (c=0.059) | 0.466 / 0.000 (c=0.356) |
| 10% | 0.801 / 0.000 (c=0.215) | FP ceiling 1.4% -> R>=0.954 | 0.942 / 0.000 (c=0.025) | 0.945 / 0.000 (c=0.046) | 0.664 / 0.000 (c=0.291) |
| 20% | 0.828 / 0.000 (c=0.165) | FP ceiling 1.4% -> R>=0.954 | FP ceiling 11.5% -> R>=0.943 | 0.957 / 0.000 (c=0.029) | 0.842 / 0.000 (c=0.198) |

| column | system |
|---|---|
| A_ctrl_E | E only (existing-data control) |
| E_ET | E + T (target aerial/forest data) |
| C_EGT | E + G + T (G, T at 2x multiplicity) |
| D_curr | curriculum: G pretrain -> E+T (+15% G replay) |
| B_Gonly | G only (generic smoke, source-only selection) |

Sources:
- `reports/v4/matched_fp_borealval.json`
