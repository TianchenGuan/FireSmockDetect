_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T06. Seed-0 arms incl. D_curr and expE_iter2, pyro-sdis VAL: recall / tiny at matched neg-image FP

Cell = recall / tiny-recall (conf at which this model sits at the FP target), interpolated on each model's own curve with the frozen matching code (IoU>=0.4). Denominators: n_images=7,335 (positives 6,177, negatives 1,158); 1% FP = ~12 negative images. Seed 0 only (3-seed means in T01). D_curr is a single run.

| neg-image FP target | A_ctrl_E | E_ET | C_EGT | D_curr | B_Gonly | expE_iter2 |
|---|---|---|---|---|---|---|
| 1% | 0.053 / 0.015 (c=0.801) | 0.046 / 0.006 (c=0.819) | 0.041 / 0.002 (c=0.812) | 0.082 / 0.024 (c=0.798) | 0.021 / 0.001 (c=0.625) | 0.059 / 0.013 (c=0.799) |
| 2% | 0.104 / 0.040 (c=0.768) | 0.073 / 0.013 (c=0.800) | 0.090 / 0.019 (c=0.781) | 0.107 / 0.038 (c=0.785) | 0.032 / 0.003 (c=0.579) | 0.092 / 0.026 (c=0.775) |
| 5% | 0.204 / 0.112 (c=0.717) | 0.161 / 0.065 (c=0.746) | 0.172 / 0.078 (c=0.735) | 0.161 / 0.075 (c=0.753) | 0.069 / 0.006 (c=0.423) | 0.190 / 0.097 (c=0.721) |
| 10% | 0.309 / 0.201 (c=0.661) | 0.247 / 0.132 (c=0.691) | 0.266 / 0.164 (c=0.684) | 0.300 / 0.192 (c=0.669) | 0.092 / 0.007 (c=0.338) | 0.309 / 0.201 (c=0.655) |
| 20% | 0.483 / 0.367 (c=0.547) | 0.415 / 0.294 (c=0.566) | 0.419 / 0.330 (c=0.587) | 0.468 / 0.368 (c=0.544) | 0.155 / 0.023 (c=0.189) | 0.480 / 0.364 (c=0.535) |

| column | system |
|---|---|
| A_ctrl_E | E only (existing-data control) |
| E_ET | E + T (target aerial/forest data) |
| C_EGT | E + G + T (G, T at 2x multiplicity) |
| D_curr | curriculum: G pretrain -> E+T (+15% G replay) |
| B_Gonly | G only (generic smoke, source-only selection) |
| expE_iter2 | iteration-2 expE YOLO11m (E only, deployed candidate) |

Sources:
- `reports/v4/matched_fp_sdisval.json`
