_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T01. Transfer matrix, pyro-sdis VAL (fixed-camera source domain): recall at matched neg-image FP, 3 seeds

Cell = mean ± std over seeds (n seeds in parentheses; n=1 arms are single runs). Matching IoU>=0.4. Denominators: n_images=7,335 (positives 6,177, negatives 1,158); 1% neg-FP = ~12 negative images. tiny = GT boxes <0.1% of image area.

| arm | recipe | seeds | R@1% | R@2% | R@5% | R@10% | R@20% | tiny R@5% | tiny R@10% |
|---|---|---|---|---|---|---|---|---|---|
| A_ctrl_E | E only (existing-data control) | 3 | 0.055 ± 0.001 (n=3) | 0.103 ± 0.004 (n=3) | 0.202 ± 0.001 (n=3) | 0.305 ± 0.003 (n=3) | 0.492 ± 0.007 (n=3) | 0.111 ± 0.001 (n=3) | 0.196 ± 0.004 (n=3) |
| E_ET | E + T (target aerial/forest data) | 3 | 0.044 ± 0.002 (n=3) | 0.080 ± 0.009 (n=3) | 0.180 ± 0.014 (n=3) | 0.277 ± 0.021 (n=3) | 0.444 ± 0.021 (n=3) | 0.076 ± 0.010 (n=3) | 0.155 ± 0.017 (n=3) |
| C_EGT | E + G + T (G, T at 2x multiplicity) | 3 | 0.043 ± 0.002 (n=3) | 0.082 ± 0.006 (n=3) | 0.165 ± 0.006 (n=3) | 0.263 ± 0.002 (n=3) | 0.422 ± 0.002 (n=3) | 0.071 ± 0.009 (n=3) | 0.161 ± 0.006 (n=3) |
| D_curr | curriculum: G pretrain -> E+T (+15% G replay) | 3 | 0.059 ± 0.017 (n=3) | 0.091 ± 0.012 (n=3) | 0.196 ± 0.029 (n=3) | 0.321 ± 0.018 (n=3) | 0.484 ± 0.019 (n=3) | 0.095 ± 0.022 (n=3) | 0.200 ± 0.021 (n=3) |
| B_Gonly | G only (generic smoke, source-only selection) | 1 | 0.021 (n=1) | 0.032 (n=1) | 0.069 (n=1) | 0.092 (n=1) | 0.155 (n=1) | 0.006 (n=1) | 0.007 (n=1) |

Sources:
- `reports/v4/seed_summary.json`
- `reports/v4/matched_fp_sdisval.json (denominators)`
