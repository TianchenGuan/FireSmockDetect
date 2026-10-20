_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T02. Transfer matrix, boreal AERIAL VAL (burn_heinola, 1 burn): recall at matched neg-image FP, 3 seeds

Cell = mean ± std (n seeds that reach the FP target; 'n/a' = no seed's curve reaches it — an FP ceiling: the model never exceeds that FP even at its lowest confidence, so recall is at least its max, see T07). Denominators: n_images=1,123 (positives 906, negatives 217); ONE independent burn; 1% neg-FP = ~2 negative images, so the 1% column has a resolution of ~0.5pt FP and a seed std of ~5pt is expected. Tiny recall is 0 for every arm (aerial plumes are never <0.1% area; boreal_val has 1 tiny + 5 small GT boxes) and is omitted.

| arm | seeds | R@1% | R@2% | R@5% | R@10% | R@20% |
|---|---|---|---|---|---|---|
| A_ctrl_E | 3 | 0.596 ± 0.033 (n=3) | 0.719 ± 0.032 (n=3) | 0.792 ± 0.007 (n=3) | 0.807 ± 0.005 (n=3) | 0.832 ± 0.004 (n=3) |
| E_ET | 3 | 0.905 ± 0.055 (n=3) | 0.898 ± 0.064 (n=2) | 0.846 (n=1) | 0.869 (n=1) | 0.928 (n=1) |
| C_EGT | 3 | 0.901 ± 0.051 (n=3) | 0.916 ± 0.034 (n=3) | 0.936 ± 0.000 (n=2) | 0.941 ± 0.001 (n=2) | n/a |
| D_curr | 3 | 0.896 ± 0.022 (n=3) | 0.906 ± 0.024 (n=3) | 0.915 ± 0.025 (n=3) | 0.922 ± 0.023 (n=3) | 0.935 ± 0.019 (n=3) |
| B_Gonly | 1 | 0.194 (n=1) | 0.215 (n=1) | 0.466 (n=1) | 0.664 (n=1) | 0.842 (n=1) |

Sources:
- `reports/v4/seed_summary.json`
- `reports/v4/matched_fp_borealval.json (denominators)`
