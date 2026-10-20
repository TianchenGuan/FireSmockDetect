_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T03. Transfer matrix, AuSmoke FOREST VAL (new region, positives-only): recall at fixed confidence, 3 seeds

forest_val has NO negative images, so matched-FP is undefined; recall is reported at fixed YOLO confidences (comparable across YOLO arms only — never against DINO scores). Denominators: n_images=1,471 (positives 1,471, negatives 0); 4 units / ~3 independent events (Axis pair = same fixed camera). False-alarm behaviour on this domain is unmeasured.

| arm | seeds | R@conf0.10 | R@conf0.20 | R@conf0.30 | R@conf0.50 | tiny R@0.20 | tiny R@0.30 |
|---|---|---|---|---|---|---|---|
| A_ctrl_E | 3 | 0.332 ± 0.018 (n=3) | 0.246 ± 0.015 (n=3) | 0.169 ± 0.018 (n=3) | 0.056 ± 0.008 (n=3) | 0.095 ± 0.009 (n=3) | 0.060 ± 0.014 (n=3) |
| E_ET | 3 | 0.343 ± 0.018 (n=3) | 0.277 ± 0.014 (n=3) | 0.221 ± 0.015 (n=3) | 0.111 ± 0.010 (n=3) | 0.243 ± 0.011 (n=3) | 0.217 ± 0.008 (n=3) |
| C_EGT | 3 | 0.378 ± 0.035 (n=3) | 0.279 ± 0.038 (n=3) | 0.204 ± 0.035 (n=3) | 0.086 ± 0.015 (n=3) | 0.183 ± 0.016 (n=3) | 0.151 ± 0.010 (n=3) |
| D_curr | 3 | 0.334 ± 0.023 (n=3) | 0.245 ± 0.016 (n=3) | 0.177 ± 0.010 (n=3) | 0.061 ± 0.001 (n=3) | 0.196 ± 0.038 (n=3) | 0.155 ± 0.034 (n=3) |
| B_Gonly | 1 | 0.007 (n=1) | 0.001 (n=1) | 0.000 (n=1) | 0.000 (n=1) | 0.000 (n=1) | 0.000 (n=1) |

Sources:
- `reports/v4/seed_summary.json`
- `reports/v4/matched_fp_forestval.json (denominators)`
