_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T04. Pre-registered selection quantities (computed on VAL only)

S2 rule: the v4 data arm with the higher 3-seed mean boreal-val R@1% neg-FP; tie <1pt -> E_ET. Median-seed rule: the seed whose sdis-val R@10% is the median of its arm is the one carried to TEST. Denominators as in T01/T02 (boreal 1% = ~2 of 217 negatives).

| quantity | value |
|---|---|
| C_EGT.boreal_R@1%.mean | mean 0.9006, std 0.0512, min 0.8283, max 0.9399, n=3 |
| E_ET.boreal_R@1%.mean | mean 0.9047, std 0.0550, min 0.8281, max 0.9550, n=3 |
| S2_pick | E_ET |
| S2_rule | higher 3-seed mean boreal R@1%; tie <1pt -> E_ET (diff=-0.0041) |
| A_ctrl_E.median_seed(sdis R@10%) | ckpt `v4a_s2` (sdis R@10% = 0.3040; n_seeds=3) |
| E_ET.median_seed(sdis R@10%) | ckpt `v4e_s1` (sdis R@10% = 0.2913; n_seeds=3) |
| C_EGT.median_seed(sdis R@10%) | ckpt `v4c_s1` (sdis R@10% = 0.2620; n_seeds=3) |
| D_curr.median_seed(sdis R@10%) | ckpt `v4dada_s1` (sdis R@10% = 0.3180; n_seeds=3) |

Sources:
- `reports/v4/seed_summary.json (selection block)`
- `reports/v4/PREREGISTRATION_V4.md (rules)`
