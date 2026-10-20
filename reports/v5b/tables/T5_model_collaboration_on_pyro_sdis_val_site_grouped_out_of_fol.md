_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

### T5. Model collaboration on pyro-sdis VAL: site-grouped out-of-fold (OOF) mean recall over FP 1/2/5/10/20% (selection statistic = macro over the two held-out sites)

Fusions are fit on one VAL site and evaluated on the other (calibration maps + WBF params); single models need no fitting. Chosen ACCURATE recipe = A+B4+B6+B1 WBF (macro 0.3535 vs best single B6 0.3258). Learned fusion (logreg) beats WBF by only +0.46 pt (< 1 pt bar) -> not adopted. Pooled-OOF numbers are pessimistic because calibration does not transfer between sites (ENSEMBLE_DEV_V5B.md T8). Latency (batch-1 L40S fp32 @1280, from reports): components A 11.2 ms, B4 157.5 ms, B6 351.6 ms, B1 156.6 ms; fusion 0.381 ms CPU; PARALLEL 352.0 ms; serial 677.3 ms.

| system | OOF cabanelle | OOF marguerite | macro | whole-VAL / pooled | whole-VAL R @1/2/5/10/20% | WBF/union params (all-VAL fit) |
|---|---|---|---|---|---|---|
| single A (yolo_A) | 0.207 | 0.436 | 0.3213 | 0.230 | 0.056 / 0.100 / 0.200 / 0.302 / 0.493 | - |
| single B4 (dino_B4) | 0.180 | 0.397 | 0.2885 | 0.242 | 0.083 / 0.123 / 0.208 / 0.309 / 0.486 | - |
| single B6 (dino_B6) | 0.213 | 0.438 | 0.3258 | 0.260 | 0.103 / 0.133 / 0.228 / 0.340 / 0.495 | - |
| single RF (rfdetr_L) | 0.244 | 0.383 | 0.3136 | 0.232 | 0.053 / 0.102 / 0.201 / 0.314 / 0.486 | - |
| single expE (yolo_expE) | 0.208 | 0.435 | 0.3215 | 0.226 | 0.058 / 0.092 / 0.188 / 0.308 / 0.483 | - |
| single B1 (dino_B1) | 0.192 | 0.404 | 0.2980 | 0.220 | 0.065 / 0.095 / 0.176 / 0.299 / 0.465 | - |
| OR/union+NMS A+B4 | 0.209 | 0.434 | 0.3216 | 0.161 | - | iou0.7 |
| WBF A+B4 | 0.204 | 0.453 | 0.3285 | 0.142 | - | avg,iou0.4,skip0.8 |
| OR/union+NMS A+B4+B6 | 0.220 | 0.458 | 0.3386 | 0.172 | - | iou0.7 |
| WBF A+B4+B6 | 0.221 | 0.460 | 0.3401 | 0.173 | - | max,iou0.3,skip0.0 |
| OR/union+NMS A+B4+B6+RF | 0.224 | 0.443 | 0.3333 | 0.174 | - | iou0.7 |
| WBF A+B4+B6+RF | 0.225 | 0.447 | 0.3362 | 0.175 | - | max,iou0.3,skip0.0 |
| OR/union+NMS A+B4+B6+expE | 0.221 | 0.455 | 0.3380 | 0.170 | - | iou0.7 |
| WBF A+B4+B6+expE | 0.222 | 0.459 | 0.3407 | 0.118 | - | max,iou0.3,skip0.0 |
| OR/union+NMS A+B4+B6+B1 | 0.230 | 0.473 | 0.3513 | 0.172 | - | iou0.7 |
| WBF A+B4+B6+B1 | 0.230 | 0.477 | 0.3535 | 0.111 | - | max,iou0.3,skip0.0 |

**model-addition protocol (keep if >= +1.0 pt macro-OOF)**

| candidate | set tested | union macro | WBF macro | gain (pt) | kept |
|---|---|---|---|---|---|
| dino_B6 | A+B4+B6 | 0.3386 | 0.3401 | +1.16 | yes |
| rfdetr_L | A+B4+B6+RF | 0.3333 | 0.3362 | -0.39 | no |
| yolo_expE | A+B4+B6+expE | 0.3380 | 0.3407 | +0.06 | no |
| dino_B1 | A+B4+B6+B1 | 0.3513 | 0.3535 | +1.34 | yes |

**bootstrap (500 image resamples within the held-out site), difference in OOF mean recall (pt)**

| held-out site | comparison | point | 95% CI | P(gain<=0) |
|---|---|---|---|---|
| cabanelle | fusion_vs_rfdetr_L | -1.43 | [-4.21, +1.76] | 0.770 |
| cabanelle | fusion_vs_dino_B6 | +1.77 | [-0.90, +3.97] | 0.086 |
| cabanelle | fusion_vs_yolo_A | +2.33 | [-0.24, +4.69] | 0.042 |
| cabanelle | fusion_vs_lean(A+B4+B6 wbf) | +1.08 | [-1.12, +3.01] | 0.180 |
| cabanelle | lean(A+B4+B6 wbf)_vs_dino_B6 | +0.69 | [-0.91, +2.22] | 0.194 |
| marguerite | fusion_vs_dino_B6 | +3.86 | [+1.00, +8.34] | 0.004 |
| marguerite | fusion_vs_yolo_A | +4.12 | [-0.90, +8.48] | 0.064 |
| marguerite | fusion_vs_lean(A+B4+B6 wbf) | +1.73 | [-1.60, +7.16] | 0.136 |
| marguerite | lean(A+B4+B6 wbf)_vs_dino_B6 | +2.13 | [-3.65, +6.11] | 0.206 |

Sources:
- `reports/v5b/ensemble_dev_v5b.json`
- `reports/v5b/ENSEMBLE_DEV_V5B.md`
- `models/v5b/fusion/accurate_recipe.json`
