# Iteration 4 — all report tables

_generated 2026-09-15T03:55Z by `scripts/v4/make_report_tables.py`; every number is read from the artifact(s) listed under each table. Re-run the script after new eval JSONs land (e.g. reports/v4/test/)._

## Index

- [T01](#t01) Transfer matrix, pyro-sdis VAL (fixed-camera source domain): recall at matched neg-image FP, 3 seeds
- [T02](#t02) Transfer matrix, boreal AERIAL VAL (burn_heinola, 1 burn): recall at matched neg-image FP, 3 seeds
- [T03](#t03) Transfer matrix, AuSmoke FOREST VAL (new region, positives-only): recall at fixed confidence, 3 seeds
- [T04](#t04) Pre-registered selection quantities (computed on VAL only)
- [T05](#t05) Per-seed detail (every checkpoint behind T01-T03) incl. the val-selected confidences used as thresholds
- [T06](#t06) Seed-0 arms incl. D_curr and expE_iter2, pyro-sdis VAL: recall / tiny at matched neg-image FP
- [T07](#t07) Seed-0 arms incl. D_curr, boreal AERIAL VAL (burn_heinola): recall at matched neg-image FP
- [T08](#t08) Seed-0 arms incl. D_curr, AuSmoke FOREST VAL: recall at fixed confs and under both threshold regimes
- [T09](#t09) DINOv3 adaptation runs vs YOLO arms, pyro-sdis VAL: recall / tiny at matched neg-image FP
- [T10](#t10) DINOv3 adaptation runs vs YOLO arms, boreal AERIAL VAL: recall at matched neg-image FP
- [T11](#t11) DINOv3 adaptation runs vs YOLO arms, AuSmoke FOREST VAL: fixed confs and both threshold regimes
- [T12](#t12) DINOv3 runs: adaptation mode, parameter counts, training cost
- [T13](#t13) Batch-1 latency (1x L40S fp32 unless stated; n=50 sdis_val frames; preprocessing included, JPEG decode excluded)
- [T14](#t14) FIgLib-39 early-detection replay (HISTORICAL set, used since iteration 1 — not a lockbox)
- [T15](#t15) FIgLib-39 paired detection delay vs expE (events detected by BOTH systems)
- [T16](#t16) FIgLib-39 per-event detection delay (s from first-visible smoke; MISS = never alerted; PRE = pre-visible alert)
- [T17](#t17) Early warning: pyro-sdis VAL frame-level recall at matched frame FPR (all-negative pool)
- [T18](#t18) Early warning: episode detection and onset latency on pyro-sdis VAL, plus the deployed discrete operating point
- [T19](#t19) Early warning: v4 VAL replay per source — the domain-shortcut evidence (thresholds from T17, sdis-val all-neg pool)
- [T20](#t20) Early warning: false-alert EPISODES per hour on continuous negative footage
- [T21](#t21) YOLO every frame + independent DINO pass every Nth frame (sdis VAL sequences)
- [T22](#t22) Compute accounting by iteration (scheduler-derived allocated hours; whole-node accounting)
- [T23](#t23) Iteration-4 compute by kind and by track
- [T24](#t24) Compute incidents: non-terminal / stopped / never-allocated jobs, and iteration-4 FAILED jobs by track
- [T25](#t25) FINAL TEST — pyro-sdis TEST (fixed-camera source domain; sites ferion + croix-augas, frozen since iteration 1)
- [T26](#t26) FINAL TEST — boreal AERIAL TEST (burn_ruokolahti, ONE burn, 9 DJI flights): recall at the pre-registered thresholds
- [T27](#t27) FINAL TEST — AuSmoke FOREST TEST (new region, ACT Australia; 9 units / 4 independent events incl. ONE DJI aerial flight)
- [T28](#t28) FINAL TEST roster: systems, checkpoints, selection basis, fixed thresholds, provenance
- [T29](#t29) AERIAL: boreal VAL (heinola) -> boreal TEST (ruokolahti), exact TEST checkpoints

## Artifacts not found at generation time

- none

## Notes

- none

---

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


---

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


---

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


---

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


---

### T05. Per-seed detail (every checkpoint behind T01-T03) incl. the val-selected confidences used as thresholds

conf@X% = the checkpoint's own confidence at X% neg-image FP on that val set (these are the SOURCE-ONLY regime thresholds from sdis_val and the TARGET-ADAPTED thresholds from boreal_val in PREREGISTRATION_V4.md). Denominators: sdis n_images=7,335 (positives 6,177, negatives 1,158); boreal n_images=1,123 (positives 906, negatives 217); forest n_images=1,471 (positives 1,471, negatives 0). 'n/a' = FP target not reachable.

| arm | ckpt | sdis R@1% | sdis R@10% | sdis tiny@10% | sdis conf@5% | sdis conf@10% | boreal R@1% | boreal R@5% | boreal conf@1% | forest R@0.20 | forest R@0.30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_ctrl_E | v4a | 0.053 | 0.309 | 0.201 | 0.717 | 0.661 | 0.640 | 0.783 | 0.408 | 0.268 | 0.194 |
| A_ctrl_E | v4a_s1 | 0.056 | 0.302 | 0.191 | 0.729 | 0.673 | 0.562 | 0.800 | 0.498 | 0.234 | 0.156 |
| A_ctrl_E | v4a_s2 | 0.056 | 0.304 | 0.196 | 0.728 | 0.673 | 0.584 | 0.795 | 0.488 | 0.237 | 0.158 |
| E_ET | v4e | 0.046 | 0.247 | 0.132 | 0.746 | 0.691 | 0.931 | n/a | 0.068 | 0.266 | 0.218 |
| E_ET | v4e_s1 | 0.040 | 0.291 | 0.172 | 0.715 | 0.660 | 0.955 | n/a | 0.048 | 0.297 | 0.241 |
| E_ET | v4e_s2 | 0.045 | 0.293 | 0.161 | 0.728 | 0.673 | 0.828 | 0.846 | 0.334 | 0.267 | 0.205 |
| C_EGT | v4c | 0.041 | 0.266 | 0.164 | 0.735 | 0.684 | 0.934 | 0.936 | 0.084 | 0.306 | 0.233 |
| C_EGT | v4c_s1 | 0.045 | 0.262 | 0.151 | 0.734 | 0.677 | 0.828 | 0.936 | 0.298 | 0.224 | 0.155 |
| C_EGT | v4c_s2 | 0.044 | 0.262 | 0.167 | 0.737 | 0.686 | 0.940 | n/a | 0.048 | 0.305 | 0.223 |
| D_curr | v4dada | 0.082 | 0.300 | 0.192 | 0.753 | 0.669 | 0.909 | 0.939 | 0.128 | 0.233 | 0.171 |
| D_curr | v4dada_s1 | 0.055 | 0.318 | 0.180 | 0.729 | 0.653 | 0.914 | 0.924 | 0.168 | 0.234 | 0.169 |
| D_curr | v4dada_s2 | 0.042 | 0.345 | 0.229 | 0.698 | 0.625 | 0.864 | 0.881 | 0.338 | 0.267 | 0.191 |
| B_Gonly | v4b | 0.021 | 0.092 | 0.007 | 0.423 | 0.338 | 0.194 | 0.466 | 0.458 | 0.001 | 0.000 |

Sources:
- `reports/v4/seed_summary.json`


---

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


---

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


---

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


---

### T09. DINOv3 adaptation runs vs YOLO arms, pyro-sdis VAL: recall / tiny at matched neg-image FP

Cell = recall / tiny-recall (conf at which this model sits at the FP target), interpolated on each model's own curve with the frozen matching code (IoU>=0.4). Denominators: n_images=7,335 (positives 6,177, negatives 1,158); 1% FP = ~12 negative images. v4B* = iteration-4 DINOv3 Faster R-CNN runs on the E+T corpus (B1 frozen ViT-L control, B4 LoRA r16 ViT-L, B5 full unfreeze ViT-L, B6 LoRA r16 ViT-H+); YOLO_* = seed-0 iteration-4 YOLO arms; v3_* = iteration-3 E-only DINO runs. Compare ONLY at matched FP: adapted DINO confidences saturate >=0.99 (see conf in cells) while YOLO sits at 0.5-0.8.

| neg-image FP target | v4B1_frozen_L | v4B4_lora_L | v4B5_full_L | v4B6_lora_Hplus | YOLO_A_ctrlE | YOLO_E_ET | YOLO_C_EGT | expE_y11m_iter2 | v3_B1_frozenL_Eonly12ep | v3_B3_top4L_Eonly12ep |
|---|---|---|---|---|---|---|---|---|---|---|
| 1% | 0.060 / 0.027 (c=0.986) | 0.085 / 0.051 (c=0.997) | n/a (FP floor 1.4%) | 0.097 / 0.038 (c=0.997) | 0.053 / 0.015 (c=0.801) | 0.046 / 0.006 (c=0.819) | 0.041 / 0.002 (c=0.812) | 0.059 / 0.013 (c=0.799) | 0.071 / 0.029 (c=0.967) | 0.033 / 0.017 (c=0.995) |
| 2% | 0.099 / 0.049 (c=0.980) | 0.115 / 0.073 (c=0.996) | 0.094 / 0.039 (c=0.999) | 0.133 / 0.064 (c=0.996) | 0.104 / 0.040 (c=0.768) | 0.073 / 0.013 (c=0.800) | 0.090 / 0.019 (c=0.781) | 0.092 / 0.026 (c=0.775) | 0.100 / 0.042 (c=0.958) | 0.079 / 0.044 (c=0.992) |
| 5% | 0.175 / 0.104 (c=0.968) | 0.210 / 0.144 (c=0.993) | 0.209 / 0.131 (c=0.998) | 0.227 / 0.144 (c=0.994) | 0.204 / 0.112 (c=0.717) | 0.161 / 0.065 (c=0.746) | 0.172 / 0.078 (c=0.735) | 0.190 / 0.097 (c=0.721) | 0.173 / 0.082 (c=0.938) | 0.154 / 0.089 (c=0.986) |
| 10% | 0.301 / 0.200 (c=0.941) | 0.311 / 0.222 (c=0.988) | 0.316 / 0.231 (c=0.996) | 0.340 / 0.252 (c=0.990) | 0.309 / 0.201 (c=0.661) | 0.247 / 0.132 (c=0.691) | 0.266 / 0.164 (c=0.684) | 0.309 / 0.201 (c=0.655) | 0.275 / 0.161 (c=0.903) | 0.264 / 0.159 (c=0.976) |
| 20% | 0.464 / 0.341 (c=0.868) | 0.478 / 0.392 (c=0.973) | 0.488 / 0.413 (c=0.992) | 0.494 / 0.416 (c=0.980) | 0.483 / 0.367 (c=0.547) | 0.415 / 0.294 (c=0.566) | 0.419 / 0.330 (c=0.587) | 0.480 / 0.364 (c=0.535) | 0.457 / 0.329 (c=0.802) | 0.423 / 0.280 (c=0.945) |

Sources:
- `reports/v4/matched_fp_dino_sdis_val.json`


---

### T10. DINOv3 adaptation runs vs YOLO arms, boreal AERIAL VAL: recall at matched neg-image FP

Cell = recall / tiny-recall (conf at which this model sits at the FP target), interpolated on each model's own curve with the frozen matching code (IoU>=0.4). Denominators: n_images=1,123 (positives 906, negatives 217); 1% FP = ~2 negative images. ONE burn; differences among DINO runs are within a handful of images. Tiny recall structurally 0.

| neg-image FP target | v4B1_frozen_L | v4B4_lora_L | v4B5_full_L | v4B6_lora_Hplus | YOLO_A_ctrlE | YOLO_E_ET | YOLO_C_EGT |
|---|---|---|---|---|---|---|---|
| 1% | 0.974 / 0.000 (c=0.498) | 0.973 / 0.000 (c=0.868) | 0.963 / 0.000 (c=0.568) | 0.964 / 0.000 (c=0.888) | 0.640 / 0.000 (c=0.408) | 0.931 / 0.000 (c=0.068) | 0.934 / 0.000 (c=0.084) |
| 2% | 0.975 / 0.000 (c=0.406) | 0.983 / 0.000 (c=0.496) | 0.963 / 0.000 (c=0.406) | 0.966 / 0.000 (c=0.837) | 0.674 / 0.000 (c=0.377) | FP ceiling 1.4% -> R>=0.954 | 0.935 / 0.000 (c=0.073) |
| 5% | 0.982 / 0.000 (c=0.056) | 0.983 / 0.000 (c=0.146) | 0.968 / 0.000 (c=0.031) | 0.972 / 0.000 (c=0.511) | 0.783 / 0.000 (c=0.257) | FP ceiling 1.4% -> R>=0.954 | 0.936 / 0.000 (c=0.052) |
| 10% | FP ceiling 7.8% -> R>=0.982 | 0.986 / 0.000 (c=0.038) | FP ceiling 5.1% -> R>=0.968 | 0.980 / 0.000 (c=0.043) | 0.801 / 0.000 (c=0.215) | FP ceiling 1.4% -> R>=0.954 | 0.942 / 0.000 (c=0.025) |
| 20% | FP ceiling 7.8% -> R>=0.982 | FP ceiling 12.4% -> R>=0.987 | FP ceiling 5.1% -> R>=0.968 | FP ceiling 12.0% -> R>=0.985 | 0.828 / 0.000 (c=0.165) | FP ceiling 1.4% -> R>=0.954 | FP ceiling 11.5% -> R>=0.943 |

Sources:
- `reports/v4/matched_fp_dino_boreal_val.json`


---

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


---

### T12. DINOv3 runs: adaptation mode, parameter counts, training cost

Parameter counts from checkpoints + HF config; node-hours = orchestrator-observed segment wall time on 8x L40S (incl. pod start and eval passes). best.pt selected by COCO mAP on a 1,500-image sdis-val subset. B5 trains at bs 2/GPU (2x the steps) and evaluates only at the end -> confounded vs B1/B4/B6 (see source).

| run | backbone | adaptation | backbone trainable / total | neck | head | ALL trainable | node-h (8xL40S) | epochs (exposures) | best COCO mAP @step | OOM/retries |
|---|---|---|---|---|---|---|---|---|---|---|
| v4B1_frozen_L | dinov3-vitl16 | frozen (head+neck only) | 0 / 303,129,600 | 7,805,184 | 14,499,865 | 22,305,049 | 1.66 | 10.00 (514,240) | 0.231 @10744 | none |
| v4B4_lora_L | dinov3-vitl16 | LoRA r16 a32 on q/k/v/o | 3,145,728 / 303,129,600 | 7,805,184 | 14,499,865 | 25,450,777 | 3.84 | 10.00 (514,240) | 0.409 @15026 | none |
| v4B5_full_L | dinov3-vitl16 | all blocks trainable | 303,128,576 / 303,129,600 | 7,805,184 | 14,499,865 | 325,433,625 | 8.35 | 10.00 (514,240) | 0.403 @32140 | none |
| v4B6_lora_Hplus | dinov3-vith16plus | LoRA r16 a32 on q/k/v/o | 5,242,880 / 840,592,640 | 10,640,192 | 14,499,865 | 30,382,937 | 8.02 | 10.00 (514,240) | 0.422 @8428 | none |

Sources:
- `reports/v4/dino_adapt_results.md (tables 'Parameter counts', 'Training outcome / cost', 'Runs')`


---

### T13. Batch-1 latency (1x L40S fp32 unless stated; n=50 sdis_val frames; preprocessing included, JPEG decode excluded)

v3 protocol. Merged form folds LoRA into the base linears (deploy form; outputs identical to 1e-6). expE YOLO11m reference on the same protocol: see source text (11.2 ms). Last row: devbox CPU (192 vCPU host, torch 8 threads) from the serving-backend verification.

| system | form | mean ms | median ms | p90 ms | infer-only median ms |
|---|---|---|---|---|---|
| v4B4_lora_L | LoRA unmerged (train form) | 161.8 | 161.6 | 162.9 | 155.8 |
| v4B4_lora_L | LoRA merged into base (deploy form) | 157.5 | 157.6 | 158.7 | 151.7 |
| v4B6_lora_Hplus | LoRA unmerged (train form) | 357.2 | 357.6 | 359.5 | 353.5 |
| v4B6_lora_Hplus | LoRA merged into base (deploy form) | 351.6 | 351.4 | 354.1 | 346.8 |
| v3 B1 frozen ViT-L (reference) | plain | 156.6 | 156.4 | 158.2 | 150.8 |
| v3 B2 frozen ViT-H+ (reference) | plain | 372.7 | 371.2 | 383.5 | 366.3 |
| v4B4_lora_L (serving backend `dino3`, CPU 8 threads) | LoRA unmerged | 5633.6 | 5616.6 | 5714.1 | 5612.1 |

Sources:
- `reports/v4/dino_adapt_results.md (Batch-1 latency table)`
- `reports/v4/dino3_backend_latency_B4_8thr.json`


---

### T14. FIgLib-39 early-detection replay (HISTORICAL set, used since iteration 1 — not a lockbox)

n_events = 39 HPWREN fire events (~80 frames each, 60 s cadence, 40 min before/after first visible smoke). detected = any alert at/after first-visible smoke OR a pre-visible alert that persists; pre-visible = alert before the labelled first-visible frame (unverifiable: early smoke or false alarm; NOT counted as false). R@N min = events with 0 <= delay <= N min / n_events (pre-visible alerts enter at delay 0 s — the same rule as scripts/v4/figlib_replay_report.py). YOLO systems use the deployed 3-of-5 policy at 0.30 / 0.15; DINO runs use their own sdis-val confs at 10% / 20% neg-FP (the v3 threshold-transfer rule). Median/mean delay over detected events. 'TEST ckpt' rows are the exact checkpoints evaluated on the sealed test sets (median seeds; STATUS.md 03:20Z); seed0 rows are the earlier iteration-4 replays.

| system | conf-single / conf-seq | detected | pre-visible | R@1 min | R@3 min | R@5 min | median delay | mean delay | misses |
|---|---|---|---|---|---|---|---|---|---|
| S0 expE_iter2 (deployed thresholds; paired-delay reference) | 0.3 / 0.15 | 30/39 | 5 | 7/39 = 0.179 | 12/39 = 0.308 | 17/39 = 0.436 | 240 s | 404 s | 9 |
| S1 A_ctrl TEST ckpt (v4a_s2) | 0.3 / 0.15 | 30/39 | 6 | 9/39 = 0.231 | 14/39 = 0.359 | 17/39 = 0.436 | 240 s | 456 s | 9 |
| S2 E_ET TEST ckpt (v4e_s1) | 0.3 / 0.15 | 32/39 | 12 | 14/39 = 0.359 | 15/39 = 0.385 | 20/39 = 0.513 | 240 s | 367 s | 7 |
| S2c C_EGT TEST ckpt (v4c_s1) | 0.3 / 0.15 | 32/39 | 14 | 17/39 = 0.436 | 18/39 = 0.462 | 21/39 = 0.538 | 60 s | 364 s | 7 |
| S3 D_curr TEST ckpt (v4dada_s1, post-hoc) | 0.3 / 0.15 | 32/39 | 11 | 13/39 = 0.333 | 15/39 = 0.385 | 20/39 = 0.513 | 240 s | 476 s | 7 |
| A_ctrl_E seed0 (v4a) | 0.3 / 0.15 | 30/39 | 7 | 10/39 = 0.256 | 13/39 = 0.333 | 17/39 = 0.436 | 240 s | 394 s | 9 |
| E_ET seed0 (v4e) | 0.3 / 0.15 | 28/39 | 9 | 11/39 = 0.282 | 12/39 = 0.308 | 18/39 = 0.462 | 240 s | 353 s | 11 |
| C_EGT seed0 (v4c) | 0.3 / 0.15 | 31/39 | 14 | 15/39 = 0.385 | 17/39 = 0.436 | 22/39 = 0.564 | 180 s | 313 s | 8 |
| S4p v4B4_lora_L (DINO, sdis-val-selected confs) | 0.9881 / 0.973 | 26/39 | 0 | 1/39 = 0.026 | 3/39 = 0.077 | 5/39 = 0.128 | 481 s | 667 s | 13 |
| S4 v4B6_lora_Hplus (DINO, sdis-val-selected confs) | 0.9901 / 0.9801 | 24/39 | 0 | 1/39 = 0.026 | 3/39 = 0.077 | 6/39 = 0.154 | 539 s | 705 s | 15 |
| v3 B1 frozen ViT-L (iteration-3 reference) | 0.5624 / 0.3454 | 34/39 | 4 | 6/39 = 0.154 | 15/39 = 0.385 | 22/39 = 0.564 | 240 s | 455 s | 5 |

Sources:
- `reports/replay_expE_figlib.json`
- `reports/v4/replay/replay_v4a_s2_figlib39.json`
- `reports/v4/replay/replay_v4e_s1_figlib39.json`
- `reports/v4/replay/replay_v4c_s1_figlib39.json`
- `reports/v4/replay/replay_v4dada_s1_figlib39.json`
- `reports/v4/replay/replay_v4a_figlib39.json`
- `reports/v4/replay/replay_v4e_figlib39.json`
- `reports/v4/replay/replay_v4c_figlib39.json`
- `reports/v4/replay_v4B4_lora_L_figlib39.json`
- `reports/v4/replay_v4B6_lora_Hplus_figlib39.json`
- `reports/v3/replay_B1_figlib_valmatched.json`


---

### T15. FIgLib-39 paired detection delay vs expE (events detected by BOTH systems)

delta = system delay - expE delay per event (negative = earlier than expE); n_both = events detected by both of the 39. Pre-visible detections carry delay 0 s on both sides. median = statistics.median (mean of the two middle values for even n; scripts/v4/figlib_replay_report.py uses the upper middle value, so its medians can differ by <=1 s).

| system | n both | earlier | tie | later | median delta | mean delta |
|---|---|---|---|---|---|---|
| S1 A_ctrl TEST ckpt (v4a_s2) | 30 | 5 | 20 | 5 | +0 s | +52 s |
| S2 E_ET TEST ckpt (v4e_s1) | 29 | 11 | 11 | 7 | +0 s | -79 s |
| S2c C_EGT TEST ckpt (v4c_s1) | 29 | 10 | 12 | 7 | +0 s | -13 s |
| S3 D_curr TEST ckpt (v4dada_s1, post-hoc) | 29 | 7 | 11 | 11 | +0 s | +99 s |
| A_ctrl_E seed0 (v4a) | 30 | 5 | 19 | 6 | +0 s | -10 s |
| E_ET seed0 (v4e) | 27 | 7 | 12 | 8 | +0 s | +2 s |
| C_EGT seed0 (v4c) | 29 | 12 | 10 | 7 | +0 s | -23 s |
| S4p v4B4_lora_L (DINO, sdis-val-selected confs) | 25 | 3 | 3 | 19 | +181 s | +288 s |
| S4 v4B6_lora_Hplus (DINO, sdis-val-selected confs) | 24 | 2 | 5 | 17 | +61 s | +290 s |
| v3 B1 frozen ViT-L (iteration-3 reference) | 29 | 11 | 10 | 8 | +0 s | -64 s |

Sources:
- `reports/replay_expE_figlib.json`
- `reports/v4/replay/replay_v4a_s2_figlib39.json`
- `reports/v4/replay/replay_v4e_s1_figlib39.json`
- `reports/v4/replay/replay_v4c_s1_figlib39.json`
- `reports/v4/replay/replay_v4dada_s1_figlib39.json`
- `reports/v4/replay/replay_v4a_figlib39.json`
- `reports/v4/replay/replay_v4e_figlib39.json`
- `reports/v4/replay/replay_v4c_figlib39.json`
- `reports/v4/replay_v4B4_lora_L_figlib39.json`
- `reports/v4/replay_v4B6_lora_Hplus_figlib39.json`
- `reports/v3/replay_B1_figlib_valmatched.json`


---

### T16. FIgLib-39 per-event detection delay (s from first-visible smoke; MISS = never alerted; PRE = pre-visible alert)

n_events = 39. One row per event; a system's cell is its delay in seconds (0 for pre-visible alerts, prefixed PRE).

| event | S0 expE_iter2 | S1 A_ctrl TEST ckpt | S2 E_ET TEST ckpt | S2c C_EGT TEST ckpt | S3 D_curr TEST ckpt | A_ctrl_E seed0 | E_ET seed0 | C_EGT seed0 | S4p v4B4_lora_L | S4 v4B6_lora_Hplus | v3 B1 frozen ViT-L |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20160604_FIRE_rm-n-mobo-c | 60 | 60 | 60 | 60 | 60 | 60 | 60 | 60 | 120 | 120 | 60 |
| 20160722_FIRE_mg-s-iqeye | MISS | MISS | MISS | MISS | MISS | MISS | MISS | MISS | MISS | MISS | 1440 |
| 20161113_FIRE_bl-n-mobo-c | 180 | 180 | 1080 | PRE 0 | 1980 | 180 | 1080 | 1080 | 960 | 2340 | 900 |
| 20170625_BBM_bm-n-mobo | 420 | 480 | 480 | 480 | 480 | 480 | 480 | PRE 0 | 960 | 480 | 420 |
| 20170708_Whittier_syp-n-mobo-m | 361 | 421 | 421 | 421 | 421 | 361 | 421 | 421 | 481 | 481 | 241 |
| 20170807_FIRE_bh-n-mobo-c | MISS | MISS | MISS | MISS | MISS | MISS | PRE 0 | PRE 0 | MISS | MISS | 600 |
| 20180517_FIRE_smer-tcs10-mobo-c | MISS | MISS | MISS | MISS | MISS | MISS | MISS | MISS | 1140 | MISS | 1140 |
| 20180806_Holy_sp-s-mobo-c | PRE 0 | PRE 0 | PRE 0 | PRE 0 | PRE 0 | PRE 0 | PRE 0 | PRE 0 | 360 | 300 | PRE 0 |
| 20180827_Holyflareup_sp-e-mobo-c | PRE 0 | 2400 | PRE 0 | 2160 | 720 | PRE 0 | 720 | 2280 | 2340 | MISS | 1380 |
| 20180910_FIRE_smer-tcs8-mobo-c | 240 | 240 | PRE 0 | PRE 0 | 240 | 240 | 240 | PRE 0 | 300 | 540 | 240 |
| 20190717_FIRE_pi-w-mobo-c | 180 | 180 | 61 | 180 | 180 | 180 | 120 | 180 | MISS | MISS | 180 |
| 20190826_FIRE_pi-s-mobo-c | MISS | MISS | 1080 | PRE 0 | PRE 0 | MISS | MISS | MISS | MISS | MISS | MISS |
| 20190924_FIRE_ml-w-mobo-c | 120 | 120 | 300 | 300 | 240 | 240 | 300 | 300 | 480 | 360 | 240 |
| 20191007_FIRE_sm-s-mobo-c | 240 | 480 | 240 | 420 | 480 | 480 | 240 | 420 | 540 | MISS | 300 |
| 20200202_FIRE_hp-w-mobo-c | PRE 0 | PRE 0 | PRE 0 | PRE 0 | 1740 | PRE 0 | MISS | PRE 0 | MISS | MISS | PRE 0 |
| 20200614_DrumCanyon_syp-w-mobo-c | 240 | 240 | 240 | 240 | 240 | 240 | 240 | 240 | 360 | 1440 | 240 |
| 20200705_FIRE_wc-n-mobo-c | 180 | 180 | PRE 0 | PRE 0 | 180 | 180 | PRE 0 | 180 | 180 | 180 | 120 |
| 20200806_SpringsFire_sm-e-mobo-m | MISS | MISS | MISS | PRE 0 | MISS | MISS | MISS | PRE 0 | MISS | MISS | MISS |
| 20210107_Miguelfire_om-w-mobo-c | 1080 | 960 | 960 | 60 | PRE 0 | 840 | PRE 0 | 960 | 360 | 420 | 120 |
| 20210302_FIRE_lp-e-mobo-m | MISS | MISS | MISS | MISS | PRE 0 | MISS | MISS | MISS | MISS | MISS | 180 |
| 20210711_FIRE_wc-e-mobo-c | 240 | 240 | 300 | 240 | 240 | 300 | 300 | 240 | 360 | 300 | 180 |
| 20220210-EmeraldFire-marconi-w-mobo-c | MISS | MISS | 1080 | MISS | MISS | MISS | MISS | MISS | MISS | MISS | MISS |
| 20220405-fire-in-Fallbrook-rm-s-mobo-m | 1740 | 1740 | MISS | MISS | PRE 0 | 1860 | MISS | MISS | MISS | MISS | 420 |
| 20220831-Border32fire-pi-s-mobo-m | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 300 | 540 | 300 | 300 |
| 20230809.111619-BunnieFire-bm-s-mobo-c | 301 | PRE 0 | 301 | PRE 0 | 241 | PRE 0 | 301 | PRE 0 | 361 | 361 | 61 |
| 20231027.095514-prescribed-bi-s-mobo-c | 899 | 1019 | PRE 0 | 1560 | PRE 0 | 1019 | 960 | PRE 0 | 1620 | 1620 | 1139 |
| 20231108.221213-AmagoFire-bi-s-mobo-m | MISS | MISS | MISS | 1621 | 1440 | MISS | MISS | MISS | MISS | MISS | MISS |
| 20231123.134319-SageFire-sm-e-mobo-c | PRE 0 | PRE 0 | PRE 0 | PRE 0 | PRE 0 | PRE 0 | PRE 0 | 662 | 719 | 662 | 239 |
| 20240130.091751-FIRE-wilson-e-mobo-c | 1200 | 1142 | 902 | 1320 | MISS | 1200 | 1200 | PRE 0 | MISS | 1262 | 1502 |
| 20240201.064352-FIRE-mlo-n-mobo-c | MISS | MISS | 1500 | MISS | MISS | MISS | MISS | MISS | MISS | MISS | 1858 |
| 20240721_EagleFire_stgo-n-mobo-c | PRE 0 | PRE 0 | PRE 0 | PRE 0 | PRE 0 | PRE 0 | PRE 0 | PRE 0 | MISS | MISS | MISS |
| 20241007_Posta3Fire_ws-w-mobo-c | 361 | 361 | PRE 0 | PRE 0 | PRE 0 | 361 | PRE 0 | PRE 0 | 900 | 900 | 480 |
| 20250123_GilmanFire_tdlln-s-mobo-c | 61 | 61 | PRE 0 | PRE 0 | PRE 0 | PRE 0 | PRE 0 | PRE 0 | 420 | 780 | 61 |
| 20250709_SteeleFire_sm-n-mobo-c | 1200 | 1200 | PRE 0 | 1200 | 1681 | 1200 | 1200 | PRE 0 | 1200 | 1200 | 1081 |
| 20250710_SpringsFire_lp-n-mobo-c | 0 | 0 | PRE 0 | PRE 0 | PRE 0 | 0 | PRE 0 | PRE 0 | 0 | 0 | 0 |
| 20251102_ScissorsFire_mp-n-mobo-m | 538 | PRE 0 | 718 | 538 | 2338 | 419 | MISS | 419 | 358 | 419 | 179 |
| 20260618_MissionFire_rm-s-mobo-c | 1020 | 1020 | 1020 | PRE 0 | 1020 | 1020 | 1020 | 901 | 1201 | 1020 | PRE 0 |
| 20260622_BernardoFire_wc-w-mobo-c | 480 | 480 | 300 | 359 | 539 | 480 | 239 | 300 | 300 | 539 | 179 |
| 20260820_ChiaFire_hp-n-mobo-c | 475 | 475 | 715 | 475 | 775 | 475 | 775 | 775 | 775 | 896 | PRE 0 |

Sources:
- `reports/replay_expE_figlib.json`
- `reports/v4/replay/replay_v4a_s2_figlib39.json`
- `reports/v4/replay/replay_v4e_s1_figlib39.json`
- `reports/v4/replay/replay_v4c_s1_figlib39.json`
- `reports/v4/replay/replay_v4dada_s1_figlib39.json`
- `reports/v4/replay/replay_v4a_figlib39.json`
- `reports/v4/replay/replay_v4e_figlib39.json`
- `reports/v4/replay/replay_v4c_figlib39.json`
- `reports/v4/replay_v4B4_lora_L_figlib39.json`
- `reports/v4/replay_v4B6_lora_Hplus_figlib39.json`
- `reports/v3/replay_B1_figlib_valmatched.json`


---

### T17. Early warning: pyro-sdis VAL frame-level recall at matched frame FPR (all-negative pool)

Frames 7,335 (positive 6,177 / negative 1,158; clean-negative subset 560); tiny-GT positive frames 5,881. s1_* = single-frame detector max score; s2_persist_3of5 = deployed persistence rule; tmp_* = learned causal temporal head (v3only / v4 = +AuSmoke extension / v4cap = capped 5-seq extension), t1_* = same module on the current frame only; _s0/_s1 = training seeds. Thresholds chosen on these val negatives.

| system | params | PR-AUC | ROC-AUC | R@FPR0.5% | R@FPR1% | R@FPR2% | R@FPR5% | R@FPR10% | tinyR@FPR5% |
|---|---|---|---|---|---|---|---|---|---|
| s1_expE | 0 (rule/raw) | 0.9351 | 0.7685 | 0.0353 | 0.0622 | 0.1004 | 0.2053 | 0.3400 | 0.2095 |
| s1_v4a_ctrlE | 0 (rule/raw) | 0.9364 | 0.7718 | 0.0414 | 0.0580 | 0.1101 | 0.2194 | 0.3371 | 0.2234 |
| s1_v4B1_dinoL | 0 (rule/raw) | 0.9327 | 0.7535 | 0.0537 | 0.0699 | 0.1078 | 0.1967 | 0.3324 | 0.1988 |
| s2_persist_3of5 | 0 (rule/raw) | 0.9169 | 0.7056 | 0.0123 | 0.0515 | 0.0826 | 0.1632 | 0.2752 | 0.1632 |
| t1_v3only_t1only_s0 | 281,153 | 0.9431 | 0.7798 | 0.0503 | 0.0639 | 0.1264 | 0.2589 | 0.4266 | 0.2573 |
| t1_v3only_t1only_s1 | 281,153 | 0.9399 | 0.7761 | 0.0401 | 0.0541 | 0.1130 | 0.2449 | 0.3987 | 0.2450 |
| t1_v4_t1only_s0 | 281,153 | 0.9473 | 0.7980 | 0.0448 | 0.0599 | 0.1305 | 0.3225 | 0.4646 | 0.3192 |
| t1_v4_t1only_s1 | 281,153 | 0.9501 | 0.8088 | 0.0283 | 0.0989 | 0.1965 | 0.3463 | 0.4967 | 0.3333 |
| t1_v4cap_t1only_s0 | 281,153 | 0.9469 | 0.7932 | 0.0380 | 0.1030 | 0.1463 | 0.2899 | 0.4412 | 0.2906 |
| t1_v4cap_t1only_s1 | 281,153 | 0.9426 | 0.7897 | 0.0204 | 0.0539 | 0.1260 | 0.2671 | 0.4345 | 0.2644 |
| tmp_v3only_temporal_s0 | 281,153 | 0.9404 | 0.7736 | 0.0423 | 0.0711 | 0.1512 | 0.2498 | 0.3983 | 0.2483 |
| tmp_v3only_temporal_s1 | 281,153 | 0.9391 | 0.7667 | 0.0299 | 0.0835 | 0.1813 | 0.2462 | 0.3861 | 0.2460 |
| tmp_v4_temporal_s0 | 281,153 | 0.9388 | 0.7662 | 0.0400 | 0.0737 | 0.1350 | 0.2576 | 0.3983 | 0.2472 |
| tmp_v4_temporal_s1 | 281,153 | 0.9457 | 0.7955 | 0.0303 | 0.0860 | 0.1376 | 0.2812 | 0.4133 | 0.2761 |
| tmp_v4cap_temporal_s0 | 281,153 | 0.9426 | 0.7810 | 0.0291 | 0.0565 | 0.1439 | 0.2699 | 0.4208 | 0.2683 |
| tmp_v4cap_temporal_s1 | 281,153 | 0.9430 | 0.7781 | 0.0562 | 0.0908 | 0.1462 | 0.2815 | 0.4102 | 0.2758 |

Sources:
- `reports/v4/trackd_v4_eval.json (A_sdis_val_frame_level)`


---

### T18. Early warning: episode detection and onset latency on pyro-sdis VAL, plus the deployed discrete operating point

episodes = 469 positive smoke episodes in sdis_val sequences; latency in frames (29 s cadence) from episode start to first alert, median over detected episodes. Thresholds matched on frame FPR over the 1,158 all-negative frames. Deployed point = 0.30 single / 0.15 seq, 3-of-5.

| system | episodes | det@FPR1% | median latency (frames) | det@FPR5% | median latency (frames) |
|---|---|---|---|---|---|
| s1_expE | 469 | 0.1748 | 4.0 | 0.4606 | 1.5 |
| s1_v4a_ctrlE | 469 | 0.1898 | 4.0 | 0.4840 | 1.0 |
| s1_v4B1_dinoL | 469 | 0.1663 | 4.5 | 0.3348 | 2.0 |
| s2_persist_3of5 | 469 | 0.1173 | 6.0 | 0.2665 | 3.0 |
| t1_v3only_t1only_s0 | 469 | 0.1343 | 4.0 | 0.3795 | 2.0 |
| t1_v3only_t1only_s1 | 469 | 0.1429 | 4.0 | 0.4136 | 1.0 |
| t1_v4_t1only_s0 | 469 | 0.0981 | 6.5 | 0.4478 | 1.0 |
| t1_v4_t1only_s1 | 469 | 0.1855 | 5.0 | 0.4328 | 2.0 |
| t1_v4cap_t1only_s0 | 469 | 0.1620 | 3.5 | 0.4200 | 1.0 |
| t1_v4cap_t1only_s1 | 469 | 0.1322 | 4.0 | 0.4009 | 2.0 |
| tmp_v3only_temporal_s0 | 469 | 0.1322 | 6.0 | 0.3710 | 2.0 |
| tmp_v3only_temporal_s1 | 469 | 0.1514 | 5.0 | 0.3731 | 2.0 |
| tmp_v4_temporal_s0 | 469 | 0.1087 | 6.0 | 0.3454 | 2.0 |
| tmp_v4_temporal_s1 | 469 | 0.1620 | 5.0 | 0.4136 | 2.0 |
| tmp_v4cap_temporal_s0 | 469 | 0.1023 | 7.0 | 0.3667 | 2.0 |
| tmp_v4cap_temporal_s1 | 469 | 0.1322 | 6.0 | 0.3646 | 2.0 |

**Deployed discrete operating point (expE)**

| point | FPR all-neg | FPR clean-neg | recall pos frames | recall tiny frames |
|---|---|---|---|---|
| confirmed_only(s2>=0.15) | 0.5345 | 0.2768 | 0.8342 | 0.8383 |
| confirmed_or_provisional(s2>=0.15|s1>=0.30) | 0.5872 | 0.3589 | 0.8982 | 0.9043 |

Sources:
- `reports/v4/trackd_v4_eval.json (A_sdis_val_frame_level)`


---

### T19. Early warning: v4 VAL replay per source — the domain-shortcut evidence (thresholds from T17, sdis-val all-neg pool)

For NEGATIVE footage (boreal/negative_footage = heinola non-smoke clips, firesense/negative_footage) every alert is false: the frame/clip alert rate IS the false-alarm rate. For positive sources the alert rate is a recall proxy only (boreal/positive is video-level truth; ausmoke/positive is positive from frame 0). Heads trained with the v4 AuSmoke extension (tmp_v4*, t1_v4*) fire on a large share of negative frames where v3-only heads fire on ~0% — the shortcut. NOTE: firesense negatives are also training negatives inside the YOLO corpora B_G/C_EGT/D_adapt (not inside these heads' data).

**ausmoke/positive** — 4 sequences / 1,471 frames, wall clock unknown (original fps never recorded (order units) -> no per-hour rate is emitted for this source), truth_level=frame_boxes, frame_truth=True, parent units=4
  - CAVEAT: positive from frame 0 — NO pre-onset segment exists; contributes nothing to the negative-footage pool

| system | thr@FPR1% | frame alert @1% | clip alert @1% | episodes/h @1% | frame alert @5% | clip alert @5% |
|---|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.3% | 50.0% | n/a | 2.2% | 50.0% |
| s2_persist_3of5 | 0.7874 | 0.1% | 25.0% | n/a | 1.7% | 50.0% |
| t1_v3only_t1only_s0 | 0.9367 | 0.6% | 75.0% | n/a | 5.9% | 75.0% |
| t1_v3only_t1only_s1 | 0.9601 | 0.0% | 0.0% | n/a | 2.4% | 75.0% |
| t1_v4_t1only_s0 | 0.9951 | 0.0% | 0.0% | n/a | 10.4% | 100.0% |
| t1_v4_t1only_s1 | 0.9637 | 1.8% | 25.0% | n/a | 14.3% | 75.0% |
| t1_v4cap_t1only_s0 | 0.9960 | 1.6% | 50.0% | n/a | 8.5% | 75.0% |
| t1_v4cap_t1only_s1 | 0.9347 | 0.9% | 50.0% | n/a | 6.5% | 75.0% |
| tmp_v3only_temporal_s0 | 0.9988 | 0.3% | 25.0% | n/a | 3.2% | 75.0% |
| tmp_v3only_temporal_s1 | 0.9556 | 0.0% | 0.0% | n/a | 2.0% | 75.0% |
| tmp_v4_temporal_s0 | 0.9998 | 3.3% | 50.0% | n/a | 11.3% | 75.0% |
| tmp_v4_temporal_s1 | 0.9857 | 0.7% | 25.0% | n/a | 7.7% | 75.0% |
| tmp_v4cap_temporal_s0 | 0.9931 | 3.5% | 75.0% | n/a | 10.6% | 75.0% |
| tmp_v4cap_temporal_s1 | 0.9750 | 3.1% | 25.0% | n/a | 11.3% | 75.0% |
| DEPLOYED expE (prov OR conf, 0.30/0.15) | - | 68.2% | 100.0% | n/a | - | - |

**boreal/negative_footage** — 15 sequences / 471 frames, 0.131 h, truth_level=videolabel, frame_truth=True, parent units=2
  - CAVEAT: clip from a TRAIN burn site (evo/karkkila) — background seen at train time; weaker negative evidence than val-site footage

| system | thr@FPR1% | frame alert @1% | clip alert @1% | episodes/h @1% | frame alert @5% | clip alert @5% |
|---|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.0% | 0.0% | 0.00 | 1.7% | 6.7% |
| s2_persist_3of5 | 0.7874 | 0.0% | 0.0% | 0.00 | 1.7% | 6.7% |
| t1_v3only_t1only_s0 | 0.9367 | 0.0% | 0.0% | 0.00 | 0.9% | 6.7% |
| t1_v3only_t1only_s1 | 0.9601 | 0.0% | 0.0% | 0.00 | 0.6% | 6.7% |
| t1_v4_t1only_s0 | 0.9951 | 26.3% | 33.3% | 38.23 | 41.6% | 66.7% |
| t1_v4_t1only_s1 | 0.9637 | 44.2% | 60.0% | 68.81 | 46.3% | 66.7% |
| t1_v4cap_t1only_s0 | 0.9960 | 5.3% | 33.3% | 38.23 | 25.5% | 46.7% |
| t1_v4cap_t1only_s1 | 0.9347 | 15.7% | 33.3% | 38.23 | 31.2% | 53.3% |
| tmp_v3only_temporal_s0 | 0.9988 | 0.0% | 0.0% | 0.00 | 1.5% | 6.7% |
| tmp_v3only_temporal_s1 | 0.9556 | 0.0% | 0.0% | 0.00 | 0.6% | 6.7% |
| tmp_v4_temporal_s0 | 0.9998 | 43.3% | 73.3% | 84.10 | 56.9% | 86.7% |
| tmp_v4_temporal_s1 | 0.9857 | 42.7% | 60.0% | 68.81 | 46.1% | 60.0% |
| tmp_v4cap_temporal_s0 | 0.9931 | 16.4% | 40.0% | 45.87 | 30.1% | 46.7% |
| tmp_v4cap_temporal_s1 | 0.9750 | 27.6% | 46.7% | 53.52 | 37.4% | 66.7% |
| DEPLOYED expE (prov OR conf, 0.30/0.15) | - | 30.1% | 60.0% | 68.81 | - | - |

**boreal/positive** — 111 sequences / 3,432 frames, 0.953 h, truth_level=videolabel, frame_truth=False, parent units=1

| system | thr@FPR1% | frame alert @1% | clip alert @1% | episodes/h @1% | frame alert @5% | clip alert @5% |
|---|---|---|---|---|---|---|
| s1_expE | 0.7997 | 4.6% | 16.2% | 18.88 | 17.1% | 35.1% |
| s2_persist_3of5 | 0.7874 | 6.0% | 15.3% | 17.83 | 15.7% | 28.8% |
| t1_v3only_t1only_s0 | 0.9367 | 0.1% | 0.9% | 1.05 | 1.8% | 6.3% |
| t1_v3only_t1only_s1 | 0.9601 | 0.0% | 0.0% | 0.00 | 0.4% | 3.6% |
| t1_v4_t1only_s0 | 0.9951 | 6.6% | 37.8% | 44.06 | 55.3% | 96.4% |
| t1_v4_t1only_s1 | 0.9637 | 79.5% | 98.2% | 114.34 | 90.7% | 100.0% |
| t1_v4cap_t1only_s0 | 0.9960 | 10.7% | 35.1% | 40.91 | 55.2% | 92.8% |
| t1_v4cap_t1only_s1 | 0.9347 | 25.1% | 54.9% | 63.99 | 56.0% | 91.0% |
| tmp_v3only_temporal_s0 | 0.9988 | 0.0% | 0.9% | 1.05 | 0.7% | 3.6% |
| tmp_v3only_temporal_s1 | 0.9556 | 0.1% | 1.8% | 2.10 | 2.4% | 11.7% |
| tmp_v4_temporal_s0 | 0.9998 | 84.1% | 99.1% | 115.39 | 89.8% | 100.0% |
| tmp_v4_temporal_s1 | 0.9857 | 76.4% | 99.1% | 115.39 | 86.9% | 100.0% |
| tmp_v4cap_temporal_s0 | 0.9931 | 36.8% | 76.6% | 89.16 | 66.9% | 98.2% |
| tmp_v4cap_temporal_s1 | 0.9750 | 58.3% | 92.8% | 108.05 | 76.4% | 98.2% |
| DEPLOYED expE (prov OR conf, 0.30/0.15) | - | 80.8% | 98.2% | 114.34 | - | - |

**firesense/negative_footage** — 9 sequences / 292 frames, 0.081 h, truth_level=videolabel, frame_truth=True, parent units=9
  - CAVEAT: frames used as background negatives in v4 YOLO train corpora (e.g. CORPUS_B_G_v1: 292 firesense background_neg lines) — false-alert numbers are OPTIMISTIC for models trained on those corpora

| system | thr@FPR1% | frame alert @1% | clip alert @1% | episodes/h @1% | frame alert @5% | clip alert @5% |
|---|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| s2_persist_3of5 | 0.7874 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| t1_v3only_t1only_s0 | 0.9367 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| t1_v3only_t1only_s1 | 0.9601 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| t1_v4_t1only_s0 | 0.9951 | 82.5% | 77.8% | 86.31 | 96.2% | 88.9% |
| t1_v4_t1only_s1 | 0.9637 | 98.3% | 100.0% | 110.97 | 100.0% | 100.0% |
| t1_v4cap_t1only_s0 | 0.9960 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| t1_v4cap_t1only_s1 | 0.9347 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| tmp_v3only_temporal_s0 | 0.9988 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| tmp_v3only_temporal_s1 | 0.9556 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| tmp_v4_temporal_s0 | 0.9998 | 92.5% | 100.0% | 110.97 | 96.2% | 100.0% |
| tmp_v4_temporal_s1 | 0.9857 | 88.4% | 88.9% | 98.64 | 94.5% | 100.0% |
| tmp_v4cap_temporal_s0 | 0.9931 | 4.5% | 11.1% | 12.33 | 10.6% | 11.1% |
| tmp_v4cap_temporal_s1 | 0.9750 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| DEPLOYED expE (prov OR conf, 0.30/0.15) | - | 11.6% | 33.3% | 36.99 | - | - |

Sources:
- `reports/v4/trackd_v4_eval.json (B_v4_val_replay)`


---

### T20. Early warning: false-alert EPISODES per hour on continuous negative footage

Episode = maximal run of alert-on frames; runs separated by < 60.0 s (deployed cooldown) merge. CI = exact Poisson 95%. Pools are tiny (minutes of footage) — read as order-of-magnitude.

**pyrosdis_val_all_neg_sequences** — 12 sequences / 219 frames / 1.521 h wall clock. pyro-sdis val sequences whose every frame is negative; real 29 s-cadence wall clock; expE never saw the val sites

| system | thr@FPR1% | false episodes/h @FPR1% | CI95 | false episodes/h @FPR5% | CI95 |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| s1_v4a_ctrlE | 0.8000 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| s1_v4B1_dinoL | 0.9858 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| s2_persist_3of5 | 0.7874 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| t1_v3only_t1only_s0 | 0.9367 | 0.00 | [0.0, 2.42] | 0.66 | [0.02, 3.66] |
| t1_v3only_t1only_s1 | 0.9601 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| t1_v4_t1only_s0 | 0.9951 | 0.00 | [0.0, 2.42] | 0.66 | [0.02, 3.66] |
| t1_v4_t1only_s1 | 0.9637 | 0.00 | [0.0, 2.42] | 0.66 | [0.02, 3.66] |
| t1_v4cap_t1only_s0 | 0.9960 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| t1_v4cap_t1only_s1 | 0.9347 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.00 | [0.0, 2.42] | 1.31 | [0.16, 4.75] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| tmp_v4_temporal_s0 | 0.9998 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| tmp_v4_temporal_s1 | 0.9857 | 0.00 | [0.0, 2.42] | 0.66 | [0.02, 3.66] |
| tmp_v4cap_temporal_s0 | 0.9931 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| tmp_v4cap_temporal_s1 | 0.9750 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| DEPLOYED expE (0.30/0.15) | - | 3.29 | [1.07, 7.67] | - | - |

**boreal_val_nonsmoke_heinola_only** — 14 sequences / 439 frames / 0.122 h wall clock. boreal heinola (VAL burn site) non-smoke clips, 1 fps

| system | thr@FPR1% | false episodes/h @FPR1% | CI95 | false episodes/h @FPR5% | CI95 |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.00 | [0.0, 30.25] | 8.20 | [0.21, 45.69] |
| s2_persist_3of5 | 0.7874 | 0.00 | [0.0, 30.25] | 8.20 | [0.21, 45.69] |
| t1_v3only_t1only_s0 | 0.9367 | 0.00 | [0.0, 30.25] | 8.20 | [0.21, 45.69] |
| t1_v3only_t1only_s1 | 0.9601 | 0.00 | [0.0, 30.25] | 8.20 | [0.21, 45.69] |
| t1_v4_t1only_s0 | 0.9951 | 41.00 | [13.31, 95.69] | 82.00 | [39.32, 150.81] |
| t1_v4_t1only_s1 | 0.9637 | 73.80 | [33.75, 140.1] | 82.00 | [39.32, 150.81] |
| t1_v4cap_t1only_s0 | 0.9960 | 41.00 | [13.31, 95.69] | 57.40 | [23.08, 118.27] |
| t1_v4cap_t1only_s1 | 0.9347 | 41.00 | [13.31, 95.69] | 65.60 | [28.32, 129.27] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.00 | [0.0, 30.25] | 8.20 | [0.21, 45.69] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.00 | [0.0, 30.25] | 8.20 | [0.21, 45.69] |
| tmp_v4_temporal_s0 | 0.9998 | 90.21 | [45.03, 161.4] | 106.61 | [56.76, 182.3] |
| tmp_v4_temporal_s1 | 0.9857 | 73.80 | [33.75, 140.1] | 73.80 | [33.75, 140.1] |
| tmp_v4cap_temporal_s0 | 0.9931 | 49.20 | [18.06, 107.09] | 57.40 | [23.08, 118.27] |
| tmp_v4cap_temporal_s1 | 0.9750 | 57.40 | [23.08, 118.27] | 82.00 | [39.32, 150.81] |
| DEPLOYED expE (0.30/0.15) | - | 73.80 | [33.75, 140.1] | - | - |

**boreal_nonsmoke_all_incl_train_site** — 15 sequences / 471 frames / 0.131 h wall clock. adds the karkkila clip, whose background is a TRAIN burn site

| system | thr@FPR1% | false episodes/h @FPR1% | CI95 | false episodes/h @FPR5% | CI95 |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.00 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| s2_persist_3of5 | 0.7874 | 0.00 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| t1_v3only_t1only_s0 | 0.9367 | 0.00 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| t1_v3only_t1only_s1 | 0.9601 | 0.00 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| t1_v4_t1only_s0 | 0.9951 | 38.22 | [12.41, 89.18] | 76.43 | [36.65, 140.56] |
| t1_v4_t1only_s1 | 0.9637 | 68.79 | [31.46, 130.58] | 76.43 | [36.65, 140.56] |
| t1_v4cap_t1only_s0 | 0.9960 | 38.22 | [12.41, 89.18] | 53.50 | [21.51, 110.24] |
| t1_v4cap_t1only_s1 | 0.9347 | 38.22 | [12.41, 89.18] | 61.15 | [26.4, 120.48] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.00 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.00 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| tmp_v4_temporal_s0 | 0.9998 | 84.08 | [41.97, 150.44] | 99.36 | [52.91, 169.91] |
| tmp_v4_temporal_s1 | 0.9857 | 68.79 | [31.46, 130.58] | 68.79 | [31.46, 130.58] |
| tmp_v4cap_temporal_s0 | 0.9931 | 45.86 | [16.83, 99.82] | 53.50 | [21.51, 110.24] |
| tmp_v4cap_temporal_s1 | 0.9750 | 53.50 | [21.51, 110.24] | 76.43 | [36.65, 140.56] |
| DEPLOYED expE (0.30/0.15) | - | 68.79 | [31.46, 130.58] | - | - |

**firesense_negatives_CONTAMINATED** **[CONTAMINATED — see note]** — 9 sequences / 292 frames / 0.081 h wall clock. these 292 frames are background negatives inside v4 YOLO CORPUS_B_G_v1 / C_EGT_v1 / D_adapt_v1 -> optimistic for any model trained on those corpora. The temporal heads here read frozen expE (v3) features and expE never saw them, but the numbers are still flagged.

| system | thr@FPR1% | false episodes/h @FPR1% | CI95 | false episodes/h @FPR5% | CI95 |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| s2_persist_3of5 | 0.7874 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| t1_v3only_t1only_s0 | 0.9367 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| t1_v3only_t1only_s1 | 0.9601 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| t1_v4_t1only_s0 | 0.9951 | 86.30 | [34.7, 177.81] | 98.63 | [42.58, 194.34] |
| t1_v4_t1only_s1 | 0.9637 | 110.96 | [50.74, 210.63] | 110.96 | [50.74, 210.63] |
| t1_v4cap_t1only_s0 | 0.9960 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| t1_v4cap_t1only_s1 | 0.9347 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| tmp_v4_temporal_s0 | 0.9998 | 110.96 | [50.74, 210.63] | 110.96 | [50.74, 210.63] |
| tmp_v4_temporal_s1 | 0.9857 | 98.63 | [42.58, 194.34] | 110.96 | [50.74, 210.63] |
| tmp_v4cap_temporal_s0 | 0.9931 | 12.33 | [0.31, 68.69] | 12.33 | [0.31, 68.69] |
| tmp_v4cap_temporal_s1 | 0.9750 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| DEPLOYED expE (0.30/0.15) | - | 36.99 | [7.63, 108.09] | - | - |

Sources:
- `reports/v4/trackd_v4_eval.json (C_false_alert_episodes_per_hour)`


---

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


---

### T22. Compute accounting by iteration (scheduler-derived allocated hours; whole-node accounting)

Allocated node-hours = scheduler completion - start (queue wait excluded); GPU-hours = node-hours x GPUs/node. No pod utilization telemetry exists (utilization column = n/a). Iterations 3/4 attributed by exact window start; 1 vs 2 by job-name pattern (approx). 316 job records (12 flagged in_scope=False: same requester, other project, excluded from every iteration total); ledger snapshot = 2026-09-15T03:32:00+00:00 (0 job(s) still open at dump). Iteration-4 ceiling: 384 node-h / 3072 GPU-h (cluster/v4/BUDGET_V4.json). 'failed/retry' = FAILED outcome or training(failed/retry) kind.

| iteration | jobs | node-h | GPU-h | node-h by GPU type | failed/retry node-h | attribution |
|---|---|---|---|---|---|---|
| 1 | 83 | 40.1 | 320 | L40S 35.3, B200 4.5, L4 0.3, H200 0.1 | 9.3 (39 jobs) | name-based (approx) |
| 2 | 18 | 10.1 | 80 | L40S 10.1 | 1.6 (1 jobs) | name-based (approx) |
| 3 | 63 | 20.6 | 165 | L40S 20.6 | 0.1 (6 jobs) | time-window (exact) |
| 4 | 140 | 80.3 | 642 | L40S 80.3 | 3.8 (24 jobs) | time-window (exact) |
| ALL in-scope | 304 | 151.0 | 1207 |  | 14.8 (70 jobs) |  |
| external (other project; excluded) | 12 | 0.6 | 5 | B200 | - | name-prefix: probe-bindle |

Sources:
- `reports/v4/COMPUTE_LEDGER.csv`
- `cluster/v4/BUDGET_V4.json`
- `reports/v4/ledger/kraken_*_{raw.json,table.txt} (raw dumps)`


---

### T23. Iteration-4 compute by kind and by track

140 in-scope iteration-4 jobs in the ledger snapshot (2026-09-15T03:32:00+00:00); 80.3 node-h = 642 GPU-h of the 3072 GPU-h ceiling (21%). External same-requester jobs in the window are excluded (see T22). No job was open at the dump. 

**by kind**

| kind | jobs | node-h | GPU-h | share of v4 node-h |
|---|---|---|---|---|
| training | 106 | 74.3 | 595 | 93% |
| training(failed/retry) | 22 | 3.5 | 28 | 4% |
| evaluation | 11 | 2.2 | 18 | 3% |
| feature-extraction | 1 | 0.2 | 2 | 0% |
| ALL v4 | 140 | 80.3 | 642 | 21% of cap |

**by track**

| track | jobs | node-h | GPU-h | failed/retry node-h |
|---|---|---|---|---|
| yolo-matrix | 78 | 53.6 | 429 | 3.5 |
| dino-adapt | 29 | 21.7 | 173 | 0.0 |
| eval | 15 | 2.5 | 20 | 0.3 |
| yolo-train | 15 | 1.7 | 14 | 0.0 |
| temporal | 3 | 0.7 | 6 | 0.0 |

Sources:
- `reports/v4/COMPUTE_LEDGER.csv`
- `cluster/v4/BUDGET_V4.json`


---

### T24. Compute incidents: non-terminal / stopped / never-allocated jobs, and iteration-4 FAILED jobs by track

Top: every ledger row whose outcome is neither SUCCEEDED nor FAILED (5 rows; QUEUED(never allocated) = idle allocation request with 0 hours; STOPPED = manually killed; RUNNING(open at dump) = truncated at the snapshot). Bottom: iteration-4 FAILED jobs (24 jobs, 3.80 node-h) — the arm-D DDP crash loop (14 segments, exit 101, ~1.2 node-h per STATUS.md) sits in yolo-matrix.

| job | iter | track | kind | outcome | created | alloc start | alloc end | node-h |
|---|---|---|---|---|---|---|---|---|
| tianrgua-sv4-dgnx | 1 | infra-probe | staging/probe | STOPPED(never allocated) | 2026-09-11T07:16:49+00:00 | 2026-09-11T07:16:00+00:00 | 2026-09-11T07:16:00+00:00 | 0.00 |
| tianrgua-rfdetrlsanitya1-i2ki | 3 | rfdetr/deimv2 | training | STOPPED | 2026-09-13T09:01:58+00:00 | 2026-09-13T09:01:00+00:00 | 2026-09-13T09:13:00+00:00 | 0.20 |
| tianrgua-d3c17bheadcas2-zfpu | 3 | dino-v3 | training | STOPPED | 2026-09-14T03:17:53+00:00 | 2026-09-14T03:17:00+00:00 | 2026-09-14T03:32:00+00:00 | 0.25 |
| tianrgua-d3b1cachednos1-pubh | 3 | dino-v3 | training | STOPPED | 2026-09-14T03:18:48+00:00 | 2026-09-14T03:18:00+00:00 | 2026-09-14T03:31:00+00:00 | 0.22 |
| probe-bindle-cqbg | external | external(other project) | external | QUEUED(never allocated) | 2026-09-14T23:44:43+00:00 | 2026-09-14T23:44:43+00:00 | 2026-09-14T23:44:43+00:00 | 0.00 |

**iteration-4 FAILED jobs by track**

| track | failed jobs | node-h |
|---|---|---|
| yolo-matrix | 19 | 3.45 |
| eval | 3 | 0.32 |
| yolo-train | 2 | 0.03 |

Sources:
- `reports/v4/COMPUTE_LEDGER.csv`


---

### T25. FINAL TEST — pyro-sdis TEST (fixed-camera source domain; sites ferion + croix-augas, frozen since iteration 1)

One-shot evaluation. Denominators: n_images=4,124 (positives 3,207, negatives 917); 1% neg-FP = ~9 negative images. Matched-FP interpolated on each model's own curve (IoU>=0.4, --top-fine grid for saturated DINO scores). Compare across architectures only at matched FP.

**(a) matched neg-image FP — recall / tiny-recall (conf)**

| neg-image FP target | S0_expE | S0b_expB | S1_A | S2_E_ET | S2c_C_EGT | S3_B_G | S3_D_curr | S4_B6_loraH | S4c_B1_frozenL | S4p_B4_loraL |
|---|---|---|---|---|---|---|---|---|---|---|
| 1% | 0.107 / 0.034 (c=0.766) | 0.118 / 0.026 (c=0.759) | 0.130 / 0.054 (c=0.759) | 0.100 / 0.025 (c=0.756) | 0.116 / 0.025 (c=0.732) | 0.019 / 0.001 (c=0.549) | 0.122 / 0.040 (c=0.719) | 0.051 / 0.005 (c=0.997) | 0.068 / 0.033 (c=0.985) | 0.080 / 0.020 (c=0.996) |
| 2% | 0.172 / 0.081 (c=0.727) | 0.171 / 0.052 (c=0.721) | 0.170 / 0.084 (c=0.732) | 0.128 / 0.042 (c=0.736) | 0.150 / 0.047 (c=0.705) | 0.025 / 0.001 (c=0.508) | 0.170 / 0.071 (c=0.672) | 0.090 / 0.016 (c=0.996) | 0.115 / 0.059 (c=0.978) | 0.127 / 0.039 (c=0.994) |
| 5% | 0.256 / 0.154 (c=0.671) | 0.258 / 0.115 (c=0.655) | 0.273 / 0.175 (c=0.660) | 0.237 / 0.136 (c=0.654) | 0.214 / 0.110 (c=0.648) | 0.037 / 0.005 (c=0.416) | 0.271 / 0.149 (c=0.561) | 0.203 / 0.071 (c=0.991) | 0.192 / 0.103 (c=0.963) | 0.209 / 0.090 (c=0.990) |
| 10% | 0.350 / 0.257 (c=0.599) | 0.354 / 0.206 (c=0.577) | 0.385 / 0.287 (c=0.564) | 0.310 / 0.204 (c=0.587) | 0.318 / 0.214 (c=0.554) | 0.059 / 0.010 (c=0.306) | 0.362 / 0.235 (c=0.456) | 0.363 / 0.189 (c=0.980) | 0.322 / 0.179 (c=0.929) | 0.329 / 0.182 (c=0.980) |
| 20% | 0.506 / 0.406 (c=0.457) | 0.501 / 0.364 (c=0.409) | 0.549 / 0.441 (c=0.409) | 0.424 / 0.324 (c=0.459) | 0.448 / 0.349 (c=0.413) | 0.098 / 0.018 (c=0.160) | 0.478 / 0.361 (c=0.311) | 0.558 / 0.376 (c=0.943) | 0.514 / 0.325 (c=0.820) | 0.509 / 0.343 (c=0.941) |

**(b) R@10% neg-FP deltas**

| system | R@10% | vs S0_expE | vs S1_A |
|---|---|---|---|
| S0_expE | 0.350 | +0.0 pt | -3.6 pt |
| S0b_expB | 0.354 | +0.4 pt | -3.1 pt |
| S1_A | 0.385 | +3.6 pt | +0.0 pt |
| S2_E_ET | 0.310 | -4.0 pt | -7.6 pt |
| S2c_C_EGT | 0.318 | -3.1 pt | -6.7 pt |
| S3_B_G | 0.059 | -29.1 pt | -32.7 pt |
| S3_D_curr | 0.362 | +1.3 pt | -2.3 pt |
| S4_B6_loraH | 0.363 | +1.4 pt | -2.2 pt |
| S4c_B1_frozenL | 0.322 | -2.8 pt | -6.3 pt |
| S4p_B4_loraL | 0.329 | -2.1 pt | -5.7 pt |

**(c) recall / tiny at the pre-registered VAL-selected thresholds, with the FP realised on test negatives**

| system | SOURCE-ONLY sdis-val conf@5% | SOURCE-ONLY sdis-val conf@10% | SOURCE-ONLY sdis-val conf@20% | TARGET-ADAPTED boreal-val conf@1% | TARGET-ADAPTED boreal-val conf@5% |
|---|---|---|---|---|---|
| S0 expE (iteration-2 YOLO11m candidate) | 0.182 / 0.086 (c=0.7210; neg flagged 22/917) | 0.274 / 0.174 (c=0.6546; neg flagged 56/917) | 0.428 / 0.332 (c=0.5354; neg flagged 125/917) | 0.464 / 0.370 (c=0.4983; neg flagged 159/917) | 0.670 / 0.578 (c=0.2615; neg flagged 328/917) |
| S0b expB (deployed default, YOLO11s@1280) | 0.167 / 0.050 (c=0.7231; neg flagged 17/917) | 0.264 / 0.121 (c=0.6503; neg flagged 49/917) | 0.397 / 0.244 (c=0.5323; neg flagged 123/917) | 0.424 / 0.275 (c=0.4983; neg flagged 137/917) | 0.553 / 0.429 (c=0.3477; neg flagged 220/917) |
| S1 A_ctrl_E, median seed | 0.175 / 0.087 (c=0.7283; neg flagged 19/917) | 0.255 / 0.161 (c=0.6727; neg flagged 41/917) | 0.413 / 0.310 (c=0.5392; neg flagged 106/917) | 0.476 / 0.368 (c=0.4883; neg flagged 138/917) | 0.660 / 0.568 (c=0.2558; neg flagged 298/917) |
| S2 E_ET, median seed | 0.155 / 0.065 (c=0.7154; neg flagged 28/917) | 0.229 / 0.128 (c=0.6603; neg flagged 44/917) | 0.347 / 0.241 (c=0.5534; neg flagged 111/917) | 0.718 / 0.672 (c=0.0483; neg flagged 514/917) | n/a(val FP floor) |
| S2c C_EGT, median seed | 0.113 / 0.022 (c=0.7344; neg flagged 7/917) | 0.181 / 0.075 (c=0.6768; neg flagged 33/917) | 0.294 / 0.191 (c=0.5767; neg flagged 74/917) | 0.535 / 0.448 (c=0.2983; neg flagged 274/917) | 0.711 / 0.671 (c=0.0705; neg flagged 535/917) |
| S3 B_Gonly, seed 0 | 0.036 / 0.005 (c=0.4231; neg flagged 45/917) | 0.051 / 0.008 (c=0.3380; neg flagged 77/917) | 0.091 / 0.017 (c=0.1885; neg flagged 160/917) | 0.032 / 0.002 (c=0.4583; neg flagged 38/917) | 0.048 / 0.006 (c=0.3558; neg flagged 68/917) |
| S3 D_curr, median seed | 0.112 / 0.033 (c=0.7292; neg flagged 7/917) | 0.189 / 0.085 (c=0.6530; neg flagged 21/917) | 0.303 / 0.177 (c=0.5258; neg flagged 58/917) | 0.588 / 0.492 (c=0.1683; neg flagged 280/917) | 0.622 / 0.538 (c=0.1303; neg flagged 317/917) |
| S4 v4B6_lora_Hplus (DINOv3 LoRA-H+) | 0.143 / 0.040 (c=0.9939; neg flagged 30/917) | 0.225 / 0.083 (c=0.9901; neg flagged 56/917) | 0.367 / 0.192 (c=0.9801; neg flagged 92/917) | 0.651 / 0.495 (c=0.8883; neg flagged 269/917) | 0.808 / 0.705 (c=0.5115; neg flagged 464/917) |
| S4c v4B1_frozen_L (DINOv3 frozen-L) | 0.170 / 0.090 (c=0.9677; neg flagged 32/917) | 0.280 / 0.154 (c=0.9407; neg flagged 79/917) | 0.459 / 0.286 (c=0.8680; neg flagged 150/917) | 0.664 / 0.471 (c=0.4983; neg flagged 370/917) | 0.822 / 0.678 (c=0.0558; neg flagged 642/917) |
| S4p v4B4_lora_L (DINOv3 LoRA-L) | 0.145 / 0.047 (c=0.9934; neg flagged 22/917) | 0.239 / 0.112 (c=0.9881; neg flagged 54/917) | 0.382 / 0.228 (c=0.9730; neg flagged 112/917) | 0.620 / 0.459 (c=0.8683; neg flagged 273/917) | 0.838 / 0.729 (c=0.1458; neg flagged 564/917) |

Sources:
- `reports/v4/test/matched_fp_sdistest.json`
- `reports/v4/test/yolo_test_at_preregistered_thresholds.json`
- `reports/v4/test/dino_test_at_preregistered_thresholds.json`


---

### T26. FINAL TEST — boreal AERIAL TEST (burn_ruokolahti, ONE burn, 9 DJI flights): recall at the pre-registered thresholds

Denominators: n_images=1,764 (positives 1,762, negatives 2) — ONE independent burn; with 2 negatives the false-alarm side is UNMEASURABLE ('neg flagged x/2' is anecdotal). Tiny recall is undefined (no tiny GT boxes) and omitted. SOURCE-ONLY = sdis-val conf applied unchanged; TARGET-ADAPTED = boreal-val (heinola) conf. 'n/a(val FP floor)' = the model never reached that FP on boreal val.

| system | SOURCE-ONLY sdis-val conf@5% | SOURCE-ONLY sdis-val conf@10% | SOURCE-ONLY sdis-val conf@20% | TARGET-ADAPTED boreal-val conf@1% | TARGET-ADAPTED boreal-val conf@5% |
|---|---|---|---|---|---|
| S0 expE (iteration-2 YOLO11m candidate) | 0.720 (c=0.7210; neg flagged 0/2) | 0.877 (c=0.6546; neg flagged 0/2) | 0.917 (c=0.5354; neg flagged 0/2) | 0.920 (c=0.4983; neg flagged 0/2) | 0.928 (c=0.2615; neg flagged 0/2) |
| S0b expB (deployed default, YOLO11s@1280) | 0.782 (c=0.7231; neg flagged 0/2) | 0.886 (c=0.6503; neg flagged 0/2) | 0.918 (c=0.5323; neg flagged 0/2) | 0.921 (c=0.4983; neg flagged 0/2) | 0.928 (c=0.3477; neg flagged 0/2) |
| S1 A_ctrl_E, median seed | 0.839 (c=0.7283; neg flagged 0/2) | 0.906 (c=0.6727; neg flagged 0/2) | 0.922 (c=0.5392; neg flagged 0/2) | 0.924 (c=0.4883; neg flagged 0/2) | 0.928 (c=0.2558; neg flagged 0/2) |
| S2 E_ET, median seed | 0.916 (c=0.7154; neg flagged 0/2) | 0.922 (c=0.6603; neg flagged 0/2) | 0.925 (c=0.5534; neg flagged 0/2) | 0.934 (c=0.0483; neg flagged 0/2) | n/a(val FP floor) |
| S2c C_EGT, median seed | 0.916 (c=0.7344; neg flagged 0/2) | 0.921 (c=0.6768; neg flagged 0/2) | 0.925 (c=0.5767; neg flagged 0/2) | 0.932 (c=0.2983; neg flagged 0/2) | 0.934 (c=0.0705; neg flagged 0/2) |
| S3 B_Gonly, seed 0 | 0.788 (c=0.4231; neg flagged 0/2) | 0.865 (c=0.3380; neg flagged 0/2) | 0.933 (c=0.1885; neg flagged 0/2) | 0.762 (c=0.4583; neg flagged 0/2) | 0.854 (c=0.3558; neg flagged 0/2) |
| S3 D_curr, median seed | 0.921 (c=0.7292; neg flagged 0/2) | 0.923 (c=0.6530; neg flagged 0/2) | 0.924 (c=0.5258; neg flagged 0/2) | 0.931 (c=0.1683; neg flagged 0/2) | 0.933 (c=0.1303; neg flagged 0/2) |
| S4 v4B6_lora_Hplus (DINOv3 LoRA-H+) | 0.928 (c=0.9939; neg flagged 0/2) | 0.929 (c=0.9901; neg flagged 0/2) | 0.931 (c=0.9801; neg flagged 0/2) | 0.934 (c=0.8883; neg flagged 0/2) | 0.934 (c=0.5115; neg flagged 0/2) |
| S4c v4B1_frozen_L (DINOv3 frozen-L) | 0.927 (c=0.9677; neg flagged 0/2) | 0.929 (c=0.9407; neg flagged 0/2) | 0.933 (c=0.8680; neg flagged 0/2) | 0.934 (c=0.4983; neg flagged 0/2) | 0.937 (c=0.0558; neg flagged 0/2) |
| S4p v4B4_lora_L (DINOv3 LoRA-L) | 0.931 (c=0.9934; neg flagged 0/2) | 0.932 (c=0.9881; neg flagged 0/2) | 0.933 (c=0.9730; neg flagged 0/2) | 0.934 (c=0.8683; neg flagged 0/2) | 0.935 (c=0.1458; neg flagged 1/2) |

Sources:
- `reports/v4/test/yolo_test_at_preregistered_thresholds.json`
- `reports/v4/test/dino_test_at_preregistered_thresholds.json`
- `reports/v4/LOCKBOX_V4_FROZEN.md (composition)`


---

### T27. FINAL TEST — AuSmoke FOREST TEST (new region, ACT Australia; 9 units / 4 independent events incl. ONE DJI aerial flight)

Denominators: n_images=2,871 (positives 2,871, negatives 0) — positives-only, so false alarms are UNMEASURABLE on this domain and matched-FP is undefined; recall-at-threshold only. Independent events: g_20241021, g_20240326_fb769, g_20240517_fb884 (ground burns) and g_dji_20230922 (one aerial flight, 3 videos, one of them a staged test-run reported separately). Never pool the 9 units as 9 events; the EVENT rows are frame-weighted over a group's units.

**(a) whole set — recall / tiny-recall at the pre-registered thresholds**

| system | SOURCE-ONLY sdis-val conf@5% | SOURCE-ONLY sdis-val conf@10% | SOURCE-ONLY sdis-val conf@20% | TARGET-ADAPTED boreal-val conf@1% | TARGET-ADAPTED boreal-val conf@5% |
|---|---|---|---|---|---|
| S0 expE (iteration-2 YOLO11m candidate) | 0.009 / 0.023 (c=0.7210) | 0.033 / 0.048 (c=0.6546) | 0.076 / 0.052 (c=0.5354) | 0.092 / 0.058 (c=0.4983) | 0.174 / 0.095 (c=0.2615) |
| S0b expB (deployed default, YOLO11s@1280) | 0.001 / 0.000 (c=0.7231) | 0.021 / 0.001 (c=0.6503) | 0.084 / 0.029 (c=0.5323) | 0.101 / 0.047 (c=0.4983) | 0.166 / 0.075 (c=0.3477) |
| S1 A_ctrl_E, median seed | 0.006 / 0.012 (c=0.7283) | 0.030 / 0.047 (c=0.6727) | 0.077 / 0.056 (c=0.5392) | 0.092 / 0.065 (c=0.4883) | 0.171 / 0.093 (c=0.2558) |
| S2 E_ET, median seed | 0.039 / 0.001 (c=0.7154) | 0.080 / 0.009 (c=0.6603) | 0.156 / 0.072 (c=0.5534) | 0.462 / 0.268 (c=0.0483) | n/a(val FP floor) |
| S2c C_EGT, median seed | 0.032 / 0.000 (c=0.7344) | 0.067 / 0.009 (c=0.6768) | 0.153 / 0.105 (c=0.5767) | 0.266 / 0.183 (c=0.2983) | 0.355 / 0.237 (c=0.0705) |
| S3 B_Gonly, seed 0 | 0.000 / 0.000 (c=0.4231) | 0.000 / 0.000 (c=0.3380) | 0.017 / 0.008 (c=0.1885) | 0.000 / 0.000 (c=0.4583) | 0.000 / 0.000 (c=0.3558) |
| S3 D_curr, median seed | 0.025 / 0.001 (c=0.7292) | 0.070 / 0.006 (c=0.6530) | 0.183 / 0.095 (c=0.5258) | 0.351 / 0.209 (c=0.1683) | 0.370 / 0.215 (c=0.1303) |
| S4 v4B6_lora_Hplus (DINOv3 LoRA-H+) | 0.145 / 0.068 (c=0.9939) | 0.210 / 0.101 (c=0.9901) | 0.280 / 0.131 (c=0.9801) | 0.416 / 0.180 (c=0.8883) | 0.508 / 0.222 (c=0.5115) |
| S4c v4B1_frozen_L (DINOv3 frozen-L) | 0.139 / 0.025 (c=0.9677) | 0.214 / 0.066 (c=0.9407) | 0.307 / 0.119 (c=0.8680) | 0.447 / 0.185 (c=0.4983) | 0.571 / 0.246 (c=0.0558) |
| S4p v4B4_lora_L (DINOv3 LoRA-L) | 0.109 / 0.038 (c=0.9934) | 0.180 / 0.079 (c=0.9881) | 0.281 / 0.139 (c=0.9730) | 0.440 / 0.207 (c=0.8683) | 0.570 / 0.252 (c=0.1458) |

**(b) per unit and per independent EVENT at the TARGET-ADAPTED boreal-val-1% confidence** (conf per system: S1_A_ctrl 0.4883, S2_E_ET 0.0483, S3_D_curr 0.1683, S0_expE 0.4983, S4_B6_loraH 0.8883)

| event (group) | unit | frames | S1_A_ctrl | S2_E_ET | S3_D_curr | S0_expE | S4_B6_loraH |
|---|---|---|---|---|---|---|---|
| g_20240326_fb769 | aus_27032024_S1020005 | 167 | 0.057 | 0.500 | 0.366 | 0.062 | 0.598 |
| g_20240326_fb769 | aus_FB769_Bonshaw_March_26_202403251112 | 530 | 0.372 | 0.812 | 0.673 | 0.316 | 0.650 |
| g_20240326_fb769 | aus_FB769_Bonshaw_March_26_202403261044_Panasonic | 566 | 0.043 | 0.723 | 0.500 | 0.045 | 0.574 |
| **g_20240326_fb769 — EVENT (frame-weighted, 3 units)** |  | 1263 | **0.183** | **0.731** | **0.555** | **0.161** | **0.609** |
| g_20240517_fb884 | aus_20240517_FB884_Sony4K_1300_MAH00006 | 404 | 0.002 | 0.038 | 0.014 | 0.000 | 0.072 |
| **g_20240517_fb884 — EVENT (frame-weighted, 1 unit)** |  | 404 | **0.002** | **0.038** | **0.014** | **0.000** | **0.072** |
| g_20241021 | aus_20241021_S1050004 | 670 | 0.003 | 0.579 | 0.387 | 0.004 | 0.554 |
| g_20241021 | aus_2024_10_21_06_1fps | 276 | 0.004 | 0.118 | 0.100 | 0.003 | 0.068 |
| **g_20241021 — EVENT (frame-weighted, 2 units)** |  | 946 | **0.003** | **0.445** | **0.303** | **0.004** | **0.412** |
| g_dji_20230922 (aerial flight) | aus_DJI_20230922112303_0008_D | 25 | 0.407 | 0.667 | 0.667 | 0.222 | 0.556 |
| g_dji_20230922 (aerial flight) | aus_DJI_20230922112554_0009_D | 143 | 0.685 | 1.000 | 1.000 | 0.944 | 1.000 |
| g_dji_20230922 (aerial flight) | aus_DJI_20230922115131_0004_D_TEST_RUN_SMOKE **[staged TEST_RUN]** | 90 | 0.086 | 0.796 | 0.806 | 0.129 | 0.946 |
| **g_dji_20230922 — EVENT (frame-weighted, 3 units)** |  | 258 | **0.449** | **0.896** | **0.900** | **0.590** | **0.938** |
| g_dji_20230922 — EVENT excl. staged TEST_RUN |  | 168 | 0.644 | 0.950 | 0.950 | 0.837 | 0.934 |

Sources:
- `reports/v4/test/yolo_test_at_preregistered_thresholds.json`
- `reports/v4/test/dino_test_at_preregistered_thresholds.json`
- `reports/v4/test/forest_test_per_unit.json`
- `data/v4/manifests/forest_holdout_v4.json (groups)`
- `data/v4/eval/EVALTEST_MANIFEST.json (aerial_event, staged_test_run)`


---

### T28. FINAL TEST roster: systems, checkpoints, selection basis, fixed thresholds, provenance

Once-guards: YOLO `.EVALUATED_ONCE` = 2026-09-15T03:32:30Z; DINO `.DINO_EVALUATED_ONCE` = DINO final test eval started 2026-09-15T02:31:44Z by tianrgua on ip-172-19-234-75.amazon.com | runs: v4B6_lora_Hplus v4B1_frozen_L v4B4_lora_L | thresholds: reports/v4/test/dino_test_thresholds.json | completed 2026-09-15T02:42:52Z failed=0. The DINO half was launched at 02:31Z by the pre-restart coordinator session with exactly the pre-registered systems/thresholds and was NOT re-run (STATUS.md 02:40Z, 03:00Z). D_curr is a POST-HOC addition: it was not a pre-registered candidate and its 3-seed median became available only after the test job launched; the shipped checkpoint coincidentally equals the median-seed pick (STATUS.md 03:00Z). Thresholds were fixed on VAL before any test prediction (reports/v4/test/dino_test_thresholds.json; yolo_test_at_preregistered_thresholds.json `thresholds`).

| id | system | checkpoint | selection basis (val) | fixed thresholds (conf) | role |
|---|---|---|---|---|---|
| S0 | expE (iteration-2 YOLO11m candidate) | `models/trained/expE_mix_y11m_1280_v2/seg05/best.pt` (iteration-2 expE YOLO11m) | deployment baseline; boreal/forest-val thresholds from reports/v4/eval_expE_{borealval,forestval}.json (v4evalr7) | sdis@5/10/20%: 0.721 / 0.6546 / 0.5354; boreal@1/5%: 0.4983 / 0.2615 | deployment baseline (pre-registered) |
| S0b | expB (deployed default, YOLO11s@1280) | `models/expB_best.pt` (byte-identical to models/trained/expB_mix_y11s_1280/seg03/best.pt; RECONCILIATION_V4.md) | deployed default; boreal/forest-val thresholds from reports/v4/eval_expB_{borealval,forestval}.json (v4evalr7) | sdis@5/10/20%: 0.7231 / 0.6503 / 0.5323; boreal@1/5%: 0.4983 / 0.3477 | deployment baseline (pre-registered) |
| S1 | A_ctrl_E, median seed | `v4a_s2` -> models/trained/expV4A_ctrl_s2/seg05/best.pt | median-seed rule: sdis-val R@10% 0.304 (n_seeds=3) | sdis@5/10/20%: 0.7283 / 0.6727 / 0.5392; boreal@1/5%: 0.4883 / 0.2558 | existing-data control (pre-registered) |
| S2 | E_ET, median seed | `v4e_s1` -> models/v4/runs/expV4E_et_s1/seg05/best.pt | median-seed rule: sdis-val R@10% 0.2913 (n_seeds=3); S2 rule -> E_ET (higher 3-seed mean boreal R@1%; tie <1pt -> E_ET (diff=-0.0041)) | sdis@5/10/20%: 0.7154 / 0.6603 / 0.5534; boreal@1/5%: 0.0483 / n/a (val FP floor) | best v4 data arm by S2 rule (pre-registered) |
| S2c | C_EGT, median seed | `v4c_s1` -> models/v4/runs/expV4C_egt_s1/seg05/best.pt | median-seed rule: sdis-val R@10% 0.262 (n_seeds=3) | sdis@5/10/20%: 0.7344 / 0.6768 / 0.5767; boreal@1/5%: 0.2983 / 0.0705 | matched control = the G-value comparison (pre-registered) |
| S3 | B_Gonly, seed 0 | `v4b` (seed 0) | only seed trained | sdis@5/10/20%: 0.4231 / 0.338 / 0.1885; boreal@1/5%: 0.4583 / 0.3558 | completeness (pre-registered) |
| S3 | D_curr, median seed | `v4dada_s1` -> models/v4/runs/expV4Dada_s1/seg03/best.pt | median-seed rule: sdis-val R@10% 0.318 (n_seeds=3) | sdis@5/10/20%: 0.7292 / 0.653 / 0.5258; boreal@1/5%: 0.1683 / 0.1303 | POST-HOC addition (not a pre-registered candidate) |
| S4 | v4B6_lora_Hplus (DINOv3 LoRA-H+) | `/scratch/scratch/tianrgua-sandbox/smoke/runs_v3/v4B6_lora_Hplus/best.pt` (--lora 16) | pre-registered pick (rule: best sdis-val matched-FP) | sdis@5/10/20%: 0.9939 / 0.9901 / 0.9801; boreal@1/5%: 0.8883 / 0.5115 | DINO rule pick: best sdis-val R@5% (pre-registered) |
| S4c | v4B1_frozen_L (DINOv3 frozen-L) | `/scratch/scratch/tianrgua-sandbox/smoke/runs_v3/v4B1_frozen_L/best.pt` (plain) | control (frozen backbone) | sdis@5/10/20%: 0.9677 / 0.9407 / 0.868; boreal@1/5%: 0.4983 / 0.0558 | DINO control (pre-registered) |
| S4p | v4B4_lora_L (DINOv3 LoRA-L) | `/scratch/scratch/tianrgua-sandbox/smoke/runs_v3/v4B4_lora_L/best.pt` (--lora 16) | post-hoc efficiency candidate (labelled post-hoc) | sdis@5/10/20%: 0.9934 / 0.9881 / 0.973; boreal@1/5%: 0.8683 / 0.1458 | POST-HOC efficiency candidate (labelled) |

Sources:
- `reports/v4/test/yolo_test_at_preregistered_thresholds.json`
- `reports/v4/test/dino_test_thresholds.json`
- `reports/v4/seed_summary.json (selection)`
- `reports/v4/replay/replay_{v4a_s2,v4e_s1,v4c_s1,v4dada_s1}_figlib39.json (weights)`
- `reports/v4/test/.EVALUATED_ONCE, .DINO_EVALUATED_ONCE`
- `STATUS.md 02:40Z/03:00Z/03:55Z`


---

### T29. AERIAL: boreal VAL (heinola) -> boreal TEST (ruokolahti), exact TEST checkpoints

VAL columns = recall at matched neg-image FP on heinola (n_images=1,123: 906 pos / 217 neg; 1% = ~2 negatives); TEST columns = recall at the pre-registered thresholds on ruokolahti (n_images=1,764: 1,762 pos / 2 neg; FP unmeasurable). Both are ONE burn each. GT box sizes — val: {'tiny': 1, 'small': 5, 'medium': 9, 'large': 902}; test: {'tiny': 0, 'small': 39, 'medium': 52, 'large': 1793} (eval JSON gt_by_size; 'large' = >5% of image area): both are large-plume sets. The delta column compares test recall at the boreal-val-1% conf with val recall at 1% FP (same threshold, different burn). expE/expB val recall recomputed from their boreal-val parquet with the frozen matching code (v4evalr7).

| system | VAL R@1% FP | VAL R@5% FP | TEST R @ sdis-val-10% conf | TEST R @ boreal-val-1% conf | TEST(ta1) - VAL(1%) | val source |
|---|---|---|---|---|---|---|
| S0 expE | 0.553 | 0.751 | 0.877 (c=0.6546; neg flagged 0/2) | 0.920 (c=0.4983; neg flagged 0/2) | +36.7 pt | reports/v4/eval_expE_borealval.raw.parquet |
| S0b expB (deployed) | 0.663 | 0.762 | 0.886 (c=0.6503; neg flagged 0/2) | 0.921 (c=0.4983; neg flagged 0/2) | +25.8 pt | reports/v4/eval_expB_borealval.raw.parquet |
| S1 A_ctrl (v4a_s2) | 0.584 | 0.795 | 0.906 (c=0.6727; neg flagged 0/2) | 0.924 (c=0.4883; neg flagged 0/2) | +34.0 pt | seed_summary.json |
| S2 E_ET (v4e_s1) | 0.955 | n/a (FP ceiling) | 0.922 (c=0.6603; neg flagged 0/2) | 0.934 (c=0.0483; neg flagged 0/2) | -2.1 pt | seed_summary.json |
| S2c C_EGT (v4c_s1) | 0.828 | 0.936 | 0.921 (c=0.6768; neg flagged 0/2) | 0.932 (c=0.2983; neg flagged 0/2) | +10.4 pt | seed_summary.json |
| S3 B_Gonly (v4b) | 0.194 | 0.466 | 0.865 (c=0.3380; neg flagged 0/2) | 0.762 (c=0.4583; neg flagged 0/2) | +56.8 pt | seed_summary.json |
| S3 D_curr (v4dada_s1, post-hoc) | 0.914 | 0.924 | 0.923 (c=0.6530; neg flagged 0/2) | 0.931 (c=0.1683; neg flagged 0/2) | +1.7 pt | seed_summary.json |
| S4 B6 LoRA-H+ | 0.964 | 0.972 | 0.929 (c=0.9901; neg flagged 0/2) | 0.934 (c=0.8883; neg flagged 0/2) | -3.0 pt | matched_fp_dino_boreal_val.json |
| S4c B1 frozen-L | 0.974 | 0.982 | 0.929 (c=0.9407; neg flagged 0/2) | 0.934 (c=0.4983; neg flagged 0/2) | -4.0 pt | matched_fp_dino_boreal_val.json |
| S4p B4 LoRA-L | 0.973 | 0.983 | 0.932 (c=0.9881; neg flagged 0/2) | 0.934 (c=0.8683; neg flagged 0/2) | -3.8 pt | matched_fp_dino_boreal_val.json |

Sources:
- `reports/v4/seed_summary.json`
- `reports/v4/matched_fp_dino_boreal_val.json`
- `reports/v4/eval_exp{E,B}_borealval.raw.parquet`
- `reports/v4/test/yolo_test_at_preregistered_thresholds.json`
- `reports/v4/test/dino_test_at_preregistered_thresholds.json`
- `reports/v4/eval_v4a_s2_borealval.json, reports/v4/test/eval_S1_A_ctrl_borealtest.json (gt_by_size)`
