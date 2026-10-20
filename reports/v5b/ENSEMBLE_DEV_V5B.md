# V5b — Model-collaboration (ensemble) study on pyro-sdis VAL, from saved predictions

Date: 2026-10-02 (V5b phase 1). Author: ensemble agent (one of ten parallel V5b agents).
Inputs: SAVED per-image predictions only — no inference, no GPU, no cluster submission
(registry not touched). TEST sites, lockboxes, FIgLib, Boreal ruokolahti, AuSmoke test:
never read. Nothing under reports/v4, reports/v5 or STATUS.md edited.

## 0. Headline

* **Fusing YOLO11m (arm A) with the DINOv3 detectors (B4 LoRA-L, B6 LoRA-H+, B1 frozen-L)
  by calibrated-score WBF (`conf_type=max`, `iou_thr=0.3`) raises frame recall at matched
  negative-image FP by +2.8 pt on average over the two VAL sites (fold-averaged OOF mean recall
  over FP 1/2/5/10/20%: 0.354 vs 0.326 for the best single model B6; vs A 0.322, RF-DETR-L 0.314).**
  Gains are site-dependent: held-out **marguerite** +3.9 pt vs B6 (95% bootstrap CI [+1.0, +8.3],
  P(gain<=0)=0.004); held-out **cabanelle** +1.8 pt vs B6 (CI [-0.9, +4.0]) but **-1.4 pt vs
  RF-DETR-L alone** (CI [-4.2, +1.8]), which is the strongest single model on that site at 5-20% FP.
* Tiny-smoke recall (<0.1% area) does **not** improve uniformly: on cabanelle the fusion is below
  B6 at 5-10% FP (0.066/0.121 vs 0.091/0.176); on marguerite it is above B6 at 5-20%
  (0.390/0.535/0.714 vs 0.324/0.449/0.659) and below at 1-2%.
* OR/union+NMS with calibrated scores and WBF are within 0.2 pt of each other for the 3- and
  4-model sets (`conf_type=max` makes WBF = union with averaged coordinates). Agreement weighting
  (`avg`-type conf_types, which discount a box found by one of M models by 1/M) **helps the 2-model
  pair** A+B4 at 1-5% FP (+1.4 to +2.8 pt vs union on both held-out sites, T3) but **loses with
  >= 3 models** (in-sample 4-model set: max 0.224/0.496 vs avg 0.226/0.466 cab/mar; with RF-DETR
  in the set max wins on both sites) — with more models the uniform discount removes too much of
  the recall carried by boxes that only one model finds at tight FP budgets.
* Model addition protocol (keep if >= +1.0 pt macro-OOF): B6 kept (+1.16), RF-DETR-L rejected
  (-0.39), expE rejected (+0.06, same architecture/data as A), B1 kept (+1.34). Both "kept"
  margins are inside their bootstrap CIs (T6) — the +1 pt rule was pre-stated and is applied as
  stated, but a 3-model "lean" recipe (A+B4+B6, same parallel latency) is the conservative choice.
* A small learned fusion (logistic regression on per-box calibrated scores, log-FP, agreement,
  size, aspect) reaches 0.358 macro-OOF = **+0.44 pt over WBF — below the 1 pt bar, not adopted**;
  a tiny GBM is worse (0.347) and unstable across sites.
* **Calibration does not transfer between sites.** A threshold set for 5% neg-image FP on
  marguerite realises 19-27% FP on cabanelle (every model and the fusion); set on cabanelle it
  realises 0-1% on marguerite (T8). Consequently any pooled-over-sites number for a calibrated
  system is pessimistic (pooled single-model OOF-calibrated B4 0.167 vs raw 0.242), so **all
  comparisons in this report are made within the held-out site**, and the recipe states that a
  deployed camera must have its operating threshold set on its own verified negatives.
* Image-level alerting (max over fused boxes = max of calibrated per-model max scores) is
  +2.1 pt (cabanelle, mean 0.266 vs B6 0.245; RF-DETR-L alone 0.283) and +4.0 pt (marguerite,
  0.515 vs 0.475) over the best single model; noisy-OR is +0.85 pt macro over max — under the bar,
  max kept for consistency with the box thresholds.
* FULL-FRAME PARALLEL latency of the recipe = max(component) + fusion = **352.0 ms** per frame
  (B6 351.6 + 0.4 ms fusion), i.e. ~31x the YOLO11m FAST path (11.2 ms); serial on one GPU
  677.3 ms. Selective execution (cheap gate -> expensive models) is a later test, not measured.

## 1. Setup and discipline

* **Data**: pyro-sdis VAL, 7,335 images (6,177 positive / 1,158 GT-negative), 6,603 GT boxes
  (2,277 tiny), identical `image`/`gt` columns in all six parquets (verified byte-equal GT), boxes
  `[x1n,y1n,x2n,y2n,score]`, saved with conf >= 0.02 (DINO coordinates up to 1.02 clipped to [0,1]).
* **Folds by SITE** (image paths `force-06_cabanelle-*` / `sdis-07_marguerite-*`): T1. Every
  calibration map, fusion hyper-parameter, learned-fusion weight and image-level combiner is fit on
  one site and evaluated on the other; "OOF" below always means that. Per-site single-model numbers
  need no fitting (recall at matched FP is invariant to a monotone transform of one model's score),
  so single-vs-fusion comparisons within a held-out site are fair and, if anything, pessimistic for
  the fusion.
* **Matching**: `src.eval.eval_detection.sweep_report` semantics (greedy by score, IoU >= 0.4, size
  buckets, neg-image FP = fraction of GT-negative images with any box above threshold), re-implemented
  as a one-pass matcher (a box's TP/FP status does not depend on lower-scored boxes) and verified to
  reproduce `sweep_report` exactly; matched-FP interpolation = `scripts/v4/compare_matched_fp.interp_at_fp`
  on a 1,300-point threshold grid (0.001 steps, 0.00005 above 0.99). Single-model T2 reproduces
  FINAL_REPORT_V4 T09 within grid resolution (B4 at 5%: 0.208/0.142 here vs 0.210/0.144).
* **Calibration**: per model, raw score -> mid-rank ECDF of the fitting site's negative-image MAX
  scores (`scripts/v5b/fusion.ScoreCalibrator`), so 1-c = that model's single-model neg-image FP;
  scores above the worst negative keep their order (interpolated to 1.0) because the DINO heads
  saturate >= 0.99. Deployed recipe maps are refit on all 1,158 VAL negatives (740-898 knots).
* **Fusions**: (2) union of calibrated boxes + NMS (iou 0.3-0.7); (3) WBF — `ensemble-boxes` 1.0.9
  (MIT) installed into the venv; the recipe uses a numpy port in `scripts/v5b/fusion.py` verified
  against it to 1e-7 for all four `conf_type`s (unit test) — grid iou_thr {0.3,0.4,0.5,0.6,0.7} x
  skip_box_thr {0,0.5,0.8,0.9} x conf_type {avg,max,box_and_model_avg,absent_model_aware_avg} = 80
  configs, selected on the fitting site by mean recall over FP 1/2/5/10/20%, evaluated on the other;
  (4) learned fusion on union anchors (IoU 0.5): features = per-model calibrated max, raw max,
  -log10(1-c), max/sum log-FP, agreement count, mean/max c, log area, log aspect, tiny flag;
  label = anchor IoU >= 0.4 with any GT; LR (standardised, C=1) and HistGradientBoosting (300 it,
  depth 3, leaf 50). Image-level: max / mean / noisy-OR / LR over per-model calibrated max scores,
  recall at matched neg-image FP computed exactly (threshold = (k+1)-th largest negative score).
* **Selection statistic**: fold-averaged (macro) OOF mean recall over the five FP levels; the +1 pt
  model-addition rule and the "learned must beat WBF" rule use it. Pooled-OOF is reported (T4, T7b)
  but not used because of the cross-site calibration offset (above).
* **Noise**: 1% FP = 7.0 negative images on cabanelle and 4.6 on marguerite; bootstrap CIs (T6)
  resample images within the held-out site (500 draws). The sdis GT is known to omit visible smoke
  on some "negative" frames (STATUS: "sdis flicker = label omission"; e.g. the first VAL image,
  GT-empty, is fired at 0.98 by every model) — such frames count as FP for every system alike.
* Labels used: author labels (pyro-sdis). No human review, no model-assisted labels, no label
  conversion anywhere in this study.

## 2. Tables

### T1. pyro-sdis VAL split by site (grouped folds)

| fold | site | images | negative images | GT boxes | tiny GT (<0.1% area) | 1% FP = n neg images |
|---|---|---|---|---|---|---|
| 0 | cabanelle | 2,701 | 698 | 2,252 | 592 | 7.0 |
| 1 | marguerite | 4,634 | 460 | 4,351 | 1,685 | 4.6 |
| all | both | 7,335 | 1,158 | 6,603 | 2,277 | 11.6 |

### T2. Single models, own raw confidence, whole VAL (recall / tiny-recall at matched neg-image FP; IoU>=0.4)

| system | 1% | 2% | 5% | 10% | 20% | mean |
|---|---|---|---|---|---|---|
| A (yolo_A) | 0.056 / 0.014 | 0.100 / 0.041 | 0.200 / 0.110 | 0.302 / 0.194 | 0.493 / 0.373 | **0.230** |
| B4 (dino_B4) | 0.083 / 0.048 | 0.123 / 0.080 | 0.208 / 0.142 | 0.309 / 0.216 | 0.486 / 0.400 | **0.242** |
| B6 (dino_B6) | 0.103 / 0.040 | 0.133 / 0.062 | 0.228 / 0.145 | 0.340 / 0.253 | 0.495 / 0.416 | **0.260** |
| RF (rfdetr_L) | 0.053 / 0.008 | 0.102 / 0.030 | 0.201 / 0.098 | 0.314 / 0.194 | 0.486 / 0.364 | **0.232** |
| expE (yolo_expE) | 0.058 / 0.012 | 0.092 / 0.026 | 0.188 / 0.094 | 0.308 / 0.200 | 0.483 / 0.366 | **0.226** |
| B1 (dino_B1) | 0.065 / 0.031 | 0.095 / 0.044 | 0.176 / 0.104 | 0.299 / 0.198 | 0.465 / 0.341 | **0.220** |

### T3-cab. Held-out site = **cabanelle** — single models (raw conf) vs fusions fit on the other site (OOF)

| system | 1% | 2% | 5% | 10% | 20% | mean |
|---|---|---|---|---|---|---|
| single A | 0.059 / 0.010 | 0.084 / 0.015 | 0.166 / 0.051 | 0.288 / 0.125 | 0.439 / 0.241 | **0.207** |
| single B4 | 0.065 / 0.027 | 0.101 / 0.052 | 0.146 / 0.096 | 0.228 / 0.142 | 0.360 / 0.246 | **0.180** |
| single B6 | 0.070 / 0.019 | 0.114 / 0.033 | 0.177 / 0.091 | 0.286 / 0.176 | 0.420 / 0.309 | **0.213** |
| single RF | 0.079 / 0.002 | 0.122 / 0.022 | 0.222 / 0.064 | 0.323 / 0.142 | 0.474 / 0.295 | **0.244** |
| single expE | 0.053 / 0.005 | 0.091 / 0.012 | 0.173 / 0.041 | 0.286 / 0.112 | 0.437 / 0.242 | **0.208** |
| single B1 | 0.064 / 0.023 | 0.088 / 0.032 | 0.156 / 0.064 | 0.247 / 0.126 | 0.405 / 0.269 | **0.192** |
| OR/union+NMS A+B4 | 0.059 / 0.010 | 0.086 / 0.016 | 0.167 / 0.051 | 0.302 / 0.155 | 0.432 / 0.260 | **0.209** <br>`iou0.7` |
| WBF A+B4 | 0.073 / 0.025 | 0.091 / 0.032 | 0.174 / 0.069 | 0.254 / 0.135 | 0.429 / 0.277 | **0.204** <br>`avg,iou0.4,skip0.8` |
| OR/union+NMS A+B4+B6 | 0.074 / 0.019 | 0.102 / 0.029 | 0.186 / 0.072 | 0.297 / 0.139 | 0.439 / 0.283 | **0.220** <br>`iou0.7` |
| WBF A+B4+B6 | 0.074 / 0.019 | 0.102 / 0.029 | 0.186 / 0.072 | 0.298 / 0.139 | 0.443 / 0.281 | **0.221** <br>`max,iou0.3,skip0.0` |
| OR/union+NMS A+B4+B6+RF | 0.074 / 0.019 | 0.102 / 0.029 | 0.186 / 0.072 | 0.299 / 0.137 | 0.459 / 0.290 | **0.224** <br>`iou0.7` |
| WBF A+B4+B6+RF | 0.074 / 0.019 | 0.102 / 0.029 | 0.187 / 0.072 | 0.300 / 0.137 | 0.462 / 0.290 | **0.225** <br>`max,iou0.3,skip0.0` |
| OR/union+NMS A+B4+B6+expE | 0.074 / 0.019 | 0.102 / 0.029 | 0.186 / 0.072 | 0.299 / 0.139 | 0.444 / 0.272 | **0.221** <br>`iou0.7` |
| WBF A+B4+B6+expE | 0.074 / 0.019 | 0.102 / 0.029 | 0.187 / 0.072 | 0.300 / 0.139 | 0.449 / 0.272 | **0.222** <br>`max,iou0.3,skip0.0` |
| OR/union+NMS A+B4+B6+B1 | 0.089 / 0.027 | 0.118 / 0.035 | 0.193 / 0.066 | 0.297 / 0.122 | 0.452 / 0.280 | **0.230** <br>`iou0.7` |
| WBF A+B4+B6+B1 | 0.089 / 0.027 | 0.118 / 0.035 | 0.194 / 0.066 | 0.296 / 0.121 | 0.454 / 0.280 | **0.230** <br>`max,iou0.3,skip0.5` |
| learned logreg A+B4+B6+B1 | 0.075 / 0.003 | 0.097 / 0.015 | 0.201 / 0.071 | 0.287 / 0.136 | 0.455 / 0.254 | **0.223** |
| learned hgb A+B4+B6+B1 | 0.056 / 0.017 | 0.109 / 0.039 | 0.189 / 0.091 | 0.295 / 0.177 | 0.451 / 0.284 | **0.220** |

### T3-mar. Held-out site = **marguerite** — single models (raw conf) vs fusions fit on the other site (OOF)

| system | 1% | 2% | 5% | 10% | 20% | mean |
|---|---|---|---|---|---|---|
| single A | 0.179 / 0.120 | 0.244 / 0.177 | 0.440 / 0.358 | 0.564 / 0.493 | 0.751 / 0.697 | **0.436** |
| single B4 | 0.131 / 0.087 | 0.234 / 0.164 | 0.373 / 0.294 | 0.533 / 0.456 | 0.714 / 0.642 | **0.397** |
| single B6 | 0.237 / 0.165 | 0.332 / 0.258 | 0.398 / 0.324 | 0.516 / 0.449 | 0.708 / 0.659 | **0.438** |
| single RF | 0.116 / 0.062 | 0.268 / 0.187 | 0.356 / 0.269 | 0.514 / 0.425 | 0.661 / 0.596 | **0.383** |
| single expE | 0.207 / 0.147 | 0.253 / 0.193 | 0.405 / 0.331 | 0.570 / 0.499 | 0.740 / 0.685 | **0.435** |
| single B1 | 0.162 / 0.108 | 0.227 / 0.153 | 0.353 / 0.262 | 0.576 / 0.475 | 0.701 / 0.607 | **0.404** |
| OR/union+NMS A+B4 | 0.169 / 0.105 | 0.281 / 0.191 | 0.410 / 0.298 | 0.568 / 0.475 | 0.741 / 0.674 | **0.434** <br>`iou0.3` |
| WBF A+B4 | 0.197 / 0.139 | 0.295 / 0.220 | 0.460 / 0.377 | 0.570 / 0.491 | 0.741 / 0.668 | **0.453** <br>`avg,iou0.6,skip0.5` |
| OR/union+NMS A+B4+B6 | 0.208 / 0.128 | 0.316 / 0.225 | 0.450 / 0.350 | 0.573 / 0.481 | 0.741 / 0.680 | **0.458** <br>`iou0.3` |
| WBF A+B4+B6 | 0.208 / 0.128 | 0.317 / 0.225 | 0.451 / 0.350 | 0.576 / 0.482 | 0.746 / 0.682 | **0.460** <br>`max,iou0.4,skip0.0` |
| OR/union+NMS A+B4+B6+RF | 0.219 / 0.132 | 0.265 / 0.172 | 0.429 / 0.321 | 0.566 / 0.474 | 0.733 / 0.668 | **0.443** <br>`iou0.3` |
| WBF A+B4+B6+RF | 0.220 / 0.132 | 0.268 / 0.172 | 0.434 / 0.321 | 0.572 / 0.474 | 0.743 / 0.673 | **0.447** <br>`max,iou0.3,skip0.0` |
| OR/union+NMS A+B4+B6+expE | 0.208 / 0.125 | 0.314 / 0.223 | 0.447 / 0.348 | 0.564 / 0.467 | 0.742 / 0.676 | **0.455** <br>`iou0.3` |
| WBF A+B4+B6+expE | 0.215 / 0.143 | 0.285 / 0.212 | 0.455 / 0.375 | 0.580 / 0.501 | 0.760 / 0.700 | **0.459** <br>`avg,iou0.6,skip0.5` |
| OR/union+NMS A+B4+B6+B1 | 0.240 / 0.147 | 0.330 / 0.234 | 0.457 / 0.352 | 0.598 / 0.504 | 0.741 / 0.670 | **0.473** <br>`iou0.3` |
| WBF A+B4+B6+B1 | 0.203 / 0.134 | 0.281 / 0.200 | 0.491 / 0.390 | 0.629 / 0.535 | 0.780 / 0.714 | **0.477** <br>`avg,iou0.3,skip0.5` |
| learned logreg A+B4+B6+B1 | 0.232 / 0.098 | 0.349 / 0.201 | 0.466 / 0.329 | 0.634 / 0.530 | 0.786 / 0.715 | **0.493** |
| learned hgb A+B4+B6+B1 | 0.259 / 0.101 | 0.347 / 0.181 | 0.430 / 0.287 | 0.571 / 0.481 | 0.758 / 0.726 | **0.473** |

### T4. Fold-averaged (macro) OOF mean recall — the selection statistic — and pooled OOF

| model set | union macro | WBF macro | union pooled | WBF pooled | WBF params chosen on all VAL |
|---|---|---|---|---|---|
| A+B4 | 0.3216 | 0.3285 | 0.1609 | 0.1421 | `avg,iou0.4,skip0.8` |
| A+B4+B6 | 0.3386 | 0.3401 | 0.1724 | 0.1731 | `max,iou0.3,skip0.0` |
| A+B4+B6+RF | 0.3333 | 0.3362 | 0.1742 | 0.1750 | `max,iou0.3,skip0.0` |
| A+B4+B6+expE | 0.3380 | 0.3407 | 0.1700 | 0.1184 | `max,iou0.3,skip0.0` |
| A+B4+B6+B1 | 0.3513 | 0.3535 | 0.1715 | 0.1110 | `max,iou0.3,skip0.0` |
| *single models (macro of per-site raw)* | A 0.3213, B4 0.2885, B6 0.3258, RF 0.3136, expE 0.3215, B1 0.2980 | | | | |
| learned logreg (A+B4+B6+B1) | | 0.3581 | | 0.1410 | |
| learned hgb (A+B4+B6+B1) | | 0.3465 | | 0.1327 | |
| learned logreg (all 6) | | 0.3508 | | 0.1279 | |
| learned hgb (all 6) | | 0.3292 | | 0.1142 | |

### T5. Model-addition protocol (keep if >= +1.0 pt macro-OOF mean recall over current best)

| candidate added to | union macro | WBF macro | gain (pt) | kept |
|---|---|---|---|---|
| A+B4 + **B6** | 0.3386 | 0.3401 | +1.16 | yes |
| A+B4+B6 + **RF** | 0.3333 | 0.3362 | -0.39 | no |
| A+B4+B6 + **expE** | 0.3380 | 0.3407 | +0.06 | no |
| A+B4+B6 + **B1** | 0.3513 | 0.3535 | +1.34 | yes |

### T6. Bootstrap (500 image resamples within the held-out site): difference in OOF mean recall, pts

| held-out site | comparison | point | 95% CI | P(gain <= 0) |
|---|---|---|---|---|
| cabanelle | ACCURATE fusion (A+B4+B6+B1 WBF)_vs_rfdetr_L | -1.43 | [-4.21, +1.76] | 0.770 |
| cabanelle | ACCURATE fusion (A+B4+B6+B1 WBF)_vs_dino_B6 | +1.77 | [-0.90, +3.97] | 0.086 |
| cabanelle | ACCURATE fusion (A+B4+B6+B1 WBF)_vs_yolo_A | +2.33 | [-0.24, +4.69] | 0.042 |
| cabanelle | ACCURATE fusion (A+B4+B6+B1 WBF)_vs_lean(A+B4+B6 wbf) | +1.08 | [-1.12, +3.01] | 0.180 |
| cabanelle | lean(A+B4+B6 wbf)_vs_dino_B6 | +0.69 | [-0.91, +2.22] | 0.194 |
| marguerite | ACCURATE fusion (A+B4+B6+B1 WBF)_vs_dino_B6 | +3.86 | [+1.00, +8.34] | 0.004 |
| marguerite | ACCURATE fusion (A+B4+B6+B1 WBF)_vs_yolo_A | +4.12 | [-0.90, +8.48] | 0.064 |
| marguerite | ACCURATE fusion (A+B4+B6+B1 WBF)_vs_lean(A+B4+B6 wbf) | +1.73 | [-1.60, +7.16] | 0.136 |
| marguerite | lean(A+B4+B6 wbf)_vs_dino_B6 | +2.13 | [-3.65, +6.11] | 0.206 |

### T7. Image-level alert score (max over boxes) — recall of positive images at matched neg-image FP, WITHIN each site (thresholds per site)

| method | site | 1% | 2% | 5% | 10% | 20% | mean |
|---|---|---|---|---|---|---|---|
| single:yolo_A | cabanelle | 0.062 | 0.097 | 0.192 | 0.337 | 0.510 | **0.240** |
| single:yolo_A | marguerite | 0.188 | 0.259 | 0.476 | 0.616 | 0.822 | **0.472** |
| single:dino_B4 | cabanelle | 0.074 | 0.116 | 0.172 | 0.272 | 0.426 | **0.212** |
| single:dino_B4 | marguerite | 0.138 | 0.249 | 0.400 | 0.582 | 0.796 | **0.433** |
| single:dino_B6 | cabanelle | 0.070 | 0.128 | 0.200 | 0.337 | 0.490 | **0.245** |
| single:dino_B6 | marguerite | 0.253 | 0.355 | 0.431 | 0.559 | 0.777 | **0.475** |
| single:rfdetr_L | cabanelle | 0.090 | 0.138 | 0.256 | 0.377 | 0.553 | **0.283** |
| single:rfdetr_L | marguerite | 0.123 | 0.288 | 0.386 | 0.557 | 0.731 | **0.417** |
| max_cal[chosen] | cabanelle | 0.098 | 0.132 | 0.221 | 0.347 | 0.530 | **0.266** |
| max_cal[chosen] | marguerite | 0.255 | 0.358 | 0.493 | 0.655 | 0.815 | **0.515** |
| mean_cal[chosen] | cabanelle | 0.080 | 0.108 | 0.211 | 0.309 | 0.535 | **0.249** |
| mean_cal[chosen] | marguerite | 0.225 | 0.326 | 0.529 | 0.680 | 0.850 | **0.522** |
| noisy_or[chosen] | cabanelle | 0.094 | 0.131 | 0.245 | 0.348 | 0.525 | **0.269** |
| noisy_or[chosen] | marguerite | 0.252 | 0.345 | 0.525 | 0.679 | 0.844 | **0.529** |
| logreg_cal[chosen] | cabanelle | 0.083 | 0.122 | 0.207 | 0.310 | 0.513 | **0.247** |
| logreg_cal[chosen] | marguerite | 0.260 | 0.318 | 0.549 | 0.702 | 0.857 | **0.537** |
| max_cal[all6] | cabanelle | 0.098 | 0.132 | 0.222 | 0.348 | 0.547 | **0.269** |
| max_cal[all6] | marguerite | 0.266 | 0.305 | 0.467 | 0.635 | 0.804 | **0.496** |
| mean_cal[all6] | cabanelle | 0.069 | 0.137 | 0.220 | 0.354 | 0.537 | **0.263** |
| mean_cal[all6] | marguerite | 0.248 | 0.369 | 0.544 | 0.706 | 0.853 | **0.544** |
| noisy_or[all6] | cabanelle | 0.076 | 0.129 | 0.249 | 0.357 | 0.524 | **0.267** |
| noisy_or[all6] | marguerite | 0.266 | 0.377 | 0.516 | 0.672 | 0.857 | **0.537** |
| logreg_cal[all6] | cabanelle | 0.078 | 0.136 | 0.217 | 0.320 | 0.514 | **0.253** |
| logreg_cal[all6] | marguerite | 0.265 | 0.366 | 0.505 | 0.644 | 0.855 | **0.527** |
| fused_box_max[chosen,wbf] | cabanelle | 0.098 | 0.132 | 0.221 | 0.347 | 0.530 | **0.266** |
| fused_box_max[chosen,wbf] | marguerite | 0.255 | 0.358 | 0.493 | 0.655 | 0.815 | **0.515** |

### T7b. Image-level, POOLED over sites with one threshold (single models raw; fusions OOF-calibrated)

| method | 1% | 2% | 5% | 10% | 20% | mean |
|---|---|---|---|---|---|---|
| single:yolo_A | 0.059 | 0.109 | 0.218 | 0.331 | 0.547 | **0.253** |
| single:dino_B4 | 0.088 | 0.133 | 0.229 | 0.344 | 0.542 | **0.267** |
| single:dino_B6 | 0.113 | 0.147 | 0.251 | 0.374 | 0.549 | **0.287** |
| single:rfdetr_L | 0.059 | 0.113 | 0.221 | 0.346 | 0.541 | **0.256** |
| single:yolo_expE | 0.062 | 0.100 | 0.205 | 0.341 | 0.537 | **0.249** |
| single:dino_B1 | 0.070 | 0.108 | 0.197 | 0.332 | 0.533 | **0.248** |
| max_cal[chosen] | 0.061 | 0.099 | 0.141 | 0.238 | 0.402 | **0.188** |
| mean_cal[chosen] | 0.037 | 0.057 | 0.111 | 0.189 | 0.357 | **0.150** |
| noisy_or[chosen] | 0.043 | 0.071 | 0.134 | 0.211 | 0.383 | **0.168** |
| logreg_cal[chosen] | 0.037 | 0.050 | 0.087 | 0.145 | 0.231 | **0.110** |
| max_cal[all6] | 0.061 | 0.100 | 0.143 | 0.218 | 0.390 | **0.182** |
| mean_cal[all6] | 0.038 | 0.065 | 0.116 | 0.191 | 0.344 | **0.151** |
| noisy_or[all6] | 0.049 | 0.071 | 0.138 | 0.207 | 0.373 | **0.168** |
| logreg_cal[all6] | 0.040 | 0.054 | 0.091 | 0.148 | 0.235 | **0.114** |
| fused_box_max[chosen,wbf] | 0.061 | 0.099 | 0.141 | 0.238 | 0.402 | **0.188** |

### T8. Threshold transfer between sites: threshold set for target FP on one site -> realized neg-image FP on the other

| system | direction | @1% | @2% | @5% | @10% | @20% |
|---|---|---|---|---|---|---|
| A | set on cabanelle -> realized on marguerite | 0.000 | 0.000 | 0.000 | 0.011 | 0.028 |
| A | set on marguerite -> realized on cabanelle | 0.085 | 0.133 | 0.271 | 0.383 | 0.582 |
| B4 | set on cabanelle -> realized on marguerite | 0.006 | 0.006 | 0.011 | 0.024 | 0.057 |
| B4 | set on marguerite -> realized on cabanelle | 0.034 | 0.087 | 0.188 | 0.331 | 0.497 |
| B6 | set on cabanelle -> realized on marguerite | 0.000 | 0.000 | 0.002 | 0.013 | 0.050 |
| B6 | set on marguerite -> realized on cabanelle | 0.086 | 0.153 | 0.193 | 0.284 | 0.484 |
| RF | set on cabanelle -> realized on marguerite | 0.002 | 0.002 | 0.011 | 0.015 | 0.050 |
| RF | set on marguerite -> realized on cabanelle | 0.050 | 0.148 | 0.206 | 0.338 | 0.463 |
| expE | set on cabanelle -> realized on marguerite | 0.000 | 0.000 | 0.004 | 0.009 | 0.039 |
| expE | set on marguerite -> realized on cabanelle | 0.109 | 0.135 | 0.256 | 0.394 | 0.569 |
| B1 | set on cabanelle -> realized on marguerite | 0.000 | 0.000 | 0.002 | 0.017 | 0.041 |
| B1 | set on marguerite -> realized on cabanelle | 0.079 | 0.115 | 0.212 | 0.433 | 0.609 |
| fusion[chosen, all-VAL calibration] | set on cabanelle -> realized on marguerite | 0.000 | 0.004 | 0.006 | 0.011 | 0.043 |
| fusion[chosen, all-VAL calibration] | set on marguerite -> realized on cabanelle | 0.099 | 0.155 | 0.246 | 0.354 | 0.539 |

### T9. ACCURATE-mode latency, batch-1 1x L40S fp32 @1280 (component numbers from reports; fusion measured here)

| component | ms | source |
|---|---|---|
| yolo_A | 11.2 | models/v3/evals/yolo_latency/latency_gpu.json (expE weights, same arch) |
| dino_B4 | 157.5 | FINAL_REPORT_V4 T13, LoRA merged |
| dino_B6 | 351.6 | FINAL_REPORT_V4 T13, LoRA merged |
| dino_B1 | 156.6 | FINAL_REPORT_V4 T13 (v3 B1 frozen reference) |
| fusion (calibration + WBF, numpy, CPU) | 0.381 median / 0.653 p90 / 1.403 max | measured over 7,335 VAL images, devbox CPU |
| **FULL-FRAME PARALLEL = max(components) + fusion** | **352.0** | one GPU per component, all run on every frame |
| serial (one GPU, components back to back) + fusion | 677.3 | |
| not-yet-tested: selective execution (cheap model gates expensive ones) | — | later test |

### T10. Recommended ACCURATE recipe (models/v5b/fusion/accurate_recipe.json)

| field | value |
|---|---|
| models | yolo_A, dino_B4, dino_B6, dino_B1 |
| fusion | {'method': 'wbf', 'conf_type': 'max', 'iou_thr': 0.3, 'skip_box_thr': 0.0, 'weights': {'yolo_A': 1.0, 'dino_B4': 1.0, 'dino_B6': 1.0, 'dino_B1': 1.0}} |
| calibration | per-model monotone map raw -> neg-image-FP quantile, refit on all 1,158 VAL negatives (740, 898, 886, 872 knots) |
| raw score at calibrated 0.95 / 0.99 | yolo_A: 0.7284 / 0.8087; dino_B4: 0.9935 / 0.9971; dino_B6: 0.9939 / 0.9968; dino_B1: 0.9674 / 0.9858 |
| fused-score thresholds (pooled OOF) | {'fp0.01': 0.9994, 'fp0.02': 0.9993, 'fp0.05': 0.999, 'fp0.10': 0.9918, 'fp0.20': 0.9657} |
| fused-score thresholds by held-out site | {'cabanelle': {'fp0.01': 0.9995, 'fp0.02': 0.9994, 'fp0.05': 0.9992, 'fp0.10': 0.999, 'fp0.20': 0.9871}, 'marguerite': {'fp0.01': 0.8534, 'fp0.02': 0.8078, 'fp0.05': 0.658, 'fp0.10': 0.505, 'fp0.20': 0.212}} |
| image-level | max_calibrated (selected by max_cal[chosen]); thresholds {'fp0.01': 0.9994, 'fp0.02': 0.9993, 'fp0.05': 0.999, 'fp0.10': 0.9929, 'fp0.20': 0.968} |
| in-sample check (all-VAL calibration, whole VAL) recall/tiny | 0.01: 0.082 / 0.027, 0.02: 0.138 / 0.065, 0.05: 0.234 / 0.131, 0.10: 0.337 / 0.218, 0.20: 0.532 / 0.407 |

## 3. Recommended ACCURATE-mode recipe and how to use it

`models/v5b/fusion/accurate_recipe.json` (T10): models yolo_A + dino_B4 + dino_B6 + dino_B1, all at
imgsz 1280 on the full frame; per-model calibration knots; WBF `conf_type=max, iou_thr=0.3,
skip_box_thr=0.0`, equal weights; fused-score thresholds (pooled OOF and per held-out site);
image-level = `max_calibrated`. Conservative alternative with the same 352 ms parallel latency:
drop dino_B1 (`A+B4+B6`, macro-OOF 0.340; its +1.3 pt is inside the CI).

```python
from scripts.v5b.fusion import fuse, image_score, load_recipe
rec = load_recipe()                                   # or path / $SMOKE_FUSION_RECIPE
boxes = fuse({"yolo_A": yolo_boxes, "dino_B4": b4_boxes, "dino_B6": b6_boxes, "dino_B1": b1_boxes}, rec,
             min_score=rec.threshold_for_fp(0.05))    # -> [[x1n,y1n,x2n,y2n,calibrated_score], ...]
alert = image_score({...}, rec) >= rec.image_level["thresholds_oof"]["fp0.05"]
```
Inputs are normalised xyxy + raw score per model (the serving `Detection` is pixel xyxy — divide by
W,H). `fuse` is strict about the model set (ACCURATE = every component runs on every frame; pass
`strict=False` only for an explicit selective-execution experiment). Fused scores are on the
calibrated scale: 1 - score ~ single-model neg-image FP on VAL; the `thresholds_oof` entries map
the fused score to 1/2/5/10/20% neg-image FP, but see the threshold-transfer caveat — per-site
thresholds differ by >0.1 on the calibrated scale (T10) and must be set on the deployed camera's
own verified negatives (PROTOCOL_V5 §3 calibration objective).

Unit tests: `.venv/bin/python -m pytest scripts/v5b/test_fusion.py -q` — 15 passed (calibrator
monotonicity/round-trip, NMS, WBF semantics per conf_type, parity with `ensemble_boxes`, recipe
load/strictness/clipping/determinism, threshold monotonicity, checkpoint paths exist).

## 4. What this does and does not claim

* Claims are VAL-only, OOF across the two VAL sites. pyro-sdis TEST was not touched; the ACCURATE
  recipe has **not** been tested on any held-out set — that is a one-shot final-test item.
* The fusion fits only (a) ~800-knot monotone score maps on VAL negatives and (b) 3 WBF
  hyper-parameters; the component models are the frozen v3/v4 checkpoints (paths in T10 / recipe).
  Training-data disclosure: yolo_A = E corpus; dino_B1/B4/B6 = E+T where T includes AuSmoke train
  frames whose licence is PENDING (PROTOCOL_V5 §2: existing v4 use disclosed, not extended — this
  study adds no training); rfdetr_L / yolo_expE = expB list. No FIgLib-derived material anywhere.
* The +2.8 pt macro gain is the average of a well-supported marguerite gain and a cabanelle gain
  that is within noise; on cabanelle RF-DETR-L alone remains the best single system at 5-20% FP,
  yet adding it to the fusion lowered the macro statistic (-0.39) — its benefit does not survive
  cross-site calibration. A per-site "best model" is not a deployable recipe without per-site
  calibration, which this study shows is required anyway.
* Latency numbers are reported batch-1 L40S fp32 figures copied from v3/v4 reports plus the fusion
  cost measured here on CPU; parallel = max assumes one GPU per component and ignores transfer /
  scheduling overhead. Selective execution is a later test.
* The learned fusion is reported but not recommended (+0.44 pt < 1 pt). It also depends on GT
  labels that contain omissions.

## 5. Artifacts and reproduction

* `reports/v5b/ENSEMBLE_DEV_V5B.md` (this file), `reports/v5b/ensemble_dev_tables.md` (T1-T10),
  `reports/v5b/ensemble_dev_v5b.json` (every curve point, config objective, bootstrap, transfer
  table), `reports/v5b/ensemble_dev_run.log`.
* `models/v5b/fusion/accurate_recipe.json` — calibration maps, fusion params, thresholds,
  OOF results, latency, provenance.
* `scripts/v5b/fusion.py` — `fuse()`, `image_score()`, `ScoreCalibrator`, numpy `wbf`/`union_nms`,
  `FusionRecipe`; `scripts/v5b/test_fusion.py` — pytest; `scripts/v5b/ensemble_dev.py` — the study;
  `scripts/v5b/make_ensemble_tables.py` — table renderer.
* venv additions (uv): ensemble-boxes 1.0.9 (MIT), scikit-learn 1.9.1, pytest 9.1.1 (+ numba/llvmlite
  pulled by ensemble-boxes). An accidental `pip install` into the system python3.9 user site was
  reverted.
* Repro (repo root, `.venv` active, ~2 min on CPU, 24 workers):
  `python scripts/v5b/ensemble_dev.py --workers 24 && python scripts/v5b/make_ensemble_tables.py && python -m pytest scripts/v5b/test_fusion.py -q`
