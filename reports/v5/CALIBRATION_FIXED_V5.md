# V5 Experiment A — FIXED-CAMERA threshold calibration baseline (no training)

Generated 2026-10-02 01:20 UTC by `scripts/v5/make_calibration_fixed_report.py` from `reports/v5/calib/`. Scenario: FIXED-CAMERA only (PROTOCOL_V5 §1). All computation on CPU; no cluster reservation was made. Every number below is a HISTORICAL REPLAY of saved per-image predictions (pyro-sdis VAL) or of saved/re-run FIgLib frames — nothing here is a field or live run.

## 0. Headline

* Anchor (§3): deployed expB at 0.3/0.15 has frame recall **R = 0.6803** on pyro-sdis VAL (IoU≥0.4; tiny 0.5806, small 0.7499, medium 0.5615, large 0.1026); engine-aware recall R_eff = 0.753; **31.074 false-alert episodes per valid hour** on the embedded-negative set (4.1514 valid h, 29 s sampled cadence).
* Holding R ≥ anchor and minimising episodes/h, **every system lands at 29.4–30.6 episodes/h** — the five detectors are indistinguishable on this negative set at matched recall, and re-calibration moves the deployed model by only 5% (paired run-bootstrap ratio CI [0.906, 0.979]). The §0 target (episode false alerts ↓ ≥ 50 % at non-inferior recall) is **not reachable by thresholds alone**; along the recall/FP frontier, halving episodes costs ≈ 26–29 pt of frame recall for every system (R 0.68 → ≈ 0.41; §6).
* The deployed point's recall constraint pins conf_single: expB cannot move above 0.30, expE above 0.32, v4a_s2 above 0.33, B4 above 0.91, B6 above 0.94. Only conf_seq is free, and it changes episode counts by a few units (and partly by *bridging* — extra confirmed frames merge neighbouring episodes — see §3).
* The v4 DINO operating points (B4 0.93/0.82, B6 0.95/0.88, 'expB-matched negFP') have R = 0.6541 / 0.6585 < anchor — they FAIL the §3 recall constraint; the protocol-feasible DINO points are B4 0.91/0.89 and B6 0.94/0.88.
* FIgLib-39 constraint (≥ 30/39, historical set, checked AFTER selection, never used to select): expB **30/39** (pass); expE **30/39** (pass); v4a_s2 **29/39** (FAIL); v4B4_lora_L **31/39** (pass); v4B6_lora_Hplus **30/39** (pass).
* These rates are NOT field false-alert rates: the negatives are no-box frames *inside pyro-sdis alert sequences* (before/after/between smoke, 63 % from 'flicker' sequences) — the hardest possible negatives, with an unknown share of faint unlabelled smoke. Continuous clean footage is still required for Q1 (§10).

## 1. Systems and inputs

| system | kind | weights | per-image preds (7,335 VAL images) |
|---|---|---|---|
| expB (deployed default, YOLO11s@1280) | yolo | `models/expB_best.pt` | `reports/eval_expB_val.raw.parquet` |
| expE (v3 S0, YOLO11m@1280) | yolo | `models/trained/expE_mix_y11m_1280_v2/seg05/best.pt` | `reports/eval_expE_val.raw.parquet` |
| v4 A control v4a_s2 (YOLO11m@1280) | yolo | `models/trained/expV4A_ctrl_s2/seg05/best.pt` | `reports/v4/eval_v4a_s2_sdisval.raw.parquet` |
| DINOv3-L LoRA (B4) | dino | `models/v4/dino/v4B4_lora_L (cluster ckpt)` | `reports/v4/eval_v4B4_lora_L_sdis_val.raw.parquet` |
| DINOv3-H+ LoRA (B6) | dino | `models/v4/dino/v4B6_lora_Hplus (cluster ckpt)` | `reports/v4/eval_v4B6_lora_Hplus_sdis_val.raw.parquet` |

Predictions were produced in iterations 3–4 (imgsz 1280; DINO 1280×736 letterbox) with conf_floor 0.02; the service floor 0.05 is applied here before the alert engine. GT boxes are the frozen YOLO labels; positive/negative flags agree 1:1 with `data/v3/sequences/sequences_val.json` (6,177 / 1,158).

## 2. VAL sequences and the negative observation set

* 266 sequences, 8 camera-azimuth units, sites cabanelle, marguerite; 7335 frames, 6177 with smoke (6603 boxes), 1158 without.
* Cadence: global median 29 s (p10 3 s — detection bursts; p90 31 s); per-sequence median cadence quantiles p10/p25/p50/p75/p90 = 7/13/25/30/30 s. Sequences were cut at gaps > 600 s (= the engine's stale reset), so no in-sequence gap triggers a history reset.
* Negative runs (maximal consecutive no-box frames inside a sequence): **394**, of which **194 have valid time** (949 frames) and **191 are isolated single frames** (zero covered time under §3). 53 inter-frame gaps > 3× sequence-median cadence were excluded.
* **Valid observed time = 4.1514 h** (§3 rule). Sensitivity: 4.3225 h if the 3× rule used the global 29 s cadence instead. Per camera (h): cabanelle-125 0.8308, cabanelle-244 0.3619, cabanelle-327 1.1536, cabanelle-46 0.1353, marguerite-185 0.0925, marguerite-282 1.0144, marguerite-29 0.1022, marguerite-339 0.4606.
* Negative frames by sequence label pattern (all / in timed runs / valid h): flicker 727 / 562 / 2.38 h; all_neg 219 / 219 / 1.24 h; appears_disappears 115 / 92 / 0.29 h; disappears 49 / 31 / 0.15 h; appears 48 / 45 / 0.11 h. Only 1.24 h (30 %) comes from sequences with no smoke at all; the rest sits next to labelled smoke.
* embedded negatives = no-GT-box frames inside pyro-sdis VAL alert sequences (before/after/between smoke), NOT continuous clean footage.

## 3. Metric definitions as implemented (`scripts/v5/calibrate_fixed.py`)

* Alert engine = `src/serve/alerting.py` with production settings: K-of-N 3/5, cooldown 60 s, stale 600 s, conf_floor 0.05; scene-change (phash) not simulated — sequences are per camera-azimuth. An independent re-implementation reproduced the engine replay's episode counts exactly at four operating points.
* **R_single** — frame recall at conf_single, computed by `src.eval.eval_detection.sweep_report` (IoU ≥ 0.4 greedy; the v1–v4 metric). **R_eff** — engine-aware recall: the engine is replayed over each sequence in time order (fresh per sequence); a frame's boxes are matched at conf_seq when the engine is CONFIRMED on that frame, else at conf_single. R_eff = R_single when conf_seq = conf_single.
* **Alert states counted** = ['provisional', 'confirmed'] (the states that notify a user). 'watch' (weak detection below conf_single) is NOT an alert; a *strict* variant counting it is reported for transparency — under it conf_single would be irrelevant and the sweep degenerate.
* **Episode**: starts on an alert-state frame; a later alert frame joins the same episode iff ts − last_alert_ts < cooldown (60 s), i.e. the device had been out of alert for ≥ cooldown. Fresh engine per negative run, so trailing alerts after real smoke are not counted. **Bridging**: lowering conf_seq adds confirmed frames that can merge two episodes into one, so episodes/h is not monotone in conf_seq — 'fraction of valid time in alert' is reported alongside to expose this.
* **Valid time** = Σ inter-frame gaps within a run that are ≤ 3× the sequence's median cadence; larger gaps excluded; runs never concatenated; 1-frame runs contribute 0 h. **eps/h** = episodes in timed runs / valid h (primary). **eps/h (all)** additionally counts episodes from isolated frames over the same hours (upper bound). **frac alert** = accepted gaps whose starting frame is in alert / valid time. **neg-frame alert frac** = share of all 1,158 negative frames in an alert state.
* **Selection** (§3): feasible = R_single ≥ anchor; minimise eps/h; ties → higher R_eff, higher R_single, lower confs. *Engine-aware variant*: additionally R_eff ≥ expB's R_eff at 0.30/0.15. Grid: 0.05–0.95 step 0.01, 0.95–0.99 step 0.001, 0.99–1 step 0.0005, plus the deployed/v4 points (154 thresholds, 11935 pairs with conf_seq ≤ conf_single).

## 4. Anchor (deployed expB at 0.30/0.15)

| R_single | tiny | small | medium | large | R_eff | neg-image FP rate | episodes (timed/all) | eps/h | eps/h all | strict | frac alert | neg-frame alert |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0.6803** | 0.5806 | 0.7499 | 0.5615 | 0.1026 | 0.753 | 0.3402 | 129 / 213 | **31.074** | 51.308 | 36.855 | 0.3832 | 0.3946 |

This matches the iteration-1 value in `reports/eval_expB_val.json` (R 0.6803 @ conf 0.30), computed with the same code path.

## 5. Calibration results per system (pyro-sdis VAL, embedded negatives)

| system | point | conf_single/conf_seq | R_single | R_eff | episodes timed/all | **eps/h** | eps/h all | eps/h strict | frac alert | neg-frame alert | feasible pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| expB (deployed default, YOLO11s@1280) | YOLO deployed 0.30/0.15 | 0.3/0.15 | 0.6803 | 0.753 | 129/213 | 31.074 | 51.308 | 36.855 | 0.3832 | 0.3946 | |
| | **CHOSEN (§3 rule)** | **0.3/0.23** | 0.6803 | 0.7127 | 122/206 | **29.388** | 49.622 | 33.242 | 0.358 | 0.3705 | 351 |
| | engine-aware variant | 0.3/0.08 | 0.6803 | 0.8001 | 127/211 | 30.592 | 50.826 | 41.673 | 0.4062 | 0.4119 | 245 |
| expE (v3 S0, YOLO11m@1280) | YOLO deployed 0.30/0.15 | 0.3/0.15 | 0.6964 | 0.7848 | 132/211 | 31.797 | 50.826 | 37.819 | 0.4397 | 0.4154 | |
| | **CHOSEN (§3 rule)** | **0.32/0.2** | 0.6808 | 0.7536 | 127/202 | **30.592** | 48.658 | 34.446 | 0.4184 | 0.3877 | 406 |
| | engine-aware variant | 0.32/0.2 | 0.6808 | 0.7536 | 127/202 | 30.592 | 48.658 | 34.446 | 0.4184 | 0.3877 | 338 |
| v4 A control v4a_s2 (YOLO11m@1280) | YOLO deployed 0.30/0.15 | 0.3/0.15 | 0.7017 | 0.7783 | 130/212 | 31.315 | 51.067 | 35.651 | 0.4272 | 0.4033 | |
| | **CHOSEN (§3 rule)** | **0.33/0.28** | 0.6826 | 0.7068 | 122/198 | **29.388** | 47.695 | 32.037 | 0.3647 | 0.3497 | 435 |
| | engine-aware variant | 0.33/0.19 | 0.6826 | 0.7537 | 123/199 | 29.629 | 47.936 | 35.651 | 0.4052 | 0.3782 | 344 |
| DINOv3-L LoRA (B4) | YOLO deployed 0.30/0.15 | 0.3/0.15 | 0.8798 | 0.9056 | 189/339 | 45.527 | 81.659 | 49.622 | 0.58 | 0.639 | |
| DINOv3-L LoRA (B4) | v4 sdis-val-selected (expB-matched negFP) 0.93/0.82 | 0.93/0.82 | 0.6541 ✗<anchor | 0.7497 | 121/206 | 29.147 | 49.622 | 38.06 | 0.3716 | 0.3756 | |
| | **CHOSEN (§3 rule)** | **0.91/0.89** | 0.6907 | 0.7118 | 126/217 | **30.351** | 52.272 | 32.037 | 0.3715 | 0.3826 | 3828 |
| | engine-aware variant | 0.91/0.75 | 0.6907 | 0.7799 | 127/218 | 30.592 | 52.513 | 40.228 | 0.4119 | 0.4128 | 3799 |
| DINOv3-H+ LoRA (B6) | YOLO deployed 0.30/0.15 | 0.3/0.15 | 0.9043 | 0.9219 | 211/365 | 50.826 | 87.922 | 52.031 | 0.6422 | 0.6874 | |
| DINOv3-H+ LoRA (B6) | v4 sdis-val-selected (expB-matched negFP) 0.95/0.88 | 0.95/0.88 | 0.6585 ✗<anchor | 0.7439 | 122/200 | 29.388 | 48.177 | 36.855 | 0.3578 | 0.3661 | |
| | **CHOSEN (§3 rule)** | **0.94/0.88** | 0.6827 | 0.7475 | 126/210 | **30.351** | 50.585 | 36.855 | 0.3703 | 0.3808 | 4095 |
| | engine-aware variant | 0.94/0.87 | 0.6827 | 0.7533 | 128/212 | 30.833 | 51.067 | 37.578 | 0.3824 | 0.3877 | 4073 |

Paired bootstrap (2,000 resamples over the 194 timed runs; cluster version over the 8 cameras) of episodes at the chosen point vs the reference point (YOLO: deployed 0.30/0.15; DINO: v4 val-selected point):

| system | reference → chosen | eps/h ref → chosen | ratio | 95 % CI (runs) | 95 % CI (camera cluster) | runs with ≥1 episode ref → chosen |
|---|---|---|---|---|---|---|
| expB (deployed default, YOLO11s@1280) | 0.3/0.15 → 0.3/0.23 | 31.07 → 29.39 | 0.946 | [0.906, 0.979] | [0.92, 0.968] | 111 → 109 / 194 |
| expE (v3 S0, YOLO11m@1280) | 0.3/0.15 → 0.32/0.2 | 31.8 → 30.59 | 0.962 | [0.921, 1.007] | [0.923, 1.0] | 114 → 111 / 194 |
| v4 A control v4a_s2 (YOLO11m@1280) | 0.3/0.15 → 0.33/0.28 | 31.31 → 29.39 | 0.938 | [0.888, 0.991] | [0.879, 0.984] | 111 → 107 / 194 |
| DINOv3-L LoRA (B4) | 0.93/0.82 → 0.91/0.89 | 29.15 → 30.35 | 1.041 | [1.007, 1.087] | [1.0, 1.132] | 101 → 105 / 194 |
| DINOv3-H+ LoRA (B6) | 0.95/0.88 → 0.94/0.88 | 29.39 → 30.35 | 1.033 | [0.992, 1.089] | [1.005, 1.108] | 107 → 111 / 194 |

The absolute eps/h has a wide run-bootstrap interval (≈ 22–43 /h for every system) — 4.15 h and 8 cameras do not support fine comparisons between systems; the paired ratios are tighter because the same frames are scored.

Size-bucket recall at the chosen conf_single (GT boxes: tiny < 0.1 % area, small 0.1–1 %, medium 1–5 %, large > 5 %):

| system | conf_single | tiny | small | medium | large | precision |
|---|---|---|---|---|---|---|
| expB (deployed default, YOLO11s@1280) | 0.3 | 0.5806 | 0.7499 | 0.5615 | 0.1026 | 0.8033 |
| expE (v3 S0, YOLO11m@1280) | 0.32 | 0.5775 | 0.7514 | 0.5538 | 0.2564 | 0.817 |
| v4 A control v4a_s2 (YOLO11m@1280) | 0.33 | 0.5859 | 0.7509 | 0.55 | 0.1538 | 0.8311 |
| DINOv3-L LoRA (B4) | 0.91 | 0.6052 | 0.7422 | 0.7346 | 0.0769 | 0.8294 |
| DINOv3-H+ LoRA (B6) | 0.94 | 0.6148 | 0.7281 | 0.6731 | 0.0256 | 0.8398 |

## 6. Recall / false-episode frontier (best conf_seq per conf_single)

What FP reduction costs in frame recall if the recall constraint were relaxed — reported, not chosen.

| system | cs=0.3 | cs=0.35 | cs=0.4 | cs=0.5 | cs=0.6 | cs=0.7 | | cs=0.9 | cs=0.93 | cs=0.95 | cs=0.97 | cs=0.98 | cs=0.99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| expB (deployed default, YOLO11s@1280) | R 0.680 / 29.4 | R 0.643 / 27.9 | R 0.608 / 25.1 | R 0.519 / 20.0 | R 0.405 / 14.7 | R 0.244 / 7.2 |  | R 0.000 / 0.0 | R 0.000 / 0.0 | R 0.000 / 0.0 | R 0.000 / 0.0 | R 0.000 / 0.0 | R 0.000 / 0.0 |
| expE (v3 S0, YOLO11m@1280) | R 0.696 / 31.6 | R 0.661 / 28.7 | R 0.622 / 26.5 | R 0.523 / 19.8 | R 0.393 / 13.2 | R 0.228 / 7.7 |  | R 0.000 / 0.0 | R 0.000 / 0.0 | R 0.000 / 0.0 | R 0.000 / 0.0 | R 0.000 / 0.0 | R 0.000 / 0.0 |
| v4 A control v4a_s2 (YOLO11m@1280) | R 0.702 / 31.1 | R 0.665 / 28.2 | R 0.625 / 23.1 | R 0.539 / 19.5 | R 0.417 / 14.2 | R 0.254 / 8.7 |  | R 0.000 / 0.0 | R 0.000 / 0.0 | R 0.000 / 0.0 | R 0.000 / 0.0 | R 0.000 / 0.0 | R 0.000 / 0.0 |
| DINOv3-L LoRA (B4) | R 0.880 / 45.3 | R 0.872 / 44.3 | R 0.865 / 43.6 | R 0.853 / 42.6 | R 0.833 / 42.4 | R 0.811 / 40.7 |  | R 0.706 / 31.8 | R 0.654 / 28.7 | R 0.603 / 24.6 | R 0.503 / 19.0 | R 0.421 / 15.2 | R 0.286 / 8.9 |
| DINOv3-H+ LoRA (B6) | R 0.904 / 50.6 | R 0.900 / 49.9 | R 0.893 / 48.7 | R 0.879 / 47.2 | R 0.859 / 45.3 | R 0.838 / 43.1 |  | R 0.742 / 35.9 | R 0.703 / 31.3 | R 0.658 / 29.4 | R 0.577 / 23.1 | R 0.496 / 19.0 | R 0.342 / 10.6 |

Cell = frame recall / episodes per valid hour. Halving episodes relative to the anchor (≈ 15.5 /h) requires conf_single ≈ 0.6 for the YOLO systems (R ≈ 0.39–0.42, i.e. −26 to −29 pt) and ≈ 0.98 for the DINO systems (R ≈ 0.42–0.50). Equal-recall comparison: at R ≈ 0.68 all five systems sit at 29–31 /h; the DINO models have higher recall at YOLO-style thresholds only because their scores are saturated, not because they separate smoke from these negatives better.

## 7. FIgLib-39 constraint check (HISTORICAL set — HPWREN fixed cameras, 60 s cadence, no boxes, open since iteration 1)

Checked AFTER selection; FIgLib is banned from tuning/selection (§2), so no point below was chosen using it. Official = YOLO re-inference with `src/eval/replay_eval.py --weights` at imgsz 1280 (CPU), or DINO `--scores-jsonl` on the saved v4 cluster frame scores (the only form available for DINO). Diagnostic rows replay saved per-frame max scores (4-dp rounded) and are sensitivity only.

| system | point | source | detected /39 | missed | pre-visible alerts | median delay s | within 1/3/5 min | §3 ≥30 |
|---|---|---|---|---|---|---|---|---|
| expB (deployed default, YOLO11s@1280) | YOLO deployed 0.30/0.15 (v3/v4 replay, weights, imgsz 1280) | reference (weights) | 32 | 7 | 5 | 475 | 7/10/15 | |
| | **CHOSEN 0.3/0.23** | official (replay_eval --weights) | **30** | 9 | 5 | 420 | 7/10/15 | **pass** |
| | engine_aware | diagnostic (saved scores) | 32 | 7 | 9 | 240 | 10/13/20 | |
| | frontier conf_single=0.25 (R_single=0.7118, eps/h=32.037) | diagnostic (saved scores) | 30 | 9 | 5 | 240 | 7/10/17 | |
| | frontier conf_single=0.3 (R_single=0.6803, eps/h=29.388) | diagnostic (saved scores) | 30 | 9 | 5 | 420 | 7/10/15 | |
| | frontier conf_single=0.35 (R_single=0.643, eps/h=27.942) | diagnostic (saved scores) | 29 | 10 | 4 | 475 | 6/9/13 | |
| | frontier conf_single=0.4 (R_single=0.6079, eps/h=25.052) | diagnostic (saved scores) | 29 | 10 | 4 | 475 | 6/9/13 | |
| | frontier conf_single=0.5 (R_single=0.5186, eps/h=19.993) | diagnostic (saved scores) | 26 | 13 | 3 | 480 | 5/8/11 | |
| expE (v3 S0, YOLO11m@1280) | YOLO deployed 0.30/0.15 (v3/v4 replay, weights, imgsz 1280) | reference (weights) | 30 | 9 | 5 | 240 | 7/12/17 | |
| | **CHOSEN 0.32/0.2** | official (replay_eval --weights) | **30** | 9 | 5 | 240 | 7/12/17 | **pass** |
| | engine_aware | diagnostic (saved scores) | 30 | 9 | 5 | 240 | 7/12/17 | |
| | frontier conf_single=0.25 (R_single=0.7354, eps/h=32.76) | diagnostic (saved scores) | 31 | 8 | 6 | 240 | 9/14/19 | |
| | frontier conf_single=0.3 (R_single=0.6964, eps/h=31.556) | diagnostic (saved scores) | 30 | 9 | 5 | 240 | 7/12/17 | |
| | frontier conf_single=0.35 (R_single=0.6611, eps/h=28.665) | diagnostic (saved scores) | 30 | 9 | 5 | 300 | 7/12/16 | |
| | frontier conf_single=0.4 (R_single=0.6224, eps/h=26.497) | diagnostic (saved scores) | 28 | 11 | 4 | 300 | 6/11/15 | |
| | frontier conf_single=0.5 (R_single=0.5229, eps/h=19.752) | diagnostic (saved scores) | 24 | 15 | 2 | 420 | 4/6/10 | |
| v4 A control v4a_s2 (YOLO11m@1280) | YOLO deployed 0.30/0.15 (v3/v4 replay, weights, imgsz 1280) | reference (weights) | 30 | 9 | 6 | 240 | 9/14/17 | |
| | **CHOSEN 0.33/0.28** | official (replay_eval --weights) | **29** | 10 | 4 | 240 | 7/11/15 | **FAIL** |
| | engine_aware | diagnostic (saved scores) | 29 | 10 | 5 | 240 | 8/12/16 | |
| | frontier conf_single=0.25 (R_single=0.7359, eps/h=33.242) | diagnostic (saved scores) | 30 | 9 | 7 | 240 | 10/14/17 | |
| | frontier conf_single=0.3 (R_single=0.7017, eps/h=31.074) | diagnostic (saved scores) | 30 | 9 | 7 | 240 | 10/14/17 | |
| | frontier conf_single=0.35 (R_single=0.6655, eps/h=28.183) | diagnostic (saved scores) | 29 | 10 | 3 | 301 | 6/10/14 | |
| | frontier conf_single=0.4 (R_single=0.6253, eps/h=23.125) | diagnostic (saved scores) | 28 | 11 | 2 | 361 | 5/9/13 | |
| | frontier conf_single=0.5 (R_single=0.5387, eps/h=19.512) | diagnostic (saved scores) | 26 | 13 | 2 | 480 | 4/7/11 | |
| DINOv3-L LoRA (B4) | v4 v3-rule 0.9881/0.973 | reference (saved scores) | 26 | 13 | 0 | 481 | 1/3/5 | |
| DINOv3-L LoRA (B4) | v4 expB-matched-negFP 0.93/0.82 | reference (saved scores) | 31 | 8 | 0 | 300 | 2/8/18 | |
| | **CHOSEN 0.91/0.89** | official (replay_eval --scores-jsonl on saved v4 cluster frame scores) | **31** | 8 | 0 | 300 | 2/8/18 | **pass** |
| | engine_aware | diagnostic (saved scores) | 32 | 7 | 1 | 300 | 3/9/18 | |
| | frontier conf_single=0.85 (R_single=0.7547, eps/h=36.614) | diagnostic (saved scores) | 32 | 7 | 3 | 240 | 5/12/21 | |
| | frontier conf_single=0.9 (R_single=0.7062, eps/h=31.797) | diagnostic (saved scores) | 32 | 7 | 1 | 300 | 3/9/18 | |
| | frontier conf_single=0.93 (R_single=0.6541, eps/h=28.665) | diagnostic (saved scores) | 30 | 9 | 0 | 300 | 2/8/18 | |
| | frontier conf_single=0.95 (R_single=0.6031, eps/h=24.57) | diagnostic (saved scores) | 30 | 9 | 0 | 300 | 2/7/18 | |
| | frontier conf_single=0.97 (R_single=0.5027, eps/h=19.03) | diagnostic (saved scores) | 30 | 9 | 0 | 358 | 2/6/15 | |
| DINOv3-H+ LoRA (B6) | v4 v3-rule 0.9901/0.9801 | reference (saved scores) | 24 | 15 | 0 | 539 | 1/3/6 | |
| DINOv3-H+ LoRA (B6) | v4 expB-matched-negFP 0.95/0.88 | reference (saved scores) | 30 | 9 | 1 | 358 | 3/8/14 | |
| | **CHOSEN 0.94/0.88** | official (replay_eval --scores-jsonl on saved v4 cluster frame scores) | **30** | 9 | 1 | 300 | 3/9/16 | **pass** |
| | engine_aware | diagnostic (saved scores) | 30 | 9 | 1 | 300 | 3/10/16 | |
| | frontier conf_single=0.85 (R_single=0.7812, eps/h=37.578) | diagnostic (saved scores) | 33 | 6 | 2 | 240 | 5/12/21 | |
| | frontier conf_single=0.9 (R_single=0.7416, eps/h=35.892) | diagnostic (saved scores) | 31 | 8 | 1 | 300 | 4/10/17 | |
| | frontier conf_single=0.93 (R_single=0.7027, eps/h=31.315) | diagnostic (saved scores) | 30 | 9 | 1 | 300 | 4/10/16 | |
| | frontier conf_single=0.95 (R_single=0.6585, eps/h=29.388) | diagnostic (saved scores) | 30 | 9 | 1 | 358 | 3/8/14 | |
| | frontier conf_single=0.97 (R_single=0.5767, eps/h=23.125) | diagnostic (saved scores) | 30 | 9 | 1 | 360 | 2/7/13 | |

**Constraint failures:** v4 A control v4a_s2 (YOLO11m@1280) at 0.33/0.28 detects 29/39. Per §3 these sdis-only points are NOT accepted; for these systems the only verified-feasible point remains the deployed/reference one. Choosing a different point *because* of FIgLib would be selection on a historical set (§2) and is not proposed.
Pre-visible alerts (alerts before FIgLib's first-visible frame) are listed, never counted as detections or as false alarms. Delays are from the labelled first-visible frame, not ignition; FIgLib frames are 60 s apart so delays quantise to minutes.

## 8. Per-camera episodes at the chosen point (timed runs)

| system | cabanelle-125 | cabanelle-244 | cabanelle-327 | cabanelle-46 | marguerite-185 | marguerite-282 | marguerite-29 | marguerite-339 |
|---|---|---|---|---|---|---|---|---|
| valid h | 0.8308 | 0.3619 | 1.1536 | 0.1353 | 0.0925 | 1.0144 | 0.1022 | 0.4606 |
| expB (deployed default, YOLO11s@1280) | 20 | 27 | 47 | 7 | 0 | 4 | 7 | 10 |
| expE (v3 S0, YOLO11m@1280) | 18 | 30 | 48 | 5 | 0 | 8 | 7 | 11 |
| v4 A control v4a_s2 (YOLO11m@1280) | 17 | 27 | 48 | 5 | 0 | 6 | 8 | 11 |
| DINOv3-L LoRA (B4) | 23 | 23 | 41 | 8 | 0 | 9 | 9 | 13 |
| DINOv3-H+ LoRA (B6) | 18 | 28 | 45 | 4 | 0 | 7 | 13 | 11 |

cabanelle-327 (1.15 h) alone produces ~35–40 % of all episodes for every system; marguerite-185 (0.09 h) produces none. With 8 camera-azimuth units on 2 sites the per-camera rate is highly heterogeneous, which is why camera-cluster intervals are given in §5.

## 9. Reading for V5

1. **Experiment A answers Q1 negatively for thresholds**: at the pre-agreed recall, no (conf_single, conf_seq) pair of the deployed engine reduces false-alert episodes by more than ~6 % for any system, and no system beats another at matched recall on these negatives. The 50 % target needs new information (matched negatives / onset supervision, experiments C/D), not a different operating point.
2. **Engine structure matters more than the model here**: the confirmed path (3-of-5 at conf_seq) is what the free parameter touches; raising conf_seq removes a few episodes but drops R_eff (expB 0.753 → 0.713) and costs FIgLib events (expB 32 → 30/39, official re-inference; v4a_s2 30 → 29/39 and fails the §3 check). A model-agnostic lever (longer windows in seconds, per-camera baselines) should be evaluated with the same harness before training decisions.
3. **DINO LoRA scores are saturated** (median max-score 0.97–0.98 on VAL; feasible conf_single ≥ 0.9); their feasible points are no better than YOLO's on episodes/h and pass FIgLib only at 30–31/39 with median delay 300 s vs 240 s for expE/v4a_s2. Threshold transfer from sdis to HPWREN is fragile for them.
4. **The embedded-negative set is a stress test, not a field proxy**: 31 episodes per hour on frames adjacent to labelled smoke is consistent with those frames containing faint/unlabelled smoke as often as true false positives. Human review of a sample of alerted negative frames would resolve how much of the 31/h is annotation leakage — a cheap Day-2 task that does not need new data.

## 10. Limitations

* **Embedded negatives ≠ continuous clean footage.** 70 % of valid time is adjacent to labelled smoke within alert sequences (63 % 'flicker'); no negative time comes from hours of a camera seeing nothing. Rates are upper-bound-like for FP and cannot be compared with field rates or with continuous-footage rates measured later (those will be reported separately with their own cadence).
* **Sampled cadence.** Frames are ~29 s apart with 1–5 s bursts; 'episode' and 'time in alert' are interpolated from samples; the 60 s cooldown spans only ~2 samples. 191 isolated negative frames have no covered time and are excluded from the primary rate (their episodes appear in 'eps/h all').
* **Valid-time rule sensitivity.** Per-sequence 3× median cadence excludes 53 gaps and gives 4.15 h; a global 29 s cadence would give 4.32 h (+4 %). Bursty sequences (per-sequence median down to 2 s) make the per-sequence rule strict for them.
* **Small observation set**: 4.15 h, 8 camera-azimuth units, 2 sites, 194 timed runs → absolute eps/h CI ≈ ±10 /h; only paired comparisons are informative.
* **Recall anchor is self-referential for expB** (its own VAL recall), so expB can only move conf_seq; and the anchor is a frame-box metric that does not reward the confirmed path — hence the engine-aware variant is shown but not used for the decision.
* **DINO score saturation**: operating points at 0.91–0.94 sit in a compressed tail (grid 0.001 there); small score shifts between domains move recall a lot (v4 finding reproduced: sdis-selected DINO thresholds lose FIgLib events).
* **FIgLib-39 is historical and open** (used since v1; 65/89 of our broader FIgLib events are byte-identical inside public MultiNatSmoke); it is a constraint check, not evidence of early-warning performance, and its 39 events make ±2 events noise.
* **Not simulated**: phash scene-change resets, tiled inference, service-side image decoding; the YOLO FIgLib official runs re-infer on CPU at imgsz 1280 (same weights, same letterbox as the deployed service).
* **Bridging artefact** in the episode definition (§3) can lower episode counts while raising time-in-alert; the chosen points have lower frac-alert than the deployed ones for all YOLO systems, so they are not bridging artefacts, but the engine-aware picks for expB (0.30/0.08) and B4 (0.91/0.75) partly are.

## 11. Artifacts and reproduction

* `scripts/v5/calibrate_fixed.py` — sweep + §3 metrics (`--systems all --workers 64`, ~1 min CPU total).
* `scripts/v5/figlib_check_fixed.py --diag` — FIgLib gather/check; official YOLO runs launched as `OMP_NUM_THREADS=40 python -m src.eval.replay_eval --manifest data/manifests/figlib_subset_manifest.parquet --weights <w> --imgsz 1280 --conf-single <cs> --conf-seq <cq> --out reports/v5/calib/figlib/replay_<sys>_chosen_<cs>_<cq>.json`.
* `scripts/v5/make_calibration_fixed_report.py` — this file.
* `reports/v5/calib/<system>.json` (anchor, deployed/chosen/engine-aware points, frontier, data bookkeeping, figlib_check), `<system>_sweep.csv` (all pairs), `figlib_check.json`, `bootstrap_eps.json`, `figlib/replay_*.json` + `diag_*.json`, `figlib_scores/` (saved-score JSONLs + PROVENANCE.json).
* Inputs untouched: `reports/eval_*_val.raw.parquet`, `reports/v4/eval_*_sdis_val.raw.parquet`, `data/v3/sequences/sequences_val.json`, `models/v4/evals/*_figlib/figlib39_preds.jsonl`. No pyro-sdis TEST, Boreal, AuSmoke or lockbox material was read.
