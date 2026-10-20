# Iteration 4 — DINOv3 visual adaptation on the v4 E+T recipe (VAL results)

_generated 2026-09-14T23:39Z by scripts/v4/dino_adapt_report.py — validation only; no test/lockbox data touched._

## Headline (val only; see confounds below)

- **Adapting the backbone helps on sdis_val at every matched-FP level.** At 5% neg-image FP: B6 LoRA-H+ R 0.227 / tiny 0.144, B4 LoRA-L 0.210 / 0.144, B5 full-L 0.209 / 0.131 vs frozen B1 0.175 / 0.104 and the best YOLO arm (A ctrl-E) 0.204 / 0.112. At 1-2% FP the LoRA runs lead all YOLO arms and the v3 E-only DINO runs (B6 0.097 / B4 0.085 vs YOLO-A 0.053, expE 0.059 at 1%). B5 cannot reach 1% FP on sdis_val (floor 1.4%).
- **H+ LoRA (B6) is the best single config on sdis_val** (also best COCO mAP 0.422) but costs 2.1x the node-hours of B4 (8.0 vs 3.8) and ~2.4x its inference latency class (v3: 371 vs 156 ms b1 L40S). B4 LoRA-L gets most of the gain for +3.1M trainable params and the same wall time as a frozen run would with ckpt (3.8 node-h vs 1.7 for B1).
- **Full unfreeze (B5) does not beat LoRA** despite 100x the trainable params and 2.2x the node-hours; its bs-2/16 schedule and end-only checkpoint are confounds (below), but there is no evidence it is worth it.
- **boreal_val (aerial burn, large plumes)**: every DINO run sits at R 0.96-0.99 by 1-5% FP; YOLO arms trained with T data (E/C) reach 0.93-0.95, the E-only YOLO control 0.64-0.83. Differences among DINO runs here are within a handful of images (217 negatives, 906 positives).
- **forest_val (no negatives)**: at a fixed conf 0.50 adapted runs recall far more (B6 0.553, B4 0.519, B5 0.489 vs B1 0.383, YOLO 0.07-0.12), but under sdis-anchored thresholds the frozen B1 transfers best because adapted scores saturate near 1.0 — a calibration effect, not a detection-quality ranking. Forest false-alarm behavior is unmeasured (no forest negatives exist in the package).
- **Incidents**: no training OOM/retries at the planned batch sizes. One EVAL incident: B4's first val eval job crashed in every shard (`ckpt mismatch ... unexpected lora_A/B`) because `dino_infer.py` built a plain backbone; fixed by adding `--lora R` injection (CPU-verified against B4/B5 checkpoints), an empty-preds hard-fail in the driver, and re-running (bogus empty sweeps were deleted; the failed eval dir is kept as `models/v3/evals/v4B4_lora_L_val.FAILED_lora_keys`). Cost: +0.2 node-h.

## Setup

- Train corpus: frozen `data/v4/yolo/E_ET_v1` -> `data/v4/coco/instances_train_eet.json` (51,420 images: E 38,454 / T 12,966; 48,480 boxes; 0 label files missing; 7 E boxes dropped <1px after clamping). file_name = absolute pod paths.
- Val sets: `sdis_val` (7,335 img / 1,158 neg; v3 json unchanged), `boreal_val` = burn_heinola (1,123 img / 217 explicit-empty neg / 917 boxes, **902 of 917 GT boxes are 'large' >5% area**), `forest_val` = AuSmoke forest-holdout VAL units (1,471 img / **0 negatives** / 1,975 mask-derived boxes). Package: `data/v4/stage/v4_evalval.tar` staged once at `/scratch/.../smoke/v4data/v4_evalval/`.
- Common: 1280x736 letterbox, 10 epochs planned, warmup 500, wd 1e-4, eval-max 1500, 8x L40S per run, segmented 1h jobs via `cluster/v3/orchestrate_dino.py`, image smoke-dino3:v1.
- Inference: `cluster/v3/dino_infer.py` (multi-root `--root-map` added), best.pt (COCO-mAP-selected on the 1500-img sdis-val subset), conf floor 0.02; sweeps by `src/eval/eval_detection.py` (IoU>=0.4) and `scripts/v4/compare_matched_fp.py --top-fine`.

## Runs

| run | backbone | adaptation | bs/GPU (eff.) | lr head / backbone | grad-ckpt | extra args |
|---|---|---|---|---|---|---|
| v4B1_frozen_L | dinov3-vitl16 | frozen (head+neck only) | 4 (32) | 1e-4 / - | no | `(none)` |
| v4B4_lora_L | dinov3-vitl16 | LoRA r16 a32 on q/k/v/o | 4 (32) | 1e-4 / 1e-4 | yes | `--lora 16 --backbone-lr 1e-4 --grad-ckpt` |
| v4B5_full_L | dinov3-vitl16 | all blocks trainable | 2 (16) | 1e-4 / 1e-5 | yes | `--unfreeze-all --grad-ckpt --backbone-lr 1e-5` |
| v4B6_lora_Hplus | dinov3-vith16plus | LoRA r16 a32 on q/k/v/o | 4 (32) | 1e-4 / 1e-4 | yes | `--lora 16 --backbone-lr 1e-4 --grad-ckpt` |

## Parameter counts (from checkpoints + HF config meta-count)

| run | backbone trainable / total | neck | head | ALL trainable |
|---|---|---|---|---|
| v4B1_frozen_L | 0 / 303,129,600 | 7,805,184 | 14,499,865 | 22,305,049 |
| v4B4_lora_L | 3,145,728 / 303,129,600 | 7,805,184 | 14,499,865 | 25,450,777 |
| v4B5_full_L | 303,128,576 / 303,129,600 | 7,805,184 | 14,499,865 | 325,433,625 |
| v4B6_lora_Hplus | 5,242,880 / 840,592,640 | 10,640,192 | 14,499,865 | 30,382,937 |

LoRA r16 = layers x 4 proj x 16 x (d+d): ViT-L 24x4x16x2048 = 3,145,728; ViT-H+ 32x4x16x2560 = 5,242,880. Full unfreeze leaves only the unused 1,024-param mask_token frozen.

## Training outcome / cost

| run | status | segments | node-hours (8 L40S) | steps | epochs achieved (exposures) | best COCO mAP (1500-img sdis-val) @step | OOM / retries |
|---|---|---|---|---|---|---|---|
| v4B1_frozen_L | complete | 3 | 1.66 | 16070/16070 | 10.00 (514,240) | 0.231 @10744 | none |
| v4B4_lora_L | complete | 5 | 3.84 | 16070/16070 | 10.00 (514,240) | 0.409 @15026 | none |
| v4B5_full_L | complete | 11 | 8.35 | 32140/32140 | 10.00 (514,240) | 0.403 @32140 | none |
| v4B6_lora_Hplus | complete | 10 | 8.02 | 16070/16070 | 10.00 (514,240) | 0.422 @8428 | none |

All four first segments exited rc=0 at the planned batch sizes — no OOM, no retries, no fallbacks used. Node-hours = sum of orchestrator-observed segment wall times (includes pod start, code/json fetch, eval passes).

## Val matched-FP results (recall / tiny-recall at neg-image FP rate)

Cell = `R / tinyR (c=conf)` interpolated on each model's own confidence curve at the FP target. `FP ceiling X% -> R>=r`: even at the 0.02 conf floor the model produces only X% neg-image FP, so its recall at any higher FP target is at least r (already maximal). `n/a (FP floor X%)`: the model cannot get below X% FP on this split.

### sdis_val  (negatives: 1158)

| FP(neg-img) | v4B1_frozen_L | v4B4_lora_L | v4B5_full_L | v4B6_lora_Hplus | YOLO_A_ctrlE | YOLO_E_ET | YOLO_C_EGT | expE_y11m_iter2 | v3_B1_frozenL_Eonly12ep | v3_B3_top4L_Eonly12ep |
|---|---|---|---|---|---|---|---|---|---|---|
| 1% | 0.060 / 0.027 (c=0.986) | 0.085 / 0.051 (c=0.997) | n/a (FP floor 1.4%) | 0.097 / 0.038 (c=0.997) | 0.053 / 0.015 (c=0.801) | 0.046 / 0.006 (c=0.819) | 0.041 / 0.002 (c=0.812) | 0.059 / 0.013 (c=0.799) | 0.071 / 0.029 (c=0.967) | 0.033 / 0.017 (c=0.995) |
| 2% | 0.099 / 0.049 (c=0.980) | 0.115 / 0.073 (c=0.996) | 0.094 / 0.039 (c=0.999) | 0.133 / 0.064 (c=0.996) | 0.104 / 0.040 (c=0.768) | 0.073 / 0.013 (c=0.800) | 0.090 / 0.019 (c=0.781) | 0.092 / 0.026 (c=0.775) | 0.100 / 0.042 (c=0.958) | 0.079 / 0.044 (c=0.992) |
| 5% | 0.175 / 0.104 (c=0.968) | 0.210 / 0.144 (c=0.993) | 0.209 / 0.131 (c=0.998) | 0.227 / 0.144 (c=0.994) | 0.204 / 0.112 (c=0.717) | 0.161 / 0.065 (c=0.746) | 0.172 / 0.078 (c=0.735) | 0.190 / 0.097 (c=0.721) | 0.173 / 0.082 (c=0.938) | 0.154 / 0.089 (c=0.986) |
| 10% | 0.301 / 0.200 (c=0.941) | 0.311 / 0.222 (c=0.988) | 0.316 / 0.231 (c=0.996) | 0.340 / 0.252 (c=0.990) | 0.309 / 0.201 (c=0.661) | 0.247 / 0.132 (c=0.691) | 0.266 / 0.164 (c=0.684) | 0.309 / 0.201 (c=0.655) | 0.275 / 0.161 (c=0.903) | 0.264 / 0.159 (c=0.976) |
| 20% | 0.464 / 0.341 (c=0.868) | 0.478 / 0.392 (c=0.973) | 0.488 / 0.413 (c=0.992) | 0.494 / 0.416 (c=0.980) | 0.483 / 0.367 (c=0.547) | 0.415 / 0.294 (c=0.566) | 0.419 / 0.330 (c=0.587) | 0.480 / 0.364 (c=0.535) | 0.457 / 0.329 (c=0.802) | 0.423 / 0.280 (c=0.945) |

### boreal_val  (negatives: 217)

| FP(neg-img) | v4B1_frozen_L | v4B4_lora_L | v4B5_full_L | v4B6_lora_Hplus | YOLO_A_ctrlE | YOLO_E_ET | YOLO_C_EGT |
|---|---|---|---|---|---|---|---|
| 1% | 0.974 / 0.000 (c=0.498) | 0.973 / 0.000 (c=0.868) | 0.963 / 0.000 (c=0.568) | 0.964 / 0.000 (c=0.888) | 0.640 / 0.000 (c=0.408) | 0.931 / 0.000 (c=0.068) | 0.934 / 0.000 (c=0.084) |
| 2% | 0.975 / 0.000 (c=0.406) | 0.983 / 0.000 (c=0.496) | 0.963 / 0.000 (c=0.406) | 0.966 / 0.000 (c=0.837) | 0.674 / 0.000 (c=0.377) | FP ceiling 1.4% -> R>=0.954 | 0.935 / 0.000 (c=0.073) |
| 5% | 0.982 / 0.000 (c=0.056) | 0.983 / 0.000 (c=0.146) | 0.968 / 0.000 (c=0.031) | 0.972 / 0.000 (c=0.511) | 0.783 / 0.000 (c=0.257) | FP ceiling 1.4% -> R>=0.954 | 0.936 / 0.000 (c=0.052) |
| 10% | FP ceiling 7.8% -> R>=0.982 | 0.986 / 0.000 (c=0.038) | FP ceiling 5.1% -> R>=0.968 | 0.980 / 0.000 (c=0.043) | 0.801 / 0.000 (c=0.215) | FP ceiling 1.4% -> R>=0.954 | 0.942 / 0.000 (c=0.025) |
| 20% | FP ceiling 7.8% -> R>=0.982 | FP ceiling 12.4% -> R>=0.987 | FP ceiling 5.1% -> R>=0.968 | FP ceiling 12.0% -> R>=0.985 | 0.828 / 0.000 (c=0.165) | FP ceiling 1.4% -> R>=0.954 | FP ceiling 11.5% -> R>=0.943 |

boreal_val has 1 tiny + 5 small GT boxes in total — tiny-recall there is noise; read only overall recall. With 217 negatives, 1% FP = 2 images.

### forest_val  (0 negatives -> matched-FP undefined; sdis-anchored thresholds)

Threshold = the confidence at which the SAME model sits at the given neg-image FP rate on sdis_val; recall / tiny-recall on forest_val evaluated at that threshold. Last row: fixed conf 0.50 for reference.

| sdis-FP anchor | v4B1_frozen_L | v4B4_lora_L | v4B5_full_L | v4B6_lora_Hplus | YOLO_A_ctrlE | YOLO_E_ET | YOLO_C_EGT |
|---|---|---|---|---|---|---|---|
| 1% | 0.042 / 0.000 (c=0.986) | 0.006 / 0.000 (c=0.997) | n/a | 0.010 / 0.000 (c=0.997) | 0.002 / 0.000 (c=0.801) | 0.002 / 0.000 (c=0.819) | 0.001 / 0.000 (c=0.812) |
| 2% | 0.060 / 0.000 (c=0.980) | 0.012 / 0.000 (c=0.996) | 0.020 / 0.000 (c=0.999) | 0.020 / 0.000 (c=0.996) | 0.005 / 0.000 (c=0.768) | 0.002 / 0.000 (c=0.800) | 0.002 / 0.000 (c=0.781) |
| 5% | 0.097 / 0.000 (c=0.968) | 0.035 / 0.000 (c=0.993) | 0.057 / 0.002 (c=0.998) | 0.047 / 0.000 (c=0.994) | 0.013 / 0.000 (c=0.717) | 0.009 / 0.000 (c=0.746) | 0.007 / 0.000 (c=0.735) |
| 10% | 0.147 / 0.007 (c=0.941) | 0.081 / 0.001 (c=0.988) | 0.093 / 0.015 (c=0.996) | 0.091 / 0.007 (c=0.990) | 0.020 / 0.000 (c=0.661) | 0.015 / 0.000 (c=0.691) | 0.014 / 0.000 (c=0.684) |
| 20% | 0.224 / 0.034 (c=0.868) | 0.160 / 0.017 (c=0.973) | 0.184 / 0.091 (c=0.992) | 0.185 / 0.076 (c=0.980) | 0.046 / 0.000 (c=0.547) | 0.072 / 0.082 (c=0.566) | 0.051 / 0.014 (c=0.587) |
| conf=0.50 | 0.383 / 0.147 | 0.519 / 0.266 | 0.489 / 0.280 | 0.553 / 0.261 | 0.066 / 0.002 | 0.124 / 0.140 | 0.105 / 0.077 |

## Confounds and caveats (read before comparing)

- **Batch size**: B5 (full unfreeze) trains at bs 2/GPU (eff. 16) vs bs 4/GPU (eff. 32) for B1/B4/B6 -> 2x the optimizer steps for the same 10 epochs (32,140 vs 16,070), different warmup fraction, different effective noise. B5-vs-others is confounded by batch size and LR schedule granularity.
- **Backbone LR**: LoRA runs use 1e-4 on the adapters, B5 uses 1e-5 on all weights; not a like-for-like LR sweep — each is the notes' recommended default for its mode.
- **Checkpoint selection**: best.pt is picked by COCO mAP on a 1,500-image sdis-val subset at epoch boundaries. B5's dataloader pass (3,214 steps at bs 2) never completes inside a ~50-min segment, so B5 evaluates ONLY at the end -> its best.pt == final weights, while B1/B4/B6 got several epoch-boundary evals and may benefit from selection. Selecting on val also mildly optimistic-biases every sdis-val number here (same as v3).
- **Segment resume**: on resume the sampler restarts the current epoch's permutation from the start (v3 behavior), so per-image exposure is uneven across the corpus; epochs above are step-derived.
- **Backbone size (B6 vs B4)**: H+ also changes the neck input dim (1280 vs 1024 -> 10.6M vs 7.8M neck params) and LoRA param count (5.2M vs 3.1M); ~2x slower per step.
- **YOLO baselines**: different input pipeline (1280 square vs 1280x736 letterbox), different training length/recipe; compare only at matched neg-image FP, never by confidence.
- **boreal_val** is a single burn (heinola) with 97% large boxes; **forest_val** has no negatives, so its recall is only meaningful at externally anchored thresholds (sdis-anchored above) and says nothing about false alarms on Australian forest backgrounds.
- **Score saturation / threshold transfer**: adapted backbones (LoRA, full) push Faster R-CNN softmax scores toward 1.0, so the confidence that gives 1-5% neg-FP on sdis_val is >=0.99 for B4/B6 vs ~0.97 for frozen B1 and ~0.8 for YOLO. The sdis-anchored forest_val rows therefore penalize saturated models (a single threshold transfers worse across domains) even where fixed-threshold recall is higher — compare the conf=0.50 row and the boreal_val table before concluding on forest generalization; a deployment would calibrate per domain (or use a per-domain FP target), which needs forest NEGATIVES that this val package does not have.
- expE_y11m_iter2 was trained on E only (iteration 2); arms A/E/C are the iteration-4 YOLO arms. v3_B1/v3_B3 are the iteration-3 DINOv3 ViT-L runs (E-only corpus, 12 epochs, bs 4 / bs 2) — the v4B1-vs-v3_B1 delta isolates the E+T corpus effect for the frozen recipe (10 vs 12 epochs confound).

## Artifacts

- preds: `models/v4/evals/<run>/{sdis_val,boreal_val,forest_val}_preds.jsonl`
- sweeps: `reports/v4/eval_<run>_<set>.json` (+ `.raw.parquet`)
- matched-FP: `reports/v4/matched_fp_dino_{sdis_val,boreal_val,forest_val}.json`
- run/cost summary: `models/v4/dino_run_summary.json`; param counts: `models/v4/param_counts_from_ckpt.json`, `models/v4/backbone_totals.json`
- checkpoints: `models/v3/runs/<run>/seg*/{best,last}.pt` (local) and `/scratch/scratch/tianrgua-sandbox/smoke/runs_v3/<run>/` (cluster)
- logs: `logs/v4/dino_<run>.log`, `logs/v4/eval_<run>.log`; data builders `scripts/v4/yolo_to_coco_eet.py`, `scripts/v4/build_evalval.py`; eval driver `scripts/v4/eval_dino_run.sh`

<!-- figlib39-replay:begin -->
## FIgLib-39 early-detection replay (historical eval set; val-selected thresholds)

Frame scores from the cluster (`cluster/v3/submit_dino_eval.py --frames-tar`, fixed LoRA inference path), replay by `src/eval/replay_eval.py --scores-jsonl` with each run's own sdis-VAL thresholds: conf-single = conf at 10% neg-image FP, conf-seq = conf at 20% (the v3 rule that set B1 to 0.5624/0.3454). R@Nmin = events alerted within N min of the first-visible frame / 39. Pre-visible alerts are listed separately (not counted as detections or as false). FIgLib-39 is historical/open (used since iteration 1) — not a lockbox.

| system | conf-single / conf-seq | detected /39 | R@1 / 3 / 5 min | median delay (detected) | misses | pre-visible alerts |
|---|---|---|---|---|---|---|
| v4B4_lora_L | 0.9881 / 0.973 | 26/39 | 0.026 / 0.077 / 0.128 | 481 s | 13 | 0 |
| v4B6_lora_Hplus | 0.9901 / 0.9801 | 24/39 | 0.026 / 0.077 / 0.154 | 539 s | 15 | 0 |
| expE_y11m (reference) | (v3/v2 val-selected) | 30/39 | 0.179 / 0.308 / 0.436 | 240 s | 9 | 5 |
| v3_B1_frozenL (reference) | (v3/v2 val-selected) | 34/39 | 0.154 / 0.385 / 0.564 | 240 s | 5 | 4 |

**Paired delay vs expE** (events detected by both; delta = run - expE, negative = earlier):

| run | n both | earlier | tie | later | median delta | mean delta |
|---|---|---|---|---|---|---|
| v4B4_lora_L | 25 | 3 | 3 | 19 | +181 s | +288 s |
| v4B4_lora_L vs v3_B1_frozenL | 26 | 0 | 2 | 24 | +240 s | +309 s |
| v4B6_lora_Hplus | 24 | 2 | 5 | 17 | +62 s | +290 s |
| v4B6_lora_Hplus vs v3_B1_frozenL | 24 | 1 | 2 | 21 | +300 s | +372 s |

**Score-transfer diagnostic** (no tuning: distribution of per-frame max smoke score on FIgLib frames at/after first-visible smoke vs the val-selected conf-single):

| system | conf-single | post-visible frames | median max score | p90 | frac >= conf-single | pre-visible frac >= conf-single |
|---|---|---|---|---|---|---|
| v4B4_lora_L | 0.9881 | 1555 | 0.856 | 0.997 | 0.253 | 0.000 |
| v4B6_lora_Hplus | 0.9901 | 1555 | 0.925 | 0.997 | 0.252 | 0.000 |
| expE_y11m (reference) | 0.3 | 1555 | 0.094 | 0.626 | 0.366 | 0.050 |
| v3_B1_frozenL (reference) | 0.5624 | 1555 | 0.656 | 0.963 | 0.548 | 0.014 |

Reading: the adapted runs' sdis-val operating points sit at conf >= 0.97, where only a small fraction of FIgLib post-visible frames score — so events are detected late or missed under the v3 threshold-transfer rule, even though the same runs win on sdis_val at matched FP. This is the calibration/saturation effect flagged in the confounds, now measured on an independent historical set; it argues for per-domain threshold selection (needs domain negatives) before any deployment claim.

- v4B4_lora_L misses: 20160722_FIRE_mg-s-iqeye, 20170807_FIRE_bh-n-mobo-c, 20190717_FIRE_pi-w-mobo-c, 20190826_FIRE_pi-s-mobo-c, 20200202_FIRE_hp-w-mobo-c, 20200806_SpringsFire_sm-e-mobo-m, 20210302_FIRE_lp-e-mobo-m, 20220210-EmeraldFire-marconi-w-mobo-c, 20220405-fire-in-Fallbrook-rm-s-mobo-m, 20231108.221213-AmagoFire-bi-s-mobo-m, 20240130.091751-FIRE-wilson-e-mobo-c, 20240201.064352-FIRE-mlo-n-mobo-c, 20240721_EagleFire_stgo-n-mobo-c
- v4B4_lora_L pre-visible alerts: none
- v4B6_lora_Hplus misses: 20160722_FIRE_mg-s-iqeye, 20170807_FIRE_bh-n-mobo-c, 20180517_FIRE_smer-tcs10-mobo-c, 20180827_Holyflareup_sp-e-mobo-c, 20190717_FIRE_pi-w-mobo-c, 20190826_FIRE_pi-s-mobo-c, 20191007_FIRE_sm-s-mobo-c, 20200202_FIRE_hp-w-mobo-c, 20200806_SpringsFire_sm-e-mobo-m, 20210302_FIRE_lp-e-mobo-m, 20220210-EmeraldFire-marconi-w-mobo-c, 20220405-fire-in-Fallbrook-rm-s-mobo-m, 20231108.221213-AmagoFire-bi-s-mobo-m, 20240201.064352-FIRE-mlo-n-mobo-c, 20240721_EagleFire_stgo-n-mobo-c
- v4B6_lora_Hplus pre-visible alerts: none

### Batch-1 latency, 1x L40S (v3 protocol: first 50 sdis_val images, preprocessing included, decode excluded)

| system | form | mean ms | median ms | p90 ms | infer-only median ms | source |
|---|---|---|---|---|---|---|
| v4B4_lora_L | LoRA unmerged (train form) | 161.8 | 161.6 | 162.9 | 155.8 | `models/v4/evals/v4B4_lora_L_figlib/latency_gpu.json` |
| v4B4_lora_L | LoRA merged into base (deploy form) | 157.5 | 157.6 | 158.7 | 151.7 | `models/v4/evals/v4B4_lora_L_figlib/latency_gpu_merged.json` |
| v4B6_lora_Hplus | LoRA unmerged (train form) | 357.2 | 357.6 | 359.5 | 353.5 | `models/v4/evals/v4B6_lora_Hplus_figlib/latency_gpu.json` |
| v4B6_lora_Hplus | LoRA merged into base (deploy form) | 351.6 | 351.4 | 354.1 | 346.8 | `models/v4/evals/v4B6_lora_Hplus_figlib/latency_gpu_merged.json` |
| v3 B1 frozen ViT-L (reference) | plain | 156.6 | 156.4 | 158.2 | 150.8 | `models/v3/evals/B1val/latency_gpu.json` |
| v3 B2 frozen ViT-H+ (reference) | plain | 372.7 | 371.2 | 383.5 | 366.3 | `models/v3/evals/B2val/latency_gpu.json` |

Merging folds W += (alpha/r)*B@A into the 96 (L) / 128 (H+) attention linears; outputs verified identical to the unmerged model on CPU (max |delta| 1e-6 on box coords/scores). v3 expE YOLO11m reference on the same protocol: 11.2 ms.

Outputs: `reports/v4/replay_{v4B4_lora_L,v4B6_lora_Hplus}_figlib39.json`; frame scores `models/v4/evals/<run>_figlib/figlib39_preds.jsonl`.
<!-- figlib39-replay:end -->


