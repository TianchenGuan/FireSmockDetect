# RF-DETR-Large @1280 — post-training evaluation (iteration 3, Track A)

Model: **rfdetrL_1280_e48** — RF-DETR-Large (Apache-2.0 code *and* weights)
fine-tuned from `rf-detr-large-2026.pth` on the frozen iteration-2 train list
(`expB_train.txt`, 38,454 images = pyro-sdis train sites + D-Fire smoke), 48
epochs at resolution 1280, 8x L40S, recipe defaults (EMA, multi-scale
1120..1440, lr 1e-4). Job `tianrgua-rfdetrl1280e48a1-gs9y`, SUCCEEDED,
48/48 epochs, 5h34m wall (20,044 s), one job, no resumes.

Control models (unchanged, re-used as-is): **expE = YOLO11m @1280, 60 ep** and
**expB = YOLO11s @1280**, both trained on the identical image list.

> **Headline.** On our unified protocol at *matched* false-alert rate,
> RF-DETR-Large does **not** beat YOLO11m: recall is within noise on the frozen
> test split (partial AUC 0.501 vs 0.502) and slightly *worse* on val (0.496 vs
> 0.521), and tiny-smoke recall is consistently **worse** (test pAUC 0.382 vs
> 0.413). It *is* better on FIgLib event-level early detection (33/39 vs 30/39
> events at val-matched thresholds, paired delay ≈ equal). It costs ~2.8x the
> GPU latency and ~3.1x the CPU latency of YOLO11m. Verdict: **architecture
> swap alone does not buy the faint/small-smoke gain we are after**; keep
> YOLO11m as the serving default, keep the RF-DETR backend available.

---

## 1. Artifacts (task 1)

`models/v3/rfdetr/` (fetched with `cluster/v3/fetch_run.py`, run dir on
/scratch left intact):

| file | note |
|---|---|
| `checkpoint_best_ema.pth` | EMA weights, best val mAP50:95 = **epoch 37**, 141,649,947 B, sha256 `2cca1fe8cd381b5da3fa4919c118da59b789ee22f9eac8ed562a3f11000743d0` |
| `metrics.csv` | full Lightning metrics log (1,249 rows) |
| `val_curve_training_protocol.csv` | per-epoch val curve, RF-DETR's own (torchmetrics COCO) protocol |
| `train_loss_by_epoch.csv` | mean train loss per epoch (8.83 → 4.74) |
| `console.log`, `scratch_listing.txt`, `PROVENANCE.md` | job console, run-dir listing, provenance |

Still on /scratch (not fetched, ~4.5 GB): `last.ckpt`,
`checkpoint_best_total.pth` (= the EMA best, per console),
`checkpoint_{4,9,…,44}.ckpt`, `last_ema.pth`.

### Training-protocol val mAP curve (EMA, torchmetrics COCO)

| epoch | 0 | 1 | 5 | 10 | 16 | 22 | **37** | 43 | 47 |
|---|---|---|---|---|---|---|---|---|---|
| mAP50 | 0.7632 | **0.7856** | 0.7748 | 0.7751 | 0.7701 | 0.7677 | 0.7654 | 0.7415 | 0.7345 |
| mAP50:95 | 0.4464 | 0.4617 | 0.4677 | 0.4694 | 0.4698 | 0.4695 | **0.4701** | 0.4534 | 0.4474 |
| mAP75 | 0.4655 | 0.4830 | 0.4995 | 0.4983 | 0.5006 | 0.5020 | 0.5022 | 0.4906 | 0.4759 |

Shape of the curve (full table in `val_curve_training_protocol.csv`): mAP50
**peaks at epoch 1** (0.786) and then drifts *down* to 0.735 by epoch 47, while
mAP50:95 creeps up to a very flat optimum around epochs 10-37 (0.469-0.470) and
then also declines. Constant lr 1e-4 (recipe default `lr_drop=100` > 48 epochs,
so no decay ever fires) + 48 epochs on 38k images = mild over-fitting past
~epoch 20. Checkpoint selection (`best_ema` by mAP50:95, epoch 37) is doing
real work here; a 12-16 epoch schedule with decay would very likely reach the
same place for 1/3 of the GPU-hours. This is the *training* protocol, not ours
— see §6.

## 2. Backend implementation (task 2)

`src/serve/backend_rfdetr.py` implements the frozen `src/serve/backends.py`
contract: `detect(img_bgr) -> (list[Detection], timing)`, `filter_by_conf`,
`class_names`, `Detection` from `src.serve.detector`, every detection tagged
`source="full"` (no tiling pathway). `rfdetr` is imported from site-packages if
present, otherwise from the pinned clone `third_party/rf-detr/src`
(@`a57ad2345c07c7822d110ef963b3bc330beb9937`). Only two pure-python packages
had to be added to the devbox venv (`pydeprecate`, `supervision`); torch /
torchvision / transformers pins were untouched.

- The checkpoint records `device="cuda"`; the backend always passes `device`
  explicitly, so CPU serving works.
- Preprocessing mirrors `rfdetr.detr.RFDETR.predict()` exactly (BGR→RGB,
  square resize to 1280x1280 with `antialias=False` bilinear = the training
  resize, ImageNet mean/std) but returns plain floats and needs no
  `supervision` at runtime.
- **Verified**: `scripts/v3/verify_rfdetr_backend.py` compares the backend
  against upstream `predict()` on real val images: identical detection counts,
  **max |score| delta 0.0, max |box| delta 0.0 px**.
- Nothing else was wired: `Settings.backend` default stays `yolo`; the demo
  server, YOLO detector and existing eval defaults are unchanged.

## 3. Unified evaluation protocol (task 3)

Same protocol as every other model: `src/eval/eval_detection.py` semantics
(greedy one-to-one matching, IoU >= 0.4, `CONF_GRID`, size buckets tiny <0.1% /
small 0.1-1% / medium 1-5% / large >5% of image area, image-level FP rate on
annotated negatives), frozen splits (val = cabanelle+marguerite 7,335 imgs /
1,158 negatives; test = ferion+croix-augas 4,124 / 917).

Two small, backwards-compatible extensions were added instead of forking the
eval:

1. `eval_detection.py --backend <name>` (default `yolo`, unchanged path) and
   `--preds-jsonl <file>` — compute the sweep from predictions produced
   elsewhere. GT is always read locally from the frozen YOLO label files.
   The matching/sweep code was factored into `sweep_report()`; **regression
   check: re-running it over the frozen `reports/eval_{expE_val,expE_test,
   expB_val}.raw.parquet` reproduces the published `conf_sweep` blocks
   byte-identically.**
2. `replay_eval.py --backend <name>` and `--scores-jsonl <file>` — replay the
   AlertEngine from precomputed per-frame scores. The detector is single-frame
   and `AlertEngine` is a deterministic function of the per-frame max score, so
   scoring and replay can be separated with no change of result. **Validation:
   rebuilding expE's frame scores from the frozen
   `reports/replay_expE_figlib.json` and replaying them offline reproduces that
   file's summary exactly (30/39 detected, median 240 s, 5 pre-visible).**

`.raw.parquet` files are written by both paths, so cascade / threshold analyses
work as usual.

Both in-process paths were exercised end-to-end on CPU as well
(`--backend rfdetr --weights models/v3/rfdetr/checkpoint_best_ema.pth`): the
live-model replay of event `20160604_FIRE_rm-n-mobo-c` at the val-matched
thresholds returns first alert at +60 s — identical to the offline
scores-JSONL replay of the same event.

**Where inference ran.** RF-DETR at 1280 on CPU is ~1.0 s/img, so val+test
(11,459 images) ran as a GPU cluster job — `cluster/v3/submit_rfdetr_eval.py` +
`rfdetr_eval_entry.sh` + `rfdetr_infer.py`, 8 single-GPU shards on one g6e node
(smoke-det3:v1, weights read straight from the training run dir on /scratch),
~21 img/s/GPU, **3 minutes wall** for both splits including the latency probe.
The pod runs the *same* `src/serve/backend_rfdetr.py` (shipped in the job
tarball), so cluster and devbox numbers cannot drift.
The 3,056 FIgLib frames are **not** staged on the cluster (they live only on the
devbox), so they were scored on the devbox with the identical driver in 22
parallel CPU shards (`scripts/v3/score_frames_cpu.py`, 282 s), then replayed
offline.

**CPU vs GPU numeric parity** (64 val images scored both ways): max |per-image
max-score delta| **8.4e-3**, mean |per-box score delta| 1.3e-3 — TF32 matmuls
on CUDA (`torch.set_float32_matmul_precision("high")` inside `rfdetr`) vs fp32
on CPU. Consequence for FIgLib is bounded in §5.

### 3a. Val sweep (threshold selection happens here, only here)

P / R / tiny-recall / negative-image-FP, IoU>=0.4:

| conf | rfdetrL | expE (y11m) | expB (y11s) |
|---|---|---|---|
| 0.05 | 0.406 / 0.930 / 0.890 / 0.843 | 0.508 / 0.870 / 0.808 / 0.596 | 0.522 / 0.847 / 0.785 / 0.544 |
| 0.10 | 0.550 / 0.905 / 0.857 / 0.743 | 0.621 / 0.830 / 0.760 / 0.524 | 0.625 / 0.804 / 0.737 / 0.489 |
| 0.15 | 0.624 / 0.881 / 0.831 / 0.675 | 0.694 / 0.800 / 0.724 / 0.479 | 0.692 / 0.769 / 0.693 / 0.453 |
| 0.20 | 0.674 / 0.857 / 0.803 / 0.622 | 0.741 / 0.769 / 0.683 / 0.433 | 0.741 / 0.742 / 0.660 / 0.418 |
| 0.25 | 0.714 / 0.832 / 0.773 / 0.586 | 0.777 / 0.735 / 0.641 / 0.390 | 0.778 / 0.712 / 0.623 / 0.382 |
| 0.30 | 0.744 / 0.810 / 0.748 / 0.543 | 0.806 / 0.696 / 0.597 / 0.358 | 0.803 / 0.680 / 0.581 / 0.340 |
| 0.40 | 0.791 / 0.759 / 0.688 / 0.459 | 0.851 / 0.622 / 0.514 / 0.294 | 0.848 / 0.608 / 0.492 / 0.275 |
| 0.50 | 0.821 / 0.699 / 0.615 / 0.380 | 0.882 / 0.523 / 0.410 / 0.226 | 0.878 / 0.519 / 0.390 / 0.216 |
| 0.60 | 0.840 / 0.640 / 0.540 / 0.319 | 0.904 / 0.393 / 0.282 / 0.147 | 0.906 / 0.405 / 0.279 / 0.149 |
| 0.70 | 0.863 / 0.576 / 0.463 / 0.256 | 0.927 / 0.228 / 0.129 / 0.065 | 0.933 / 0.244 / 0.135 / 0.066 |

GT boxes by size (val): tiny 2,277 / small 4,027 / medium 260 / large 39.

Raw confidences are **not comparable across architectures** — at the same
nominal conf RF-DETR fires far more often (conf 0.30 → 54% of negative images
vs YOLO11m's 36%). Every cross-model claim below is therefore made at matched
negative-image FP (`scripts/v3/compare_matched_fp.py`: fine 0.01 conf grid,
identical matching code, linear interpolation to the FP target).

**Val, matched FP** (conf / recall / tiny-recall / precision):

| neg-img FP | rfdetrL | expE (y11m) | expB (y11s) |
|---|---|---|---|
| 0.05 | 0.87 / 0.203 / 0.102 / 0.930 | 0.72 / 0.190 / 0.097 / 0.932 | 0.72 / 0.203 / 0.104 / 0.941 |
| 0.10 | 0.85 / 0.315 / 0.194 / 0.910 | 0.65 / 0.309 / 0.201 / 0.915 | 0.65 / 0.329 / 0.203 / 0.925 |
| 0.15 | 0.81 / 0.415 / 0.292 / 0.897 | 0.60 / 0.399 / 0.287 / 0.904 | 0.60 / 0.406 / 0.280 / 0.906 |
| 0.20 | 0.77 / 0.489 / 0.368 / 0.881 | 0.54 / 0.480 / 0.364 / 0.888 | 0.53 / 0.484 / 0.359 / 0.886 |
| 0.30 | 0.63 / 0.623 / 0.519 / 0.845 | 0.38 / 0.635 / 0.529 / 0.845 | 0.36 / 0.633 / 0.523 / 0.833 |
| 0.40 | 0.48 / 0.710 / 0.626 / 0.815 | 0.24 / **0.744** / **0.651** / 0.771 | 0.23 / 0.727 / 0.644 / 0.758 |
| 0.50 | 0.36 / 0.777 / 0.706 / 0.773 | 0.13 / **0.811** / **0.740** / 0.668 | 0.09 / 0.813 / 0.744 / 0.609 |

Partial AUC of recall vs neg-image FP over FP<=0.5 (normalized):
**rfdetrL 0.496 / expE 0.521 / expB 0.523**; tiny-recall pAUC **0.398 / 0.422 /
0.420**. RF-DETR is *ahead* only in the very tight FP<=0.2 corner (by ~1-2 pts
recall) and behind in the useful 0.3-0.5 range; it does hold a large precision
advantage at high FP (0.77 vs 0.67 at FP 0.5), i.e. fewer *extra boxes* per
alerting image, which our image-level FP metric does not reward.

**Protocol-artifact check (val only).** DETR heads emit 300 queries with no
NMS, and our matcher charges every unmatched prediction as an FP. Applying
class-agnostic NMS at IoU 0.55 (the YOLO backend's `merge_iou`) removes 22% of
predicted boxes (31,182 → 24,191) and changes the matched-FP table by **<=0.001
in every cell** (`reports/v3/eval_rfdetrL_e48_val_nms.json`,
`matched_fp_val_nms.json`) — the duplicates sit below the operating
thresholds. The gap is real, not a protocol artifact.

### 3b. Frozen test sweep (single sweep, run once)

| conf | rfdetrL | expE (y11m) | expB (y11s) |
|---|---|---|---|
| 0.05 | 0.324 / 0.885 / 0.820 / 0.789 | 0.435 / 0.845 / 0.783 / 0.644 | 0.483 / 0.771 / 0.712 / 0.509 |
| 0.10 | 0.463 / 0.827 / 0.749 / 0.656 | 0.537 / 0.794 / 0.725 / 0.551 | 0.578 / 0.721 / 0.647 / 0.442 |
| 0.15 | 0.537 / 0.787 / 0.696 / 0.579 | 0.609 / 0.749 / 0.670 / 0.482 | 0.631 / 0.679 / 0.586 / 0.394 |
| 0.20 | 0.595 / 0.754 / 0.665 / 0.497 | 0.663 / 0.711 / 0.627 / 0.429 | 0.684 / 0.646 / 0.541 / 0.353 |
| 0.25 | 0.637 / 0.725 / 0.625 / 0.450 | 0.711 / 0.677 / 0.588 / 0.366 | 0.721 / 0.618 / 0.503 / 0.313 |
| 0.30 | 0.675 / 0.700 / 0.592 / 0.410 | 0.744 / 0.643 / 0.549 / 0.328 | 0.761 / 0.586 / 0.466 / 0.270 |
| 0.40 | 0.737 / 0.644 / 0.524 / 0.336 | 0.815 / 0.560 / 0.461 / 0.236 | 0.815 / 0.508 / 0.373 / 0.206 |
| 0.50 | 0.774 / 0.593 / 0.462 / 0.289 | 0.850 / 0.462 / 0.367 / 0.172 | 0.856 / 0.423 / 0.275 / 0.148 |
| 0.60 | 0.796 / 0.534 / 0.401 / 0.236 | 0.895 / 0.348 / 0.256 / 0.099 | 0.898 / 0.328 / 0.177 / 0.085 |
| 0.70 | 0.828 / 0.471 / 0.330 / 0.185 | 0.936 / 0.213 / 0.112 / 0.037 | 0.942 / 0.202 / 0.073 / 0.026 |

GT boxes by size (test): tiny 1,550 / small 1,740 / medium 219 / large 53.

**Test, matched FP** (conf / recall / tiny-recall / precision):

| neg-img FP | rfdetrL | expE (y11m) | expB (y11s) |
|---|---|---|---|
| 0.05 | 0.86 / 0.190 / 0.076 / 0.890 | 0.67 / **0.256** / **0.154** / 0.924 | 0.66 / 0.258 / 0.115 / 0.923 |
| 0.10 | 0.81 / 0.307 / 0.164 / 0.866 | 0.60 / **0.350** / **0.258** / 0.895 | 0.58 / 0.354 / 0.206 / 0.887 |
| 0.15 | 0.75 / 0.417 / 0.276 / 0.841 | 0.52 / 0.446 / **0.353** / 0.861 | 0.50 / 0.426 / 0.277 / 0.855 |
| 0.20 | 0.67 / 0.492 / 0.353 / 0.818 | 0.46 / 0.506 / **0.406** / 0.835 | 0.41 / 0.501 / 0.364 / 0.820 |
| 0.30 | 0.47 / 0.610 / 0.482 / 0.764 | 0.34 / 0.614 / 0.514 / 0.769 | 0.26 / 0.610 / 0.490 / 0.732 |
| 0.40 | 0.31 / 0.695 / 0.585 / 0.682 | 0.22 / 0.697 / 0.613 / 0.680 | 0.14 / 0.689 / 0.603 / 0.622 |
| 0.50 | 0.20 / 0.755 / 0.666 / 0.594 | 0.13 / 0.767 / 0.693 / 0.585 | 0.06 / 0.767 / 0.708 / 0.496 |

pAUC (FP<=0.5): recall **rfdetrL 0.501 / expE 0.502 / expB 0.508**; tiny-recall
**0.382 / 0.413 / 0.394**. Overall recall is a tie; **tiny-smoke recall is
clearly worse for RF-DETR on the test sites** (e.g. -0.09 absolute at FP 0.10,
-0.05 at FP 0.20) — the opposite of what the architecture was tried for.

## 4. Threshold selection for RF-DETR (val only)

Score distributions differ, so the deployed YOLO thresholds (single-frame 0.30 /
seq 0.15, 3-of-5) are not transferable. Selection rule (val only, documented,
applied once): **match expE's val negative-image FP rate at each deployed
threshold**.

| role | expE conf | expE val neg-FP | rfdetrL conf (matched) |
|---|---|---|---|
| single-frame provisional | 0.30 | 0.358 | **0.535** |
| sequence "hit" (3-of-5) | 0.15 | 0.479 | **0.382** |

## 5. FIgLib causal replay — 39 events / 3,056 frames (task 3b)

Frame scores from the devbox CPU shards; alert logic = unmodified
`src/serve/alerting.AlertEngine`, fresh state per event, chronological order,
`seq_window=5`, `seq_hits_required=3`, `conf_floor=0.05`. Both threshold sets
reported (the second is the val-selected one from §4):

| model / thresholds | detected | missed | median delay (s) | mean delay (s) | pre-visible alerts |
|---|---|---|---|---|---|
| rfdetrL @ deployed 0.30/0.15 | **35/39** | 4 | **120** | 250 | 13 |
| rfdetrL @ val-matched 0.535/0.382 | **33/39** | 6 | 300 | 449 | 6 |
| expE y11m @ 0.30/0.15 | 30/39 | 9 | 240 | 404 | 5 |
| expB y11s @ 0.30/0.15 | 32/39 | 7 | 475 | 549 | 5 |
| baseline pyronear y11s @ 0.30/0.15 (CONTAMINATED — trained on FIgLib) | 37/39 | 2 | 61 | 261 | 14 |

Paired per-event comparison, rfdetrL @val-matched vs expE @deployed: 28 events
detected by both, **5 detected only by RF-DETR**, 2 only by expE. On the 28
shared events the delay difference (RF-DETR − expE) is median **0 s**, mean
−32 s (12 earlier / 8 later / 8 identical). So the gain is "a few more events
found", not "systematically earlier".

Pre-visible alerts (before FIgLib's first-visible-plume frame) rise from 5
(expE) to 6 at val-matched thresholds, but to 13 at the deployed 0.30 — another
sign the raw thresholds are not transferable; those are reported, not counted
as errors (smoke can be faintly visible earlier), but 13/35 would need review
before shipping.

**Sensitivity to the CPU-vs-GPU score noise (§3):** perturbing every frame score
by ±0.01 (>= the observed 8.4e-3 max deviation) moves the deployed-threshold
result by at most one event (35 → 34) and the val-matched median delay by one
frame interval (300 → 240 s); the val-matched detected count is unchanged
(33/39). Frames sitting within ±0.01 of a threshold: 18 at 0.30, 27 at 0.535,
9 at 0.382, out of 3,056.

FIgLib caveats unchanged: fixed-camera imagery (no drone generalization claim),
event-level alerting only (no boxes), delay measured from the first-visible
frame, not ignition. The iteration-3 lockbox (25 fresh events) was **not**
touched.

## 6. Latency — batch 1, including preprocessing (task 3c)

Decode excluded, resize+normalize+forward+postprocess included; 3 warmup
frames; 12-50 timed frames from val.

| config | RF-DETR-L @1280 | YOLO11m (expE) @1280 | ratio |
|---|---|---|---|
| 1x L40S GPU (g6e pod, fp32) | **31.4 ms** median (mean 31.5, p90 31.6; 26.4 ms model + 4.3 ms preprocess) | **11.2 ms** median (mean 11.4, p90 11.5) | **2.8x** |
| devbox CPU, 8 threads | **1016 ms** median (1006 ms model) | **324 ms** median | **3.1x** |
| devbox CPU, 32 threads | **416 ms** median | **312 ms** median | 1.3x |
| devbox CPU, all 192 threads | ~6,400 ms (thread thrash; not a usable config) | — | — |

Throughput in the sharded eval job: 21 img/s per L40S including JPEG decode
(11,459 images on 8 GPUs in ~3 min). RF-DETR-L has 35.6M params at 1280 (PE
grows from the 704-native 33.9M) vs YOLO11m's 20.1M; no fp16/TensorRT
optimization was applied (`model.inference(dtype=torch.float16)` is available
and would likely close much of the GPU gap — untested).

Reference points from earlier iterations: YOLO11s @1024 CPU 140-230 ms; the
9B VLM 6.5 s/img on a sharded L40S.

## 7. Training-protocol vs our-protocol numbers

| protocol | metric | rfdetrL | expE y11m |
|---|---|---|---|
| framework-internal, val | mAP50 | 0.765 (best-EMA epoch 37; 0.786 peak @ep1) | 0.747 |
| framework-internal, val | mAP50:95 | 0.470 | 0.502 |
| framework-internal, val | P / R | 0.722 / 0.781 | 0.735 / 0.691 |
| ours (IoU>=0.4, conf 0.40), val | P / R | 0.791 / 0.759 | 0.851 / 0.622 |

Why the framework numbers cannot be compared to each other or to ours:

1. **Different implementations of "mAP"**: RF-DETR logs torchmetrics COCO mAP
   over its 300 un-NMS'd queries; ultralytics computes its own mAP after NMS
   with its own conf/IoU defaults and its own P/R definition (max-F1 point).
   RF-DETR wins on mAP50 (+0.018) and loses on mAP50:95 (−0.032) — i.e. the two
   frameworks even disagree about which model is better, at the same nominal
   task.
2. **Averaged over IoU thresholds vs one fuzzy threshold**: our protocol fixes
   IoU>=0.4 deliberately (smoke has no crisp boundary), so mAP50:95's
   localization-tightness component — where YOLO leads — is intentionally out
   of scope. Conversely mAP integrates over *all* score thresholds, including
   the useless high-FP tail, while we evaluate at operating points.
3. **One-to-one greedy matching + image-level FP on annotated negatives**: our
   metric charges duplicate boxes and any box on a negative image; COCO mAP
   only pushes them down the ranking. This is why RF-DETR's superior raw recall
   ceiling (0.930 val recall @conf 0.05 vs 0.870) does not convert into a win
   at matched alert rate.
4. **Selection metric**: the checkpoint we evaluate was selected by the
   framework on *its* metric (mAP50:95, epoch 37) — not the operating point we
   care about. Epoch 1 had the best mAP50; epochs 5-22 were within noise. A
   selection-on-our-metric sweep over the retained epoch checkpoints was not
   run (each is 566 MB on /scratch; ~3 min of GPU per candidate would make it
   cheap if this model is revisited).

Net: the "+0.018 mAP50 over YOLO11m" headline a framework-metric comparison
would have produced is **not real** under the protocol that matches our product
(alert at controlled false-alert rate on faint/small smoke).

## 8. Files

Code (new):
- `src/serve/backend_rfdetr.py` — serving backend (contract-compliant).
- `scripts/v3/verify_rfdetr_backend.py` — parity vs upstream `predict()` + CPU latency.
- `cluster/v3/rfdetr_infer.py` — shard inference driver (pod + devbox, one code path).
- `cluster/v3/rfdetr_eval_entry.sh`, `cluster/v3/submit_rfdetr_eval.py` — GPU eval job.
- `cluster/v3/yolo_latency.py`, `cluster/v3/submit_yolo_latency.py` — YOLO11m latency control (smoke-detect:v6 has ultralytics; smoke-det3:v1 does not).
- `scripts/v3/score_frames_cpu.py` — parallel CPU frame scoring (FIgLib).
- `scripts/v3/compare_matched_fp.py` — matched-FP comparison + partial AUCs.
- `scripts/v3/nms_preds.py` — NMS sensitivity check.

Code (extended, defaults unchanged): `src/eval/eval_detection.py`
(`--backend`, `--preds-jsonl`, `sweep_report()`), `src/eval/replay_eval.py`
(`--backend`, `--scores-jsonl`).

Results:
- `reports/v3/eval_rfdetrL_e48_val.json` (+`.raw.parquet`), `..._val_nms.json`, `..._test.json` (+`.raw.parquet`)
- `reports/v3/matched_fp_val.json`, `matched_fp_val_nms.json`, `matched_fp_test.json`
- `reports/v3/replay_rfdetrL_e48_figlib_deployedthr.json`, `..._valmatched.json`
- `reports/v3/latency_rfdetr_cpu8.json`, `latency_rfdetr_cpu32.json`, `latency_yolo11m_cpu8.json`, `latency_yolo11m_cpu32.json`
- `models/v3/evals/rfdetrLe48vt/{sdis_val,sdis_test}_preds.jsonl(.gz)`, `models/v3/evals/rfdetrLe48_figlib/figlib39_preds.jsonl`, `models/v3/evals/rfdetrLe48lat/latency_gpu.json`, `models/v3/evals/yolo_latency/latency_gpu_yolo.json`
- `data/v3/figlib39_frames.txt` (frame list used for scoring)

Reproduce:
```bash
# GPU inference for val + test (3 min on one g6e node) -> preds jsonl
python cluster/v3/submit_rfdetr_eval.py --eval-name rfdetrLe48vt \
    --splits sdis_val --splits sdis_test --latency-n 50
# sweeps (offline, identical matching code as every other model)
python -m src.eval.eval_detection --data-yaml data/yolo/yamls/local/expA.yaml \
    --split val --backend rfdetr --imgsz 1280 \
    --preds-jsonl models/v3/evals/rfdetrLe48vt/sdis_val_preds.jsonl \
    --out reports/v3/eval_rfdetrL_e48_val.json
# FIgLib: score on devbox CPU, replay offline
python scripts/v3/score_frames_cpu.py --weights models/v3/rfdetr/checkpoint_best_ema.pth \
    --list data/v3/figlib39_frames.txt --workers 22 --threads 8 \
    --out models/v3/evals/rfdetrLe48_figlib/figlib39_preds.jsonl
python -m src.eval.replay_eval --manifest data/manifests/figlib_subset_manifest.parquet \
    --scores-jsonl models/v3/evals/rfdetrLe48_figlib/figlib39_preds.jsonl \
    --backend rfdetr --imgsz 1280 --conf-single 0.5349 --conf-seq 0.382 \
    --out reports/v3/replay_rfdetrL_e48_figlib_valmatched.json
# serving (not enabled by default)
SMOKE_BACKEND=rfdetr SMOKE_WEIGHTS_PATH=models/v3/rfdetr/checkpoint_best_ema.pth \
SMOKE_IMGSZ=1280 SMOKE_SINGLE_FRAME_ALERT_CONF=0.535 SMOKE_SEQ_FRAME_CONF=0.382 ...
```

## 9. Limits / what this does and does not show

- One architecture point, one recipe, one seed. RF-DETR was run with its own
  default augmentations (multi-scale + square resize) and YOLO with
  ultralytics' — the comparison is "architecture+recipe", as stated in
  `trackA_notes.md` §3, not a controlled-augmentation ablation.
- No lr-decay / shorter-schedule variant was tried, and the checkpoint was
  selected by the framework's mAP50:95. A shorter schedule with decay, or
  selection on our matched-FP metric, could move RF-DETR by a small amount; it
  would not plausibly close a tiny-recall gap of 0.03-0.09 absolute.
- No fp16/TensorRT inference optimization (would improve GPU latency only).
- Frozen test was swept exactly **once**; the iteration-3 FIgLib lockbox was
  not touched. FIgLib numbers for our own models are contamination-free; the
  pyronear baseline row is contaminated and shown only for orientation.
- `n=39` events: differences of 2-3 events are within binomial noise
  (95% CI on 33/39 is roughly ±0.13 in rate).
