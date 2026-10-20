# Track D (iteration 3): learned causal temporal detection — working notes

Goal: on REAL chronological pyro-sdis sequences, compare
(1) single-frame expE YOLO11m score, (2) deployed 3-of-5 persistence,
(3) a LEARNED causal temporal head on frozen expE features.
Everything selected on VAL (cabanelle, marguerite). Test/FIgLib/lockbox untouched.

## 1. Temporal sequence dataset (DONE, devbox)
Built by `scripts/v3/build_sequences.py` from `data/manifests/pyro_sdis_manifest.parquet`
(timestamps parsed from filenames; `name_meta_match` true for all 33,636 imgs).
Rules: camera = `{partner}_{site}-{azimuth}`; sort by capture time; cut at gap
> 600 s; keep runs >= 3 frames; TRAIN sites only for train, VAL sites for val;
test sites never read. Output: `data/v3/sequences/sequences_{train,val}.json`,
audit in `data/v3/sequences/audit.json`.

### Sequence stats
| | train | val |
|---|---|---|
| sites | brison 467, courmettes 338, valbonne 87, serre-de-barre 23 | cabanelle 108, marguerite 158 |
| sequences | 915 | 266 |
| frames in sequences | 22,170 (99.97% of split; 6 short runs / 7 frames dropped) | 7,335 (100%) |
| cameras | 21 | 8 |
| pos / neg frames | 18,748 / 3,422 | 6,177 / 1,158 |
| seq length min/med/p90/max | 3 / 14 / 53 / 221 | 4 / 15.5 / 62.5 / 218 |
| median duration | — | 390 s (p90 ~2,071 s) |

### Actual cadence (documented, from kept intra-sequence intervals)
Median interval **29 s** (train & val), p25 13 s, p75 30 s, p90 31–33 s.
NOT 1–2 min: the network cadence is ~30 s, with burst pairs 1–4 s apart
(~22% of all raw intervals <= 10 s — pyro-sdis alert sequences often contain
near-consecutive frames). 4.0% of raw per-camera intervals exceed 600 s and
became sequence cuts. Max kept gap = 600/599 s by construction.

### Label continuity (smoke appearing/disappearing mid-sequence)
| pattern | train | val |
|---|---|---|
| all_pos | 378 | 95 |
| flicker (pos interrupted by neg) | 265 | 97 |
| disappears | 95 | 26 |
| appears | 76 | 12 |
| appears_disappears | 60 | 24 |
| all_neg | 41 | 12 |

~29% (train) / ~36% (val) of sequences FLICKER: smoke-presence labels toggle
mid-sequence. Spot interpretation: faint plumes at the annotation boundary —
per-frame labels are noisy near onset/decay. Consequence: (a) the temporal
head must tolerate label noise; (b) image-level FPR on neg frames embedded in
positive sequences penalizes temporal smoothing on decay — eval reports FPR on
all neg frames AND on "clean" neg frames (no positive frame in the past-5
window) separately.

## 2. pyronear/temporal-model audit (DONE — verdict: inspiration only)
- Pinned: `main` @ `fb7f6ca855cafa8a2ca0981ac54487c63892f6ad` (2026-09-10).
  License: **Apache-2.0** (repo; released model on HF `pyronear/temporal-model`
  also apache-2.0 tagged). Clone kept at /tmp/temporal_model_audit (not vendored).
- Architecture (from source): YOLO11s dets @conf 0.1 -> greedy IoU tube
  linking (miss tolerance 2, fragment merge, gap interpolation) -> ONE
  stabilized 224x224 crop per tube (union of boxes, 1.5x context) -> per-frame
  DINOv2 ViT-S/14 (last block finetuned) -> **bidirectional** 2-layer
  transformer w/ learned pos-emb (hard 20-frame cap, symmetric pad to 6) ->
  one logit per tube -> 4-feature logistic calibrator. NOT causal — online
  behavior is emulated at eval by re-scoring growing prefixes.
- Data: DVC import of pyro-dataset v4.1.0 (~1.6k+1.6k train seqs) on a
  PRIVATE S3 remote; not publicly downloadable. **Overlap with pyro-sdis is
  essentially certain** — their docs hardcode sequences from `brison` (one of
  our TRAIN sites) and reference sdis-07/sdis-77 orgs; their val is NOT held
  out w.r.t. our sites. No FIgLib/HPWREN references found.
- DECISION: use for architecture inspiration only; do NOT download their data
  (private anyway); do NOT compare against their released model on our val
  (contaminated). Ideas adopted: per-frame label-noise-tolerant training,
  temporal augmentation (frame drop / stride), time-to-detection as metric.
  Ideas rejected for Track D: non-causal attention (we need causal, valid at
  T=1), tube-crop pipeline (heavier than our budget; we operate image-level).

## 3. Architecture (frozen expE YOLO11m features + causal window head)
Frozen detector: `models/trained/expE_mix_y11m_1280_v2/seg05/best.pt`,
imgsz 1280 (720p letterboxed to 736x1280), conf floor 0.01.
Per-frame features (extracted once, `src/v3/trackd_features.py`):
- neck maps via forward hooks on layers 16/19/22 (Detect inputs): P3 256x92x160,
  P4 512x46x80, P5 512x23x40; each spatially avg- AND max-pooled to 12x20,
  channels group-mean-reduced to 32 -> 64 ch/scale, 192 ch total;
- 2 detection channels rasterized to 12x20: max-conf paint + det count;
- scalars: per-frame max det conf (= the single-frame baseline score, so all
  three systems share one detector pass), n dets, full det list to jsonl.
Stored fp16 [T,194,12,20] per sequence (~92 KB/frame; ~2.7 GB total).

Temporal head (`src/v3/trackd_temporal.py`): causal, window T<=5 (current +
<=4 past), valid at T=1 by construction:
- shared frame encoder: 1x1 conv 194->64 + GN/SiLU + 3x3 conv 64->64;
- ConvGRU (hidden 64, 3x3 gates) unrolled over the <=5-frame window, front
  padding masked (hidden stays 0 until first real frame); per-step dt-channel
  (dt to previous frame /60, clipped) appended to input;
- head: spatial avg+max pool of final hidden + current-frame max-conf scalar
  -> MLP -> per-frame smoke logit. ~0.3M params.
- gaps: sequences are pre-cut at >600 s; within-window dt is a feature; at
  serving, history reset on stale/scene-change is inherited from AlertEngine.
- augmentation (train only): random window stride in {1,2}, random frame drop
  p=0.1 (dt recomputed) — simulates missed uploads.
- loss: BCEWithLogits, pos_weight = n_neg/n_pos (train frames are 84.6% pos).
Known bias, documented: expE was TRAINED on these train-split images, so
train-time detector features are optimistically confident; head hyperparams /
checkpoint selection therefore on VAL (val sites unseen by expE).

## 4. Training plan (cluster, green g6e, <=1 node at a time)
- Job 1 `trackd_extract`: one GPU job, fetch code+weights+sequence manifests
  via presigned GET, images already on /scratch (`smoke_data_v2`, staged v2,
  contains all 33,636 pyro-sdis imgs), write per-sequence npz to
  /scratch/.../smoke/trackd_v3/features/, resumable per-sequence; tar+PUT
  features back to S3 (train+val) so devbox holds a full copy.
- Job 2 `trackd_train`: tiny-overfit -> profile -> real run (bounded epochs,
  resumable, best-on-val selection); artifacts via presigned PUT.
- Eval on devbox from fetched features + head (`src/v3/trackd_eval.py`).

## 5. Eval protocol (val sequences; matched FPR on val neg frames)
- s1 single-frame: per-frame expE max conf.
- s2 persistence: causal 3rd-largest score of the last-5 window (sweeping a
  threshold over this order statistic traces the exact 3-of-5 ROC); frames
  with <3 frames of history score 0 (deployed logic cannot confirm there);
  deployed operating point (seq 0.15 3-of-5, OR single-frame 0.30) also
  reported as a discrete point.
- s3 learned head logit.
- Report: recall@FPR {0.5,1,2,5,10}% on pos frames (FPR on all 1,158 neg
  frames; secondary FPR on clean negs), PR-AUC/ROC-AUC, tiny-GT-only frame
  recall (max GT area frac < 0.01), and detection latency (frames from smoke
  onset to first alert at matched FPR) on appears/appears_disappears+flicker
  onsets. The learned head must beat s2 to matter.

## 6. What ran (cluster, green g6e us-east-1, 1 node per job, in-quota)
- `tianrgua-extextracts1-q27r` (extract): 8-way GPU-sharded feature
  extraction over all 29,505 sequence frames, ~7 min wall; wrote
  /scratch/.../smoke/trackd_v3/features and PUT feats_train.tgz (1.85 GB) +
  feats_val.tgz (0.61 GB) -> fetched to data/v3/features/ (devbox copy).
- `tianrgua-traheadoverfs1-ui4b` (tiny overfit, 8 seqs): loss 0.22->0.16,
  0.9 s/epoch — pod GPU path + /scratch features verified.
- `tianrgua-tratrackdheas1-eo3b` (real run `trackd_head_v1`): 281,153-param
  head, 12 epochs x 86 steps (bs 256, lr 1e-3 cosine, aug on), 158 s wall,
  train loss 0.152->0.029, best val AP 0.9404 @epoch 8 (train loss keeps
  falling while val AP plateaus — detector-saw-train optimism as predicted).
- `trackd_head_t1` (ablation control: same head/recipe, window = current
  frame only) and seed-1 replicates of both (`trackd_head_v1_seed1`,
  `trackd_head_t1_seed1`) — see §8.
Artifacts: models/v3/trackd/<run>/unpacked/{best.pt,train_log.jsonl};
S3 v3/runs/trackd/. Total GPU spend ≈ 0.5 node-h (5 train jobs ~3 min each +
extract ~7 min), never more than 1 node at a time.

## 7. VAL results (matched FPR on val negative frames; primary = all_neg pool)
Frames: 7,335 (6,177 pos / 1,158 neg; clean negs — no pos frame in past-5
window — 560). 95% of pos frames have only tiny GT (<1% area), so tiny-recall
~= overall recall here. Full numbers: reports/v3/trackD_eval_val.json.

| system | PR-AUC | ROC-AUC | R@FPR1% | R@FPR2% | R@FPR5% |
|---|---|---|---|---|---|
| s1 single-frame expE | 0.9351 | 0.7685 | 0.062 | 0.100 | 0.205 |
| s2 deployed 3-of-5 persistence | 0.9169 | 0.7056 | 0.051 | 0.083 | 0.163 |
| s3 learned causal head | **0.9404** | **0.7736** | **0.071** | **0.151** | **0.250** |

- The learned head beats persistence at every matched FPR (+2.0/+6.9/+8.7 pts
  absolute at 1/2/5% FPR) and beats single-frame at every point too
  (+0.9/+5.1/+4.5 pts).
- **Persistence is WORSE than single-frame on this metric.** Cause: val
  "negative" frames mostly sit inside smoke sequences (flicker/decay), and
  the 3-of-5 window keeps firing on a neg frame whose neighbors are pos.
  On the clean-neg pool (560 negs, no pos in window) the ordering flips:
  s2 R@1%=0.219 > s1 0.141 > s3 0.119 (small-n: threshold set on 560 negs,
  quantile k=5; noisy). Honest read: persistence trades onset latency
  (median 3 vs 1-2 frames) for robustness on isolated negatives; the learned
  head dominates on embedded hard negatives but does not dominate the clean
  pool at the tightest FPR.
- Episode-level (secondary; thresholds matched on FRAME FPR, not episode FP):
  at all_neg@5% s1 detects 46% of pos episodes (median latency 1.5 frames),
  s3 37% (2 frames), s2 27% (3 frames) — s1 fires spiky (few frames/episode,
  more episodes), the head is smoother (more frames, fewer episodes).
- Deployed operating point on these hard negs (context, not a curve):
  confirmed-only FPR 0.535 / recall 0.834; +provisional FPR 0.587 / 0.898.

## 8. Ablation (t1-only control) + seed replicates — VERDICT
Same head capacity/recipe with the window forced to the CURRENT FRAME ONLY,
plus seed-1 replicates of both variants (all_neg pool):

| run | PR-AUC | R@FPR1% | R@FPR2% | R@FPR5% |
|---|---|---|---|---|
| temporal seed0 | 0.9404 | 0.071 | 0.151 | 0.250 |
| temporal seed1 | 0.9391 | 0.084 | 0.181 | 0.246 |
| t1-only seed0 | 0.9431 | 0.064 | 0.126 | 0.259 |
| t1-only seed1 | 0.9399 | 0.054 | 0.113 | 0.245 |
| s1 single-frame | 0.9351 | 0.062 | 0.100 | 0.205 |
| s2 persistence 3-of-5 | 0.9169 | 0.051 | 0.083 | 0.163 |

Honest reading:
1. **The learned head beats deployed persistence at every matched FPR in
   both seeds** — that was the bar. It also beats raw single-frame.
2. **Most of the gain is learned spatial readout, not temporal context**:
   the t1-only control (no history at all) already beats both baselines,
   and PR-AUC differences between temporal and t1-only are within seed noise
   (0.939-0.943).
3. **Temporal context does add a consistent slice at tight FPR**: temporal >
   t1-only at 1% and 2% FPR in 2/2 seeds (mean R@2% 0.166 vs 0.120,
   mean R@1% 0.077 vs 0.059); at 5% FPR the advantage vanishes (0.248 vs
   0.252). Median onset latency at all_neg@5%: 2 frames (~1 min), same as
   t1-only, vs 3+ frames for persistence.
4. Clean-neg pool (560 negs) is too small for stable ordering — numbers
   swing wildly across seeds (temporal cleanR@1% 0.119/0.181, t1-only
   0.056/0.134); reported but inconclusive.
5. Caveats: expE saw the train-split images during its own training, so
   train-time features are optimistic (mitigated by aug + val selection,
   disclosed); per-frame labels are noisy near onset/decay (§1); val negs
   are mostly embedded in smoke sequences — an ops FP rate per camera-hour
   still cannot be derived from this data. Test/FIgLib/lockbox untouched.
RECOMMENDATION: if a temporal head ships, prefer it over 3-of-5 persistence
at tight alert budgets; but the cheaper first step is replacing the raw
max-conf score with a learned spatial readout of the frozen features
(t1-only head) — most of the benefit, no buffer or state machine.

## Log
- [2026-09-13 ~01:40] sequences built + audited (above); temporal-model audit
  returned; feature/head design frozen.
- [~01:55] extractor + head wiring de-risked on devbox CPU first: 2-seq
  extraction (89 frames, 29 s CPU; per-frame score EXACTLY matches
  reports/eval_expE_val.raw.parquet for the same image -> same detector pass
  semantics as v2 evals), 2-seq tiny overfit (loss 0.69->0.05, AP 0.995).
  Added --t1-only ablation control (same head, current frame only) to
  isolate temporal contribution if the head wins.
- [~02:00] cluster extract job submitted (green g6e 1 node, smoke-detect:v6,
  8-way GPU-sharded, resumable per-sequence on /scratch, feature tars
  presigned-PUT back): job tianrgua-extextracts1-q27r.
- [~02:05] extract SUCCEEDED in one segment (~7 min): 915+266 npz, tars
  fetched to data/v3/features/ (2.4 GB unpacked).
- [~02:10] baselines evaluated locally (s1/s2); pod tiny-overfit passed.
- [~02:15] real run trackd_head_v1 done (158 s, best val AP 0.9404 @ep8).
- [~02:25] t1-only ablation + seed-1 replicates done; §7/§8 finalized.
  Eval artifacts: reports/v3/trackD_eval_val*.json.
