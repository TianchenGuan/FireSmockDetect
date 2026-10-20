# Track B/C results: DINOv3 backbones for smoke detection (iteration 3)

Date: 2026-09-14. Gated access to the three DINOv3 checkpoints landed at
~01:08 UTC; staging, five training runs, all evaluation and this report
happened inside the remaining ~7h of the iteration budget (boundary 08:10
UTC). Prep, stack selection and licensing: `reports/v3/trackBC_prep.md`.

> **Headline.** With head, data, resolution and protocol held fixed, **no
> DINOv3 backbone variant beats the YOLO11m baseline (expE) where it counts.**
> The most interesting run — C1, the frozen **7B** backbone with a head
> trained on cached features — **wins on val** (recall pAUC 0.531 vs expE
> 0.520; tiny-smoke pAUC 0.430 vs 0.422, with striking gains at tight FP:
> tiny-recall 0.161 vs 0.097 at FP 0.05) and was therefore selected, per the
> pre-registered rule, for the single frozen-test sweep — where it
> **collapses: pAUC 0.469 vs expE 0.526, tiny 0.304 vs 0.435.** The val win
> is site-specific and does not transfer (an inference-path artifact was
> ruled out by an exact-parity check, §2d). The best *robust* variant, frozen
> ViT-H+ (B2), reaches only val parity (0.517) at 33x expE's GPU latency.
> Frozen-backbone scaling L→H+→7B does buy monotonic val gains
> (0.487→0.517→0.531-with-caveats); top-4-block unfreezing (B3) improves
> COCO-style mAP but *hurts* the operating-point metric — the same
> framework-metric mirage documented for RF-DETR. FIgLib event replay:
> B1 finds more events than expE (34/39 vs 30/39) at equal median delay, but
> C1 does not (30/39, slower). **YOLO11m stays the serving default.**

## 1. What trained

All runs: frozen expB_train list (38,454 imgs = pyro-sdis train sites +
D-Fire smoke, incl. 11,257 negatives), 1280x736 letterbox (patch-16 grid
80x46), identical ViTDet-style SimpleFeaturePyramid + torchvision Faster
R-CNN head, AdamW lr 1e-4 (head), cosine schedule + 500 warmup, bs 32
global, bf16 autocast, seed 0, 8x L40S (green g6e), segmented ~50-min jobs
with /scratch resume. Backbones: pinned gated snapshots (L `ea8dc286`,
H+ `c807c9ee`, 7B `b8036775`), staged byte-verified devbox→S3→/scratch.

| run | backbone | trainable / total | backbone (train/total) | head+neck | epochs | steps | s/it | wall |
|---|---|---|---|---|---|---|---|---|
| B1 | dinov3-vitl16 **frozen** | **22.31M** / 325.43M | 0 / 303.13M | 22.31M | 12 | 14,412 | 0.33 | 1.4h, 2 segs |
| B2 | dinov3-vith16plus **frozen** | **25.14M** / 865.73M | 0 / 840.59M | 25.14M | 12 | 14,412 | 0.54 | 2.5h, 4 segs |
| B3 | vitl16, **top-4 of 24 blocks unfrozen** @0.1x LR, grad-ckpt, bs 2/GPU | **72.70M** / 325.43M | 50.39M / 303.13M | 22.31M | 12 | 28,836 | 0.29 | 2.6h, 4 segs |
| C1 | dinov3-vit7b16 **frozen — head trained on cached bf16 features, no aug** | **95.25M** / 6.811B | 0 / 6.716B | 95.25M (4096-d in) | 6 | 7,206 | ~1.02 (I/O-bound) | 2.4h, 4 segs (+2 segs precompute) |
| B1cached | vitl16 frozen, cached features, no aug — **control for C1** | **22.31M** / 325.43M | 0 / 303.13M | 22.31M | 6 | 7,206 | ~0.42 | 0.8h, 2 segs |

**None of these is a full fine-tune** — B1/B2/C1/B1cached are frozen-backbone
runs, B3 is a partial (top-4-block) unfreeze. Checkpoints store trainable
state only; frozen weights come from the pinned snapshots.

Schedule decisions forced by the 24h boundary (disclosed):
- **C1 and B1cached ran 6 epochs, not 12.** The 7B cached head is I/O-bound
  (~1.02 s/it on 30.1 MB/img reads); 12 epochs would have overrun the budget.
  Both cached runs were cut to the SAME exposure so the 7B-vs-L cached
  comparison stays clean. All 12-epoch runs plateau by epoch ~5 on the
  in-training metric. Both cached runs briefly ran on a 12-epoch cosine
  schedule (first ~1.5-2.8k steps) before a checkpoint-resume onto the
  6-epoch schedule (a documented LR-schedule splice). C1's in-training curve
  was still rising at cutoff (0.255→0.259 over the last epoch) — it is
  mildly undertrained relative to the others.
- 7B feature precompute (train+val, 45,789 imgs): 2 segments, 1.95 img/s/GPU
  (0.51 s/img — 1.6x faster than the prep estimate), ~1.3 TB bf16 cache on
  /scratch. L-feature cache for the control: 1 segment, ~0.35 TB.
- Best checkpoints by in-training COCO mAP on a fixed 1,500-img val subset:
  B1 step 6,005 (mAP 0.223 / AP50 0.557), B2 11,806 (0.240 / 0.596),
  B3 16,453 (0.269 / 0.626), B1cached 5,603 (0.246 / 0.593),
  C1 7,206 (0.259 / 0.622).

One real bug found on the way in (and the reason profiles exist):
`det_modeling._blocks()` did not know `DINOv3ViTModel`'s layout
(`hf.model.layer`; the dinov2 stand-in uses `hf.encoder.layer`), so the B3
profile crashed at model build. Fixed and CPU-verified (grads reach exactly
the top-4 blocks; 72.7M trainable) before the long run — the
"weight-layout surprise" flagged in prep §7, cost: one 6-minute profile job.

## 2. Unified evaluation

Protocol identical to every model in this project (`src/eval/eval_detection.py`:
greedy 1-1 matching, IoU>=0.4, size buckets, image-level FP on annotated
negatives; val = frozen sdis_val 7,335 imgs / 1,158 neg; test = frozen
ferion+croix-augas 4,124 / 917). GPU jobs produce predictions only
(`cluster/v3/dino_infer.py` — same letterbox/normalize code as training —
via `submit_dino_eval.py`); sweeps run offline on the devbox; matched-FP
comparison via `scripts/v3/compare_matched_fp.py`.

### 2a. Val, matched negative-image FP (conf / recall / tiny-recall / precision)

| FP | B1 frozen-L | B2 frozen-H+ | B3 L-top4 | B1cached noaug | C1 7B cached | expE y11m | expB y11s | rfdetrL |
|---|---|---|---|---|---|---|---|---|
| 0.05 | .94/.173/.082/.901 | .97/.196/.124/.906 | .99/.154/.089/.916 | .96/.154/.082/.918 | .97/**.228**/**.161**/.909 | .72/.190/.097/.932 | .72/.203/.104/.941 | .87/.203/.102/.930 |
| 0.10 | .90/.275/.161/.869 | .95/.292/.205/.875 | .98/.264/.159/.896 | .94/.281/.175/.880 | .94/**.353**/**.257**/.868 | .65/.309/.201/.915 | .65/.329/.203/.925 | .85/.315/.194/.910 |
| 0.15 | .85/.387/.260/.837 | .92/.394/.291/.848 | .96/.356/.228/.875 | .91/.387/.269/.848 | .93/**.429**/**.319**/.842 | .60/.399/.287/.904 | .60/.406/.280/.906 | .81/.415/.292/.897 |
| 0.20 | .80/.457/.329/.808 | .86/.506/.392/.811 | .95/.423/.280/.856 | .88/.459/.325/.812 | .89/**.520**/**.399**/.817 | .54/.480/.364/.888 | .53/.484/.359/.886 | .77/.489/.368/.881 |
| 0.30 | .65/.599/.482/.729 | .73/.618/.499/.741 | .88/.564/.415/.805 | .80/.577/.440/.733 | .82/.625/.506/.751 | .38/**.635**/**.529**/.845 | .36/.633/.523/.833 | .63/.623/.519/.845 |
| 0.40 | .49/.685/.579/.638 | .49/.723/.608/.637 | .70/.698/.582/.729 | .67/.668/.536/.646 | .65/.727/.620/.654 | .24/**.744**/**.651**/.771 | .23/.727/.644/.758 | .48/.710/.626/.815 |
| 0.50 | .31/.750/.653/.519 | .27/.786/.688/.522 | .38/.782/.678/.609 | .45/.757/.627/.515 | .44/.796/.701/.550 | .13/**.811**/**.740**/.668 | .09/.813/.744/.609 | .36/.777/.706/.773 |

Partial AUC of (tiny-)recall vs neg-image FP over FP<=0.5, normalized — ONE
consistent trapezoid integration for every model in this report (values for
expE/expB/rfdetrL differ by <=0.02 from the integration used in
`rfdetr_eval.md`; all orderings unchanged):

| model | val pAUC recall | val pAUC tiny | test pAUC recall | test pAUC tiny |
|---|---|---|---|---|
| **C1 7B cached noaug** | **0.531** | **0.430** | **0.469** | **0.304** |
| expB y11s | 0.523 | 0.420 | 0.521 | 0.405 |
| **expE y11m (baseline)** | 0.520 | 0.422 | **0.526** | **0.435** |
| B2 frozen ViT-H+ | 0.517 | 0.417 | — (not tested, §2b) | — |
| rfdetrL | 0.515 | 0.415 | 0.505 | 0.385 |
| B1 frozen ViT-L | 0.487 | 0.378 | — | — |
| B1cached noaug (control) | 0.477 | 0.358 | — | — |
| B3 L top-4 unfrozen | 0.474 | 0.357 | — | — |

Readings (val):
- **Frozen-backbone scaling is real on val and monotonic**: L 0.487 → H+
  0.517 → 7B 0.531 (cached pair: L-cached 0.477 → 7B-cached 0.531, +0.054 at
  identical head/exposure/no-aug). C1's tight-FP tiny-smoke numbers are the
  best this project has produced on val (+0.066 tiny-recall over expE at
  FP 0.05, +0.056 at 0.10) — exactly the product corner. But see §2b.
- **B3 (partial unfreeze) is a COCO-metric mirage**: best in-training mAP of
  all runs (0.269 vs B1's 0.223) yet the worst operating-point pAUC (0.474
  vs B1's 0.487). Unfreezing tightened localization/confidence — rewarded by
  mAP, worthless at matched image-level FP. Same lesson as `rfdetr_eval.md`
  §7; framework-style metrics would have called B3 the winner.
- **Caching/no-aug control**: B1cached (0.477) sits 0.010 below end-to-end
  B1 (0.487) — the frozen-augmentation cost of the cached route is about one
  pAUC point on val (on the in-training subset metric the control looked
  *better*, 0.246 vs 0.223 — another COCO-metric inversion).

### 2b. Frozen test — swept ONCE, for the single best-on-val B/C model (C1)

Selection rule (pre-registered in the runbook): the single best B/C model on
val gets the one test sweep. That was **C1** (val pAUC 0.531). Result:

| FP | C1 7B cached | expE y11m | expB y11s | rfdetrL |
|---|---|---|---|---|
| 0.05 | .97/.155/.094/.834 | .67/**.256**/**.154**/.924 | .66/.258/.115/.923 | .86/.190/.076/.890 |
| 0.10 | .94/.300/.164/.780 | .60/**.350**/**.258**/.895 | .58/.354/.206/.887 | .81/.307/.164/.866 |
| 0.15 | .91/.399/.234/.721 | .52/**.446**/**.353**/.861 | .50/.426/.277/.855 | .75/.417/.276/.841 |
| 0.20 | .87/.474/.290/.687 | .46/**.506**/**.406**/.835 | .41/.501/.364/.820 | .67/.492/.353/.818 |
| 0.30 | .76/.569/.370/.615 | .34/**.614**/**.514**/.769 | .26/.610/.490/.732 | .47/.610/.482/.764 |
| 0.40 | .55/.648/.444/.527 | .22/**.697**/**.613**/.680 | .14/.689/.603/.622 | .31/.695/.585/.682 |
| 0.50 | .34/.698/.501/.428 | .13/**.767**/**.693**/.585 | .06/.767/.708/.496 | .20/.755/.666/.594 |

**C1's val win does not transfer**: test pAUC 0.469 vs expE 0.526 (tiny
0.304 vs 0.435 — a 0.13 deficit). Every baseline HELD its pAUC from val to
test (expE 0.520→0.526, expB 0.523→0.521, rfdetrL 0.515→0.505); C1 dropped
0.062. The val advantage was site-specific.

**Not an inference artifact (checked)**: val predictions came from the
feature cache, test from a recompute route (bf16 backbone forward + bf16
feature rounding + fp32 head, `dino_infer.py --bf16-backbone`). A 240-image
val parity job through the recompute route reproduced the cached-route
predictions **exactly** (top-score |delta| max 0.0000, identical detection
counts on all 240 images; `models/v3/evals/C1valcheck/`). The collapse is
real generalization failure, not plumbing.

Interpretation (cannot be fully separated with the runs that exist):
(a) 7B features may transfer worse to unseen sites for this task;
(b) the C1 configuration is the most overfit-prone of the set — the largest
head (95M vs 22-25M) trained with NO augmentation; B1cached (same handicap,
smaller head) was already the second-worst val model. FIgLib (§2c), also
out-of-domain cameras, is consistent with the transfer failure: C1 detects
no more events than expE and is slower. B2's test row deliberately does not
exist (one-test rule); its val margin (-0.003 vs expE) plus 33x latency
means even a perfect transfer would not make it the serving choice.

### 2c. FIgLib 39-event causal replay (val-selected thresholds, clean models)

Thresholds selected on val only, identical rule to `rfdetr_eval.md` §4
(match expE's val neg-image FP at the deployed 0.30/0.15 thresholds):
B1 0.5624/0.3454, B2 0.5911/0.3079, C1 0.7377/0.4918. Replay = unmodified
AlertEngine (seq 3-of-5), frame scores from the same inference code (B1/B2
on devbox CPU shards, C1 as a GPU job with shipped frames), offline replay.

| model @ val-matched thr | detected | missed | median delay | mean delay | pre-visible |
|---|---|---|---|---|---|
| B1 frozen-L | **34/39** | 5 | 240 s | 455 s | 4 |
| B2 frozen-H+ | 32/39 | 7 | 240 s | 399 s | 3 |
| C1 7B cached | 30/39 | 9 | 360 s | 406 s | 2 |
| expE y11m @0.30/0.15 deployed | 30/39 | 9 | 240 s | 404 s | 5 |
| rfdetrL @ val-matched | 33/39 | 6 | 300 s | 449 s | 6 |

Paired per-event vs expE (shared events / found-only-by-X / only-by-expE,
median delay delta): B1 29 / **+5** / 1, median 0 s (11 earlier, 8 later,
10 equal); B2 28 / +4 / 2, median 0 s; C1 27 / +3 / 3, median 0 s.
Pattern matches RF-DETR: the ViT detectors convert their higher raw-recall
ceiling into *a few more events found*, not into earlier detection — and B1,
the smallest, is the best event-finder of the three (34/39 is the best
clean-model count this project has recorded; the contaminated pyronear
baseline's 37/39 remains excluded). n=39: ±2-3 events is binomial noise.

### 2d. Val→test transfer summary (the actual story of this track)

| | val pAUC | test pAUC | delta |
|---|---|---|---|
| C1 7B cached | 0.531 | 0.469 | **-0.062** |
| expE y11m | 0.520 | 0.526 | +0.006 |
| expB y11s | 0.523 | 0.521 | -0.002 |
| rfdetrL | 0.515 | 0.505 | -0.010 |

## 3. Latency (batch-1, includes preprocessing, excludes JPEG decode)

| config | B1 (L, 325M) | B2 (H+, 866M) | C1 (7B, 6.8B, bf16 backbone) | rfdetrL | expE y11m |
|---|---|---|---|---|---|
| 1x L40S GPU | 156.4 ms (150.8 model + 5.6 pre) | 371.2 ms | 368.5 ms | 31.4 ms | **11.2 ms** |
| devbox CPU, 8 threads | 5,670 ms | 11,670 ms | not viable | 1,016 ms | **324 ms** |

DINOv3 GPU numbers are fp32 except C1's backbone (bf16, matching its
training regime); no TensorRT/compile optimization on any model in the
table. Even the smallest variant is 14x expE on GPU and 17x on CPU.

## 4. Verdicts

1. **Serving default unchanged: YOLO11m (expE).** No DINOv3 variant beats it
   at matched false-alert rate on held-out sites, and all cost 14-33x the
   latency.
2. **The val-only "win" (C1) is a cautionary result, not a product
   candidate**: +0.011 val pAUC over expE, -0.057 on frozen test. Without
   the frozen-test discipline this track would have shipped a wrong
   conclusion ("7B backbone beats YOLO"). This is the second time this
   iteration a headline number failed under protocol (RF-DETR's mAP50 was
   the first).
3. **Frozen-backbone scaling does move val operating-point metrics
   monotonically** (L→H+→7B: 0.487→0.517→0.531) — DINOv3 features contain
   real signal for faint/small smoke — but the gain, at least in the no-aug
   cached configuration we could afford, does not survive a site shift.
   If revisited: 7B end-to-end-style training WITH augmentation (needs the
   `--flip` 2x cache or on-the-fly features), a smaller C1 head, and
   multi-site regularization are the candidates to separate hypotheses (a)
   and (b) of §2b.
4. **Partial unfreezing (B3) hurt** the metric that matters while flattering
   COCO mAP; do not spend further GPU on head-unfreeze variants judged by
   framework metrics.
5. **FIgLib**: B1 finds 34/39 events (best clean result; expE 30/39) at
   equal median delay — if event coverage on fixed cameras ever becomes the
   product priority over frame-level FP control, a frozen-L DINOv3 branch is
   worth a second look (156 ms would need distillation/quantization work).
6. License note (from prep §2): B3's checkpoint and both feature caches are
   DINOv3-License derivatives — redistribution only under the same terms;
   commercial use itself is fine, prohibited-use list (military etc.) n/a.

## 5. What got cut / limits

- C1 and B1cached at 6 epochs (schedule splice documented in §1); C1 still
  rising at cutoff — its val number is a floor, which makes the test
  collapse MORE striking, not less.
- B2/B1/B3/B1cached have no test rows (single-test rule; spent on C1).
- The planned `--flip` augmented 7B cache (2x storage) was not built — no
  time; it is the first thing to try if C1 is revisited.
- No bf16/TensorRT latency optimization; GPU latencies are unoptimized.
- One seed, one recipe per variant; in-training eval on a fixed 1,500-img
  val subset (seed 0) for checkpoint selection only.
- FIgLib is fixed-camera HPWREN imagery — no drone-generalization claim; the
  iteration-3 lockbox (25 events) remains UNTOUCHED.

## 6. Files

Runs/artifacts (`models/v3/runs/<run>/seg*/`): B1_dino3vitl16_frozen_1280,
B2_dino3vith16plus_frozen_1280, B3_dino3vitl16_top4_1280,
C1_7b_head_cached_1280, B1cached_noaug_head_1280, C1_7b_precompute_1280,
B1L_feats_precompute_1280 (+ B1prof/B2prof/B3prof profiles). Feature caches
on /scratch: `smoke/feats_v3/dino3_7b_1280x736` (~1.3 TB),
`dino3_L_1280x736` (~0.35 TB).

Code (new): `cluster/v3/dino_infer.py` (shared devbox/pod inference driver;
end-to-end, bf16-recompute and cached-feature modes; latency probe),
`cluster/v3/dino_eval_entry.sh`, `cluster/v3/submit_dino_eval.py` (eval jobs
incl. shipped-frames mode for FIgLib), `scripts/v3/score_frames_cpu_dino.py`
(parallel CPU FIgLib scoring). Fixed: `src/v3/det_modeling.py::_blocks`
(DINOv3ViTModel layout).

Results: `reports/v3/eval_{B1,B2,B3,B1cached,C1}_val.json` (+`.raw.parquet`),
`eval_C1_test.json` (+parquet), `matched_fp_val_trackBC.json`,
`matched_fp_test_trackBC.json`, `replay_{B1,B2,C1}_figlib_valmatched.json`,
`models/v3/evals/{B1val,B2val,B3val,B1cachedval,C1val,C1test,C1figlib,
C1valcheck,B1_figlib,B2_figlib}/` (preds jsonl, latency_gpu.json).

Reproduce (pattern, per model):
```bash
# GPU inference -> preds jsonl (val; test only for the chosen model)
python cluster/v3/submit_dino_eval.py --eval-name B1val \
    --backbone-dir /scratch/scratch/tianrgua-sandbox/models/dinov3-vitl16 \
    --ckpt /scratch/.../smoke/runs_v3/B1_dino3vitl16_frozen_1280/best.pt \
    --splits sdis_val --latency-n 50
# offline sweep (identical matching code as every other model)
python -m src.eval.eval_detection --data-yaml data/yolo/yamls/local/expA.yaml \
    --split val --backend dino3_B1 --imgsz 1280 \
    --preds-jsonl models/v3/evals/B1val/sdis_val_preds.jsonl \
    --out reports/v3/eval_B1_val.json
# FIgLib: CPU scoring (L/H+) or GPU job with shipped frames (7B), then replay
python scripts/v3/score_frames_cpu_dino.py --backbone-dir models/v3/hf/dinov3-vitl16 \
    --ckpt models/v3/runs/B1_dino3vitl16_frozen_1280/seg02/best.pt \
    --list data/v3/figlib39_frames.txt --workers 22 --threads 8 \
    --out models/v3/evals/B1_figlib/figlib39_preds.jsonl
python -m src.eval.replay_eval --manifest data/manifests/figlib_subset_manifest.parquet \
    --scores-jsonl models/v3/evals/B1_figlib/figlib39_preds.jsonl \
    --backend dino3_B1 --imgsz 1280 --conf-single 0.5624 --conf-seq 0.3454 \
    --out reports/v3/replay_B1_figlib_valmatched.json
```
