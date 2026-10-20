# Iteration 3 — Track A: new detection architectures vs YOLO11m
Working notes, updated as work proceeds. Started 2026-09-13.

## 1. Licenses (read from the actual repos, pinned commits)

### DEIMv2 — RESTRICTIVE (non-commercial only)
- Repo: github.com/Intellindust-AI-Lab/DEIMv2, pinned commit
  `1d2ca42171570c713e78fc6a766ec5104b7f4724` (2026-08-24), clone at
  `third_party/DEIMv2`.
- GitHub API "NOASSERTION" explained: `LICENSE.md` is a **custom "DEIMv2
  License"**, Copyright (c) 2026 INTELLINDUST INFORMATION TECHNOLOGY
  (SHENZHEN) CO., LTD. Key terms (verbatim-relevant):
  - Grant: "use, reproduce, modify, and distribute the DEIMv2 Materials
    **solely for Non-Commercial Purposes**". Non-Commercial explicitly
    includes "academic research, scientific research, education, personal
    experimentation, **benchmarking, evaluation**".
  - §3: "Commercial Use of the DEIMv2 Materials is **not permitted** under
    this License", incl. "deploying DEIMv2 in a commercial production
    environment" and "incorporating DEIMv2 into commercial hardware or
    software". Separate commercial license required (shenxi@intellindust.com).
  - §5: **Model Weights are subject to the same license** — "downloaded,
    evaluated, fine-tuned, modified, and redistributed solely for
    Non-Commercial Purposes."
  - Governing law: People's Republic of China.
- CONSEQUENCE: any DEIMv2-derived checkpoint is benchmark/evaluation-only
  for us; it can never ship in the product. Track A is therefore weighted
  toward RF-DETR; DEIMv2 (if run) is a benchmark-only data point.

### RF-DETR — Apache-2.0 for repo AND the Large model weights
- Repo: github.com/roboflow/rf-detr, pinned commit
  `a57ad2345c07c7822d110ef963b3bc330beb9937` (2026-09-13), clone at
  `third_party/rf-detr`. Repo LICENSE = Apache License 2.0.
- Weights license is split by model (README "License" section, verbatim):
  "The open-source `rfdetr` package and Apache-designated model weights are
  licensed under Apache License 2.0. ... Plus components, including the
  `rfdetr_plus` extension and RF-DETR-XL / RF-DETR-2XL detection models,
  are licensed under PML 1.0."
- Model table lists **RF-DETR-L (`RFDETRLarge`, `rfdetr-large`): Apache 2.0**
  (also N/S/M Apache 2.0; XL/2XL = PML 1.0, requires `rfdetr[plus]` — NOT used).
- Pretrain weight asset (src/rfdetr/assets/model_weights.py):
  `rf-detr-large.pth` from `https://storage.googleapis.com/rfdetr/rf-detr-large.pth`,
  md5 `992c8e862aa733a7bb2777e45d49f1a0`.
- CONSEQUENCE: RF-DETR-Large is a **co-equal architecture point** (usable
  weights, usable code), and the primary launch for this track.

## 2. DEIMv2 variants: gated vs ungated backbones
Per README §2.3 "Backbone Preparation" + configs/deimv2/*.yml:
- **HGNetv2 variants** (Atto 0.5M / Femto 1.0M / Pico 1.5M / N 3.6M, plus
  unlisted hgnetv2_{s,m,l,x} configs): backbone auto-downloads, ungated.
  Trainable without any gated weights, but all are small (<10M).
- **S (9.7M, AP 50.9)**: backbone = their ViT-Tiny distilled from DINOv3-S,
  `ckpts/vitt_distill.pt`, hosted on **Google Drive** (ungated link in README).
- **M (18.1M, AP 53.0)**: ViT-Tiny+ `ckpts/vittplus_distill.pt`, Google
  Drive (ungated).
- **L (32.2M)**: backbone `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` —
  **gated DINOv3-S** (Meta license/access request). SKIP.
- **X (50.3M)**: `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` —
  **gated DINOv3-S+**. SKIP until access lands (task brief already noted X;
  note that **L is gated too**, not just X).
- HuggingFace (huggingface.co/Intellindust): hosts full trained COCO
  checkpoints (DEIMv2_DINOv3_{S,M,L,X}_COCO, DEIMv2_HGNetv2_*_COCO) — full
  models incl. backbone, no gate on the HF repos themselves; usable as
  fine-tune init for S/M without touching gated raw DINOv3. All under the
  same non-commercial DEIMv2 license.
- Strongest ungated variant = **DEIMv2-M** (ViT-Tiny+), 18.1M params.

## 3. Plan (bounded)
- Primary: **RF-DETR-Large** fine-tune from Apache COCO weights at high
  resolution (~1288, divisible-by-56 constraint), 8x L40S single node,
  segmented/resumable via /scratch, green g6e queue.
- Secondary (benchmark-only, license-noted): DEIMv2-M at ~1280 if time
  allows within the 2-node ceiling.
- Control: reuse **expE YOLO11m** (60ep @1280 on identical expB_train list,
  eval reports already in reports/eval_expE_*). Augmentation parity note:
  we do NOT modify our data or YOLO augs; each architecture uses its own
  recipe's augmentations (that is part of the "architecture+recipe" claim,
  same as expE used ultralytics defaults).
- Data: convert frozen YOLO lists to COCO json (scripts/v3/), images stay
  at /scratch/scratch/tianrgua-sandbox/smoke_data_v2/ (already staged,
  path map /local/home/tianrgua/cvproject/data/yolo=<pod smoke_data_v2>).
- NEVER touch test/FIgLib/lockbox.

## 4. COCO conversion (DONE, verified)
- `scripts/v3/yolo_to_coco.py` (deterministic, list-order ids, single
  category id=1 'smoke'; YOLO cxcywh-norm -> clamp -> COCO xywh top-left px;
  <1px boxes dropped; label-less/empty-label images kept as negatives).
- train (expB_train.txt): 38,454 imgs / 31,487 boxes / 11,257 negatives /
  7 sub-px boxes dropped. sha256 20a6bd2c…
- val (sdis_val.txt): 7,335 imgs / 6,603 boxes / 1,158 negatives / 0 dropped.
  sha256 3d7eea73…
- Outputs: `data/v3/coco/instances_{train,val}.json` +
  `data/v3/rf/{train,valid}/_annotations.coco.json` (Roboflow layout;
  file_name prefixed `data/` -> resolved via a per-split `data` symlink to
  the pod image tree; RF-DETR's loader does `Path(root)/file_name`).
- Verification: `scripts/v3/verify_coco_contact_sheet.py` — 20-sample
  draw-back per split (14 pos + 6 neg, seed 0), exact arithmetic cross-check
  vs original YOLO labels: 0 mismatches; visual pass on
  `reports/v3/coco_verify_{train,val}.jpg` (boxes on plumes, negatives clean).
- RF-DETR keeps annotation-free images (torchvision CocoDetection semantics,
  no empty-image filtering found in rfdetr/datasets/coco.py) — negatives
  train as pure background, same as ultralytics.

## 5. Infra (iteration-3)
- Container `smoke-det3:v1` (cluster/v3/Dockerfile.det): pytorch 2.8.0-cu128
  runtime base (same digest as v1/v2 images), curl+git, rf-detr[train]
  installed from the pinned clone (pulls transformers 5.x, pytorch_lightning
  2.6.x, torchmetrics, hotcoco, faster-coco-eval, pycocotools,
  torch-hungarian), DEIMv2 deps (faster-coco-eval PyYAML tensorboard scipy
  calflops), both repos baked, rf-detr-large-2026.pth baked at
  /opt/rfhome/models (RF_HOME=/opt/rfhome; md5 verified 5cb72153…),
  DEIMv2 ckpts (vitt_distill.pt 1c52722c…, vittplus_distill.pt 3f831977…,
  deimv2_dinov3_m_coco.pth e1d2b7eb…) at /opt/DEIMv2/ckpts, COCO jsons at
  /opt/data. uid-1001 env fixes live in the entry script.
- RF-DETR-Large geometry: patch 16, num_windows 2 -> resolution must be
  divisible by 32 -> **1280 exactly** (PE auto-scales 1280/16=80; default
  704). multi_scale+expanded_scales at 1280 spans 1120..1440 (kept, recipe
  default).
- Image verified on devbox CPU inside the container (mounted local data,
  448 res, 16-img overfit ds, device=cpu): dataset symlink resolution works,
  LWDETR 33.2M params, pretrain load OK (PE resize OK), 8 train steps + val
  + checkpoint + RFDETR_TRAIN_DONE. Pushed as
  000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-det3:v1; repo created
  with the exact LeviathanPull policy read back from the us-east-1
  smoke-detect repo (both cluster accounts).
- GOTCHA found by reading TrainConfig: `devices` defaults to **1** in the
  Lightning migration — 8-GPU DDP requires devices=8 explicitly (driver
  flag --devices, entry env NGPUS).
- GOTCHA (pod, job deimv2msanitya1-vv7x exit 109): gdown writes files mode
  600 and docker COPY preserves it -> uid-1001 pod PermissionError on
  /opt/DEIMv2/ckpts/*.pt (invisible in root docker CPU tests). Fixed with a
  `chmod -R a+rX` layer; image digest now c36065782b14…, pushed 02:04 PDT.
- DEIMv2-M pod config `cluster/v3/deimv2_m_smoke_1280.yml`: 48-epoch
  fine-tune @1280 from official M COCO ckpt, scaled scheduler/aug epochs
  (documented in the file header), num_classes 1, our COCO jsons, ViT-Tiny+
  RoPE backbone = input-size agnostic (verified 1280 forward on CPU in
  container: 18,236,011 params, pred_logits [1,300,1]). BENCHMARK-ONLY
  (license). Profile note: 2-epoch profile runs pre-Mosaic (augs start ep4),
  so its rate slightly underestimates full-recipe epoch time.
- JOB PATTERN (documented choice): single LONG job (12h wall via timeout)
  instead of 55-min segments — Lightning checkpoints ('last.ckpt') are
  epoch-granular, and one epoch may exceed a presigned-URL segment. Run dir
  on /scratch persists; resume = resubmit same run name. Status/console
  beacons best-effort for first ~50 min; afterwards fetch via
  cluster/v3/fetch_run.py (v2 fetch-job pattern). Files:
  cluster/v3/{det_entry.sh,orchestrate_det.py,fetch_run.py},
  scripts/v3/train_rfdetr.py.

## 6. Cluster debug ledger (sanity phase, 4 failed attempts -> all fixed)
1. deimv2msanitya1 exit 109: gdown files mode 600 + uid-1001 -> chmod layer.
2. rfdetrlsanitya1: torch_linear_assignment (REQUIRED by rf-detr matcher on
   CUDA) JIT-compiles a Triton driver shim -> "Failed to find C compiler"
   -> added gcc+libc6-dev layer.
3. deimv2msanitya2: DEIMv2 custom v2 transforms use torchvision<=0.20
   private hooks _transform/_get_params, removed in tv 0.23 -> local patch
   in third_party/DEIMv2 (alias transform/make_params; 16-line diff, kept
   in-tree, documented in file). Dataloader then verified on CPU incl.
   mosaic epoch: multiscale set spans 960..1600 @base 1280.
4. deimv2msanitya3: CUDA index-out-of-bounds in matcher — with
   remap_mscoco_category=False DEIMv2 uses RAW category_id (ours=1) as
   label -> num_classes must be 2 (slot 0 unused). Config fixed.
5. rfdetrlsanitya2 overfit stage: Lightning re-execs the driver per rank ->
   8-way rmtree race on the overfit ds -> rank-0-only build + ready-marker.

## 7. RF-DETR sanity/profile results (job rfdetrlsanitya2, SUCCEEDED)
- PARAMS at 1280: total=trainable=35,647,394 (PE grows vs 704-native 33.9M).
- Pretrain load verified: md5-checked local file; PE-resize warnings are the
  documented fine-tune path.
- profile bs=2/gpu: did not finish 1 epoch (2403 steps) in 720s (killed).
- profile bs=4/gpu (global 32): FULL epoch (1202 steps) + FULL 7,335-img
  val INSIDE 720s incl. startup; **val (EMA) after 1 epoch: mAP50:95 0.4468,
  mAP50 0.7678, mAP75 0.4635, mAR@500 0.6951, F1 0.7496 (P 0.7436/R 0.7557)**
  (torchmetrics protocol — not directly comparable to our ultralytics-based
  eval; independent eval with src/eval tooling will follow training).
  No OOM at 1280 multi-scale (1120..1440) with bs4.
- Budget: epoch ~6-7 min incl. val -> 48 epochs ≈ 6h << 12h wall.

## 7b. DEIMv2-M profile (deimv2msanitya4, SUCCEEDED rc=124=clean timeout)
- Tuning load VERIFIED from console: backbone vittplus_distill.pt loaded;
  COCO-M checkpoint loaded with only score heads/anchors re-initialized
  ('missed': denoising_class_embed; 'unmatched': enc/dec_score_heads,
  anchors, valid_mask — exactly the 80->2-class + 640->1280 deltas).
- calflops @1280: 18.05M params, fwd 199.7 GFLOPs.
- Trainable params 18,237,550.
- GAP -> DIAGNOSED AS HANG: attempts 4/5/6 (print_freq 500/25/5, a6 with
  num_workers=0) all produced ZERO progress lines AND zero tensorboard
  scalar writes (writer logs every 10 steps; a6 tfevents mtime never
  advanced in 15 min) while gpu_util beacon showed **100% on all 8 GPUs**
  = NCCL collective spin-wait: DDP hang in the FIRST training step, not
  slow training. (Contrast: a3 with num_classes=1 CRASHED inside step 1's
  loss in <2 min, so the input pipeline itself reaches the criterion.)
  Suspects: sync_bn / find_unused_parameters interplay under torch 2.8
  (repo pins torch 2.5.1). Attempt 7 = single bounded retry with
  sync_bn=False find_unused_parameters=False; else DEIMv2 is declared
  blocked-with-evidence for this iteration.

## 7c. DEIMv2-M VERDICT: BLOCKED (with evidence), 2026-09-13
- a7 (sync_bn=False, find_unused_parameters=False, print_freq=5): SAME
  hang (100% util, no steps). Overrides verified applied in the cfg dump.
- a8 (STOCK 640 recipe config, print_freq=5): SAME hang — resolution-
  independent (util 100%, mem ~2GB, no steps).
- Summary of evidence: 5 attempts (a4-a8) all spin at 100% GPU with zero
  completed iterations (no MetricLogger prints at print_freq 5/25/500, no
  tensorboard scalars though the writer fires every 10 steps, no last.pth);
  varying num_workers (0/4), sync_bn, find_unused_parameters, resolution
  (640/1280) changes nothing. The pipeline itself demonstrably reaches the
  loss (a3 crashed inside step-1 loss when num_classes was wrong), and the
  full model forward + dataloader pass on CPU in the same container.
  => first-step NCCL/DDP hang specific to torch 2.8.0+cu128 (repo pins
  torch 2.5.1/torchvision 0.20.1). Follow-up if DEIMv2 is ever needed:
  dedicated torch-2.5.1 container (different base image), or bisect their
  DINOv3STAs optimized attention / ms_deform_attn under torch 2.8, or
  single-GPU training (NGPUS=1) as a slow benchmark fallback. Given the
  non-commercial license already caps DEIMv2 at benchmark-only, this is
  not worth more of the track budget: Track A's architecture point is
  RF-DETR-Large.
- GPU-hours burned on DEIMv2 attempts: ~8 short jobs x ~15-25 min x 1 node
  (~2.5 node-h).

## 8. LAUNCHED
- **rfdetrL_1280_e48** (job tianrgua-rfdetrl1280e48a1-gs9y, submitted
  09:44 UTC 2026-09-13): RF-DETR-Large fine-tune from rf-detr-large-2026
  (Apache-2.0), res 1280, 8x L40S, bs 4/gpu ga 1 (global 32), lr 1e-4
  (recipe default incl. EMA/multi-scale/expanded-scales), epochs 48
  (bounded 40-60 < recipe 100), ckpt every 5 ep + last.ckpt every ep,
  val every ep, wall 12h, run dir
  /scratch/.../smoke/runs_v3/rfdetrL_1280_e48 (resume = resubmit attempt 2).
- In-flight health (metrics.csv fetched ~10:40 UTC): 7 epochs in ~55 min
  (~7.5 min/ep -> ~6h total, wall OK). val mAP50: ep0 0.7632, ep1 0.7856,
  ep6 0.7795; train loss 8.83 -> 6.41. last.ckpt + checkpoint_4.ckpt +
  checkpoint_best_ema.pth present on /scratch.
- Overfit-16 note: the formal 50-iter overfit was superseded by a STRONGER
  check — the bs4 profile trained a full real epoch and hit val mAP50 0.767
  (sanity a2); the 8-rank overfit-ds race found there is fixed in the
  driver (rank-0 build + marker). Not re-run to save node-hours.

## 8b. Monitoring / resume runbook (rfdetrL_1280_e48)
- Orchestrator: detached pid on devbox (logs/v3_rfdetrL_1280_e48.log),
  polls kraken every 5 min; fetches S3 beacons at terminal state.
- Live checks: `kraken -p green jobs list -r tianrgua | grep rfdetrl1280`;
  artifacts: `python cluster/v3/fetch_run.py --run rfdetrL_1280_e48
  --files metrics.csv --files console.log` (add `--files last.ckpt`
  [540MB] or `--files checkpoint_best_ema.pth` [135MB] at the end).
- If the job dies or hits the 12h wall before epoch 48:
  `python cluster/v3/orchestrate_det.py --run rfdetrL_1280_e48 --mode
  train --attempt 2 --epochs 48 --bs 4 --ga 1` (driver auto-resumes from
  /scratch last.ckpt).
- NOTE: baked /opt/{det_entry.sh,train_rfdetr.py} in smoke-det3:v1 are one
  fix behind the repo copies (rank-guard + gpu_util beacon); jobs fetch the
  repo copies via presigned GET, baked ones are fetch-failure fallback only.
  Rebake on next image update.
- After training: evaluate best_ema with OUR protocol (src/eval) on the
  frozen val list; thresholds on val only; then the usual test/FIgLib flow
  (NOT the lockbox — that stays untouched).

## 8c. TRAINING COMPLETE + EVALUATED (2026-09-13 15:19 UTC)
- rfdetrL_1280_e48 SUCCEEDED: 48/48 epochs, 5h34m (20,044 s), single job, no
  resume needed. Best EMA = **epoch 37** (mAP50:95 0.4701, mAP50 0.7654);
  mAP50 PEAKED AT EPOCH 1 (0.7856) then drifted down to 0.7345 by ep47 (lr
  never decays: recipe lr_drop=100 > 48 epochs) -> ~12-16 epochs with decay
  would likely suffice next time.
- Artifacts in `models/v3/rfdetr/` (checkpoint_best_ema.pth sha256 2cca1fe8…,
  metrics.csv, val_curve_training_protocol.csv, PROVENANCE.md); other
  checkpoints stay on /scratch.
- Serving backend `src/serve/backend_rfdetr.py` verified BIT-EXACT vs upstream
  `rfdetr.predict()` (0.0 score/box delta) via
  `scripts/v3/verify_rfdetr_backend.py`. Default backend still yolo.
- Unified eval: GPU inference job (`cluster/v3/submit_rfdetr_eval.py`, 8 shards
  on one g6e, 21 img/s/GPU, 3 min for val+test) -> preds JSONL -> offline sweep
  via `eval_detection.py --preds-jsonl` (new flag; `sweep_report()` refactor
  reproduces frozen expE/expB conf_sweep byte-identically). FIgLib 3,056 frames
  scored on devbox CPU (22 shards, 282 s) because those frames are not staged
  on the cluster; `replay_eval.py --scores-jsonl` (new flag) reproduces the
  frozen expE replay summary exactly.
- **RESULT (full report: reports/v3/rfdetr_eval.md)**: at MATCHED negative-image
  FP, RF-DETR-L does NOT beat YOLO11m. Recall pAUC(FP<=0.5) val 0.496 vs expE
  0.521, test 0.501 vs 0.502; tiny-recall pAUC val 0.398 vs 0.422, test 0.382
  vs 0.413. NMS on the 300 un-NMS'd queries changes nothing (<=0.001/cell), so
  it is not a protocol artifact. FIgLib IS better: 33/39 events at val-matched
  thresholds (35/39 at the raw 0.30/0.15) vs expE 30/39, paired delay ~equal
  (median 0 s over 28 shared events). Latency 31.4 ms/img GPU (vs y11m 11.2)
  and 1016 ms CPU@8 threads (vs 324) -> 2.8-3.1x cost.
- Threshold transfer: RF-DETR scores are NOT on the YOLO scale (conf 0.30 =>
  54% neg-image FP vs YOLO 36%). Val-matched equivalents: single-frame 0.535,
  seq hit 0.382 (selected on val only).

## 9. Control (YOLO11m)
- Reusing **expE_mix_y11m_1280_v2** (iteration 2): identical expB_train
  list (the exact images the COCO jsons were converted from), 60ep @1280,
  results in reports/eval_expE_{val,test}.json. AUGMENTATION PARITY: our
  data and YOLO pipeline are untouched this iteration; each architecture
  runs its own recipe's augmentations (RF-DETR: multi-scale + square
  resize + its default photometric/flip stack). No new YOLO control run
  needed. Final comparison must use OUR eval tooling (src/eval) on the
  frozen val list at matched FP, not the frameworks' internal metrics.
