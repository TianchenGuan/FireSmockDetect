# Wildfire Smoke Detection Prototype — STATUS

Last updated: 2026-09-10 (session start)

## Goal
Drone uploads RGB image → server returns smoke boxes, scores, alert status.
Optimized for faint/small/distant smoke. Single-image inference standalone;
recent-frame buffer optional strengthener.

## Environment (verified 2026-09-10)
- Devbox: no GPU, 192 CPU, 1.5TB RAM, 7TB free disk, has internet. Python 3.12.12 via uv.
- GPU: Leviathan cluster (kraken CLI, Midway auth OK). Cluster has NO internet.
  - Chosen queue: `obsidian-2-hyperpod-bom-110-p6-b200.48xlarge-ap-south-1a`
    (project obsidian, POC=SYSTEM shared, access=NONE, 100/106 nodes free in-quota
    as of 2026-09-10 08:45 UTC). 1 node = 8× B200. In-quota jobs not preemptible.
  - Storage: FSx `/data` (ro), `/scratch` (rw), `/checkpoints`; sandbox
    `/scratch/scratch/tianrgua-sandbox/`. GPU nodes have local NVMe `/mnt/local`.
  - Image must come from ECR-allowlisted accounts; using 000000000001
    (agi-training-toolkit-dev) in ap-south-1.
- Data staging plan: devbox (internet) → S3 (ap-south-1, acct 000000000001) → pod.
  Pod S3 access untested (pod identity = cluster account); fallback = bake data into image.

## Pinned artifacts
- pyronear/yolov11s @ rev `a3db632886688f78eb814c5d5a080c906771aad5` (Apache-2.0).
  Trained on pyronear/pyro-dataset → LIKELY OVERLAPS pyro-sdis; its eval numbers on
  our splits must be flagged as potentially contaminated.
- pyronear/pyro-sdis @ rev `a1e553ec4d806f71fc6db744cc22bc3469487382` (Apache-2.0),
  3.28 GB, 6 train + 1 val parquet shards, ultralytics label format.
- Devbox deps: python 3.12.12, torch 2.8.0 (cpu on devbox / cu128 in cluster image),
  ultralytics 8.3.202, fastapi 0.117.1 — full pin list in `.venv` + cluster/Dockerfile.

## Work log
- [2026-09-10 08:40] Environment recon done (GPU/queues/internet/dataset sizes).
- [2026-09-10 08:50] venv ready (torch 2.8.0+cpu, ultralytics 8.3.202, py3.12).
- [2026-09-10 09:00] Phase-1 workflow launched (run wf_bad7cdbf-528), 6 parallel
  tracks: A baseline demo, B pyro-sdis, C dfire, D figlib, E cluster-B200, F fasdd.
- [2026-09-10 09:15] Serving stack written and unit-sanity-checked:
  src/serve/{config,detector,alerting,app,replay_client}.py + static upload UI.
  Alert engine: provisional from single frame, confirmed from K-of-N persistence,
  cooldown, stale/scene-change reset. Tiled inference in detector.py.
- [2026-09-10 09:20] COCO-pretrained yolo11s/yolo11m downloaded + load-verified.
  Training wrapper src/train/train_yolo.py (AMP, resumable, small-smoke-safe augs).
  Eval: src/eval/eval_detection.py (PR sweep, size-bucket recall, neg-image FP),
  src/eval/replay_eval.py (causal FIgLib replay, delay from first-visible-smoke).

- [2026-09-10 ~12:30] Phase-1 workflow done (6/6 tracks, ~3.5h, details in
  reports/*): baseline demoed + CONTAMINATION CONFIRMED (pyronear model trained
  on pyro-sdis AND FIgLib); pyro-sdis 33,636 imgs converted+audited (0 corrupt);
  D-Fire 21,527 verified (CC0); FIgLib 39 events/3,056 frames with verified
  offset-0=first-visible-plume; FASDD downloaded but ~half of "UAV" subset is
  HPWREN webcam imagery -> SKIPPED this iteration; B200 GPU fwd/bwd PASS
  (8x B200 sm_100, torch 2.8.0+cu128); pod->S3 = AccessDenied (unfixable our
  side); NO pod log access -> exit-code bitmask pattern.
- [2026-09-10 12:50] SPLITS FROZEN (data/manifests/splits_frozen.json):
  pyro-sdis site-disjoint train 22,177 (4 sites) / val 7,335 (cabanelle,
  marguerite) / test 4,124 (ferion winter, croix-augas sdis-77). D-Fire =
  train-only supplement (16,277: smoke boxes kept, fire boxes dropped, 944
  fire-only imgs excluded, 7,833 negatives kept). Cross-dup audit D-Fire vs
  sdis val/test: 0 exact, 0 near (phash hamming<=6). FIgLib eval-only.
- [2026-09-10 12:55] Tiny end-to-end CPU training test PASSED (TRAIN_DONE,
  checkpoints written) on frozen expA yaml.
- [2026-09-10 13:05] Demo server live on 127.0.0.1:8000 with baseline model:
  negatives->none, dense smoke->provisional (0.50), faint t+2min missed
  (fine-tuning target). ~140-230ms/img CPU.
- [2026-09-10 13:15] smoke-detect:v2 built + pushed (data+code+weights baked;
  pods can't reach S3). Training plan: ONE B200 node (allocation granularity
  is whole 8-GPU nodes; minimal possible), expA then expB sequential, DDP
  8 GPU, segmented <=~45min jobs (presigned URL TTL = ada cred 1h), resume
  from /scratch sandbox, artifacts out via presigned PUT.

- [2026-09-10 15:00-17:00] GPU e2e test #1 FAILED silently (no artifacts).
  Root cause via exit-code probe jobs: pytorch/pytorch base image has NO
  curl/wget -> every presigned upload was a no-op; also presigned URLs must
  use the REGIONAL S3 endpoint (global unreachable from no-internet pod VPC;
  URLs verified working from devbox). Fixes: smoke-detect:v3 (curl + baked
  yolo11n.pt/Arial.ttf for offline ultralytics AMP check + cd /opt/weights),
  orchestrator presigns regional + robust status polling.
- [2026-09-10 17:15] GPU e2e test #2 (gputest2, v3, 2ep @640, 2-GPU DDP)
  submitted. Baseline val sweep done (reports/eval_baseline_sdis_val.json).

- [2026-09-10 18:00-19:30] EGRESS SAGA RESOLVED by moving to green/g6e:
  probe10 proved presigned PUT from obsidian B200 pods = HTTP 403 (S3 VPC
  endpoint filters by principal; URL verified valid from devbox). probe11/12:
  pod ECR auth denied; CodeArtifact reachable ONLY for the shared amazon
  domain host (exact-hostname allowlist; our domains unreachable in both
  regions). CodeArtifact round-trip works devbox-side (201 + md5 match) but
  unusable from pods. ALSO: bom110 shared queue quota was clawed back
  (106 -> 0 allocated; pulsar expanded to 471/472) -> our B200 jobs were on
  borrowed, preemptible capacity. DECISION: moved training to green project
  queue green-hyperpod-g6e.48xlarge-us-east-1-tas (8x L40S, 38/39 free,
  IN-QUOTA -> not preemptible). g6e probe exit 208 = presigned PUT 200 + 8
  GPUs. Pushed smoke-detect:v4/v5 to us-east-1 ECR (repo policy grants pull
  to both Leviathan accounts), new us-east-1 bucket. Entry v5: writable
  run-root autodetect (/scratch -> /mnt/local -> /tmp) + cross-segment
  resume via presigned GET. Orchestrator parameterized (project/queue/
  bucket/region) + stale-pending resubmit. gputest2 "SUCCEEDED" was a false
  positive (budget-skip branch bug, fixed: now exits 90/91).
- [2026-09-10 19:35] gputest3 (g6e, v5, 2ep @640, 8-GPU DDP): seg1 trained
  2/2 epochs in ~4min wall, val mAP50=0.498 after 2ep, 19MB checkpoint + all
  artifacts retrieved from S3 -- FULL PIPELINE VERIFIED. green /scratch
  sandbox exists & persists (local resume verified by seg2). One bug found:
  TRAIN_DONE print unreliable under DDP -> completion now judged by
  results.csv epoch count (entry v6, pushed).
- [2026-09-10 20:00] REAL TRAINING LAUNCHED on green g6e (in-quota, not
  preemptible): expA_sdis_y11s_1280 + expB_mix_y11s_1280 in parallel
  (1 node x 8 L40S each, 60ep @1280 batch 64, v6 image, segmented+resumable).

- [2026-09-10 20:35-21:25] BOTH TRAININGS COMPLETE on green g6e:
  expA 60/60 ep (2 segs), final val mAP50=0.698; expB 60/60 ep (3 segs),
  val mAP50=0.738 (D-Fire supplement wins). Segmented resume worked across
  a devbox session restart (jobs unaffected; orchestrators relaunched with
  --start-seg/--resume-key, now setsid-detached).
- [2026-09-10 21:30-22:40] Evals: expA/expB val+test sweeps done, FIgLib
  causal replays done (tiled variants still running). Thresholds selected on
  VAL only (reports/threshold_selection.md): single-frame 0.30, persistence
  0.15 x 3-of-5. 18 annotated examples (reports/examples/). Demo server now
  runs expB @1280 (verified: known-positive FIgLib frame -> 0.67 provisional,
  ~180ms CPU).

## Training experiments (status)
- expA_sdis_y11s_1280: DONE. test conf=0.2: P=0.686 R=0.552 tiny=0.445; FIgLib 30/39, median delay 180s.
- expB_mix_y11s_1280: DONE, PRIMARY. test conf=0.2: P=0.684 R=0.646 tiny=0.541; FIgLib 32/39, median 475s (different detected-event sets; not directly comparable to expA delay).
- expC (tiled inference): REJECTED at current thresholds — pre-smoke frame
  alert rate 0.10 -> 0.45 (expA) / 0.53 (expB). Needs a separate tile-pathway
  threshold + hard-negative round next iteration.

## DELIVERED (iteration 1 complete, 2026-09-10)
Trained checkpoints (expA/expB), frozen split manifests, evaluation reports
(reports/eval_*, replay_*), threshold selection doc, 18 annotated examples,
FINAL_REPORT.md, live demo server (expB @1280 on 127.0.0.1:8000), replay
client, reproducible commands (README.md).

## Open blockers
- No pod log access (ScientistRole unassumable; Bindles CanAssume needed) — workaround in place.
- Pod→S3 denied — workaround: bake data in image, presigned-URL uploads out.
- Queue admission non-deterministic (25min+ ADMITTING stalls observed).
- FIgLib replay for the BASELINE model is contaminated (trained on FIgLib) — only our own models get clean early-detection numbers.


# ===== ITERATION 2 (v2): VLM domain post-training — started 2026-09-11 =====
Objective: does domain SFT of open VLMs (Qwen3.5 family) beat YOLO baselines
on faint/small/distant smoke at matched false-alert level. v1 artifacts and
service PRESERVED; everything new under */v2/.

- [2026-09-11 ~08:20 UTC] Model cards verified (all Apache-2.0, all
  image-text-to-text): Qwen3.5-9B @c2022362 19GB / 27B @fc05daec 56GB /
  35B-A3B @59d61f3c 72GB / 122B-A10B @dc4d3484 250GB; Qwen-Image-Edit-2511
  @6f3ccc0b. Arch qwen3_5 needs transformers>=4.57 (pin 4.57.6).
  L40S plan: 9B LoRA zero2 1 node; 27B LoRA zero3 1 node; 35B-A3B LoRA
  zero3 1-2 nodes; 122B LoRA zero3(+offload)/QLoRA prep; NO cross-node sync.
- [08:30] 5 prep agents launched: VLM container (transformers 4.57.6 stack),
  weight staging (HF->S3->presigned GET->/scratch), multimodal SFT dataset
  (JSON boxes in 0-1000 thousandths, full-image + small-smoke crops,
  train-list only), Track E YOLO11m launch + tile-offset val tuning,
  FIgLib v2 lockbox (~25 fresh event-disjoint events, frozen uninspected).
- [08:40] Written: src/train/train_vlm.py (LoRA SFT, vision frozen by
  default, param counts logged, resumable), cluster/v2/{vlm_entry.sh,
  zero2.json, zero3.json, orchestrate_vlm.py} (segment pattern reused from
  v1; code/data fetched per-job via presigned GET so iterations need no
  image rebuild). Training images tar (5.9GB) uploaded to S3; staging job
  tianrgua-dstage-9mdq untarring to /scratch/.../smoke_data_v2 (RUNNING).
Planned first jobs: overfit-16 sanity (9B, 60 steps) -> 100-step profile ->
trkA_9b_lora + trkB_27b_lora launches.
- [2026-09-11 ~09:40 UTC] Track E part 1: expE_mix_y11m_1280_v2 (YOLO11m,
  expB recipe, 60ep @1280 batch 48, v6 image) LAUNCHED on green g6e via
  setsid-detached orchestrator (logs/v2_expE_orchestration.log). seg1
  SUCCEEDED (13 ep, val mAP50=0.740 already); seg2 resumed + RUNNING.
  S3 beacon verified (runs/expE_mix_y11m_1280_v2/seg01/status.json).
- [2026-09-11 ~09:45 UTC] Track E part 2: tile-pathway threshold tuning on
  VAL (2,000-img seeded sample, expB weights, 1280/1024). detector.py gained
  tile_conf_offset + filter_by_conf post-filter (default = old behavior).
  Sweep (src/eval/tile_tuning.py, reports/v2/tile_tuning_val.{md,json}):
  shared threshold (offset 0) confirms iter-1 FP explosion; RECOMMENDED
  (base_conf 0.25, tile_offset 0.2) beats no-tiling tiny-recall
  (0.613 vs 0.594) at lower neg-image FP (0.295 vs 0.299). Gains modest;
  3x CPU latency; FIgLib replay confirmation required before shipping.
- [09:05] FIgLib v2 LOCKBOX FROZEN (data/v2/manifests/LOCKBOX_FROZEN.md):
  25 fresh events / 1,979 frames / 25 new cameras (0 camera-id, 0 view,
  0 (cam,date), 0 event overlap vs iter-1 39-event set; 14 towers shared
  at different headings — disclosed). Years 2016-2026, >=2/yr. 25/25 pass
  iter-1-identical file validation (22 exact +00000, 3 near <=4s). phash
  spot-check 200 frames vs iter-1: 0 exact, 0 near (h<=6), min h=8.
  Manifest sha256 1c777845...c9ebab. 1.8GB frames (+1.8GB tars kept).
  NO model touches it until final report, ONCE. Uninspected (no contact
  sheets, file-level checks only).
- [2026-09-11 09:20 UTC] SFT DATASET v2 BUILT (data/v2/sft/, scripts/
  build_sft_v2.py, seed 0): train.jsonl 59,999 (38,454 full-image from
  expB_train + 19,294 pos 640-crops for area_frac<0.01 smoke, random
  in-crop placement + 2,251 neg crops = 20% of negs; under 70k cap ->
  0 dropped); val_mini.jsonl 600 FROM-VAL (fromval: ids, 200 tiny +200
  pos +200 neg, training-time eval only); overfit_16.jsonl 8pos/8neg.
  Answers = compact JSON, boxes in integer thousandths [x1,y1,x2,y2] of
  model input (crop-frame for crops); crop_box recorded, no crop files.
  Round-trip verified (scripts/test_sft_roundtrip.py, 50 samples/56
  boxes): exact answer re-derivation, IoU mean .975 / min .935, every
  box >= its 1/1000-quantization bound (0.99 unattainable for tiny
  boxes at this resolution; boxes >=100 thou/side: 100% >=0.99).
  Contact sheets + DATASET.md + README.md in data/v2/sft/.

- [2026-09-11 10:00-15:45 UTC] VLM training path debugged end-to-end (all
  fixes to code fetched per-job; zero image rebuilds except +bitsandbytes):
  (1) uid 1001 has no passwd entry -> pin HOME/cache env; (2) transformers
  4.57 lacks qwen3_5 -> container built on 5.17.0 (agent caught it);
  (3) OOM root cause 1: gradient checkpointing inert without
  enable_input_require_grads + non-reentrant under PEFT; (4) OOM root cause
  2: per-image max_pixels hint is a qwen-vl-utils convention the bare
  processor IGNORES -> 12MP dfire images became ~16k vision tokens and a
  single 11GiB alloc; enforced budget by resizing in the dataset;
  (5) zero3 27B loads sharded only if TrainingArguments precedes
  from_pretrained (zero.Init) — fixed, but measured 391 s/it: PCIe-only
  L40S all-gather kills ZeRO-3 -> switched 27B+ to QLoRA (nf4 base, bf16
  LoRA, pure DDP) = 20.2 s/it, 19x faster. vlm:v2 image adds bitsandbytes.
  9B profile: 10.5 s/it zero2 (~6.1 samples/s).
- [2026-09-11 14:05] trkA_9b_lora REAL RUN training (1 epoch = ~937 steps,
  seg1 ckpt@200 loss 0.366, segment resume verified for VLM path).
- [2026-09-11 15:50] trkB_27b_qlora REAL RUN launched (~7 segs est).
  prof_35b_q (Track C MoE) submitted. Cluster use: 4/39 nodes (A, B, C-prof,
  expE YOLO11m). Track D 122B: 4bit ~65GB won't fit one L40S; zero3 measured
  impractical -> plan device_map multi-GPU pipeline profile, else report
  blocked-with-evidence.

- [2026-09-11 21:00-01:40 UTC] TRACK F (synthetic-data pilot) COMPLETE —
  Qwen-Image-Edit-2511 @6f3ccc0b (QwenImageEditPlusPipeline, diffusers
  0.40.0): smoke-diffusion:v1 built+pushed (digest c35e10b3..., CPU
  class-load verified); 57.7GB weights staged HF->S3->/scratch byte-exact
  (job -29ue). 450 imgs generated on green g6e (5 segments synthseg1-5, all
  SUCCEEDED, 25.5 GPU-h): 150 TRAIN-site-negative backgrounds x
  (faint-smoke @region / matched control / fog-dust-cloud confounder),
  spec+seeds+prompts frozen in data/v2/synth/spec.jsonl, ALL outputs marked
  unverified. Steady rate ~123 s/img/GPU (30 steps, ~1MP, cpu-offload;
  first img/worker ~1000s mmap warmup). Screening (automated/model-assisted,
  NOT human; contact sheets + diff-locality): realism good but ~53% plumes
  oversized vs "faint", region adherence ~ZERO (median 4% edit mass in 2x
  box -> requested boxes UNUSABLE as labels), 25% empty/global-haze, 4%
  visible flames, controls carry re-render fingerprint (mean |d| 6-10/255).
  72/450 quarantined. VERDICT: NO-GO on 1-2k expansion / YOLO ablation with
  this protocol; conditional-GO for masked-edit + post-hoc-box revision
  (data/v2/synth/PILOT.md). No synthetic training launched.
- [2026-09-11 19:50 UTC] HEADLINE: 9B SFT val eval (2,000 imgs, IoU0.4):
  base -> SFT: img_recall 0.384->0.753, box_recall 0.272->0.589,
  tiny_recall 0.090->0.460 (5x), box_precision 0.722->0.766, neg-img FP
  0.094->0.330, parse_fail 112->96, latency ~7.3s/img (8-way sharded
  L40S). At roughly matched hard-neg FP (~0.33-0.34), YOLO11m still leads
  tiny recall (~0.66 val) and is ~30x faster -> VLM positioning: async
  verifier, not primary detector. 27B (trkB, ~440/937 steps) pending.
  35B-A3B: 3 bounded retries all OOM (evidence in models/v2/runs/prof_35b_q)
  -> marked blocked, not retrying further.

- [2026-09-11 21:00-22:30 UTC] Cascade analysis (offline, val 2000):
  SFT-9B decisions highly correlated with YOLO11m; AND/OR/hybrid fusions
  all sit ON the YOLO threshold curve -> no ROC gain from 9B as verifier
  (reports/v2/cascade_analysis_val.txt). Track F diffusion pilot agent
  launched: container smoke-diffusion:v1, weights staged, generation jobs
  running (synthseg1-4).
- [2026-09-11 ~22:40 UTC] trkB_27b_qlora TRAINING COMPLETE: 938/937 steps,
  10 segments, final loss 0.0146 (vs 9B 0.053). Adapter retrieved
  (models/v2/runs/trkB_27b_qlora/seg10/). eval_vlm.py gained --quant 4bit
  (27B bf16 = 56GB > 44GB L40S; eval quantization matches training nf4).
  Submitted eval27b_base + eval27b_sft (same 2,000-img val subset).
- GPU-hours so far (v2, approx): trkA 9B 5 segs + trkB 27B 10 segs +
  profiles/smoke ~6 jobs + evals ~4 jobs + expE 5 segs + staging/synth —
  all single-node 8x L40S, sequential segments; peak concurrent 4/39 nodes.

- [2026-09-12 08:00-10:00 UTC] GRPO FEASIBILITY GATE (§6): VERDICT GO
  (reports/v2/grpo_feasibility.md). trl 1.13.0 (compatible w/ pinned
  transformers 5.17.0; native VLM GRPO verified from source) baked into
  smoke-detect-vlm:v3 (digest e2cfd653...). Verifiable reward
  src/train/grpo_reward.py (+1/-1 presence, +0.5/matched box IoU>=0.4,
  -0.5/unmatched pred box, clip [-2,2]; reuses eval parse_answer;
  18/18 unit tests). Probe grpo_probe_32.jsonl (16 pos/16 neg, seed 0).
  Wiring de-risked on devbox CPU with a tiny random qwen3_5
  (cluster/v2/make_tiny_qwen35.py) before spending cluster attempts.
  Cluster probe SUCCEEDED on attempt 1 (tianrgua-grpoe2ea1-wyyj, 10 min):
  20/20 GRPO steps on 9B+SFT-adapter (trained in place, beta=0), reward
  non-constant (7 levels, grad>0 on 14/20 steps), no always-negative
  collapse (pred-neg 0.50->0.53), parse_ok 0.86->0.96, 10.7 s/step steady.
  500-step run NOT launched; prerequisites: segment/resume in grpo_entry,
  G=8, max_new_tokens 192.

- [2026-09-12 ~06:30-07:40 UTC] 27B evals v2 (512 tok + fallback parser):
  SFT 27B/nf4/0.75MP: tiny 0.302, box R 0.438, P 0.754, parse_fail 361 (18%)
  vs 9B/bf16/1MP SFT tiny 0.460 -- larger model LOSES under the L40S
  quantization+resolution constraints; fair bf16 27B comparison requires
  H200/B200 (obsidian egress still blocked). Root cause of the earlier
  false 'failure': 27B keeps a chatty CoT preamble that overflowed 192
  tokens; also first eval attempt lost artifacts to URL expiry (evals now
  persist to /scratch + fetch jobs).
- [2026-09-12 07:45] LOCKBOX one-time eval (25 fresh events, first
  inspection): expE y11m 20/25 median 299s pre-vis 2; v1 expB 20/25 median
  240s pre-vis 4 -> image-level y11m gains do NOT show as event-level gains
  at n=25; tiled config finishing. Reported as-is.
- [2026-09-12 07:50] GRPO feasibility spike: GO (trl 1.13 native VLM GRPO,
  vlm:v3 image, reward fn 18/18 tests incl. anti-gaming cases, e2e 20 steps
  clean, 10.7 s/step). Launched 500-step controlled run grpo_9b_500
  (G=8, 192 tok) per brief section 6.

- [2026-09-12 13:30-14:30 UTC] GRPO 500/500 done (attempt02, reward
  0.586->0.994). HELD-OUT val: tiny 0.460->0.548, box R 0.589->0.646,
  parse_fail 96->0, negFP 0.330->0.413 (no compute-matched SFT control —
  stated). Cascade with GRPO-9B: hybrid R 0.862@FP0.365, ~1-2pt above
  YOLO's own curve (noise-level). Tiled config REJECTED on lockbox
  (pre-smoke alert rate 0.376 vs 0.044 full-image). FINAL_REPORT_V2.md
  finalized. Iteration 2 complete.

# ===== ITERATION 3 (v3) — started 2026-09-13 =====
- [2026-09-13 08:57 UTC] FIgLib v3 LOCKBOX FROZEN
  (data/v3/manifests/LOCKBOX_V3_FROZEN.md): 25 fresh events / 1,978 frames /
  25 new cameras. Disjoint vs ALL 64 used events (iter-1 39 + iter-2 lockbox
  25): 0 event, 0 camera, 0 (cam,date), 0 site+view overlap (mobo/iqeye
  hardware suffixes normalized). Selection deterministic seed 20260913
  (src/figlib_lockbox_v3_select.py; pool 127 disjoint events). 25/25 pass
  iter-1-identical file validation (24 exact +00000, 1 near +1s JunctionFire
  hp-s); counts match remote listings; 2024+ tars have a new nested layout,
  normalized at extraction. Years 2016-2026, >=2/yr except 2022=1 (provably
  maximal after disjointness — disclosed). phash 200 frames vs BOTH prior
  sets: vs iter-2 clean (min h=8); vs iter-1 0 exact/h0, 61 h==6 pairs ALL
  from one near-black night frame vs an entire dark iter-1 event = degenerate
  dark-sky hash, not duplication (excluding it: 0 h<=6, min h=8); disclosed
  in xdup JSON. Disclosures: 14 shared towers at different headings; 8 events
  share a DATE with a used event (same fire, different tower/scene — allowed
  by iter-2 criteria). Manifest sha256 66514b03...c07608. 1.6GB frames
  (+1.6GB tars). NO model touches it until final report, ONCE. Uninspected
  (no contact sheets; file-level checks + hash statistics only).


# ===== ITERATION 3 (v3): visual backbones, architectures, temporal — started 2026-09-13 =====
Objective: improve the VISUAL detection problem (faint/small/distant smoke at
controlled FP) via DEIMv2/RF-DETR architectures, DINOv3 L/H+/7B backbones,
and a learned causal temporal head. Baselines: YOLO11s/m + v2 VLMs. 24h
first-pass wall budget. All new work under */v3/.

- [2026-09-13 ~08:10 UTC] Recon: green g6e quota RECHECKED: us-east-1 39/39
  free, us-west-2 16/16 free (in-quota). DEIMv2 repo license = NOASSERTION
  per GitHub API (needs in-repo check); DEIMv2-X backbone is DINOv3-S+
  (gated!); rf-detr repo Apache-2.0 (weights license to verify).
  BLOCKER: all 3 DINOv3 HF models gated:manual; our token (user
  TianchenGuan) gets 403 -> user asked to click through the three model
  pages. Tracks B/C build+validate plumbing with ungated stand-in meanwhile.
- [08:20] 4 track agents launched: A (DEIMv2 ungated variants + RF-DETR-L,
  COCO conversion, real training within ~3h), D (temporal: pyro-sdis
  camera+timestamp sequences from TRAIN sites only, frozen-expE features +
  causal head, vs K-of-N persistence), E (lockbox v3: ~25 fresh events
  disjoint from all 64 used), B/C prep (stack selection, container,
  stand-in validation, 7B precompute-features plan).
- v2 25-event lockbox is now OPENED/historical: banned from training,
  mining, tuning, and selection in v3.

- [2026-09-13 ~09:45 UTC] TRACK B/C PREP (DINOv3 detection) — scaffold done
  (reports/v3/trackBC_prep.md). Stack CHOSEN: HF DINOv3ViTModel (frozen) +
  our ViTDet-style SimpleFeaturePyramid + torchvision FasterRCNN head
  (src/v3/), own DDP loop on the v1/v2 segment pattern. dinov3 repo
  detection code = INFERENCE-ONLY (no loss/matcher; verified @6876159a);
  DEIMv2 REJECTED for B/C: license is proprietary NON-COMMERCIAL (read in
  full; relicensing dependency for any shipping path) + S/S+-only DINOv3
  integration. DINOv3 License covers code AND weights identically:
  commercial OK, no military/ITAR uses, redistribute only under same terms,
  publication acknowledgment; ckpts/cached features = derivatives.
  Container smoke-dino3:v1 built+pushed (us-east-1, digest aaf5b9d0...,
  transformers 5.17.0 + pycocotools + curl + uid fixes; in-image fwd
  verified). Gated revisions pinned (L ea8dc286/1.2GB, H+ c807c9ee/3.4GB,
  7B b8036775/26.9GB; still 403 as of 09:12 UTC). Gate watch:
  scripts/v3/dinov3_gate_check.sh; post-access one-shot staging:
  scripts/v3/stage_dinov3.sh; launch configs cluster/v3/launch/{B1,B2,B3,
  C1_precompute,C1_head}.sh (B3 = top-4 unfreeze over LoRA; C1 = 7B
  precompute-features route: 13.4GB bf16 fits ONE L40S, cache 30.1MB/img =
  1.35TB — /scratch has 95TB free per pod probe — no-aug tradeoff + B1-cached
  control documented). Stand-in dinov2-large (UNGATED, Apache-2.0,
  @47b73eef) staged to /scratch byte-exact; CPU validations ALL PASS (grid
  80x46@1280x736 patch16 / 92x52@1288x728 patch14, frozen-grad audit, ckpt
  resume, cached-feature route); cluster overfit-16: seg1/2 failed in EVAL
  ONLY (pycocotools 2.0.10 needs 'info' key — fixed; training step + DDP +
  artifact channel all worked, status.json now reports scratch_free_gb),
  seg3 SUCCEEDED: loss 0.027@300 steps, train-mAP 0.807/AP50 1.00 on the
  16 imgs (plumbing-only numbers, stand-in != DINOv3), artifacts + resume
  channel verified. Data: consumed Track A's data/v3/coco jsons as-is.
  B/C is now GATED ONLY on HF access; ~20-30 min from access to training.

- [2026-09-13 ~02:30 UTC] TRACK D (learned causal temporal detection) COMPLETE
  (reports/v3/trackD_notes.md, reports/v3/trackD_eval_val*.json):
  (1) Temporal dataset built from pyro-sdis manifest timestamps: train 915
  seqs / 22,170 frames (4 train sites, 21 cams), val 266 / 7,335 (2 val
  sites, 8 cams); cut at >10 min gaps, >=3 frames; TRUE cadence median 29 s
  (p90 ~31 s, bursts 1-4 s); 29-36% of seqs have label flicker (noisy
  onset/decay labels) — data/v3/sequences/.
  (2) pyronear/temporal-model audited @fb7f6ca (Apache-2.0): NOT causal
  (bidirectional transformer over DINOv2 tube crops, 20-frame cap); their
  data = private-DVC pyro-dataset v4.1.0, overlap with pyro-sdis essentially
  certain (brison hardcoded in their docs) -> inspiration only, no data reuse.
  (3) Frozen-expE features (neck P3/P4/P5 pooled 12x20 + det conf maps,
  194ch fp16) extracted for all 29,505 seq frames in ONE g6e job (8-way
  shard, 7 min); tars fetched (data/v3/features/, 2.4GB). 281k-param causal
  ConvGRU head (T<=5, valid @T=1, dt feature, stride/drop aug) trained on
  cluster: overfit->real (158 s) + t1-only control + seed replicates;
  total ~0.5 node-h.
  (4) VAL @matched FPR on 1,158 embedded hard negs: learned head beats
  BOTH baselines at every FPR in 2/2 seeds (R@2% 0.151/0.181 vs single-frame
  0.100 vs persistence 0.083; PR-AUC 0.940 vs 0.935 vs 0.917). Deployed
  3-of-5 persistence is WORSE than single-frame on this frame-level metric
  (window keeps firing on neg frames inside smoke sequences). ABLATION:
  t1-only control also beats baselines -> most of the gain is learned
  spatial readout of frozen features; temporal context adds consistently
  only at tight FPR 1-2% (mean R@2% 0.166 vs 0.120). Cheapest win for
  serving: t1-only spatial-readout head (no buffer/state needed).

- [2026-09-13 15:19-17:00 UTC] TRACK A COMPLETE — RF-DETR-Large trained AND
  evaluated (reports/v3/rfdetr_eval.md; notes reports/v3/trackA_notes.md §8c).
  (1) rfdetrL_1280_e48 SUCCEEDED 48/48 epochs in 5h34m, single job. Best EMA =
  epoch 37 (training-protocol val mAP50 0.7654 / mAP50:95 0.4701); mAP50 peaked
  at EPOCH 1 (0.7856) and decayed to 0.7345 by ep47 — constant lr (recipe
  lr_drop=100 never fires) => 12-16 epochs w/ decay would do. Artifacts in
  models/v3/rfdetr/ (ckpt sha256 2cca1fe8…, metrics.csv, val curve,
  PROVENANCE.md); rest stays on /scratch.
  (2) src/serve/backend_rfdetr.py implements the backends.py contract and is
  BIT-EXACT vs upstream rfdetr.predict() (0.0 score/box delta). Default backend
  unchanged (yolo). Devbox venv touched only by +pydeprecate +supervision.
  (3) UNIFIED EVAL (same protocol as every model): val+test inference as ONE
  GPU job (cluster/v3/submit_rfdetr_eval.py, 8 shards, 21 img/s/GPU, 3 min for
  11,459 imgs), sweeps computed offline via new eval_detection.py --backend /
  --preds-jsonl (refactor reproduces frozen expE/expB conf_sweep
  byte-identically); FIgLib 3,056 frames scored on devbox CPU (22 shards, 282 s;
  frames are not staged on the cluster) and replayed via new replay_eval.py
  --scores-jsonl (reproduces the frozen expE replay summary exactly). Test swept
  ONCE; lockbox untouched.
  (4) HEADLINE: at MATCHED negative-image FP, RF-DETR-L does NOT beat YOLO11m —
  recall pAUC(FP<=0.5): val 0.496 vs expE 0.521 / expB 0.523; test 0.501 vs
  0.502 / 0.508. Tiny-smoke recall WORSE: pAUC val 0.398 vs 0.422; test 0.382
  vs 0.413 (e.g. -0.09 abs at FP 0.10 on test). Its raw recall ceiling IS higher
  (val R 0.930 @conf 0.05 vs 0.870) but does not convert at controlled FP; NMS
  on the 300 un-NMS'd DETR queries changes the matched-FP table by <=0.001/cell
  (not a protocol artifact).
  (5) FIgLib 39-event causal replay (clean): rfdetrL 33/39 detected, median
  delay 300 s, 6 pre-visible at VAL-MATCHED thresholds (single 0.535 / seq
  0.382, selected on val only — RF-DETR scores are NOT on the YOLO scale:
  conf 0.30 => 54% neg-img FP vs YOLO's 36%); 35/39, median 120 s, 13
  pre-visible at the raw deployed 0.30/0.15. vs expE 30/39 median 240 s. Paired
  on 28 shared events: median delay delta 0 s (12 earlier / 8 later / 8 equal),
  +5 events found only by RF-DETR, 2 only by expE. ±0.01 score perturbation
  moves detection count by <=1 event.
  (6) Latency batch-1 incl. preprocessing: L40S 31.4 ms (vs YOLO11m 11.2 ms,
  2.8x); devbox CPU 8 threads 1016 ms (vs 324 ms, 3.1x); CPU 32 threads 416 ms
  (vs 312). No fp16/TensorRT applied.
  VERDICT: architecture swap alone does not buy the faint/small-smoke gain;
  YOLO11m stays the serving default, RF-DETR backend available behind
  SMOKE_BACKEND=rfdetr. Framework-metric comparison would have claimed a win
  (+0.018 mAP50 over ultralytics' 0.747) — it does not survive our protocol
  (§7 of the report explains the four reasons).

- [2026-09-14 01:35 UTC] TRACK B/C EXECUTION STARTED — DINOv3 gated access
  LANDED (all 3 models 200 at 01:08). Staged devbox->S3->/scratch byte-verified
  in 20 min: L 1,212,584,721 B / H+ 3,362,457,720 B / 7B 26,864,283,673 B at
  pinned revs. 120-step profiles on real weights: B1 frozen-L 0.33 s/it (bs32
  global), B2 frozen-H+ 0.54 s/it, B3 top4-unfreeze 0.29 s/it (bs16) — 12
  epochs fits every run before the 08:10 boundary. BUG found by B3 profile:
  det_modeling._blocks() didn't know DINOv3ViTModel's hf.model.layer path
  (stand-in dinov2 uses encoder.layer) — fixed + CPU-verified (top-4 grads
  flow, bottom frozen, 72.7M/325.4M trainable). LAUNCHED: B1+B2+B3 12-epoch
  runs + C1 7B feature precompute (4 nodes). Param counts (exact): B1 22.31M
  trainable/325.43M total; B2 25.14M/865.73M; B3 72.70M/325.43M (50.39M
  backbone); C1 head 95.25M, 7B backbone 6.716B frozen. Eval infra built:
  cluster/v3/{dino_infer.py,dino_eval_entry.sh,submit_dino_eval.py} (same
  preds-jsonl contract as rfdetr eval; CPU smoke test passed).

- [2026-09-14 06:20 UTC] TRACK B/C COMPLETE (reports/v3/trackBC_results.md).
  5 runs trained + unified-eval'd in the 7h window: B1 frozen-L (22.31M/325M,
  12ep), B2 frozen-H+ (25.14M/866M, 12ep), B3 L-top4-unfreeze (72.70M/325M,
  12ep), C1 7B-frozen cached-features noaug (95.25M head/6.81B, 6ep — I/O
  bound, cut from 12 w/ schedule splice, disclosed), B1cached-noaug control
  (6ep). 7B precompute: 45,789 imgs, 1.95 img/s/GPU, 1.3TB cache.
  VAL matched-FP pAUC(R/tiny): C1 0.531/0.430 > expB 0.523/0.420 >
  expE 0.520/0.422 > B2 0.517/0.417 > rfdetrL 0.515/0.415 > B1 0.487/0.378 >
  B1cached 0.477/0.358 > B3 0.474/0.357. Frozen scaling L->H+->7B monotonic
  on val; C1 tiny-recall@FP0.05 0.161 vs expE 0.097. B3 = best in-training
  COCO mAP (0.269) but WORST operating pAUC — framework-metric mirage again.
  TEST (swept ONCE, best-on-val = C1 per pre-registered rule): **C1
  COLLAPSES 0.469/0.304 vs expE 0.526/0.435** — every baseline held val->test
  (expE +0.006), C1 dropped -0.062. NOT an inference artifact: 240-img val
  parity job (bf16 recompute vs cache) matched EXACTLY (max top-score delta
  0.0000). Val win was site-specific (no-aug + 95M head overfit and/or 7B
  feature transfer — inseparable w/ existing runs). FIgLib 39-event replay
  @val-matched thr: B1 34/39 med 240s (best clean count; expE 30/39), B2
  32/39, C1 30/39 med 360s; paired: +5/-1 events vs expE (B1), delays equal.
  Latency batch-1 L40S incl. preprocess: B1 156ms / B2 371ms / C1-bf16 369ms
  vs expE 11.2ms (CPU 8t: 5.7s / 11.7s / n-a vs 324ms). VERDICT: YOLO11m
  stays serving default; DINOv3 val gains don't survive site shift; do NOT
  judge variants by COCO mAP; B1's event coverage (34/39) noted for a future
  fixed-camera track. Test used once; lockbox v3 UNTOUCHED. ~20 node-jobs,
  ~14 node-hours total.

- [2026-09-14 06:55 UTC] LOCKBOX V3 OPENED — one-time evaluation executed
  (reports/v3/LOCKBOX_V3_RESULTS.md). Pre-registered systems, no re-runs, no
  threshold changes: expE@deployed 0.30/0.15, rfdetrL@val-sel 0.5349/0.382,
  B1-dinov3L@val-sel 0.5624/0.3454. Manifest sha256 re-verified before run.
  RESULTS (25 events): B1 24/25 detected (1 miss) / expE 21/25 / rfdetrL
  20/25; R@1min 8-9/25 and R@3min 11-12/25 for ALL three (early detection =
  tie); paired median delay delta 0s in every pairing — systems differ in
  COVERAGE not speed. B1's extra events are found late (477-1560s tail);
  raw medians (B1 477s vs expE 120s) are over different detected sets and
  must not be compared unpaired. B1's FIgLib-39 coverage edge REPLICATES on
  pristine data (2 independent sets agree); rfdetrL's does NOT (20/25 <
  expE 21/25). Pre-visible alerts 4-5 per system (reported separately,
  uninspected). Scores: models/v3/evals/lockbox_v3/, replays
  reports/v3/replay_lockboxv3_*.json. Lockbox v3 is now HISTORICAL — banned
  from training/mining/tuning/selection. No further iteration on these
  numbers.

- [2026-09-14 06:50 UTC] ITERATION 3 CLOSED 80 min inside budget.
  FINAL_REPORT_V3.md finalized. Headlines: B1 frozen-DINOv3-L event
  coverage replicates on pristine lockbox (24/25 vs expE 21/25); C1-7B val
  win collapsed on frozen test (discipline win); learned temporal head >
  persistence everywhere; YOLO11m stays serving default. All jobs terminal.

## ITERATION 4 — data transfer & visual adaptation (opened 2026-09-14 08:33 UTC)

- [2026-09-14 08:45 UTC] ITERATION 4 OPENED. Window 2026-09-14T08:33:10Z ->
  2026-09-15T08:33:10Z (24h). Ceilings: 16 nodes / 128 L40S concurrent,
  384 node-hours (3,072 GPU-h). Green g6e queue verified: 39 nodes free,
  0 queued. Scaffolding: cluster/v4/{BUDGET_V4.json,registry.py (flock'd
  shared limiter — ALL v4 submissions reserve first),run_yolo_arm.sh,
  run_v4_arm.sh,orchestrate_v4.py,train_entry_v4.sh}. v4 entry adds SEED,
  DATA_TARS (S3->/scratch staging w/ .done markers), DATA_YAML_B64 (inline
  mixture yamls) — fetched via presigned URL_ENTRY, no image rebuild.
- [2026-09-14 08:38 UTC] ARM A (existing-data control) LAUNCHED:
  expV4A_ctrl_s0 = YOLO11m from COCO, expB.yaml corpus (38,454 imgs), 60ep
  b48 @1280 seed0 — v4 matched budget = ~2.31M image exposures per arm.
  Job tianrgua-v4actrls0s1 RUNNING. Doubles as pipeline smoke test.
- [2026-09-14 08:42 UTC] DATA ACQUISITION WORKFLOW launched (6 parallel
  agents): DFS, FIRESENSE (zenodo), MmodalFire (figshare), Boreal UAV
  (Sci Data), MultiNatSmoke (contamination-critical: recover FIgLib/D-Fire/
  Boreal origins, quarantine unknowns), + v1-v3 report/artifact
  reconciliation note (reports/v4/RECONCILIATION_V4.md).
- [2026-09-14 09:00 UTC] v4 STAGING CHAIN VERIFIED end-to-end: toy job
  tianrgua-v4toytests1 SUCCEEDED — DATA_TARS tar fetched via presigned GET,
  extracted to /scratch v4data w/ .done marker, DATA_YAML_B64 decoded,
  trained 2/2 epochs on 8 GPU, artifacts uploaded (status.json entry:v4,
  phase done). Fixed job-name parse (trailing '.') in orchestrate_v4.py.
  Arms B-E can launch the moment corpora exist. Strict-FP eval tooling added
  (scripts/v4/compare_matched_fp.py, targets now incl. 1%/2%): expE val
  baseline @1% negFP = R 0.059 / tiny 0.013 — the strict region is where v4
  must move. Background: DINO-adapt code agent + corpus-builder agent
  running; acquisition workflow (6 agents) running; arm A seg1 RUNNING.
- [2026-09-14 11:40 UTC] G CORPUS FROZEN + ARMS B/D_pre LAUNCHED. Corpus
  builder agent delivered scripts/v4/build_corpus.py (self-test 22/22).
  Dedup vs eval material: ZERO usable-row collisions (16,493 eval frames
  hashed; all 53 eval-hit units were pre-quarantined multinatsmoke subsets).
  DFS = BLOCKED (dead OneDrive/Baidu-only). mmodalfire = 0 trainable smoke
  boxes (video-labels only). B_G_v1 frozen: 14,901 imgs (14,609 mask-derived
  multinatsmoke non-forest positives + 292 firesense verified negatives),
  16,794 boxes, tar 3.4GB. Strict source-only val added (make_gval.py):
  unit-grouped 105 units -> gval 310 imgs (thin — logged as limitation);
  arm B model selection NEVER touches target data. Arm B = 158ep (exposure-
  matched 2.31M), D_pre = 63ep (40% of budget; stage-2 gets 60% on E+T).
  URL_INIT added to entry for curriculum stage-2 init-from-ckpt.
  Boreal (125GB, 4 burn sites) still downloading; arms C/E + aerial split
  wait on it. Verified-empty-negatives policy flag added to builder.
  Active: armA seg3 (ep~25+/60), armB seg1 training, D_pre submitting.
- [2026-09-14 13:35 UTC] ALL FIVE MATRIX ARMS LAUNCHED + DINO/TEMPORAL
  TRACKS DELEGATED. Boreal landed (4,963 imgs, 4 burns, 4,693 human boxes,
  CC-BY-4.0, 125GB verified) — genuine UAV aerial. MultiNatSmoke audited:
  usable 29,857; FIgLib quarantine held (65/89 of our eval events found
  byte-identical inside it — banned); AuSmoke prize = 15,248f/29 videos
  incl. 4 DJI drone units. Splits frozen: aerial (boreal) evo+karkkila
  train / heinola val / ruokolahti TEST; forest holdout (AuSmoke) manually
  burn-grouped (md5 split had same-burn-multi-camera leaks: FB769/FB867/
  FB746/FB902/2024-10-21 — fixed) -> 16 units train / 4 val / 9 test
  (4 independent test events incl. 1 aerial flight). Corpora frozen:
  E_ET_v1 51,420 lines (45ep), C_EGT_v1 94,188 effective w/ 2x G/T
  multiplicity (25ep); both w/ verified-empty boreal negatives. Arms E/C
  relaunched after yaml-path miss (builder layout differed). Reconciliation
  note delivered (reports/v4/RECONCILIATION_V4.md, 5 discrepancies; D1
  fixed: src/serve/config.py file defaults now = deployed expB@1280, was
  contaminated-pyronear fallback). DINO track agent launched: 4 runs
  (frozen-L control / LoRA-L / full-L / LoRA-H+) on E_ET recipe @10ep,
  1 node each, registry-reserved. Temporal data-prep agent launched (CPU):
  v4 sequences from AuSmoke (per-frame boxes!), boreal clips, FIRESENSE.
  Active nodes: A(seg5) B Dpre E C = 5; DINO adds 4 when staged.
- [2026-09-14 15:45 UTC] BUG FOUND+FIXED: silent DDP crash-loop on
  init-from-checkpoint (arm D stage-2). 14 segments failed with exit 101 /
  train_rc 1 and NO console artifact. Cause: ultralytics clears the run dir
  when starting a fresh run there under DDP, which deleted the fetched
  init.pt AND unlinked the shell's already-open segment_console.log — so all
  ultralytics output went to a dead inode and every diagnosis channel was
  empty. Fixes in cluster/v4/train_entry_v4.sh: init.pt + console now live in
  /tmp (outside the run dir); entry self-logs its own stdout (exec >
  /tmp/entry.log) and ships entry.log + run-dir ls + ultralytics tail through
  the console channel. Verified: toy job w/ --init-key now rc=0 TRAIN_DONE.
  Also fixed invalid taxonomy role 'evaluation' -> 'inference' (kraken
  rejects; had silently blocked the batch-eval submitter too). D_adapt
  relaunched as expV4Dada_s0b. Wasted ~1.2 node-h on the crash loop.
- [2026-09-14 17:30 UTC] FIRST TRANSFER RESULTS (val only, arms A/B/D_pre).
  Batch-eval infra now working (cluster/v4/{yolo_eval_entry.sh,
  submit_yolo_eval.py}; two bugs fixed: eval pods never staged /opt/smoke_data
  so val lists 404'd, and preds env-var names carried a wrong 'ckpt_' prefix
  — both invisible until the entry started self-logging). 9 preds sets
  (3 ckpts x sdis-val/boreal-val/forest-val) fetched and evaluated.
  HEADLINE: generic-smoke pretraining does NOT transfer.
  * sdis val @1%/10% negFP: A_ctrl_E R 0.053/0.309 tiny 0.015/0.201 vs
    B_Gonly 0.021/0.092 tiny 0.001/0.007 vs Dpre_G 0.034/0.093.
  * boreal AERIAL val (1,123 imgs, 217 real negatives) @1% negFP:
    A_ctrl_E R 0.640 vs B_Gonly 0.194 vs Dpre_G 0.133. Existing recipe
    transfers to genuine UAV imagery FAR better than generic smoke data.
    (tiny-bucket recall is 0.000 for all: aerial plumes are never <0.1% area
    — the aerial task is NOT a tiny-object task.)
  * AuSmoke forest val (new region, positives-only -> matched-FP undefined;
    script now emits recall-at-conf instead): at each model's sdis-val-
    selected conf, A_ctrl_E R 0.020, B/Dpre 0.000; at conf 0.20 A 0.268
    (tiny 0.103) vs B 0.0005 vs Dpre 0.000.
  Arms with target data still training: E_ET ep36/45, C_EGT ep20/25,
  D_adapt ep9/26. Those decide whether T (and G on top of T) actually help.
- [2026-09-14 18:15 UTC] SESSION RESTART RECOVERY. The coordinator session
  ended ~17:40Z; all cluster work SURVIVED because every orchestrator runs
  nohup'd (5 jobs RUNNING at 18:09Z: D_adapt seg3, DINO B4/B5/B6 seg4,
  trackd headsbal). Only the two monitoring agents (DINO track, temporal
  track) died — both resumed from their transcripts with explicit state.
  Completed while unattended: arm E_ET (45ep, 5 segs) and arm C_EGT (25ep,
  5 segs) TRAINING COMPLETE; DINO v4B1_frozen_L complete + evaluated on all
  3 val sets. Launched 3-valset eval for E/C (v4evalr3). Registry: 6 nodes
  active, 33.8 node-h spent of 384; deadline 2026-09-15T08:33Z unchanged.
- [2026-09-14 18:35 UTC] EARLY-WARNING TRACK COMPLETE (reports/v4/
  EARLY_WARNING_V4.md; 3 cluster jobs, 0.72 node-h). Frozen-expE features
  for 16,571 v4 frames (byte-identical extractor to v3). Heads 281,153
  params. pyro-sdis val matched-FPR recall @1/2/5%: expE single 0.062/
  0.100/0.205; persistence 3-of-5 0.051/0.083/0.163; temporal v3-only
  0.077/0.166/0.248; t1-only v4-ext 0.079/0.164/0.334. Temporal head beats
  persistence 30/30 cells (as v3); t1-only still matches it -> gain is
  spatial readout, not history. NEGATIVE RESULT: the v4 AuSmoke extension
  (10,905 all-positive frames, no matched negatives) is a DOMAIN SHORTCUT —
  extended heads fire on 43-47% of boreal non-smoke frames and 83-98% of
  firesense negative frames where v3-only heads fire on 0% (40-90 false
  episodes/h vs 0); cap-5-seqs ablation shows dose-response. Invisible to
  pyro-sdis val selection; caught only by the small v4 replay negatives.
  DO NOT SHIP v4-extended heads. YOLO + 1-in-70 DINO sweep: wins 60/60
  matched-FP cells but +0.4-0.8pt episode detection (2-4/469) — negligible,
  no deployment case. v4 added NO observed-onset supervision.
- [2026-09-14 18:50 UTC] TRANSFER MATRIX seed-0 RESULTS (val only; arms
  A/B/C/E; D_adapt eval job running). reports/v4/matched_fp_*.json.
  * sdis val (fixed-camera, source domain) R/tiny @10% negFP:
    A 0.309/0.201 > C_EGT 0.266/0.164 > E_ET 0.247/0.132 >> B 0.092/0.007.
    Adding target data costs ~4-6pt source-domain recall.
  * boreal AERIAL val (1 held-out burn, 217 real negs) R @1% negFP:
    C_EGT 0.934 ≈ E_ET 0.931 >> A 0.640 >> B 0.194. E_ET's FP never
    exceeds ~1% at any conf (n/a rows) — near-saturated. This is TARGET-
    ADAPTED transfer (2 boreal burns in T); A is the source-only number.
  * AuSmoke forest val (new region, positives-only) R/tiny @conf0.20:
    C 0.306/0.163, E 0.266/0.238, A 0.268/0.103, B 0.0005/0. At sdis-val-
    selected confs (0.66-0.75) ALL arms get 1-2% recall: fixed-camera
    thresholds do not transfer to forest/aerial — threshold choice is
    itself the adaptation step (brief §4 distinction confirmed empirically).
  * G's marginal value on top of T ≈ 0 (C vs E within noise everywhere;
    C +4pt forest R, E +7pt forest tiny). Generic-only B is the worst arm
    on every set. VERDICT (pending seeds): the research hypothesis fails —
    curriculum/generic pretraining does not beat straightforward E+T.
  Launching 3-seed repeats: A/E/C x seeds 1,2 (6 nodes, ~27 node-h).
- [2026-09-14 19:40 UTC] LEDGER + HOUSEKEEPING. scripts/v4/make_compute_
  ledger.py builds reports/v4/COMPUTE_LEDGER.csv from kraken --json + text
  tables (start/end from scheduler; queue wait excluded; whole-node GPU-h;
  utilization = n/a, no pod telemetry). Raw dumps under reports/v4/ledger/
  with presigned URLs/tokens SCRUBBED before saving. Iter 3/4 attribution is
  time-based (exact window starts); iter 1-vs-2 is name-based (approx) —
  column attribution_method says which. Totals (allocated): iter1 41.9
  node-h (incl. 4.5 B200), iter2 10.1, iter3 20.6, iter4 so far 34.5 node-h
  = 276 GPU-h of 3,072 cap (77 jobs; 3.5 node-h were failed/retry). Found
  and STOPPED a stale iteration-1 probe (tianrgua-sv4-dgnx) QUEUED in
  obsidian-pu since 09-11 — idle allocation request, never ran.
  Pre-registration written BEFORE any test prediction exists:
  reports/v4/PREREGISTRATION_V4.md (systems by val rules, two threshold
  regimes, default-replacement criteria). TRANSFER_RESULTS_V4.csv generated
  from saved JSONs (scripts/v4/make_transfer_csv.py). FIgLib-39 CPU replay
  for A/E/C seed0 running locally (~2.7h/model). Agents: DINO track (B4/B5/
  B6 seg4+), sealed test package builder, DATA_CARD_V4 writer. 9 cluster
  jobs live: 6 seed repeats + 3 DINO.
- [2026-09-14 19:55 UTC] D_adapt (curriculum G->E+T, 15% G replay) seed-0
  evaluated. On sdis val it is the BEST v4 arm at strict FP: R 0.082 @1%
  (A 0.053, expE 0.059, C 0.041, E 0.046) and ties A at 10% (0.300 vs
  0.309, tiny 0.192 vs 0.201) — unlike C/E it does NOT pay the ~5pt source-
  domain cost — while keeping the aerial gain (boreal 0.909 @1%, 0.957 @20%).
  Caveats: 1 seed; the 1% point rests on ~12 negative images. Launched
  D_pre seeds 1,2 (D_adapt seeds follow when they finish) so the curriculum
  arm gets the same 3-seed treatment as A/C/E. 11 nodes active.
- [2026-09-14 20:05 UTC] DATA_CARD_V4.md delivered (332 lines, all numbers
  artifact-cited). Key totals: 27,538 new positives vs 329 negatives — ZERO
  negatives from any new camera domain (the root of the temporal-track
  shortcut). Sealed-test section defers to LOCKBOX_V4_FROZEN.md (builder
  agent still running).
- [2026-09-14 20:25 UTC] TEST PACKAGE SEALED (reports/v4/LOCKBOX_V4_FROZEN.md,
  frozen 18:38Z, tar 6.75GB sha f7cc4ad7...). boreal_test = ruokolahti
  1,764 fr / 1,762 pos / ONLY 2 negatives / 1 burn; forest_test = 2,871 fr,
  positives-only, 4 events (3 ground + 1 DJI flight incl. 90-fr staged
  test-run, broken out); sdis_test list sha recorded. Overlap vs 84,680
  train/val images: 0 by path, 0 by sha256. CONSEQUENCE for final eval:
  neither new test set supports matched-FP — thresholds MUST come from val
  (both regimes in PREREGISTRATION_V4.md); test reports recall-at-threshold
  only, with boreal FP shown on its 2 negatives as anecdotal. FINDING: public
  MultiNatSmoke ships Boreal byte-for-byte (2,215 ruokolahti frames, 1,748 in
  its official TRAIN split) — our corpora are clean (0 in any list), but any
  model pretrained on public MultiNatSmoke is contaminated for boreal test.
- [2026-09-14 20:40 UTC] DINOv3 ADAPTATION TRACK COMPLETE (reports/v4/
  dino_adapt_results.md; 4 runs x 10ep on E_ET recipe, ~23 node-h train +
  ~1 eval; no OOM). Trainable params: B1 frozen 0 (22.3M head), B4 LoRA-L
  3.15M (25.5M total trainable), B5 full-L 303.1M (325.4M), B6 LoRA-H+
  5.24M (30.4M). sdis val matched-FP R @1%/5%: B6 0.097/0.227 > B4 0.085/
  0.210 ≈ B5 (n/a@1%, floor 1.4%)/0.209 > YOLO-A 0.053/0.204 > expE 0.059/
  0.190 > B1 (n/a)/0.175. Adaptation beats frozen at every FP level; LoRA-L
  captures most of the gain at half B6's cost; FULL UNFREEZE DOES NOT BEAT
  LoRA (confound: bs2/2x steps/end-only ckpt). boreal val: all DINO 0.96-
  0.99 R @1-5% vs YOLO E/C 0.93-0.95. forest val @conf0.5: B6 0.553 > B4
  0.519 > B5 0.489 > B1 0.383 >> YOLO 0.07-0.12 — BUT adapted DINO scores
  saturate >=0.99 (calibration; sdis-anchored thresholds then favour B1),
  flagged. Eval incident fixed: dino_infer.py lacked LoRA injection ->
  empty preds silently swept; now hard-fails on empty preds. Pre-registered
  rule (sdis-val R@5%) selects B6 LoRA-H+ as S4; B1 = control; B4 LoRA-L
  will ALSO be run on test as a clearly-labelled POST-HOC efficiency
  candidate (latency), not the pre-registered pick.
- [2026-09-14 23:50 UTC] SEEDS DONE + FIgLib REPLAY (seed 0). All six A/E/C
  seed-1/2 runs TRAINING COMPLETE (5 segs each, full epochs). D_pre s1
  stopped at ep 60/63, s2 EARLY-STOPPED at ep 32 (ultralytics patience 20,
  phase done) — D_adapt s2 therefore starts from a shorter stage-1; noted as
  seed-variance factor. Launched: 3-valset eval for the 6 seed ckpts
  (v4evalr5) and D_adapt seeds 1,2 (init from D_pre s1 seg02 / s2 seg01).
  FIgLib-39 replay @deployed 0.30/0.15 (historical set, CPU, 3 models):
  expE(v3) 30/39 det, 5 pre-visible, med 240s | A 30/39, 7 pre-vis, 240s |
  E_ET 28/39, 9 pre-vis, 240s | C_EGT 31/39, 8 missed, **14 pre-visible**,
  med 180s. C's extra event + earlier median come WITH 2-3x the pre-visible
  (unverified) alerts at the same thresholds — consistent with its lower
  sdis precision; not an early-detection win until matched on FP. Registry
  74.9 node-h spent (of 384); 3 nodes active; deadline in 8h40m.
- [2026-09-15 00:40 UTC] SERVING BACKEND dino3 IS NOW REAL (reports/v4/
  BACKEND_DINO3_LORA.md). Finding: iteration-3's `dino3` registry entry was
  a DANGLING IMPORT (src/serve/backend_dino3.py never existed) — SMOKE_
  BACKEND=dino3 would have crashed at startup; the v3 report's "registered"
  claim was a stub. Now: Dino3Backend mirrors dino_infer.py LoRA path, auto-
  detects rank/alpha/geometry/backbone snapshot, strict state-dict load.
  PARITY vs cluster preds (same 1280x736 letterbox): B4 19/19 boxes, max
  |dscore| 2.2e-4, min IoU 0.9995; B6 9/9, 3.0e-4. LoRA-merge equivalence
  3.3e-6. INTEGRATION on :8011 via env vars only: /healthz backend=dino3,
  3 real images through /detect, service on :8000 untouched. CPU b1 latency
  B4 5.62s@8thr (= v3 frozen ViT-L; LoRA adds no cost), B6 4.1s@96thr.
  Thresholds: at YOLO's 0.30/0.15 B4 flags 61%/66% of sdis-val negatives —
  DINO backends MUST use val-selected confs (B4 0.93/0.82, B6 0.95/0.88 at
  expB-matched negFP). New config fields all default to auto/off.
- [2026-09-15 01:05 UTC] DINO FOLLOW-UPS DONE (appended to reports/v4/
  dino_adapt_results.md). FIgLib-39 replay at sdis-val-selected thresholds
  (v3 rule: conf@10%/20% negFP -> B4 0.9881/0.973, B6 0.9901/0.9801):
  B4 26/39 (13 missed, med 481s), B6 24/39 (15 missed, 539s) vs expE 30/39
  (240s) and v3 frozen-B1 34/39 (240s); paired vs expE B4 median +181s
  LATER, B6 +62s later; 0 pre-visible alerts. Diagnostic: only 25% of post-
  visible FIgLib frames clear B4/B6's conf-single vs 55% for B1's 0.5624,
  although B4/B6 raw scores are HIGHER (median 0.86/0.93 vs 0.66) — the
  adapted models' operating points sit in the saturated tail and do not
  transfer to HPWREN frames. Negative result for THRESHOLD TRANSFER (per-
  domain negatives needed), not proof the detectors are worse. Latency 1x
  L40S b1 incl. preprocessing: B4 merged 157.5ms (= frozen-L 156.6; LoRA
  free after merge, equivalence 1e-6), B6 merged 351.6ms, expE 11.2ms.
  --lora-merge device bug fixed in dino_infer.py. ~0.6 node-h.
- [2026-09-15 01:15 UTC] 3-SEED RESULTS (reports/v4/seed_summary.json; A/E/C
  n=3, D n=1 pending, B n=1). mean±std:
  * sdis val R@1%/10%: A 0.055±.001/0.305±.003 | E 0.044±.002/0.277±.021 |
    C 0.043±.002/0.263±.002 | D(1) 0.082/0.300 | B 0.021/0.092.
  * boreal val R@1%/10%: A 0.596±.033/0.807 | E 0.905±.055/0.869(FP floor) |
    C 0.901±.051/0.941 | D(1) 0.909/0.945 | B 0.194/0.664.
  * forest val R@0.20/0.30: A 0.246±.015/0.169 | E 0.277±.014/0.221 |
    C 0.279±.038/0.204 | D(1) 0.233/0.171 | B 0.001/0.
  VERDICTS (seeds): G on top of T adds NOTHING (C vs E: -0.4pt boreal,
  -1.4pt sdis, +0.2pt forest — all inside seed std). T costs ~3pt sdis
  R@10% and buys +31pt boreal R@1%. Curriculum D is the only arm that keeps
  sdis performance AND gains aerial — still 1 seed. PRE-REGISTERED RULE:
  S2 = E_ET (diff -0.0041 < 1pt tie -> E_ET). Median seeds: A=v4a_s2 (0.304),
  E=v4e_s1 (0.291), C=v4c_s1 (0.262). boreal @1% rests on ~2 of 217 neg
  images (std 0.05) — report with that denominator.
- [2026-09-15 00:50 UTC] END-GAME PREP COMPLETE. (a) cluster/v4/run_final_
  test_eval.sh (YOLO systems, one-shot guard) + cluster/v4/run_final_test_
  eval_dino.sh (B6/B1/B4, dry-run passed, no submission, guard .DINO_
  EVALUATED_ONCE) + reports/v4/test/dino_test_thresholds.json (val-selected
  confs for both regimes). (b) scripts/v4/make_report_tables.py -> 27
  tables reports/v4/tables/ (all from artifacts; T25-27 TEST = PENDING
  auto-fill). (c) reports/v4/FINAL_REPORT_V4.md DRAFT (353 lines) with
  {{TEST:...}} placeholders + Chinese exec summary skeleton. Ledger note:
  CSV sums iter-1 to 40.1 node-h vs 41.9 quoted at 19:40Z (CSV = record).
  Waiting only on D_adapt s1/s2 (seg2/3).
- [2026-09-15 02:22 UTC] D_adapt seeds 1,2 TRAINING COMPLETE (3 segs each);
  val eval launched (v4evalr6). FINAL TEST — DINO half STARTED (one-shot,
  guard armed): cluster/v4/run_final_test_eval_dino.sh for B6 (pre-reg
  pick) / B1 (control) / B4 (post-hoc efficiency, labelled) on sdis_test +
  boreal_test + forest_test, thresholds fixed in reports/v4/test/dino_test_
  thresholds.json BEFORE this run. YOLO half waits ~20 min for the D_curr
  median seed (pre-registered median-seed rule needs 3 seeds).
- [2026-09-15 02:40 UTC] D_curr 3 SEEDS + FINAL TEST STARTED. D_adapt s1/s2
  complete (26/26 ep). D_curr 3-seed: sdis R@1%/10% 0.059±.017/0.321±.018
  (seed-0's 0.082 was the high draw; @10% it is now the BEST arm, A 0.305),
  boreal R@1% 0.896±.022, forest R@0.20 0.245±.016. Verdict holds:
  curriculum keeps source-domain performance and gains aerial; G-as-init
  helps, G-as-mixture does not. Median seed v4dada_s1 (0.318).
  UNEXPECTED: at 02:20Z a duplicate D_adapt eval job (v4evalr6-kmkj) and at
  02:31Z the DINO final test (run_final_test_eval_dino.sh --skip-stage,
  jobs dev4b6lorahplust/dev4b1frozenltes/dev4b4loraltest, after a staging
  job stgv4evaltest) were launched from this host by a process other than
  the coordinator or its monitor agent (provenance under investigation; see
  next entry). The DINO run uses exactly the pre-registered systems and
  thresholds and armed its once-guard, so it is being allowed to finish
  rather than killed and re-run (a re-run would break "evaluate once").
  Duplicate eval cost ~0.2 node-h. YOLO FINAL TEST launched 02:40Z (one-
  shot, run_final_test_eval.sh): S0 expE, S0b expB(deployed), S1 A median
  v4a_s2, S2 E_ET median v4e_s1, S2c C_EGT median v4c_s1 (matched control),
  S3 B seed0, S3 D_curr median v4dada_s1 (POST-HOC addition, 3 seeds now
  available; not the pre-registered pick — labelled).
- [2026-09-15 03:00 UTC] CORRECTION + COORDINATION. The "other process" was
  the pre-restart coordinator session (cvproject-9d), still alive and
  executing the same plan. Reconciled by message: it launched the DINO one-
  shot final test (pre-registered systems/thresholds; first attempt 02:20Z
  aborted on a validator bug BEFORE any inference, only staged the test tar)
  and the FIRST D_adapt val eval (v4evalr6-kmkj). The duplicate was
  therefore MINE (monitor agent's v4evalr6-t726, ~0.2 node-h) — corrected.
  It has stood down (no further jobs/edits); this session owns the close.
  D_curr checkpoint shipped to test (v4dada_s1) happens to equal the median-
  seed pick (0.318) computed after launch — D stays labelled POST-HOC because
  it was not a pre-registered candidate, and the deviation is stated in §6.
  All three DINO test jobs SUCCEEDED; YOLO final test RUNNING.
- [2026-09-15 03:20 UTC] FIgLib-39 replay for the TEST CHECKPOINTS (historical
  set, deployed 0.30/0.15, CPU 40thr x4 parallel ~10min): A v4a_s2 30/39
  (6 pre-visible, med 240s) | E v4e_s1 32/39 (12 pre-vis, 240s) | C v4c_s1
  32/39 (14 pre-vis, med 60s) | D v4dada_s1 32/39 (11 pre-vis, 240s) |
  expE 30/39 (5). Every T-trained arm finds +2 events but doubles the
  unverified pre-visible alerts at the same thresholds — reported as
  historical comparison with that caveat, not an early-detection win.
  expE/expB evaluated on boreal/forest VAL (v4evalr7) so the target-adapted
  threshold regime exists for the deployment baselines too. YOLO final test
  job RUNNING (38 min); DINO test done. Registry 82 node-h.
- [2026-09-15 03:55 UTC] FINAL TEST EVALUATED ONCE (YOLO job v4finaltest-m2k5
  03:32Z; DINO jobs 02:31-03:00Z). reports/v4/test/. Systems: S0 expE, S0b
  expB(deployed), S1 A=v4a_s2, S2 E_ET=v4e_s1 (pre-registered pick), S2c
  C_EGT=v4c_s1, S3 B seed0, S3 D_curr=v4dada_s1 (post-hoc), S4 B6 LoRA-H+,
  S4c B1, S4p B4 (post-hoc).
  * sdis TEST (917 neg) matched-FP R/tiny @1/10/20%: A 0.130/0.054 0.385/
    0.287 0.549/0.441 = BEST at every FP; expE 0.107 0.350 0.506; expB 0.118
    0.354 0.501; E_ET 0.100 0.310 0.424; C 0.116 0.318 0.448; D 0.122 0.362
    0.478; B 0.019 0.059 0.098; B6 0.051 0.363 0.558; B1 0.068 0.322 0.514;
    B4 0.080 0.329 0.509. Target data COSTS fixed-camera recall (E -7.5pt
    @10%); curriculum costs least (-2.3pt); DINO LoRA ties at 10-20% but
    loses at strict FP (confs 0.98-0.997, saturation).
  * boreal TEST (1 burn, 2 negs): source-only-regime recall A 0.906, E 0.922,
    C 0.921, D 0.923, expE 0.877, expB 0.886, B 0.865; target-adapted A 0.924
    E 0.934 D 0.931; B6 0.93. The +31pt VAL gap (heinola) did NOT reproduce
    on the test burn (ruokolahti): T buys +1.6-2.9pt; everyone ~0.9.
  * forest TEST (new region, 0 negs): source-only regime ALL <=0.08
    (thresholds don't transfer). target-adapted (boreal-val 1% conf): E_ET
    0.462 (tiny 0.27), D 0.351, C 0.266, A 0.092, expE 0.092, B 0.000, B6
    0.416. Per-event: E_ET ground burns 0.04-0.81, aerial flight 0.67-1.00.
    Those YOLO confs (E 0.07, D 0.17) are unusable on fixed cameras.
  DECISIONS: default-replacement criteria NOT met -> deployed default
  UNCHANGED (expB). A beats expE on test but tied on val -> seed variance;
  registered as option only. Aerial generalization PARTIALLY demonstrated
  (recall ~0.9 on one UAV burn for all systems; FP side UNVERIFIED); new-
  region forest gains real but require target-domain threshold calibration.
  Earlier detection NOT demonstrated. Research question: generic smoke
  pretraining does NOT beat the existing recipe or straightforward mixing;
  curriculum (G as init) is the only recipe keeping source performance.
  Ledger: 12 'probe-bindle-*' jobs (same requester, other project) ran in
  the window — flagged external, excluded from v4 totals. v4 in-scope:
  ~81 node-h / 647 GPU-h of 3,072 (152 jobs, 3.5 node-h failed/retry).
- [2026-09-15 04:00 UTC] Ledger final (external probe-bindle excluded):
  iteration-4 in-scope 140 jobs, 80.3 node-h = 642 L40S GPU-h of 3,072 cap
  (training 74.3, eval 2.2, feat-extract 0.2, failed/retry 3.5). Registry:
  0 active reservations, 0 live smoke-detect jobs. Report fill in progress.

- [2026-09-15 04:00 UTC] ITERATION 4 CLOSED — 4h33m inside the 24h window.
  Deliverables: reports/v4/FINAL_REPORT_V4.md (§0 中文执行摘要, §13 repro),
  DATA_CARD_V4.md, TRANSFER_RESULTS_V4.csv, AERIAL_EVAL_V4.md,
  EARLY_WARNING_V4.md, COMPUTE_LEDGER.csv, LOCKBOX_V4_FROZEN.md (OPENED
  once; now historical), PREREGISTRATION_V4.md, RECONCILIATION_V4.md,
  BACKEND_DINO3_LORA.md, tables/ (29), test/ (all preds+evals). Headlines:
  generic-smoke pretraining does NOT beat the existing recipe; G-as-mixture
  adds nothing; curriculum keeps source performance; T's +31pt aerial VAL
  gain shrank to +0.3-1.7pt on the test burn; new-region forest gains need
  target-domain thresholds; DINO LoRA > frozen but saturated scores break
  threshold transfer; temporal head > persistence but v4 extension is a
  domain shortcut; earlier detection NOT demonstrated; aerial generalization
  PARTIAL (recall yes, FP unverified). Deployed default UNCHANGED (expB).
  Compute 80.3 node-h / 642 GPU-h (21% of cap), 3.5 node-h failed/retry.
  0 live jobs, 0 reservations. Two coordinator sessions were active after
  the 17:40Z restart; reconciled, no duplicated test inference.

## ITERATION 5 — low-false-alert real early warning (prep opened 2026-10-02 00:35 UTC)

- [2026-10-02 00:45 UTC] V5 DAY-1 OPENED on user approval (two parallel
  lines: do-now work + proactive acquisition). Host had rebooted 5 days ago:
  service on :8000 was DOWN — restarted with file defaults (expB@1280,
  healthz OK). Cluster: 39 nodes free, 0 queued; other sessions present
  (cvproject-75 idle) — coordination message sent. Scaffolding: cluster/v5/
  BUDGET_V5.json (prep cap = 1 node / 24 node-h, deadline 2026-10-03T00:35Z),
  registry made iteration-selectable (SMOKE_REGISTRY_DIR; cluster/v5/
  registry.sh), reports/v5/PROTOCOL_V5.md FROZEN (two questions, scenarios
  separated, data-use-by-purpose incl. AuSmoke/MultiNatSmoke = PERMISSION
  PENDING, episodes/h with valid-time rules, calibration = hold recall then
  minimise FP, acceptance criteria). Negative-footage inventory on disk:
  ~13 min from new domains; UAV = 0.
- [2026-10-02 00:55 UTC] DAY-1 WORKFLOW LAUNCHED (8 parallel agents, CPU-first,
  ≤1 node): fixed-camera calibration baseline (expB/expE/A/B4/B6 on sdis VAL
  sequences, episodes/valid-hour, FIgLib ≥30/39 constraint), aerial
  calibration baseline (heinola 217 negs + 14 non-smoke clips, recall ≥0.90
  rule), shadow-mode service (source=historical_replay tagging, demo on
  :8012), review/onset/export tools, Boreal train-clip + sdis flicker onset
  audit, MmodalFire mist/early-fire review + aux package, data inventory by
  purpose (permission-pending marked), acquisition leads + FGI/AuSmoke email
  drafts for user review. Deliverables due within 24 h.
- [2026-10-02 02:00 UTC] DAY-1 RESULTS (7/8 agents; MmodalFire review retried
  after an input-size error). All CPU, 0 node-h used.
  * FIXED-CAMERA CALIBRATION (reports/v5/CALIBRATION_FIXED_V5.md): sdis VAL
    embedded negatives = 4.15 valid h @29 s cadence, 8 cameras. Anchor expB
    @0.30/0.15: frame R 0.680, 31.1 episodes/h, 38% of time in alert.
    Holding R>=0.680 and minimising episodes: EVERY system lands at 29.4-30.6
    eps/h (expB 0.30/0.23 -5%; B4 0.91/0.89; B6 0.94/0.88); halving episodes
    costs 26-29pt recall for all. v4a_s2's sdis-only point fails FIgLib
    (29/39) -> not accepted. VERDICT: thresholds alone cannot deliver -50%
    episodes at non-inferior recall; embedded negatives are a stress test
    (adjacent to labelled smoke), continuous clean footage still required.
  * AERIAL CALIBRATION (reports/v5/CALIBRATION_AERIAL_V5.md): heinola VAL
    217 negs + 14 non-smoke clips (7.3 min, ONE burn). At R>=0.90: expB needs
    conf 0.107 and flags 73/217 negs, 11/14 clips (90 eps/h); E_ET 0.216 ->
    0/217, 6/14 (49/h); D 0.212 -> 1/217, 7/14; DINO B4/B6 >=0.989 -> 0/217,
    4/14 (33/h) BUT 0.000 recall on the AuSmoke DJI val unit (saturated
    thresholds don't transfer). 3 "non-smoke" clips (heinola_13/53/62) fire
    for EVERY system at DINO 0.999-1.000 -> must be human-reviewed before
    being counted as false alerts (video-level label may be wrong).
  * SHADOW MODE (reports/v5/SHADOW_DEMO.md): src/serve/shadow.py; /detect
    `source` field {live_upload, historical_replay}; response byte-identical
    with/without shadow (test PASS 4/4); demo on :8012 = 61 records, ALL
    historical replay; :8000 untouched.
  * TOOLS (reports/v5/TOOLS_V5.md): negative_review.py (19 units, 45 sheets,
    verdict CSV; verified 0 h — no human verdicts yet), onset_annotate.py
    (blind mode works; 3 provisional annotations), export_candidates.py (10
    onset + 10 high-FP sdis VAL candidates; 7/10 onset candidates have models
    firing BEFORE the first positive label -> label-onset leakage suspected).
  * ONSET AUDIT (reports/v5/AUDIT_ONSET_V5.md): Boreal train clips 30/30
    already-smoke-at-start, 0 true onsets, 0 turn-induced; pyro-sdis flicker
    = annotation OMISSION (96/140 islands, 1 genuine fade); 0/6 label-
    "appears" starts verified. True onset candidates in train material: 0.
    PROTOCOL §6 onset acceptance NOT reachable from disk material.
  * INVENTORY (data/v5/manifests/DATA_INVENTORY_V5.{csv,md}): verified
    aerial negatives 0.12 h (need 2 h/3 units); forest-ground 0; human-
    reviewed onsets 0 (need 10). Permission-pending: AuSmoke 15,248 fr +
    MNS G 14,609. Quarantine 68,008 frames.
  * ACQUISITION (reports/v5/ACQUISITION_LEADS_V5.md, EMAIL_DRAFTS_V5.md):
    18 ranked leads; downloaded 6.2 GB licensed candidates to data/v5/raw/
    (BAMBI UAV nadir sample CC-BY 96 s; MONIMET Finnish phenocam CC-BY 6,375
    JPG/168 camera-days @30 min; US-gov burn B-roll; Zenodo smoke-bomb clips)
    — all CALIBRATION/VAL candidates, UNVERIFIED, no model has seen them.
    Best leads: BAMBI 45 h UAV no-smoke (CC-BY 4.0), RxCADRE 2012 onset w/
    ignition times (public domain, >20 GB), FLAME3 No-Fire UAV (login).
    LICENCE CORRECTION: HPWREN/FIgLib and ALERTCalifornia state CC BY-NC-ND
    4.0 (v4 card said "credit-reference") -> amendment below. Emails NOT sent.
- [2026-10-02 02:40 UTC] MmodalFire review DONE (retry; reports/v5/
  MMODALFIRE_REVIEW_V5.md). 47 sheets / 710 frames / 16 sessions viewed.
  Mist1/2/4 = clear ground-hugging white plume in every frame (strong smoke
  look-alike); Mist3 no visible plume (uncertain); dry-ice 1-4 visually =
  blank. 4 onset-interval proposals (cotton/wood/PU/n-heptane); 3 of 4 are
  FLAME-FIRST with thin plume — human must rule whether that counts as
  "clear smoke". AUX_LABEL_PACKAGE: 300 frames, origin=ai_agent_proposal.
  Indoor-only auxiliary candidate; never forest/UAV eval. Running: new-
  footage candidate-FP scoring + REVIEW_QUEUE_V5; BAMBI expansion (≤40 GB).
- [2026-10-02 02:50 UTC] NEW-FOOTAGE CANDIDATE-FP TRIAGE DONE (reports/v5/
  NEWFOOTAGE_CANDIDATE_FP_V5.md; 0.31 node-h on the one allowed node for
  DINO; YOLO on CPU). MONIMET phenocam (6,375 fr, 183 camera-days, 30-min
  SAMPLED cadence -> frame-FP source, not episodes): alert-frame rate at
  Day-1 operating points expB 5.40%, expE 5.38%, B4 0.77%, B6 0.52%; YOLO/
  DINO overlap only 23 frames; YOLO alerts cluster in leafless/overcast
  months. KEY: the "no-smoke" MONIMET set CONTAINS REAL SMOKE (2020-04-21
  17:01 dense field-burning plume, all 4 systems fired; 04-20 thin column)
  -> confirms that neither dataset labels nor detector silence verify
  no-smoke; human review mandatory. BAMBI flight_98 (96 s): YOLO 1-3/96
  frames (some over houses), DINO emits nothing. REVIEW_QUEUE_V5.md
  consolidated: 1,793 frames / 91 sheets / 20 sequences / 3 annotations +
  MmodalFire 300 labels + 4 intervals (row (e) patched after the package
  landed). §6 acceptance still NOT MET (1 camera-year, 1.6 min of flight).
- [2026-10-02 03:10 UTC] BAMBI EXPANSION DONE: 12 new flights (39.6 GB, md5 OK,
  zips deleted after 1-fps extraction) + flight_98 = 13 flights / 4,140 fr /
  1.15 h (61.5 min reviewable; flight_367 pre-dawn black), 11 physical sites
  (10 usable; BAMBI metadata mislabels flight_144's site — fixed from SRT
  GPS), 5 Austrian states, 4 seasons, CC-BY-4.0 re-verified per Zenodo
  record (17 of 47 raw parts unpublished). Habitat mixed (farmyard take-
  offs, enclosures; clean canopy = 98/58/61); flight_269 tractor exhaust
  flagged. Review package data/v5/review/bambi/ (179 sheets). Inventory
  regenerated with V5 rows (63) and FIgLib licence = CC BY-NC-ND.
  DAY-1 DELIVERABLES INDEX: reports/v5/DAY1_DELIVERABLES_V5.md. All Day-1
  work complete at 03:10Z (2h35m after open); 0.31 node-h; 0 live jobs.

## ITERATION 5b — labeled-data repair, model collaboration, distillation (opened 2026-10-02 07:26 UTC)

- [2026-10-02 07:35 UTC] V5b OPENED (execution brief). Window to 2026-10-03
  07:26Z; ceiling 128 GPUs = ≤15 g6e L40S nodes + ≤1 p6 B200 node
  (cluster/v5b/BUDGET_V5B.json; registry now enforces per-GPU-type caps and
  tracks node-hours by type; use cluster/v5b/registry.sh or
  SMOKE_REGISTRY_DIR=cluster/v5b). State: service :8000 OK (expB default),
  green 39 nodes free, obsidian-pu B200 queue
  obsidian-pu-hyperpod-p6-b200.48xlarge-ap-south-1-tas shows 1 node
  available (ap-south-1; egress/ECR-region/CUDA stack must be re-verified);
  no Kaggle credentials on host (WSDataset original via Kaggle likely
  blocked). Arm A control LAUNCHED: expV5bA_ctrl_s3 (YOLO11m, expB recipe,
  60 ep, seed 3) job tianrgua-v5bactrls3s1. Parallel workflow launching:
  checkpoint-registry verification, 4 data acquisitions (FireSpot-CNX,
  Sheffield ORDA, FireMan-UAV-RGBT, WSDataset), B200 probe (90-min bound),
  ensemble development on sdis VAL (grouped OOF), FAST/ACCURATE demo infra,
  teacher inference over the E train corpus (DINO B4/B6 + RF-DETR) for arm
  D, DINOv3-L segmentation branch on eligible Boreal masks.
- [2026-10-02 09:00 UTC] PHASE 1 COMPLETE (10/10 agents).
  * REGISTRY (models/v5b/MODEL_REGISTRY.{json,md}): 12 checkpoints load on
    CPU (6 YOLO, 5 dino3, RF-DETR); 7B head + Qwen adapters cluster-only.
    FLAG R1: v4 arm-A "seeds" were all seed 0 (run_yolo_arm.sh never passed
    --seed) -> reports/v5b/RECONCILIATION_V5B.md; launcher now guarded; the
    running v5b control expV5bA_ctrl_s3 is a seed-0 replicate (labelled).
    FLAG R2: shipped B6 ckpt (seg10) != run best (seg05) — registered.
  * DATA: FireMan-UAV-RGBT 33 GB CC BY 4.0 CONFIRMED — 69,871 RGB frames,
    64,867 smoke-box frames / 69,232 boxes (CVAT; mostly interpolated large
    region boxes), 4,979 explicit negatives, by-video units, YOLO export
    train/val/holdout under data/v5b/yolo/fireman/. Sheffield UAV 1,336 fr
    CC BY 4.0, 5,283 smoke boxes, 30 negatives — ONE burn session (6
    recordings) -> single incident. FireSpot-CNX 3,996 (2,817 pos / 1,179
    author negatives, 5 sites) — NO licence anywhere -> PENDING, not
    trainable; its positives were already in v4 corpora via MNS (R3).
    WSDataset BLOCKED (Kaggle login; 18.4 GB archive incl. ~13k author
    negatives + 16k boxes) — unblock = user's Kaggle API token.
  * ENSEMBLE (reports/v5b/ENSEMBLE_DEV_V5B.md): site-grouped OOF on sdis
    VAL; best single B6 macro-recall 0.326; recommended ACCURATE = A+B4+B6+B1
    calibrated WBF 0.354 (+2.8pt; marguerite +3.9 CI[+1.0,+8.3], cabanelle
    +1.8 CI[-0.9,+4.0], RF-DETR alone best on cabanelle). Recipe + fuse()
    shipped (models/v5b/fusion/, scripts/v5b/fusion.py, 15 tests pass).
  * SEG: DINOv3-L frozen + FPN-lite head TRAINED on 665 eligible Boreal
    train-burn masks (628 masks + 37 negatives; 13 flights), val heinola
    382 frames; real weights models/v5b/seg/segV5b_a/; 19.5 node-min.
  * DEMO: FAST/ACCURATE modes (src/serve/modes.py, demo.py, static/
    demo.html); FAST byte-identical to golden; tests pass in 3 configs;
    screenshots reports/v5b/demo/. Production :8000 untouched.
  * TEACHER POOL: B4, B6, RF-DETR scored all 40,516 train images (E +
    Boreal train; AuSmoke excluded, permission pending); 5 jobs, 0 missing.
  * B200 PROBE: WORKS. 8x B200 183 GB, torch 2.8 cu128 sm_100 via ap-south-1
    smoke-detect:v4; real DDP fwd/bwd + save + --resume + reload PASS;
    retrieval via `kraken jobs connect -c` (new channel). 0.62 node-h B200.
    Caveat: toy val loss nan — reproduce on L40S before trusting B200 val.
- [2026-10-02 09:40 UTC] ARMS B/C/D LAUNCHED (green, 1 node each, seed 0,
  YOLO11m COCO @1280 b48, 43 ep = 2.30M exposures, matched to A ±0.3%).
  Base corpus 53,511 lines: E x1 38,454 + Boreal-train x2 4,124 + FireMan
  burn-A train capped 600/video 8,279 (1,572 explicit negs) + Sheffield x2
  2,654 -> new-source share 39.2% of E. Val = sdis val (as A).
  C (ignore regions, ≥2/3 teachers ≥10%-FP conf, IoU<0.1 to GT, fill with
  mean colour): only 148 images / 300 rectangles masked (145 sdis) — effect
  small by construction. D (pseudo boxes, ≥2/3 teachers agree ≥20%-FP conf,
  no GT within IoU 0.3): 495 pseudo boxes / 495 images (85% rest on the two
  correlated DINO teachers); 3,755 disagreement images kept unchanged; all
  explicit negatives kept. No teacher pool existed for FireMan/Sheffield ->
  C/D touch only E+Boreal (limitation). Jobs v5bbs0s1-rb6i, v5bcs0s1-bh5g,
  v5bds0s1-ftbh; manifests data/v5b/manifests/CORPUS_{B,C,D}_v1.json.
  Pre-registration frozen (reports/v5b/PREREGISTRATION_V5B.md; H1 FireMan
  burn C has 0 negatives -> recall-only). Launching D2 (looser, all-source
  ensemble pseudo-labels) as the "compute-assisted" arm.
- [2026-10-02 09:24 UTC] (clock note: STATUS timestamps written
  between 09:00 and 09:40 "UTC" above were coordinator estimates ~40 min
  AHEAD of the system clock — e.g. the pre-registration file's mtime is
  08:48:43Z; order of events is correct, absolute times in those entries are
  approximate. From here on timestamps come from `date`.)
  EVAL PACKAGES BUILT (data prep only): fireman_val_s5 1,706 fr (231 author
  negatives, 1 video, VAL); fireman_holdout_s10 1,949 fr (5 videos, 0 negs,
  SEALED H1 -> recall only; reports/v5b/LOCKBOX_V5B_FROZEN.md); sheffield_
  holdout rec_0008 327 fr — CONTAMINATED for arms B/C/D/D2 (B corpus used
  all Sheffield frames) -> H3 valid only for A/expB/expE/DINO; recorded.
  Overlap check vs 66,035 training images: FireMan sets 0/0. Tar 445 MiB.
- [2026-10-02 09:39 UTC] ARM D2 LAUNCHED (job v5bd2s0s1-ekb4, 43 ep)
  but is a near-replicate of D by construction: teachers scored the 9,606
  new-source train images (0.37 node-h) and added only 408 pseudo boxes
  total — FireMan 0 (every teacher box sits on/inside an author region box),
  Sheffield 2 (teacher plumes span several author tiles -> guard drops them),
  rest on E/Boreal. FINDING: on already-labelled data the teachers agree
  with the authors; pseudo-labelling can only add value on UNLABELLED
  positives. -> launching D3 = B + teacher pseudo-boxes on genuinely
  unlabelled smoke-positive material we hold (Boreal train-burn Subset-B
  clip frames, video-level smoke=yes, no boxes; FIRESENSE smoke_pos frames,
  video-level; D-Fire fire-only train images, smoke unknown). That is the
  real teacher-to-student test of the core story.
- [2026-10-02 10:06 UTC] ARM D3 LAUNCHED (job v5bd3s0s1-vk2d, 40 ep,
  57,307 lines, exposures -0.65% vs A). Teachers scored 4,844 unlabelled
  images (0.19 node-h): pseudo boxes on Boreal train-clip frames 3,254/3,338
  (all 104 smoke clips; the single no-smoke clip 0/32 fired — consistent
  with its video label), FIRESENSE smoke_pos 350/530 (near-field staged
  smoke, labelled), D-Fire fire-only 192/944 (752 excluded, never negatives).
  3,564/3,797 boxes have all three teachers in-cluster; boxes are LARGE
  plume regions (median 25-36% of frame) -> D3 adds whole-plume supervision,
  not faint-smoke boxes (limitation to state). Boreal frames stored as 1920-
  max resized copies (2.9 GB tar). 7 L40S nodes active, ~10.4 node-h.
- [2026-10-02 10:40 UTC] INCIDENT (recoverable): arms B/C/D segment-1
  jobs ended FAILED with phase stuck at 'training' and no final upload — the
  7.4 GB corpus tar staging ate the first minutes, the segment timeout was
  computed BEFORE staging, so the presigned upload URLs (ttl 3540 s) expired
  before the end-of-segment upload. Same quirk hit v4's first segments.
  Training progress is NOT lost: the run dir is on persistent /scratch and
  the entry resumes from its local last.pt (seg-2 jobs running). Fixed for
  future segments: train_entry_v4.sh now recomputes the budget after
  staging. Will verify seg-2 status shows resumed_from=local, epoch>9.
  Baseline val-eval job (expE/expB, 3 val sets) running 40+ min.
- [2026-10-02 10:57 UTC] Eval pipeline verified end-to-end on the v5b
  packages (job v5bval0, 6/6 preds). BASELINES on FireMan VAL (1,706 fr, 231
  author negatives, one video): expE R 0.963/0.993 @1%/10% negFP, expB
  0.994/1.000 — FireMan is NOT a faint-smoke benchmark (large plume region
  boxes; recall saturates for v1-era models); precision 0.36-0.52 is an
  annotation-convention artefact (one author region vs several predicted
  sub-boxes). Faint-smoke evidence therefore rests on sdis VAL tiny-recall,
  FIgLib replay and the sealed FireMan holdout (recall only). Arm B seg-2
  confirmed resumed_from=local at epoch 17 (no progress lost).
- [2026-10-02 11:29 UTC] REPORT PREP DONE: scripts/v5b/make_report_tables_v5b.py
  (18 tables, test tables PENDING), FINAL_REPORT_V5B.md DRAFT (14 {{TEST}}
  placeholders), PRESENTATION_OUTLINE_V5B.md (11 slides + "what did NOT
  work"), DEMO_SCRIPT_5MIN.md, COMPUTE_LEDGER_V5B.csv (fresh scrubbed dumps;
  5b so far L40S 16.6 node-h / 133 GPU-h, B200 2.7 node-h / 21.7 GPU-h).
  Artifact cross-check found 9 minor inconsistencies (all documented in the
  agent report; two clarified by dated notes in LOCKBOX_V5B_FROZEN.md and
  PREREGISTRATION_V5B.md: the ACCURATE recipe is H3-clean; H1 RGB = 19,459
  fr). B200 high-res DINOv3-L LoRA run at step 8,480/12,660, train mAP 0.405.
  Arms: A seg5, B/C/D seg3, D2/D3 seg2; monitor agent will run val eval.
- [2026-10-02 11:51 UTC] Arm A control complete (60 ep, seg05). Launched
  seeded A repeats s1/s2 via run_v4_arm.sh --seed (genuine seeds this time;
  no tar, pod-local expB.yaml) ahead of student selection. 8 L40S nodes.
- [2026-10-02 12:35 UTC] B200 TRACK (A) RUN COMPLETE — v5bB200_lora_L_1920: DINOv3-L
  LoRA r16 (v4 B4 recipe, bs 4/GPU eff 32, lr 1e-4, 10 ep) at 1920x1088 on the
  obsidian-pu B200 node, corpus E + Boreal-train (AuSmoke dropped by path ->
  data/v5b/coco/instances_train_eb.json, 40,516 img / 33,548 boxes). Image
  ap-south-1 smoke-dino3-b200:v1 (FROM smoke-detect:v4 + transformers 5.17 +
  pycocotools + DINOv3-L + 2,062 Boreal frames + code/jsons; 28.8 GB, pushed in
  7.4 min = 31 MB/s). Memprobe: bs 16/GPU = 44.5 GiB, bs 8 = 23.1, bs 4 = 12.4
  (178 GiB avail), throughput flat ~45 img/s/node -> kept bs 4 for schedule
  parity. 12,660 steps in 2.80 h (0.76 s/it, 42 img/s/node), no OOM; best
  mAP(1500-subset) 0.416 @ep7 (B4 0.409) BUT matched-FP on full sdis_val is
  WORSE than B4 at every FP level (5% FP: R 0.165/tiny 0.096 vs 0.210/0.144;
  conf 0.5: higher recall 0.885 vs 0.853 and higher neg-img FP 0.63 vs 0.57).
  Batch-1 latency 284 ms (B200, 1920) vs 162 ms (L40S, 1280). 3.18 B200
  node-h. Artifacts models/v5b/b200/v5bB200_lora_L_1920/{best.pt,best_model.pt,
  last_model.pt,metrics.jsonl,sdis_val_preds.jsonl.gz}, reports/v5b/
  {B200_DINO_HIGHRES.md,eval_v5bB200_lora_L_1920_sdis_val.json,
  matched_fp_b200_v5bB200_lora_L_1920_sdis_val.json}. Retrieval = kraken connect
  (base64+sha256; cluster/v5b/orchestrate_b200_dino.py). Incident: bare `wait`
  in the entry also waited on the tee procsub -> pod held at phase infer,
  finished via connect, job stopped manually; entries fixed (wait $PIDS).
  CONTROL LAUNCHED 12:28Z: v5bB200_lora_L_1280 (same corpus/recipe/node,
  1280x736; 0.30 s/it) to isolate resolution from the AuSmoke/steps/hardware
  confounds. Qwen plan written: reports/v5b/B200_QWEN27B_PLAN.md (image
  smoke-detect-vlm-b200:v1 70.7 GB built+pushed 33 min; smoke test pending node).
- [2026-10-02 13:50 UTC] B200 CONTROL COMPLETE — v5bB200_lora_L_1280 (same E+Boreal
  corpus/recipe/node, 1280x736): 12,660 steps in 1.15 h (0.315 s/it, 102 img/s
  node, 6.6 GiB/GPU), best mAP1500 0.416 @ep9 (== the 1920 run), pod exited 0,
  1.25 B200 node-h. VERDICT (reports/v5b/B200_DINO_HIGHRES.md): at identical
  data/steps/hardware, 1920x1088 is WORSE than 1280x736 at every matched
  neg-image FP level on sdis_val (5%: R 0.165/tiny 0.096 vs 0.193/0.135; 1%:
  0.055/0.019 vs 0.068/0.041) for 2.4x train time and 2.7x latency; subset
  COCO mAP is blind to it. AuSmoke removal (ctrl vs v4 B4) costs ~1.7 pt at
  1-5% FP, ~0 at 10-20%. DINO arm total 4.43 B200 node-h (cap 8). Qwen 27B
  bf16 200-step SMOKE TEST submitted 13:46Z (job tianrgua-v5bqwensmk-2g5n,
  res-b200qwensmk, image 70.7 GB — first node-side pull measurement).
- [2026-10-02 14:03 UTC] INTERIM VAL (job v5bval1; reports/v5b/val/
  SELECTION_V5B.md INTERIM): R@5%/R@10% sdis | boreal | FireMan —
  expE 0.190/0.309 | 0.751/0.779 | 0.981/0.993; expB 0.204/0.329 | 0.762/
  0.771 | 0.997/1.000; A 0.202/0.302 | 0.800/0.814 | 0.997/1.000;
  B 0.183/0.302 | 0.955/n/a | 0.995/0.997; C 0.183/0.276 | 0.946/0.948 |
  0.998/0.998; D 0.170/0.287 | n/a | 0.996/0.998 (boreal n/a = FP never
  reaches that level; B/C/D boreal R@1% 0.82-0.86 vs 0.55-0.56 for A/expE).
  Rule picks B (mean sdis+FireMan R@10% 0.650 vs D 0.643, C 0.637).
  READING: new labelled data does NOT lift fixed-camera faint-smoke recall
  (B ties A @10%, -2pt @5%) but lifts aerial recall by +15-30pt at matched
  FP; conservative C/D add nothing (as expected by construction). Launched B
  seeds 1,2 (A seeds 1,2 already running). Registry note: arm A's release
  did not run (launcher edited mid-run) — released manually by the monitor.
- [2026-10-02 14:45 UTC] B200 TRACK (B) Qwen3.5-27B bf16 LoRA SMOKE TEST (not a
  campaign) DONE: job tianrgua-v5bqwensmk-2g5n SUCCEEDED, 0.64 B200 node-h.
  Image smoke-detect-vlm-b200:v1 (vlm:v3 + 55.6 GB weights in 2 layers + SFT
  data/images + code; 70.7 GB) pulled in 10m36s; 200 steps at eff 64 (bs2xga4x8),
  1 MP, LoRA r16 on 256 linears (79.7M trainable): 7.69 s/step (v2 L40S nf4
  17-20 s), 66.5 GB/GPU peak, loss 0.877->0.392; adapter 294 MB pulled;
  eval_vlm plumbing OK (200 imgs, 14.6-16.3 s/img at 192 tok; 72-88% parse
  fails at 0.2 epoch -> no quality claim). Plan reports/v5b/B200_QWEN27B_PLAN.md
  = GO for a follow-up window, ~3.3-4.5 node-h per matched arm (eval-dominated;
  use batched generation). B200 TRACK TOTAL: 5.07 node-h job-RUNNING time
  (3.18 highres + 1.25 control + 0.64 smoke; registry reservation-based 6.13
  incl. queue waits); L40S 0. All B200 reservations released; no jobs running.
- [2026-10-02 14:51 UTC] FINAL STUDENT SELECTION (reports/v5b/val/
  SELECTION_V5B.md): rule picks ARM B (0.650 vs D 0.643, C 0.637, D2 0.636,
  D3 0.617). D2 0.158/0.274 and D3 0.164/0.239 sdis R@5%/10% — the teacher-
  assisted arms are the WORST on fixed-camera faint smoke; D3 (pseudo-boxes
  on unlabelled positives) loses 6pt @10% vs B. NEGATIVE RESULT for teacher-
  to-student as implemented: teachers add nothing on author-labelled data
  and their large whole-plume pseudo-boxes on unlabelled footage shift the
  student away from faint/small smoke. All student arms keep the +15-30pt
  aerial (boreal) gain. Seeds: A s1/s2 and B s1/s2 running (4 nodes).
- [2026-10-02 14:52 UTC] FIgLib-39 replay (historical, 0.30/0.15) for v5b
  arms: A 30/39 (6 pre-vis, med 240 s), B 29/39 (6), C 31/39 (7, 361 s),
  D 29/39 (4, 361 s), D2 31/39 (4, 240 s); D3 running. No arm beats the v4
  numbers (expE 30/39); event coverage unchanged by the new data.
- [2026-10-02 22:51 UTC] INCIDENT: Midway/ada credentials expired ~18:10Z.
  Two agents (seed monitor, DINO final-script prep) died on API auth; the B
  seed orchestrators crashed at their next segment submit (ada call) after
  seg 3 (epoch 27/43 for both; pods resume from local /scratch last.pt).
  A seeds s1/s2 had already completed (60 ep). User refreshed credentials
  at ~22:45Z; B_s1/B_s2 orchestrators relaunched with --start-seg 4; both
  agents resumed. ~4.5 h of wall time lost; window deadline 07:26Z holds.
- [2026-10-02 22:52 UTC] D3 FIgLib-39 replay: det 32/39, missed 7, pre-vis 14, med 60.0 s
  (reports/v5b/replay/replay_S4c_D3_figlib39.json). Service :8000 healthy.
- [2026-10-03 00:13 UTC] SEEDS DONE (reports/v5b/val/SEEDS_V5B.md; B s1/s2
  finished in one resumed segment, seeds verified 1/2). 3-seed mean±std,
  paired B-A: sdis R@5% -1.0pt [-1.9,-0.4], R@10% +0.7pt [0.0,+1.6], tiny@10%
  +0.5pt [-1.7,+2.4] -> within seed noise on fixed-camera; boreal R@5% +21pt
  [+15.5,+31.0] -> large and consistent. Median seeds: A = s3 (seed 0 rep,
  0.302), B = s0 (0.303). FINAL TEST LAUNCHED (one-shot, per-system guard,
  two disjoint batches to stay inside URL lifetime): v5bfinal1 = expE, expB,
  A s3/s1/s2, C; v5bfinal2 = B s0/s1/s2, D, D2, D3; 5 sets each (sdis TEST,
  FireMan holdout s10, Sheffield holdout, boreal test, forest test). Seeds
  other than the median are labelled seed-spread, not selected. DINO
  components (B4/B6/B1) on the two new holdouts + S5 fusion follow when the
  prepared script lands; v4 DINO test preds reused for sdis/boreal/forest.
- [2026-10-03 03:20 UTC] FINAL TEST batches v5bfinal1/2 (6 ckpts each) FAILED
  exit 97 with no uploads: ~70 min runtime exceeded the 59-min presigned-URL
  lifetime, so preds and logs could not be uploaded. No system was marked
  evaluated (guard intact; no results seen). Relaunched as four 3-ckpt
  batches (v5bf_b1..b4). DINO one-shot (B4/B6/B1 on FireMan/Sheffield
  holdouts) and the recipe yolo component v4a_s2 are running concurrently.
- [2026-10-03 03:55 UTC] FINAL TEST COMPLETE: four YOLO batches (12 ckpts) +
  DINO one-shot (B4/B6/B1 on FireMan/Sheffield holdouts; v4 preds reused for
  sdis/boreal/forest) + recipe yolo v4a_s2 + S5 fusion on all 5 sets. 65 eval
  JSONs in reports/v5b/test/. Guards armed; nothing re-run. Tables regenerated.
- [2026-10-03 03:56 UTC] FINAL TEST HEADLINE (H2 pyro-sdis TEST, 917 neg, matched
  FP R/tiny @1/5/10/20%): A(existing recipe, YOLO11m) 0.108/0.035 0.287/0.184
  0.388/0.283 0.553/0.435 = BEST YOLO at every level; expE 0.107 0.256 0.350
  0.506; expB(deployed 11s) 0.118 0.258 0.354 0.501; B(selected new-data)
  0.089 0.221 0.309 0.442; C 0.085 0.221 0.318 0.461; D 0.089 0.228 0.342
  0.462; D2 0.116 0.255 0.348 0.482; D3 0.096 0.260 0.358 0.466; S5 ensemble
  0.072 0.234 0.366 0.552; B6 LoRA-H+ 0.051 0.203 0.363 0.558. READING: no
  v5b data/teacher arm beats the existing recipe on fixed-camera faint smoke
  (B -8pt @10%); the val-selected arm B is also the worst student on test
  (val->test rank reversal among students = seed/site noise, stated); the
  ensemble does not beat the best single model at matched FP. H1/H3/H4 in
  reports/v5b/tables/T8b-e. Report fill launched.
- [2026-10-03 04:17 UTC] ITERATION 5b CLOSED (~3 h inside the 24 h window).
  Correction to the 03:56Z entry: at 1% FP (9 negative images) v4a_s2 0.130,
  expB 0.118 and D2 0.116 exceed A 0.108; A is best at 5/10/20%. The two
  failed final batches cost 5.0 node-h incl. scheduler auto-restarts (not
  ~1.1); the bad-tag relaunch never created kraken jobs (no cost).
  DELIVERABLES: reports/v5b/FINAL_REPORT_V5B.md (§0 中文执行摘要, 0 placeholders),
  COMPARISON_V5B.csv (523 rows, contamination flags), per-example preds
  reports/v5b/test/*.raw.parquet + models/v5b/evals/, models/v5b/
  MODEL_REGISTRY.{json,md} (34 entries, all loadable verified), data/v5b/
  manifests/* (provenance incl. annotation_origin), B200_PROBE.md +
  B200_DINO_HIGHRES.md + B200_QWEN27B_PLAN.md, DEMO_V5B.md (FAST/ACCURATE,
  tests pass), PRESENTATION_OUTLINE_V5B.md (11 slides), DEMO_SCRIPT_5MIN.md,
  COMPUTE_LEDGER_V5B.csv (scheduler-derived, scrubbed). DECISIONS: core-story
  answer NO; FAST default candidate = S1_A (rule met, +3.4pt vs expB on H2;
  production NOT flipped — user decision, env block + tests in §8); ACCURATE
  ensemble NOT shipped (0.366 vs 0.388 @10%); data/distillation/high-res arms
  are negative results. Compute (registry): L40S 58.2 node-h, B200 6.1
  node-h; 0 active reservations; 0 live jobs; service :8000 unchanged.

## ITERATION 6 — SMOKE-EDGE-SYNTH (window 2026-10-20T19:50:37Z → 2026-10-21T19:50:37Z)

- [2026-10-20 20:12 UTC] V6 OPEN. Scaffolding: cluster/v6/{BUDGET_V6.json,
  registry.sh,run_yolo.sh,train_entry_v6.sh,orchestrate_v6.py} (entry adds
  TRAIN_SCRIPT/EXTRA_ARGS for the KD arm; outputs under models/v6/runs),
  scripts/v6/bedrock_ledger.py (USD 2000 cap, USD 200 gencmp sub-cap,
  reserve-before-submit), reports/v6/PREREGISTRATION_V6.md FROZEN (arms
  A/A-long/B/C/D/E at YOLO11n 1280/48, step-matched via corpus-size+epochs;
  decision rules; synthetic-data rules; edge targets).
- [2026-10-20 20:12 UTC] ARM A (real-only YOLO11n baseline) LAUNCHED ~20 min
  into the window: expV6A_realn_s0, baked expB recipe (pyro-sdis+D-Fire),
  60 ep / imgsz 1280 / batch 48 / seed 0, init s3 v6/weights/yolo11n.pt via
  URL_INIT. Job tianrgua-v6an0s1-hmxw RUNNING on green g6e use1 (3 free of
  39 at launch). Registry: v6a-realn-s0 1 node (1/16).
- [2026-10-20 20:12 UTC] Recon workflow wf_ec0cd73d-268 running: (1) AWS/
  Bedrock routing+entitlement+pricing for 000000000002/000000000003/
  000000000001 (read-only, no agreements); (2) Qwen-Image-2.1 / GLM-Image /
  Stability-on-Bedrock licence+capability research; (3) real-data inventory:
  expB_train counts, frozen test sets, fresh-holdout search, background pool
  with neg-evidence, teacher selection by val; (4) B200 generator plan
  (queue, ECR bake pattern, sizes, fallback L40S).
- [2026-10-20 21:30 UTC] Recon landed (wf_ec0cd73d-268, 4 agents): (a) Bedrock:
  all Stability image models/services AUTHORIZED+ENTITLED in all 3 accounts
  (inpaint $0.07 20RPM via us.* profile; SD3.5L $0.08/Core $0.04/Ultra $0.14
  usw2-only; Nova Canvas EOL); no new agreements needed; AWS Service Terms
  50.5 training-use ambiguity -> Bedrock outputs are comparison/demo only
  (prereg A1). (b) Licences: GLM-Image MIT (clean) = arm-C generator;
  Qwen-Image-2.1 research-only+attribution = excluded; LICENSES_V6.md.
  (c) Inventory: N_R=38,454 (27,197/11,257); bg pool 12,890 author-negative
  train images; teacher by val rule = expB_y11s (R@10 0.329); fresh holdout
  H-NEW = FireSpot-CNX-r1 eval-only (licence pending, disclosed).
  (d) B200: queue idle, GLM fits resident; plan reports/v6/B200_GEN_PLAN.md.
- [2026-10-20 21:30 UTC] JOBS: arm A (v6an0s1) RUNNING ~35min; A-long
  (v6al0s1) RUNNING; usw2 probes SUCCEEDED (ECR cross-region pull + S3
  GET/PUT via use1 presigned URLs verified) -> usw2 g6e (8 free) usable.
  GLM-Image 34GB weights uploaded to S3; smoke job v6glmsmk RUNNING (8
  black-bg prompts, 2 GPUs, measures s/img). Bedrock probes+comparison:
  8 probe imgs (mask polarity WHITE=edit confirmed; black-polarity inverts
  scene = auto-rejectable; t2i gives studio smoke w/ spurious objects ->
  needs source-crop filtering) + 96 t2i + 30 inpaint, 0 failures, ledger
  $10.38 of $200 gencmp. Arm-B corpus v1 built (38,454, 0 err) then REBUILT
  after visual QC caught sky-floating plumes: placement now anchored to the
  EMPIRICAL real-plume base distribution (per-camera pyro-sdis, global
  dfire, uniform-terrain aerial) — rebuild running.
- [2026-10-20 21:58 UTC] GENCMP interim (126 Bedrock imgs, $10.38): SD3.5L
  85% acceptance $0.094/acc 5.8s; Core 17% (frame-filling) $0.24/acc; Inpaint
  100% localized, median outside-mask change 0.0000, $0.07, 17.6s. Automated
  filter: coverage/colored-object/flat-blob/near-dup (thresholds tuned after
  visual inspection of rejects: sat 0.28, cov<=0.85). Arm-B corpus v2
  (anchored placement) packed: data/v6/stage/v6synthB.tar 11.1GB,
  CORPUS_B_v1.json; arm B launched on usw2 (v6b0, 30ep matched-steps).
  Edge toolchain installed (tf 2.21 cpu); export validation running on
  expB_y11s (ONNX parity + TFLite fp16/int8 PTQ, calib=train/val only).
  VLM review of 300 stratified arm-B composites running (Haiku 4.5, ledger
  tag review). GLM full queue q_glm_full.jsonl ready (3,480 items).
- [2026-10-20 22:35 UTC] GLM-Image VALIDATED on L40S: v6glmsmk 8/8 imgs,
  ~91 s/img @1024 (DiT 50 steps = 40s, AR stage ~50s; steps-cut not useful).
  Probe quality: usable volumetric smoke, but purple-cast + lit ground plane
  + tiny source objects -> filter patched (bg-floor subtraction, 75%
  desaturation, bottom-crop); probe acceptance 5/8 (3 correct rejects).
  VLM review of 300 arm-B composites: negatives clean (<=5% false smoke);
  positives plausible 44%, med quality 1/5 (copy-paste edges/halos; faint
  stratum often below VLM visibility) — recorded in SYNTH_DATA_CARD_V6.md as
  the crude-control characterization. FULL GLM GEN LAUNCHED: 3,480 assets,
  3 shards x 1 usw2 node, segmented resume via /scratch (v6glma/b/c, ~3.7h).
  B200 validation track staged: ECR repo smoke-diffusion-b200 (ap-south-1,
  cluster pull policy), 42GB image building (weights+worker+400-item queue
  baked), entry+orchestrator mirror the v5b connect/ack pattern. Ledger
  $10.88 (review $0.50). Registry 6/16 nodes.
