# V5b — High-resolution DINOv3-L LoRA detection on the B200 node (1920×1088 vs v4 B4 1280×736)

_V5b B200 track, 2026-10-02. Val only (sdis_val); no test/lockbox/FIgLib/AuSmoke/heinola-for-training touched._
_Status: **FINAL for the DINO arm** (high-res run + same-corpus 1280×736 control both complete; sdis_val only)._

## 0. Question and design
Does giving the same detector (DINOv3-L, LoRA r16 on q/k/v/o, SimpleFeaturePyramid neck, torchvision
Faster R-CNN head — the v4 B4 recipe) a 1.5× larger letterboxed input help small-smoke recall at matched
false-positive rate? Resolution is the variable: 1920×1088 (patch-16 grid 120×68 = 8,160 tokens) vs
1280×736 (80×46 = 3,680 tokens). Everything else is held at the B4 values (bs 4/GPU → eff. 32, lr 1e-4 head
and LoRA, wd 1e-4, warmup 500, cosine, 10 epochs, grad-ckpt, eval on the same 1,500-image sdis-val subset at
epoch boundaries, best.pt = COCO-mAP-selected).

**Stated confound (data):** v4 B4 trained on E + T = 51,420 images (T = Boreal train 2,062 + AuSmoke 10,904).
AuSmoke may not be used in V5b (permission pending), so this run trains on **E + Boreal-train = 40,516 images**
(pyro-sdis 22,177 / D-Fire 16,277 / Boreal 2,062; 33,548 boxes; 29,222 positive / 11,294 negative images;
`data/v5b/coco/instances_train_eb.json`, built by `scripts/v5b/build_train_eb.py` from the frozen v4 json by
dropping every `multinatsmoke__*AuSmoke*` path). Per epoch this is 21 % fewer images and 21 % fewer optimizer
steps (1,266 vs 1,607 per epoch; 12,660 vs 16,070 total). Any difference vs B4 is therefore
resolution **and** corpus; the AuSmoke removal is expected to hurt forest-type generalisation but has no
obvious direction on sdis_val (fixed-camera, European). A same-corpus 1280×736 control was not run (single
B200 node, cap 1; it would be the natural next ablation on an L40S node).

**Second confound (hardware/precision):** B200 vs L40S, same torch 2.8.0+cu128 / bf16 autocast / cuDNN 9.10.2
stack; numerics are not bit-comparable but the training recipe is identical.

Note on what "higher resolution" means per source: pyro-sdis frames are native 1280×720 → the 1920×1088
letterbox **upsamples them 1.5×** (scale 1.5, pad 8 px) — the ViT sees finer tokens (≈10.7 native px per
16-px patch instead of 16), not more sensor pixels. D-Fire images have mixed sizes; Boreal frames are
4K (3840/4096×2160) and are downscaled less aggressively (0.5× instead of 0.33×).

## 1. Infrastructure (all verified this session; details in `reports/v5b/B200_PROBE.md`)
- Node: obsidian-pu, queue `obsidian-pu-hyperpod-p6-b200.48xlarge-ap-south-1-tas`, 8× B200 (183,359 MiB each).
- Image `000000000001.dkr.ecr.ap-south-1.amazonaws.com/smoke-dino3-b200:v1` (`cluster/v5b/Dockerfile.dino3_b200`),
  FROM ap-south-1 `smoke-detect:v4` (torch 2.8.0+cu128 with sm_100, Python 3.11.13; pyro-sdis + D-Fire images
  already baked at `/opt/smoke_data`) + transformers 5.17.0 / pycocotools 2.0.10 (wheels resolved by the image's
  own pip on the devbox, `cluster/v5b/build/wheels`, installed `--no-index`) + DINOv3-L snapshot (HF rev
  `ea8dc286`, 1.2 GB, `/opt/models/dinov3-vitl16`) + the 2,062 Boreal-train frames (5.5 GB, the only corpus part
  missing from v4) + `src/v3` code, `cluster/v3/dino_infer.py`, the COCO jsons and the local sdis_val list.
  Build-time check: all 40,516 train and 7,335 val paths resolve inside the image. 28.8 GB (7 GB of which is a
  wasted duplicate layer from `RUN chmod -R` — use `COPY --chmod` next time). Pushed in 7.4 min (≈31 MB/s).
  New repo `smoke-dino3-b200` created with the 3-account pull policy (000000000005, 000000000006, 000000000004).
- S3 egress is denied from these pods (probe §3) → nothing is staged at run time; all retrieval through
  `kraken -p obsidian-pu jobs connect -c "sh -c '...'"` with base64 + sha256 verification
  (`cluster/v5b/orchestrate_b200_dino.py`). The pod entry (`cluster/v5b/dino_b200_entry.sh`) always exits 0
  (non-zero ⇒ automatic restart), runs `torchrun --standalone` with the operator `PET_*` rendezvous vars unset,
  `HOME=/tmp`, offline HF flags, and holds the pod for pulls at the end.
- Registry: `res-b200dinohr`, `--gpu-type B200 --track b200`; kraken job `tianrgua-v5bdinohr-mssz`.

## 2. Memory / batch-size audit at 1920×1088 (8× B200, LoRA r16 + grad-ckpt, 40-step probes) — measured
| bs/GPU | eff. batch | peak allocated GiB/GPU | s/it | img/s per node | OOM |
|---|---|---|---|---|---|
| 16 | 128 | 44.5 | 2.81 | 45.5 | no |
| 8 | 64 | 23.1 | 1.45 | 44.2 | no |
| **4 (used)** | **32** | **12.4** | **0.75–0.78** | **~42** | no |

Throughput is flat in batch size → compute-bound at ≈5.5 img/s per GPU; memory grows ≈2.7 GiB per image, so
bs ≈ 48–56/GPU would still fit in 178 GiB. **bs 4 was kept deliberately**: a larger batch would change the
optimizer-step count/LR schedule relative to B4 and confound the resolution comparison, while buying no
throughput. Reference: v4 B4 on 8× L40S at 1280×736, bs 4 → median 0.81 s/it = 39.4 img/s per node (from its
metrics.jsonl); the B200 node therefore processes 2.2× more tokens per image at ≈1.1× the image rate
(attention cost grows faster than linearly, so the per-FLOP speed-up is ≈2.5–3×; MFU is low, ≈12 %, with HF
SDPA on sm_100 — not tuned here).

## 3. Training run `v5bB200_lora_L_1920` (kraken `tianrgua-v5bdinohr-mssz`) — measured
- 10 epochs = **12,660 steps** at eff. batch 32 (1,266 steps/epoch), single segment, no OOM/retry, `DET_TRAIN_DONE`,
  train rc 0. Wall: training 2.80 h (first→last log line; 0.762 s/it median = **42.0 img/s per node**, peak
  12.4 GiB/GPU) + 10 epoch-boundary evals (1,500 sdis-val images each, < 1 min) + 8-shard full-val inference
  4.8 min. Pod RUNNING 09:16:02Z → stopped 12:26:35Z = **3.18 B200 node-hours (25.4 GPU-h)** incl. the 2
  memory probes (≈ 4 min) and ~15 min of retrieval/hold. Loss (mean of logged 20-step means) per epoch:
  0.184, 0.094, 0.083, 0.078, 0.072, 0.069, 0.064, 0.061, 0.057, 0.057.
- COCO mAP on the 1,500-image sdis-val subset (same subset/seed as v4) per epoch:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| step | 1,266 | 2,532 | 3,798 | 5,064 | 6,330 | 7,596 | **8,862** | 10,128 | 11,394 | 12,660 |
| mAP 1920×1088 | 0.312 | 0.372 | 0.384 | 0.401 | 0.403 | 0.405 | **0.416** | 0.412 | 0.416 | 0.413 |
| v4 B4 1280×736 (step) | 0.324 (1,607) | 0.380 (3,214) | 0.398 (4,840) | 0.393 (6,447) | 0.400 (8,230) | 0.395 (9,837) | 0.399 (11,628) | 0.401 (13,235) | **0.409** (15,026) | 0.408 (16,070) |

  best.pt = epoch 7 (step 8,862, mAP 0.4162 vs B4's best 0.4086) → `models/v5b/b200/v5bB200_lora_L_1920/best.pt`
  (305 MB with optimizer; `best_model.pt` / `last_model.pt` = 102 MB trainable-only), `metrics.jsonl`,
  `sdis_val_preds.jsonl.gz` (7,335 rows, conf floor 0.02). All pulled through `connect`, sha256-verified.
- Incident: the entry's bare `wait` after the 8 inference shards also waited on the `tee` process substitution
  (bash 5), so the pod never reached its retain phase; the post-inference steps (concat/gzip, status) were done
  through `connect`, the pod was stopped manually (≈ 10 min of idle node time). Both entry scripts now
  `wait $PIDS`.

## 4. sdis_val results — same eval code as v4 (`src.eval.eval_detection`, IoU ≥ 0.4, GT from the frozen local
## YOLO labels; 7,335 images: 6,177 positive / 1,158 negative; 2,277 tiny / 4,027 small / 260 medium / 39 large GT boxes)

### Matched negative-image-FP table (recall / tiny-recall / small-recall, conf interpolated on each model's own curve)
| FP(neg-img) | **v5bB200 1920×1088 (E+Boreal)** | v4B4 1280×736 (E+T) | v4B6 H+ 1280×736 (E+T) | v4B1 frozen-L 1280×736 (E+T) |
|---|---|---|---|---|
| 1 % | 0.055 / 0.019 / 0.075 (c=0.9953) | **0.085 / 0.051 / 0.105** (c=0.9971) | 0.097 / 0.038 / 0.134 | 0.060 / 0.027 / 0.079 |
| 2 % | 0.087 / 0.035 / 0.117 (c=0.9942) | **0.115 / 0.073 / 0.139** (c=0.9963) | 0.133 / 0.064 / 0.177 | 0.099 / 0.049 / 0.127 |
| 5 % | 0.165 / 0.096 / 0.207 (c=0.9917) | **0.210 / 0.144 / 0.249** (c=0.9934) | 0.227 / 0.144 / 0.277 | 0.175 / 0.104 / 0.219 |
| 10 % | 0.280 / 0.203 / 0.329 (c=0.9870) | **0.311 / 0.222 / 0.362** (c=0.9881) | 0.340 / 0.252 / 0.394 | 0.301 / 0.200 / 0.365 |
| 20 % | 0.460 / 0.382 / 0.513 (c=0.9768) | **0.478 / 0.392 / 0.528** (c=0.9730) | 0.494 / 0.416 / 0.543 | 0.464 / 0.341 / 0.540 |

Source: `reports/v5b/matched_fp_b200_v5bB200_lora_L_1920_sdis_val.json` (`scripts/v4/compare_matched_fp.py --top-fine`),
sweep `reports/v5b/eval_v5bB200_lora_L_1920_sdis_val.json` (+ `.raw.parquet`).

### Fixed-confidence view (why the two metrics disagree)
| conf | model | precision | recall | neg-image FP | tiny-R | small-R | medium-R | large-R |
|---|---|---|---|---|---|---|---|---|
| 0.50 | v5bB200 1920 | 0.647 | **0.885** | **0.631** | **0.845** | 0.920 | 0.815 | 0.103 |
| 0.50 | v4B4 1280 | 0.680 | 0.853 | 0.567 | 0.791 | 0.895 | 0.846 | 0.154 |

**Reading.** At any fixed threshold the 1920×1088 model finds more smoke (incl. +5 pts tiny recall at conf 0.5),
which is why its COCO mAP is higher — but it also fires on more negative images, and once the comparison is
made at matched negative-image FP (the project's deployment-relevant metric) it is **worse than B4 at every FP
level (−3 to −4.5 pts recall, −2 to −5 pts tiny recall)**, and at 1–10 % FP it is even at/below the frozen B1.
The medium/large recall drop (0.815 vs 0.846, 0.103 vs 0.154 at conf 0.5) is consistent with the anchor set
(16–256 px) not being rescaled with the input: a 1.5× larger canvas makes the few large plumes exceed the
largest anchor/RoI statistics the head was tuned for at 1280.

## 5. Throughput, latency, node-hours
| item | v5bB200 1920×1088 (8× B200) | v4B4 1280×736 (8× L40S) |
|---|---|---|
| train step time (bs 4/GPU, eff. 32) | 0.762 s/it → 42.0 img/s/node | 0.811 s/it → 39.4 img/s/node |
| peak train memory | 12.4 GiB/GPU (bs 16: 44.5 GiB; bs 8: 23.1 GiB) | ~40 GiB class (bs 4 needed grad-ckpt on 44 GB) |
| 10-epoch training wall | 2.80 h (12,660 steps, 40,516 img/epoch) | ≈ 3.3 h across 5 segments (16,070 steps, 51,420 img/epoch) |
| node-hours | **3.18 B200** (incl. probes, infer, hold) | 3.84 L40S |
| batch-1 inference, LoRA unmerged, preprocessing incl. | **283.8 ms median** (infer-only 259 ms; 1 B200, bf16-off fp32 head) | 161.6 ms (1 L40S, 1280×736) |
| full sdis_val inference | 7,335 img in 287 s on 8 GPUs = 3.2 img/s/GPU | — |

Reading: per image, the B200 node at 2.2× the tokens is only ≈1.07× faster than the L40S node at 1280×736 —
the ViT-L+Faster R-CNN stack is far from B200 peak (≈12 % MFU; HF SDPA, fp32 neck/head, RPN/NMS on a
480×272 stride-4 map). Inference latency grows 1.76× with resolution (2.2× tokens).

## 6. Verdict and caveats
- **Resolution alone did not buy matched-FP recall on sdis_val with this recipe — it cost some.** Against v4 B4
  the high-res run is behind at every FP level, and the same-corpus, same-node 1280×736 control (§7) shows that
  this is the resolution, not the AuSmoke removal or the hardware: at identical data, steps, LR schedule and
  hardware, 1920×1088 loses 1.3–2.8 pts recall and 2.2–3.9 pts tiny recall at 1–20 % neg-image FP, while the
  two runs have the same subset COCO mAP (0.416). Removing AuSmoke / 21 % fewer steps (control vs v4 B4) costs
  ≈1–2 pts at 1–5 % FP and nothing at 10–20 %.
- Native resolution matters for the interpretation: pyro-sdis frames are 1280×720, so 1920×1088 is a 1.5×
  upsample of the dominant val/train source — finer tokens over the same pixels. The size-bucket pattern
  (tiny/small up at fixed conf, medium/large down, negatives up) says the head's anchor/RoI statistics were
  implicitly tuned for 1280 and were not rescaled; a fair high-res variant should scale the anchor sizes
  (×1.5) and probably the RPN/RoI sampling, which was deliberately NOT done here to keep "same head".
- Threshold saturation (conf ≈ 0.99 at 1–5 % FP) is as in v4 — LoRA-adapted Faster R-CNN scores saturate;
  the matched-FP interpolation at the extreme tail is sensitive but identical for all models in the table.
- val-selected best.pt (epoch 7 by subset mAP) — same mild optimism as all v3/v4 DINO numbers.
- No test/lockbox/FIgLib/AuSmoke/heinola-for-training data was used; sdis_val only.

## 7. Control run `v5bB200_lora_L_1280` (kraken `tianrgua-v5bdinoctl-57zb`) — same corpus, recipe and node, 1280×736
- Launched 12:28Z on the same node right after the high-res run (image cached: PENDING→RUNNING in 36 s), no
  memprobe; identical entry/orchestrator (`--width 1280 --height 736`). 12,660 steps in 1.15 h (median
  **0.315 s/it = 101.7 img/s per node**, peak **6.6 GiB/GPU**), 10 evals, full-val inference 7,335 images in
  110 s (8.4–8.8 img/s/GPU), entry reached `retain` on its own (fixed `wait $PIDS`), pod exited 0 → job
  SUCCEEDED 13:43:58Z. **1.25 B200 node-hours.** Mean loss per epoch 0.148, 0.078, 0.068, 0.065, 0.061, 0.059,
  0.056, 0.052, 0.049, 0.050 (lower than the 1920 run's at every epoch — the 1920 model fits the training set
  less well, not more). Batch-1 latency 103.9 ms median on 1 B200 (infer-only 100.5 ms; preprocessing 3.5 ms
  vs 24.8 ms at 1920 — PIL resize/pad of a 1920×1088 canvas is 7× the 1280 cost).
- Subset COCO mAP per epoch: 0.291, 0.368, 0.374, 0.390, 0.380, 0.404, 0.406, 0.416, **0.416 (step 11,394 →
  best.pt)**, 0.415 — the same plateau as the 1920 run (0.416 @ step 8,862). Artifacts:
  `models/v5b/b200/v5bB200_lora_L_1280/{best.pt,best_model.pt,last_model.pt,metrics.jsonl,sdis_val_preds.jsonl.gz}`.

### Matched-FP, all five models (recall / tiny-R / small-R; `reports/v5b/matched_fp_b200_dino_sdis_val.json`)
| FP(neg-img) | **1920×1088 E+Boreal B200** | **1280×736 E+Boreal B200 (control)** | Δ (1920 − 1280) | v4B4 1280 E+T L40S | v4B6 H+ 1280 E+T | v4B1 frozen 1280 E+T |
|---|---|---|---|---|---|---|
| 1 % | 0.055 / 0.019 / 0.075 | **0.068 / 0.041 / 0.082** | −1.3 / −2.2 / −0.7 | 0.085 / 0.051 / 0.105 | 0.097 / 0.038 / 0.134 | 0.060 / 0.027 / 0.079 |
| 2 % | 0.087 / 0.035 / 0.117 | **0.104 / 0.061 / 0.125** | −1.7 / −2.6 / −0.8 | 0.115 / 0.073 / 0.139 | 0.133 / 0.064 / 0.177 | 0.099 / 0.049 / 0.127 |
| 5 % | 0.165 / 0.096 / 0.207 | **0.193 / 0.135 / 0.225** | −2.8 / −3.9 / −1.8 | 0.210 / 0.144 / 0.249 | 0.227 / 0.144 / 0.277 | 0.175 / 0.104 / 0.219 |
| 10 % | 0.280 / 0.203 / 0.329 | **0.307 / 0.232 / 0.350** | −2.7 / −2.9 / −2.1 | 0.311 / 0.222 / 0.362 | 0.340 / 0.252 / 0.394 | 0.301 / 0.200 / 0.365 |
| 20 % | 0.460 / 0.382 / 0.513 | **0.480 / 0.400 / 0.531** | −2.0 / −1.8 / −1.8 | 0.478 / 0.392 / 0.528 | 0.494 / 0.416 / 0.543 | 0.464 / 0.341 / 0.540 |

Fixed conf 0.5: control P 0.674 / R 0.872 / neg-FP 0.588 (tiny 0.815, small 0.915, medium 0.808, large 0.128)
vs 1920: P 0.647 / R 0.885 / neg-FP 0.631 (tiny 0.845, small 0.920, medium 0.815, large 0.103) vs v4 B4:
P 0.680 / R 0.853 / neg-FP 0.567.

### What this settles
1. **Resolution (this recipe): negative.** Holding data, schedule and hardware fixed, 1920×1088 is worse than
   1280×736 at every matched negative-image FP level (−1.3 to −2.8 pts recall; −1.8 to −3.9 pts tiny recall;
   the tiny bucket, the one resolution was supposed to help, loses most). The high-res model detects more at a
   fixed score (incl. tiny) but fires on proportionally more negatives, so after re-thresholding it is behind.
   The subset COCO mAP (positives only) is identical and therefore blind to this.
2. **Corpus/steps (control vs v4 B4):** dropping AuSmoke (−10,904 images, −21 % steps) costs ≈1.7 pts at 1–5 %
   FP (0.068→0.085, 0.193→0.210) and ≈0 at 10–20 %; AuSmoke positives apparently help FP discrimination on
   European fixed cameras a little. Hardware (B200 vs L40S) is a second, smaller unknown in that gap.
3. **Why high-res may still be the right direction with a different recipe:** anchors (16–256 px), RPN/RoI
   sampling and the 3-level feature pyramid were tuned at 1280; at 1920 every object/anchor relationship shifts
   by 1.5× without the head knowing. The 1920 run's higher training loss at every epoch and its medium/large
   recall drop are the fingerprint of that mismatch. A fair follow-up is anchors ×1.5 (24–384 px) + larger RoI
   count, or training at 1920 only for the Boreal/4K sources (where there are real extra pixels) and 1280 for
   pyro-sdis (native 1280×720 — upsampling adds no information).
4. **Cost:** 2.2× the tokens buys 0.76 vs 0.32 s/it (2.4× train time) and 284 vs 104 ms batch-1 latency
   (2.7×) on the same B200 — a steep price for a negative result.

## 8. B200 accounting and artifacts (this track)
| job | run | RUNNING → end | node-h (B200) |
|---|---|---|---|
| `tianrgua-v5bdinohr-mssz` | v5bB200_lora_L_1920 (incl. 2 memprobes, infer, ~10 min hold + manual stop) | 09:16:02–12:26:35Z | 3.18 |
| `tianrgua-v5bdinoctl-57zb` | v5bB200_lora_L_1280 control (SUCCEEDED) | 12:28:56–13:43:58Z | 1.25 |
| **DINO arm total** | | | **4.43 node-h = 35.4 GPU-h** (budget was ≤ 8) |

L40S: none used by this track. Registry entries `res-b200dinohr`, `res-b200dinoctl` (released).
Code: `cluster/v5b/{Dockerfile.dino3_b200,dino_b200_entry.sh,orchestrate_b200_dino.py}`, `scripts/v5b/{build_train_eb.py,
eval_b200_dino.sh}`, `src/v3/train_det.py` (+ peak-memory field in the step log/metrics — additive). Data:
`data/v5b/coco/instances_train_eb.json` (+ `.stats.json`), `instances_val.json` (copy of v3). Reports/logs:
`reports/v5b/b200_dino/<run>/` (orchestrator summaries, entry logs, latency, memprobe), `logs/v5b/b200_dino_*.log`,
`reports/v5b/eval_v5bB200_lora_L_{1920,1280}_sdis_val.json(.raw.parquet)`, `reports/v5b/matched_fp_b200_dino_sdis_val.json`.
