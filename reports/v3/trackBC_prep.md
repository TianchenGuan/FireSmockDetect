# Track B/C prep: DINOv3 detection integration (iteration 3)

Date: 2026-09-13. Status: **READY TO LAUNCH** — everything except the gated
weights themselves is built, validated, and staged; training starts with two
commands once HF access lands (see §6).

BLOCKER (unchanged): `facebook/dinov3-{vitl16,vith16plus,vit7b16}-pretrain-lvd1689m`
are `gated: manual` on HF; our token gets HTTP 403 on file access (verified
again 2026-09-13 09:12 UTC via `scripts/v3/dinov3_gate_check.sh`). No gating
bypass was attempted; user has been asked to accept the terms on the three
model pages.

## 1. Stack selection

**CHOSEN: HF-transformers backbone (frozen `DINOv3ViTModel`) + our ViTDet-style
SimpleFeaturePyramid neck (~120 lines, `src/v3/det_modeling.py`) + torchvision
0.23 Faster R-CNN head, trained by our own compact DDP loop reusing the proven
v1/v2 segment/resume/presigned-URL machinery.**

Routes evaluated:

(a) **dinov3 repo's own detection adapter — REJECTED (inference-only).**
Pinned `facebookresearch/dinov3` @ `6876159a11b4df116f30f667f8c9888617df0751`
(2026-07-15, cloned to `third_party/dinov3`). `dinov3/eval/detection/` contains
a Plain-DETR-style model (deformable-DETR tricks + global RPE-decomposed
decoder, `models/detr.py`, hidden_dim 768, 1500 one2one + 1500 one2many
queries) and `dinov3/hub/detectors.py` loads a COCO2017-trained 7B detector —
but there is **no criterion, no Hungarian matcher, no loss, and no training
loop anywhere under eval/detection** (grep for Criterion/Matcher/loss_labels:
zero hits). Training it would mean porting Plain-DETR's whole training side;
high effort, and the head is dimensioned for the 7B.

(b1) **DEIMv2 adaptation — REJECTED, primarily on license.** Pinned
`Intellindust-AI-Lab/DEIMv2` @ `1d2ca42171570c713e78fc6a766ec5104b7f4724`
(`third_party/DEIMv2`). Technically it IS the shortest maintained DINOv3
integration: `engine/backbone/dinov3_adapter.py` (`DINOv3STAs`) wraps a
vendored copy of Meta's ViT and feeds a lite "Spatial Prior Module" conv
pyramid + DEIM transformer. But:
- **License** (LICENSE.md, read in full): custom **non-commercial-only**
  license by INTELLINDUST (Shenzhen) — Commercial Use "not permitted",
  explicitly including "integrating DEIMv2 into a commercial product or
  service" and commercial deployment; commercial use requires a separate
  license (contact shenxi@intellindust.com). Their Model Weights carry the
  same restriction. Third-party materials (the vendored DINOv3 code) stay
  under Meta's DINOv3 License. GitHub's "NOASSERTION" from recon is resolved:
  it is a proprietary NC license. Anything that wins under Track B/C is a
  candidate for the product path — embedding NC-licensed architecture/training
  code there creates a relicensing dependency. (Track A's use of DEIMv2 as an
  architecture *benchmark* is defensible under their enumerated
  "benchmarking, evaluation" purposes; a different call than building on it.)
- Their DINOv3 integration only defines S/S+ configs
  (`engine/backbone/dinov3/vision_transformer.py` has `dinov3_vits16`/`
  dinov3_vits16plus` entries only); L/H+/7B would need config surgery, Meta
  .pth-format loading, and STA/hidden-dim changes anyway — DEIMv2-X trains at
  640x640, so our ~1280 regime is untested there too.
- `SpatialPriorModulev2` consumes the RGB image in parallel with the ViT, so
  the C1 cached-feature route would not cleanly separate backbone from head.
- Pins torch 2.5.1 (we ship 2.8.0+cu128).
DEIMv2's deformable attention falls back to pure-torch `grid_sample` (no CUDA
ext), so containerizing it was feasible — license, size coverage, and the 7B
route decided this, not build risk.

(b2) **detectron2 / mmdet ViTDet — REJECTED on build+integration cost.** Both
need compiled CUDA extensions built against torch 2.8/cu128; our pinned base
image has **no nvcc** (runtime image; iteration-2 lesson) and the devbox has
no GPU, so we'd move to a -devel base and cross-build. Their ViTDet configs
also assume classic abs-pos-embed ViTs with window attention, not HF RoPE
DINOv3 — a custom backbone adapter is needed regardless, which erases most of
the "maintained stack" benefit.

(b3) **Chosen route details.** transformers 5.17.0 (already the project pin
from v2) natively ships `DINOv3ViTModel`/`DINOv3ViTConfig` — the gated HF
snapshots load with `from_pretrained` the minute access lands, with RoPE
handling arbitrary input sizes (NO pos-embed interpolation needed; DINOv3 has
no learned pos-embed). torchvision 0.23.0+cu128 is **already in the pinned
base image** — the Faster R-CNN head, RPN, RoIAlign and losses are maintained
library code with zero new compiled deps. Our code is only: backbone wrapper
(CLS + 4 register tokens stripped, frozen/partial-unfreeze), the ViTDet SFP
neck (strides ~4/8/16/32/64 from the single stride-16 map), dataset/letterbox,
and the training loop (pattern already burned in twice in this project).
Head is identical across B1/B2/B3/C1 — the comparison isolates the backbone.
Known tradeoff: Faster R-CNN is not a SOTA DETR head; Track B/C measures
*backbone* deltas under a fixed head, and Track A covers modern DETR heads
(DEIMv2/RF-DETR) — the two tracks triangulate.

Window/global attention & patch grid: released DINOv3 backbones use global
attention in all blocks (windowing is a ViTDet trick for MAE ViTs — not
needed here); at our token counts (3680) SDPA/flash handles it. Inputs must
be multiples of patch 16: we letterbox to **1280x736 -> grid 80x46 = 3680
tokens** (pyro-sdis native 1280x720 scales 1.0 with a 16px bottom pad;
D-Fire's mixed sizes letterbox in). Stand-in dinov2-large is patch **14**:
**1288x728 -> grid 92x52 = 4784 tokens**.

## 2. License findings

**DINOv3 (code AND weights): one license, the "DINOv3 License"**
(`third_party/dinov3/LICENSE.md`, last updated 2025-08-19; MODEL_CARD.md and
README §License confirm code and weights are BOTH under it — the weights
license does NOT differ). It is a Meta custom license, NOT OSI open source.
Key terms:
- Non-exclusive, worldwide, royalty-free right to use, reproduce, distribute,
  and create derivatives of the "DINO Materials" (explicitly includes trained
  model weights, training/fine-tuning code, inference code, docs).
- **Commercial use is NOT restricted** (no revenue/MAU clauses, unlike Llama).
- Redistribution (incl. derivatives) only under the same Agreement, with a
  copy of the Agreement attached.
- Research publications must acknowledge use of DINO Materials.
- No reverse-engineering of the Materials; compliance with Trade
  Controls/sanctions; **prohibited: military/warfare, nuclear, espionage,
  weapons uses** (ITAR-related). Wildfire smoke detection is none of these.
- Patent-litigation termination clause + licensee indemnifies Meta; AS-IS, no
  warranty; on termination you must delete the Materials.
- Derivative works you make are yours, subject to Meta's ownership of the
  base Materials.
- Access to the HF checkpoints additionally requires the click-through
  acceptance (hence the manual gating).
FLAG for the final report: fine-tuned checkpoints (B3) and cached features are
derivatives of DINO Materials — redistributable only under the DINOv3 License.

**DEIMv2**: proprietary non-commercial (see §1(b1)). **RF-DETR** (Track A):
repo Apache-2.0 (weights license verification is with Track A). **dinov2-large
stand-in**: Apache-2.0, ungated (`gated: False` verified), so no restriction on
the plumbing validation. **torchvision** BSD-3; **transformers** Apache-2.0.

## 3. Pinned artifacts

| artifact | pin | note |
|---|---|---|
| facebookresearch/dinov3 | `6876159a11b4df116f30f667f8c9888617df0751` | third_party/dinov3, reference only (specs, norm constants) |
| DEIMv2 | `1d2ca42171570c713e78fc6a766ec5104b7f4724` | third_party/DEIMv2, evaluated + rejected for B/C |
| facebook/dinov3-vitl16-pretrain-lvd1689m | rev `ea8dc2863c51be0a264bab82070e3e8836b02d51`, 1.2 GB | GATED; rev pinned in scripts/v3/stage_dinov3.sh |
| facebook/dinov3-vith16plus-pretrain-lvd1689m | rev `c807c9eeea853df70aec4069e6f56b28ddc82acc`, 3.4 GB | GATED |
| facebook/dinov3-vit7b16-pretrain-lvd1689m | rev `b80367753773648a6793235ab9c65cdbb029506f`, 26.9 GB | GATED |
| facebook/dinov2-large (stand-in) | rev `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` | UNGATED, Apache-2.0; staged to /scratch (byte-verified, job tianrgua-wsdinov2large-r6kp) |
| container smoke-dino3:v1 | `000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-dino3:v1` digest `sha256:aaf5b9d06d6fe76339013e7687430c8767989808da7153a5b8539b2caaea5eb8` | torch 2.8.0+cu128 base + transformers 5.17.0 + pycocotools + curl + uid/offline env fixes; in-image import+forward verified |
| model hyperparams (for random-init wiring) | vitl16: d1024/L24/h16/mlp4; vith16plus: d1280/L32/h20/swiglu6; vit7b16: d4096/L40/h32/swiglu64x3; all patch16 + 4 registers + RoPE | from dinov3 repo hub/backbones.py @ pin |
| normalization | mean (0.485,0.456,0.406) / std (0.229,0.224,0.225) | dinov3/data/transforms.py; identical to dinov2-large's preprocessor_config.json — one constant set for stand-in and real runs |
| data | Track A's `data/v3/coco/instances_{train,val}.json` (train=expB_train 38,454 imgs/31,487 boxes incl. 11,257 negatives; val=sdis_val 7,335/6,603) | consumed as-is; sha256s in CONVERSION_STATS.json |

## 4. Stand-in validation (PLUMBING ONLY — none of these numbers are DINOv3 results)

CPU (devbox, `scripts/v3/validate_det_cpu.py`, all PASS):
- A: tiny random DINOv3ViT e2e — dataset->detector fwd/bwd, loss dict finite,
  frozen backbone got NO grads / >20 head+neck tensors got grads,
  trainable-only ckpt round-trip bit-identical, eval postprocess, and the
  cached-feature path produced the same loss keys via the same RPN/RoI heads.
- B: random-init `dinov3_vitl16` (303M params — matches ViT-L) @ 1280x736:
  grid 80x46, register stripping (1 CLS + 4 reg) verified by token-count
  assert, SFP levels (184,320)/(92,160)/(46,80)/(23,40)/(12,20) = strides
  4/8/16/32/64. This exercises the EXACT class (`DINOv3ViTModel`) the gated
  weights will load into.
- C: **facebook/dinov2-large real weights** @ 1288x728 (patch-14 grid 92x52):
  frozen fwd+bwd on real train images, losses finite (cls 0.672 / box 0.000 /
  obj 0.696 / rpn 0.026 at init), fmap stats sane (mean 0.03, std 1.45),
  `unfreeze_top=4` -> 50M/304M trainable (B3 mechanics).
- Trainer loop: 6-step run + resume-to-9-steps on real data (tiny spec),
  metrics.jsonl + last/best.pt + DONE marker written, resume verified.

Cluster (green g6e, 1 node, smoke-dino3:v1): `standin_d2l_overfit16` — 16
train images (12 pos/4 neg), 300 steps, eval-on-train — RESULT BELOW.

**Cluster overfit-16: PASS** (job `tianrgua-d3standind2ls3-a96c`, SUCCEEDED,
~2 min train on 8x L40S DDP, smoke-dino3:v1): loss 0.816@20 -> 0.076@100 ->
**0.027@300**; train-set COCO eval 0.061 mAP@100 -> 0.725@200 ->
**0.807 mAP / AP50 = 1.00 @300** on the same 16 images — head learns,
detections postprocess correctly, in-pod COCO eval works. Artifacts round-trip
verified: last/best.pt (268 MB), metrics.jsonl, status.json (now incl.
`scratch_free_gb`: **95,738 GB free** -> the 1.35 TB 7B cache fits trivially).
Two plumbing bugs found and fixed by this run (that is what it is for):
(1) pycocotools 2.0.10 `loadRes` requires an `info` key in the GT dict
(segs 1-2 failed in eval AFTER a clean training step; devbox pycocotools is
older -> not caught on CPU); (2) status.json corrupted by quotes in the
grep-extracted loss field (entry now strips to bare numbers; orchestrator
hardened). Both fixes live in per-job-fetched files — no image rebuild.
Segment-resume machinery is byte-identical to the battle-tested v1/v2 pattern
(/scratch run-dir first, S3-presigned last.pt fallback) and ckpt resume was
verified on devbox; a multi-segment run will exercise it end-to-end in B1.

REMINDER: all numbers above validate PLUMBING with dinov2-large; they are
never to be reported as DINOv3 results.

## 5. C1: DINOv3-7B plan (sharding -> NOT needed; precompute -> CHOSEN)

Sharding analysis: 7B params in bf16 = **~13.4 GB** -> fits ONE 44.4-GiB L40S
for inference with room for 3680-token activations (est. peak < 20 GB at
bs 2). **No tensor/pipeline sharding is needed for the frozen-backbone use**;
we run 8 independent replicas (one per GPU) sharding the image list.
(End-to-end 7B *fine-tuning* would need FSDP across the PCIe-only node —
measured 19x slowdown territory in v2 — explicitly out of scope.)

**Chosen: precompute-features route.**
- Cost math (est.): 7B fwd ~= 2*6.7e9*3680 ~= 49 TFLOP/img -> ~0.8 s/img/GPU
  -> one pass over train+val (45,789 imgs) ~= **1.3-2 node-hours, paid once**.
  End-to-end frozen training would pay that EVERY epoch (12 epochs ~= 15-20
  node-hours per hyperparameter setting). Head-only training on cached
  features is I/O-bound minutes/epoch — a full head sweep costs ~1 node-hour.
- Storage: last-layer bf16 [4096, 46, 80] = **30.1 MB/img -> 1.35 TB** total
  on /scratch (`/scratch/.../smoke/feats_v3/dino3_7b_1280x736`). The entry
  script now reports `scratch_free_gb` in every status.json; **check >= ~1600
  GB before launching**. Fallbacks (in order): fp8-e4m3 storage (0.7 TB,
  needs a small quant/dequant shim), 1024x576 features (grid 64x36, 18.9
  MB/img -> 0.85 TB), train-subset caching.
- **Augmentation implication (the real tradeoff)**: cached features freeze
  augmentation — no flip/color jitter for the C1 head, unlike B1/B2. Decision:
  accept for C1 (it is a scaling probe, not the production candidate), and
  de-confound by running a **B1-cached-noaug control** (L features are 7.5
  MB/img -> 345 GB; same precompute script with the vitl16 dir) before
  comparing C1 against B1/B2. `--flip` in precompute_feats.py can bake
  mirrored copies (2x storage) if the control shows aug matters a lot.
- Files: one bf16 .pt per image keyed by the COCO json's `flat_file_name`;
  resumable (skips existing; tmp+rename writes); `.done_rank*` markers signal
  completion through the no-logs exit-code channel.

## 6. Launch readiness — what happens when access lands

1. `bash scripts/v3/dinov3_gate_check.sh` (or `--loop 300` to watch) — tests
   all 3 models in one call; exits 0 when all accessible.
2. `bash scripts/v3/stage_dinov3.sh` — pinned-revision snapshot download ->
   S3 -> byte-verified /scratch staging (reuses v2 stage_weights_job.py; L+H+
   land in ~5 min, 7B in ~10-15 min).
3. `setsid nohup bash cluster/v3/launch/B1_vitl16.sh > logs/v3_B1.log 2>&1 &`
   (then B2; B3 after B1's first eval sanity; C1_7b_precompute.sh can run in
   parallel on a second node, C1_7b_head.sh after it).
Total human path from access to B1 training steps: **~20-30 min** (dominated
by the 7B download; B1 alone needs only the 1.2-GB vitl16).

Run matrix (all: identical data/exposure — frozen expB_train, 12 epochs, bs 32
global, lr 1e-4 head, 1280x736, seed 0; forced differences noted):
| run | backbone | trainable | forced differences |
|---|---|---|---|
| B1 | dinov3-vitl16 frozen | SFP+head (~33M) | — (anchor run) |
| B2 | dinov3-vith16plus frozen | SFP+head (~35M) | SFP input dim 1280 vs 1024 |
| B3 | vitl16, top-4 blocks unfrozen @0.1x LR | +50M backbone | bs 2/GPU (grad-ckpt on); LoRA rejected: full-rank adaptation, zero new deps (no peft in image), native ckpting via trainable-state filter; validated on stand-in |
| C1 | vit7b16 frozen, CACHED features | SFP+head (~45M; 4096-dim input) | no augmentation (see §5); B1-cached-noaug control required |

Monitoring: `models/v3/runs/<run>/seg*/status.json` (phase/step/loss/
scratch_free_gb), `metrics.jsonl` (loss curve + per-epoch val COCO mAP/AP50 on
a fixed 1,500-img val subset), `console_tail.txt` (error excerpts). Exit-code
bitmask channel identical to v2. Full-val + project-standard PR/size-bucket
eval (src/eval/eval_detection.py) happens devbox-side after training.

## 7. Open risks
- Gating timing unknown; Meta approval latency is typically minutes-hours for
  auto after acceptance, but manual review can take days. Everything else is
  DONE, so this is the only critical-path item.
- HF `DINOv3ViTModel` weight-name mapping for the *real* snapshots is
  untested by definition (403). Mitigated: transformers 5.17.0 model code is
  itself the converter target for these exact repos; wiring is validated on
  the exact class; any surprise appears in the first 2 minutes of B1 and only
  affects `from_pretrained`.
- /scratch capacity for the 1.35 TB 7B cache unverified until the first
  status.json reports `scratch_free_gb` (overfit job provides it).
- 45k small files in one FSx dir (feature cache) may be slow to enumerate;
  loader opens by exact name (no listing) — acceptable; shard into subdirs if
  ls-latency becomes a problem.
