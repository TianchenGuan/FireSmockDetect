# Iteration 4 — `dino3` serving backend for LoRA-adapted DINOv3 checkpoints

_2026-09-14, devbox CPU only (192 vCPU, no GPU). Validation images only; no test/lockbox
data touched. The production service on :8000 (pid 3411031, yolo/expB) was never
touched; all integration runs used :8011 and were stopped afterwards._

## TL;DR

- `dino3` was a **registered name with a dangling import** (`src/serve/backends.py`
  imported `src.serve.backend_dino3`, which did not exist -> `ImportError` at startup).
  It is now a working backend: `src/serve/backend_dino3.py::Dino3Backend`.
- Loads the iteration-4 LoRA checkpoints **without any "unexpected keys"** by applying the
  same LoRA injection as training before `load_state_dict` (mirrors
  `cluster/v3/dino_infer.py --lora R`), with **rank / alpha / geometry / backbone snapshot
  auto-detected from the checkpoint** — an operator only sets `SMOKE_WEIGHTS_PATH`.
- **Parity with the cluster predictions (L40S fp32) is numerically exact for practical
  purposes**: B4 (ViT-L r16) 10 sdis_val images, 19/19 boxes matched, 0 unmatched,
  max |score delta| **2.2e-4**, min IoU **0.9995**; B6 (ViT-H+ r16) 5 images, 9/9 matched,
  max |score delta| 3.0e-4, min IoU 0.9996. Same 1280x736 letterbox geometry — no difference.
- FastAPI integration on :8011 (env vars only): `/healthz` -> `backend=dino3` + LoRA info,
  3 real images through `/detect` with the full JSON contract (provisional / watch / none).
- CPU batch-1 latency (B4, incl. preprocessing, excl. JPEG decode): **2.17 s median**
  (torch default 96 threads), **5.62 s at 8 threads** — identical to the v3 frozen ViT-L
  (5,670 ms): unmerged LoRA r16 adds no measurable serving cost. Not viable for CPU
  real-time; GPU numbers come from the cluster (v3 frozen-L: 156 ms on L40S).
- Thresholds MUST be val-selected for this model (scores saturate near 1.0): at YOLO's
  0.30/0.15 B4 would flag **61% / 66% of sdis_val negative images**. Used here: 0.93 / 0.82
  (v3 matching rule vs the deployed expB@1280). Details in §6.

## 1. What changed

| file | change |
|---|---|
| `src/serve/backend_dino3.py` | **new** — `Dino3Backend` (contract: `detect(img_bgr) -> (list[Detection], timing)`, `filter_by_conf`, `class_names={0:"smoke"}`, plus `info()`); `inspect_checkpoint`, `resolve_backbone_dir`, `merge_lora_` helpers |
| `src/serve/config.py` | **new optional fields only**, all defaulting to "auto / off" so yolo/rfdetr behavior is unchanged: `dino_backbone_dir=""`, `lora_r=None`, `lora_alpha=None`, `lora_merge=False`, `det_width=0`, `det_height=0`. No existing default modified (verified: `Settings()` still gives backend=yolo, weights=models/expB_best.pt, 0.05/0.30/0.15, imgsz 1280) |
| `src/serve/app.py` | `_load()` forwards the new settings as kwargs (`yolo` branch ignores them, other backends accept `**kw`); `/healthz` adds `backend_info` **only** when the backend exposes `info()` — yolo's healthz payload is byte-identical (TestClient-verified) |
| `scripts/v4/verify_dino3_backend.py` | **new** — parity vs cluster preds jsonl + CPU latency + LoRA-merge equivalence check |
| `scripts/v4/integration_dino3_service.sh` | **new** — starts the app on a spare port from env vars, POSTs images, asserts the JSON contract, stops the server; hard-refuses port 8000 |
| `src/serve/backends.py`, `src/v3/*`, `cluster/*` | untouched |

### How the LoRA checkpoint is loaded (mirrors `dino_infer.py` exactly)

```python
bb = ViTBackbone(backbone_dir, frozen=True, lora_r=R, lora_alpha=A)   # inject_lora on attention q/k/v/o
model = build_detector(bb, image_size=(W, H), box_score_thresh=conf_floor)
load_trainable_state_dict(model, ck["model"])   # strict: only frozen backbone keys may be missing;
                                                # ANY unexpected key (e.g. lora_A/B into a plain
                                                # backbone) is a hard RuntimeError, never swallowed
model.to(device).eval()
```

Auto-detection (each overridable; a mismatching override is a hard error):

| what | source in the checkpoint | override |
|---|---|---|
| LoRA rank `R` | shape of the `*.lora_A` tensors (`[R, d]`); 0 keys -> plain (frozen / full-unfreeze ckpts load unchanged) | `SMOKE_LORA_R` (must equal detected) |
| LoRA alpha `A` | `ck["args"]["lora_alpha"]` (None -> `2*R`, the training default) | `SMOKE_LORA_ALPHA` |
| letterbox `W x H` | `ck["args"]["width"/"height"]` (1280x736 for every v3/v4 run) | `SMOKE_DET_WIDTH/HEIGHT` |
| backbone snapshot | `basename(ck["args"]["backbone_dir"])` under `models/v3/hf/`, else by the neck input dim (`backbone.neck.stages.2.0.weight`: 1024 -> dinov3-vitl16, 1280 -> dinov3-vith16plus, 4096 -> dinov3-vit7b16) | `SMOKE_DINO_BACKBONE_DIR` |

Gated weights are **never downloaded**; a missing snapshot raises `FileNotFoundError`
naming the directories tried. `SMOKE_LORA_MERGE=1` folds `W += (A/R) * B @ A` into the base
linears after loading (deployment form; mirrors `dino_infer.py --lora-merge`; the merged
`nn.Linear` is created on the base weight's device so it also works after `.to("cuda")`).

## 2. How to run

```bash
cd /local/home/tianrgua/cvproject && source .venv/bin/activate

# B4: LoRA r16 on dinov3-vitl16 (best.pt = step 15026, COCO mAP 0.4086 on the 1500-img sdis-val subset)
SMOKE_BACKEND=dino3 \
SMOKE_WEIGHTS_PATH=models/v3/runs/v4B4_lora_L/seg05/best.pt \
SMOKE_MODEL_VERSION=v4B4_lora_L@seg05-step15026 \
SMOKE_SINGLE_FRAME_ALERT_CONF=0.93 SMOKE_SEQ_FRAME_CONF=0.82 \
uvicorn src.serve.app:app --host 127.0.0.1 --port 8011

# B6: LoRA r16 on dinov3-vith16plus (best.pt = step 11714, mAP 0.4166)
SMOKE_BACKEND=dino3 \
SMOKE_WEIGHTS_PATH=models/v3/runs/v4B6_lora_Hplus/seg10/best.pt \
SMOKE_MODEL_VERSION=v4B6_lora_Hplus@seg10-step11714 \
SMOKE_SINGLE_FRAME_ALERT_CONF=0.95 SMOKE_SEQ_FRAME_CONF=0.88 \
uvicorn src.serve.app:app --host 127.0.0.1 --port 8011
```

Optional: `SMOKE_DEVICE=cuda:0`, `SMOKE_LORA_MERGE=1`, `SMOKE_DINO_BACKBONE_DIR=<hf snapshot>`,
`SMOKE_LORA_R=16 SMOKE_LORA_ALPHA=32` (explicit; must match), `SMOKE_CONF_FLOOR` (default 0.05).
`SMOKE_IMGSZ` is ignored by this backend in favor of the checkpoint geometry (a note is printed).

Re-verify:
```bash
python scripts/v4/verify_dino3_backend.py --n 8 --n-empty 2 --latency-n 10 --merge-check \
    --out reports/v4/dino3_backend_parity_B4.json            # parity + latency (B4 defaults)
python scripts/v4/verify_dino3_backend.py --weights models/v3/runs/v4B6_lora_Hplus/seg10/best.pt \
    --preds models/v4/evals/v4B6_lora_Hplus/sdis_val_preds.jsonl --n 4 --n-empty 1
bash scripts/v4/integration_dino3_service.sh [img ...]      # PORT=8011 by default
```

## 3. Structural load test (both LoRA checkpoints)

| ckpt | backbone (auto) | LoRA modules | LoRA params | ckpt tensors | unexpected keys | load |
|---|---|---|---|---|---|---|
| `v4B4_lora_L/seg05/best.pt` | `models/v3/hf/dinov3-vitl16` | 96 (24 layers x q/k/v/o) | **3,145,728** | 238 | 0 | 1.9 s |
| `v4B6_lora_Hplus/seg10/best.pt` | `models/v3/hf/dinov3-vith16plus` | 128 (32 x 4) | **5,242,880** | 302 | 0 | 2.2 s |

LoRA param counts equal the exact values in `reports/v4/dino_adapt_results.md` and
`cluster/v4/DINO_ADAPT_NOTES.md` (r16: layers x 4 x 16 x 2d). alpha resolved to 32.0 (2r) as
trained. Passing `lora_r=8` raises
`ValueError: lora_r=8 requested but checkpoint has rank 16 (...)`. Both snapshots' real
`model.safetensors` are on this host (1.2 GB / 3.4 GB), so nothing was downloaded.

## 4. Parity with the cluster predictions

Reference: `models/v4/evals/<run>/sdis_val_preds.jsonl` produced by `dino_infer.py` on
8x L40S (fp32, `--lora 16`, conf floor 0.02, 1280x736). Backend run here on CPU fp32 with the
same conf floor; boxes divided by the letterbox scale and normalized by the original W/H
exactly as `dino_infer.py`; greedy IoU>=0.5 matching in score order. Images decoded with PIL
on both sides (the FastAPI app also decodes with PIL). Deterministic sample: seed 0,
8 images with >=1 cluster prediction + 2 with none.

**Geometry: identical.** Letterbox `r = min(1280/iw, 736/ih)`, bilinear resize, top-left paste
on a 114-gray 1280x736 canvas, ImageNet mean/std; Faster R-CNN transform is a no-op at fixed
size (`build_detector`). No geometry difference to report.

### B4 (`v4B4_lora_L`, 10 images) — `reports/v4/dino3_backend_parity_B4.json`

| image (sdis_val) | ours / cluster boxes | max abs score delta | min IoU | top scores ours vs cluster |
|---|---|---|---|---|
| force-06_cabanelle-244_2024-02-16T10-45-33 | 1 / 1 | 1e-06 | 0.9999 | 0.9712 vs 0.9712 |
| force-06_cabanelle-46_2024-02-23T09-50-10 | 3 / 3 | 7e-05 | 0.9997 | 0.9129, 0.5254, 0.1009 vs same |
| sdis-07_marguerite-29_2024-02-05T13-20-54 | 3 / 3 | 4.1e-05 | 0.9998 | 0.9907, 0.9174, 0.0313 vs ..., 0.0312 |
| force-06_cabanelle-244_2024-09-12T13-57-28 | 3 / 3 | 2.0e-04 | 0.9995 | 0.9664, 0.3217, 0.1289 vs 0.9664, 0.3219, 0.1287 |
| sdis-07_marguerite-282_2024-02-08T10-41-12 | 3 / 3 | 6.1e-05 | 0.9995 | 0.9815, 0.8415, 0.0265 vs ..., 0.0266 |
| force-06_cabanelle-125_2024-04-10T08-39-07 | 2 / 2 | 1.8e-05 | 0.9999 | 0.972, 0.0239 vs same |
| force-06_cabanelle-125_2024-06-14T07-25-12 | 3 / 3 | 5.9e-05 | 0.9997 | 0.8876, 0.2621, 0.093 vs same |
| sdis-07_marguerite-339_2024-02-07T15-56-12 | 1 / 1 | 2.23e-04 | 0.9998 | 0.3983 vs 0.3981 |
| val/force-06_cabanelle-327_2024-01-01T16-51-09 (neg) | 0 / 0 | - | - | none vs none |
| sdis-07_marguerite-29_2024-01-27T11-21-35 (neg) | 0 / 0 | - | - | none vs none |

**Summary B4:** 19/19 pairs matched, 0 unmatched on either side, 10/10 identical counts;
**max |score delta| 2.23e-4** (median 4.3e-5); **min IoU 0.9995** (mean 0.99979); max box
delta 4.7e-5 normalized (~0.06 px at 1280). Residuals are GPU (TF32 convs / fused SDPA) vs
CPU fp32 kernel differences — far below the 0.001 grid of the threshold sweeps.

**Summary B6** (`v4B6_lora_Hplus`, 4 + 1 images, `reports/v4/dino3_backend_parity_B6.json`):
9/9 matched, 0 unmatched, 5/5 identical counts; max |score delta| 2.95e-4 (median 2.2e-5);
min IoU 0.9996 (mean 0.99983); max box delta 7e-6.

**LoRA merge equivalence** (B4, `lora_merge=True`, 96 linears folded): vs unmerged on 2
images, max |score delta| 3.3e-6, max box delta 6.1e-5 px, identical counts.

## 5. Integration test (FastAPI on :8011, env vars only)

`bash scripts/v4/integration_dino3_service.sh <hi> <mid> <neg>` — full output in
`logs/v4/integration_dino3/{run.log,server.log,healthz.json,detect_1..3.json}`.
Server ready 4 s after launch (safetensors are mmap'd).

`GET /healthz` ->
```json
{"status":"ok","backend":"dino3","model_version":"v4B4_lora_L@seg05-step15026","device":"cpu",
 "imgsz":1280,"tile_size":0,"uptime_s":2.1,
 "backend_info":{"backbone_dir":".../models/v3/hf/dinov3-vitl16","letterbox":[1280,736],
   "lora_r":16,"lora_alpha":32.0,"lora_modules":96,"lora_params":3145728,"lora_merged_linears":0,
   "ckpt_step":15026,"ckpt_best_map":0.40856,"ckpt_tensors":238,"conf_floor":0.05,"device":"cpu"}}
```

`POST /detect` (device `itest-dino3`, increasing `capture_ts`, thresholds 0.93 / 0.82):

| # | image | dets | max_smoke_score (cluster top) | alert.state | latency_ms (server) | wall incl. upload |
|---|---|---|---|---|---|---|
| 1 | force-06_cabanelle-327_2024-02-27T11-18-57 | 2 | 0.9984 (0.998441) | **provisional** ("single frame score 0.998 >= 0.93") | 2095.6 | 2131.5 |
| 2 | force-06_cabanelle-125_2024-02-27T13-58-25 | 4 | 0.8735 (0.873608) | **watch** (below 0.93, >= 0.82; `history_reset: scene_change` — different camera, expected) | 2017.4 | 2039.6 |
| 3 | force-06_cabanelle-125_2024-06-14T08-40-18 (neg) | 0 | 0.0 (none) | **none** | 2061.7 | 2082.9 |

Contract asserted on every response: `detections[]` (`bbox_xyxy`, `score`, `cls_id`,
`cls_name="smoke"`, `source="full"`), `n_smoke`, `max_smoke_score`, `alert{state, reason,
window_hits, window_size, history_reset, cooldown_active, note}`, `model_version`,
`image_size`, `device_id`, `capture_ts`, `client_metadata_echo`, `score_semantics`, timing
(`latency_ms`, `preprocess_ms`, `infer_ms`, `postprocess_ms`, `full_pass_ms`, `tile_pass_ms`,
`n_tiles`). Server stopped by the script (uvicorn "Application shutdown complete"), port 8011
released; :8000 (pid 3411031) still listening before and after.

Regression: with no `SMOKE_*` set, `Settings()` defaults are unchanged, the app loads
`SmokeDetector` (yolo/expB), `/healthz` has no `backend_info` key and `/detect` works
(FastAPI TestClient run in this session; not persisted as a file).

## 6. CPU latency (batch-1, includes preprocessing, excludes JPEG decode)

B4 unmerged, 1280x736, torch 2.8.0+cpu, sdis_val images, 2 warm-ups then n=10:

| threads | median | mean | p90 | min | max | preprocess | infer |
|---|---|---|---|---|---|---|---|
| 96 (torch default on this 192-vCPU host) | **2168.8 ms** | 2175.9 | 2215.3 | 2131.4 | 2238.3 | 3.8 | 2165.0 |
| 8 (protocol of the v3 table) | **5616.6 ms** | 5633.6 | 5714.1 | 5554.7 | 5730.5 | 4.6 | 5612.1 |

- v3 frozen ViT-L (B1) on the same devbox at 8 threads: 5,670 ms -> LoRA r16 (unmerged, 96
  rank-16 side branches) costs nothing measurable; merged form measured 2050-2198 ms at
  default threads (same as unmerged within noise).
- B6 (ViT-H+): 4108.9 ms median at default threads (n=5).
- Service-side `latency_ms` in the integration run: 2017-2096 ms; end-to-end wall incl.
  upload/decode/phash 2040-2132 ms.
- GPU: not measurable on this host. The L40S number for v4B4 comes from the cluster latency
  job, which was still PENDING when this was written (`logs/v4/latency_v4B4_lora_L.log`,
  job tianrgua-dev4b4lorallat-qtnq); v3 frozen-L reference: 156.4 ms b1 on L40S.
  As in the v3 verdict, ~2 s/frame on CPU is an offline/batch or GPU-serving configuration,
  not a drop-in replacement for the 324 ms YOLO CPU path.

## 7. Thresholds and limitations

**Score saturation -> thresholds must be val-selected for THIS model.** Adapted backbones push
Faster R-CNN softmax scores toward 1.0 (`reports/v4/dino_adapt_results.md`). On sdis_val
(1,158 negative images):

| operating rule | B4 single / seq | B6 single / seq |
|---|---|---|
| YOLO service defaults 0.30 / 0.15 (**do not use**) | neg-image FP **61.4% / 66.2%** | 65.1% / 70.3% |
| v3 rule — match the deployed expB@1280's sdis_val neg-FP at 0.30 / 0.15 (34.0% / 45.3%) — **used here** | **0.93 / 0.82** (FP 33.0% / 44.7%, R 0.654 / 0.770) | **0.95 / 0.88** (FP 32.3% / 44.1%) |
| low-FP targets 1% / 2% / 5% / 10% / 20% | 0.9971 / 0.9963 / 0.9934 / 0.9881 / 0.973 | 0.997 / 0.9962 / 0.9939 / 0.9901 / 0.9801 |

(from `reports/v4/matched_fp_dino_sdis_val.json`; grid 0.01 below 0.99, 0.001 above.) Pick the
FP target for the deployment and read the conf off the model's own curve — never transfer a
YOLO threshold. Whatever pair is chosen goes in `SMOKE_SINGLE_FRAME_ALERT_CONF` /
`SMOKE_SEQ_FRAME_CONF`; `conf_floor` 0.05 is fine (the >=0.05 subset of a 0.02-floor run is
identical: the score threshold is applied before NMS and the 100-box cap is never reached).

Other limitations:
- Thresholds are sdis_val-selected and mildly optimistic (best.pt selected on a 1,500-image
  sdis-val subset); saturated models transfer a single threshold across domains worse than
  frozen B1 (forest_val rows in `dino_adapt_results.md`); per-domain calibration needs
  negatives that the forest package lacks. No FIgLib/lockbox evaluation was done here.
- Boxes are not clipped to the original image: a box reaching into the 16-px bottom pad
  (1280x720 -> 736) maps to y2 slightly > 720. Kept deliberately so service output equals the
  cluster jsonl convention; clip client-side if needed.
- Single-frame, single-class, no tiling (all detections `source="full"`), no batching.
- CUDA path (`SMOKE_DEVICE=cuda:0`) is written (device sync, merge on device) but untested on
  this host. Note for the DINO agent: `dino_infer.py --lora-merge` creates the merged
  `nn.Linear` on CPU after `model.to(dev)`, so it would fail on a GPU run; the backend's
  `merge_lora_` creates it on the base weight's device. Not changed here.
- transformers 5.17 prints a "Loading weights" progress bar to stderr at startup (cosmetic).
- The backend relies on the transformers-5.17 DINOv3 module layout
  (`hf.model.layer.N.attention.{q,k,v,o}_proj`) — the same one training used; a layout change
  would fail loudly in `inject_lora` / `load_trainable_state_dict`, not silently.

## 8. Artifacts

- code: `src/serve/backend_dino3.py`, `src/serve/config.py` (new fields), `src/serve/app.py`,
  `scripts/v4/verify_dino3_backend.py`, `scripts/v4/integration_dino3_service.sh`
- parity / latency: `reports/v4/dino3_backend_parity_B4.json`,
  `reports/v4/dino3_backend_parity_B6.json`, `reports/v4/dino3_backend_latency_B4_8thr.json`;
  logs `logs/v4/dino3_parity_B4.log`, `logs/v4/dino3_parity_B6.log`, `logs/v4/dino3_latency_B4_8thr.log`
- integration: `logs/v4/integration_dino3/{run.log,server.log,healthz.json,detect_1.json,detect_2.json,detect_3.json}`
