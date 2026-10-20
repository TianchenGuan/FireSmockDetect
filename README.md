# FireSmockDetect — early wildfire-smoke detection

Drone or fixed camera uploads an RGB image → a FastAPI service returns smoke
boxes, scores and an alert state. This repo contains the code, experiment
protocols and reports of a six-iteration research project on **faint/early
smoke detection**: detectors (YOLO11 n/s/m, RF-DETR, DINOv3 + Faster R-CNN
heads), weak supervision and distillation, false-alarm calibration,
synthetic-data generation, and edge export.

一个野火早期烟雾检测课程项目：无人机/固定摄像头上传 RGB 图像，服务端返回烟雾框、
置信度与告警状态。仓库包含 6 个迭代的全部代码、冻结协议与评测报告：检测器训练、
弱监督/蒸馏、误报治理、可控合成数据（V6）、以及端侧导出。模型权重发布在
Hugging Face（见下），数据集与权重文件不随仓库分发。

## Results in one paragraph (honest summary)

Across v1–v5b the strongest *fixed-camera* model remained the simple recipe
(pyro-sdis + D-Fire) — retraining it (YOLO11m, "arm A") beat every new-data,
distillation and ensemble variant on the frozen test at matched
false-positive rates (R@10%FP 0.388 vs 0.354 deployed); the only robust gain
from new labelled data was **aerial** recall (+21 pt @5%FP on the Boreal val).
Thresholds alone cannot fix false alarms (v5: halving false-alert episodes
costs ~26–29 pt recall), and no true smoke-onset footage existed on disk, so
early-warning claims are explicitly deferred. V6 (in progress) tests whether
**controllable synthetic smoke** (procedural vs generative compositing, 50/50
with real data) improves a compact YOLO11n student, with pre-registered
metrics and an edge (ONNX/TFLite-INT8) deliverable.

## Layout

| path | content |
|---|---|
| `src/` | serving (FastAPI, FAST/ACCURATE modes, shadow replay), eval (matched-FP sweeps), training |
| `scripts/` | per-iteration tooling (v3–v6): eval, fusion, corpus builders, synthetic compositor, VLM review, edge export |
| `cluster/` | training orchestration (segmented cluster jobs, resource registry, budget ledgers) |
| `reports/` | per-iteration frozen protocols, pre-registrations, final reports, tables |
| `data/*/manifests`, `data/*.md` | dataset/corpus manifests and inventories (no images) |
| `models/*.md` | model registry (metadata only; weights on Hugging Face) |

Key reading order: `reports/v5b/FINAL_REPORT_V5B.md` →
`reports/v6/PREREGISTRATION_V6.md` → `STATUS.md` (living log).

## What is NOT in this repo

- **Datasets** (pyro-sdis, D-Fire, FIgLib/HPWREN, FireMan-UAV-RGBT, Sheffield
  ORDA, Boreal, AuSmoke, FireSpot-CNX, …): fetch from their original sources;
  several carry restrictive or pending licences (FIgLib is CC BY-NC-ND and was
  never used for training here).
- **Model weights**: published separately on Hugging Face
  (`TianchenGuan/FireSmockDetect-models`), AGPL-3.0 (Ultralytics YOLO
  derivatives). DINOv3-based checkpoints are not redistributed (Meta licence).
- **Generated job configs, credentials, state files** — see
  `SANITIZATION.md`: cloud account IDs, role names and presigned URLs were
  replaced with placeholders before publication.

## Serving quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt   # or: pip install ultralytics fastapi uvicorn pillow numpy
# place a checkpoint (see HF repo) under models/
uvicorn src.serve.app:app --port 8000
# POST an image to /detect ; demo UI at /demo
```

## Licence

Code: AGPL-3.0 (forced by Ultralytics-derived training/inference paths).
Reports and documentation: CC BY 4.0. Dataset licences remain with their
owners.
