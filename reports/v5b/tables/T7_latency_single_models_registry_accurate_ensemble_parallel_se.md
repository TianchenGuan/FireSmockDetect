_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

### T7. Latency: single models (registry), ACCURATE ensemble (parallel / serial), demo replays on CPU, segmentation on CPU

GPU numbers are batch-1 1x L40S fp32 @1280 copied from v3/v4 reports (no new GPU latency was measured in V5b); CPU numbers are devbox measurements under load. ACCURATE on CPU is 5.8 s/frame (2-model demo) to 12.3 s/frame (4-model recipe) -> ACCURATE is a GPU or offline configuration (DEMO_V5B.md section 5-6).

| system | architecture | configuration | CPU ms | L40S ms | source / note |
|---|---|---|---|---|---|
| expB_y11s | YOLO11s | single model, batch 1 @1280 | 140 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured for expB (YOLO11s); expE YOLO11m same protocol  |
| expE_y11m | YOLO11m | single model, batch 1 @1280 | 309 | 11.2 | CPU measured by build_model_registry.py (8 threads); L40S: models/v3/evals/yolo_latency/latency_gpu_yolo.json (n=50, fp |
| v4A_ctrl_s2 | YOLO11m | single model, batch 1 @1280 | 327 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; same architecture/imgsz as expE (11.2 ms) |
| v4E_et_s1 | YOLO11m | single model, batch 1 @1280 | 308 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; same architecture/imgsz as expE (11.2 ms) |
| v4C_egt_s1 | YOLO11m | single model, batch 1 @1280 | 321 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; same architecture/imgsz as expE (11.2 ms) |
| v4Dada_s1 | YOLO11m | single model, batch 1 @1280 | 302 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; same architecture/imgsz as expE (11.2 ms) |
| rfdetrL_e48 | RF-DETR-Large (DINOv2-S backbone, 1280) | single model, batch 1 @1280 | 1000 | 31.4 | CPU measured by build_model_registry.py (8 threads); L40S: models/v3/evals/rfdetrLe48lat/latency_gpu.json (n=50, fp32) |
| v4B1_frozen_L | DINOv3 ViT-L/16 frozen + SFP + Faster R-CNN | single model, batch 1 @1280 | 5693 | 156.6 | CPU measured by build_model_registry.py (8 threads); L40S: reports/v4/dino_adapt_results.md latency table (v3 B1 frozen |
| v4B4_lora_L | DINOv3 ViT-L/16 LoRA r16 a32 (q/k/v/o) + SFP  | single model, batch 1 @1280 | 5824 | 157.5 | CPU measured by build_model_registry.py (8 threads); L40S: reports/v4/dino_adapt_results.md (LoRA merged 157.5 / unmerg |
| v4B5_full_L | DINOv3 ViT-L/16 full unfreeze + SFP + Faster  | single model, batch 1 @1280 | 5006 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; plain ViT-L form == B1/B4-merged (~157 ms expe |
| v4B6_lora_Hplus | DINOv3 ViT-H+/16 LoRA r16 a32 (q/k/v/o) + SFP | single model, batch 1 @1280 | 11946 | 351.6 | CPU measured by build_model_registry.py (8 threads); L40S: reports/v4/dino_adapt_results.md (LoRA merged 351.6 / unmerg |
| v4B6_lora_Hplus_seg05_globalbest | DINOv3 ViT-H+/16 LoRA r16 a32 (q/k/v/o) + SFP | single model, batch 1 @1280 | 11779 | 351.6 | CPU measured by build_model_registry.py (8 threads); L40S: same architecture as v4B6_lora_Hplus (merged 351.6 ms) |
| C1_7b_head_cached | DINOv3 ViT-7B/16 frozen (cached bf16 feats) + | single model, batch 1 @1280 | n/a | 368.5 | CPU measured by build_model_registry.py (8 threads); L40S: reports/v3/trackBC_results.md s3 (bf16 backbone, 1x L40S) |
| qwen3.5-9b_sft_lora | Qwen3.5-9B VLM + LoRA adapter (SFT) | single model, batch 1 @1280 | n/a | 7300.0 | CPU measured by build_model_registry.py (8 threads); L40S: reports/FINAL_REPORT_V2.md (7.3 s/img, 1x L40S bf16) |
| qwen3.5-9b_sft_grpo500 | Qwen3.5-9B VLM + LoRA adapter (SFT -> GRPO 50 | single model, batch 1 @1280 | n/a | 6500.0 | CPU measured by build_model_registry.py (8 threads); L40S: reports/FINAL_REPORT_V2.md (6.5 s/img, 1x L40S bf16) |
| qwen3.5-27b_sft_qlora | Qwen3.5-27B VLM nf4 + QLoRA adapter (SFT) | single model, batch 1 @1280 | n/a | 17800.0 | CPU measured by build_model_registry.py (8 threads); L40S: reports/FINAL_REPORT_V2.md (17.8 s/img, 1x L40S) |
| v5bA_ctrl_s3 | YOLO11m | single model, batch 1 @1280 | 275 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; same architecture/imgsz as expE (11.2 ms L40S) |
| v5bA_ctrl_s1 | YOLO11m | single model, batch 1 @1280 | 296 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; same architecture/imgsz as expE (11.2 ms L40S) |
| v5bA_ctrl_s2 | YOLO11m | single model, batch 1 @1280 | 342 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; same architecture/imgsz as expE (11.2 ms L40S) |
| v5bB_s0 | YOLO11m | single model, batch 1 @1280 | 334 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; same architecture/imgsz as expE (11.2 ms L40S) |
| v5bB_s1 | YOLO11m | single model, batch 1 @1280 | 338 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; same architecture/imgsz as expE (11.2 ms L40S) |
| v5bB_s2 | YOLO11m | single model, batch 1 @1280 | 341 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; same architecture/imgsz as expE (11.2 ms L40S) |
| v5bC_s0 | YOLO11m | single model, batch 1 @1280 | 351 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; same architecture/imgsz as expE (11.2 ms L40S) |
| v5bD_s0 | YOLO11m | single model, batch 1 @1280 | 314 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; same architecture/imgsz as expE (11.2 ms L40S) |
| v5bD2_s0 | YOLO11m | single model, batch 1 @1280 | 311 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; same architecture/imgsz as expE (11.2 ms L40S) |
| v5bD3_s0 | YOLO11m | single model, batch 1 @1280 | 319 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured; same architecture/imgsz as expE (11.2 ms L40S) |
| v5bB200_lora_L_1920 | DINOv3 ViT-L/16 LoRA r16 (q/k/v/o) + SFP + Fa | single model, batch 1 @1280 | 17498 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: NOT measured on L40S; B200 batch-1 283.8 ms median (reports/ |
| v5bB200_lora_L_1280 | DINOv3 ViT-L/16 LoRA r16 (q/k/v/o) + SFP + Fa | single model, batch 1 @1280 | 5757 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: NOT measured on L40S; B200 batch-1 103.9 ms median (reports/ |
| segV5b_a_frozen | SEG: DINOv3 ViT-L/16 frozen + SFP + FPNLite s | single model, batch 1 @1280 | 5265 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured on GPU (CPU demo path only) |
| segV5b_a_lora | SEG: DINOv3 ViT-L/16 LoRA r16 (q/k/v/o) + SFP | single model, batch 1 @1280 | 5084 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured on GPU |
| segV5b_a_r50 | SEG: ResNet-50 (ImageNet, frozen) + FPNLite s | single model, batch 1 @1280 | 577 | n/a | CPU measured by build_model_registry.py (8 threads); L40S: not measured on GPU |
| ensemble_accurate_v5b | ACCURATE ensemble: calibrated WBF of yolo_A + | single model, batch 1 @1280 | n/a | 352.0 | CPU measured by build_model_registry.py (8 threads); L40S: FULL-FRAME PARALLEL = max(component) + fusion = 351.6 (B6) + |
| ACCURATE recipe A+B4+B6+B1 | calibrated WBF fusion | PARALLEL = max(component) + fusion (one GPU per component) | - | 352.0 | batch-1 1x L40S fp32 @1280 from reports (T13 / v3 latency json); parallel = max(components |
| ACCURATE recipe A+B4+B6+B1 | calibrated WBF fusion | SERIAL on one GPU + fusion | - | 677.3 | components back to back |
| fusion step only | numpy WBF + calibration | per image, devbox CPU | 0.381 (p90 0.653, max 1.403) | - | n=7335 |
| demo replay fast_sdis (FAST: expB-mix-y11s-1280@seg03) | service /detect, CPU host shared with other agents | 29 frames, historical_replay | 146 median / 294 max | - | server-reported wall; not a GPU figure |
| demo replay acc_sdis (ACCURATE: accurate:or[expV4A_ctrl_s2@seg05+v4B4_lora_L@seg05) | service /detect, CPU host shared with other agents | 29 frames, historical_replay | 5816 median / 7286 max (yolo_A 319, dino_B4 5815) | - | server-reported wall; not a GPU figure |
| demo replay fast_heinola (FAST: expB-mix-y11s-1280@seg03) | service /detect, CPU host shared with other agents | 30 frames, historical_replay | 169 median / 203 max | - | server-reported wall; not a GPU figure |
| demo replay acc_heinola (ACCURATE: accurate:or[expV4A_ctrl_s2@seg05+v4B4_lora_L@seg05) | service /detect, CPU host shared with other agents | 30 frames, historical_replay | 5764 median / 6372 max (yolo_A 339, dino_B4 5764) | - | server-reported wall; not a GPU figure |
| segV5b_a frozen/LoRA (seg_infer.py) | DINOv3-L + FPN-lite seg head | CPU, 2048x1080 frame at 1280x736, 32 threads | 2.6–2.9 s (text figure, not a JSON artifact) | - | SEGMENTATION_V5B.md section 0 |
| segV5b_a r50 baseline (seg_infer.py) | ResNet-50 + FPN-lite | CPU, same frame | 0.15 s (text figure) | - | SEGMENTATION_V5B.md section 0 |

Sources:
- `data/v5b/demo/runs/acc_heinola.json`
- `data/v5b/demo/runs/acc_sdis.json`
- `data/v5b/demo/runs/fast_heinola.json`
- `data/v5b/demo/runs/fast_sdis.json`
- `models/v5b/MODEL_REGISTRY.json`
- `reports/v5b/SEGMENTATION_V5B.md (text, section 0 'CPU inference')`
- `reports/v5b/ensemble_dev_v5b.json (latency_table, fusion_latency_ms_cpu)`
- `reports/v5b/DEMO_V5B.md`
