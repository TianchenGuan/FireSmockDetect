# Iteration 5b - all report tables

_generated 2026-10-03T03:55Z by `scripts/v5b/make_report_tables_v5b.py`; every number is read from the artifact(s) listed under each table. Re-run after new eval artifacts land (reports/v5b/val, reports/v5b/test) or to refresh the compute ledger._

## Index

- [T1](#t1) Data sources, permissions and use in V5b training
- [T2](#t2) Training corpora per arm: lines, multiplicities, epochs, exposures (matched budget), masked / pseudo-label counts
- [T3a](#t3a) Model registry (V5b phase 1): checkpoints that exist, load, and their frozen VAL numbers
- [T3b](#t3b) V5b systems in training at generation time (not yet in the registry; numbers are training-time only)
- [T4a](#t4a) VAL matched-FP: pyro-sdis VAL (fixed camera; sites cabanelle + marguerite; 7,335 imgs / 1,158 negatives; 1% FP = ~12 negatives)
- [T4b](#t4b) VAL matched-FP: Boreal heinola aerial VAL (1,123 imgs / 217 negatives; 1% FP = ~2 negatives -> 1-2% rows are weakly supported)
- [T4c](#t4c) VAL matched-FP: FireMan val_s5 (burn D video_015, every 5th frame; 1,706 imgs / 231 author negatives; one video = one unit; author boxes are large region boxes)
- [T5](#t5) Model collaboration on pyro-sdis VAL: site-grouped out-of-fold (OOF) mean recall over FP 1/2/5/10/20% (selection statistic = macro over the two held-out sites)
- [T6](#t6) Thin-smoke SEGMENTATION branch (segV5b_a): heinola aerial VAL, thr 0.5, best epoch selected on VAL (selection-optimistic)
- [T7](#t7) Latency: single models (registry), ACCURATE ensemble (parallel / serial), demo replays on CPU, segmentation on CPU
- [T8a](#t8a) FINAL TEST roster: systems found under reports/v5b/test, H3 contamination status, pre-registered VAL confidences
- [T8b](#t8b) FINAL TEST H1 - H1 fireman_holdout_s10 (burn C Taivalkoski, 5 videos, 1,949 frames, 0 negatives -> RECALL ONLY): recall at the pre-registered VAL confidences
- [T8c](#t8c) FINAL TEST H2 - H2 pyro-sdis TEST (ferion + croix-augas, 4,124 imgs / 917 negatives; second opening after v4): matched-FP recall / tiny / precision
- [T8d](#t8d) FINAL TEST H3 - H3 sheffield_holdout rec_0008 (327 frames, 0 negatives -> recall only; WITHIN-INCIDENT; CONTAMINATED for arms whose corpus holds Sheffield)
- [T8e](#t8e) FINAL TEST H4 - historical continuity: Boreal ruokolahti TEST and AuSmoke forest TEST at the pre-registered confidences
- [T9a](#t9a) Compute by iteration and GPU type (scheduler-derived allocated hours; whole-node accounting)
- [T9b](#t9b) Iteration 5b compute by GPU type, track, kind and arm (scheduler allocation) with the reservation-registry cross-check
- [T9c](#t9c) Iteration 5b jobs that did not end SUCCEEDED (FAILED segments, STOPPED probes, RUNNING at dump)

## PENDING

- none

## Artifacts not found at generation time

- none

## Notes

- none

---

### T1. Data sources, permissions and use in V5b training

Counts are read from the V5b manifests (agent-built, nothing human-reviewed; `annotation_origin=author` everywhere except WSDataset masks = model-assisted and the D3 pseudo boxes = model-assisted). 'Used in training' names the arms whose train lists contain the source (data/v5b/manifests/CORPUS_*_v1.json). FireMan boxes are interpolated region boxes (presence-grade, not tight instances); Sheffield boxes are fragmented sub-boxes per plume.

| source | frames (rows) | smoke boxes | explicit negatives | units | licence (verbatim/short) | permission_status | used in V5b training | eval role |
|---|---|---|---|---|---|---|---|---|
| E corpus = pyro-sdis TRAIN sites + D-Fire smoke (frozen expB_train.txt) | 38,454 | 21,944 + 9,550 | 11,257 | 4 sdis TRAIN sites + D-Fire (no units) | pyro-sdis: Apache-2.0; D-Fire: CC0, provenance caveat in repo LICENSE (reports/FINAL_REPORT.md) | inherited (v1-v4) | YES - every arm (A,B,C,D,D2,D3), x1 | sdis VAL/TEST sites held out |
| Boreal Forest Fire (FGI) TRAIN burns evo + karkkila, Subset-A boxes | 2,062 | 2,061 | 37 (karkkila Empty-Images clip) | 2 burns / 13 flights | - CC BY 4.0 (Creative Commons Attribution 4.0 International). | confirmed | YES - arms B,C,D,D2,D3 x2 (and v4 E+T models) | heinola = aerial VAL; ruokolahti = historical TEST |
| Boreal Subset-C masks (SAM model-assisted + 27 human) + author negatives [SEG branch] | 665 train / 382 val | masks: 628 / 336 | 37 / 46 | train evo+karkkila 13 flights; val heinola 7 flights | CC BY 4.0 (same source) | confirmed | YES - segV5b_a only (not detectors) | heinola masks = VAL; ruokolahti masks (453) never staged |
| FireMan-UAV-RGBT (Oulu) RGB, Multiclass CVAT | 69,871 | 69,234 on 64,867 frames (incl. 4 on 4 conflict frames; CARD says 69,232) | 4,979 | 4 burns / 34 videos | CC BY 4.0 | confirmed | YES - burn A train capped 600/video = 8,279 lines (arms B,C,D,D2,D3) | burn D video_015 = aerial VAL (fireman_val_s5); burn C = H1 holdout (sealed); burn B = quarantine (Ruokolahti candidate) |
| FireMan thermal (no smoke supervision) | 33,052 | 0 smoke boxes | 0 | 10 videos | CC BY 4.0 | confirmed | NO (not exported) | future RGB-T ablation only |
| Sheffield UAV (ORDA, Roboflow re-deposit) | 1,336 | 5,283 | 30 (24 exported; 6 smoke-visible excluded) | 1 session / 6 recordings | CC BY 4.0 | confirmed | YES - ALL 1,327 exported frames x2 (arms B,C,D,D2,D3) incl. rec_0008 | rec_0008 = H3 (CONTAMINATED for B/C/D/D2/D3; within-incident for everyone) |
| FireSpot-CNX-r1 (NECTEC) | 3,996 | 4,058 | 1,179 | 5 sites / 29 sessions | UNSPECIFIED (no LICENSE anywhere; MNS 'CC BY 4.0' is an aggregator claim) | pending | NO (guarded converter refuses); NOTE 2,862 MNS copies were in v4 G corpora (R3) | none |
| WSDataset (Kaggle gloryvu; MNS copies + GitHub demo) | 11,753 rows (129 new files) | 12,929 (mask-derived, model-assisted; author boxes BLOCKED) | 126 new author negatives | 70 units (66 videos recovered) | MIT declared on Kaggle page (text unverified); GitHub demo unlicensed | pending | NO in v5b (MNS copies were v4 G material); original 18.4 GB archive BLOCKED (Kaggle token) | none |
| UNLABELLED smoke-positive material for D3: Boreal train-burn Subset-B clip frames (104 smoke clips), FIRESENSE smoke_pos, D-Fire fire-only | 3,796 | 0 author boxes -> 3,797 teacher pseudo boxes (model-assisted) | 0 (no teacher-silence negatives created) | 3,370 + 530 + 944 target frames (TEACHER_POOL.md D3 section) | Boreal CC BY 4.0; FIRESENSE / D-Fire inherited | inherited | YES - arm D3 only (x1, pseudo boxes) | none |
| AuSmoke (via MultiNatSmoke) T list | 10,904 | - | - | - | permission PENDING (PROTOCOL_V5 S2) | pending | NO new use in v5b (v4 B1/B4/B6/E_ET models already trained on it - disclosed) | AuSmoke forest TEST = H4 historical |

Sources:
- `data/v4/manifests/boreal_CARD.md`
- `data/v5b/manifests/CORPUS_B_v1.json`
- `data/v5b/manifests/CORPUS_D3_v1.json`
- `data/v5b/manifests/fireman_CARD.md`
- `data/v5b/manifests/fireman_manifest.parquet`
- `data/v5b/manifests/firespot_CARD.md`
- `data/v5b/manifests/firespot_manifest.parquet`
- `data/v5b/manifests/seg_v5b_manifest.json`
- `data/v5b/manifests/sheffield_uav_CARD.md`
- `data/v5b/manifests/sheffield_uav_manifest.parquet`
- `data/v5b/manifests/wsdataset_CARD.md`
- `data/v5b/manifests/wsdataset_manifest.parquet`
- `models/v5b/MODEL_REGISTRY.json (train_data_recipe of expB_y11s)`
- `reports/FINAL_REPORT.md (iteration-1 data section: licence lines)`


---

### T2. Training corpora per arm: lines, multiplicities, epochs, exposures (matched budget), masked / pseudo-label counts

Exposures = lines x epochs, matched to arm A (38,454 x 60 = 2,307,240) within -0.3%/-0.65%. C = B + conservative ignore regions (teacher-agreed, GT-free rectangles filled with image mean); D = B + filtered teacher pseudo boxes on already-labelled E/Boreal images; D2 = D rules + looser all-source pseudo-labels (near-replicate of D: only 408 boxes, 0 on FireMan); D3 = B + teacher pseudo boxes on genuinely UNLABELLED smoke-positive material (the real teacher-to-student test). C/D are small by construction: on already-labelled data the teachers agree with the authors.

| arm | train lines | components raw x multiplicity = effective | epochs | exposures | delta vs A | masked (C) | pseudo boxes (D/D2/D3) | label origin |
|---|---|---|---|---|---|---|---|---|
| A (control; expV5bA_ctrl_s3, seed-0 replicate) | 38,454 | E x1 | 60 | 2,307,240 | 0.000% | - | - | author only |
| B | 53,511 | E 38,454x1=38,454, boreal 2,062x2=4,124, fireman 8,279x1=8,279, sheffield 1,327x2=2,654 | 43 | 2,300,973 | -0.272% | - | - | author 50,085, author_verified_empty 37 |
| C | 53,511 | E 38,454x1=38,454, boreal 2,062x2=4,124, fireman 8,279x1=8,279, sheffield 1,327x2=2,654 | 43 | 2,300,973 | -0.272% | 300 rects on 148 imgs ({'sdis': 145, 'dfire': 2, 'boreal': 1}) | - | author 49,937, author_labels+masked_pixels 148, author_verified_empty 37 |
| D | 53,508 | E 38,454x1=38,454, boreal 2,062x2=4,121, fireman 8,279x1=8,279, sheffield 1,327x2=2,654 | 43 | 2,300,844 | -0.277% | - | 495 boxes on 495 imgs; by source {'sdis': 449, 'dfire': 43, 'boreal': 3} | author 49,590, author+model-assisted 495, author_verified_empty 37 |
| D2 | 53,508 | E 38,454x1=38,454, boreal 2,062x2=4,123, fireman 8,279x1=8,279, sheffield 1,327x2=2,652 | 43 | 2,300,844 | -0.277% | - | 408 boxes on 408 imgs; by source {'sdis': 394, 'dfire': 11, 'boreal': 1, 'sheffield': 2} | author 49,677, author+model-assisted 408, author_verified_empty 37 |
| D3 | 57,307 | E 38,454x1=38,454, boreal 2,062x2=4,124, fireman 8,279x1=8,279, sheffield 1,327x2=2,654, d3 3,796x1=3,796 | 40 | 2,292,280 | -0.648% | - | 3796 unlabelled imgs kept with pseudo boxes (rule: D2 rules without GT filter (no author boxes exist): (a) FUSED calibrated WBF >= 0.9918 (ac...) | author 50,085, author_verified_empty 37, model-assisted 3,796 |

**new-source share and exclusions**

| arm | FireMan explicit negs | Sheffield explicit negs | new-source effective lines | new-source / E | excluded (from the manifest) |
|---|---|---|---|---|---|
| B | 1,572 | 24 | 15,057 | 39.2% | FireSpot (licence pending); WSDataset (blocked); AuSmoke/MNS (permission pending; 10,904 T lines); FIgLib (eval-only); Boreal heinola (aerial VAL) / ruokolahti ... |
| C | 1,572 | 24 | 15,057 | 39.2% | FireSpot (licence pending); WSDataset (blocked); AuSmoke/MNS (permission pending; 10,904 T lines); FIgLib (eval-only); Boreal heinola (aerial VAL) / ruokolahti ... |
| D | 1,572 | 24 | 15,054 | 39.1% | FireSpot (licence pending); WSDataset (blocked); AuSmoke/MNS (permission pending; 10,904 T lines); FIgLib (eval-only); Boreal heinola (aerial VAL) / ruokolahti ... |
| D2 | 1,572 | 24 | 15,054 | 39.1% | FireSpot (licence pending); WSDataset (blocked); AuSmoke/MNS (permission pending; 10,904 T lines); FIgLib (eval-only); Boreal heinola (aerial VAL) / ruokolahti ... |
| D3 | 1,572 | 24 | 15,057 | 39.2% | FireSpot (licence pending); WSDataset (blocked); AuSmoke/MNS (permission pending; 10,904 T lines); FIgLib (eval-only); Boreal heinola (aerial VAL) / ruokolahti ... |

Sources:
- `data/v5b/manifests/CORPUS_B_v1.json`
- `data/v5b/manifests/CORPUS_C_v1.json`
- `data/v5b/manifests/CORPUS_D_v1.json`
- `data/v5b/manifests/CORPUS_D2_v1.json`
- `data/v5b/manifests/CORPUS_D3_v1.json`


---

### T3a. Model registry (V5b phase 1): checkpoints that exist, load, and their frozen VAL numbers

Generated 2026-10-02T23:10:50Z by scripts/v5b/build_model_registry.py on the CPU devbox. Eligibility rules: fast = loads on CPU + CPU batch-1 <= 1000 ms; accurate = loads + L40S batch-1 <= 400 ms + licence permits internal serving; teacher = loads + licence permits derivative use + not FIgLib-trained + (sdis-VAL R@5% >= 0.202 or FIgLib-39 coverage > 30/39). Flags R1 (arm-A seeds all 0) and R2 (B6 shipped ckpt != run best) are in reports/v5b/RECONCILIATION_V5B.md.

| id | family | params trainable / total | train recipe | sdis-VAL matched-FP [artifact] | FIgLib-39 | exists | loads | backend | latency CPU / L40S ms | licence | fast/acc/teacher | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| expB_y11s | YOLO11s | 9428179 / 9428179 | corpus E = frozen expB_train list: pyro-sdis TRAIN sites + D-Fire smok... | R@5% 0.203 (tiny 0.104); R@10% 0.329 (tiny 0.203) [reports/v3/matched_fp_val.json] | 32/39 (med 475.0 s) | yes | yes | yolo | 140 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | Y/Y/n |  |
| expE_y11m | YOLO11m | 20053779 / 20053779 | corpus E = frozen expB_train list: pyro-sdis TRAIN sites + D-Fire smok... | R@5% 0.190 (tiny 0.097); R@10% 0.309 (tiny 0.201) [reports/v3/matched_fp_val.json] | 30/39 (med 240.0 s) | yes | yes | yolo | 309 / 11.2 | AGPL-3.0 (ultralytics YOLO11 code + COCO | Y/Y/n |  |
| v4A_ctrl_s2 | YOLO11m | 20053779 / 20053779 | corpus E = frozen expB_train list: pyro-sdis TRAIN sites + D-Fire smok... | R@5% 0.201 (tiny 0.111); R@10% 0.304 (tiny 0.196) [reports/v4/seed_summary.json] | 30/39 (med 240.0 s) | yes | yes | yolo | 327 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | Y/Y/n | SEED PROVENANCE: args.yaml records seed=0 for ALL THREE arm-A runs (expV4A_ctrl_ |
| v4E_et_s1 | YOLO11m | 20053779 / 20053779 | corpus E+T (data/v4/manifests/CORPUS_E_ET_v1.json): E 38,454 + T 12,96... | R@5% 0.191 (tiny 0.089); R@10% 0.291 (tiny 0.172) [reports/v4/seed_summary.json] | 32/39 (med 240.0 s) | yes | yes | yolo | 308 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | Y/Y/n |  |
| v4C_egt_s1 | YOLO11m | 20053779 / 20053779 | corpus E+G+T (CORPUS_C_EGT_v1.json): E + G (WSDataset/FireSpot/FESB/fi... | R@5% 0.157 (tiny 0.059); R@10% 0.262 (tiny 0.151) [reports/v4/seed_summary.json] | 32/39 (med 60.0 s) | yes | yes | yolo | 321 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | Y/Y/n |  |
| v4Dada_s1 | YOLO11m | 20053779 / 20053779 | curriculum: stage-1 G-only pretrain 63 ep (expV4Dpre_g_s1) -> stage-2 ... | R@5% 0.195 (tiny 0.084); R@10% 0.318 (tiny 0.180) [reports/v4/seed_summary.json] | 32/39 (med 240.0 s) | yes | yes | yolo | 302 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | Y/Y/n |  |
| rfdetrL_e48 | RF-DETR-Large (DINOv2-S backbone, 1280) | 35327172 / 35327172 | corpus E = frozen expB_train list: pyro-sdis TRAIN sites + D-Fire smok... | R@5% 0.203 (tiny 0.102); R@10% 0.315 (tiny 0.194) [reports/v3/matched_fp_val.json] | 33/39 (med 300.0 s) | yes | yes | rfdetr | 1000 / 31.4 | Apache-2.0 (RF-DETR-Large code + weights | Y/Y/Y |  |
| v4B1_frozen_L | DINOv3 ViT-L/16 frozen + SFP + Faster R-CNN | 22305049 / 325434649 | corpus E+T (data/v4/manifests/CORPUS_E_ET_v1.json): E 38,454 + T 12,96... | R@5% 0.175 (tiny 0.104); R@10% 0.301 (tiny 0.200) [reports/v4/matched_fp_dino_sdis_val.json] | - | yes | yes | dino3 | 5693 / 156.6 | DINOv3 License (custom Meta, 2025-08-19; | n/Y/n |  |
| v4B4_lora_L | DINOv3 ViT-L/16 LoRA r16 a32 (q/k/v/o) + SFP + Faster R-CNN | 25450777 / 328580377 | corpus E+T (data/v4/manifests/CORPUS_E_ET_v1.json): E 38,454 + T 12,96... | R@5% 0.210 (tiny 0.144); R@10% 0.311 (tiny 0.222) [reports/v4/matched_fp_dino_sdis_val.json] | 26/39 (med 481.0 s) | yes | yes | dino3 | 5824 / 157.5 | DINOv3 License (custom Meta, 2025-08-19; | n/Y/Y |  |
| v4B5_full_L | DINOv3 ViT-L/16 full unfreeze + SFP + Faster R-CNN | 325433625 / 325434649 | corpus E+T (data/v4/manifests/CORPUS_E_ET_v1.json): E 38,454 + T 12,96... | R@5% 0.209 (tiny 0.131); R@10% 0.316 (tiny 0.231) [reports/v4/matched_fp_dino_sdis_val.json] | - | yes | yes | dino3 | 5006 / n/a | DINOv3 License (custom Meta, 2025-08-19; | n/Y/Y | confounded vs B1/B4/B6 (bs 2/GPU, 2x steps, end-only checkpoint); best.pt exists |
| v4B6_lora_Hplus | DINOv3 ViT-H+/16 LoRA r16 a32 (q/k/v/o) + SFP + Faster R-CNN | 30382937 / 870975577 | corpus E+T (data/v4/manifests/CORPUS_E_ET_v1.json): E 38,454 + T 12,96... | R@5% 0.227 (tiny 0.144); R@10% 0.340 (tiny 0.252) [reports/v4/matched_fp_dino_sdis_val.json] | 24/39 (med 539.0 s) | yes | yes | dino3 | 11946 / 351.6 | DINOv3 License (custom Meta, 2025-08-19; | n/Y/Y | CHECKPOINT != RUN BEST: shipped best.pt is step 11714 (best_map 0.4166) but the  |
| v4B6_lora_Hplus_seg05_globalbest | DINOv3 ViT-H+/16 LoRA r16 a32 (q/k/v/o) + SFP + Faster R-CNN | 30382937 / 870975577 | corpus E+T (data/v4/manifests/CORPUS_E_ET_v1.json): E 38,454 + T 12,96... | none | - | yes | yes | dino3 | 11779 / 351.6 | DINOv3 License (custom Meta, 2025-08-19; | n/Y/n |  |
| C1_7b_head_cached | DINOv3 ViT-7B/16 frozen (cached bf16 feats) + SFP + Faster R-CNN head | 95253529 / 6811000000 | corpus E = frozen expB_train list: pyro-sdis TRAIN sites + D-Fire smok... | R@5% 0.228 (tiny 0.161); R@10% 0.353 (tiny 0.257) [reports/v3/matched_fp_val_trackBC.json] | 30/39 (med 360.0 s) | yes | cluster-only | none (cluster-only; cached-feature head, no serving backend) | n/a / 368.5 | DINOv3 License (custom Meta, 2025-08-19; | n/n/n | 7B backbone snapshot IS present locally (models/v3/hf/dinov3-vit7b16, 26 GB, pin |
| qwen3.5-9b_sft_lora | Qwen3.5-9B VLM + LoRA adapter (SFT) | ~0.9% LoRA (FINAL_REPORT_V2) / 9400000000 | SFT on data/v2/sft (60k examples built from corpus E labels; no FIgLib... | none | - | yes | cluster-only | none (cluster-only VLM; no serving backend) | n/a / 7300.0 | Apache-2.0 (Qwen3.5 base, models/v2/PINS | n/n/n |  |
| qwen3.5-9b_sft_grpo500 | Qwen3.5-9B VLM + LoRA adapter (SFT -> GRPO 500 steps) | ~0.9% LoRA (FINAL_REPORT_V2) / 9400000000 | GRPO 500 steps on top of the SFT adapter, reward from corpus-E boxes (... | none | - | yes | cluster-only | none (cluster-only VLM; no serving backend) | n/a / 6500.0 | Apache-2.0 (Qwen3.5 base, models/v2/PINS | n/n/n | status.json snapshot says phase=training (stale); final_adapter.tgz present and  |
| qwen3.5-27b_sft_qlora | Qwen3.5-27B VLM nf4 + QLoRA adapter (SFT) | QLoRA adapter (FINAL_REPORT_V2) / 27000000000 | SFT QLoRA (nf4, 0.75 MP) on data/v2/sft; no FIgLib... | none | - | yes | cluster-only | none (cluster-only VLM; no serving backend) | n/a / 17800.0 | Apache-2.0 (Qwen3.5 base, models/v2/PINS | n/n/n |  |
| deimv2_M_coco_pretrain | DEIMv2-M (DINOv3 distilled) COCO pretrain | n/a | upstream COCO weights; sanity fine-tune models/v3/runs/deimv2M_sanity ... | none | - | yes | excluded | deimv2 (registered in backends.AVAILABLE, never implemented) | n/a / n/a | DEIMv2 License - NON-COMMERCIAL ONLY, we | n/n/n |  |
| v5bA_ctrl_s3 | YOLO11m | 20053779 / 20053779 | corpus E = frozen expB_train list: pyro-sdis TRAIN sites + D-Fire smok... | R@5% 0.202 (tiny 0.109); R@10% 0.302 (tiny 0.191) [reports/v5b/val/matched_fp_sdisval.json] | 30/39 (med 240.0 s) | yes | yes | yolo | 275 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | Y/Y/n | SEED PROVENANCE: seed-0 REPLICATE of expV4A_ctrl_s{0,1,2} (run_yolo_arm.sh canno |
| v5bA_ctrl_s1 | YOLO11m | 20053779 / 20053779 | corpus E = frozen expB_train list: pyro-sdis TRAIN sites + D-Fire smok... | none | - | yes | yes | yolo | 296 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | Y/Y/n |  |
| v5bA_ctrl_s2 | YOLO11m | 20053779 / 20053779 | corpus E = frozen expB_train list: pyro-sdis TRAIN sites + D-Fire smok... | none | - | yes | yes | yolo | 342 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | Y/Y/n |  |
| v5bB_s0 | YOLO11m | 20053779 / 20053779 | corpus B_v1 (data/v5b/manifests/CORPUS_B_v1.json): E 38,454 + Boreal t... | R@5% 0.183 (tiny 0.075); R@10% 0.302 (tiny 0.174) [reports/v5b/val/matched_fp_sdisval.json] | 29/39 (med 240.0 s) | yes | yes | yolo | 334 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | Y/Y/n |  |
| v5bB_s1 | YOLO11m | 20053779 / 20053779 | corpus B_v1 (data/v5b/manifests/CORPUS_B_v1.json): E 38,454 + Boreal t... | none | - | yes | yes | yolo | 338 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | n/n/n | IN PROGRESS: latest seg03 epoch 27/43, phase segment_end (rc 124) at 2026-10-02T |
| v5bB_s2 | YOLO11m | 20053779 / 20053779 | corpus B_v1 (data/v5b/manifests/CORPUS_B_v1.json): E 38,454 + Boreal t... | none | - | yes | yes | yolo | 341 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | n/n/n | IN PROGRESS: latest seg03 epoch 27/43, phase segment_end (rc 124) at 2026-10-02T |
| v5bC_s0 | YOLO11m | 20053779 / 20053779 | corpus C_v1 (CORPUS_C_v1.json): B + conservative teacher IGNORE region... | R@5% 0.183 (tiny 0.087); R@10% 0.276 (tiny 0.172) [reports/v5b/val/matched_fp_sdisval.json] | 31/39 (med 361.0 s) | yes | yes | yolo | 351 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | Y/Y/n |  |
| v5bD_s0 | YOLO11m | 20053779 / 20053779 | corpus D_v1 (CORPUS_D_v1.json): B + filtered teacher PSEUDO-boxes (495... | R@5% 0.170 (tiny 0.084); R@10% 0.287 (tiny 0.175) [reports/v5b/val/matched_fp_sdisval.json] | 29/39 (med 361.0 s) | yes | yes | yolo | 314 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | Y/Y/n |  |
| v5bD2_s0 | YOLO11m | 20053779 / 20053779 | corpus D2_v1 (CORPUS_D2_v1.json): B + pseudo-boxes from calibrated WBF... | R@5% 0.158 (tiny 0.074); R@10% 0.274 (tiny 0.167) [reports/v5b/val/matched_fp_sdisval.json] | 31/39 (med 240.0 s) | yes | yes | yolo | 311 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | Y/Y/n |  |
| v5bD3_s0 | YOLO11m | 20053779 / 20053779 | corpus D3_v1 (CORPUS_D3_v1.json): B unchanged + 3,796 pseudo-boxed UNL... | R@5% 0.164 (tiny 0.052); R@10% 0.239 (tiny 0.113) [reports/v5b/val/matched_fp_sdisval.json] | 32/39 (med 60.0 s) | yes | yes | yolo | 319 / n/a | AGPL-3.0 (ultralytics YOLO11 code + COCO | Y/Y/n | 40 ep (not 43): exposure-matched to A on 57,307 lines (2,292,280 vs 2,307,240 ex |
| v5bB200_lora_L_1920 | DINOv3 ViT-L/16 LoRA r16 (q/k/v/o) + SFP + Faster R-CNN @ 1920x1088 | 25450777 / 328580377 | corpus E + Boreal-train (data/v5b/coco/instances_train_eb.json = froze... | R@5% 0.165 (tiny 0.096); R@10% 0.280 (tiny 0.203) [reports/v5b/matched_fp_b200_dino_sdis_val.json] | - | yes | yes | dino3 | 17498 / n/a | DINOv3 License (custom Meta, 2025-08-19; | n/n/n | CONFOUNDED vs v4B4: resolution (1920x1088 vs 1280x736) AND corpus (E+Boreal 40,5 |
| v5bB200_lora_L_1280 | DINOv3 ViT-L/16 LoRA r16 (q/k/v/o) + SFP + Faster R-CNN @ 1280x736 | 25450777 / 328580377 | corpus E + Boreal-train (data/v5b/coco/instances_train_eb.json = froze... | R@5% 0.193 (tiny 0.135); R@10% 0.307 (tiny 0.232) [reports/v5b/matched_fp_b200_dino_sdis_val.json] | - | yes | yes | dino3 | 5757 / n/a | DINOv3 License (custom Meta, 2025-08-19; | n/Y/n | best_model.pt (102 MB, trainable-only) registered; best.pt (305 MB) = same 238 m |
| segV5b_a_frozen | SEG: DINOv3 ViT-L/16 frozen + SFP + FPNLite seg head (pixel smoke mask) | 10102401 / 313232001 | data/v5b/manifests/seg_v5b_manifest.json: 665 Boreal TRAIN-burn (karkk... | none | - | yes | yes | seg_infer (scripts/v5b/seg_infer.py SmokeSegmenter; demo /segment slot, NOT a src.serve detector backend) | 5265 / n/a | DINOv3 License (custom Meta, 2025-08-19; | n/n/n | best.pt = best of 10 epoch evals on heinola VAL (selection-optimistic; final-epo |
| segV5b_a_lora | SEG: DINOv3 ViT-L/16 LoRA r16 (q/k/v/o) + SFP + FPNLite seg head | 13248129 / 316377729 | data/v5b/manifests/seg_v5b_manifest.json: 665 Boreal TRAIN-burn (karkk... | none | - | yes | yes | seg_infer (scripts/v5b/seg_infer.py SmokeSegmenter) | 5084 / n/a | DINOv3 License (custom Meta, 2025-08-19; | n/n/n | LoRA overfits after epoch 2 with 665 train frames; best.pt = epoch 2 (step 166); |
| segV5b_a_r50 | SEG: ResNet-50 (ImageNet, frozen) + FPNLite seg head - baseline | 5410177 / 28918209 | data/v5b/manifests/seg_v5b_manifest.json: 665 Boreal TRAIN-burn (karkk... | none | - | yes | yes | seg_infer (scripts/v5b/seg_infer.py SmokeSegmenter) | 577 / n/a | BSD-3-Clause (torchvision resnet50 IMAGE | n/n/n | trunk weights NOT in best.pt: SmokeSegmenter reloads models/v5b/seg/pretrained/r |
| qwen3.5-27b_bf16_lora_SMOKETEST | Qwen3.5-27B VLM bf16 + LoRA r16 a32 adapter - 200-step SMOKE TEST | 79691776 / 27436420336 | SMOKE TEST ONLY: 200 steps of bf16 LoRA SFT (eff. batch 64 = 12,800 ex... | none | - | yes | cluster-only | none (cluster-only VLM; no serving backend) | n/a / n/a | Apache-2.0 (Qwen3.5 base, models/v2/PINS | n/n/n | SMOKE TEST ONLY: 200 steps (0.21 epoch) = plumbing/throughput measurement, NOT a |
| ensemble_accurate_v5b | ACCURATE ensemble: calibrated WBF of yolo_A + dino_B4 + dino_B6 + dino_B1 | 0 / 1545044382 | NO TRAINING: recipe fit on pyro-sdis VAL only (grouped 2-fold by site ... | none | - | yes | yes | composite (src.serve fusion of 4 registered backends; parallel full-frame WBF) | n/a / 352.0 | composite: AGPL-3.0 (YOLO11m v4A_ctrl_s2 | n/Y/n | thresholds did not transfer between the two VAL sites (accurate_recipe.json thre |

Sources:
- `models/v5b/MODEL_REGISTRY.json`
- `models/v5b/MODEL_REGISTRY.md`
- `reports/v5b/RECONCILIATION_V5B.md`


---

### T3b. V5b systems in training at generation time (not yet in the registry; numbers are training-time only)

Reservation hours are registry wall-clock (reserve -> release/now), not scheduler allocation (see T9). 'last segment' = the last job the orchestrator submitted (segments resume from /scratch last.pt). The B200 run's COCO mAP is the training-time 1,500-image sdis-val subset metric (same as v4 B4's 0.4086 best), not the matched-FP protocol.

| system | run | recipe | corpus | registry state | reserved node-h | last segment / progress |
|---|---|---|---|---|---|---|
| A | expV5bA_ctrl_s3 | YOLO11m COCO-init @1280 b48, seed 0 (REPLICATE #4 of seed 0, R1) | 38,454 lines / 60 ep (expB recipe) | released | 4.38 | [seg 5] submitted tianrgua-v5bactrls3s5-2euv (url ttl 3540s, resume=ye |
| B | expV5bB_s0 | YOLO11m COCO-init @1280 b48, seed 0 | 53,511 lines / 43 ep | released | 4.52 | [seg 5] submitted tianrgua-v5bbs0s5-iuov (url ttl 3540s, resume=yes) |
| C | expV5bC_s0 | YOLO11m COCO-init @1280 b48, seed 0 | 53,511 lines / 43 ep | released | 4.43 | [seg 5] submitted tianrgua-v5bcs0s5-dnf5 (url ttl 3540s, resume=yes) |
| D | expV5bD_s0 | YOLO11m COCO-init @1280 b48, seed 0 | 53,508 lines / 43 ep | released | 4.53 | [seg 5] submitted tianrgua-v5bds0s5-4zva (url ttl 3539s, resume=yes) |
| D2 | expV5bD2_s0 | YOLO11m COCO-init @1280 b48, seed 0 | 53,508 lines / 43 ep | released | 4.34 | [seg 5] submitted tianrgua-v5bd2s0s5-cz7g (url ttl 3540s, resume=yes) |
| D3 | expV5bD3_s0 | YOLO11m COCO-init @1280 b48, seed 0 | 57,307 lines / 40 ep | released | 4.51 | [seg 5] submitted tianrgua-v5bd3s0s5-3hfp (url ttl 3540s, resume=yes) |
| S6 | v5bB200_lora_L_1920 | DINOv3-L LoRA r16 + SFP + Faster R-CNN @1920x1088, bs 4/GPU, 10 ep (B4 recipe) | E + Boreal-train 40,516 imgs (no AuSmoke) | released | 18.68 (B200) | 10 epoch evals; last train step 12660 (of 12,660), best 1,500-img sdis-val COCO mAP 0.4162 @ step 8862 |

Sources:
- `cluster/v5b/registry.json`
- `data/v5b/manifests/CORPUS_B_v1.json`
- `data/v5b/manifests/CORPUS_C_v1.json`
- `data/v5b/manifests/CORPUS_D2_v1.json`
- `data/v5b/manifests/CORPUS_D3_v1.json`
- `data/v5b/manifests/CORPUS_D_v1.json`
- `logs/v5b/arm*.log (last 'submitted' line)`
- `models/v5b/b200/v5bB200_lora_L_1920/metrics.jsonl`


---

### T4a. VAL matched-FP: pyro-sdis VAL (fixed camera; sites cabanelle + marguerite; 7,335 imgs / 1,158 negatives; 1% FP = ~12 negatives)

Recall / tiny-recall (<0.1% area GT) / precision at matched negative-image FP, interpolated on each system's own confidence sweep (src.eval.eval_detection.sweep_report, IoU>=0.4, grid 0.01..0.99 step 0.01; +0.991..0.999 for DINO-family tokens, as scripts/v4/compare_matched_fp.py --top-fine). 'tiny n/a' = the set has no tiny GT boxes (FireMan/Boreal author boxes are large plume regions). The eval JSONs record `imgsz: 1152`, which is eval_detection's CLI default when fed --preds-jsonl; the pod inference ran at IMGSZ=1280 (cluster/v4/yolo_eval_entry.sh default). FireMan VAL recall saturates for v1-era models -> it is NOT a faint-smoke benchmark (STATUS 10:57 UTC).

| system | 1% FP | 2% FP | 5% FP | 10% FP | 20% FP | denominators |
|---|---|---|---|---|---|---|
| S0_expE | c=0.799 R 0.059 / tiny 0.013 / P 0.961 | c=0.775 R 0.092 / tiny 0.026 / P 0.946 | c=0.721 R 0.190 / tiny 0.097 / P 0.932 | c=0.655 R 0.309 / tiny 0.201 / P 0.915 | c=0.535 R 0.479 / tiny 0.364 / P 0.888 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S0b_expB | c=0.808 R 0.058 / tiny 0.011 / P 0.961 | c=0.780 R 0.098 / tiny 0.030 / P 0.954 | c=0.723 R 0.204 / tiny 0.104 / P 0.940 | c=0.650 R 0.329 / tiny 0.203 / P 0.925 | c=0.532 R 0.485 / tiny 0.359 / P 0.887 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S1_A | c=0.812 R 0.056 / tiny 0.012 / P 0.954 | c=0.780 R 0.108 / tiny 0.041 / P 0.950 | c=0.729 R 0.202 / tiny 0.109 / P 0.938 | c=0.673 R 0.302 / tiny 0.191 / P 0.921 | c=0.534 R 0.499 / tiny 0.374 / P 0.892 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S1_A_s1 | c=0.807 R 0.043 / tiny 0.007 / P 0.957 | c=0.785 R 0.077 / tiny 0.021 / P 0.948 | c=0.727 R 0.184 / tiny 0.089 / P 0.938 | c=0.666 R 0.287 / tiny 0.167 / P 0.916 | c=0.542 R 0.458 / tiny 0.331 / P 0.887 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S1_A_s2 | c=0.809 R 0.050 / tiny 0.011 / P 0.963 | c=0.789 R 0.080 / tiny 0.025 / P 0.953 | c=0.717 R 0.204 / tiny 0.100 / P 0.940 | c=0.646 R 0.313 / tiny 0.179 / P 0.924 | c=0.500 R 0.493 / tiny 0.338 / P 0.891 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S2_B | c=0.814 R 0.056 / tiny 0.011 / P 0.965 | c=0.785 R 0.103 / tiny 0.027 / P 0.957 | c=0.740 R 0.183 / tiny 0.075 / P 0.941 | c=0.672 R 0.302 / tiny 0.174 / P 0.916 | c=0.557 R 0.457 / tiny 0.335 / P 0.881 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S2_B_s1 | c=0.813 R 0.037 / tiny 0.005 / P 0.946 | c=0.784 R 0.082 / tiny 0.022 / P 0.948 | c=0.728 R 0.180 / tiny 0.085 / P 0.933 | c=0.659 R 0.292 / tiny 0.173 / P 0.908 | c=0.528 R 0.449 / tiny 0.325 / P 0.871 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S2_B_s2 | c=0.816 R 0.042 / tiny 0.006 / P 0.950 | c=0.784 R 0.093 / tiny 0.025 / P 0.950 | c=0.728 R 0.196 / tiny 0.089 / P 0.941 | c=0.656 R 0.328 / tiny 0.204 / P 0.916 | c=0.559 R 0.465 / tiny 0.329 / P 0.882 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S3_C | c=0.819 R 0.049 / tiny 0.009 / P 0.957 | c=0.796 R 0.089 / tiny 0.027 / P 0.952 | c=0.745 R 0.183 / tiny 0.087 / P 0.934 | c=0.691 R 0.276 / tiny 0.172 / P 0.913 | c=0.585 R 0.433 / tiny 0.313 / P 0.880 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S4_D | c=0.827 R 0.043 / tiny 0.008 / P 0.959 | c=0.804 R 0.082 / tiny 0.023 / P 0.949 | c=0.758 R 0.170 / tiny 0.084 / P 0.932 | c=0.695 R 0.287 / tiny 0.175 / P 0.919 | c=0.578 R 0.454 / tiny 0.336 / P 0.884 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S4b_D2 | c=0.831 R 0.044 / tiny 0.011 / P 0.956 | c=0.808 R 0.083 / tiny 0.033 / P 0.946 | c=0.765 R 0.158 / tiny 0.074 / P 0.927 | c=0.703 R 0.274 / tiny 0.167 / P 0.911 | c=0.572 R 0.444 / tiny 0.325 / P 0.881 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S4c_D3 | c=0.788 R 0.055 / tiny 0.006 / P 0.962 | c=0.774 R 0.074 / tiny 0.013 / P 0.948 | c=0.723 R 0.164 / tiny 0.052 / P 0.934 | c=0.679 R 0.239 / tiny 0.113 / P 0.909 | c=0.564 R 0.419 / tiny 0.276 / P 0.879 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S5_ensemble_S1_A | n/a (FP floor 3.4%) | n/a (FP floor 3.4%) | c=0.982 R 0.237 / tiny 0.132 / P 0.941 | c=0.962 R 0.344 / tiny 0.225 / P 0.921 | c=0.904 R 0.533 / tiny 0.409 / P 0.892 | 7,335 imgs / 1,158 neg; tiny GT 2277 |
| S5_ensemble_v4a_s2 | n/a (FP floor 3.3%) | n/a (FP floor 3.3%) | c=0.982 R 0.232 / tiny 0.131 / P 0.940 | c=0.962 R 0.343 / tiny 0.226 / P 0.920 | c=0.905 R 0.533 / tiny 0.407 / P 0.891 | 7,335 imgs / 1,158 neg; tiny GT 2277 |

Sources:
- `reports/v5b/val/eval_S0_expE_sdisval.raw.parquet`
- `reports/v5b/val/eval_S0b_expB_sdisval.raw.parquet`
- `reports/v5b/val/eval_S1_A_sdisval.raw.parquet`
- `reports/v5b/val/eval_S1_A_s1_sdisval.raw.parquet`
- `reports/v5b/val/eval_S1_A_s2_sdisval.raw.parquet`
- `reports/v5b/val/eval_S2_B_sdisval.raw.parquet`
- `reports/v5b/val/eval_S2_B_s1_sdisval.raw.parquet`
- `reports/v5b/val/eval_S2_B_s2_sdisval.raw.parquet`
- `reports/v5b/val/eval_S3_C_sdisval.raw.parquet`
- `reports/v5b/val/eval_S4_D_sdisval.raw.parquet`
- `reports/v5b/val/eval_S4b_D2_sdisval.raw.parquet`
- `reports/v5b/val/eval_S4c_D3_sdisval.raw.parquet`
- `reports/v5b/val/eval_S5_ensemble_S1_A_sdisval.raw.parquet`
- `reports/v5b/val/eval_S5_ensemble_v4a_s2_sdisval.raw.parquet`


---

### T4b. VAL matched-FP: Boreal heinola aerial VAL (1,123 imgs / 217 negatives; 1% FP = ~2 negatives -> 1-2% rows are weakly supported)

Recall / tiny-recall (<0.1% area GT) / precision at matched negative-image FP, interpolated on each system's own confidence sweep (src.eval.eval_detection.sweep_report, IoU>=0.4, grid 0.01..0.99 step 0.01; +0.991..0.999 for DINO-family tokens, as scripts/v4/compare_matched_fp.py --top-fine). 'tiny n/a' = the set has no tiny GT boxes (FireMan/Boreal author boxes are large plume regions). The eval JSONs record `imgsz: 1152`, which is eval_detection's CLI default when fed --preds-jsonl; the pod inference ran at IMGSZ=1280 (cluster/v4/yolo_eval_entry.sh default). FireMan VAL recall saturates for v1-era models -> it is NOT a faint-smoke benchmark (STATUS 10:57 UTC).

| system | 1% FP | 2% FP | 5% FP | 10% FP | 20% FP | denominators |
|---|---|---|---|---|---|---|
| S0_expE | c=0.498 R 0.553 / tiny 0.000 / P 0.901 | c=0.356 R 0.698 / tiny 0.000 / P 0.814 | c=0.262 R 0.751 / tiny 0.000 / P 0.743 | c=0.223 R 0.779 / tiny 0.000 / P 0.711 | c=0.145 R 0.810 / tiny 0.000 / P 0.618 | 1,123 imgs / 217 neg; tiny GT 1 |
| S0b_expB | c=0.498 R 0.663 / tiny 0.000 / P 0.925 | c=0.436 R 0.709 / tiny 0.000 / P 0.915 | c=0.348 R 0.762 / tiny 0.000 / P 0.864 | c=0.333 R 0.771 / tiny 0.000 / P 0.848 | c=0.312 R 0.782 / tiny 0.000 / P 0.813 | 1,123 imgs / 217 neg; tiny GT 1 |
| S1_A | c=0.498 R 0.562 / tiny 0.000 / P 0.901 | c=0.346 R 0.743 / tiny 0.000 / P 0.786 | c=0.255 R 0.800 / tiny 0.000 / P 0.717 | c=0.226 R 0.814 / tiny 0.000 / P 0.684 | c=0.179 R 0.837 / tiny 0.000 / P 0.639 | 1,123 imgs / 217 neg; tiny GT 1 |
| S1_A_s1 | c=0.439 R 0.582 / tiny 0.000 / P 0.854 | c=0.416 R 0.598 / tiny 0.000 / P 0.835 | c=0.382 R 0.626 / tiny 0.000 / P 0.801 | c=0.319 R 0.673 / tiny 0.000 / P 0.760 | c=0.259 R 0.724 / tiny 0.000 / P 0.709 | 1,123 imgs / 217 neg; tiny GT 1 |
| S1_A_s2 | c=0.438 R 0.682 / tiny 0.000 / P 0.855 | c=0.387 R 0.717 / tiny 0.000 / P 0.819 | c=0.331 R 0.748 / tiny 0.000 / P 0.781 | c=0.308 R 0.757 / tiny 0.000 / P 0.760 | c=0.259 R 0.783 / tiny 0.000 / P 0.711 | 1,123 imgs / 217 neg; tiny GT 1 |
| S2_B | c=0.308 R 0.852 / tiny 0.000 / P 0.866 | c=0.157 R 0.901 / tiny 0.000 / P 0.784 | c=0.021 R 0.955 / tiny 0.000 / P 0.583 | n/a | n/a | 1,123 imgs / 217 neg; tiny GT 1 |
| S2_B_s1 | c=0.408 R 0.828 / tiny 0.000 / P 0.917 | c=0.117 R 0.917 / tiny 0.000 / P 0.803 | c=0.031 R 0.937 / tiny 0.000 / P 0.668 | n/a | n/a | 1,123 imgs / 217 neg; tiny GT 1 |
| S2_B_s2 | c=0.218 R 0.850 / tiny 0.000 / P 0.896 | c=0.178 R 0.863 / tiny 0.000 / P 0.873 | c=0.049 R 0.925 / tiny 0.000 / P 0.711 | c=0.039 R 0.930 / tiny 0.000 / P 0.677 | c=0.024 R 0.939 / tiny 0.000 / P 0.618 | 1,123 imgs / 217 neg; tiny GT 1 |
| S3_C | c=0.318 R 0.861 / tiny 0.000 / P 0.910 | c=0.197 R 0.891 / tiny 0.000 / P 0.859 | c=0.028 R 0.946 / tiny 0.000 / P 0.626 | c=0.021 R 0.948 / tiny 0.000 / P 0.595 | n/a | 1,123 imgs / 217 neg; tiny GT 1 |
| S4_D | c=0.308 R 0.822 / tiny 0.000 / P 0.909 | c=0.146 R 0.868 / tiny 0.000 / P 0.859 | n/a | n/a | n/a | 1,123 imgs / 217 neg; tiny GT 1 |
| S4b_D2 | c=0.328 R 0.881 / tiny 0.000 / P 0.944 | c=0.117 R 0.943 / tiny 0.000 / P 0.809 | n/a | n/a | n/a | 1,123 imgs / 217 neg; tiny GT 1 |
| S4c_D3 | c=0.388 R 0.821 / tiny 0.000 / P 0.928 | c=0.236 R 0.865 / tiny 0.000 / P 0.895 | n/a | n/a | n/a | 1,123 imgs / 217 neg; tiny GT 1 |
| S5_ensemble_S1_A | PENDING | PENDING | PENDING | PENDING | PENDING | - |
| S5_ensemble_v4a_s2 | PENDING | PENDING | PENDING | PENDING | PENDING | - |

Sources:
- `reports/v5b/val/eval_S0_expE_borealval.raw.parquet`
- `reports/v5b/val/eval_S0b_expB_borealval.raw.parquet`
- `reports/v5b/val/eval_S1_A_borealval.raw.parquet`
- `reports/v5b/val/eval_S1_A_s1_borealval.raw.parquet`
- `reports/v5b/val/eval_S1_A_s2_borealval.raw.parquet`
- `reports/v5b/val/eval_S2_B_borealval.raw.parquet`
- `reports/v5b/val/eval_S2_B_s1_borealval.raw.parquet`
- `reports/v5b/val/eval_S2_B_s2_borealval.raw.parquet`
- `reports/v5b/val/eval_S3_C_borealval.raw.parquet`
- `reports/v5b/val/eval_S4_D_borealval.raw.parquet`
- `reports/v5b/val/eval_S4b_D2_borealval.raw.parquet`
- `reports/v5b/val/eval_S4c_D3_borealval.raw.parquet`


---

### T4c. VAL matched-FP: FireMan val_s5 (burn D video_015, every 5th frame; 1,706 imgs / 231 author negatives; one video = one unit; author boxes are large region boxes)

Recall / tiny-recall (<0.1% area GT) / precision at matched negative-image FP, interpolated on each system's own confidence sweep (src.eval.eval_detection.sweep_report, IoU>=0.4, grid 0.01..0.99 step 0.01; +0.991..0.999 for DINO-family tokens, as scripts/v4/compare_matched_fp.py --top-fine). 'tiny n/a' = the set has no tiny GT boxes (FireMan/Boreal author boxes are large plume regions). The eval JSONs record `imgsz: 1152`, which is eval_detection's CLI default when fed --preds-jsonl; the pod inference ran at IMGSZ=1280 (cluster/v4/yolo_eval_entry.sh default). FireMan VAL recall saturates for v1-era models -> it is NOT a faint-smoke benchmark (STATUS 10:57 UTC).

| system | 1% FP | 2% FP | 5% FP | 10% FP | 20% FP | denominators |
|---|---|---|---|---|---|---|
| S0_expE | c=0.256 R 0.963 / tiny n/a / P 0.523 | c=0.208 R 0.971 / tiny n/a / P 0.503 | c=0.177 R 0.981 / tiny n/a / P 0.489 | c=0.070 R 0.993 / tiny n/a / P 0.439 | n/a | 1,706 imgs / 231 neg; tiny GT 0 |
| S0b_expB | c=0.287 R 0.994 / tiny n/a / P 0.360 | c=0.214 R 0.996 / tiny n/a / P 0.346 | c=0.144 R 0.997 / tiny n/a / P 0.334 | c=0.063 R 1.000 / tiny n/a / P 0.315 | c=0.022 R 1.000 / tiny n/a / P 0.295 | 1,706 imgs / 231 neg; tiny GT 0 |
| S1_A | c=0.197 R 0.971 / tiny n/a / P 0.489 | c=0.147 R 0.984 / tiny n/a / P 0.474 | c=0.091 R 0.997 / tiny n/a / P 0.450 | c=0.030 R 1.000 / tiny n/a / P 0.405 | n/a | 1,706 imgs / 231 neg; tiny GT 0 |
| S1_A_s1 | c=0.277 R 0.960 / tiny n/a / P 0.527 | c=0.262 R 0.965 / tiny n/a / P 0.523 | c=0.236 R 0.972 / tiny n/a / P 0.510 | c=0.165 R 0.983 / tiny n/a / P 0.475 | c=0.040 R 1.000 / tiny n/a / P 0.389 | 1,706 imgs / 231 neg; tiny GT 0 |
| S1_A_s2 | c=0.427 R 0.923 / tiny n/a / P 0.750 | c=0.404 R 0.929 / tiny n/a / P 0.728 | c=0.282 R 0.960 / tiny n/a / P 0.638 | c=0.179 R 0.977 / tiny n/a / P 0.582 | c=0.043 R 0.996 / tiny n/a / P 0.495 | 1,706 imgs / 231 neg; tiny GT 0 |
| S2_B | c=0.178 R 0.992 / tiny n/a / P 0.546 | c=0.168 R 0.993 / tiny n/a / P 0.544 | c=0.132 R 0.995 / tiny n/a / P 0.538 | c=0.076 R 0.997 / tiny n/a / P 0.521 | c=0.026 R 0.997 / tiny n/a / P 0.485 | 1,706 imgs / 231 neg; tiny GT 0 |
| S2_B_s1 | c=0.053 R 0.983 / tiny n/a / P 0.590 | c=0.029 R 0.988 / tiny n/a / P 0.575 | n/a | n/a | n/a | 1,706 imgs / 231 neg; tiny GT 0 |
| S2_B_s2 | c=0.437 R 0.968 / tiny n/a / P 0.704 | c=0.374 R 0.980 / tiny n/a / P 0.685 | c=0.194 R 0.993 / tiny n/a / P 0.636 | c=0.044 R 0.997 / tiny n/a / P 0.576 | n/a | 1,706 imgs / 231 neg; tiny GT 0 |
| S3_C | c=0.197 R 0.994 / tiny n/a / P 0.588 | c=0.182 R 0.995 / tiny n/a / P 0.585 | c=0.072 R 0.998 / tiny n/a / P 0.563 | c=0.045 R 0.998 / tiny n/a / P 0.549 | c=0.021 R 1.000 / tiny n/a / P 0.524 | 1,706 imgs / 231 neg; tiny GT 0 |
| S4_D | c=0.193 R 0.992 / tiny n/a / P 0.606 | c=0.114 R 0.995 / tiny n/a / P 0.593 | c=0.082 R 0.996 / tiny n/a / P 0.583 | c=0.036 R 0.998 / tiny n/a / P 0.558 | n/a | 1,706 imgs / 231 neg; tiny GT 0 |
| S4b_D2 | c=0.227 R 0.970 / tiny n/a / P 0.630 | c=0.114 R 0.988 / tiny n/a / P 0.600 | c=0.051 R 0.994 / tiny n/a / P 0.572 | c=0.024 R 0.997 / tiny n/a / P 0.548 | n/a | 1,706 imgs / 231 neg; tiny GT 0 |
| S4c_D3 | c=0.319 R 0.990 / tiny n/a / P 0.621 | c=0.284 R 0.992 / tiny n/a / P 0.611 | c=0.262 R 0.993 / tiny n/a / P 0.606 | c=0.199 R 0.994 / tiny n/a / P 0.587 | c=0.037 R 0.999 / tiny n/a / P 0.502 | 1,706 imgs / 231 neg; tiny GT 0 |
| S5_ensemble_S1_A | PENDING | PENDING | PENDING | PENDING | PENDING | - |
| S5_ensemble_v4a_s2 | PENDING | PENDING | PENDING | PENDING | PENDING | - |

Sources:
- `reports/v5b/val/eval_S0_expE_firemanval.raw.parquet`
- `reports/v5b/val/eval_S0b_expB_firemanval.raw.parquet`
- `reports/v5b/val/eval_S1_A_firemanval.raw.parquet`
- `reports/v5b/val/eval_S1_A_s1_firemanval.raw.parquet`
- `reports/v5b/val/eval_S1_A_s2_firemanval.raw.parquet`
- `reports/v5b/val/eval_S2_B_firemanval.raw.parquet`
- `reports/v5b/val/eval_S2_B_s1_firemanval.raw.parquet`
- `reports/v5b/val/eval_S2_B_s2_firemanval.raw.parquet`
- `reports/v5b/val/eval_S3_C_firemanval.raw.parquet`
- `reports/v5b/val/eval_S4_D_firemanval.raw.parquet`
- `reports/v5b/val/eval_S4b_D2_firemanval.raw.parquet`
- `reports/v5b/val/eval_S4c_D3_firemanval.raw.parquet`


---

### T5. Model collaboration on pyro-sdis VAL: site-grouped out-of-fold (OOF) mean recall over FP 1/2/5/10/20% (selection statistic = macro over the two held-out sites)

Fusions are fit on one VAL site and evaluated on the other (calibration maps + WBF params); single models need no fitting. Chosen ACCURATE recipe = A+B4+B6+B1 WBF (macro 0.3535 vs best single B6 0.3258). Learned fusion (logreg) beats WBF by only +0.46 pt (< 1 pt bar) -> not adopted. Pooled-OOF numbers are pessimistic because calibration does not transfer between sites (ENSEMBLE_DEV_V5B.md T8). Latency (batch-1 L40S fp32 @1280, from reports): components A 11.2 ms, B4 157.5 ms, B6 351.6 ms, B1 156.6 ms; fusion 0.381 ms CPU; PARALLEL 352.0 ms; serial 677.3 ms.

| system | OOF cabanelle | OOF marguerite | macro | whole-VAL / pooled | whole-VAL R @1/2/5/10/20% | WBF/union params (all-VAL fit) |
|---|---|---|---|---|---|---|
| single A (yolo_A) | 0.207 | 0.436 | 0.3213 | 0.230 | 0.056 / 0.100 / 0.200 / 0.302 / 0.493 | - |
| single B4 (dino_B4) | 0.180 | 0.397 | 0.2885 | 0.242 | 0.083 / 0.123 / 0.208 / 0.309 / 0.486 | - |
| single B6 (dino_B6) | 0.213 | 0.438 | 0.3258 | 0.260 | 0.103 / 0.133 / 0.228 / 0.340 / 0.495 | - |
| single RF (rfdetr_L) | 0.244 | 0.383 | 0.3136 | 0.232 | 0.053 / 0.102 / 0.201 / 0.314 / 0.486 | - |
| single expE (yolo_expE) | 0.208 | 0.435 | 0.3215 | 0.226 | 0.058 / 0.092 / 0.188 / 0.308 / 0.483 | - |
| single B1 (dino_B1) | 0.192 | 0.404 | 0.2980 | 0.220 | 0.065 / 0.095 / 0.176 / 0.299 / 0.465 | - |
| OR/union+NMS A+B4 | 0.209 | 0.434 | 0.3216 | 0.161 | - | iou0.7 |
| WBF A+B4 | 0.204 | 0.453 | 0.3285 | 0.142 | - | avg,iou0.4,skip0.8 |
| OR/union+NMS A+B4+B6 | 0.220 | 0.458 | 0.3386 | 0.172 | - | iou0.7 |
| WBF A+B4+B6 | 0.221 | 0.460 | 0.3401 | 0.173 | - | max,iou0.3,skip0.0 |
| OR/union+NMS A+B4+B6+RF | 0.224 | 0.443 | 0.3333 | 0.174 | - | iou0.7 |
| WBF A+B4+B6+RF | 0.225 | 0.447 | 0.3362 | 0.175 | - | max,iou0.3,skip0.0 |
| OR/union+NMS A+B4+B6+expE | 0.221 | 0.455 | 0.3380 | 0.170 | - | iou0.7 |
| WBF A+B4+B6+expE | 0.222 | 0.459 | 0.3407 | 0.118 | - | max,iou0.3,skip0.0 |
| OR/union+NMS A+B4+B6+B1 | 0.230 | 0.473 | 0.3513 | 0.172 | - | iou0.7 |
| WBF A+B4+B6+B1 | 0.230 | 0.477 | 0.3535 | 0.111 | - | max,iou0.3,skip0.0 |

**model-addition protocol (keep if >= +1.0 pt macro-OOF)**

| candidate | set tested | union macro | WBF macro | gain (pt) | kept |
|---|---|---|---|---|---|
| dino_B6 | A+B4+B6 | 0.3386 | 0.3401 | +1.16 | yes |
| rfdetr_L | A+B4+B6+RF | 0.3333 | 0.3362 | -0.39 | no |
| yolo_expE | A+B4+B6+expE | 0.3380 | 0.3407 | +0.06 | no |
| dino_B1 | A+B4+B6+B1 | 0.3513 | 0.3535 | +1.34 | yes |

**bootstrap (500 image resamples within the held-out site), difference in OOF mean recall (pt)**

| held-out site | comparison | point | 95% CI | P(gain<=0) |
|---|---|---|---|---|
| cabanelle | fusion_vs_rfdetr_L | -1.43 | [-4.21, +1.76] | 0.770 |
| cabanelle | fusion_vs_dino_B6 | +1.77 | [-0.90, +3.97] | 0.086 |
| cabanelle | fusion_vs_yolo_A | +2.33 | [-0.24, +4.69] | 0.042 |
| cabanelle | fusion_vs_lean(A+B4+B6 wbf) | +1.08 | [-1.12, +3.01] | 0.180 |
| cabanelle | lean(A+B4+B6 wbf)_vs_dino_B6 | +0.69 | [-0.91, +2.22] | 0.194 |
| marguerite | fusion_vs_dino_B6 | +3.86 | [+1.00, +8.34] | 0.004 |
| marguerite | fusion_vs_yolo_A | +4.12 | [-0.90, +8.48] | 0.064 |
| marguerite | fusion_vs_lean(A+B4+B6 wbf) | +1.73 | [-1.60, +7.16] | 0.136 |
| marguerite | lean(A+B4+B6 wbf)_vs_dino_B6 | +2.13 | [-3.65, +6.11] | 0.206 |

Sources:
- `reports/v5b/ensemble_dev_v5b.json`
- `reports/v5b/ENSEMBLE_DEV_V5B.md`
- `models/v5b/fusion/accurate_recipe.json`


---

### T6. Thin-smoke SEGMENTATION branch (segV5b_a): heinola aerial VAL, thr 0.5, best epoch selected on VAL (selection-optimistic)

Train = 665 Boreal TRAIN-burn frames (628 masks: 601 SAM model-assisted + 27 human; 37 author negatives), VAL = 382 heinola frames (329 SAM + 7 human + 46 author negatives). 19.5 node-minutes on 1 L40S node (res-segv5b-a). IoU vs SAM masks measures agreement with SAM, not with humans; the 7 human masks show diffuse haze is under-segmented. Mask->box recall vs the authors' one-big-box labels (A: component bbox IoU>=0.4 / B: union bbox / C: localisation hit); the GT masks themselves reach only 0.887 / 0.946 / 1.000. Negative-frame false alarms are driven by ONE clear-cut flight (DJI_0024: 21/21 frames for every model) -> the map is an explanation overlay, not a stand-alone alarm.

| config | checkpoint | best | IoU | Dice | P | R | mIoU pos | IoU vs SAM (n) | IoU vs human (n) | neg-frame FA any / >=1% | mask->box A / B / C | params trainable / total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DINOv3-L LoRA r16 + SFP + FPN-lite | models/v5b/seg/segV5b_a/seg01/lora/best.pt | ep 2 / step 166 | 0.551 | 0.710 | 0.855 | 0.608 | 0.591 | 0.556 (329) | 0.442 (7) | 0.457 / 0.065 | 0.804 / 0.860 / 0.994 | 13,248,129 / 316,377,729 |
| DINOv3-L frozen + SFP + FPN-lite | models/v5b/seg/segV5b_a/seg02/frozen/best.pt | ep 2 / step 166 | 0.568 | 0.724 | 0.842 | 0.636 | 0.604 | 0.571 (329) | 0.474 (7) | 0.457 / 0.000 | 0.830 / 0.878 / 0.997 | 10,102,401 / 313,232,001 |
| ResNet-50 (ImageNet, frozen) + FPN-lite (baseline) | models/v5b/seg/segV5b_a/seg01/r50/best.pt | ep 3 / step 123 | 0.510 | 0.675 | 0.742 | 0.620 | 0.534 | 0.514 (329) | 0.453 (7) | 1.000 / 0.109 | 0.854 / 0.878 / 1.000 | 5,410,177 / 28,918,209 |

**threshold sweep (selected rows)**

| config | thr | IoU | Dice | P | R | neg FA any | neg FA >=1% | neg FP pixel frac |
|---|---|---|---|---|---|---|---|---|
| frozen | 0.3 | 0.572 | 0.728 | 0.819 | 0.655 | 0.457 | 0.000 | 0.0046 |
| frozen | 0.5 | 0.567 | 0.724 | 0.842 | 0.635 | 0.457 | 0.000 | 0.0027 |
| frozen | 0.7 | 0.558 | 0.716 | 0.862 | 0.613 | 0.457 | 0.000 | 0.0016 |
| lora | 0.3 | 0.553 | 0.712 | 0.831 | 0.623 | 0.500 | 0.391 | 0.0064 |
| lora | 0.5 | 0.550 | 0.709 | 0.854 | 0.606 | 0.457 | 0.065 | 0.0047 |
| lora | 0.7 | 0.541 | 0.702 | 0.876 | 0.586 | 0.457 | 0.000 | 0.0030 |
| r50 | 0.3 | 0.508 | 0.674 | 0.703 | 0.647 | 1.000 | 0.174 | 0.0090 |
| r50 | 0.5 | 0.508 | 0.674 | 0.742 | 0.618 | 1.000 | 0.109 | 0.0064 |
| r50 | 0.7 | 0.504 | 0.670 | 0.780 | 0.588 | 0.913 | 0.087 | 0.0046 |

**per-epoch heinola VAL IoU**

| config | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | ep9 | ep10 |
|---|---|---|---|---|---|---|---|---|---|---|
| lora | 0.522 | 0.551 | 0.506 | 0.484 | 0.523 | 0.507 | 0.520 | 0.526 | 0.526 | 0.529 |
| frozen | 0.481 | 0.568 | 0.484 | 0.499 | 0.519 | 0.517 | 0.511 | 0.515 | 0.523 | 0.523 |
| r50 | 0.444 | 0.496 | 0.510 | 0.495 | 0.498 | 0.479 | 0.490 | 0.490 | 0.493 | 0.494 |

Sources:
- `models/v5b/seg/segV5b_a/MANIFEST.json`
- `models/v5b/seg/segV5b_a/seg01/frozen/thr_sweep.json`
- `models/v5b/seg/segV5b_a/seg01/lora/metrics.jsonl`
- `models/v5b/seg/segV5b_a/seg01/lora/thr_sweep.json`
- `models/v5b/seg/segV5b_a/seg01/r50/metrics.jsonl`
- `models/v5b/seg/segV5b_a/seg01/r50/thr_sweep.json`
- `models/v5b/seg/segV5b_a/seg02/frozen/metrics.jsonl`
- `reports/v5b/SEGMENTATION_V5B.md`
- `data/v5b/manifests/seg_v5b_manifest.json`


---

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


---

### T8a. FINAL TEST roster: systems found under reports/v5b/test, H3 contamination status, pre-registered VAL confidences

One-shot evaluation ran at 2026-10-03T03:20:07Z
2026-10-03T03:48:01Z
2026-10-03T03:49:22Z
2026-10-03T03:49:43Z
2026-10-03T03:50:03Z (reports/v5b/test/.EVALUATED_ONCE). Pre-registered confs (PREREGISTRATION_V5B.md 'Thresholds per system') are computed from the VAL parquets by this script (reports/v5b/tables/_derived/preregistered_confs_v5b.json). H3 contamination: arms B/C/D/D2/D3 trained on ALL Sheffield frames incl. rec_0008 (LOCKBOX_V5B_FROZEN.md); the heuristic maps pre-registered tokens S2/S3/S4 and arm names B/C/D/D2/D3 to CONTAMINATED - drop a reports/v5b/test/ROSTER_V5B.json {"systems": {token: {"h3_contaminated": bool}}} to override. NOTE: LOCKBOX wording 'the ACCURATE ensemble, which contains B' conflates v5b arm B with the v4 DINO B-series; the shipped recipe (yolo_A + dino_B4/B6/B1) holds no Sheffield frame and is therefore NOT contaminated by its training lists.

| system token | sets present | H3 status | sdis-VAL conf @5/10/20% | FireMan-VAL conf @5/10% |
|---|---|---|---|---|
| S0_expE | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | clean (heuristic) | 0.05: 0.721, 0.10: 0.655, 0.20: 0.535 | 0.05: 0.177, 0.10: 0.070 |
| S0b_expB | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | clean (heuristic) | 0.05: 0.723, 0.10: 0.650, 0.20: 0.532 | 0.05: 0.144, 0.10: 0.063 |
| S1_A | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | clean (heuristic) | 0.05: 0.729, 0.10: 0.673, 0.20: 0.534 | 0.05: 0.091, 0.10: 0.030 |
| S1_A_s1 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | clean (heuristic) | 0.05: 0.727, 0.10: 0.666, 0.20: 0.542 | 0.05: 0.236, 0.10: 0.165 |
| S1_A_s2 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | clean (heuristic) | 0.05: 0.717, 0.10: 0.646, 0.20: 0.500 | 0.05: 0.282, 0.10: 0.179 |
| S2_B | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | CONTAMINATED (heuristic) | 0.05: 0.740, 0.10: 0.672, 0.20: 0.557 | 0.05: 0.132, 0.10: 0.076 |
| S2_B_s1 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | CONTAMINATED (heuristic) | 0.05: 0.728, 0.10: 0.659, 0.20: 0.528 | 0.05: n/a, 0.10: n/a |
| S2_B_s2 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | CONTAMINATED (heuristic) | 0.05: 0.728, 0.10: 0.656, 0.20: 0.559 | 0.05: 0.194, 0.10: 0.044 |
| S3_C | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | CONTAMINATED (heuristic) | 0.05: 0.745, 0.10: 0.691, 0.20: 0.585 | 0.05: 0.072, 0.10: 0.045 |
| S4_D | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | CONTAMINATED (heuristic) | 0.05: 0.758, 0.10: 0.695, 0.20: 0.578 | 0.05: 0.082, 0.10: 0.036 |
| S4b_D2 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | CONTAMINATED (heuristic) | 0.05: 0.765, 0.10: 0.703, 0.20: 0.572 | 0.05: 0.051, 0.10: 0.024 |
| S4c_D3 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | CONTAMINATED (heuristic) | 0.05: 0.723, 0.10: 0.679, 0.20: 0.564 | 0.05: 0.262, 0.10: 0.199 |
| S5_ensemble_v4a_s2 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | clean (heuristic) | 0.05: 0.982, 0.10: 0.962, 0.20: 0.905 | no fireman-val parquet -> n/a |
| v4B1_frozen_L | firemanholdout, sheffieldholdout | UNKNOWN (unknown) | no sdis-val parquet -> n/a | no fireman-val parquet -> n/a |
| v4B4_lora_L | firemanholdout, sheffieldholdout | UNKNOWN (unknown) | no sdis-val parquet -> n/a | no fireman-val parquet -> n/a |
| v4B6_lora_Hplus | firemanholdout, sheffieldholdout | UNKNOWN (unknown) | no sdis-val parquet -> n/a | no fireman-val parquet -> n/a |
| v4a_s2 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | UNKNOWN (unknown) | no sdis-val parquet -> n/a | no fireman-val parquet -> n/a |

Sources:
- `reports/v5b/test/eval_<sys>_<set>.{json,raw.parquet}`
- `reports/v5b/tables/_derived/preregistered_confs_v5b.json`
- `reports/v5b/LOCKBOX_V5B_FROZEN.md`
- `reports/v5b/PREREGISTRATION_V5B.md`


---

### T8b. FINAL TEST H1 - H1 fireman_holdout_s10 (burn C Taivalkoski, 5 videos, 1,949 frames, 0 negatives -> RECALL ONLY): recall at the pre-registered VAL confidences

One-shot evaluation ran at 2026-10-03T03:20:07Z
2026-10-03T03:48:01Z
2026-10-03T03:49:22Z
2026-10-03T03:49:43Z
2026-10-03T03:50:03Z (reports/v5b/test/.EVALUATED_ONCE). H1 has 0 explicit negatives (amendment 09:35 UTC) -> RECALL ONLY (image-level + box recall @ IoU>=0.4); no holdout FP claim. Author boxes are interpolated region boxes -> box IoU metrics are weak; one burn, 640x512 rectified RGB, every 10th frame. Exact recall at each conf is recomputed from the test parquet with the frozen matching code.

| system | R @ sdis-val 5% | R @ sdis-val 10% | R @ sdis-val 20% | R @ fireman-val 5% | R @ fireman-val 10% | denominators |
|---|---|---|---|---|---|---|
| S0_expE | c=0.721 R 0.464 (tiny n/a) | c=0.655 R 0.663 (tiny n/a) | c=0.535 R 0.888 (tiny n/a) | c=0.177 R 0.995 (tiny n/a) | c=0.070 R 1.000 (tiny n/a) | 1,949 imgs / 0 neg |
| S0b_expB | c=0.723 R 0.388 (tiny n/a) | c=0.650 R 0.590 (tiny n/a) | c=0.532 R 0.831 (tiny n/a) | c=0.144 R 0.992 (tiny n/a) | c=0.063 R 0.999 (tiny n/a) | 1,949 imgs / 0 neg |
| S1_A | c=0.729 R 0.612 (tiny n/a) | c=0.673 R 0.736 (tiny n/a) | c=0.534 R 0.907 (tiny n/a) | c=0.091 R 0.998 (tiny n/a) | c=0.030 R 1.000 (tiny n/a) | 1,949 imgs / 0 neg |
| S1_A_s1 | c=0.727 R 0.201 (tiny n/a) | c=0.666 R 0.438 (tiny n/a) | c=0.542 R 0.761 (tiny n/a) | c=0.236 R 0.985 (tiny n/a) | c=0.165 R 0.996 (tiny n/a) | 1,949 imgs / 0 neg |
| S1_A_s2 | c=0.717 R 0.444 (tiny n/a) | c=0.646 R 0.630 (tiny n/a) | c=0.500 R 0.908 (tiny n/a) | c=0.282 R 0.994 (tiny n/a) | c=0.179 R 0.999 (tiny n/a) | 1,949 imgs / 0 neg |
| S2_B | c=0.740 R 0.522 (tiny n/a) | c=0.672 R 0.718 (tiny n/a) | c=0.557 R 0.900 (tiny n/a) | c=0.132 R 0.991 (tiny n/a) | c=0.076 R 0.999 (tiny n/a) | 1,949 imgs / 0 neg |
| S2_B_s1 | c=0.728 R 0.426 (tiny n/a) | c=0.659 R 0.650 (tiny n/a) | c=0.528 R 0.912 (tiny n/a) | n/a (no VAL conf) | n/a (no VAL conf) | 1,949 imgs / 0 neg |
| S2_B_s2 | c=0.728 R 0.402 (tiny n/a) | c=0.656 R 0.680 (tiny n/a) | c=0.559 R 0.876 (tiny n/a) | c=0.194 R 0.986 (tiny n/a) | c=0.044 R 0.999 (tiny n/a) | 1,949 imgs / 0 neg |
| S3_C | c=0.745 R 0.709 (tiny n/a) | c=0.691 R 0.856 (tiny n/a) | c=0.585 R 0.950 (tiny n/a) | c=0.072 R 1.000 (tiny n/a) | c=0.045 R 1.000 (tiny n/a) | 1,949 imgs / 0 neg |
| S4_D | c=0.758 R 0.153 (tiny n/a) | c=0.695 R 0.438 (tiny n/a) | c=0.578 R 0.799 (tiny n/a) | c=0.082 R 0.997 (tiny n/a) | c=0.036 R 0.999 (tiny n/a) | 1,949 imgs / 0 neg |
| S4b_D2 | c=0.765 R 0.158 (tiny n/a) | c=0.703 R 0.480 (tiny n/a) | c=0.572 R 0.837 (tiny n/a) | c=0.051 R 0.996 (tiny n/a) | c=0.024 R 0.996 (tiny n/a) | 1,949 imgs / 0 neg |
| S4c_D3 | c=0.723 R 0.614 (tiny n/a) | c=0.679 R 0.725 (tiny n/a) | c=0.564 R 0.889 (tiny n/a) | c=0.262 R 0.975 (tiny n/a) | c=0.199 R 0.986 (tiny n/a) | 1,949 imgs / 0 neg |
| S5_ensemble_v4a_s2 | c=0.982 R 0.981 (tiny n/a) | c=0.962 R 0.983 (tiny n/a) | c=0.905 R 0.983 (tiny n/a) | n/a (no VAL conf) | n/a (no VAL conf) | 1,949 imgs / 0 neg |
| v4B1_frozen_L | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 1,949 imgs / 0 neg |
| v4B4_lora_L | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 1,949 imgs / 0 neg |
| v4B6_lora_Hplus | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 1,949 imgs / 0 neg |
| v4a_s2 | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 1,949 imgs / 0 neg |

Sources:
- `reports/v5b/test/eval_S0_expE_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S0b_expB_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S1_A_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S1_A_s1_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S1_A_s2_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S2_B_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S2_B_s1_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S2_B_s2_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S3_C_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S4_D_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S4b_D2_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S4c_D3_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_S5_ensemble_v4a_s2_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_v4B1_frozen_L_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_v4B4_lora_L_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_v4B6_lora_Hplus_firemanholdout.raw.parquet`
- `reports/v5b/test/eval_v4a_s2_firemanholdout.raw.parquet`


---

### T8c. FINAL TEST H2 - H2 pyro-sdis TEST (ferion + croix-augas, 4,124 imgs / 917 negatives; second opening after v4): matched-FP recall / tiny / precision

One-shot evaluation ran at 2026-10-03T03:20:07Z
2026-10-03T03:48:01Z
2026-10-03T03:49:22Z
2026-10-03T03:49:43Z
2026-10-03T03:50:03Z (reports/v5b/test/.EVALUATED_ONCE). Deployment rule (PREREGISTRATION_V5B.md): replace the FAST default only if a candidate beats expB on H2 R@10% by >= 2 pt AND does not lose on H1 image-level recall AND passes integration tests. v4 comparison points: reports/v4/tables/T25_*.md (same split, same protocol).

| system | 1% FP | 2% FP | 5% FP | 10% FP | 20% FP | denominators |
|---|---|---|---|---|---|---|
| S0_expE | c=0.766 R 0.107 / tiny 0.034 / P 0.968 | c=0.727 R 0.172 / tiny 0.081 / P 0.957 | c=0.671 R 0.256 / tiny 0.154 / P 0.924 | c=0.599 R 0.350 / tiny 0.257 / P 0.895 | c=0.457 R 0.506 / tiny 0.406 / P 0.835 | 4,124 imgs / 917 neg |
| S0b_expB | c=0.759 R 0.118 / tiny 0.026 / P 0.963 | c=0.721 R 0.171 / tiny 0.052 / P 0.951 | c=0.655 R 0.258 / tiny 0.115 / P 0.923 | c=0.577 R 0.354 / tiny 0.206 / P 0.887 | c=0.409 R 0.501 / tiny 0.364 / P 0.819 | 4,124 imgs / 917 neg |
| S1_A | c=0.773 R 0.108 / tiny 0.035 / P 0.970 | c=0.723 R 0.183 / tiny 0.092 / P 0.963 | c=0.644 R 0.287 / tiny 0.184 / P 0.928 | c=0.555 R 0.388 / tiny 0.283 / P 0.893 | c=0.391 R 0.553 / tiny 0.435 / P 0.833 | 4,124 imgs / 917 neg |
| S1_A_s1 | c=0.778 R 0.091 / tiny 0.020 / P 0.966 | c=0.756 R 0.124 / tiny 0.040 / P 0.942 | c=0.675 R 0.261 / tiny 0.149 / P 0.927 | c=0.591 R 0.373 / tiny 0.253 / P 0.893 | c=0.452 R 0.521 / tiny 0.426 / P 0.837 | 4,124 imgs / 917 neg |
| S1_A_s2 | c=0.782 R 0.102 / tiny 0.026 / P 0.964 | c=0.736 R 0.172 / tiny 0.065 / P 0.952 | c=0.663 R 0.280 / tiny 0.155 / P 0.928 | c=0.571 R 0.383 / tiny 0.244 / P 0.890 | c=0.426 R 0.511 / tiny 0.384 / P 0.833 | 4,124 imgs / 917 neg |
| S2_B | c=0.776 R 0.089 / tiny 0.019 / P 0.960 | c=0.739 R 0.141 / tiny 0.051 / P 0.951 | c=0.676 R 0.221 / tiny 0.116 / P 0.915 | c=0.602 R 0.309 / tiny 0.199 / P 0.879 | c=0.474 R 0.442 / tiny 0.325 / P 0.821 | 4,124 imgs / 917 neg |
| S2_B_s1 | c=0.779 R 0.085 / tiny 0.017 / P 0.955 | c=0.739 R 0.145 / tiny 0.053 / P 0.951 | c=0.662 R 0.242 / tiny 0.126 / P 0.916 | c=0.567 R 0.349 / tiny 0.228 / P 0.875 | c=0.428 R 0.470 / tiny 0.349 / P 0.805 | 4,124 imgs / 917 neg |
| S2_B_s2 | c=0.766 R 0.091 / tiny 0.020 / P 0.955 | c=0.729 R 0.140 / tiny 0.045 / P 0.947 | c=0.654 R 0.229 / tiny 0.110 / P 0.912 | c=0.551 R 0.332 / tiny 0.213 / P 0.872 | c=0.388 R 0.471 / tiny 0.359 / P 0.790 | 4,124 imgs / 917 neg |
| S3_C | c=0.782 R 0.085 / tiny 0.020 / P 0.959 | c=0.742 R 0.144 / tiny 0.062 / P 0.943 | c=0.682 R 0.221 / tiny 0.135 / P 0.912 | c=0.596 R 0.318 / tiny 0.214 / P 0.875 | c=0.457 R 0.461 / tiny 0.365 / P 0.820 | 4,124 imgs / 917 neg |
| S4_D | c=0.786 R 0.089 / tiny 0.022 / P 0.962 | c=0.746 R 0.149 / tiny 0.070 / P 0.947 | c=0.682 R 0.228 / tiny 0.144 / P 0.919 | c=0.589 R 0.342 / tiny 0.249 / P 0.877 | c=0.457 R 0.462 / tiny 0.356 / P 0.823 | 4,124 imgs / 917 neg |
| S4b_D2 | c=0.776 R 0.116 / tiny 0.038 / P 0.962 | c=0.749 R 0.153 / tiny 0.067 / P 0.949 | c=0.669 R 0.255 / tiny 0.151 / P 0.920 | c=0.582 R 0.348 / tiny 0.245 / P 0.880 | c=0.429 R 0.482 / tiny 0.374 / P 0.809 | 4,124 imgs / 917 neg |
| S4c_D3 | c=0.744 R 0.096 / tiny 0.015 / P 0.954 | c=0.722 R 0.129 / tiny 0.037 / P 0.941 | c=0.628 R 0.260 / tiny 0.143 / P 0.908 | c=0.550 R 0.358 / tiny 0.250 / P 0.876 | c=0.451 R 0.466 / tiny 0.364 / P 0.802 | 4,124 imgs / 917 neg |
| S5_ensemble_v4a_s2 | n/a | n/a | c=0.969 R 0.234 / tiny 0.102 / P 0.914 | c=0.935 R 0.366 / tiny 0.202 / P 0.880 | c=0.854 R 0.552 / tiny 0.367 / P 0.819 | 4,124 imgs / 917 neg |
| v4B1_frozen_L | PENDING | PENDING | PENDING | PENDING | PENDING | - |
| v4B4_lora_L | PENDING | PENDING | PENDING | PENDING | PENDING | - |
| v4B6_lora_Hplus | PENDING | PENDING | PENDING | PENDING | PENDING | - |
| v4a_s2 | c=0.759 R 0.130 / tiny 0.054 / P 0.970 | c=0.732 R 0.170 / tiny 0.084 / P 0.959 | c=0.660 R 0.273 / tiny 0.175 / P 0.927 | c=0.564 R 0.385 / tiny 0.287 / P 0.890 | c=0.409 R 0.549 / tiny 0.441 / P 0.834 | 4,124 imgs / 917 neg |

Sources:
- `reports/v5b/test/eval_S0_expE_sdistest.raw.parquet`
- `reports/v5b/test/eval_S0b_expB_sdistest.raw.parquet`
- `reports/v5b/test/eval_S1_A_sdistest.raw.parquet`
- `reports/v5b/test/eval_S1_A_s1_sdistest.raw.parquet`
- `reports/v5b/test/eval_S1_A_s2_sdistest.raw.parquet`
- `reports/v5b/test/eval_S2_B_sdistest.raw.parquet`
- `reports/v5b/test/eval_S2_B_s1_sdistest.raw.parquet`
- `reports/v5b/test/eval_S2_B_s2_sdistest.raw.parquet`
- `reports/v5b/test/eval_S3_C_sdistest.raw.parquet`
- `reports/v5b/test/eval_S4_D_sdistest.raw.parquet`
- `reports/v5b/test/eval_S4b_D2_sdistest.raw.parquet`
- `reports/v5b/test/eval_S4c_D3_sdistest.raw.parquet`
- `reports/v5b/test/eval_S5_ensemble_v4a_s2_sdistest.raw.parquet`
- `reports/v5b/test/eval_v4a_s2_sdistest.raw.parquet`


---

### T8d. FINAL TEST H3 - H3 sheffield_holdout rec_0008 (327 frames, 0 negatives -> recall only; WITHIN-INCIDENT; CONTAMINATED for arms whose corpus holds Sheffield)

One-shot evaluation ran at 2026-10-03T03:20:07Z
2026-10-03T03:48:01Z
2026-10-03T03:49:22Z
2026-10-03T03:49:43Z
2026-10-03T03:50:03Z (reports/v5b/test/.EVALUATED_ONCE). Rows marked CONTAMINATED must not enter any comparison or decision (train/test overlap: all 327 frames are in the B/C/D train lists). Even clean rows are within-incident generalisation (same burn, site, 12-minute session as rec_0003-0007). Author boxes are fragmented sub-boxes -> box recall at IoU 0.4 is convention-limited (sheffield_uav_CARD.md section 7: 91% of author boxes are >=50% inside a predicted box while IoU>=0.4 recall is 0.085 for expE).

| system | R @ sdis-val 5% | R @ sdis-val 10% | R @ sdis-val 20% | R @ fireman-val 5% | R @ fireman-val 10% | denominators |
|---|---|---|---|---|---|---|
| S0_expE | c=0.721 R 0.008 (tiny n/a) | c=0.655 R 0.017 (tiny n/a) | c=0.535 R 0.073 (tiny n/a) | c=0.177 R 0.229 (tiny n/a) | c=0.070 R 0.270 (tiny n/a) | 327 imgs / 0 neg |
| S0b_expB | c=0.723 R 0.000 (tiny n/a) | c=0.650 R 0.006 (tiny n/a) | c=0.532 R 0.051 (tiny n/a) | c=0.144 R 0.206 (tiny n/a) | c=0.063 R 0.243 (tiny n/a) | 327 imgs / 0 neg |
| S1_A | c=0.729 R 0.018 (tiny n/a) | c=0.673 R 0.023 (tiny n/a) | c=0.534 R 0.070 (tiny n/a) | c=0.091 R 0.264 (tiny n/a) | c=0.030 R 0.299 (tiny n/a) | 327 imgs / 0 neg |
| S1_A_s1 | c=0.727 R 0.000 (tiny n/a) | c=0.666 R 0.013 (tiny n/a) | c=0.542 R 0.050 (tiny n/a) | c=0.236 R 0.186 (tiny n/a) | c=0.165 R 0.230 (tiny n/a) | 327 imgs / 0 neg |
| S1_A_s2 | c=0.717 R 0.005 (tiny n/a) | c=0.646 R 0.024 (tiny n/a) | c=0.500 R 0.096 (tiny n/a) | c=0.282 R 0.174 (tiny n/a) | c=0.179 R 0.226 (tiny n/a) | 327 imgs / 0 neg |
| S2_B **CONTAMINATED** | c=0.740 R 0.425 (tiny n/a) | c=0.672 R 0.504 (tiny n/a) | c=0.557 R 0.644 (tiny n/a) | c=0.132 R 0.939 (tiny n/a) | c=0.076 R 0.955 (tiny n/a) | 327 imgs / 0 neg |
| S2_B_s1 **CONTAMINATED** | c=0.728 R 0.374 (tiny n/a) | c=0.659 R 0.476 (tiny n/a) | c=0.528 R 0.665 (tiny n/a) | n/a (no VAL conf) | n/a (no VAL conf) | 327 imgs / 0 neg |
| S2_B_s2 **CONTAMINATED** | c=0.728 R 0.428 (tiny n/a) | c=0.656 R 0.511 (tiny n/a) | c=0.559 R 0.631 (tiny n/a) | c=0.194 R 0.877 (tiny n/a) | c=0.044 R 0.968 (tiny n/a) | 327 imgs / 0 neg |
| S3_C **CONTAMINATED** | c=0.745 R 0.372 (tiny n/a) | c=0.691 R 0.435 (tiny n/a) | c=0.585 R 0.598 (tiny n/a) | c=0.072 R 0.960 (tiny n/a) | c=0.045 R 0.973 (tiny n/a) | 327 imgs / 0 neg |
| S4_D **CONTAMINATED** | c=0.758 R 0.319 (tiny n/a) | c=0.695 R 0.457 (tiny n/a) | c=0.578 R 0.579 (tiny n/a) | c=0.082 R 0.949 (tiny n/a) | c=0.036 R 0.975 (tiny n/a) | 327 imgs / 0 neg |
| S4b_D2 **CONTAMINATED** | c=0.765 R 0.313 (tiny n/a) | c=0.703 R 0.410 (tiny n/a) | c=0.572 R 0.603 (tiny n/a) | c=0.051 R 0.962 (tiny n/a) | c=0.024 R 0.980 (tiny n/a) | 327 imgs / 0 neg |
| S4c_D3 **CONTAMINATED** | c=0.723 R 0.252 (tiny n/a) | c=0.679 R 0.324 (tiny n/a) | c=0.564 R 0.470 (tiny n/a) | c=0.262 R 0.792 (tiny n/a) | c=0.199 R 0.838 (tiny n/a) | 327 imgs / 0 neg |
| S5_ensemble_v4a_s2 | c=0.982 R 0.199 (tiny n/a) | c=0.962 R 0.212 (tiny n/a) | c=0.905 R 0.224 (tiny n/a) | n/a (no VAL conf) | n/a (no VAL conf) | 327 imgs / 0 neg |
| v4B1_frozen_L | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 327 imgs / 0 neg |
| v4B4_lora_L | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 327 imgs / 0 neg |
| v4B6_lora_Hplus | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 327 imgs / 0 neg |
| v4a_s2 | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 327 imgs / 0 neg |

Sources:
- `reports/v5b/test/eval_S0_expE_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S0b_expB_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S1_A_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S1_A_s1_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S1_A_s2_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S2_B_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S2_B_s1_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S2_B_s2_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S3_C_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S4_D_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S4b_D2_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S4c_D3_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_S5_ensemble_v4a_s2_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_v4B1_frozen_L_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_v4B4_lora_L_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_v4B6_lora_Hplus_sheffieldholdout.raw.parquet`
- `reports/v5b/test/eval_v4a_s2_sheffieldholdout.raw.parquet`


---

### T8e. FINAL TEST H4 - historical continuity: Boreal ruokolahti TEST and AuSmoke forest TEST at the pre-registered confidences

One-shot evaluation ran at 2026-10-03T03:20:07Z
2026-10-03T03:48:01Z
2026-10-03T03:49:22Z
2026-10-03T03:49:43Z
2026-10-03T03:50:03Z (reports/v5b/test/.EVALUATED_ONCE). Reported for continuity with v4 (reports/v4/tables/T26_*.md, T27_*.md). Ruokolahti has 2 negatives (FP unmeasurable); AuSmoke is positives-only and permission-pending -> numbers shown, not used for decisions. FIgLib-39 causal replay is a separate historical artifact (not produced by this script).

**H4 boreal ruokolahti TEST (1,764 imgs / 2 negatives -> FP unmeasurable; historical continuity)**

| system | R @ sdis-val 5% | R @ sdis-val 10% | R @ sdis-val 20% | R @ fireman-val 5% | R @ fireman-val 10% | denominators |
|---|---|---|---|---|---|---|
| S0_expE | c=0.721 R 0.720 (tiny n/a; imgFP 0.000) | c=0.655 R 0.877 (tiny n/a; imgFP 0.000) | c=0.535 R 0.917 (tiny n/a; imgFP 0.000) | c=0.177 R 0.929 (tiny n/a; imgFP 0.000) | c=0.070 R 0.931 (tiny n/a; imgFP 0.000) | 1,764 imgs / 2 neg |
| S0b_expB | c=0.723 R 0.782 (tiny n/a; imgFP 0.000) | c=0.650 R 0.886 (tiny n/a; imgFP 0.000) | c=0.532 R 0.918 (tiny n/a; imgFP 0.000) | c=0.144 R 0.931 (tiny n/a; imgFP 0.000) | c=0.063 R 0.933 (tiny n/a; imgFP 0.000) | 1,764 imgs / 2 neg |
| S1_A | c=0.729 R 0.838 (tiny n/a; imgFP 0.000) | c=0.673 R 0.906 (tiny n/a; imgFP 0.000) | c=0.534 R 0.923 (tiny n/a; imgFP 0.000) | c=0.091 R 0.931 (tiny n/a; imgFP 0.000) | c=0.030 R 0.933 (tiny n/a; imgFP 0.000) | 1,764 imgs / 2 neg |
| S1_A_s1 | c=0.727 R 0.538 (tiny n/a; imgFP 0.000) | c=0.666 R 0.806 (tiny n/a; imgFP 0.000) | c=0.542 R 0.918 (tiny n/a; imgFP 0.000) | c=0.236 R 0.930 (tiny n/a; imgFP 0.000) | c=0.165 R 0.933 (tiny n/a; imgFP 0.000) | 1,764 imgs / 2 neg |
| S1_A_s2 | c=0.717 R 0.830 (tiny n/a; imgFP 0.000) | c=0.646 R 0.911 (tiny n/a; imgFP 0.000) | c=0.500 R 0.925 (tiny n/a; imgFP 0.000) | c=0.282 R 0.929 (tiny n/a; imgFP 0.000) | c=0.179 R 0.930 (tiny n/a; imgFP 0.000) | 1,764 imgs / 2 neg |
| S2_B | c=0.740 R 0.863 (tiny n/a; imgFP 0.000) | c=0.672 R 0.890 (tiny n/a; imgFP 0.000) | c=0.557 R 0.913 (tiny n/a; imgFP 0.000) | c=0.132 R 0.929 (tiny n/a; imgFP 0.000) | c=0.076 R 0.931 (tiny n/a; imgFP 0.000) | 1,764 imgs / 2 neg |
| S2_B_s1 | c=0.728 R 0.839 (tiny n/a; imgFP 0.000) | c=0.659 R 0.862 (tiny n/a; imgFP 0.000) | c=0.528 R 0.892 (tiny n/a; imgFP 0.000) | n/a (no VAL conf) | n/a (no VAL conf) | 1,764 imgs / 2 neg |
| S2_B_s2 | c=0.728 R 0.860 (tiny n/a; imgFP 0.000) | c=0.656 R 0.882 (tiny n/a; imgFP 0.000) | c=0.559 R 0.897 (tiny n/a; imgFP 0.000) | c=0.194 R 0.931 (tiny n/a; imgFP 0.000) | c=0.044 R 0.935 (tiny n/a; imgFP 0.000) | 1,764 imgs / 2 neg |
| S3_C | c=0.745 R 0.842 (tiny n/a; imgFP 0.000) | c=0.691 R 0.868 (tiny n/a; imgFP 0.000) | c=0.585 R 0.887 (tiny n/a; imgFP 0.000) | c=0.072 R 0.934 (tiny n/a; imgFP 0.000) | c=0.045 R 0.935 (tiny n/a; imgFP 0.000) | 1,764 imgs / 2 neg |
| S4_D | c=0.758 R 0.869 (tiny n/a; imgFP 0.000) | c=0.695 R 0.894 (tiny n/a; imgFP 0.000) | c=0.578 R 0.914 (tiny n/a; imgFP 0.000) | c=0.082 R 0.932 (tiny n/a; imgFP 0.000) | c=0.036 R 0.934 (tiny n/a; imgFP 0.000) | 1,764 imgs / 2 neg |
| S4b_D2 | c=0.765 R 0.856 (tiny n/a; imgFP 0.000) | c=0.703 R 0.888 (tiny n/a; imgFP 0.000) | c=0.572 R 0.911 (tiny n/a; imgFP 0.000) | c=0.051 R 0.934 (tiny n/a; imgFP 0.000) | c=0.024 R 0.934 (tiny n/a; imgFP 0.000) | 1,764 imgs / 2 neg |
| S4c_D3 | c=0.723 R 0.856 (tiny n/a; imgFP 0.000) | c=0.679 R 0.878 (tiny n/a; imgFP 0.000) | c=0.564 R 0.906 (tiny n/a; imgFP 0.000) | c=0.262 R 0.931 (tiny n/a; imgFP 0.000) | c=0.199 R 0.934 (tiny n/a; imgFP 0.000) | 1,764 imgs / 2 neg |
| S5_ensemble_v4a_s2 | c=0.982 R 0.929 (tiny n/a; imgFP 0.000) | c=0.962 R 0.930 (tiny n/a; imgFP 0.000) | c=0.905 R 0.931 (tiny n/a; imgFP 0.000) | n/a (no VAL conf) | n/a (no VAL conf) | 1,764 imgs / 2 neg |
| v4B1_frozen_L | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| v4B4_lora_L | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| v4B6_lora_Hplus | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| v4a_s2 | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 1,764 imgs / 2 neg |

**H4 AuSmoke forest TEST (2,871 imgs, positives only; permission pending -> shown, not used for decisions)**

| system | R @ sdis-val 5% | R @ sdis-val 10% | R @ sdis-val 20% | R @ fireman-val 5% | R @ fireman-val 10% | denominators |
|---|---|---|---|---|---|---|
| S0_expE | c=0.721 R 0.009 (tiny 0.023) | c=0.655 R 0.033 (tiny 0.048) | c=0.535 R 0.076 (tiny 0.052) | c=0.177 R 0.211 (tiny 0.112) | c=0.070 R 0.280 (tiny 0.155) | 2,871 imgs / 0 neg |
| S0b_expB | c=0.723 R 0.001 (tiny 0.000) | c=0.650 R 0.021 (tiny 0.001) | c=0.532 R 0.084 (tiny 0.029) | c=0.144 R 0.246 (tiny 0.116) | c=0.063 R 0.301 (tiny 0.157) | 2,871 imgs / 0 neg |
| S1_A | c=0.729 R 0.008 (tiny 0.016) | c=0.673 R 0.030 (tiny 0.047) | c=0.534 R 0.079 (tiny 0.056) | c=0.091 R 0.245 (tiny 0.130) | c=0.030 R 0.309 (tiny 0.171) | 2,871 imgs / 0 neg |
| S1_A_s1 | c=0.727 R 0.001 (tiny 0.001) | c=0.666 R 0.019 (tiny 0.022) | c=0.542 R 0.077 (tiny 0.056) | c=0.236 R 0.197 (tiny 0.110) | c=0.165 R 0.241 (tiny 0.130) | 2,871 imgs / 0 neg |
| S1_A_s2 | c=0.717 R 0.010 (tiny 0.017) | c=0.646 R 0.041 (tiny 0.047) | c=0.500 R 0.094 (tiny 0.053) | c=0.282 R 0.185 (tiny 0.093) | c=0.179 R 0.241 (tiny 0.119) | 2,871 imgs / 0 neg |
| S2_B | c=0.740 R 0.005 (tiny 0.001) | c=0.672 R 0.041 (tiny 0.035) | c=0.557 R 0.098 (tiny 0.067) | c=0.132 R 0.247 (tiny 0.147) | c=0.076 R 0.273 (tiny 0.159) | 2,871 imgs / 0 neg |
| S2_B_s1 | c=0.728 R 0.005 (tiny 0.001) | c=0.659 R 0.027 (tiny 0.006) | c=0.528 R 0.093 (tiny 0.056) | n/a (no VAL conf) | n/a (no VAL conf) | 2,871 imgs / 0 neg |
| S2_B_s2 | c=0.728 R 0.003 (tiny 0.001) | c=0.656 R 0.022 (tiny 0.005) | c=0.559 R 0.079 (tiny 0.050) | c=0.194 R 0.217 (tiny 0.107) | c=0.044 R 0.313 (tiny 0.174) | 2,871 imgs / 0 neg |
| S3_C | c=0.745 R 0.002 (tiny 0.001) | c=0.691 R 0.016 (tiny 0.007) | c=0.585 R 0.081 (tiny 0.054) | c=0.072 R 0.279 (tiny 0.160) | c=0.045 R 0.301 (tiny 0.169) | 2,871 imgs / 0 neg |
| S4_D | c=0.758 R 0.001 (tiny 0.001) | c=0.695 R 0.011 (tiny 0.005) | c=0.578 R 0.072 (tiny 0.056) | c=0.082 R 0.284 (tiny 0.131) | c=0.036 R 0.339 (tiny 0.160) | 2,871 imgs / 0 neg |
| S4b_D2 | c=0.765 R 0.005 (tiny 0.002) | c=0.703 R 0.035 (tiny 0.051) | c=0.572 R 0.090 (tiny 0.061) | c=0.051 R 0.307 (tiny 0.179) | c=0.024 R 0.340 (tiny 0.198) | 2,871 imgs / 0 neg |
| S4c_D3 | c=0.723 R 0.003 (tiny 0.001) | c=0.679 R 0.012 (tiny 0.001) | c=0.564 R 0.080 (tiny 0.055) | c=0.262 R 0.184 (tiny 0.098) | c=0.199 R 0.209 (tiny 0.116) | 2,871 imgs / 0 neg |
| S5_ensemble_v4a_s2 | c=0.982 R 0.141 (tiny 0.047) | c=0.962 R 0.203 (tiny 0.079) | c=0.905 R 0.308 (tiny 0.148) | n/a (no VAL conf) | n/a (no VAL conf) | 2,871 imgs / 0 neg |
| v4B1_frozen_L | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| v4B4_lora_L | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| v4B6_lora_Hplus | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| v4a_s2 | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | n/a (no VAL conf) | 2,871 imgs / 0 neg |

Sources:
- `reports/v5b/test/eval_S0_expE_borealtest.raw.parquet`
- `reports/v5b/test/eval_S0b_expB_borealtest.raw.parquet`
- `reports/v5b/test/eval_S1_A_borealtest.raw.parquet`
- `reports/v5b/test/eval_S1_A_s1_borealtest.raw.parquet`
- `reports/v5b/test/eval_S1_A_s2_borealtest.raw.parquet`
- `reports/v5b/test/eval_S2_B_borealtest.raw.parquet`
- `reports/v5b/test/eval_S2_B_s1_borealtest.raw.parquet`
- `reports/v5b/test/eval_S2_B_s2_borealtest.raw.parquet`
- `reports/v5b/test/eval_S3_C_borealtest.raw.parquet`
- `reports/v5b/test/eval_S4_D_borealtest.raw.parquet`
- `reports/v5b/test/eval_S4b_D2_borealtest.raw.parquet`
- `reports/v5b/test/eval_S4c_D3_borealtest.raw.parquet`
- `reports/v5b/test/eval_S5_ensemble_v4a_s2_borealtest.raw.parquet`
- `reports/v5b/test/eval_v4a_s2_borealtest.raw.parquet`
- `reports/v5b/test/eval_S0_expE_foresttest.raw.parquet`
- `reports/v5b/test/eval_S0b_expB_foresttest.raw.parquet`
- `reports/v5b/test/eval_S1_A_foresttest.raw.parquet`
- `reports/v5b/test/eval_S1_A_s1_foresttest.raw.parquet`
- `reports/v5b/test/eval_S1_A_s2_foresttest.raw.parquet`
- `reports/v5b/test/eval_S2_B_foresttest.raw.parquet`
- `reports/v5b/test/eval_S2_B_s1_foresttest.raw.parquet`
- `reports/v5b/test/eval_S2_B_s2_foresttest.raw.parquet`
- `reports/v5b/test/eval_S3_C_foresttest.raw.parquet`
- `reports/v5b/test/eval_S4_D_foresttest.raw.parquet`
- `reports/v5b/test/eval_S4b_D2_foresttest.raw.parquet`
- `reports/v5b/test/eval_S4c_D3_foresttest.raw.parquet`
- `reports/v5b/test/eval_S5_ensemble_v4a_s2_foresttest.raw.parquet`
- `reports/v5b/test/eval_v4a_s2_foresttest.raw.parquet`


---

### T9a. Compute by iteration and GPU type (scheduler-derived allocated hours; whole-node accounting)

Allocated node-hours = scheduler completion - start from the kraken text table (queue wait excluded); jobs RUNNING at the dump are truncated at 2026-10-03T03:54:35+00:00. GPU-hours = node-hours x GPUs/node from the nodepool (g6e = 8x L40S, p6-b200 = 8x B200). Iteration 5b = jobs CREATED >= 2026-10-02T07:26:24+00:00 (cluster/v5b/BUDGET_V5B.json window_start); iteration 5 = created >= 2026-10-02T00:35:00+00:00; 3/4 by allocation start as in v4; 1 vs 2 by job-name pattern. 434 job records across projects ['green', 'obsidian', 'obsidian-pu']; 12 external (other project) rows excluded. Presigned URLs/tokens were scrubbed from the dumps in memory before writing (<scrubbed-url>). No pod utilization telemetry exists.

| iteration | GPU type | jobs | node-h | GPU-h | open at dump |
|---|---|---|---|---|---|
| 1 | B200 | 26 | 4.54 | 36.3 | 0 |
| 1 | H200 | 1 | 0.05 | 0.4 | 0 |
| 1 | L4 | 4 | 0.27 | 1.3 | 0 |
| 1 | L40S | 52 | 35.26 | 282.2 | 0 |
| 2 | L40S | 18 | 10.05 | 80.4 | 0 |
| 3 | L40S | 63 | 20.62 | 164.9 | 0 |
| 4 | A100 | 7 | 2.47 | 19.7 | 0 |
| 4 | B200 | 5 | 22.48 | 179.9 | 0 |
| 4 | CPU-node | 5 | 0.98 | 0.0 | 0 |
| 4 | H200 | 12 | 2.38 | 19.1 | 0 |
| 4 | L40S | 145 | 103.01 | 823.9 | 0 |
| 5 | L40S | 2 | 0.28 | 2.3 | 0 |
| 5b | B200 | 4 | 5.97 | 47.7 | 0 |
| 5b | L40S | 78 | 55.30 | 442.4 | 0 |

Sources:
- `reports/v5b/COMPUTE_LEDGER_V5B.csv`
- `reports/v5b/ledger/kraken_{green,obsidian,obsidian-pu}_{raw.json,table.txt,dump.meta}`
- `cluster/v5b/BUDGET_V5B.json`
- `cluster/v5/BUDGET_V5.json`


---

### T9b. Iteration 5b compute by GPU type, track, kind and arm (scheduler allocation) with the reservation-registry cross-check

82 iteration-5b jobs at the snapshot (2026-10-03T03:54:35+00:00; 0 still open -> truncated). B200: 5.97 node-h = 47.7 GPU-h; L40S: 55.30 node-h = 442.4 GPU-h. Window cap: 384 node-h / 3072 GPU-h total (max concurrent {'L40S': 15, 'B200': 1}). Registry (cluster/v5b/registry.json, reserve->release wall-clock, includes queue wait): B200 6.12 node-h, L40S 58.11 node-h. The two differ by queue wait / pod start and by segments whose reservation outlives the job.

**by track**

| track | GPU | jobs | node-h | GPU-h | failed node-h |
|---|---|---|---|---|---|
| yolo-matrix | L40S | 50 | 44.03 | 352.2 | 2.98 |
| eval | L40S | 7 | 6.67 | 53.3 | 5.00 |
| b200-dino | B200 | 1 | 3.25 | 26.0 | 0.00 |
| teacher | L40S | 11 | 2.33 | 18.6 | 0.00 |
| yolo-train | L40S | 7 | 1.93 | 15.5 | 0.00 |
| yolo-train | B200 | 1 | 1.25 | 10.0 | 0.00 |
| b200-qwen | B200 | 1 | 0.82 | 6.5 | 0.00 |
| infra-probe | B200 | 1 | 0.65 | 5.2 | 0.00 |
| seg | L40S | 3 | 0.33 | 2.7 | 0.20 |

**by kind**

| kind | GPU | jobs | node-h | GPU-h | failed node-h |
|---|---|---|---|---|---|
| training | L40S | 56 | 43.12 | 344.9 | 0.00 |
| evaluation | L40S | 7 | 6.67 | 53.3 | 5.00 |
| training | B200 | 3 | 5.32 | 42.5 | 0.00 |
| training(failed/retry) | L40S | 4 | 3.18 | 25.5 | 3.18 |
| inference(teacher pool) | L40S | 11 | 2.33 | 18.6 | 0.00 |
| staging/probe | B200 | 1 | 0.65 | 5.2 | 0.00 |

**by arm (yolo-matrix only)**

| arm | GPU | jobs | node-h | GPU-h | failed node-h |
|---|---|---|---|---|---|
| A | L40S | 15 | 13.20 | 105.6 | 0.00 |
| B | L40S | 15 | 13.13 | 105.1 | 1.00 |
| C | L40S | 5 | 4.38 | 35.1 | 0.98 |
| D | L40S | 5 | 4.50 | 36.0 | 1.00 |
| D2 | L40S | 5 | 4.33 | 34.7 | 0.00 |
| D3 | L40S | 5 | 4.48 | 35.9 | 0.00 |

Sources:
- `reports/v5b/COMPUTE_LEDGER_V5B.csv`
- `cluster/v5b/registry.json`
- `cluster/v5b/BUDGET_V5B.json`


---

### T9c. Iteration 5b jobs that did not end SUCCEEDED (FAILED segments, STOPPED probes, RUNNING at dump)

FAILED yolo-matrix segments are the seg-1 presigned-upload-expiry incident (STATUS 10:40 UTC): training resumed from /scratch last.pt, so their node-hours are real training time, not lost work.

| job | track | arm | kind | GPU | outcome | created | alloc start | alloc end | node-h |
|---|---|---|---|---|---|---|---|---|---|
| tianrgua-sgsegv5bas1-69kq | seg |  | training(failed/retry) | L40S | FAILED | 10-02T07:44 | 10-02T07:44 | 10-02T07:56 | 0.20 |
| tianrgua-v5bb200p1-gfi3 | infra-probe |  | staging/probe | B200 | STOPPED | 10-02T07:48 | 10-02T07:48 | 10-02T08:27 | 0.65 |
| tianrgua-v5bbs0s1-rb6i | yolo-matrix | B | training(failed/retry) | L40S | FAILED | 10-02T09:01 | 10-02T09:01 | 10-02T10:01 | 1.00 |
| tianrgua-v5bds0s1-ftbh | yolo-matrix | D | training(failed/retry) | L40S | FAILED | 10-02T09:02 | 10-02T09:02 | 10-02T10:02 | 1.00 |
| tianrgua-v5bcs0s1-bh5g | yolo-matrix | C | training(failed/retry) | L40S | FAILED | 10-02T09:04 | 10-02T09:04 | 10-02T10:03 | 0.98 |
| tianrgua-v5bdinohr-mssz | b200-dino |  | training | B200 | STOPPED | 10-02T09:12 | 10-02T09:12 | 10-02T12:27 | 3.25 |
| tianrgua-v5bfinal2-eu8w | eval |  | evaluation | L40S | FAILED | 10-03T00:12 | 10-03T00:12 | 10-03T02:43 | 2.52 |
| tianrgua-v5bfinal1-6rb6 | eval |  | evaluation | L40S | FAILED | 10-03T00:12 | 10-03T00:12 | 10-03T02:41 | 2.48 |

Sources:
- `reports/v5b/COMPUTE_LEDGER_V5B.csv`
