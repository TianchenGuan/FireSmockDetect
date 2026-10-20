_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

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
