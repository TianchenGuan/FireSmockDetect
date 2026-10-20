_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T12. DINOv3 runs: adaptation mode, parameter counts, training cost

Parameter counts from checkpoints + HF config; node-hours = orchestrator-observed segment wall time on 8x L40S (incl. pod start and eval passes). best.pt selected by COCO mAP on a 1,500-image sdis-val subset. B5 trains at bs 2/GPU (2x the steps) and evaluates only at the end -> confounded vs B1/B4/B6 (see source).

| run | backbone | adaptation | backbone trainable / total | neck | head | ALL trainable | node-h (8xL40S) | epochs (exposures) | best COCO mAP @step | OOM/retries |
|---|---|---|---|---|---|---|---|---|---|---|
| v4B1_frozen_L | dinov3-vitl16 | frozen (head+neck only) | 0 / 303,129,600 | 7,805,184 | 14,499,865 | 22,305,049 | 1.66 | 10.00 (514,240) | 0.231 @10744 | none |
| v4B4_lora_L | dinov3-vitl16 | LoRA r16 a32 on q/k/v/o | 3,145,728 / 303,129,600 | 7,805,184 | 14,499,865 | 25,450,777 | 3.84 | 10.00 (514,240) | 0.409 @15026 | none |
| v4B5_full_L | dinov3-vitl16 | all blocks trainable | 303,128,576 / 303,129,600 | 7,805,184 | 14,499,865 | 325,433,625 | 8.35 | 10.00 (514,240) | 0.403 @32140 | none |
| v4B6_lora_Hplus | dinov3-vith16plus | LoRA r16 a32 on q/k/v/o | 5,242,880 / 840,592,640 | 10,640,192 | 14,499,865 | 30,382,937 | 8.02 | 10.00 (514,240) | 0.422 @8428 | none |

Sources:
- `reports/v4/dino_adapt_results.md (tables 'Parameter counts', 'Training outcome / cost', 'Runs')`
