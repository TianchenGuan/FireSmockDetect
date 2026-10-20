_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T13. Batch-1 latency (1x L40S fp32 unless stated; n=50 sdis_val frames; preprocessing included, JPEG decode excluded)

v3 protocol. Merged form folds LoRA into the base linears (deploy form; outputs identical to 1e-6). expE YOLO11m reference on the same protocol: see source text (11.2 ms). Last row: devbox CPU (192 vCPU host, torch 8 threads) from the serving-backend verification.

| system | form | mean ms | median ms | p90 ms | infer-only median ms |
|---|---|---|---|---|---|
| v4B4_lora_L | LoRA unmerged (train form) | 161.8 | 161.6 | 162.9 | 155.8 |
| v4B4_lora_L | LoRA merged into base (deploy form) | 157.5 | 157.6 | 158.7 | 151.7 |
| v4B6_lora_Hplus | LoRA unmerged (train form) | 357.2 | 357.6 | 359.5 | 353.5 |
| v4B6_lora_Hplus | LoRA merged into base (deploy form) | 351.6 | 351.4 | 354.1 | 346.8 |
| v3 B1 frozen ViT-L (reference) | plain | 156.6 | 156.4 | 158.2 | 150.8 |
| v3 B2 frozen ViT-H+ (reference) | plain | 372.7 | 371.2 | 383.5 | 366.3 |
| v4B4_lora_L (serving backend `dino3`, CPU 8 threads) | LoRA unmerged | 5633.6 | 5616.6 | 5714.1 | 5612.1 |

Sources:
- `reports/v4/dino_adapt_results.md (Batch-1 latency table)`
- `reports/v4/dino3_backend_latency_B4_8thr.json`
