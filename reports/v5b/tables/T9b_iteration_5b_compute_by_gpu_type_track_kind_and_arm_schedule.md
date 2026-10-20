_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

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
