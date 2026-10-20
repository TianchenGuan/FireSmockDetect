_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

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
