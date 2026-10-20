_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T22. Compute accounting by iteration (scheduler-derived allocated hours; whole-node accounting)

Allocated node-hours = scheduler completion - start (queue wait excluded); GPU-hours = node-hours x GPUs/node. No pod utilization telemetry exists (utilization column = n/a). Iterations 3/4 attributed by exact window start; 1 vs 2 by job-name pattern (approx). 316 job records (12 flagged in_scope=False: same requester, other project, excluded from every iteration total); ledger snapshot = 2026-09-15T03:32:00+00:00 (0 job(s) still open at dump). Iteration-4 ceiling: 384 node-h / 3072 GPU-h (cluster/v4/BUDGET_V4.json). 'failed/retry' = FAILED outcome or training(failed/retry) kind.

| iteration | jobs | node-h | GPU-h | node-h by GPU type | failed/retry node-h | attribution |
|---|---|---|---|---|---|---|
| 1 | 83 | 40.1 | 320 | L40S 35.3, B200 4.5, L4 0.3, H200 0.1 | 9.3 (39 jobs) | name-based (approx) |
| 2 | 18 | 10.1 | 80 | L40S 10.1 | 1.6 (1 jobs) | name-based (approx) |
| 3 | 63 | 20.6 | 165 | L40S 20.6 | 0.1 (6 jobs) | time-window (exact) |
| 4 | 140 | 80.3 | 642 | L40S 80.3 | 3.8 (24 jobs) | time-window (exact) |
| ALL in-scope | 304 | 151.0 | 1207 |  | 14.8 (70 jobs) |  |
| external (other project; excluded) | 12 | 0.6 | 5 | B200 | - | name-prefix: probe-bindle |

Sources:
- `reports/v4/COMPUTE_LEDGER.csv`
- `cluster/v4/BUDGET_V4.json`
- `reports/v4/ledger/kraken_*_{raw.json,table.txt} (raw dumps)`
