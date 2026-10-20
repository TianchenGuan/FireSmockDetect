_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T23. Iteration-4 compute by kind and by track

140 in-scope iteration-4 jobs in the ledger snapshot (2026-09-15T03:32:00+00:00); 80.3 node-h = 642 GPU-h of the 3072 GPU-h ceiling (21%). External same-requester jobs in the window are excluded (see T22). No job was open at the dump. 

**by kind**

| kind | jobs | node-h | GPU-h | share of v4 node-h |
|---|---|---|---|---|
| training | 106 | 74.3 | 595 | 93% |
| training(failed/retry) | 22 | 3.5 | 28 | 4% |
| evaluation | 11 | 2.2 | 18 | 3% |
| feature-extraction | 1 | 0.2 | 2 | 0% |
| ALL v4 | 140 | 80.3 | 642 | 21% of cap |

**by track**

| track | jobs | node-h | GPU-h | failed/retry node-h |
|---|---|---|---|---|
| yolo-matrix | 78 | 53.6 | 429 | 3.5 |
| dino-adapt | 29 | 21.7 | 173 | 0.0 |
| eval | 15 | 2.5 | 20 | 0.3 |
| yolo-train | 15 | 1.7 | 14 | 0.0 |
| temporal | 3 | 0.7 | 6 | 0.0 |

Sources:
- `reports/v4/COMPUTE_LEDGER.csv`
- `cluster/v4/BUDGET_V4.json`
