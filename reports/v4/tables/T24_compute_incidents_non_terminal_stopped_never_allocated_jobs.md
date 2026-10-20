_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T24. Compute incidents: non-terminal / stopped / never-allocated jobs, and iteration-4 FAILED jobs by track

Top: every ledger row whose outcome is neither SUCCEEDED nor FAILED (5 rows; QUEUED(never allocated) = idle allocation request with 0 hours; STOPPED = manually killed; RUNNING(open at dump) = truncated at the snapshot). Bottom: iteration-4 FAILED jobs (24 jobs, 3.80 node-h) — the arm-D DDP crash loop (14 segments, exit 101, ~1.2 node-h per STATUS.md) sits in yolo-matrix.

| job | iter | track | kind | outcome | created | alloc start | alloc end | node-h |
|---|---|---|---|---|---|---|---|---|
| tianrgua-sv4-dgnx | 1 | infra-probe | staging/probe | STOPPED(never allocated) | 2026-09-11T07:16:49+00:00 | 2026-09-11T07:16:00+00:00 | 2026-09-11T07:16:00+00:00 | 0.00 |
| tianrgua-rfdetrlsanitya1-i2ki | 3 | rfdetr/deimv2 | training | STOPPED | 2026-09-13T09:01:58+00:00 | 2026-09-13T09:01:00+00:00 | 2026-09-13T09:13:00+00:00 | 0.20 |
| tianrgua-d3c17bheadcas2-zfpu | 3 | dino-v3 | training | STOPPED | 2026-09-14T03:17:53+00:00 | 2026-09-14T03:17:00+00:00 | 2026-09-14T03:32:00+00:00 | 0.25 |
| tianrgua-d3b1cachednos1-pubh | 3 | dino-v3 | training | STOPPED | 2026-09-14T03:18:48+00:00 | 2026-09-14T03:18:00+00:00 | 2026-09-14T03:31:00+00:00 | 0.22 |
| probe-bindle-cqbg | external | external(other project) | external | QUEUED(never allocated) | 2026-09-14T23:44:43+00:00 | 2026-09-14T23:44:43+00:00 | 2026-09-14T23:44:43+00:00 | 0.00 |

**iteration-4 FAILED jobs by track**

| track | failed jobs | node-h |
|---|---|---|
| yolo-matrix | 19 | 3.45 |
| eval | 3 | 0.32 |
| yolo-train | 2 | 0.03 |

Sources:
- `reports/v4/COMPUTE_LEDGER.csv`
