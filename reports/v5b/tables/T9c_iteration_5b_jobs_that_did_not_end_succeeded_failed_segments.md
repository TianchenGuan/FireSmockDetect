_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

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
