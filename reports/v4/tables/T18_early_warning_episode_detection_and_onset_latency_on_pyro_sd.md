_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T18. Early warning: episode detection and onset latency on pyro-sdis VAL, plus the deployed discrete operating point

episodes = 469 positive smoke episodes in sdis_val sequences; latency in frames (29 s cadence) from episode start to first alert, median over detected episodes. Thresholds matched on frame FPR over the 1,158 all-negative frames. Deployed point = 0.30 single / 0.15 seq, 3-of-5.

| system | episodes | det@FPR1% | median latency (frames) | det@FPR5% | median latency (frames) |
|---|---|---|---|---|---|
| s1_expE | 469 | 0.1748 | 4.0 | 0.4606 | 1.5 |
| s1_v4a_ctrlE | 469 | 0.1898 | 4.0 | 0.4840 | 1.0 |
| s1_v4B1_dinoL | 469 | 0.1663 | 4.5 | 0.3348 | 2.0 |
| s2_persist_3of5 | 469 | 0.1173 | 6.0 | 0.2665 | 3.0 |
| t1_v3only_t1only_s0 | 469 | 0.1343 | 4.0 | 0.3795 | 2.0 |
| t1_v3only_t1only_s1 | 469 | 0.1429 | 4.0 | 0.4136 | 1.0 |
| t1_v4_t1only_s0 | 469 | 0.0981 | 6.5 | 0.4478 | 1.0 |
| t1_v4_t1only_s1 | 469 | 0.1855 | 5.0 | 0.4328 | 2.0 |
| t1_v4cap_t1only_s0 | 469 | 0.1620 | 3.5 | 0.4200 | 1.0 |
| t1_v4cap_t1only_s1 | 469 | 0.1322 | 4.0 | 0.4009 | 2.0 |
| tmp_v3only_temporal_s0 | 469 | 0.1322 | 6.0 | 0.3710 | 2.0 |
| tmp_v3only_temporal_s1 | 469 | 0.1514 | 5.0 | 0.3731 | 2.0 |
| tmp_v4_temporal_s0 | 469 | 0.1087 | 6.0 | 0.3454 | 2.0 |
| tmp_v4_temporal_s1 | 469 | 0.1620 | 5.0 | 0.4136 | 2.0 |
| tmp_v4cap_temporal_s0 | 469 | 0.1023 | 7.0 | 0.3667 | 2.0 |
| tmp_v4cap_temporal_s1 | 469 | 0.1322 | 6.0 | 0.3646 | 2.0 |

**Deployed discrete operating point (expE)**

| point | FPR all-neg | FPR clean-neg | recall pos frames | recall tiny frames |
|---|---|---|---|---|
| confirmed_only(s2>=0.15) | 0.5345 | 0.2768 | 0.8342 | 0.8383 |
| confirmed_or_provisional(s2>=0.15|s1>=0.30) | 0.5872 | 0.3589 | 0.8982 | 0.9043 |

Sources:
- `reports/v4/trackd_v4_eval.json (A_sdis_val_frame_level)`
