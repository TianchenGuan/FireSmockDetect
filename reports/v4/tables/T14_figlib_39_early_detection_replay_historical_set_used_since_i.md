_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T14. FIgLib-39 early-detection replay (HISTORICAL set, used since iteration 1 — not a lockbox)

n_events = 39 HPWREN fire events (~80 frames each, 60 s cadence, 40 min before/after first visible smoke). detected = any alert at/after first-visible smoke OR a pre-visible alert that persists; pre-visible = alert before the labelled first-visible frame (unverifiable: early smoke or false alarm; NOT counted as false). R@N min = events with 0 <= delay <= N min / n_events (pre-visible alerts enter at delay 0 s — the same rule as scripts/v4/figlib_replay_report.py). YOLO systems use the deployed 3-of-5 policy at 0.30 / 0.15; DINO runs use their own sdis-val confs at 10% / 20% neg-FP (the v3 threshold-transfer rule). Median/mean delay over detected events. 'TEST ckpt' rows are the exact checkpoints evaluated on the sealed test sets (median seeds; STATUS.md 03:20Z); seed0 rows are the earlier iteration-4 replays.

| system | conf-single / conf-seq | detected | pre-visible | R@1 min | R@3 min | R@5 min | median delay | mean delay | misses |
|---|---|---|---|---|---|---|---|---|---|
| S0 expE_iter2 (deployed thresholds; paired-delay reference) | 0.3 / 0.15 | 30/39 | 5 | 7/39 = 0.179 | 12/39 = 0.308 | 17/39 = 0.436 | 240 s | 404 s | 9 |
| S1 A_ctrl TEST ckpt (v4a_s2) | 0.3 / 0.15 | 30/39 | 6 | 9/39 = 0.231 | 14/39 = 0.359 | 17/39 = 0.436 | 240 s | 456 s | 9 |
| S2 E_ET TEST ckpt (v4e_s1) | 0.3 / 0.15 | 32/39 | 12 | 14/39 = 0.359 | 15/39 = 0.385 | 20/39 = 0.513 | 240 s | 367 s | 7 |
| S2c C_EGT TEST ckpt (v4c_s1) | 0.3 / 0.15 | 32/39 | 14 | 17/39 = 0.436 | 18/39 = 0.462 | 21/39 = 0.538 | 60 s | 364 s | 7 |
| S3 D_curr TEST ckpt (v4dada_s1, post-hoc) | 0.3 / 0.15 | 32/39 | 11 | 13/39 = 0.333 | 15/39 = 0.385 | 20/39 = 0.513 | 240 s | 476 s | 7 |
| A_ctrl_E seed0 (v4a) | 0.3 / 0.15 | 30/39 | 7 | 10/39 = 0.256 | 13/39 = 0.333 | 17/39 = 0.436 | 240 s | 394 s | 9 |
| E_ET seed0 (v4e) | 0.3 / 0.15 | 28/39 | 9 | 11/39 = 0.282 | 12/39 = 0.308 | 18/39 = 0.462 | 240 s | 353 s | 11 |
| C_EGT seed0 (v4c) | 0.3 / 0.15 | 31/39 | 14 | 15/39 = 0.385 | 17/39 = 0.436 | 22/39 = 0.564 | 180 s | 313 s | 8 |
| S4p v4B4_lora_L (DINO, sdis-val-selected confs) | 0.9881 / 0.973 | 26/39 | 0 | 1/39 = 0.026 | 3/39 = 0.077 | 5/39 = 0.128 | 481 s | 667 s | 13 |
| S4 v4B6_lora_Hplus (DINO, sdis-val-selected confs) | 0.9901 / 0.9801 | 24/39 | 0 | 1/39 = 0.026 | 3/39 = 0.077 | 6/39 = 0.154 | 539 s | 705 s | 15 |
| v3 B1 frozen ViT-L (iteration-3 reference) | 0.5624 / 0.3454 | 34/39 | 4 | 6/39 = 0.154 | 15/39 = 0.385 | 22/39 = 0.564 | 240 s | 455 s | 5 |

Sources:
- `reports/replay_expE_figlib.json`
- `reports/v4/replay/replay_v4a_s2_figlib39.json`
- `reports/v4/replay/replay_v4e_s1_figlib39.json`
- `reports/v4/replay/replay_v4c_s1_figlib39.json`
- `reports/v4/replay/replay_v4dada_s1_figlib39.json`
- `reports/v4/replay/replay_v4a_figlib39.json`
- `reports/v4/replay/replay_v4e_figlib39.json`
- `reports/v4/replay/replay_v4c_figlib39.json`
- `reports/v4/replay_v4B4_lora_L_figlib39.json`
- `reports/v4/replay_v4B6_lora_Hplus_figlib39.json`
- `reports/v3/replay_B1_figlib_valmatched.json`
