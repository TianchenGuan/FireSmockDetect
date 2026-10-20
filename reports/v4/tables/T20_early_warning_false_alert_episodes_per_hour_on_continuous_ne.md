_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T20. Early warning: false-alert EPISODES per hour on continuous negative footage

Episode = maximal run of alert-on frames; runs separated by < 60.0 s (deployed cooldown) merge. CI = exact Poisson 95%. Pools are tiny (minutes of footage) — read as order-of-magnitude.

**pyrosdis_val_all_neg_sequences** — 12 sequences / 219 frames / 1.521 h wall clock. pyro-sdis val sequences whose every frame is negative; real 29 s-cadence wall clock; expE never saw the val sites

| system | thr@FPR1% | false episodes/h @FPR1% | CI95 | false episodes/h @FPR5% | CI95 |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| s1_v4a_ctrlE | 0.8000 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| s1_v4B1_dinoL | 0.9858 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| s2_persist_3of5 | 0.7874 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| t1_v3only_t1only_s0 | 0.9367 | 0.00 | [0.0, 2.42] | 0.66 | [0.02, 3.66] |
| t1_v3only_t1only_s1 | 0.9601 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| t1_v4_t1only_s0 | 0.9951 | 0.00 | [0.0, 2.42] | 0.66 | [0.02, 3.66] |
| t1_v4_t1only_s1 | 0.9637 | 0.00 | [0.0, 2.42] | 0.66 | [0.02, 3.66] |
| t1_v4cap_t1only_s0 | 0.9960 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| t1_v4cap_t1only_s1 | 0.9347 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.00 | [0.0, 2.42] | 1.31 | [0.16, 4.75] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| tmp_v4_temporal_s0 | 0.9998 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| tmp_v4_temporal_s1 | 0.9857 | 0.00 | [0.0, 2.42] | 0.66 | [0.02, 3.66] |
| tmp_v4cap_temporal_s0 | 0.9931 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| tmp_v4cap_temporal_s1 | 0.9750 | 0.00 | [0.0, 2.42] | 0.00 | [0.0, 2.42] |
| DEPLOYED expE (0.30/0.15) | - | 3.29 | [1.07, 7.67] | - | - |

**boreal_val_nonsmoke_heinola_only** — 14 sequences / 439 frames / 0.122 h wall clock. boreal heinola (VAL burn site) non-smoke clips, 1 fps

| system | thr@FPR1% | false episodes/h @FPR1% | CI95 | false episodes/h @FPR5% | CI95 |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.00 | [0.0, 30.25] | 8.20 | [0.21, 45.69] |
| s2_persist_3of5 | 0.7874 | 0.00 | [0.0, 30.25] | 8.20 | [0.21, 45.69] |
| t1_v3only_t1only_s0 | 0.9367 | 0.00 | [0.0, 30.25] | 8.20 | [0.21, 45.69] |
| t1_v3only_t1only_s1 | 0.9601 | 0.00 | [0.0, 30.25] | 8.20 | [0.21, 45.69] |
| t1_v4_t1only_s0 | 0.9951 | 41.00 | [13.31, 95.69] | 82.00 | [39.32, 150.81] |
| t1_v4_t1only_s1 | 0.9637 | 73.80 | [33.75, 140.1] | 82.00 | [39.32, 150.81] |
| t1_v4cap_t1only_s0 | 0.9960 | 41.00 | [13.31, 95.69] | 57.40 | [23.08, 118.27] |
| t1_v4cap_t1only_s1 | 0.9347 | 41.00 | [13.31, 95.69] | 65.60 | [28.32, 129.27] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.00 | [0.0, 30.25] | 8.20 | [0.21, 45.69] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.00 | [0.0, 30.25] | 8.20 | [0.21, 45.69] |
| tmp_v4_temporal_s0 | 0.9998 | 90.21 | [45.03, 161.4] | 106.61 | [56.76, 182.3] |
| tmp_v4_temporal_s1 | 0.9857 | 73.80 | [33.75, 140.1] | 73.80 | [33.75, 140.1] |
| tmp_v4cap_temporal_s0 | 0.9931 | 49.20 | [18.06, 107.09] | 57.40 | [23.08, 118.27] |
| tmp_v4cap_temporal_s1 | 0.9750 | 57.40 | [23.08, 118.27] | 82.00 | [39.32, 150.81] |
| DEPLOYED expE (0.30/0.15) | - | 73.80 | [33.75, 140.1] | - | - |

**boreal_nonsmoke_all_incl_train_site** — 15 sequences / 471 frames / 0.131 h wall clock. adds the karkkila clip, whose background is a TRAIN burn site

| system | thr@FPR1% | false episodes/h @FPR1% | CI95 | false episodes/h @FPR5% | CI95 |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.00 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| s2_persist_3of5 | 0.7874 | 0.00 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| t1_v3only_t1only_s0 | 0.9367 | 0.00 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| t1_v3only_t1only_s1 | 0.9601 | 0.00 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| t1_v4_t1only_s0 | 0.9951 | 38.22 | [12.41, 89.18] | 76.43 | [36.65, 140.56] |
| t1_v4_t1only_s1 | 0.9637 | 68.79 | [31.46, 130.58] | 76.43 | [36.65, 140.56] |
| t1_v4cap_t1only_s0 | 0.9960 | 38.22 | [12.41, 89.18] | 53.50 | [21.51, 110.24] |
| t1_v4cap_t1only_s1 | 0.9347 | 38.22 | [12.41, 89.18] | 61.15 | [26.4, 120.48] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.00 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.00 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| tmp_v4_temporal_s0 | 0.9998 | 84.08 | [41.97, 150.44] | 99.36 | [52.91, 169.91] |
| tmp_v4_temporal_s1 | 0.9857 | 68.79 | [31.46, 130.58] | 68.79 | [31.46, 130.58] |
| tmp_v4cap_temporal_s0 | 0.9931 | 45.86 | [16.83, 99.82] | 53.50 | [21.51, 110.24] |
| tmp_v4cap_temporal_s1 | 0.9750 | 53.50 | [21.51, 110.24] | 76.43 | [36.65, 140.56] |
| DEPLOYED expE (0.30/0.15) | - | 68.79 | [31.46, 130.58] | - | - |

**firesense_negatives_CONTAMINATED** **[CONTAMINATED — see note]** — 9 sequences / 292 frames / 0.081 h wall clock. these 292 frames are background negatives inside v4 YOLO CORPUS_B_G_v1 / C_EGT_v1 / D_adapt_v1 -> optimistic for any model trained on those corpora. The temporal heads here read frozen expE (v3) features and expE never saw them, but the numbers are still flagged.

| system | thr@FPR1% | false episodes/h @FPR1% | CI95 | false episodes/h @FPR5% | CI95 |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| s2_persist_3of5 | 0.7874 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| t1_v3only_t1only_s0 | 0.9367 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| t1_v3only_t1only_s1 | 0.9601 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| t1_v4_t1only_s0 | 0.9951 | 86.30 | [34.7, 177.81] | 98.63 | [42.58, 194.34] |
| t1_v4_t1only_s1 | 0.9637 | 110.96 | [50.74, 210.63] | 110.96 | [50.74, 210.63] |
| t1_v4cap_t1only_s0 | 0.9960 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| t1_v4cap_t1only_s1 | 0.9347 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| tmp_v4_temporal_s0 | 0.9998 | 110.96 | [50.74, 210.63] | 110.96 | [50.74, 210.63] |
| tmp_v4_temporal_s1 | 0.9857 | 98.63 | [42.58, 194.34] | 110.96 | [50.74, 210.63] |
| tmp_v4cap_temporal_s0 | 0.9931 | 12.33 | [0.31, 68.69] | 12.33 | [0.31, 68.69] |
| tmp_v4cap_temporal_s1 | 0.9750 | 0.00 | [0.0, 45.48] | 0.00 | [0.0, 45.48] |
| DEPLOYED expE (0.30/0.15) | - | 36.99 | [7.63, 108.09] | - | - |

Sources:
- `reports/v4/trackd_v4_eval.json (C_false_alert_episodes_per_hour)`
