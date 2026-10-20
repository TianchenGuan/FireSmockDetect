_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T19. Early warning: v4 VAL replay per source — the domain-shortcut evidence (thresholds from T17, sdis-val all-neg pool)

For NEGATIVE footage (boreal/negative_footage = heinola non-smoke clips, firesense/negative_footage) every alert is false: the frame/clip alert rate IS the false-alarm rate. For positive sources the alert rate is a recall proxy only (boreal/positive is video-level truth; ausmoke/positive is positive from frame 0). Heads trained with the v4 AuSmoke extension (tmp_v4*, t1_v4*) fire on a large share of negative frames where v3-only heads fire on ~0% — the shortcut. NOTE: firesense negatives are also training negatives inside the YOLO corpora B_G/C_EGT/D_adapt (not inside these heads' data).

**ausmoke/positive** — 4 sequences / 1,471 frames, wall clock unknown (original fps never recorded (order units) -> no per-hour rate is emitted for this source), truth_level=frame_boxes, frame_truth=True, parent units=4
  - CAVEAT: positive from frame 0 — NO pre-onset segment exists; contributes nothing to the negative-footage pool

| system | thr@FPR1% | frame alert @1% | clip alert @1% | episodes/h @1% | frame alert @5% | clip alert @5% |
|---|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.3% | 50.0% | n/a | 2.2% | 50.0% |
| s2_persist_3of5 | 0.7874 | 0.1% | 25.0% | n/a | 1.7% | 50.0% |
| t1_v3only_t1only_s0 | 0.9367 | 0.6% | 75.0% | n/a | 5.9% | 75.0% |
| t1_v3only_t1only_s1 | 0.9601 | 0.0% | 0.0% | n/a | 2.4% | 75.0% |
| t1_v4_t1only_s0 | 0.9951 | 0.0% | 0.0% | n/a | 10.4% | 100.0% |
| t1_v4_t1only_s1 | 0.9637 | 1.8% | 25.0% | n/a | 14.3% | 75.0% |
| t1_v4cap_t1only_s0 | 0.9960 | 1.6% | 50.0% | n/a | 8.5% | 75.0% |
| t1_v4cap_t1only_s1 | 0.9347 | 0.9% | 50.0% | n/a | 6.5% | 75.0% |
| tmp_v3only_temporal_s0 | 0.9988 | 0.3% | 25.0% | n/a | 3.2% | 75.0% |
| tmp_v3only_temporal_s1 | 0.9556 | 0.0% | 0.0% | n/a | 2.0% | 75.0% |
| tmp_v4_temporal_s0 | 0.9998 | 3.3% | 50.0% | n/a | 11.3% | 75.0% |
| tmp_v4_temporal_s1 | 0.9857 | 0.7% | 25.0% | n/a | 7.7% | 75.0% |
| tmp_v4cap_temporal_s0 | 0.9931 | 3.5% | 75.0% | n/a | 10.6% | 75.0% |
| tmp_v4cap_temporal_s1 | 0.9750 | 3.1% | 25.0% | n/a | 11.3% | 75.0% |
| DEPLOYED expE (prov OR conf, 0.30/0.15) | - | 68.2% | 100.0% | n/a | - | - |

**boreal/negative_footage** — 15 sequences / 471 frames, 0.131 h, truth_level=videolabel, frame_truth=True, parent units=2
  - CAVEAT: clip from a TRAIN burn site (evo/karkkila) — background seen at train time; weaker negative evidence than val-site footage

| system | thr@FPR1% | frame alert @1% | clip alert @1% | episodes/h @1% | frame alert @5% | clip alert @5% |
|---|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.0% | 0.0% | 0.00 | 1.7% | 6.7% |
| s2_persist_3of5 | 0.7874 | 0.0% | 0.0% | 0.00 | 1.7% | 6.7% |
| t1_v3only_t1only_s0 | 0.9367 | 0.0% | 0.0% | 0.00 | 0.9% | 6.7% |
| t1_v3only_t1only_s1 | 0.9601 | 0.0% | 0.0% | 0.00 | 0.6% | 6.7% |
| t1_v4_t1only_s0 | 0.9951 | 26.3% | 33.3% | 38.23 | 41.6% | 66.7% |
| t1_v4_t1only_s1 | 0.9637 | 44.2% | 60.0% | 68.81 | 46.3% | 66.7% |
| t1_v4cap_t1only_s0 | 0.9960 | 5.3% | 33.3% | 38.23 | 25.5% | 46.7% |
| t1_v4cap_t1only_s1 | 0.9347 | 15.7% | 33.3% | 38.23 | 31.2% | 53.3% |
| tmp_v3only_temporal_s0 | 0.9988 | 0.0% | 0.0% | 0.00 | 1.5% | 6.7% |
| tmp_v3only_temporal_s1 | 0.9556 | 0.0% | 0.0% | 0.00 | 0.6% | 6.7% |
| tmp_v4_temporal_s0 | 0.9998 | 43.3% | 73.3% | 84.10 | 56.9% | 86.7% |
| tmp_v4_temporal_s1 | 0.9857 | 42.7% | 60.0% | 68.81 | 46.1% | 60.0% |
| tmp_v4cap_temporal_s0 | 0.9931 | 16.4% | 40.0% | 45.87 | 30.1% | 46.7% |
| tmp_v4cap_temporal_s1 | 0.9750 | 27.6% | 46.7% | 53.52 | 37.4% | 66.7% |
| DEPLOYED expE (prov OR conf, 0.30/0.15) | - | 30.1% | 60.0% | 68.81 | - | - |

**boreal/positive** — 111 sequences / 3,432 frames, 0.953 h, truth_level=videolabel, frame_truth=False, parent units=1

| system | thr@FPR1% | frame alert @1% | clip alert @1% | episodes/h @1% | frame alert @5% | clip alert @5% |
|---|---|---|---|---|---|---|
| s1_expE | 0.7997 | 4.6% | 16.2% | 18.88 | 17.1% | 35.1% |
| s2_persist_3of5 | 0.7874 | 6.0% | 15.3% | 17.83 | 15.7% | 28.8% |
| t1_v3only_t1only_s0 | 0.9367 | 0.1% | 0.9% | 1.05 | 1.8% | 6.3% |
| t1_v3only_t1only_s1 | 0.9601 | 0.0% | 0.0% | 0.00 | 0.4% | 3.6% |
| t1_v4_t1only_s0 | 0.9951 | 6.6% | 37.8% | 44.06 | 55.3% | 96.4% |
| t1_v4_t1only_s1 | 0.9637 | 79.5% | 98.2% | 114.34 | 90.7% | 100.0% |
| t1_v4cap_t1only_s0 | 0.9960 | 10.7% | 35.1% | 40.91 | 55.2% | 92.8% |
| t1_v4cap_t1only_s1 | 0.9347 | 25.1% | 54.9% | 63.99 | 56.0% | 91.0% |
| tmp_v3only_temporal_s0 | 0.9988 | 0.0% | 0.9% | 1.05 | 0.7% | 3.6% |
| tmp_v3only_temporal_s1 | 0.9556 | 0.1% | 1.8% | 2.10 | 2.4% | 11.7% |
| tmp_v4_temporal_s0 | 0.9998 | 84.1% | 99.1% | 115.39 | 89.8% | 100.0% |
| tmp_v4_temporal_s1 | 0.9857 | 76.4% | 99.1% | 115.39 | 86.9% | 100.0% |
| tmp_v4cap_temporal_s0 | 0.9931 | 36.8% | 76.6% | 89.16 | 66.9% | 98.2% |
| tmp_v4cap_temporal_s1 | 0.9750 | 58.3% | 92.8% | 108.05 | 76.4% | 98.2% |
| DEPLOYED expE (prov OR conf, 0.30/0.15) | - | 80.8% | 98.2% | 114.34 | - | - |

**firesense/negative_footage** — 9 sequences / 292 frames, 0.081 h, truth_level=videolabel, frame_truth=True, parent units=9
  - CAVEAT: frames used as background negatives in v4 YOLO train corpora (e.g. CORPUS_B_G_v1: 292 firesense background_neg lines) — false-alert numbers are OPTIMISTIC for models trained on those corpora

| system | thr@FPR1% | frame alert @1% | clip alert @1% | episodes/h @1% | frame alert @5% | clip alert @5% |
|---|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| s2_persist_3of5 | 0.7874 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| t1_v3only_t1only_s0 | 0.9367 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| t1_v3only_t1only_s1 | 0.9601 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| t1_v4_t1only_s0 | 0.9951 | 82.5% | 77.8% | 86.31 | 96.2% | 88.9% |
| t1_v4_t1only_s1 | 0.9637 | 98.3% | 100.0% | 110.97 | 100.0% | 100.0% |
| t1_v4cap_t1only_s0 | 0.9960 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| t1_v4cap_t1only_s1 | 0.9347 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| tmp_v3only_temporal_s0 | 0.9988 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| tmp_v3only_temporal_s1 | 0.9556 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| tmp_v4_temporal_s0 | 0.9998 | 92.5% | 100.0% | 110.97 | 96.2% | 100.0% |
| tmp_v4_temporal_s1 | 0.9857 | 88.4% | 88.9% | 98.64 | 94.5% | 100.0% |
| tmp_v4cap_temporal_s0 | 0.9931 | 4.5% | 11.1% | 12.33 | 10.6% | 11.1% |
| tmp_v4cap_temporal_s1 | 0.9750 | 0.0% | 0.0% | 0.00 | 0.0% | 0.0% |
| DEPLOYED expE (prov OR conf, 0.30/0.15) | - | 11.6% | 33.3% | 36.99 | - | - |

Sources:
- `reports/v4/trackd_v4_eval.json (B_v4_val_replay)`
