_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T17. Early warning: pyro-sdis VAL frame-level recall at matched frame FPR (all-negative pool)

Frames 7,335 (positive 6,177 / negative 1,158; clean-negative subset 560); tiny-GT positive frames 5,881. s1_* = single-frame detector max score; s2_persist_3of5 = deployed persistence rule; tmp_* = learned causal temporal head (v3only / v4 = +AuSmoke extension / v4cap = capped 5-seq extension), t1_* = same module on the current frame only; _s0/_s1 = training seeds. Thresholds chosen on these val negatives.

| system | params | PR-AUC | ROC-AUC | R@FPR0.5% | R@FPR1% | R@FPR2% | R@FPR5% | R@FPR10% | tinyR@FPR5% |
|---|---|---|---|---|---|---|---|---|---|
| s1_expE | 0 (rule/raw) | 0.9351 | 0.7685 | 0.0353 | 0.0622 | 0.1004 | 0.2053 | 0.3400 | 0.2095 |
| s1_v4a_ctrlE | 0 (rule/raw) | 0.9364 | 0.7718 | 0.0414 | 0.0580 | 0.1101 | 0.2194 | 0.3371 | 0.2234 |
| s1_v4B1_dinoL | 0 (rule/raw) | 0.9327 | 0.7535 | 0.0537 | 0.0699 | 0.1078 | 0.1967 | 0.3324 | 0.1988 |
| s2_persist_3of5 | 0 (rule/raw) | 0.9169 | 0.7056 | 0.0123 | 0.0515 | 0.0826 | 0.1632 | 0.2752 | 0.1632 |
| t1_v3only_t1only_s0 | 281,153 | 0.9431 | 0.7798 | 0.0503 | 0.0639 | 0.1264 | 0.2589 | 0.4266 | 0.2573 |
| t1_v3only_t1only_s1 | 281,153 | 0.9399 | 0.7761 | 0.0401 | 0.0541 | 0.1130 | 0.2449 | 0.3987 | 0.2450 |
| t1_v4_t1only_s0 | 281,153 | 0.9473 | 0.7980 | 0.0448 | 0.0599 | 0.1305 | 0.3225 | 0.4646 | 0.3192 |
| t1_v4_t1only_s1 | 281,153 | 0.9501 | 0.8088 | 0.0283 | 0.0989 | 0.1965 | 0.3463 | 0.4967 | 0.3333 |
| t1_v4cap_t1only_s0 | 281,153 | 0.9469 | 0.7932 | 0.0380 | 0.1030 | 0.1463 | 0.2899 | 0.4412 | 0.2906 |
| t1_v4cap_t1only_s1 | 281,153 | 0.9426 | 0.7897 | 0.0204 | 0.0539 | 0.1260 | 0.2671 | 0.4345 | 0.2644 |
| tmp_v3only_temporal_s0 | 281,153 | 0.9404 | 0.7736 | 0.0423 | 0.0711 | 0.1512 | 0.2498 | 0.3983 | 0.2483 |
| tmp_v3only_temporal_s1 | 281,153 | 0.9391 | 0.7667 | 0.0299 | 0.0835 | 0.1813 | 0.2462 | 0.3861 | 0.2460 |
| tmp_v4_temporal_s0 | 281,153 | 0.9388 | 0.7662 | 0.0400 | 0.0737 | 0.1350 | 0.2576 | 0.3983 | 0.2472 |
| tmp_v4_temporal_s1 | 281,153 | 0.9457 | 0.7955 | 0.0303 | 0.0860 | 0.1376 | 0.2812 | 0.4133 | 0.2761 |
| tmp_v4cap_temporal_s0 | 281,153 | 0.9426 | 0.7810 | 0.0291 | 0.0565 | 0.1439 | 0.2699 | 0.4208 | 0.2683 |
| tmp_v4cap_temporal_s1 | 281,153 | 0.9430 | 0.7781 | 0.0562 | 0.0908 | 0.1462 | 0.2815 | 0.4102 | 0.2758 |

Sources:
- `reports/v4/trackd_v4_eval.json (A_sdis_val_frame_level)`
