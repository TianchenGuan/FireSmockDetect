_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T15. FIgLib-39 paired detection delay vs expE (events detected by BOTH systems)

delta = system delay - expE delay per event (negative = earlier than expE); n_both = events detected by both of the 39. Pre-visible detections carry delay 0 s on both sides. median = statistics.median (mean of the two middle values for even n; scripts/v4/figlib_replay_report.py uses the upper middle value, so its medians can differ by <=1 s).

| system | n both | earlier | tie | later | median delta | mean delta |
|---|---|---|---|---|---|---|
| S1 A_ctrl TEST ckpt (v4a_s2) | 30 | 5 | 20 | 5 | +0 s | +52 s |
| S2 E_ET TEST ckpt (v4e_s1) | 29 | 11 | 11 | 7 | +0 s | -79 s |
| S2c C_EGT TEST ckpt (v4c_s1) | 29 | 10 | 12 | 7 | +0 s | -13 s |
| S3 D_curr TEST ckpt (v4dada_s1, post-hoc) | 29 | 7 | 11 | 11 | +0 s | +99 s |
| A_ctrl_E seed0 (v4a) | 30 | 5 | 19 | 6 | +0 s | -10 s |
| E_ET seed0 (v4e) | 27 | 7 | 12 | 8 | +0 s | +2 s |
| C_EGT seed0 (v4c) | 29 | 12 | 10 | 7 | +0 s | -23 s |
| S4p v4B4_lora_L (DINO, sdis-val-selected confs) | 25 | 3 | 3 | 19 | +181 s | +288 s |
| S4 v4B6_lora_Hplus (DINO, sdis-val-selected confs) | 24 | 2 | 5 | 17 | +61 s | +290 s |
| v3 B1 frozen ViT-L (iteration-3 reference) | 29 | 11 | 10 | 8 | +0 s | -64 s |

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
