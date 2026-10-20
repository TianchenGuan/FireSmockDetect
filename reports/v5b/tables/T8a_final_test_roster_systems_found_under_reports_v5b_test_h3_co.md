_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

### T8a. FINAL TEST roster: systems found under reports/v5b/test, H3 contamination status, pre-registered VAL confidences

One-shot evaluation ran at 2026-10-03T03:20:07Z
2026-10-03T03:48:01Z
2026-10-03T03:49:22Z
2026-10-03T03:49:43Z
2026-10-03T03:50:03Z (reports/v5b/test/.EVALUATED_ONCE). Pre-registered confs (PREREGISTRATION_V5B.md 'Thresholds per system') are computed from the VAL parquets by this script (reports/v5b/tables/_derived/preregistered_confs_v5b.json). H3 contamination: arms B/C/D/D2/D3 trained on ALL Sheffield frames incl. rec_0008 (LOCKBOX_V5B_FROZEN.md); the heuristic maps pre-registered tokens S2/S3/S4 and arm names B/C/D/D2/D3 to CONTAMINATED - drop a reports/v5b/test/ROSTER_V5B.json {"systems": {token: {"h3_contaminated": bool}}} to override. NOTE: LOCKBOX wording 'the ACCURATE ensemble, which contains B' conflates v5b arm B with the v4 DINO B-series; the shipped recipe (yolo_A + dino_B4/B6/B1) holds no Sheffield frame and is therefore NOT contaminated by its training lists.

| system token | sets present | H3 status | sdis-VAL conf @5/10/20% | FireMan-VAL conf @5/10% |
|---|---|---|---|---|
| S0_expE | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | clean (heuristic) | 0.05: 0.721, 0.10: 0.655, 0.20: 0.535 | 0.05: 0.177, 0.10: 0.070 |
| S0b_expB | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | clean (heuristic) | 0.05: 0.723, 0.10: 0.650, 0.20: 0.532 | 0.05: 0.144, 0.10: 0.063 |
| S1_A | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | clean (heuristic) | 0.05: 0.729, 0.10: 0.673, 0.20: 0.534 | 0.05: 0.091, 0.10: 0.030 |
| S1_A_s1 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | clean (heuristic) | 0.05: 0.727, 0.10: 0.666, 0.20: 0.542 | 0.05: 0.236, 0.10: 0.165 |
| S1_A_s2 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | clean (heuristic) | 0.05: 0.717, 0.10: 0.646, 0.20: 0.500 | 0.05: 0.282, 0.10: 0.179 |
| S2_B | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | CONTAMINATED (heuristic) | 0.05: 0.740, 0.10: 0.672, 0.20: 0.557 | 0.05: 0.132, 0.10: 0.076 |
| S2_B_s1 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | CONTAMINATED (heuristic) | 0.05: 0.728, 0.10: 0.659, 0.20: 0.528 | 0.05: n/a, 0.10: n/a |
| S2_B_s2 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | CONTAMINATED (heuristic) | 0.05: 0.728, 0.10: 0.656, 0.20: 0.559 | 0.05: 0.194, 0.10: 0.044 |
| S3_C | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | CONTAMINATED (heuristic) | 0.05: 0.745, 0.10: 0.691, 0.20: 0.585 | 0.05: 0.072, 0.10: 0.045 |
| S4_D | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | CONTAMINATED (heuristic) | 0.05: 0.758, 0.10: 0.695, 0.20: 0.578 | 0.05: 0.082, 0.10: 0.036 |
| S4b_D2 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | CONTAMINATED (heuristic) | 0.05: 0.765, 0.10: 0.703, 0.20: 0.572 | 0.05: 0.051, 0.10: 0.024 |
| S4c_D3 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | CONTAMINATED (heuristic) | 0.05: 0.723, 0.10: 0.679, 0.20: 0.564 | 0.05: 0.262, 0.10: 0.199 |
| S5_ensemble_v4a_s2 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | clean (heuristic) | 0.05: 0.982, 0.10: 0.962, 0.20: 0.905 | no fireman-val parquet -> n/a |
| v4B1_frozen_L | firemanholdout, sheffieldholdout | UNKNOWN (unknown) | no sdis-val parquet -> n/a | no fireman-val parquet -> n/a |
| v4B4_lora_L | firemanholdout, sheffieldholdout | UNKNOWN (unknown) | no sdis-val parquet -> n/a | no fireman-val parquet -> n/a |
| v4B6_lora_Hplus | firemanholdout, sheffieldholdout | UNKNOWN (unknown) | no sdis-val parquet -> n/a | no fireman-val parquet -> n/a |
| v4a_s2 | borealtest, firemanholdout, foresttest, sdistest, sheffieldholdout | UNKNOWN (unknown) | no sdis-val parquet -> n/a | no fireman-val parquet -> n/a |

Sources:
- `reports/v5b/test/eval_<sys>_<set>.{json,raw.parquet}`
- `reports/v5b/tables/_derived/preregistered_confs_v5b.json`
- `reports/v5b/LOCKBOX_V5B_FROZEN.md`
- `reports/v5b/PREREGISTRATION_V5B.md`
