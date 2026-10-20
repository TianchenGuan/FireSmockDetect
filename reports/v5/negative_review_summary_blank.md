# Verified no-smoke footage summary (verdicts: data/v5/review/verdicts.csv)

Counting rule: ONLY verdict == no_smoke contributes; uncertain / smoke / blank contribute 0. Hours are PROTOCOL_V5 §3 valid observed time per unit (gaps > 3x median cadence excluded, units never concatenated). Acceptance (§6) per scenario: >= 2.0 h verified AND >= 3 independent units AND licence recorded.

| scenario | units | no_smoke | smoke | uncertain | unreviewed | candidate h | VERIFIED no-smoke h | of which contaminated h | independent units verified/all | cadence_s | acceptance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aerial | 17 | 0 | 0 | 0 | 17 | 0.195 | **0.000** | 0.000 | 0/1 | 0.96,1.0 | NOT MET |
| fixed_camera_like | 2 | 0 | 0 | 0 | 2 | 0.044 | **0.000** | 0.000 | 0/2 | 1.0 | NOT MET |

| unit | scenario | source | verdict | reviewer | valid s | counted h | contamination |
|---|---|---|---|---|---|---|---|
| boreal_heinola_13 | aerial | boreal | (blank) |  | 32 | 0.0000 |  |
| boreal_heinola_21 | aerial | boreal | (blank) |  | 30 | 0.0000 |  |
| boreal_heinola_32 | aerial | boreal | (blank) |  | 30 | 0.0000 |  |
| boreal_heinola_37 | aerial | boreal | (blank) |  | 30 | 0.0000 |  |
| boreal_heinola_43 | aerial | boreal | (blank) |  | 31 | 0.0000 |  |
| boreal_heinola_49 | aerial | boreal | (blank) |  | 32 | 0.0000 |  |
| boreal_heinola_53 | aerial | boreal | (blank) |  | 32 | 0.0000 |  |
| boreal_heinola_62 | aerial | boreal | (blank) |  | 32 | 0.0000 |  |
| boreal_heinola_74 | aerial | boreal | (blank) |  | 32 | 0.0000 |  |
| boreal_heinola_76 | aerial | boreal | (blank) |  | 32 | 0.0000 |  |
| boreal_heinola_103 | aerial | boreal | (blank) |  | 30 | 0.0000 |  |
| boreal_heinola_104 | aerial | boreal | (blank) |  | 32 | 0.0000 |  |
| boreal_heinola_116 | aerial | boreal | (blank) |  | 32 | 0.0000 |  |
| boreal_heinola_124 | aerial | boreal | (blank) |  | 32 | 0.0000 |  |
| boreal_subsetA_empty_heinola_DJI_0024 | aerial | boreal | (blank) |  | 79 | 0.0000 |  |
| boreal_subsetA_empty_heinola_DJI_0028 | aerial | boreal | (blank) |  | 138 | 0.0000 |  |
| boreal_subsetA_empty_heinola_DJI_0048 | aerial | boreal | (blank) |  | 45 | 0.0000 |  |
| firesense_testneg02.808 | fixed_camera_like | firesense | (blank) |  | 46 | 0.0000 | 1-fps frames of this video are background negatives inside v4 YOLO train corpora (CORPUS_B_G_v1 et al.) -> replay FP optimistic for v4-trained models |
| firesense_testneg04.810 | fixed_camera_like | firesense | (blank) |  | 114 | 0.0000 | 1-fps frames of this video are background negatives inside v4 YOLO train corpora (CORPUS_B_G_v1 et al.) -> replay FP optimistic for v4-trained models |
