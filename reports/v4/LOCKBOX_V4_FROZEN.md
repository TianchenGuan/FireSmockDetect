# LOCKBOX V4 -- FROZEN (sealed final TEST set, iteration 4)

- Frozen (UTC): **2026-09-14T18:38:37+00:00**
- Built by: `scripts/v4/build_evaltest.py` (DATA PREP ONLY -- no model run, no prediction, no metric).
- Manifest: `/local/home/tianrgua/cvproject/data/v4/eval/EVALTEST_MANIFEST.json`
- **Evaluated once at iteration-4 close; never used for training, mining, tuning, threshold selection, or model selection.**

## What is in the sealed set

| set | units | frames | positives | negatives | boxes | independent events |
|---|---|---|---|---|---|---|
| boreal_test | burn_ruokolahti (1 unit, 9 DJI flights) | 1764 | 1762 | 2 | 1884 | **1** (ONE burn) |
| forest_test | 9 AuSmoke units in 4 groups | 2871 | 2871 | 0 | 4359 | **4** (3 ground burns + 1 aerial flight) |
| sdis_test | pyro-sdis sites ferion + croix-augas | 4124 | - | - | - | 2 sites (frozen since iteration 1; images not part of this package) |

### Caveats (read before interpreting any number)

- **boreal_test = ONE burn** (burn_ruokolahti). Every frame is from the same prescribed burn, split only by DJI flight; per-frame metrics are heavily autocorrelated and the effective sample size is 1 event. Boreal has only 4 independent burns in total (train 2 / val 1 / test 1) -- treat aerial test numbers as anecdotal (see aerial_split_v4.json tiny_n_warning).
- boreal_test negatives: only **2** verified-empty frames (Subset-A Empty-Images with explicit Empty-Labels). Matched-FP anchoring within boreal_test is effectively impossible; false-alert rates must come from sdis_test / other negative pools.
- **forest_test = 4 independent events**, of which **1 is an aerial flight** (`g_dji_20230922`: 3 DJI videos from ONE flight on 2023-09-22) that **includes a staged test-run** (`aus_DJI_20230922115131_0004_D_TEST_RUN_SMOKE`, staged_test_run=true). Report the aerial event separately from the 3 ground burns and report the staged test-run unit on its own; never pool the 9 units as 9 events.
- forest_test negatives: **0** (AuSmoke ships positive-only frames; every frame has mask_fg_px>0). Recall-only set; matched-FP is undefined within forest_test.
- Group provenance (forest_holdout_v4.json): several units are the SAME burn filmed by different cameras -> counted once per group below.

### forest_test units

| unit | group | frames | boxes | staged_test_run | aerial_event |
|---|---|---|---|---|---|
| aus_DJI_20230922112303_0008_D | g_dji_20230922 | 25 | 27 | false | g_dji_20230922 |
| aus_DJI_20230922112554_0009_D | g_dji_20230922 | 143 | 143 | false | g_dji_20230922 |
| aus_DJI_20230922115131_0004_D_TEST_RUN_SMOKE | g_dji_20230922 | 90 | 93 | **true** | g_dji_20230922 |
| aus_20241021_S1050004 | g_20241021 | 670 | 1027 | false | - |
| aus_2024_10_21_06_1fps | g_20241021 | 276 | 769 | false | - |
| aus_27032024_S1020005 | g_20240326_fb769 | 167 | 194 | false | - |
| aus_FB769_Bonshaw_March_26_202403251112 | g_20240326_fb769 | 530 | 648 | false | - |
| aus_FB769_Bonshaw_March_26_202403261044_Panasonic | g_20240326_fb769 | 566 | 606 | false | - |
| aus_20240517_FB884_Sony4K_1300_MAH00006 | g_20240517_fb884 | 404 | 852 | false | - |

Independent events (forest_test): `g_dji_20230922` (aus_DJI_20230922112303_0008_D, aus_DJI_20230922112554_0009_D, aus_DJI_20230922115131_0004_D_TEST_RUN_SMOKE), `g_20241021` (aus_20241021_S1050004, aus_2024_10_21_06_1fps), `g_20240326_fb769` (aus_27032024_S1020005, aus_FB769_Bonshaw_March_26_202403251112, aus_FB769_Bonshaw_March_26_202403261044_Panasonic), `g_20240517_fb884` (aus_20240517_FB884_Sony4K_1300_MAH00006)

### boreal_test flights

| camera_or_flight_id | positives | negatives |
|---|---|---|
| ruokolahti_DJI_0080 | 341 | 0 |
| ruokolahti_DJI_0081 | 128 | 0 |
| ruokolahti_DJI_0082 | 70 | 0 |
| ruokolahti_DJI_0084 | 23 | 0 |
| ruokolahti_DJI_0085 | 176 | 2 |
| ruokolahti_DJI_0086 | 328 | 0 |
| ruokolahti_DJI_0087 | 279 | 0 |
| ruokolahti_DJI_0088 | 328 | 0 |
| ruokolahti_DJI_0089 | 89 | 0 |

## Exclusions applied (same policy as the val builder)

- `boreal:non_image(videolabel_mp4)`: 58
- `boreal:subsetC_duplicate_of_subsetA`: 453
- `boreal:subsetC_not_in_subsetA_skipped(label_type_none)`: 4
- `boreal:unexpected_label_type_none`: 3

## Leak check (train/val vs sealed test)

- Result: **PASS** -- path_overlap=0, sha256_overlap=0, manifest_nontest_sha256_overlap=0, within_test_duplicate_bytes=0
- Reference: 18 list files, 84680 unique images (sha256 of bytes computed for all), vs 4635 sealed test images. Lists checked:
  - `data/v4/eval/boreal_val/lists/local/val.txt` (1123)
  - `data/v4/eval/forest_val/lists/local/val.txt` (1471)
  - `data/v4/yolo/B_G_v1/lists/local/gval.txt` (310)
  - `data/v4/yolo/B_G_v1/lists/local/train.txt` (14901)
  - `data/v4/yolo/B_G_v1/lists/local/train90.txt` (14591)
  - `data/v4/yolo/B_G_v1/lists/local/train_G.txt` (14901)
  - `data/v4/yolo/C_EGT_v1/lists/local/train.txt` (94188)
  - `data/v4/yolo/C_EGT_v1/lists/local/train_E.txt` (38454)
  - `data/v4/yolo/C_EGT_v1/lists/local/train_G.txt` (14901)
  - `data/v4/yolo/C_EGT_v1/lists/local/train_T.txt` (12966)
  - `data/v4/yolo/E_ET_v1/lists/local/train.txt` (51420)
  - `data/v4/yolo/E_ET_v1/lists/local/train_E.txt` (38454)
  - `data/v4/yolo/E_ET_v1/lists/local/train_T.txt` (12966)
  - `data/yolo/lists/local/dfire_test_raw.txt` (4306)
  - `data/yolo/lists/local/expB_train.txt` (38454)
  - `data/yolo/lists/local/sdis_test.txt` (4124)
  - `data/yolo/lists/local/sdis_train.txt` (22177)
  - `data/yolo/lists/local/sdis_val.txt` (7335)

- **Known re-distribution** `multinatsmoke:dup_boreal:burn_ruokolahti`: 2215 rows in the public MultiNatSmoke release are byte-identical copies of burn_ruokolahti frames (1762 match sealed test images; MultiNatSmoke official split: {'Train': 1748, 'Test': 467}). They are excluded from every corpus by `scripts/v4/build_corpus.py origin_class != usable` and verified present in 0 train/val lists. **Implication:** any checkpoint trained on the PUBLIC MultiNatSmoke release is contaminated for boreal_test (ruokolahti frames sit in its official Train split). (The same holds for burn_heinola = boreal_val.)

## SHA256

- tar `/local/home/tianrgua/cvproject/data/v4/stage/v4_evaltest.tar` (6,747,750,400 bytes): `f7cc4ad752b396c189165ccf6a389fd41bec9e52697e11eade386c10b10498c0`
  - pod extraction prefix: `/scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evaltest/`
- `/local/home/tianrgua/cvproject/data/v4/eval/boreal_test/lists/local/test.txt` (1764): `e02ab8a6bc28eb185e9a706033be46fc95a3ee3d987104213b1d6c2c5aed3343`
- `/local/home/tianrgua/cvproject/data/v4/eval/boreal_test/lists/pod/test.txt` (1764): `4d04a10c08c5ccd252f46b56a521e332266bbc438ab1aa52d3a0744acae62b53`
- `/local/home/tianrgua/cvproject/data/v4/eval/boreal_test/SHA256SUMS` (per-image sha256): `7d5ddfb7b24d15245dc64491337c2c705c15b08e4941819ecb973fd6e13a4751`
- `/local/home/tianrgua/cvproject/data/v4/coco/instances_boreal_test.json`: `f7ac04207d07c586e90876790b045935ed6ddd0eff1e1a2f4ddc311a985e69fc`
- `/local/home/tianrgua/cvproject/data/v4/eval/yamls/boreal_test.yaml`: `ac8ff7942aa5baa61f18220a9882e1276d6141484f9940602dfec8aa50b5146b`
- `/local/home/tianrgua/cvproject/data/v4/eval/forest_test/lists/local/test.txt` (2871): `3b30cf643cad0d86c00671fd473b1a7ddac2c4900f45f19787b6c5252ff4c8e1`
- `/local/home/tianrgua/cvproject/data/v4/eval/forest_test/lists/pod/test.txt` (2871): `9770cb8cac31a846fcbef14fa9182f778f8c0ae7da7d044edbd0eb590df79d76`
- `/local/home/tianrgua/cvproject/data/v4/eval/forest_test/SHA256SUMS` (per-image sha256): `2128a4e618273164ff8a01be4d6338987fd756ac3da94749fc7031cc4d723608`
- `/local/home/tianrgua/cvproject/data/v4/coco/instances_forest_test.json`: `6da28a1d4e1e5ddd30734daad95f630ef60506dea60a19b41c3de1a7cc905438`
- `/local/home/tianrgua/cvproject/data/v4/eval/yamls/forest_test.yaml`: `64091c2e9ffbb211c7123c701b832cd6bd20b97316847e211323147a35cddf3a`
- `/local/home/tianrgua/cvproject/data/yolo/lists/local/sdis_test.txt` (4124): `88ea60e2b408f8346c186242ba1115ed9b647a138288c83a19d0b6dbb180d776`  (sdis_test; defined by `/local/home/tianrgua/cvproject/data/yolo/yamls/local/expB.yaml`; images untouched)
- `/local/home/tianrgua/cvproject/data/yolo/lists/pod/sdis_test.txt` (4124): `c5e8631ca789629e2818d2c0e90998a33602b6286300f212bab1314c0354a0d0`
- `data/manifests/splits_frozen.json`: `0692b6ae9023600507aae4463b80b07f7a50b4f4cca0033497b1de09d8157c86`

## Rules

1. Evaluated once at iteration-4 close; never used for training, mining, tuning, threshold selection, or model selection.
2. No script other than the one-time iteration-4 close evaluation may read the images listed in the files above. Any per-image output produced by that evaluation is HISTORICAL afterwards (same treatment as the FIgLib lockboxes v1-v3).
3. If any list/tar sha256 above changes, the seal is broken and the set must be re-frozen and re-declared before use.
4. The package layout mirrors `v4_evalval` (images/ labels/ lists/{local,pod}/test.txt, COCO json with absolute pod paths); consumers pass the list path explicitly.
5. Any future corpus built from `multinatsmoke` MUST keep dropping `origin_class=dup_boreal` rows (they are byte-identical Boreal copies, incl. burn_ruokolahti); any model initialised from a checkpoint trained on the public MultiNatSmoke release must be declared CONTAMINATED for boreal_test.
