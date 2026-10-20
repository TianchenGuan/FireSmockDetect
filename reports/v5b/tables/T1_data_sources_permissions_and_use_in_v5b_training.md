_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

### T1. Data sources, permissions and use in V5b training

Counts are read from the V5b manifests (agent-built, nothing human-reviewed; `annotation_origin=author` everywhere except WSDataset masks = model-assisted and the D3 pseudo boxes = model-assisted). 'Used in training' names the arms whose train lists contain the source (data/v5b/manifests/CORPUS_*_v1.json). FireMan boxes are interpolated region boxes (presence-grade, not tight instances); Sheffield boxes are fragmented sub-boxes per plume.

| source | frames (rows) | smoke boxes | explicit negatives | units | licence (verbatim/short) | permission_status | used in V5b training | eval role |
|---|---|---|---|---|---|---|---|---|
| E corpus = pyro-sdis TRAIN sites + D-Fire smoke (frozen expB_train.txt) | 38,454 | 21,944 + 9,550 | 11,257 | 4 sdis TRAIN sites + D-Fire (no units) | pyro-sdis: Apache-2.0; D-Fire: CC0, provenance caveat in repo LICENSE (reports/FINAL_REPORT.md) | inherited (v1-v4) | YES - every arm (A,B,C,D,D2,D3), x1 | sdis VAL/TEST sites held out |
| Boreal Forest Fire (FGI) TRAIN burns evo + karkkila, Subset-A boxes | 2,062 | 2,061 | 37 (karkkila Empty-Images clip) | 2 burns / 13 flights | - CC BY 4.0 (Creative Commons Attribution 4.0 International). | confirmed | YES - arms B,C,D,D2,D3 x2 (and v4 E+T models) | heinola = aerial VAL; ruokolahti = historical TEST |
| Boreal Subset-C masks (SAM model-assisted + 27 human) + author negatives [SEG branch] | 665 train / 382 val | masks: 628 / 336 | 37 / 46 | train evo+karkkila 13 flights; val heinola 7 flights | CC BY 4.0 (same source) | confirmed | YES - segV5b_a only (not detectors) | heinola masks = VAL; ruokolahti masks (453) never staged |
| FireMan-UAV-RGBT (Oulu) RGB, Multiclass CVAT | 69,871 | 69,234 on 64,867 frames (incl. 4 on 4 conflict frames; CARD says 69,232) | 4,979 | 4 burns / 34 videos | CC BY 4.0 | confirmed | YES - burn A train capped 600/video = 8,279 lines (arms B,C,D,D2,D3) | burn D video_015 = aerial VAL (fireman_val_s5); burn C = H1 holdout (sealed); burn B = quarantine (Ruokolahti candidate) |
| FireMan thermal (no smoke supervision) | 33,052 | 0 smoke boxes | 0 | 10 videos | CC BY 4.0 | confirmed | NO (not exported) | future RGB-T ablation only |
| Sheffield UAV (ORDA, Roboflow re-deposit) | 1,336 | 5,283 | 30 (24 exported; 6 smoke-visible excluded) | 1 session / 6 recordings | CC BY 4.0 | confirmed | YES - ALL 1,327 exported frames x2 (arms B,C,D,D2,D3) incl. rec_0008 | rec_0008 = H3 (CONTAMINATED for B/C/D/D2/D3; within-incident for everyone) |
| FireSpot-CNX-r1 (NECTEC) | 3,996 | 4,058 | 1,179 | 5 sites / 29 sessions | UNSPECIFIED (no LICENSE anywhere; MNS 'CC BY 4.0' is an aggregator claim) | pending | NO (guarded converter refuses); NOTE 2,862 MNS copies were in v4 G corpora (R3) | none |
| WSDataset (Kaggle gloryvu; MNS copies + GitHub demo) | 11,753 rows (129 new files) | 12,929 (mask-derived, model-assisted; author boxes BLOCKED) | 126 new author negatives | 70 units (66 videos recovered) | MIT declared on Kaggle page (text unverified); GitHub demo unlicensed | pending | NO in v5b (MNS copies were v4 G material); original 18.4 GB archive BLOCKED (Kaggle token) | none |
| UNLABELLED smoke-positive material for D3: Boreal train-burn Subset-B clip frames (104 smoke clips), FIRESENSE smoke_pos, D-Fire fire-only | 3,796 | 0 author boxes -> 3,797 teacher pseudo boxes (model-assisted) | 0 (no teacher-silence negatives created) | 3,370 + 530 + 944 target frames (TEACHER_POOL.md D3 section) | Boreal CC BY 4.0; FIRESENSE / D-Fire inherited | inherited | YES - arm D3 only (x1, pseudo boxes) | none |
| AuSmoke (via MultiNatSmoke) T list | 10,904 | - | - | - | permission PENDING (PROTOCOL_V5 S2) | pending | NO new use in v5b (v4 B1/B4/B6/E_ET models already trained on it - disclosed) | AuSmoke forest TEST = H4 historical |

Sources:
- `data/v4/manifests/boreal_CARD.md`
- `data/v5b/manifests/CORPUS_B_v1.json`
- `data/v5b/manifests/CORPUS_D3_v1.json`
- `data/v5b/manifests/fireman_CARD.md`
- `data/v5b/manifests/fireman_manifest.parquet`
- `data/v5b/manifests/firespot_CARD.md`
- `data/v5b/manifests/firespot_manifest.parquet`
- `data/v5b/manifests/seg_v5b_manifest.json`
- `data/v5b/manifests/sheffield_uav_CARD.md`
- `data/v5b/manifests/sheffield_uav_manifest.parquet`
- `data/v5b/manifests/wsdataset_CARD.md`
- `data/v5b/manifests/wsdataset_manifest.parquet`
- `models/v5b/MODEL_REGISTRY.json (train_data_recipe of expB_y11s)`
- `reports/FINAL_REPORT.md (iteration-1 data section: licence lines)`
