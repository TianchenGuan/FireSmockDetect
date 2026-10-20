# INVENTORY V6 — local data inventory for the wildfire-smoke V6 iteration

Built 2026-10-20 by the v6 inventory agent. All paths are absolute local paths under
`/local/home/tianrgua/cvproject` (pod root `/mnt/local/smoke_data` ⇄ local `data/yolo`;
pod `v4data`/`v5bdata` extractions mirror `data/v4`/`data/v5b`). Counts below were
recomputed from the lists + YOLO label files on disk (label = `images→labels`, `.txt`;
positive = non-empty label; negative = empty label; broken = image file missing).
Every list below verified with **0 broken paths** unless stated.

## 1. Real training recipe ("arm A" baked list) — LOCAL mirror

Pod list `/mnt/local/smoke_data/lists/pod/expB_train.txt` ⇄ local mirror
`/local/home/tianrgua/cvproject/data/yolo/lists/local/expB_train.txt` (38,454 lines; a
byte-identical pod-path copy sits at `data/yolo/lists/pod/expB_train.txt`).

| list (local) | images | pos | neg (empty) | neg (absent) | broken |
|---|---|---|---|---|---|
| `data/yolo/lists/local/expB_train.txt` | 38,454 | 27,197 | 11,257 | 0 | 0 |
| — of which `data/yolo/pyro_sdis/` (pyro-sdis TRAIN sites) | 22,177 | 18,753 | 3,424 | 0 | 0 |
| — of which `data/yolo/dfire_smoke/` (D-Fire) | 16,277 | 8,444 | 7,833 | 0 | 0 |
| `data/yolo/lists/local/sdis_train.txt` (pyro-sdis part alone) | 22,177 | 18,753 | 3,424 | 0 | 0 |
| `data/yolo/lists/local/sdis_val.txt` (VAL, matched-FP support) | 7,335 | 6,177 | **1,158 neg** | 0 | 0 |
| `data/yolo/lists/local/sdis_test.txt` (FROZEN TEST H2) | 4,124 | 3,207 | **917 neg** | 0 | 0 |

No "other" source directories appear in any of these lists. D-Fire empty labels are
exactly the 7,833 author *none*-category train images (verified against
`data/manifests/dfire_manifest.parquet`; the 944 fire-only images were never in E).

## 2. The 5 FROZEN FINAL TEST sets (v5b lockbox + v4 eval package) — v6 must NOT train on or tune with these

| set | local list | images | pos | neg | provenance / freeze doc |
|---|---|---|---|---|---|
| pyro-sdis TEST (H2) | `data/yolo/lists/local/sdis_test.txt` | 4,124 | 3,207 | 917 | frozen since v1 (`data/manifests/splits_frozen.json`); opened v4 close + v5b close (disclosed) |
| FireMan holdout s10 (H1, burn C Taivalkoski) | `data/v5b/eval/fireman_holdout_s10/lists/local/fireman_holdout_s10.txt` | 1,949 | 1,949 | 0 | `reports/v5b/LOCKBOX_V5B_FROZEN.md` (sealed; recall-only, 0 negatives; ONE burn, 640×512) |
| Sheffield holdout (H3, rec_0008) | `data/v5b/eval/sheffield_holdout/lists/local/sheffield_holdout.txt` | 327 | 327 | 0 | lockbox; within-incident only; CONTAMINATED for v5b arms B/C/D/D2/D3 (rec_0008 was in their train lists) — clean for A/expB/expE/B200-LoRA |
| boreal test (H4, ruokolahti) | `data/v4/eval/boreal_test/lists/local/test.txt` | 1,764 | 1,762 | 2 | `data/v4/eval/EVALTEST_MANIFEST.json`; opened once at v4 close, HISTORICAL since |
| forest test (AuSmoke holdout) | `data/v4/eval/forest_test/lists/local/test.txt` | 2,871 | 2,871 | 0 | v4 eval package (`forest_holdout_v4.json`); AuSmoke permission PENDING — eval-only disclosure |

Images+labels for each set live next to the lists (`.../images`, `.../labels`,
`SHA256SUMS`, yamls under `data/v5b/eval/yamls/` and `data/v4/eval/yamls/`).
Manifest: `data/v5b/eval/EVAL_V5B_MANIFEST.json`.

## 3. VAL sets available for v6 decisions

| set | local list | images | pos | neg |
|---|---|---|---|---|
| sdis VAL (fixed-camera; cabanelle+marguerite sites) | `data/yolo/lists/local/sdis_val.txt` | 7,335 | 6,177 | 1,158 |
| boreal VAL (aerial; heinola) | `data/v4/eval/boreal_val/lists/local/val.txt` | 1,123 | 906 | 217 |
| (also available, NOT sealed) FireMan aerial VAL s5 (burn D video_015) | `data/v5b/eval/fireman_val_s5/lists/local/fireman_val_s5.txt` | 1,706 | 1,475 | 231 |
| (historical) forest val (AuSmoke) | `data/v4/eval/forest_val/lists/local/val.txt` | 1,471 | 1,471 | 0 |

Caveat on heinola negatives: `reports/v5/REVIEW_QUEUE_V5.md` (a1–a3) flags 3 of the 14
"no-smoke" heinola clips + ≥1 of the 217 stills as agent-screened *possible real smoke*;
no human verdict was ever recorded (`data/v5/review/verdicts.csv` is blank).

## 4. Fresh never-used author-labeled holdout — FOUND (with caveats)

Scanned: `CORPUS_{B,C,D,D2,D3}_v1.json` (+ v4 `CORPUS_*` manifests), every list under
`data/v5b/yolo/*/lists/local/` and `data/yolo/lists/local/`,
`reports/v5b/COMPARISON_V5B.csv` (eval splits used: borealtest, borealval,
firemanholdout, firemanval, foresttest, sdistest, sdisval, sheffieldholdout — nothing
else), `FINAL_REPORT_V5B.md`, T1 permissions table, and the source CARDs.

**Best candidate: FireSpot-CNX-r1 (NECTEC, Chiang Mai, Thailand).**
- On disk: `data/v5b/raw/firespot/dataset/images/*.jpg` (3,996 files verified) +
  `dataset/labels/*.txt`; manifest `data/v5b/manifests/firespot_manifest.parquet`;
  card `data/v5b/manifests/firespot_CARD.md`.
- 3,996 author box-labeled images: 2,817 positives (4,058 smoke boxes) + **1,179
  explicit author negatives**; 5 sites / 29 sessions; smartphone + tower-camera stills.
- NEVER in any training corpus or evaluation v1–v5b (guarded converter refused to run;
  T1: used-in-training = NO, eval role = none). NOTE: 2,862 MNS re-encode *copies* of
  FireSpot frames were in v4 G corpora → before using FireSpot as a holdout for models
  with v4-G lineage (v4C_egt, v4Dada G-pretrain), overlap vs the MNS copies must be
  honoured; for v6 students trained on E(+synthetic) it is clean.
- **Caveat: licence UNSPECIFIED, permission_status=pending** (no LICENSE anywhere; the
  "CC BY 4.0" seen in MNS is an aggregator claim). Protocol blocks *training* use;
  one-shot internal evaluation needs an explicit protocol decision + disclosure.

Other candidates checked (honest status):
- **FireMan burn B** (`fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE`): 18,619 RGB
  frames (18,171 box-pos, 429 explicit-neg), author CVAT labels, never exported/used;
  raw frames on disk (`data/v5b/raw/fireman/Multiclass/Unimodal/rgb/...`). CC BY 4.0.
  **Quarantined**: best candidate for the Ruokolahti burn = same incident as boreal
  test (H4) → unusable as an independent holdout until the Oulu authors confirm the
  video→burn mapping.
- **WSDataset (gloryvu/VMOKED)**: original 18.4 GB Kaggle archive **BLOCKED** (needs
  Kaggle token). On disk only the author GitHub demo
  (`data/v5b/raw/wsdataset_github_demo/`): 214 unique author-labeled images, 126 new
  explicit negatives; 85+3 are phash dups/near-dups of MultiNatSmoke frames = v4 TRAIN
  (G) material. MIT declared on Kaggle (unverified), permission pending. Too small /
  partially contaminated for a holdout.
- **AuSmoke / MultiNatSmoke**: NOT fresh — v4 models (B1/B4/B6/E_ET/C_EGT/D) trained on
  the T/G lists; AuSmoke holdout = forest test (frozen). Permission still pending.
- **Sheffield ORDA**: the deposit is 1,336 frames / ONE 12-min session, all already
  exported (train/val rec_0007/holdout rec_0008). No unused parts.
- **MmodalFire**: indoor combustion rig; smoke never author-annotated
  (has_smoke_label=False everywhere; session-level fire labels only) → not an
  author-labeled smoke holdout; v5 AUX_LABEL_PACKAGE (300 proposed labels) was never
  human-confirmed.
- **FASDD_UAV**: zip on disk (`data/raw/fasdd_uav/FASDD_UAV.zip`, MD5 verified), never
  extracted/used, but CC BY-SA (viral), ~half HPWREN fixed-camera imagery and mixed
  re-used web sources → unsuitable as an independent author-labeled holdout without a
  major dedup/provenance pass.
- **BAMBI** (13 flights, CC BY 4.0): wildlife UAV footage, no smoke annotation at all
  (negatives are agent-screened only, verdicts blank) → not author-labeled.

## 5. Background pool for synthetic compositing — `data/v6/background_pool_v1.json`

**12,890 images, all TRAIN-role, all author/original no-smoke evidence, all
licence-clear, all verified to exist on disk** (re-checked at build time; 0 missing).

| source | count | split_role | neg_evidence | license | empty-label==verified-negative sound? |
|---|---|---|---|---|---|
| pyro_sdis_train (`data/yolo/pyro_sdis/images/train`) | 3,424 | train | author_exhaustive | Apache-2.0 | YES — single-class smoke dataset from the Pyronear/SDIS operational pipeline; every image annotated for smoke; same semantics used for the matched-FP negative denominator since v1 |
| dfire_train (`data/yolo/dfire_smoke/images`) | 7,833 | train | author_explicit_negative | CC0-1.0 (provenance caveat in repo LICENSE) | YES — exactly the author *none*-category train images; the 944 fire-only (possible unlabeled smoke) never entered E and are excluded |
| boreal_train_karkkila_empty (`data/v4/yolo/E_ET_v1/images/...karkkila...Empty-Images...`) | 37 | train | author_explicit_negative | CC BY 4.0 | YES — author "Empty-Images" clip; corpus lineage records them as author_verified_empty |
| fireman_burnA_train (`data/v5b/yolo/B_v1/images/fireman__*`) | 1,572 | train | author_explicit_negative | CC BY 4.0 | YES as author tag — CVAT `smoke_presence=no` pre-/off-burn flights; NOT human-verified by us |
| sheffield_rec0005_train (`data/v5b/yolo/B_v1/images/sheffield__train__*`) | 24 | train | author_explicit_negative | CC BY 4.0 (depositor declaration, unknown provenance) | WEAKEST — Roboflow-user labels; 6 smoke-visible empties already excluded at export; the 24 kept are one static parked-vehicles scene |

EXCLUDED (and why): **MONIMET** MC124 (agent screening found real slash-burn plumes in
the "negative" pool; zero human verdicts) · **FIgLib/HPWREN** (CC BY-NC-ND 4.0,
eval-only, no derivatives) · everything val/test-role (sdis VAL/TEST sites, Boreal
heinola VAL flights + ruokolahti TEST, fireman_val_s5 / holdout_s10, Sheffield
rec_0007/rec_0008) · FIRESENSE smoke_neg (now negative-REPLAY role; already in v4 train
corpora) · BAMBI (no author smoke annotation) · MmodalFire (indoor; smoke never
annotated) · FireSpot 1,179 negatives (licence pending → reserved as holdout candidate
anyway) · WSDataset demo 126 negatives (licence pending/unverified) · AuSmoke/MNS
(permission pending) · D-Fire fire-only.
**v5 human-reviewed verified negatives: NONE** — `data/v5/review/verdicts.csv` and the
b1/b2 frame-verdict templates have no filled rows, so nothing qualifies under the
human_reviewed clause.

Note: the fireman count is the capped in-corpus train list (600/video); the full burn-A
export holds 3,396 explicit negatives, but only the capped 1,572 exist as local YOLO
image+label files (`B_v1/images`); the remainder exists only in `data/v5b/raw/fireman/`
(re-exportable if more aerial backgrounds are needed).

## 6. Teacher candidates (VAL evidence only — sdis VAL 7,335 imgs / 1,158 neg, box recall at matched neg-image FP, IoU≥0.4)

| id | ckpt (verified exists) | R@5% | R@10% | artifact |
|---|---|---|---|---|
| **expB_y11s (deployed, YOLO11s)** | `/local/home/tianrgua/cvproject/models/expB_best.pt` | **0.203** | **0.329** | reports/v3/matched_fp_val.json#expB_y11s |
| v5bA_ctrl_s3 (arm A YOLO11m, = final-test system S1_A seed 0) | `/local/home/tianrgua/cvproject/models/trained/expV5bA_ctrl_s3/seg05/best.pt` | 0.202 | 0.302 | reports/v5b/val/matched_fp_sdisval.json#S1_A |
| v4A_ctrl_s2 (arm A YOLO11m) | `/local/home/tianrgua/cvproject/models/trained/expV4A_ctrl_s2/seg05/best.pt` | 0.201 | 0.304 | reports/v4/seed_summary.json |
| expE_y11m (YOLO11m) | `/local/home/tianrgua/cvproject/models/trained/expE_mix_y11m_1280_v2/seg05/best.pt` | 0.190 | 0.309 | reports/v3+v4 matched_fp |
| (arm-A seeds s1/s2, never matched-FP-evaluated) | `/local/home/tianrgua/cvproject/models/v4/runs/expV5bA_ctrl_s{1,2}/seg05/best.pt` | — | — | none |

v5b DINO teachers (used for the D/D2/D3 pseudo-label pools, `models/v5b/teacher/*.jsonl`):
- **B4** `v4B4_lora_L` → `/local/home/tianrgua/cvproject/models/v3/runs/v4B4_lora_L/seg05/best.pt` (exists) — sdis VAL R@5% 0.210 / R@10% 0.311 (DINOv3 License).
- **B6** `v4B6_lora_Hplus` → `/local/home/tianrgua/cvproject/models/v3/runs/v4B6_lora_Hplus/seg10/best.pt` (exists) — sdis VAL R@5% 0.227 / R@10% 0.340 (DINOv3 License; flag: shipped best.pt = step 11714, not the run's best).

**Recommendation (one detector teacher): `expB_y11s` (`models/expB_best.pt`).**
Best sdis-VAL matched-FP R@10% of every fixed-camera YOLO (0.329 vs 0.302–0.309 for the
y11m arms, a ~+2.5 pt gap) and tied-best R@5% (0.203 vs 0.202/0.201); also the deployed
model and the fastest (140 ms CPU). Alternative if a larger same-protocol/same-era
teacher is preferred: `v5bA_ctrl_s3` (y11m, v5b-protocol artifact, S1_A); its seeds
s1/s2 were never matched-FP-evaluated. The YOLO family is AGPL-3.0 fine-tune —
consistent with the YOLO11n student. If a cross-architecture teacher is wanted, B6 is
the strongest VAL model overall (R@10% 0.340) but is 12 s/img CPU and DINOv3-licensed.

## Compact machine summary

See the orchestrator return JSON; artifacts: `data/v6/background_pool_v1.json` (12,890
entries + per-source summary + soundness notes), this file.
