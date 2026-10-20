# FireSpot-CNX-r1 — data card (V5b phase 1, agent SOURCE=firespot, 2026-10-02)

**Status: permission_status = PENDING for every row -> NOT usable for training this round (PROTOCOL_V5 §2).**
Manifest: `data/v5b/manifests/firespot_manifest.parquet` (3,996 rows, 29 cols; csv.gz copy alongside). No YOLO export was
written (`scripts/v5b/firespot_to_yolo.py` is a guarded converter that refuses to run until written permission is on file).

## 1. Provenance
- Repo: https://github.com/Biometrix-4/FireSpot-CNX (default branch `main`, last update 2024-11-12; tree = `README.md` + `trainandtest.ipynb` only;
  GitHub API `license: null`). Authors (README): "Natthaphol P., Warit P., Kanokvate T., Jessada K." — NECTEC with Pa Miang, Nong Yaeng and
  Choeng Doi municipalities, Chiang Mai, Thailand. Paper: Pornpholkullapat et al., "FireSpot: A Database for Smoke Detection in Early-stage
  Wildfires", iSAI-NLP 2023, doi:10.1109/iSAI-NLP60301.2023.10354807 (abstract via Semantic Scholar; full text not retrieved, IEEE paywall).
- Download link in README ("Download" section): Google Drive folder `1udDy_F7nGbjUiKUY0FOQBCv2lD8zp47b` ("WildFire") containing
  `FireSpot-CNX-r1.zip` (id `1s6tF48CL87TCYJQMyD6cvE0wW9UkPkjQ`) and `FireSpot-Seg-CNX-r1.zip` (id `1xP6YozUDxVEFpwNmv_aunUysHQBRnJ5d`, NOT downloaded).
- Acquired: `data/v5b/raw/firespot/FireSpot-CNX-r1.zip`, 16,614,395,090 bytes, sha256 `143fefded0860a94f943652dbe8dc70d9f6624794df98038d3655fc616062e67`
  (archive mtime 2024-11-11; image entries dated 2023-07-23). Extracted `dataset/images/*.jpg` (3,996, 17.29 GB) and `dataset/labels/*.txt` (3,996);
  `__MACOSX/` resource forks and `.DS_Store` ignored. New-data volume used by this agent: 16.6 GB download (33.9 GB on disk incl. extraction).
- Archive contents audited by listing: NO licence, readme, citation or terms file inside the zip.

## 2. Licence / permission (verbatim)
The only use-related statement anywhere in the repo or archive is the README "Citation" section:
> "Please cite the following paper if you use our image database:"
> "Pornpholkullapat, N., Phankrawee, W., Boondet, P., Thein, T.L.L., Siharath, P., Cruz, J.D., Marata, K.T., Tungpimolrut, K. and Karnjana, J., 2023, November. FireSpot: A Database for Smoke Detection in Early-stage Wildfires. In 2023 18th International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP) (pp. 1-6). IEEE."

No licence is named. The "CC BY 4.0" attributed to FireSpot in `data/v4/manifests/multinatsmoke_CARD.md` comes from the MultiNatSmoke README
licence table — an aggregator's field, not evidence. => `licence = UNSPECIFIED`, `permission_status = pending`; training use is NOT allowed
this round. Action: email the corresponding author (Jessada Karnjana, NECTEC) asking for the intended licence / permission for internal
research training use (draft to be added under reports/v5b/, not reports/v5/ which is frozen).

## 3. Label format (audited on all 3,996 files + 48 images viewed on contact sheets)
- `label_type = box`, YOLO txt, one line per box `0 xc yc w h` (normalised, 5 fields; 4,058 lines total, all class id 0). Class vocabulary: `0 = Smoke`
  (single class; README table lists only "Smoke"). **No fire class** -> `has_fire_label = False` everywhere; negatives are the authors' explicit
  "None" (no-smoke) category, i.e. true smoke negatives, not fire-only images.
- Counts: positives 2,817 files / 4,058 boxes (README states 2,817 boxes — it counts images; 1,952 images have 1 box, 652 have 2, 121 have 3,
  54 have 4, 18 have 5, 20 have 6–9). Explicit negatives (empty label file) 1,179. Unlabeled 0. Filename suffix `-1`/`-0` agrees with label content 100 %.
- Box geometry: median box area 2.2 % of the image (p10 0.28 %, p90 15.2 %); 14 boxes exceed [0,1] by <1e-3 (clip on export).
- `annotation_origin = author` for all rows (no human review by us; our inspection is model-assisted/visual spot-checking only).
- Visual spot check (data/v5b/manifests/_inspect/firespot_pos_sheet{1,2,3}.jpg, firespot_neg_sheet1.jpg; list in firespot_inspected_list.txt):
  36 positives — boxes sit on visible plumes (many tiny faint plumes on tree lines; loose boxes around diffuse near-field smoke from burning
  piles); 12 negatives — no smoke visible. Imagery = handheld smartphone-style photos (4032x3024, 3264x2448, 1479x1109) plus 682 tower-camera
  snapshots (1280x722); NOT a continuous fixed-camera feed; EXIF stripped (0/3,996 have DateTime) -> no timestamps, no valid-hours figure.

## 4. Grouping units (recovered from filenames `SS-EE-CC-NNN-L`)
- `SS` = site 01–05 (5 sites; README names 3 municipalities), `EE` = session within site (29 sessions, 123–139 images each — a designed sampling),
  `CC` = device/camera slot (`01`..`06`, `tower`; Appendix A shows each slot maps to one resolution, i.e. a device, `tower` = 1280x722 tower camera),
  `NNN` = sequence number, `L` = label. 18 names deviate (4-digit NNN / 2-digit NNN) but parse the same way.
- **Sessions are NOT independent** (phash evidence): near-dup rate (hamming<=6) 9.4 % for same-session+device pairs, 0.01 % same-session-other-device,
  0.01 % same-site-other-session (76 pairs across 20 sessions), **0 cross-site pairs**; 9 byte-identical groups span sessions, 0 span sites;
  visually site 01 is one reservoir scene recurring in sessions 02–07. => `parent_unit = firespot_site_SS` (5 units); `session_unit` kept as a column.
- `proposed_split` by SITE: train = sites 01, 02, 05 (2,754 img: 1,917 pos / 837 neg, 20 sessions); val = site 03 (690: 484/206, 5 sessions);
  holdout = site 04 (552: 416/136, 4 sessions) — 3/1/1 units, 69/17/14 by images. Holdout is author-labelled and reserved; nothing quarantined.

## 5. Overlap / dedup (`scripts/v5b/firespot_dedup.py`; reference = `_cache/firespot_ref_hashes.parquet`, 137,026 rows: pyro-sdis all sites,
D-Fire, FIgLib subset+lockbox, Boreal (heinola=val, ruokolahti=quarantine), FIRESENSE, MmodalFire, MultiNatSmoke incl. its FIgLib-family and AuSmoke units)
- vs FIgLib / held-out sets (FIgLib 5,034 local + 23,863 MNS FIgLib-family, Boreal heinola & ruokolahti, pyro-sdis TEST sites, AuSmoke val/test): **0 hits**
  (exact or phash<=6). One phash collision at hamming 0 with a pyro-sdis TRAIN image (`03-02-04-017-1` vs `sdis-07_brison-110_2024-02-21T11-37-09`) is
  visually a different scene (`_inspect/firespot_vs_sdis_hit.jpg`) -> flagged `phash_false_positive`, no action.
- vs MultiNatSmoke "FireSpot" subset (2,865 rows, v4 G component): 0 byte-identical (MNS re-encoded), 2,862/2,865 MNS rows matched by phash<=6:
  2,768 same-stem (hamming 0: 2,631; hamming 2: 137), 93 MNS stems were renamed between releases and matched at hamming 0 by 99 r1 images
  (`dup_flags=mns_renamed:<mns_stem>`); 4 same-name pairs differ in content (>6). Rows: `dup_multinat` 2,705 (positives that are MNS copies),
  `dup_multinat_conflict` 141 (+1 also an internal exact dup): **r1 EXPLICIT NEGATIVES that MNS ships WITH a non-empty smoke mask** (MNS has 115 `-0`
  stems, median mask 26,757 px). Visual check of 8 (`_inspect/firespot_neg_with_mns_mask_sheet.jpg`): masks fall on a bare hillside, cloud edges and
  tree edges with no visible smoke -> MNS pseudo-masks on these images are spurious; the author negatives stand. Consequence: every v4 model trained
  on G (arms B/C/D) has seen 2,867 of these 3,996 images by dup_flags (incl. 407 of the 552 holdout-site images) and treated 142 author-negatives as positives;
  only the 1,129 rows without an MNS flag (145 in holdout) are unseen by G-trained models. E-only models (arm A, expE v2) never saw FireSpot.
- Internal: 65 byte-identical groups (136 files; the 71 non-first copies -> `dup_class=internal_exact_dup`, `canonical_stem` given), all label-consistent.
  374 phash-identical groups (886 files, bursts); 7 groups mix labels (19 files, `internal_neardup_label_conflict`) — bursts where smoke appears/vanishes
  or inconsistent labelling; needs a human look before any use.

## 6. Model-assisted sanity DIAGNOSTIC (not review) — `scripts/v5b/firespot_sanity_diag.py`, `_cache/firespot_sanity_summary.json`
200 random labelled images (seed 20261002; 137 positives / 192 author boxes, 63 explicit negatives; 24 of them from the holdout site — scoring only,
no selection), `models/trained/expE_mix_y11m_1280_v2/seg05/best.pt`, CPU, imgsz 1280:
- author-box recall (IoU>=0.4) at conf 0.2: **30.7 %** (59/192); positive images with >=1 author box matched: 43.1 %; (conf 0.1: 35.9 % / 50.4 %).
- explicit-negative images with any detection >= 0.3: **0/63 (0.0 %)**; >= 0.2: 1/63; >= 0.1: 1/63.
Reading: the fixed-camera-trained model is silent on the author negatives (consistent with them being clean) but misses most author boxes — a
domain shift (handheld ground-level photos, near-field dense smoke, loose boxes on diffuse plumes), not evidence of label error.

## 7. Limitations
No licence text (pending); EXIF stripped, no timestamps/cadence -> unusable for episode/hour metrics; handheld photos + tower snapshots, so the data
fit a "forest-ground / near-field" scenario rather than the fixed-camera or aerial scenarios of PROTOCOL_V5 §1; only 5 grouping units; Thai dry-season
vegetation; boxes loose for diffuse smoke; 14 out-of-range boxes; MNS copies of 2,846 images already inside the v4 G corpus (contaminates G-trained
models w.r.t. any FireSpot evaluation); 19 burst frames with conflicting labels; FireSpot-Seg-CNX-r1 (masks) not acquired.
### Appendix A — resolution x camera_pos (counts)

camera_pos   01   02   03   04   05   06  tower
row_0                                          
1280x722      0    0    0    0    0    0    682
1479x1109   627  259   74    0    0    0      0
3264x2448     0    0    0    0    0  601      0
4000x3000     0    0    0  130    0    0      0
4032x2613     0    0    0    0    1    0      0
4032x2673     0    0    0    0    1    0      0
4032x2719     0    0    0    0    1    0      0
4032x2723     0    0    0    0    1    0      0
4032x2775     0    0    0    0    1    0      0
4032x3024     0  355  310  472  481    0      0

### Appendix B — EXIF

images with EXIF DateTime: 0 of 3996
capture dates (YYYY:MM:DD) top: {}
camera models: {}
dates per site: {}
