# fireman — FireMan-UAV-RGBT (University of Oulu), Multiclass package

Status: V5b phase-1 data acquisition (2026-10-02). Agent-produced; NOTHING here is human-reviewed.
Manifest: `data/v5b/manifests/fireman_manifest.parquet` · YOLO export: `data/v5b/yolo/fireman/` · units: `data/v5b/manifests/fireman_video_units.json`.

## Provenance
- Zenodo record https://zenodo.org/records/13732947 (DOI 10.5281/zenodo.13732947), version 1.0.0-beta.2, published 2024-09-09
  (concept DOI chain: 10.5281/zenodo.12628284 = 1.0.0-alpha, README only). Access right on the record: `open`.
- Authors (README): S.D.M.W. Kularatne, Kamand Sedaghat Shayegan, Constantino Álvarez Casado (contact), Janne Rajala, Tuomo Hänninen,
  Miguel Bordallo López (contact), Le Ngu Nguyen (contact) — University of Oulu. Funding: Academy of Finland 6G Flagship 346208 and grant 348008.
- Paper: Kularatne et al., "FireMan-UAV-RGBT: A Novel UAV-Based RGB-Thermal Video Dataset for the Detection of Wildfires in the Finnish Forests",
  IEEE ETFA 2024, DOI 10.1109/ETFA61755.2024.10710657 (Crossref). Full text NOT obtained (IEEE paywall; oulurepo.oulu.fi returns 403 to scripted access;
  search engines blocked) — so per-burn video tables, if any, were not available; burn identification below is from frame content only.
- README (verbatim excerpts): "Captured over four controlled burnings in Finland during 2022-2023" · "Locations: Ruokolahti, Perho, Kestilä, Taivalkoski (Finland)" ·
  "Dates: July 6, 2022; August 16, 2022; June 14, 2023; August 26, 2023" · "Equipment: DJI Matrice 30T and Matrice 300 drones equipped with Zenmuse H20T and H20N cameras" ·
  "Videos: 34 RGB videos, 20 RGB-Thermal pair videos" · "Annotation Methods: Manual: Human annotators drawing bounding boxes and tagging; Automatic: Segmentation models
  (e.g., SAM, MIL) for thermal images; Semi-Automatic: Projecting thermal annotations to RGB images using image pair registration; Tools: CVAT platform" ·
  "Classification Labels: Binary: 'Fire', 'No_Fire'; Multiclass: 'Fire', 'Smoke', 'Smoke-Fire', 'No-Fire-No-Smoke', 'Weather', 'Vehicle', 'Human', 'Lake', 'Building'".
- CHANGELOG 1.0.0-beta.2 (verbatim): "RGB-unimodal total frames: 69942 (Test - 6019, Val - 5634, Train - 58286)" · "IR-unimodal total frames: 33059 (Test - 5077, Val - 1689, Train - 26298)" ·
  "The frames in the train, test and validation subsets belong to different videos. There are not frames from the same video in the different subsets."
  (Actual jpg counts in the archive: RGB 69,865; IR 33,058 — see Counts.)

## Download (what was fetched, how, verification)
- Zenodo API file list (6 files, 39.40 GB total): Binary.zip 6.351 GB (md5 37cbb820…), **Multiclass.zip 33.045 GB (md5 ddc790b901bfc51e196976145f49ea5d)**, README.txt, CHANGELOG.txt, 2 sample JPGs.
- Fetched: README/CHANGELOG/sample JPGs, then **Multiclass.zip only** (single official 33.0 GB archive — allowed as the "single 30–35 GB archive" exception; Binary.zip NOT downloaded
  because its labels are Fire/No_Fire only, which say nothing about smoke). Its content listing and three small CSVs were read via HTTP range requests (no bulk download).
- Downloader: `scripts/v5b/fireman_download.py` (parallel ranged GET, 12 streams, ~20 MB/s); md5 verified `ddc790b901bfc51e196976145f49ea5d` = Zenodo checksum. Extracted to
  `data/v5b/raw/fireman/Multiclass/` (102,923 jpg + 44 annotations.xml + 2 csv; unzip clean). The zip is kept alongside for reproducibility.
- New acquisition volume charged to the shared budget: **33.0 GB** (+ ~27 MB of range-request peeks).

## Licence (verbatim from the source itself)
- Zenodo record metadata (API `/api/records/13732947`): `"license": {"id": "cc-by-4.0"}`, `"access_right": "open"`.
- README.txt inside the record: `License: Creative Commons Attribution 4.0 International (CC BY 4.0)`.
- permission_status = **confirmed** (open CC BY 4.0 on the record and in the README; attribution to the authors/Zenodo DOI required on any publication or redistribution).

## Package structure (audited by listing and by opening files)
```
Multiclass/Unimodal/{rgb,thermal}/{train,val,test}/video_NNN/{images/frame_XXXXXX.jpg, annotations.xml}   + rgb/rgb_modality.csv, thermal/thermal_modality.csv (plain path lists)
```
- 34 RGB videos (train 24 / val 5 / test 5), 10 thermal videos (train 6 / val 2 / test 2). Every frame of each video is included (full frame rate → heavy within-video redundancy).
- `annotations.xml` = CVAT "for images" 1.1 export (task `mode=interpolation`, 5-frame overlap). Per `<image>`: `<box label=… source="manual|semi-auto" xtl ytl xbr ybr>` and
  `<tag label=… source="manual"><attribute …>`. No polygons/masks in the Multiclass package.
- RGB label vocabulary: tags `smoke` (smoke_presence yes|no; smoke_intensity low|medium|high), `fire` (fire_presence yes|no), `camera` (camera_perspective side_view|top_view;
  smoke_interaction clear_view|through_smoke), `weather` (condition; sky_brightness); rectangles `smoke_region`, `fire_region`, `lake`, `vehicle`, `human`, `building`.
- Thermal label vocabulary: tags `smoke_tag` (smoke_presence **unknown** everywhere audited), `fire_tag`, `camera`; rectangles `fire_region`, `building`. Thermal frames carry NO smoke supervision.
- Frame resolutions: RGB 3840×2160 (native H20T wide), 1920×1080, and **rectified RGB at thermal geometry** for the 10 RGB–thermal pair videos (640×512 for H20T pairs,
  958×760/764/768 for the other camera). Thermal 640×512 and 958×7xx. Five H20T thermal videos have a burned-in DJI overlay (model, timestamp, GPS, altitude).
- JPEGs carry no EXIF; the XML `<source>` is the renamed task video (`FireMan_RGB_Train_Video_001.mp4`) — no original DJI filenames, GPS or dates anywhere in the package.

## Smoke-label semantics used in the manifest
- `has_smoke_label` = smoke_presence==yes OR ≥1 `smoke_region` box. `n_smoke_boxes` = count of `smoke_region` boxes (`smoke_boxes` column holds pixel xyxy).
- `is_explicit_negative` = smoke_presence==**no** AND no smoke box (an author statement about smoke). `fire_presence=no` or Binary `No_Fire` are NOT smoke negatives and are not used as such.
- `label_type` (smoke): `box` if smoke boxes; `image` if only a smoke tag (yes or no); `none` if no smoke tag/box (e.g. all thermal frames, frames absent from the XML).
- `annotation_origin` = `author` for every frame carrying any author tag/box, else `unlabeled`. Nothing here is `human_reviewed` (no human on our side looked at labels) and nothing is
  `model_assisted` by us. NOTE the authors' own `smoke_region` boxes are predominantly `source="semi-auto"` (thermal→RGB projection per README); the per-frame CVAT source is kept in
  `author_box_source` (manual | semi-auto | mixed | none). We did not convert any annotation type into another.
- `label_conflict` flags (excluded from YOLO export): presence_yes_no_boxes, presence_no_but_boxes, boxes_without_presence_tag, tag_conflict:<label> (two tags with different attribute values on one frame).

## Counts (manifest, 102,923 rows = 69,871 RGB + 33,052 thermal; XML↔image match 44/44 videos, 0 decode errors)
| RGB smoke label | frames | origin |
|---|---|---|
| box (`smoke_region`, 1–4 boxes; 69,232 boxes total) | 64,867 | author (CVAT `source`: file 64,341 · manual 346 · semi-auto 180) |
| image-level explicit negative (smoke_presence=no, no box) | 4,979 | author |
| image-level positive without box (presence=yes, 0 boxes) → flagged `presence_yes_no_boxes`, NOT exported | 25 | author |
| other conflicts (`presence_no_but_boxes` 3, `tag_conflict:camera` 1) → NOT exported | 4 | author |
| human_reviewed / model_assisted (by us) | 0 / 0 | — |
- Fire: `fire_presence=yes` or `fire_region` on 62,545 RGB frames; 2,244 frames have smoke=yes & fire=no; 128 have fire=yes & smoke=no (fire-only frames are NOT smoke negatives and are not exported as such).
- Thermal: 33,052 frames, `smoke_presence` = unknown/yes (yes only in three Taivalkoski videos and val_t_02, i.e. inherited video-level tags, no smoke boxes); 34,585 `fire_region` boxes. Thermal is recorded in the manifest (`modality=thermal`), NOT exported, NOT used as model input — available for a future RGB-T ablation (10 RGB–thermal pairs, 19,459+8,530+... frames per burn as below; `paired_video` column).
- Smoke intensity (author): medium 34,594 · high 27,995 · low 7,282. Camera: side_view 57,788 · top_view 10,520 · through_smoke 6,556.
- Resolutions (RGB): 3840×2160 33,575 · 1920×1080 3,233 · 640×512 20,090 (rectified H20T pairs) · 958×760/764/768 12,973 (rectified M30T pairs).
- Boxes are large-region "smoke_region" boxes (often most of the frame), predominantly imported (`source=file`) → interpolated/projected, not tight manual instances.

## Units, burn identification and proposed split (see `fireman_video_units.json` for per-video evidence)
| parent_unit | RGB frames | videos (RGB) | box-pos | explicit-neg | proposed_split | identification |
|---|---|---|---|---|---|---|
| fireman_burnC_taivalkoski_2023-08-26 | 19,459 | 5 (009,010,011,val_002,test_002) + 5 thermal | 19,459 | 0 | **holdout** | **confirmed**: DJI overlay on paired thermal frames "2023-08-26 11:58–12:30, 65.45–65.48°N 28.41–28.46°E"; RGB↔thermal pairing by identical frame counts/scenes |
| fireman_burnA_sunny_bog | 23,263 | 15 (001–008,018,020–022,val_001,val_005,test_001) + 1 thermal | 19,861 | 3,396 | **train** | visual only: one sunny midday, green string-bog, one burning treed island; 005/018/test_001 = smoke-free flights over the same landscape |
| fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 18,619 | 13 (012–014,016,017,019,023,024,val_003,val_004,test_003–005) + 3 thermal | 18,171 | 429 | **quarantine** | visual only: evening→dusk light, late-summer colours, val_003 shows a Saimaa-like archipelago → best candidate for the README's Ruokolahti / Aug-16-2022 burn (Boreal Ruokolahti = 2022-08-15 = v4 final-test site) |
| fireman_burnD_overcast_clearcut_edge | 8,530 | 1 (015) + 1 thermal | 7,376 | 1,154 | **val** | visual only: overcast, standing-pine fire at a clearcut edge; singleton |
- The README's 4 burns ↔ 4 units above is my reconstruction; only C is metadata-confirmed. Dates for A/D are "2022-07-06 or 2023-06-14"; B's date "2022-08-16" is inferred from season+README.
  No Heinola/Evo/Karkkila candidate (FireMan lists none; 2022-07-06 coincides with Boreal Karkkila's date but no FireMan scene resembles the Karkkila/Evo/Heinola Boreal frames, and the README names Perho/Kestilä/Taivalkoski/Ruokolahti).
- Split is by whole burn (protocol §2), so 70/15/15-by-units is impossible with 4 units: by units 1/1/1/1, by RGB frames train 33% · val 12% · holdout 28% · quarantine 27%.
  Holdout = the only GPS-confirmed independent burn (but 640×512 rectified RGB). Val is a single video; Boreal heinola remains the protocol's aerial VAL.
- Within-video redundancy is extreme (every frame at 25–30 fps; e.g. 7,703 consecutive frames in video_003). Any training use should stride-sample (≥1/10) and count a video, not a frame, as the sample unit.

## Overlap / dedup findings (sha256 exact + phash hamming≤6 vs 96,456 reference hashes: pyro-sdis all sites, D-Fire, FIgLib subset+lockboxes, Boreal A/C, FIRESENSE, MultiNatSmoke)
- Exact sha256 duplicates: **0**. Within-FireMan exact duplicates: 0.
- phash≤6 hits: **52**, all at distance exactly 6, all against two D-Fire web images (WEB02564, WEB04678): 31 thermal t_06 frames + 2 thermal test_t_02 frames (uniform magenta frames) and 19 consecutive RGB video_024 frames. Visually checked 6 pairs: unrelated scenes → recorded as `dup_class=dup_near_dfire` with `dup_note=...false_positive`; no split change (D-Fire is train material anyway).
- No match to FIgLib, pyro-sdis VAL/TEST, Boreal heinola/ruokolahti frames, or MultiNatSmoke (→ FireMan is NOT inside MultiNatSmoke; no `dup_multinat` rows). MmodalFire (indoor) was checked by sha256 via its manifest only: 0.
- Incident-level overlap with Boreal is therefore handled by unit assignment, not by frame hashes: B → quarantine (Ruokolahti candidate). If the authors confirm B ≠ Ruokolahti, release B; if they identify another unit as Ruokolahti, quarantine that one instead.

## YOLO export (`data/v5b/yolo/fireman/`, images symlinked, class 0 = smoke)
| split | images | box files | boxes | empty (explicit-neg) files | skipped conflicts |
|---|---|---|---|---|---|
| train (burn A) | 23,254 | 19,858 | 21,654 | 3,396 | 9 |
| val (burn D) | 8,530 | 7,376 | 7,376 | 1,154 | 0 |
| holdout (burn C) | 19,458 | 19,458 | 19,458 | 0 | 1 |
- Quarantine (B) and all thermal rows are NOT exported. 48,488 label lines validated (class 0, coords in [0,1]). Image-level-only positives: 0 exported (the 25 are conflicts → `conflicts.txt`); `image_level_only.txt` empty.
- Explicit negatives are author "smoke_presence=no" frames: train 3,396 (video_018 1,811 · 005 836 · 004 338 · test_001 329 · 003 82), val 1,154 (video_015). They are pre-/off-burn flights over the burn landscapes — exactly the matched aerial negatives the V5 protocol asks for, but **not human-verified by us**.

## Model-assisted sanity inspection — DIAGNOSTIC ONLY (not review)
`scripts/v5b/fireman_sanity_inspect.py` · `models/trained/expE_mix_y11m_1280_v2/seg05/best.pt`, CPU, imgsz 1280, 200 random labelled RGB frames (100 box-positive + 100 explicit-negative; quarantine excluded) · `data/v5b/manifests/_inspect/fireman_sanity_inspect.json`
- Author-box recall @IoU≥0.4, conf≥0.2: **100/104 = 0.96**; positive frames with any det ≥0.3: 97/100. Misses: video_015 frame 8515 (no det), video_004 frame 1821 (no det), two video_003 frames with 2 author boxes where one was hit.
- Explicit-negative frames with any det ≥0.3: **21/100 (0.21)** — all 21 from video_018 (max conf 0.30–0.52). Agent-viewed 8: the detector fires on light-grey bare rock/ground patches in the mire, not on smoke → model false positives on hard aerial negatives, not label errors (agent judgement, not human review). Negatives from 005/015/004/test_001/003 (59 frames) produced no detection ≥0.3.
- Sheets: `_inspect/sanity_neg_with_det.jpg`, `_inspect/fireman_rgb_sheet_0{0..4}.jpg`, `_inspect/fireman_thermal_overlays*.jpg`, `_inspect/cmp_*.jpg`, `_inspect/fireman_scene_embed.json` (DINOv2 ViT-B/14 per-video similarities; clusters by scene type, not location).

## Limitations / honesty notes
1. Location/date per video is NOT in the package; only burn C is confirmed (thermal overlay). Units A/B/D and the Ruokolahti quarantine are agent visual judgements. **Ask the authors** (contacts in README: constantino.alvarezcasado@oulu.fi, miguel.bordallo@oulu.fi, le.nguyen@oulu.fi) for the video→(location, date) mapping before any training use; email NOT sent by this agent.
2. The ETFA paper full text was not obtained (IEEE paywall; oulurepo 403); if it tabulates videos per burn, it supersedes §Units.
3. Author boxes are large "region" boxes, mostly `source=file` (imported/interpolated) — fine for presence, poor for localisation; IoU-based metrics against them are weak. 25 presence=yes frames have no box; 3 presence=no frames have boxes.
4. Rectified pair videos (640×512, 958×7xx) are low-resolution RGB; the holdout (burn C) is entirely 640×512.
5. Nothing is human-reviewed; `annotation_origin=author` throughout; thermal smoke labels are `unknown`.
6. CHANGELOG counts (69,942 RGB / 33,059 IR) differ slightly from the archive (69,871 / 33,052).
7. Not downloaded: Binary.zip (6.35 GB; Fire/No_Fire only — `No_Fire` ≠ no smoke; its 10 RGBT pair videos exist here already as Multiclass rectified pairs).

## Reproduction
```
python scripts/v5b/fireman_download.py "https://zenodo.org/records/13732947/files/Multiclass.zip?download=1" data/v5b/raw/fireman/Multiclass.zip ddc790b901bfc51e196976145f49ea5d 12
cd data/v5b/raw/fireman && unzip -q Multiclass.zip            # (8 parallel pattern unzips were used)
python scripts/v5b/build_ref_hashes.py                        # reference sha256/phash table with inherited roles
python scripts/v5b/fireman_build_manifest.py                  # uses data/v5b/manifests/fireman_video_units.json
python scripts/v5b/fireman_export_yolo.py && python scripts/v5b/fireman_sanity_inspect.py 200 && python scripts/v5b/fireman_summarize.py
```
