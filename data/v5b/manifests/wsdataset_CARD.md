# wsdataset (WSDataset, gloryvu / VMOKED, Vietnam) — V5b data card (2026-10-02)

Agent: V5b phase-1 data agent (SOURCE=wsdataset). Manifest: `data/v5b/manifests/wsdataset_manifest.parquet`
(11,753 rows); per-unit table `wsdataset_units.csv`; build summary `wsdataset_build_summary.json`;
provenance copies in `wsdataset_provenance/`; script `scripts/v5b/wsdataset_build_manifest.py`.

## 0. Status in one paragraph
The ORIGINAL WSDataset (Kaggle `gloryvu/wildfire-smoke-detection`, v7 2024-11-05, 18,398,231,432 bytes,
57,194 files; 11,539 author box-labelled smoke frames from 68 video segments + 13,095 explicit nonsmoke
frames from 59 segments) is **BLOCKED**: the download endpoint 302-redirects to
`https://www.kaggle.com/account/login?...returnUrl=/datasets/gloryvu/wildfire-smoke-detection/versions/1?resource=download`
and the API needs a user token (`~/.kaggle/kaggle.json`, absent on this host; no account may be created by the agent).
Author-controlled hosting other than Kaggle does not exist: the author's repo `GloryVu/VMOKED` (README: "## Dataset
[WSDataset]()" — empty link) ships only a 440-file *demo* sample; no HF / Zenodo / GitHub mirror under the author's
name (HF search for gloryvu/VMOKED/WSDataset: 0 hits; author's 21 GitHub repos checked). What we hold and manifest:
(A) the MultiNatSmoke re-encodes of all 11,539 positives (640x640, model-assisted masks, NOT the author boxes) —
`dup_multinat`, **not new data**, already v4 TRAIN (G) material; (B) the author's GitHub demo sample — 214 unique images,
author labels, of which 126 are explicit negatives (NEW) and 85 are phash-identical resized duplicates of MNS frames (3 further smoke_03 frames are same-scene near-dups, hamming 1–6, counted as new by the hamming-0 rule); new_data = 129 files / 101.1 MB.

## 1. Provenance
- Authors: Vũ Quang Vinh, Trần Công, Trần Anh Đạt (PTIT / IEC PTIT, Vietnam); funded by Lam Dong province. Data = tower
  cameras in Lam Dong (Da Lat area: greenhouses, pine forest, hills), 2022–2024; RTSP/VLC recordings and "camera_N preset"
  exports (filenames). Fixed-camera scenario.
- Papers: (1) Vu, Tran-Anh, Tran, "A resource-efficient system for rapid and accurate forest fire smoke detection using
  Video-based Multiple Object Kinetic Emission Detection", *Array* 30 (2026) 100983, doi:10.1016/j.array.2026.100983, open
  access, article licence CC BY-NC 4.0 (Elsevier API `openaccessUserLicense`; full text behind Cloudflare 403 for curl/WebFetch,
  so its data-availability statement could NOT be read). (2) Vũ et al., JSTIC 1(4) 2023, 118–124 (Vietnamese).
- Kaggle page metadata (unauthenticated `GET https://www.kaggle.com/api/v1/datasets/view/gloryvu/wildfire-smoke-detection`,
  2026-10-02, saved as `wsdataset_provenance/kaggle_api_v1_datasets_view_2026-10-02.json`), verbatim fields:
  `"ref":"gloryvu/wildfire-smoke-detection"`, `"creatorNameNullable":"Glory Vu"`, `"licenseNameNullable":"MIT"`,
  `"totalBytesNullable":18398231432`, `"currentVersionNumber":7`, `"lastUpdated":"2024-11-05T12:35:43.487Z"`,
  `"downloadCount":5818`, `"isPrivate":false`. Description (verbatim excerpt): "We developed the WSDataset through the analysis
  of 31GB of full HD video data, resulting in a comprehensive collection of 68 video segments containing 11,539 smoke-positive
  frames, each annotated with bounding boxes for training object detection models. To address the issue of class imbalance,
  we included 59 nonsmoke video segments comprising challenging samples and totaling 13,095 frames. For a fair evaluation, the
  dataset was divided by video into an approximately 80:20 split, forming a training set (58 smoke videos and 43 nonsmoke
  videos) and a test set (10 smoke videos and 16 nonsmoke videos)."
- Full file listing of the Kaggle archive obtained WITHOUT credentials (`api/v1/datasets/list`, 286 pages x 200) →
  `wsdataset_provenance/kaggle_filelist_2026-10-02.jsonl` (57,194 entries, byte total = 18,398,231,432 = page total). Layout:
  `wildfire_smoke_dataset/wildfire_smoke_dataset/classification/{train,test}/{smoke,nonsmoke}/<unit>/<frame>.jpg`
  (train/smoke 58 units 10,439 frames 6.72 GB; train/nonsmoke 43 units 10,523 frames 7.68 GB; test/smoke 10 units 1,100 frames;
  test/nonsmoke 16 units 2,572 frames 1.70 GB) and detection `{train,test}/{images,labels}/` (train 12,280 images / 12,281 YOLO
  txt, test 3,890 images incl. 16 nonsmoke units / 3,890 txt; test names `<unit>_<frame>`, train names unit-less); plus a stray
  `classification/non_smoke_74/` (218 frames) and `WSdataset_.png`. => the original holds 13,095 (+218) explicit-negative frames
  from 59 (+1) segments that MultiNatSmoke did NOT copy, and author BOXES that MultiNatSmoke replaced by masks.
- GitHub `GloryVu/VMOKED` @ 731b65bab1628b608b089e74a00dad3cdb00071d (sparse clone of `dataset/demo`, 2026-10-02 →
  `data/v5b/raw/wsdataset_github_demo/`, 426 MB on disk incl. git objects; 440 files / 215 unique sha256 because `test/` is a
  byte-identical copy of `train/`). Author notebook `train/vsmoked/train-yolo-nas-on-custom-dataset.ipynb`:
  `CLASSES = ['not smoke','smoke']` with `train/images, train/labels, test/images, test/labels` — i.e. the original detection txt
  use class index 1 = smoke (two-class vocabulary); the demo detection labels are all empty (negatives) so this could not be
  verified on real positive label lines.
- MultiNatSmoke copy: `data/v4/raw/multinatsmoke/MultiNatSmokeDataset/{Train,Test}/WSDataset/{images,masks}` (10,439 + 1,100 =
  11,539 images, 640x640, PNG masks {0,255}); `info.txt` cites the Kaggle page and `https://github.com/GloryVu/VMOKED`.

## 2. Licence / permission (quoted, by evidence class)
- MultiNatSmoke README (github.com/henryzhao0615/MultiNatSmoke, fetched 2026-10-02) licence table row, verbatim:
  `| **WSDataset** | [Link](https://www.kaggle.com/datasets/gloryvu/wildfire-smoke-detection) | MIT |` — SECOND-HAND
  attribution by the aggregator; the MNS aggregate itself carries no licence (v4 card).
- Author's own Kaggle page: `"licenseNameNullable": "MIT"` (API metadata above) — first-party DECLARATION by the uploader
  "Glory Vu", but the MIT licence TEXT inside the archive was not inspected (download blocked) and the Kaggle HTML is
  JS-rendered (WebFetch returned only the title).
- GitHub VMOKED: **no licence** (`GET /repos/GloryVu/VMOKED/license` → 404; repo tree has no LICENSE/COPYING; API
  `"license": null`). Default = all rights reserved; demo images are publicly viewable but no grant to train.
- Array 2026 paper: CC BY-NC 4.0 applies to the ARTICLE, not to the dataset.
- permission_status in the manifest = **pending** for every row (PROTOCOL_V5 §2 lists non-AuSmoke G material as
  permission-pending; nothing here upgrades it to "confirmed": no licence text was read at the source). Recommended action for the
  human: either supply a Kaggle API token (then the MIT file inside the archive can be verified and 13k author negatives + author
  boxes obtained, 18.4 GB — within the single-archive allowance) or e-mail the authors (contacts via PTIT / Kaggle profile).

## 3. What is in the manifest (counts are facts from the build, see wsdataset_build_summary.json)
| material | rows | label_type | annotation_origin | has_smoke | explicit neg | units | new_data | proposed_split |
|---|---|---|---|---|---|---|---|---|
| multinatsmoke_copy (A) | 11,539 | mask (+ derived boxes) | model_assisted | 11,539 | 0 | 67 (66 original videos recovered + 1 ambiguous bucket of 200 `frame_NNNN` frames) | False | train 11,539 (inherited v4 G role); 0 eval/FIgLib hits |
| github_demo_author (B) nonsmoke_00 | 120 | image (folder label) | author | 0 | 120 | 1 | True | train (same camera as MNS smoke_14/14_1, which is v4-train; blocked until permission) |
| github_demo_author (B) non_smoke_71 | 6 | box (empty YOLO txt) | author | 0 | 6 | 1 | True | val (new scene: overcast sky over hills/greenhouses) |
| github_demo_author (B) smoke_03 | 88 | image (folder label) | author | 88 | 0 | 1 | 85 False — `dup_multinat_resized` (phash hamming 0 vs MNS smoke_03 frames, 1280x720 vs 640x640); 3 True (same scene, hamming 1–6) | train (inherited) |
- Author labels: 214 (B); human-reviewed: 0; model-assisted: 11,539 (A masks per WACV26 §3.3: SAM/self-training pseudo-labels
  + authors' manual verification — the "manual verification" is the MNS authors' claim, not our review); unlabeled: 0.
- Smoke vs fire: smoke only everywhere; no fire labels; the original's two-class vocabulary is `not smoke`/`smoke`
  (classification by folder), so its nonsmoke frames ARE smoke negatives (not fire-only).
- Box conversion (named): A rows' YOLO txt = v4 `scripts/v4/build_corpus.py:mask_to_lines` (threshold ≥128, 8-connected
  components, min_area = max(9, 2e-5·W·H), morphological close when >8 components, ≤50 largest) copied from
  `data/v4/yolo/B_G_v1/labels/` → `data/v5b/yolo/wsdataset/labels/` (11,539 files, 12,929 boxes, 1–6 per image, median 1);
  images symlinked in `data/v5b/yolo/wsdataset/images/`. These are MASK-DERIVED, MODEL-ASSISTED boxes, not author boxes.
  B detection negatives: 6 empty label files (`gh__non_smoke_71_*.txt`, author txt copied verbatim). B classification frames: NO
  box files (image-level only, recorded in the manifest).
- Unit recovery: original video ids were recovered for 11,339/11,539 A frames by matching MNS basenames to the Kaggle listing
  (`kaggle_listing_frame_name` 10,239 = classification frame name under a unit folder; `kaggle_listing_unit_frame_name` 1,100 =
  detection `<unit>_<frame>` names); 200 `frame_NNNN` names occur in several units → `mns_wsd_frames_unknown`. MNS's own 8
  "camera" groups were coarse: camera_16_preset_2 = 6 original videos, camera_4 = 37 videos, frames_unknown = 19 videos.
  Original Kaggle split by unit: 10,239 train / 1,100 test (= MNS Train/Test exactly). Unit sizes: 15 units ≥100 frames,
  8 units <30 frames; largest smoke_61 (2,678), smoke_63 (1,673), smoke_68 (918).
- `unit_split_suggestion_70_15_15` (seed 0, by unit over the 67 A units: 47 train / 10 val / 10 "holdout_if_author_boxes_obtained")
  is ONLY a pre-registered suggestion for the day the Kaggle archive (author boxes) arrives; the effective `proposed_split`
  inherits the v4 role (train) because every A frame was in B_G_v1 (x1) and C_EGT_v1 (x2) training lists.

## 4. Overlap / dedup findings (sha256 exact + phash hamming ≤6, v4 convention; vs hash cache of pyro-sdis train/val/test, D-Fire,
FIgLib subset, FIgLib lockbox, Boreal (heinola/ruokolahti tagged), firesense, all MultiNatSmoke usable + FIgLib-family rows)
- A vs eval/held-out sets (pyro-sdis val/test, FIgLib-39, lockbox, heinola, ruokolahti, AuSmoke val/test): **0 exact, 0 near**.
  A vs FIgLib-family MNS subsets: 0. A vs other train material: 0. => nothing to quarantine.
- A internal redundancy: 11,518/11,539 frames have ≥1 other WSDataset frame within hamming ≤6 (22.85 M pairs) — static cameras,
  ~1 fps; effective diversity ≈ 66 videos on ~5 camera views, not 11.5k images.
- B: nonsmoke_00 (120) — every frame within hamming ≤2–6 of MNS smoke_14_1/smoke_14 frames (287 MNS frames) = SAME CAMERA VIEW,
  smoke-free → tagged `samescene_neardup_multinat`, kept as new explicit negatives (matched negatives of a train camera).
  smoke_03 (88) — hamming 0 vs MNS smoke_03 (85) + same-scene hits on smoke_18/23/07/10 → `dup_multinat_resized`, not new.
  non_smoke_71 (6) — min hamming 20 vs everything held → new scene.
- Lesson recorded: for fixed cameras phash ≤6 flags same-VIEW frames, not same frames; exact-frame identity needs hamming 0 / sha.

## 5. Model-assisted sanity inspection (DIAGNOSTIC — not review; `data/v5b/manifests/wsdataset_sanity_check.json`,
`scripts/v5b/wsdataset_sanity_check.py`; model `models/trained/expE_mix_y11m_1280_v2/seg05/best.pt`, CPU, imgsz 1280, seed 20261002;
this model never saw WSDataset — `expB_train.txt` has 0 WSDataset rows)
- 200 random A frames (225 derived boxes): box recall @IoU≥0.4, conf≥0.2 = **95/225 = 42.2%**; images with any detection =
  159/200 = 79.5%. By MNS group: camera_16 (tiny horizon smoke) 58/105 boxes; camera_4 **1/38 boxes but 35/38 images detected**
  (the mask-derived boxes do not coincide with where the detector fires — loose/offset pseudo-masks or detector on a different
  plume; needs a human look, not a conclusion); frames_unknown 12/42 boxes, 40/41 images; rtsp/other 22/34.
- Explicit negatives (B, 126): any detection ≥0.3 on **28/126 = 22.2%** (all 28 in nonsmoke_00 — hazy forest/lake view, max conf
  0.54; non_smoke_71 0/6). Demo smoke_03 sequence (image-level positive): 88/88 frames fired at conf ≥0.2.
- My own visual audit (not a review of record): 30 random A masks/boxes in `reports/v5b/wsdataset_audit_sheet_mns.jpg` — all 30
  masks sit on a visible plume or at the far-horizon smoke (camera_16), none on a non-smoke object; boxes are tight to the mask;
  24 B frames in `reports/v5b/wsdataset_audit_sheet_github.jpg` — nonsmoke_00/non_smoke_71 show no smoke (haze / low cloud,
  genuine hard negatives); smoke_03 shows a clear plume.

## 6. Limitations
1. No new positive SCENES; the only new material is 126 author-labelled explicit negatives (+3 same-scene smoke frames), 101.1 MB, permission pending.
2. No timestamps / fps / cadence for any frame (filenames: an export-time suffix identical on 5,395 files; `frame_NNNN` only);
   no valid-hours figure is possible, so these negatives cannot count toward PROTOCOL §6 (≥2 h, ≥3 units, human-verified).
3. A labels are MNS pseudo-masks → derived boxes; author boxes exist only in the blocked archive.
4. Licence: declared MIT (author Kaggle page) but text unverified; GitHub demo unlicensed.
5. Geographic/scene diversity is low (~5 fixed camera views, Lam Dong highlands, greenhouses) — out-of-domain for pyro-sdis
   France/Boreal Finland but relevant as fixed-camera hard negatives/confounders (haze, low cloud).

## 7. Budget
Downloaded this phase: 426 MB on disk (GitHub sparse clone incl. git objects and the byte-identical test copy); unique new images
101.1 MB (129 files). Kaggle archive (18.4 GB) NOT downloaded. No cluster jobs (CPU only; no registry reservation needed).
