# sheffield_uav — data CARD (V5b phase 1, written 2026-10-02 by the sheffield_uav data agent)

Status: DOWNLOADED, AUDITED, MANIFESTED. No model has been trained on this material. Nothing in
STATUS.md / reports/v4 / reports/v5 was edited. Acquisition volume: 1.00 GB archive (1.9 GB on disk incl. extraction).

## 1. Provenance (what the official record says)
- Record: University of Sheffield ORDA (figshare-hosted), article 33080681, v1, DOI 10.15131/shef.data.33080681.v1,
  published 2026-07-26. Title "Aerial Imagery Dataset for Wildfire Detection using UAVs". Authors Aditya Shrikhande,
  Lyudmila Mihaylova (Sheffield). Citation string as given by the record is in
  data/v5b/raw/sheffield_uav/figshare_article_33080681.json (field `citation`).
- Fetched via the figshare API https://api.figshare.com/v2/articles/33080681 (ORDA is a figshare institutional
  portal; `url_public_html` in the response is the orda.shef.ac.uk landing page). The ORDA HTML landing page returns
  an HTTP 202 bot-challenge page to curl (the orda.shef.ac.uk/v2 API mirror likewise returns 202 with an empty body),
  so licence evidence below comes from the figshare API metadata of the record and from the
  text files inside the archive itself (not from an aggregator).
- Files: Custom_YOLO_Dataset.zip (1,004,079,935 B, md5 67ca30197d03a86155432ceba33ab73e — verified after download),
  data.yaml, README.dataset.txt, README.roboflow.txt, readme.txt. All five kept under data/v5b/raw/sheffield_uav/.
- The archive is a **Roboflow export** (README.roboflow.txt: "exported via roboflow.com on July 24, 2026";
  workspace `fire-detection-wqtsj`, project `my-first-project-kvseq`). README.dataset.txt says "Provided by a Roboflow user".
  The depositors' readme.txt describes the images as "captured from drone/aerial footage (filenames show DJI drone
  video frame exports)". No capture location, platform, date, operator or source-video statement is given anywhere
  in the record. No publication by Shrikhande+Mihaylova describing this footage was found (arXiv API, Crossref).
  The Roboflow project page sits behind a Cloudflare human-verification wall (not bypassed) -> origin page NOT inspected.

## 2. Licence (verbatim, from the source itself)
- Record `license` field: `{"value": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}`
- Record description: "Uploaded on Roboflow, licensed CC BY 4.0, no pre-applied augmentation."
- Archive readme.txt, section SOURCE / LICENSE: "Exported via Roboflow on July 24, 2026. / License: CC BY 4.0
  (attribution required) / Original project: https://universe.roboflow.com/fire-detection-wqtsj/my-first-project-kvseq"
- Archive README.dataset.txt: "Provided by a Roboflow user / License: CC BY 4.0"; data.yaml: `roboflow: license: CC BY 4.0`.
- Record custom field "Sharing and access restrictions": "The uploaded data can be shared openly".
permission_status = **confirmed** (CC BY 4.0 stated on the official record and inside the archive). Caveat: the
licence is the depositor's declaration over a Roboflow-user upload whose underlying footage owner is not identified;
attribution target = the ORDA citation above.

## 3. What the data actually is (audited by parsing all 1,336 label files + viewing 53 frames with boxes)
- 1,336 JPEG frames, ALL 1920x1080 RGB, EXIF fully stripped (Roboflow re-encode; one JPEG quant/JFIF signature for
  all files) -> camera make/model/GPS cannot be read.
- Filenames: `DJI_<YYYYMMDDhhmmss>_<NNNN>_<W|Z>_MP4-<frame>_jpg.rf.<hash>.jpg`. This is the DJI Zenmuse H20-series
  (M300-class) dual-stream naming: `_W` = wide camera, `_Z` = zoom camera. **8 video streams from 6 recordings**,
  numbered 0003..0008, all stamped **2022-07-21 14:07:20 -> 14:19:32 (camera local time)**, i.e. one ~12-minute UAV
  session at one site. Recordings 0004 and 0006 have co-temporal W+Z streams of the same scene (verified visually,
  same frame indices). Frame indices run 0..N consecutively (dense video sampling); 292 frames (21.9%) have a phash
  identical to an earlier frame of the same stream.
- Scene (visual): flat temperate grassland/heath prescribed-style burn, dirt tracks, deciduous trees, crew vehicles
  (white van + dark pickup) parked on a track, nadir-to-oblique views at low altitude; white/grey smoke plumes with
  visible flame fronts. NOT boreal pine forest (Boreal/FGI), NOT the Arizona FLAME scenes, NOT fixed-camera.
- Labels: YOLO txt, 2 classes, **0 = fire, 1 = smoke** (data.yaml `names: ['fire','smoke']`). 13,715 boxes:
  8,432 fire (tiny: median 0.18% of image area) + 5,283 smoke (median 4.7% area, p90 21%). Smoke is annotated as
  MANY overlapping sub-boxes per plume (2-8 per image, mode 2-3), not one box per plume. 0 malformed / out-of-range lines.
- Per-image: 1,303 images with >=1 smoke box (1,214 smoke+fire, 89 smoke-only); **3 fire-only** images; **30 empty**
  label files. readme.txt states "zero = no fire/smoke present in that frame", so empties are author-declared negatives.
- Author split (train 938 / valid 266 / test 132) is a RANDOM FRAME split: every one of the 8 streams appears in all
  three author splits -> the author split is leaky and is NOT used; recorded as `official_split` only.

## 4. Counts by label type / annotation origin
| category | images | boxes | annotation_origin |
|---|---|---|---|
| box, has smoke | 1,303 | 5,283 smoke (+8,385 fire) | author (Roboflow-user annotations re-deposited by Sheffield) |
| box, fire-only (NOT smoke negatives -> ignore-region candidates) | 3 | 47 fire | author |
| explicit negative (empty label file) | 30 | 0 | author |
| image-level only | 0 | – | – |
| mask | 0 | – | – |
| human-reviewed (by us) | **0** | | none — nothing here has been human-reviewed |
| model-assisted | 0 labels; diagnostic only (§7) | | |
Units: 1 session / 6 recordings (parent_unit) / 8 streams. Unit frame counts: rec_0003 134, rec_0004 296,
rec_0005 103, rec_0006 297, rec_0007 179, rec_0008 327.

## 5. Agent inspection flags (automated/agent visual inspection — NOT human review)
- 6 of the 30 empty-label frames visibly contain smoke (label omissions): 0005_W#0029, 0003_Z#0042, 0003_Z#0128,
  0004_Z#0027, 0006_W#0132, 0006_Z#0084 -> `agent_visual_flag = smoke_visible_in_empty_label`; kept as
  is_explicit_negative=True (author semantics) but EXCLUDED from the YOLO export and listed in
  data/v5b/yolo/sheffield_uav/needs_human_review_empty_label_smoke_visible.txt.
- The other 24 empties are one static scene (rec_0005_W frames 80-103, ~24 consecutive frames: parked vehicles on a
  dirt track, no smoke visible) -> genuine negatives but effectively ONE negative unit, not 24 independent ones.
- The 3 fire-only frames show flame fronts with clearly visible unlabeled smoke -> ignore_region_candidate=True,
  excluded from export, listed in ignore_region_candidates_fire_only.txt. No labels were fabricated for them.

## 6. Overlap / dedup findings
Compared exact sha256 and phash (imagehash.phash 64-bit, hamming <= 6, same convention as scripts/v4/build_corpus.py)
against 161,618 reference images (160,913 with phash): MultiNatSmoke 70,816 (incl. AuSmoke, FIgLib, WSDataset,
D-Fire, SmokeSeg, Boreal, FireSpot, aiformankind, firecam, SMOKE5K, FESB), pyro_sdis 33,636, MmodalFire 23,599,
D-Fire 21,527, Boreal 6,714, FIgLib subset 3,056, FIgLib lockbox 1,978, FIRESENSE frames 292
(table: data/v5b/manifests/_cache/reference_hashes.parquet, builder scripts/v5b/build_reference_hashes.py).
Result: **0 exact and 0 near-duplicate hits** -> dup_class = unique for all 1,336; no row inherits heinola/ruokolahti/
FIgLib/sdis-test/AuSmoke-test roles; quarantine = 0. FASDD is not held locally -> not compared (limitation).
Date check vs Boreal burns (evo 2022-05-24, heinola 06-28, karkkila 07-06, ruokolahti 08-15): 2022-07-21 matches none,
and the DJI naming scheme differs from Boreal's `DJI_00xx` files.

## 7. Model-assisted sanity DIAGNOSTIC (not review) — models/trained/expE_mix_y11m_1280_v2/seg05/best.pt, CPU, imgsz 1280
Scored 200 random smoke-labelled images (828 author smoke boxes) + all 30 empties + 3 fire-only (seed 0;
scripts/v5b/model_assisted_check.py; preds data/v5b/manifests/_cache/sheffield_uav_modelcheck_preds.parquet).
- Box-level recall of author smoke boxes @conf>=0.2, IoU>=0.4: **0.085** (70/828). Image-level (>=1 author box
  matched): 0.335. BUT: 98.0% of smoke images get >=1 detection @0.2 (median max conf 0.59) and **91.1%** of author
  boxes are >=50% contained inside a predicted box -> the IoU miss is a *box-convention mismatch* (author = many
  fragmented sub-boxes per plume; detector = ~1.5 large plume boxes per image), not scene-level disagreement.
  Consequence: if used for training, the fragmented-sub-box convention conflicts with our one-box-per-plume
  convention and with v4 IoU>=0.4 evaluation; consider union/merge of overlapping author boxes before use (not done).
- Explicit negatives with any detection >=0.3: **12/30** (>=0.5: 7/30). All 6 agent-flagged smoke-visible frames
  fire (0.36-0.75) — consistent with label omission. 6/24 parked-vehicle frames fire at 0.29-0.55 on the bright dirt
  track/vehicles with no smoke visible -> genuine detector false positives (useful hard-negative cluster, 1 scene).
- Fire-only frames: 3/3 fire >=0.3 (smoke is visibly present in them).
- Visual sheet for reviewers: reports/v5b/sheffield_uav_author_vs_model_sheet.jpg (cyan author, magenta model).

## 8. Proposed split (by WHOLE recording; W+Z streams never separated)
train = rec_0003, 0004, 0005, 0006 (830 frames, 62%; incl. the 24 negatives) · val = rec_0007 (179, 13%) ·
holdout = rec_0008 (327, 25%; author-labelled, last recording) · quarantine = 0. By units 4/1/1.
**Hard caveat: all six recordings are the SAME burn, same site, same 12 minutes.** val/holdout here measure
within-event generalisation only; they must not be reported as an independent site. If an independent aerial test is
needed, use this source whole (all-train or all-eval), never split.

## 9. Deliverables
- data/v5b/manifests/sheffield_uav_manifest.parquet (1,336 rows; required columns + session_unit, video_stream,
  recording_start_local, camera, frame_idx, official_split, n_fire_boxes, ignore_region_candidate, agent_visual_flag,
  dup_detail, inherited_role, within_source_neardup, boxes[json, original class ids])
- data/v5b/yolo/sheffield_uav/{images,labels}/{train,val,holdout} — class 0 = smoke only (fire boxes dropped),
  images symlinked to raw; 24 explicit negatives = empty label files; 1,327 images exported; data.yaml (nc=1);
  sidecars ignore_region_candidates_fire_only.txt (3), needs_human_review_empty_label_smoke_visible.txt (6).
- data/v5b/manifests/_cache/{sheffield_uav_hashes,reference_hashes,sheffield_uav_modelcheck_preds}.parquet,
  sheffield_uav_modelcheck_summary.json, sheffield_uav_counts.json
- scripts/v5b/{build_reference_hashes,build_sheffield_manifest,model_assisted_check}.py (re-runnable)

## 10. Limitations
1. Provenance of the underlying footage is **unestablished**: establishable = DJI H20-class dual-stream video, one
   session 2022-07-21, one temperate grassland site, no image-level overlap with any of our 161k reference frames;
   NOT establishable = who flew it, where, and whether it was first published elsewhere (Roboflow origin page blocked;
   ORDA record silent). Fraction with established origin: 0/1,336 beyond "not a copy of our material" -> treat as
   unknown-provenance, not as an independent new site.
2. One event only (n_units=1 at the event level); weak independence; dense near-duplicate frames (292 phash-identical).
3. Author boxes: fragmented multi-sub-box smoke convention; 6/30 negatives are label omissions; 3 fire-only frames
   have unlabeled smoke; label quality otherwise plausible on 53 inspected frames; no human review performed.
4. EXIF stripped; author split leaky (frame-random) and discarded.
5. Explicit negatives: 24 frames from a single ~24 s static scene -> not a negative-hours contribution (PROTOCOL §6).
