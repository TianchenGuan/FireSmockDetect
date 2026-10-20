# mmodalfire — MmodalFire indoor fire videos + physical sensing (iteration 4)

## Provenance
- Paper: "MmodalFire: A Multimodal Dataset Comprising Video and Physical Sensing Data
  for Detecting Indoor Fires", Zhang, Jia, Guo, Chen, Zhang, Wang — Scientific Data,
  DOI 10.1038/s41597-026-06810-6 (open access, read in full for this audit).
- Dataset: figshare (Springer Nature portal), article 28804448,
  DOI 10.6084/m9.figshare.28804448.v1, published 2026-03-28.
- Download URL (from figshare API https://api.figshare.com/v2/articles/28804448):
  https://ndownloader.figshare.com/files/53696243 -> MmodalFire.rar
  (4,929,210,010 bytes). md5 88287408a7a36a25999cf6276fa902a9 verified after download.
  No registration/click-through required.
- Local: data/v4/raw/mmodalfire/ (MmodalFire.rar + extracted/原始数据/ + frames_1fps/
  + figshare_article_28804448.json + LICENSE_TERMS.txt).

## License
- CC BY 4.0 (figshare API license field: `{"name": "CC BY", "url":
  "https://creativecommons.org/licenses/by/4.0/"}`; companion paper also CC BY 4.0).
  Verbatim record: data/v4/raw/mmodalfire/LICENSE_TERMS.txt. Commercial use OK with
  attribution — fine for our internal research prototype.

## What the deposit actually contains (differs from the paper's Data Records section)
The paper describes two folders "MmodalFire (Xi'an)" / "MmodalFire (Hefei)" with
InertialData/VideoData and 65 videos across 2 labs. The actual figshare v1 RAR contains
ONE lab's raw data in a single folder 原始数据 ("raw data"):
- `Video/`: 28 raw DVR surveillance MP4s (1920x1080, 25 fps, H.264, total 1,982 min /
  33 h), filenames `Downtest_ch01_*.mp4`, burned-in OSD date-time top-left (all
  2022-08-01..2022-08-10), watermark 灭火试验室 ("fire-extinguishing laboratory")
  bottom-right. One camera, one room, one continuous fixed view.
- `Physical sensing/`: 70 CSVs (GBK encoding) — per session a timestamp file
  (UTC + local, 64 Hz) and a 6-channel value file (smoke density, temperature,
  IR 5.0/4.4/3.8 um, UV count). Names offset vs XLSX (e.g. "Cotton rope3" = XLSX
  "Cotton 1"; verified by timestamps). Plus `test1/2` (pre-session equipment test)
  and "N-Heptane (after burnning)" (post-burn residual logs).
- `SETTINGS OF THE COMBUSTION EXPERIMENT FOR DATA COLLECTION.xlsx`: THE label file —
  38 sessions with material, luminance (lx), wind speed, fire location, quantity, and
  wall-clock interval (month.day hh:mm-hh:mm, year 2022 per OSD).
- `Readme.txt`: alignment instructions (videos are raw surveillance; align via OSD
  clock and the XLSX/CSV times).
Which of the two labs this deposit is could not be determined; the second lab's data
is NOT in this deposit.

## Label semantics (audited by opening files — 28 OSD first-frames + 114 session
frames sampled early/mid/late for all 38 sessions + 4 full-res frames + 7 CSVs + XLSX)
- Classes: session-level material = cotton (9 sessions), n-heptane (8), polyurethane
  (5), wood (4), dry-ice (4), water-mist "Mist" (4), blank control (2), occlusion
  n-heptane/polyurethane (2). Paper's usable label is BINARY per sequence:
  fire (cotton/n-heptane/polyurethane/wood/occlusion) vs non-fire (dry ice, mist, blank).
- NO boxes, NO masks, NO per-frame labels anywhere in the deposit. label_type =
  videolabel (session-level), derived by mapping burned-in OSD wall-clock to XLSX
  session intervals. Mapping validated: every sampled frame's OSD time matched the
  predicted time to ~1 s.
- SMOKE IS NOT EXPLICITLY ANNOTATED. "Fire" labels imply combustion, not per-frame
  visible smoke -> has_smoke_label=False everywhere; has_fire_label=True only for
  fire-session items. Missing labels are NOT negatives; fire-session early frames can
  show no visible smoke yet, late frames can be dense room-filling smoke.
- What I actually saw: cotton = smoldering rope, small ember/flame, faint plume +
  light haze; wood = furnace-heated smolder, dense white room-filling smoke late
  (near-whiteout, e.g. Wood1 10:25); polyurethane = flame + very dense dark smoke,
  colored lighting (pink/blue/purple) in several sessions; n-heptane = clear pool-fire
  flame column, moderate smoke; dry ice = NO visible fog from this camera (sensor-only
  interference; smoke-density CSV stays flat ~1200 while cotton ramps 1377->3242);
  mist = humidifier with a distinct white vertical plume (excellent smoke look-alike
  hard negative); blank = empty room; occlusion sessions = LENS COVERED, whole video
  is gray blur -> no usable visual data (kept as whole videos only, no frames).
- Experimenters walk through frames during sessions (ignition, adjustments, one sits
  by the fan; one wood session has a person standing mid-frame) — pedestrian presence
  inside "fire" sessions is common.
- Sensor CSVs are LABEL-AUDIT AID ONLY (used above to confirm smoke onset/session
  times); they must NOT become model inputs.

## What was extracted
- ~1 fps frames ONLY inside XLSX session intervals (label established at session
  level), for 36 of 38 sessions (2 occlusion sessions skipped — lens covered):
  23,571 JPEG frames (q90, 1920x1080, ~5.6 GB) under
  data/v4/raw/mmodalfire/frames_1fps/<Session>/<Session>_YYYYMMDD_HHMMSS.jpg.
- Sessions crossing DVR file boundaries were stitched from both files (verified,
  e.g. NHeptane8, Wood1, Mist3).
- The ~26 h of footage OUTSIDE session intervals (idle room, setup, smoke clearing
  between sessions, 6 DVR files contain no session at all) was NOT extracted —
  unlabeled, not negatives (residual smoke can linger after sessions).

## Manifest
data/v4/manifests/mmodalfire_manifest.parquet — 23,599 rows = 23,571 frames
(item_kind=frame) + 28 raw videos (item_kind=video). Columns: source, rel_path,
sha256 (all unique; spot-verified), parent_unit (mmodalfire/<Session> for frames,
mmodalfire/<dvr-stem> for videos), camera_or_flight_id (ch01_room1 — single camera),
timestamp (wall clock from OSD start + pts), label_type (videolabel; none for the 7
session-free DVR files), has_smoke_label (all False — smoke never annotated),
has_fire_label, n_smoke_boxes (0), license, origin (original-human — labels are the
authors' experiment log), plus extras: material, session_label (fire/interference/
blank for frames; session list for videos), video_stem, item_kind.

## Counts (do not conflate)
- Unique images: 23,571 extracted frames (all sha256-unique). Raw videos: 28.
- Independent units (recording sessions): 38 total; 36 with frames
  (26 fire — cotton 9 / n-heptane 8 / polyurethane 5 / wood 4, 8 interference,
  2 blank), 2 occlusion sessions video-only.
- Frames by material: wood 8,091 / cotton 5,701 / n-heptane 3,661 / polyurethane
  2,222 / dry-ice 1,921 / water-mist 1,741 / blank 240. Fire-session frames 19,671;
  interference 3,660; blank 240.
- ADJACENT 1-fps FRAMES ARE NOT INDEPENDENT, and ALL 38 sessions share ONE room, ONE
  camera, ONE week. For grouped splits treat parent_unit=session as the group at
  minimum; for honest generalization claims treat the WHOLE SOURCE as a single site
  (recommend: train-only supplement for corpus G, never an eval domain).

## Confounders present (useful)
- Water-mist humidifier plume (visible white plume, labeled non-fire).
- Dry-ice sessions (labeled non-fire; visually near-identical to blank room).
- People moving in frame during fire and non-fire sessions.
- Colored/dim lighting (pink/blue/purple lamps, near-dark), wind from a floor fan,
  dense-smoke whiteout frames where the "smoke" is the whole image.
- Burned-in Chinese OSD text (top-left) and watermark (bottom-right) on every frame —
  a model could shortcut on OSD region; consider cropping at training time.

## Contamination vs pyro-sdis / D-Fire / FIgLib
- None plausible: indoor Chinese fire-lab surveillance recorded 2022-08, published
  2026-03; pyro-sdis (French wildfire towers), FIgLib (HPWREN), D-Fire (mostly
  outdoor, released before these videos were public) cannot contain these frames.
  Frames were decoded fresh here, so sha256 overlap with existing manifests is
  impossible by construction; no filename-pattern overlap either. Full near-dup pass
  deferred to the central pass as planned.

## Blockers / caveats
- No blocker for use. Caveats: (1) deposit is HALF the paper's described dataset
  (one lab, 38 sessions, 28 DVR files — not 65 curated videos, no Xi'an/Hefei split);
  (2) fire/non-fire is the ONLY annotation — per-frame smoke visibility is NOT
  labeled; (3) 2 occlusion sessions have no usable video; (4) paper's "12 fps" refers
  to their processed clips — raw files are 25 fps; (5) XLSX times are minute-granular,
  so session-boundary frames carry +/- ~30 s label slop.
