# FIRESENSE video database — dataset card (v4 acquisition, 2026-09-14)

## Provenance
- **Title**: "FIRESENSE database of videos for flame and smoke detection"
- **Creators**: Nikos Grammalidis, Kosmas Dimitropoulos (ITI-CERTH, Thessaloniki, Greece); Enis Cetin (Bilkent University, Ankara, Turkey)
- **Origin**: EU FP7-ENV project 244088 "FIRESENSE" (protection of cultural heritage areas from fire). Published on Zenodo 2017-07-31. Videos themselves date from ~2005–2011 (burned-in timestamps observed: 2005-04-29, 2010-05-31, 2010-07-28/29; zip file mtimes Jun 2011).
- **DOI**: 10.5281/zenodo.836749 (concept DOI 10.5281/zenodo.836748)
- **Download URLs** (Zenodo API, direct, no registration):
  - https://zenodo.org/api/records/836749/files/fire_videos.1406.zip/content (622,676,845 B, md5 7cfe8286cc4b45193149186bf67ea00d — VERIFIED match after download)
  - https://zenodo.org/api/records/836749/files/smoke_videos.1407.zip/content (197,589,846 B, md5 5ee6bc867c4a9332b4e75f308c26fc23 — VERIFIED match)
- Record JSON saved verbatim at `data/v4/raw/firesense/zenodo_record_836749.json`.

## License
- **CC-BY-4.0** (Creative Commons Attribution 4.0 International), per the Zenodo record's `metadata.license.id = "cc-by-4.0"`.
- Verbatim sources: `data/v4/raw/firesense/zenodo_record_836749.json` and `data/v4/raw/firesense/zenodo_license_cc-by-4.0.json` (Zenodo license vocabulary entry pointing to https://creativecommons.org/licenses/by/4.0/legalcode).
- Commercial use permitted; attribution to the creators/DOI required. No restriction blocking our internal research prototype. No separate license/README file exists inside the zips (checked: zips contain only .avi files).

## Contents on disk
- `data/v4/raw/firesense/fire_videos/{pos,neg}/*.avi` — flame task: 11 pos, 16 neg
- `data/v4/raw/firesense/smoke_videos/{pos,neg}/*.avi` — smoke task: 13 pos, 9 neg
- Total **49 videos**, ~28.9 min decoded video, resolutions 300x240 → 1600x1200 (mostly 320x240–640x480), 4.7–30 fps, codecs FMP4/MJPG/MP42/IV50.
- Extracted frames (~1 fps, JPEG q95): `data/v4/raw/firesense/frames/{smoke_pos,smoke_neg,flame_pos,flame_neg}/<video_id>/<video_id>_tSSSS.jpg` (SSSS = seconds offset into video). **1,752 frames total**: flame_pos 339, flame_neg 591, smoke_pos 530, smoke_neg 292.
- Extraction log: `data/v4/raw/firesense/frame_extraction_summary.json`. Audit contact sheets: `data/v4/raw/firesense/audit_sheets/`.
- **Decode caveat**: `testneg03.809.avi` (Indeo IV50 codec) only decodes 182 of 748 container frames with OpenCV/ffmpeg → only the first ~6 s / 7 frames extracted. All other videos decoded fully (±1 frame).

## Label semantics (audited by eye — 146 sampled frames viewed on contact sheets, first/middle/last frame of every video, plus full-res checks)
- **Video-level binary labels ONLY**, encoded purely by directory placement (pos/neg per task). There are **no boxes, no masks, no frame-level annotations, no label files of any kind** in the archives. `label_type=videolabel` everywhere; `n_smoke_boxes=0`.
- **Smoke IS explicitly the labeled target** for the smoke task (13 pos / 9 neg), separate from the flame task — this is not a fire-only dataset.
- What I actually saw:
  - **smoke_pos (13)**: mostly *staged near-field white smoke* from smoke pots/grenades on lawns, courtyards, gravel (testpos01, 04, 06–13, incl. one pink smoke jet testpos08); one NIST structure-burn with dark smoke from a window + visible flame (testpos02, "NIST" watermark); one far-field surveillance view of a fenced hillside with faint smoke + small visible flame (testpos03); one indoor warehouse clip (testpos05) that has **cyan smoke-detection contours burned into the pixels** (algorithm-output screen capture) + "C01: 04/29/05" timestamp overlay.
  - **smoke_neg (9)**: verified confounders — swaying blossom tree, people on icy lot, indoor office, fenced bushes, reeds, highway traffic x2, white *clouds* behind bare trees, waving flags. Directly usable as verified smoke-free negatives (video-level negative ⇒ every frame negative).
  - **flame_pos (11)**: burning bus with heavy black smoke, large forest crown-fire ("UltimateChase.com" watermark x2), kitchen sink fire, burning leaves, fireplace x3, small trash fire, campfire, small fire on the same fenced hillside as testpos03.
  - **flame_neg (16)**: castle pan, driving clips, offices/people, red car, highway overpass x2, train station x2, night traffic with headlight glare x2, kids/toys, glitter bokeh, "SHRED" red text card, aquarium — classic flame-color/motion confounders.
- **Annotation completeness caveats** (missing labels are NOT negatives):
  - Frames from *pos* videos carry NO frame-level truth: smoke/fire is often not visible at t0 (e.g. testpos01_t0000, testpos04_t0078 show no smoke). Manifest sets `has_smoke_label=false`, `has_fire_label=false` on ALL frame rows; video rows carry the video-level positive flag only.
  - Cross-task leakage of the *other* class: most flame_pos videos also show unlabeled SMOKE (bus fire, forest fire, trash fire); smoke_pos testpos02/testpos03 also show unlabeled FLAME. A flame-only video label implies nothing about smoke and vice versa.
  - **Burned-in overlays are a shortcut risk**: green "Live (FPS: x.xx)" screen-capture text on posVideo1, posVideo6, posVideo7, testpos06, testpos07; cyan detector contours on testpos05; watermarks "UltimateChase.com" (posVideo10/11), "NIST" (testpos02); timestamp bars on negsVideo12/13/14/15/16, testpos05.

## Counts
- **Unique images (extracted frames): 1,752** (sha256-unique, 0 duplicates). Adjacent 1-fps frames of one video are highly correlated — NOT independent.
- **Videos: 49** (11 flame_pos / 16 flame_neg / 13 smoke_pos / 9 smoke_neg).
- **Independent units (parent_unit): 43.** I merged 6 visually-verified same-site/same-session video pairs (full-res comparison + burned-in timestamps 90 s – 2 min apart):
  - firesense_site_fencedhill = {testpos03.819 (smoke_pos), posVideo7.876 (flame_pos)} — identical camera view
  - firesense_site_lakeside = {posVideo1.868, posVideo6.875} — same lakeside compound
  - firesense_site_courtyard = {testpos06.822, testpos07.823} — identical camera view
  - firesense_site_overpass_20100728 = {negsVideo12.1074, negsVideo15.1078} — timestamps 90 s apart
  - firesense_site_nightstreet = {negsVideo2.859, negsVideo3.860} — same night street session (probable)
  - firesense_site_trainstation_20100531 = {negsVideo13.1075, negsVideo14.1076} — timestamps 2 min apart
  - (testneg06.812 / testneg09.815 are both Dutch motorway clips but views differ; kept separate — treat with mild caution.)
- parent_unit spans categories in one case (fencedhill: a smoke_pos AND a flame_pos video) — grouped splits must respect this.

## Usability for our task (early wildfire smoke detection)
- **Directly usable**: 292 smoke_neg frames from 9 videos as *verified hard negatives* (clouds, fog-like reeds, flags, traffic, foliage motion). Optionally flame_neg frames (591) as generic negatives for the smoke task ONLY if one accepts video-level flame-negativity as smoke-negativity — I verified by eye that no flame_neg sampled frame shows smoke, but exhaust/dust in traffic clips was not exhaustively checked frame-by-frame.
- **NOT directly usable as smoke positives for detection training**: no boxes; smoke not visible in every frame of pos videos; smoke is mostly staged near-field white smoke, not distant wildfire plumes (domain mismatch vs pyro-sdis towers); several clips have burned-in overlays/watermarks.
- Videos left whole alongside frames; frame rows and video rows are both in the manifest (distinguish by rel_path extension .jpg vs .avi).

## Confounders present (valuable)
Clouds behind trees, moving foliage/reeds, waving flags, highway traffic (day+night), headlight glare, red objects/text, glitter bokeh, people indoors, snow/ice, overexposure.

## Contamination check vs existing sources
- Filename overlap vs data/manifests/{dfire,figlib_subset,pyro_sdis} manifests: **0 / 0 / 0**.
- Existing manifests carry no sha256/phash columns, so exact/near-dup content check deferred to the central pass (and exact sha would not match across re-encodes anyway).
- **Suspicion assessment**: pyro-sdis (fixed Pyronear towers) and FIgLib (HPWREN cameras) — very low. **D-Fire — moderate**: D-Fire aggregates internet imagery, and FIRESENSE is a classic public benchmark whose clips (fireplace, UltimateChase forest fire, Bilkent/VisiFire-style test clips co-authored by Enis Cetin) circulate widely; central near-dup pass should compare D-Fire vs these 1,752 frames.

## Blockers
None. Direct download worked; md5 checksums verified; license permissive. Only defect: testneg03.809 partial decode (first 6 s only) — 24% of that one video.

## Manifest
`data/v4/manifests/firesense_manifest.parquet` — 1,801 rows (49 video rows + 1,752 frame rows), rel_path relative to `data/v4/raw/firesense/`. Columns per v4 spec. Conventions: `has_smoke_label=true` only on smoke_pos VIDEO rows; `has_fire_label=true` only on flame_pos VIDEO rows; all FRAME rows false/false (no frame-level truth); smoke-neg status recoverable from rel_path prefix `frames/smoke_neg/` or `smoke_videos/neg/`; `origin=original-human` (labels = human curation of whole videos by the FIRESENSE consortium); `camera_or_flight_id` only derivable for testpos05.821 ("C01" burned-in); `timestamp` only where burned-in dates were legible (5 videos + their frames).
