# commons_usgov_burn_videos — US-government prescribed-burn videos via Wikimedia Commons (CARD, V5 Day-1, 2026-10-02T01:14Z)

## Provenance (per file; Commons extmetadata saved in _commons_imageinfo.json)
| local file | origin | Commons licence tag | credit/author (Commons) |
|---|---|---|---|
| NASA_FireSense_SVS14559_DRONES_BROLL.webm | NASA SVS 14559 "NASA FireSense (Fishlake National Forest)" — drones flown during the 2023-10-09 FASMEE prescribed burn, Fishlake NF, Utah (SVS page released 2024-03-29) | Public domain | NASA's Scientific Visualization Studio |
| NASA_FireSense_SVS14559_PrescribedBurn_Area2_Day2_BROLL.webm | NASA SVS 14559, "Prescribed Burn Area 2 Day2 BROLL" (ground-level) | Public domain | NASA's Scientific Visualization Studio |
| USFS_R1_Prescribed_Burn_North_Derby_Gulch_CGNF_53229395208.webm | USDA Forest Service Northern Region Flickr → Commons; Custer Gallatin NF | Public domain | Forest Service - Northern Region |
| BLM_ORWA_Timelapse_drip_torch_prescribed_burn_37896304436.webm | BLM Oregon & Washington Flickr → Commons | Public domain | BLM Oregon & Washington |
| BLM_ORWA_Loveless_Prescribed_Burn_CCBY2.webm | BLM Oregon & Washington Flickr → Commons | CC BY 2.0 | Bureau of Land Management Oregon and Washington |

- NASA Media Usage Guidelines (fetched 2026-10-02): "NASA content – images, audio, video ... generally are not subject to
  copyright in the United States. ... NASA should be acknowledged as the source of the material."
- US federal agency works (USFS, BLM) are public domain in the US (17 U.S.C. §105); the Commons tags agree. The Loveless clip
  is tagged CC BY 2.0 by BLM on Flickr — attribution required.
- Downloaded from upload.wikimedia.org originals (URLs in _download_log.jsonl). The SVS site also offers the 6.2 GB mp4 master of the drone B-roll.

## Contents (probed with OpenCV)
| file | bytes | sha256 (first 16) | md5 check | probe |
|---|---|---|---|---|
| NASA_FireSense_SVS14559_DRONES_BROLL.webm | 210,931,334 | c95e1ca82dffe4c1 | none published | 3840x2160 @ 23.976 fps, 26508 fr, 1105.6 s, decodable=True |
| NASA_FireSense_SVS14559_PrescribedBurn_Area2_Day2_BROLL.webm | 686,526,446 | 1e00cd3c07229275 | none published | 3840x2160 @ 23.976 fps, 87817 fr, 3662.7 s, decodable=True |
| USFS_R1_Prescribed_Burn_North_Derby_Gulch_CGNF_53229395208.webm | 86,489,452 | ed37fec8cd99670b | none published | 1920x1080 @ 30.0 fps, 737 fr, 24.6 s, decodable=False |
| BLM_ORWA_Timelapse_drip_torch_prescribed_burn_37896304436.webm | 29,371,243 | 6f57560d30929eb0 | none published | 1920x1080 @ 29.97 fps, 610 fr, 20.4 s, decodable=True |
| BLM_ORWA_Loveless_Prescribed_Burn_CCBY2.webm | 51,461,830 | 5010e31a38ce4497 | none published | 608x1080 @ 30.0 fps, 1416 fr, 47.2 s, decodable=True |
Total 1,064,780,305 bytes (1.06 GB).

## Semantics / intended V5 use
- All five are EDITED B-roll / short clips containing active prescribed fire and smoke; some show ignition (drip torch,
  heli-torch) → candidate material for the human onset-review tool (does a cut show a clean no-smoke → first-smoke transition?).
- Purpose tag at ingest: **human-review candidate material only**. No metric use (cuts break the valid-time rules of PROTOCOL §3);
  not negatives; not training unless a later decision records otherwise.

## Limitations
- Edited compilations: scene cuts, variable cameras, no timestamps. Fishlake burn footage is aerial WITH smoke (not a negative).
- Provenance chain is agency → Flickr/SVS → Commons; the Commons copies are transcodes (webm) of the originals.

## Probe note
- USFS_R1_Prescribed_Burn_North_Derby_Gulch_CGNF_53229395208.webm is AV1-encoded; the OpenCV build in .venv cannot decode it
  ("Failed to get pixel format"). The other four decode. Total decodable material ≈ 4,836 s (80.6 min), of which 4,768 s is the two NASA B-rolls.
