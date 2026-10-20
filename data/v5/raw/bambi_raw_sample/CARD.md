# bambi_raw_sample — BAMBI Raw Dataset, flight_98 (CARD, V5 Day-1, 2026-10-02T01:14Z)

## Provenance
- Project: BAMBI — Biodiversity Airborne Monitoring Based on Intelligent UAV sampling, University of Applied Sciences Upper Austria
  (https://www.bambi.eco/ ; code/metadata https://github.com/bambi-eco/Dataset).
- Zenodo: "The BAMBI Raw Dataset: Unprocessed Nadir UAV-Recordings of Forest Wildlife (Part 47)", https://doi.org/10.5281/zenodo.19177615
  (published 2026-03-23; creators Praschl, Schedl, Maschek, Probst, Stöckl, Böss). Record JSON saved as _zenodo_record_part47.json.
- Record description (verbatim): "It contains the raw, unprocessed nadir UAV recordings of forest wildlife. Each flight is stored as an
  individual ZIP archive named by its flight id (e.g. flight_0.zip). Inside each ZIP are the original video files captured during the
  flight, along with the associated undistortion parameters, SRT subtitle files, and Airdata flight logs."
- GitHub README (saved _github_README.md): 389 paired RGB+thermal sequences, ~45 h total flight time, 12 species, recording period
  January 2023 – November 2024, Austria (Tyrol, Upper Austria, Lower Austria, Salzburg, Styria, Carinthia); nadir view.
- Flight 98 metadata (saved 98_metadata.json, from the GitHub flight_metadata/): start 2023-04-26T17:26:45 end 2023-04-26T17:42:24 (local),
  drone "Mavic 3 Thermal", weather code 2, split "train", species present: Sus scrofa (Wild boar);
  sharepoint path indicates site "2023_04_26_NOe_Purkersdorf" (Lower Austria).
- md5 from the Zenodo record verified after download (see _download_log.jsonl).

## Licence
- docs/dataset-versions.md (saved _github_dataset-versions.md), verbatim: "Every version of this dataset is **CC-BY-4.0** except
  `environment-nc`, which is **CC-BY-NC-4.0**." The `raw` version is CC BY 4.0. Zenodo record licence id cc-by-4.0, access open.

## Contents
| file | bytes | sha256 (first 16) | md5 check | probe |
|---|---|---|---|---|
| flight_98.zip | 1,107,504,419 | 7a2dacca291eb740 | md5 6b7a1fea |  |
- Zip listing: 7 entries, 1.12 GB uncompressed; by extension: {".csv": 1, ".mp4": 2, ".srt": 2, ".json": 2}
- First entries: ['air_data.csv', 'DJI_20230426173033_0001_T_point1.MP4', 'DJI_20230426173033_0001_T_point1.SRT', 'DJI_20230426173033_0001_V_point1.MP4', 'DJI_20230426173033_0001_V_point1.SRT', 'T_calib.json', 'W_calib.json']

## Semantics / intended V5 use
- Purpose tag at ingest: **AERIAL NO-SMOKE candidate (UNVERIFIED)** → proposed CALIBRATION/VAL pool for the aerial scenario, never
  training until the human review and the inventory record exist. No model has seen it.
- What it is: wildlife-survey flight over Austrian forest; no fire by design. Thermal + RGB video from a DJI Mavic 3 Thermal,
  nadir gimbal. Smoke-free status must be established by a HUMAN watching the RGB track (detector silence is not evidence).

## Limitations
- Nadir-only (Boreal flights span 0° to −90° pitch). Temperate Austrian forest, not boreal Finland. RGB from a Mavic 3T wide camera.
- Evening flight (17:26–17:42 local, late April) — lighting differs from Boreal's daytime burns.
- One flight = one independent unit; BAMBI offers ~389 flights across sites/dates if more are wanted (1–7 GB each).

## Probe results (cv2 + SRT parsing on this host, 2026-10-02)
- DJI_20230426173033_0001_V_point1.MP4 (RGB wide): 3840x2160 @ 29.97 fps, 2,873 frames, **95.9 s**, decodable.
- DJI_20230426173033_0001_T_point1.MP4 (thermal): 640x512 @ 29.95 fps, 2,872 frames, 95.9 s, decodable.
- SRT (per-frame): 2023-04-26 17:30:33.9 → 17:32:10.7 local; rel_alt 63.3–67.0 m; gimbal pitch −90.0…−89.9°; lat 48.1945 lon 16.1709 (Purkersdorf, Lower Austria); focal_len 40.
- NOTE: the flight-metadata window (17:26:45–17:42:24) is the whole survey; this raw zip holds ONE 96-second segment ("point1").
  So this sample contributes ~1.6 min of candidate aerial no-smoke video, not 16 min. Reaching the protocol's ≥2 h / ≥3 units
  for aerial negatives would require pulling on the order of 75+ such segments from different flights/sites (1–7 GB per flight zip).
