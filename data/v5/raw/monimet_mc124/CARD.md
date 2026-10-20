# monimet_mc124 — MONIMET phenocam MC124, Lammi mixed stand, landscape view, year 2020 (CARD, V5 Day-1, 2026-10-02T01:14Z)

## Provenance
- Zenodo record "Phenological time lapse images from landscape camera MC124 in Lammi Mixed stand", https://doi.org/10.5281/zenodo.7500295
  (creators Loehr, Böttcher, Aurela, Kolari; Finnish Meteorological Institute / Natural Resources Institute Finland (Luke) network).
  Record JSON saved as _zenodo_record.json; network datasheet record 10.5281/zenodo.777952 saved as _zenodo_record_datasheet_777952.json.
- Record description (verbatim excerpts): "Camera was mounted at landscape view level at location 61.05356; 25.03898 (N;E, WGS84).
  ... Images are taken half-hourly during fixed day-time period over the year. Gaps in time series and dark images possibly exist. ...
  time indicated in UTC+2. ... collected under EU Life+ (LIFE ENV/FI/000409) Monimet project, http://monimet.fmi.fi".
  Contact given on the datasheet record: mikko.peltoniemi@luke.fi. Citation requested: Peltoniemi et al., ESSD, doi 10.5194/essd-2017-62.
- md5 from the Zenodo record verified after download.

## Licence
- Zenodo licence id cc-by-4.0 on 7500295 and on the datasheet record 777952; access open. Attribution required.

## Contents
| file | bytes | sha256 (first 16) | md5 check | probe |
|---|---|---|---|---|
| Monimet_MC124_2020_2020.zip | 3,625,940,364 | c305606abc2bbb6a | md5 a686ef6c |  |
- Zip listing: 6375 entries, 3.72 GB uncompressed; by extension: {".jpg": 6375}; 6375 timestamped files; first 2020-01-10 11:01, last 2020-11-02 20:01.
- Sample names: ['lammi_ls_mixed_2020_01_10_110111.jpg', 'lammi_ls_mixed_2020_01_10_113104.jpg', 'lammi_ls_mixed_2020_01_10_120105.jpg', 'lammi_ls_mixed_2020_01_10_123105.jpg', 'lammi_ls_mixed_2020_01_10_130107.jpg', 'lammi_ls_mixed_2020_01_10_133105.jpg', 'lammi_ls_mixed_2020_01_10_140105.jpg', 'lammi_ls_mixed_2020_01_10_143104.jpg']

## Semantics / intended V5 use
- Purpose tag at ingest: **FIXED-CAMERA NO-SMOKE candidate (UNVERIFIED), 30-min cadence** → proposed CALIBRATION/VAL pool for the
  fixed-camera scenario (Finnish boreal forest landscape), never training until human review + inventory record. No model has seen it.
- Episodes/hour must be reported as "30-min sampled" with valid-time rules (PROTOCOL §3); a 30-min cadence cannot resolve
  episode durations/cooldowns comparable to the 29-s pyro-sdis cadence — it measures frame-level FP and seasonal confounders.

## Limitations
- Research-forest phenology camera: smoke is very unlikely but NOT excluded (chimneys, slash burning, mist). Fog, snow, low sun,
  wet lens are expected confounders — exactly the kind we want, but each flagged frame needs a human label.
- One camera-year = one independent unit; other MONIMET cameras/years (15–48 GB each) are listed in ACQUISITION_LEADS_V5.md §2.3.
