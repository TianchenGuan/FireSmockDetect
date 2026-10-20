# V5 Acquisition Leads — proactive external data search (Day 1)

Prepared 2026-10-02 (UTC) by the acquisition agent. Research + email drafts only.
Everything below was checked live from this host (HTTP fetches of the landing
page / repository API / licence page; raw fetches saved under /tmp/v5acq/ during
the run). Nothing was taken from mirrors, torrents or Kaggle re-uploads. Items were
DOWNLOADED only when the licence is explicit, permits redistribution, and the file
is < 20 GB (see §5); everything else is listed.

Scope asked for: (a) NO-SMOKE forest/wildland AERIAL footage with verifiable
provenance; (b) complete smoke-EVENT recordings spanning ignition → visible smoke;
(c) continuous clean FIXED-CAMERA forest footage.

Protocol reminders that shaped the ranking (reports/v5/PROTOCOL_V5.md §2, §6):
negatives must be human-verified (detector silence is not evidence), ≥2 h valid
per scenario from ≥3 independent units; onsets need ≥10 independent events with
reviewed [last_clear_no_smoke, first_clear_smoke] intervals; licence recorded at
ingest; new footage assigned by whole site/flight/date before any model sees it.

## 0. Headline

| | finding |
|---|---|
| Best aerial no-smoke lead | **BAMBI raw UAV forest recordings** (FH Upper Austria, Zenodo, CC BY 4.0): 389 flights, ~45 h, nadir RGB+thermal, Austria 2023–2024, no fire by design. Per-flight zips 1–7 GB. One flight downloaded as a sample. Limitation: nadir-only, temperate (not boreal) forest, needs human no-smoke check. |
| Best onset lead (free) | **RxCADRE 2012 plume photos + videos** (USFS Research Data Archive, US-Government public domain): 5 cameras per burn, 1 frame / 5 s, 3 operational burns (2 grass, 1 forest) + separate **ignition point/line time data**. But the zips are 26–34 GB each (JPG) and the MTS video set is 155–385 GB (ship-a-drive). Fixed ground cameras, not UAV. |
| Best fixed-camera clean footage | **MONIMET phenocams** (FMI/Luke, Zenodo, CC BY 4.0): Finnish boreal-forest cameras, half-hourly daytime images 2014–2022, 3.6–9.4 GB per camera-year; one camera-year downloaded. Cadence 30 min → episodes/h only at that cadence. **Fintraffic weathercams** (CC BY 4.0, 811 stations / 2,277 presets, ~10-min cadence, 24 h rolling history) are the only continuous fine-cadence source found; roadside, not wildland. |
| Behind a free login (user needed) | **FLAME 1 / FLAME 2 / FLAME 3** on IEEE DataPort — all CC BY 4.0 — need an IEEE account to download. FLAME 3 CV subset has 116 explicit **No-Fire quartets of the surrounding forest** (UAV) + pre-burn nadir flights; FLAME 1 has a pile-burn video "from the start of burning". |
| Licence surprise (user needed) | **HPWREN** (FIgLib source) and **ALERTCalifornia** state **CC BY-NC-ND 4.0** on their licence pages, with for-profit users asked to contact UCSD OIC. Our v4 card recorded FIgLib as "credit reference" only. Needs a legal/licence read before any further FIgLib use beyond historical comparison. FIgLib CDN lists 486 sequences — all already in our local index (0 new onsets). |
| Dead / not usable | Nemo (dataset URL never published); DFS (dead links, from v4); AuSmoke/MultiNatSmoke still **no licence file anywhere** (GitHub tree + HF card re-checked today) → email drafted. |

## 1. Ranked leads

Score = usefulness (U, 1–5, for V5's two questions) × feasibility (F, 1–5).
"Needs the user" lists exactly what only the user can do (account, payment,
authorization, legal review, human visual review). "Me/agents" = can proceed now.

| # | source | cat. | U×F | needs the user |
|---|---|---|---|---|
| 1 | BAMBI raw UAV forest recordings | a | 4×5=20 | human no-smoke review of sampled flights (I cannot certify "clean") |
| 2 | Boreal raw flights from FGI (request) | a+b | 5×2=10 | approve + send email (reports/v5/EMAIL_DRAFTS_V5.md) |
| 3 | MONIMET phenocam camera-years | c | 3×5=15 | human review (fog/snow/low-sun confounders) |
| 4 | RxCADRE 2012 plume photos/videos + ignition data | b | 4×3=12 | none for the 26–34 GB zips (exceeds my 20 GB cap → needs a go-ahead); drive shipping for the full MTS set |
| 5 | FLAME 3 CV subset + NADIR plots | a | 4×3=12 | IEEE account (free) |
| 6 | FLAME 1 pile-burn videos + Fire/No-Fire frames | a+b | 3×3=9 | IEEE account (free) |
| 7 | FLAME 2 RGB/IR pairs + raw pairs | a | 2×3=6 | IEEE account (free); note "No Smoke" ≠ smoke-free (see row) |
| 8 | Fintraffic weathercams (live collector) | c | 2×5=10 | none; but it is a new ongoing collector, not an archive — decide if roadside views are wanted |
| 9 | Zenodo "Forest Smoke Video Dataset" (smoke bomb) | b* | 2×5=10 | none (downloaded) — artificial smoke, tool-testing only |
| 10 | NASA FireSense FASMEE 2023 drone/ground B-roll | b* | 2×5=10 | none (downloaded) — edited B-roll, review material only |
| 11 | AuSmoke / MultiNatSmoke licence confirmation | b | 4×2=8 | approve + send email |
| 12 | HPWREN camera archive (non-FIgLib days) | c | 4×1=4 | licence read (CC BY-NC-ND) + OIC contact; archive host unreachable from here |
| 13 | ALERTCalifornia footage request | b+c | 4×1=4 | licence read (CC BY-NC-ND) + request via their contact page |
| 14 | FASDD_UAV "NeitherFireNorSmoke" images | a | 2×2=4 | decision on CC BY-SA + mixed web provenance |
| 15 | NPS air-quality webcams | c | 2×2=4 | manual browser check (JS site) — archive existence unverified |
| 16 | Commons CC-BY drone forest clips (YouTube-sourced) | a | 1×4=4 | none — but provenance is uploader-asserted; not recommended |
| 17 | Zenodo trACT drone wildlife movies | a | 2×4=8 | none — not downloaded (lower priority than BAMBI; same group type of content) |
| 18 | BurnedAreaUAV (prescribed fire, DJI P4 Pro, Portugal) | b* | 2×3=6 | none — .rar; no unrar on host |

## 2. Detailed records

### 2.1 BAMBI — Raw Nadir UAV Recordings of Forest Wildlife (rank 1)
- URL: GitHub https://github.com/bambi-eco/Dataset ; Zenodo parts e.g. Part 47 https://doi.org/10.5281/zenodo.19177615 (raw), Matched Part 1 https://doi.org/10.5281/zenodo.21291168 ; project https://www.bambi.eco/
- Owner: University of Applied Sciences Upper Austria (Praschl, Schedl, Maschek, Probst, Stöckl, Böss). Recording period Jan 2023 – Nov 2024; Austria (Tyrol, Upper/Lower Austria, Salzburg, Styria, Carinthia). Drone in sampled metadata: "Mavic 3 Thermal".
- Licence (verbatim, docs/dataset-versions.md): "Every version of this dataset is **CC-BY-4.0** except `environment-nc`, which is **CC-BY-NC-4.0**." Zenodo record licence id: cc-by-4.0, access_right open.
- Contents: 389 paired RGB + thermal nadir video sequences, "Total flight time ~45 hours"; `raw` version = "The original, unprocessed videos as they came off the drone, with undistortion parameters, SRT subtitles and Airdata logs" (one folder per flight). `base` = processed 2048×1024 side-by-side (thermal left / RGB right). Flight metadata JSON per flight (start/end time, drone, weather code, species).
- Viewpoint: nadir only (camera −90°). Boreal UAV used pitch 0° to −90°, so this covers only the nadir end. Habitats: forest and forest-adjacent; wildlife surveys → flights at various times of day; sampled flights: 2023-04-26 17:26–17:42 (Purkersdorf, Lower Austria), 2023-07-27 07:15–07:43, 2023-01-19 11:29–12:00 (snow likely).
- Onset present: no (no fire anywhere — by design). Smoke-free: EXPECTED but UNVERIFIED (protocol: human review required; e.g., chimney smoke, mist, dust).
- Independent units: 389 flights (394 metadata files; 386 flights referenced by the label-transfer doc), many sites/dates.
- Size: raw parts 40–49 GB each (~47 parts listed on Zenodo); individual flight zips 1.1–7.3 GB (Part 47: flight_96 5.65, _97 7.31, _98 1.11, _99 4.61 GB). Matched-frames zips 0.6–1.8 GB.
- Downloaded: `data/v5/raw/bambi_raw_sample/flight_98.zip` (1.11 GB, md5-checked) — see CARD.
- Blocker: none for download. Gap: not boreal forest; nadir-only; RGB sensor is a Mavic 3T wide camera (lower quality than Phantom 4 4K).

### 2.2 Boreal Forest Fire raw flight recordings — FGI (rank 2, by request)
- Paper: Sci Data 2025, DOI 10.1038/s41597-025-05634-0 (published 2025-08-13). Dataset DOI 10.23729/fd-72c6cf74-b8eb-3687-860d-bf93a1ab94c9 (Fairdata/Etsin), CC BY 4.0 (Metax record). Already on disk (data/v4/raw/boreal, 125.1 GB).
- Evidence that un-released footage exists (paper Methods, fetched today): "Before cutting the videos, the original drone footage was reviewed to remove non-purposeful or sensitive segments, such as identifying information, stationary ground-level footage, or any other irrelevant content. The videos were then segmented into fixed 30-second clips using ... LosslessCut". Flights: DJI Phantom 4, ~10–200 m AGL, < 500 m from the burn, 4K.
- What to ask for: full-length originals (incl. pre-ignition and camera-turned-away segments) for the four 2022 burns, DJI flight logs / SRT timestamps, ignition times. Our split inheritance: heinola → VAL only, ruokolahti → historical test, evo/karkkila → train (PROTOCOL §2).
- Contacts verified: corresponding author **Julius Pesonen, julius.pesonen@nls.fi** (Nature article "Correspondence to Julius Pesonen"; mailto on the article page). Affiliation per Metax: FGI Dept. of Remote Sensing and Photogrammetry, National Land Survey of Finland (+ Aalto University per ORCID 0009-0000-9175-7129). Research manager / senior author **Eija Honkavaara** (ORCID 0000-0002-7236-2145, "Research Manager, Finnish Geospatial Research Institute"; email not public — FGI pattern firstname.lastname@nls.fi is UNVERIFIED). Repository: Fairdata/IDA service desk **servicedesk@csc.fi** (fairdata.fi/en/contact). Rights holder: FGI.
- Needs the user: approve and send the email (draft in EMAIL_DRAFTS_V5.md).

### 2.3 MONIMET phenological cameras — Finland (rank 3)
- URL: Zenodo community "Phenological time lapse images and data from Monimet EU Life+ project"; datasheet record https://doi.org/10.5281/zenodo.777952 ; e.g. landscape camera MC124 Lammi mixed stand https://doi.org/10.5281/zenodo.7500295 (61.05356 N, 25.03898 E).
- Owner: Finnish Meteorological Institute / Natural Resources Institute Finland (Luke) — Peltoniemi, Aurela, Böttcher, Kolari, Loehr. Contact given in the datasheet record: mikko.peltoniemi@luke.fi.
- Licence: Zenodo licence id cc-by-4.0 on every record checked (777952, 7500295). Citation requested: Peltoniemi et al., ESSD 2018 (doi 10.5194/essd-2017-62).
- Contents (record text): "Images are taken half-hourly during fixed day-time period over the year ... Gaps in time series and dark images possibly exist." MC124: 2015-11-23 → 2022, one zip per year (2015–16 9.40 GB, 2017 8.65, 2018 7.30, 2019 7.07, 2020 3.63, 2021 5.88, 2022 5.83 GB). Other cameras in the search results: MC104 Punkaharju spruce (crown), MC113 Suonenjoki pine (canopy), MC115 Kenttärova spruce (crown), MC119 Värriö pine (canopy), MC126/MC127 Lammi birch (ground), MC128 Kaamanen peatland, MC129 Lompolojänkkä peatland (ground) — 15–48 GB each. Landscape-view cameras are the relevant ones for a fixed-camera smoke scenario.
- Onset: none expected. Smoke-free: unverified (research forest; slash burning or chimney smoke not excluded). Confounders: fog, snow, low sun, lens water.
- Independent units: ≥ 7 sites; each camera-year is a continuous series at 30-min cadence (valid-time rules in PROTOCOL §3 apply; report as "30-min sampled").
- Downloaded: `data/v5/raw/monimet_mc124/Monimet_MC124_2020_2020.zip` (3.63 GB) — see CARD.
- Blocker: none. Gap: cadence 30 min (vs 29 s for pyro-sdis), so it measures frame-level FP and seasonal confounders rather than fine-grained episode behaviour.

### 2.4 RxCADRE 2012 — large-burn smoke plume photographs and videos (rank 4)
- URL: https://doi.org/10.2737/RDS-2018-0033 (catalog https://www.fs.usda.gov/rds/archive/Catalog/RDS-2018-0033); ignition data https://doi.org/10.2737/RDS-2017-0065.
- Owner: USDA Forest Service Research Data Archive (Potter, Krull, Larkin, Moore, O'Neill, Rorig, Strand; Pacific Wildland Fire Sciences Lab).
- Licence (verbatim from the catalog page): "These data were collected using funding from the U.S. Government and can be used without additional permissions or fees." (US federal work; public domain.)
- Contents (abstract): "time lapse JPEG and CR2 format photographs, taken every 5 seconds, as well as MTS videos from burns of the 2012 ... RxCADRE at Eglin Air Force Base, Florida. Images are provided from five cameras located around the fire perimeter at various distances for each burn, taken in November 2012. There were two operational grass fires named L1G and L2G, and an operational forest fire named L2F." File index: JPG L1G 11,568 imgs ~72 GB; L2F 7,020 imgs ~41 GB; L2G 11,767 imgs ~92 GB; MTS L1G ~155 GB, L2G ~385 GB; CR2 raw 166–284 GB per burn. Directly downloadable zips seen: RDS-2018-0033_JPG_L1G_CA1-CA2.zip 26.6 GB, _CA3-CA4.zip 33.6 GB (+ more listed); "only a subset of these files are available for direct download ... An external drive can be shipped to us".
- Onset present: YES — cameras ran before ignition; RDS-2017-0065 gives ignition points and lines with times (1 Nov 2012 L1G, 10 Nov L2G, 11 Nov 2012 L2F). Fixed ground-level viewpoint (not UAV). Vegetation: longleaf pine / grass, Florida — not boreal.
- Independent units: 3 burns (× 5 cameras = 15 views; same-burn cameras are NOT independent).
- Blocker: every zip > 20 GB (outside my download cap). Needs the user: a go-ahead to pull 1–2 zips (~60 GB), or authorise the ship-a-drive route for the videos.

### 2.5 FLAME 3 — Computer Vision subset (Sycan Marsh) + NADIR plots (rank 5)
- URL: https://ieee-dataport.org/open-access/flame-3-radiometric-thermal-uav-imagery-wildfire-management ; paper arXiv 2412.02831 (Hopkins et al.; Clemson / NAU / USFS Missoula Fire Lab). Author Kaggle copy exists (brycehopkins) but IEEE DataPort is the primary.
- Licence: page JSON-LD `"license": "https://creativecommons.org/licenses/by/4.0/"` (CC BY 4.0).
- Contents (verbatim): "622 image quartets labeled Fire and 116 image quartets labeled No Fire. The No Fire images are of the surrounding forestry of the prescribed burn plot. Each image quartet is composed of ... a raw RGB image, a raw thermal image, a corrected FOV RGB image, and a thermal TIFF ... This dataset also contains a NADIR Thermal Fire set ... captured by UAV every 3-5 seconds ... Pre, post, and during burn imagery are included". Files: CVSubset.zip 6.23 GB, NADIRPlots.zip 32.59 GB.
- Onset: pre-burn flights exist (nadir, thermal-centric). Aerial UAV: yes. Still images, not video.
- Independent units: 1 burn (Sycan Marsh, Oregon) in this subset; the paper lists further burns (Willamette 233 pairs, Shoetank 3,540 pairs) in the larger FLAME 3.
- Blocker: "LOGIN TO ACCESS DATASET FILES ... Create a free IEEE account. IEEE Membership is not required." → needs the user's IEEE account (or permission to create one in the user's name).

### 2.6 FLAME 1 — aerial pile-burn imagery (rank 6)
- URL: https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs (DOI 10.21227/qad6-r683; Shamsoshoara et al., NAU).
- Licence: JSON-LD CC BY 4.0; DataCite rightsList "Creative Commons Attribution 4.0 International".
- Contents (page text): raw drone videos of a prescribed pile burn, Northern Arizona (winter, snow): video 1 (966 s, 1.2 GB, "Fire-vs-NoFire" training source), video 2 (966 s, 503 MB) "shows the behavior of one pile from the start of burning", thermal videos, a 32 GB 4K MOV test video, 39,375 training frames (254×254) + 8,617 test frames labelled Fire/No-Fire, plus segmentation masks.
- Onset: pile ignition → smoke in video 2 (aerial). No-Fire frames = snowy forest with no fire (aerial). Independent units: 1 burn day.
- Blocker: IEEE login. Size: all repos < 20 GB except the 32 GB MOV.

### 2.7 FLAME 2 — RGB/IR aerial pairs (rank 7)
- URL: https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset (DOI 10.21227/swyw-6j78; IS-WiN Lab, NAU; IEEE Access 2022).
- Licence: JSON-LD CC BY 4.0.
- Contents: 53,451 RGB/IR frame pairs from an open-canopy prescribed burn, Northern Arizona, Nov 2021, labelled Fire/No-Fire and Smoke/No-Smoke; raw video pairs 291 s (3.81 GB zip) and 183 s (2.39 GB zip), unlabeled. Caveat (verbatim): "The Smoke label is applied when the classifiers visually estimate that at least 50% of the RGB frame is filled with smoke" → "No Smoke" frames can contain smoke; they are NOT clean negatives.
- Blocker: IEEE login. Usefulness for V5 low (labels not suited to faint smoke).

### 2.8 Fintraffic / Digitraffic road weather cameras — Finland (rank 8)
- URL: https://www.digitraffic.fi/en/road-traffic/ ; API https://tie.digitraffic.fi/api/weathercam/v1/stations (requires `Digitraffic-User` header and gzip); images https://weathercam.digitraffic.fi/<presetId>.jpg ; history https://tie.digitraffic.fi/api/weathercam/v1/stations/<id>/history
- Owner: Fintraffic (Finnish state).
- Licence (verbatim, digitraffic.fi terms of service): "Fintraffic's open data is licensed under the Creative Commons 4.0 By license. It gives the right to distribute, remix, tweak, and build upon our data, even commercially, as long as you credit the source for the original creation."
- Verified today: 811 stations, 2,277 presets, collectionStatus GATHERING; history for station C01503 shows versioned images every ~10 min over the last 24 h.
- Contents: roadside fixed cameras, many in forested landscapes; ~10-min cadence; no long-term archive (24 h rolling). Onset: none. Clean: unverified (traffic, exhaust, road dust, fog).
- Needs: a CPU collector running continuously (new pipeline); the user to decide whether roadside views count as a fixed-camera deployment surrogate.

### 2.9 Zenodo "Forest Smoke Video Dataset for Wildfire Monitoring" (rank 9) — DOWNLOADED
- URL: https://doi.org/10.5281/zenodo.20506617 (published 2026-06-02). Creators: Hellenic Telecommunications Organization (OTE) and NCSR "Demokritos".
- Licence: cc-by-4.0 (Zenodo record), open.
- Contents (verbatim): "video recordings of smoke generated by a smoke bomb in a forest environment ... capture the temporal and spatial evolution of smoke plumes from different viewpoints and distances ... raw visual data and does not include annotations". 7 MP4 files, 0.80 GB total.
- Onset: artificial (smoke bomb) — usable to test onset-labelling tools and temporal windows, NOT as wildfire ground truth. Independent units: 1 site/campaign (7 clips).
- Downloaded to data/v5/raw/zenodo_forest_smoke_video/ — CARD there.

### 2.10 NASA FireSense FASMEE 2023 (Fishlake NF, Utah) B-roll (rank 10) — DOWNLOADED
- URL: NASA SVS 14559 https://svs.gsfc.nasa.gov/14559 (released 2024-03-29; burn 2023-10-09); Commons copies used for download (files cite SVS 14559).
- Licence: NASA Media Usage Guidelines (fetched): "NASA content – images, audio, video ... generally are not subject to copyright in the United States ... NASA should be acknowledged as the source". Commons extmetadata: "Public domain".
- Contents: DRONES BROLL 3840×2160, 1,105 s (various drones flown during the prescribed burn; aerial footage with smoke); Prescribed Burn Area 2 Day 2 BROLL 3840×2160, 3,662 s (ground-level). Edited B-roll — cuts, not continuous.
- Also downloaded (public domain / CC BY 2.0 via BLM & USFS Flickr → Commons): USFS Northern Region "Prescribed Burn in North Derby Gulch Area, CGNF" (25 s), BLM OR/WA "Time-lapse: Using drip torch for prescribed burn" (20 s), BLM OR/WA "Loveless Prescribed Burn" (47 s, CC BY 2.0).
- Use: human-review/candidate material only; no metric use.

### 2.11 AuSmoke / MultiNatSmoke — licence confirmation (rank 11)
- Re-checked today: GitHub repo tree has no LICENSE file (root: README.md, assets, code); HF dataset card has no licence field/tag. README Contact: **henryzhao0615@gmail.com**. Paper (WACV 2026) authors: Weihao Li, Hongjin Zhao, Gao Zhu, Ge-Peng Ji, Nicholas Wilson, Marta Yebra, Nick Barnes — "Bushfire Research Centre of Excellence, Australian National University". The PDF contains no licence statement and no e-mail addresses. Data collection statement: planned burns in the ACT "managed by ACT Parks and Conservation Service and ACT Rural Fire Service"; PTZ Axis Q6318-LE on a communications tower on Mount Stromlo.
- ANU research portal (fetched): Nick Barnes — "Email nick.barnes anu.edu au"; Marta Yebra — "Email marta.yebra anu.edu au" (rendered with the @ and dot obfuscated; reconstruct as nick.barnes@anu.edu.au / marta.yebra@anu.edu.au — user to confirm). Hongjin Zhao has no portal page (404).
- Status in PROTOCOL: PERMISSION PENDING. Needs the user: approve + send the draft.

### 2.12 HPWREN FIgLib / camera archive (rank 12)
- FIgLib index https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/index.html: "Tar directory of 524 image sequences", last update 18 Dec 2024; 486 distinct sequence tokens parsed, **all 486 are already in data/manifests/figlib_events_index.csv (487 events) → 0 new onset sequences externally**. (Whether some indexed events remain unused by v1–v4 is an internal inventory question, not an acquisition one.)
- Licence (verbatim, https://www.hpwren.ucsd.edu/cc.html): "UC San Diego HPWREN program data is copyrighted under a CC BY-NC-ND 4.0 Creative Commons License. The CC BY-NC-ND 4.0 license is an open-source license BUT does not grant commercial use ... a user intending to download the data for a for-profit (commercial) use will be requested to contact OIC to negotiate a commercial license with appropriate licensing fees." The FIgLib page itself says "This data is provided as-is".
- Camera archive host c1.hpwren.ucsd.edu: no TCP response from this host (http and https) — cannot verify archive availability.
- Needs the user: licence/legal read of NC-ND for our internal for-profit prototype use (also retroactively for FIgLib-39 historical comparisons), and OIC contact if continued.

### 2.13 ALERTCalifornia (rank 13)
- https://alertcalifornia.org/terms/ (verbatim): "ALERTCalifornia's program data is copyrighted under a CC BY-NC-ND 4.0 license ... Footage Requests: If you are the media, a scientist, partner, or work for a government agency please email us through our contact page." No public bulk archive found.
- Needs the user: same NC-ND read; request through their contact page if desired.

### 2.14 FASDD (rank 14)
- DOI 10.57760/sciencedb.j00104.00103 (Science Data Bank; DataCite: "Creative Commons Attribution Share Alike 4.0 International"; size "82125550221 bytes, 6 files"; 2025). Description: "FASDD_UAV" sub-dataset; "names of all images ... prefixed with Fire, Smoke, FireAndSmoke, and NeitherFireNorSmoke". 122,634 samples total; sources include "surveillance cameras, UAVs, and satellites" — web-collected, mixed provenance; still images only. Landing page is JS-only (file sizes not retrievable here).
- Needs the user: decision on CC BY-SA (share-alike) and on web-sourced provenance; otherwise skip.

### 2.15 NPS air-quality webcams (rank 15)
- https://www.nps.gov/subjects/air/webcams.htm fetched; the camera list/archives are JavaScript-rendered (no camera links in the HTML). NPS content is generally US-Government public domain. Archive existence and cadence UNVERIFIED. Needs a manual browser check.

### 2.16 Commons CC-BY drone forest clips (rank 16)
- "Aerial views of a forest in Belarus.webm" (CC BY 3.0, 3840×2160, 103 s, source YouTube), "Aerial view of Snow Lake and Dipping Vat Campground in Gila Forest NM Aug 2018.webm" (CC BY 3.0, 1920×1080, 265 s, source YouTube). Licence explicit but provenance = uploader assertion; viewpoint oblique. Not downloaded; not recommended for any metric.

### 2.17 Zenodo trACT (rank 17)
- https://doi.org/10.5281/zenodo.23057130 (CC BY 4.0, 2026-09-30; Bimber et al., JKU Linz): high-altitude hovering drone over "densely forested wildlife habitats"; Movies.zip 1.0 GB + Data_S1/S2. Not downloaded (BAMBI covers the same content class with far more hours).

### 2.18 BurnedAreaUAV (rank 18)
- https://doi.org/10.5281/zenodo.7944963 (CC BY 4.0, v1.1 2023): DJI Phantom 4 Pro video "captured during the evolution of a prescribed fire" at Torre do Pinhão, Portugal (shrub/herbaceous), frames + burned-area masks; 0.91 GB .rar. Aerial WITH smoke; not a negative source. No `unrar` on this host → not downloaded.

### 2.19 Checked and rejected
- Nemo (SayBender/Nemo): README still says "Download ... from [ADD A URL ONCE DATASET IS PUBLIC]" → no data.
- Zenodo 22877126 "Fire & Smoke Dataset" (Phenikaa, CC BY 4.0 claimed): aggregator re-licensing of third-party corpora (FireSmokeDataset v2 etc.) — provenance not verifiable → rejected.
- Zenodo 22841226 (RT-DETR dataset): no description → rejected.
- Zenodo 17009070 Jasper South wildfire citizen video (CC BY 4.0, 0.59 GB): handheld active-fire footage, no onset → low value, not downloaded.
- HF Simuletic/Long_Distance_Wildfire_Smoke_Detection_Dataset (CC BY-NC 4.0), UniDataPro/fire-and-smoke (CC BY-NC-ND 4.0), AI-for-Mankind & Firecam (CC BY-NC-SA, FIgLib-derived): non-commercial → not for our use without review.
- HF sagecontinuum/smokedataset (MIT): derived from HPWREN imagery → inherits the HPWREN question above.
- USFS WFSI "A multiscale study ... video from 2022-2025" (10.60594/W4MG6F): liquid-pool / lab-scale fires with 360° cameras, not wildland smoke.
- Figshare, CSIRO DAP, data.gov, data.gov.au, open.canada.ca searches: no relevant video datasets (data.gov API 404; CSIRO 0 results; data.gov.au only gazette PDFs).
- DVIDS (US military video hub): bot challenge (HTTP 202, empty) — not checked.

## 3. What the user needs to do (consolidated)
1. Approve/send the FGI email (Boreal raw flights) — EMAIL_DRAFTS_V5.md §1.
2. Approve/send the ANU email (AuSmoke licence) — EMAIL_DRAFTS_V5.md §2.
3. IEEE DataPort: log in (free account) and download FLAME 3 CVSubset.zip (6.2 GB) [+ NADIRPlots 32.6 GB], FLAME 1 videos (≤1.2 GB each), optionally FLAME 2 raw pairs — or authorise an account for the agents.
4. Decide on RxCADRE: allow pulling 1–2 JPG zips (26.6 + 33.6 GB) now; optionally arrange the drive shipment for the MTS videos.
5. Licence read: HPWREN & ALERTCalifornia CC BY-NC-ND 4.0 — does our internal research-prototype use qualify; if not, contact UCSD OIC. This also affects how FIgLib-39 is cited going forward.
6. Human visual review of the downloaded candidates (BAMBI flight_98, MONIMET MC124-2020 sample) before any of it is marked "verified no-smoke".
7. Decide whether Fintraffic roadside cameras are an acceptable fixed-camera surrogate (if yes, agents can start a CPU collector).

## 4. Explicitly deferred
- UAV no-smoke in BOREAL forest with the Boreal camera/altitude profile: only via FGI (lead 2). Until then, aerial-negative acceptance (≥2 h, ≥3 units, human-verified) can only be approached with BAMBI (temperate, nadir) — a domain mismatch that must be stated in any result.
- Real wildfire ignition→smoke video in UAV view: none found with an explicit licence. FLAME 1 pile ignition (IEEE login) is the closest.

## 5. Downloads performed today (all under data/v5/raw/, each with a CARD and _download_log.jsonl)
| dir | source | licence | size | purpose tag at ingest |
|---|---|---|---|---|
| zenodo_forest_smoke_video/ | Zenodo 10.5281/zenodo.20506617 | CC BY 4.0 | 0.80 GB, 7 mp4 | onset-TOOL TESTING only (artificial smoke) — never a metric set |
| commons_usgov_burn_videos/ | NASA SVS 14559 (2 files), USFS R1, BLM OR/WA (2) via Wikimedia Commons | Public domain (US Gov) / CC BY 2.0 | ~1.05 GB, 5 webm | human-review candidate material |
| bambi_raw_sample/ | Zenodo 10.5281/zenodo.19177615 flight_98.zip | CC BY 4.0 | 1.11 GB | AERIAL NEGATIVE candidate — unverified, review before use |
| monimet_mc124/ | Zenodo 10.5281/zenodo.7500295 (2020) | CC BY 4.0 | 3.63 GB | FIXED-CAMERA NEGATIVE candidate (30-min cadence) — unverified |
Assignment (PROTOCOL §2, "assigned by whole site/flight/date at ingest"): proposed CALIBRATION/VAL for all four (none to training) until the human review and the inventory agent record them in data/v5/manifests; nothing here has been seen by any model.

## 6. Measured numbers for the downloaded items (cv2 / zip inspection, 2026-10-02)
| dir | measured |
|---|---|
| zenodo_forest_smoke_video | 7 clips, 1440x1080 (6) / 1920x1080 (1), 30 fps, 43.8–93.5 s each, **total 474 s = 7.9 min**; md5 verified 7/7 |
| commons_usgov_burn_videos | NASA drones B-roll 3840x2160, 1,105.6 s; NASA ground B-roll 3840x2160, 3,662.7 s; BLM drip-torch time-lapse 20.4 s; BLM Loveless 47.2 s; USFS North Derby Gulch 24.6 s (AV1, not decodable by our cv2) — **edited cuts, ~80 min decodable** |
| bambi_raw_sample | flight_98.zip = ONE 96-s segment: RGB 3840x2160 @ 29.97 fps + thermal 640x512, per-frame SRT (17:30:33–17:32:10 local, rel_alt 63–67 m, gimbal pitch −90°), Airdata CSV; md5 verified. **~1.6 min** of candidate aerial no-smoke video, unverified |
| monimet_mc124 | 6,375 JPGs, 2020-01-10 11:01 → 2020-11-02 20:01 (UTC+2), captures 04:00–22:30 at 30-min cadence (uniform ~330–350 per hour-of-day), months uneven (May 27, Sep 94, Nov 71, Dec 0 images → gaps); ≈ **168 camera-days**; md5 verified |

Implication for acceptance (PROTOCOL §6: ≥2 h valid, ≥3 independent units, human-verified, per scenario):
- AERIAL: today's sample is 1.6 min / 1 unit. BAMBI can supply the hours (each flight zip 1–7 GB; ~389 flights) but every
  minute still needs a human no-smoke pass, and the domain is nadir/temperate. The boreal-matched route remains the FGI request.
- FIXED-CAMERA: MONIMET gives ~168 camera-days at 30-min cadence from ONE unit; two more landscape cameras/years (15–48 GB
  each) would meet the unit count, at a cadence that only supports frame-level FP, not 29-s episode dynamics.
