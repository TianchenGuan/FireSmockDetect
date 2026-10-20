# V5 Day-1/2 — candidate false positives on NEW, UNVERIFIED no-smoke footage (MONIMET MC124, BAMBI flight_98)

Written 2026-10-02 02:30 UTC. **Every number here is "candidate FP on UNVERIFIED footage: detector output is not evidence of
smoke or of its absence" (PROTOCOL_V5 §6).** Nothing below verifies either set as no-smoke; it sizes and prioritises the human
review. Scores are historical replays (source_tag = historical_replay), never field runs. No model has trained on either source.

## 0. Headline
* MONIMET (fixed camera, 6,375 frames, 183 camera-days, 30-min SAMPLED cadence): at the Day-1 fixed-camera operating points the
  YOLO systems flag **5.4 % of frames** (expB 344, expE 343; 100 / 92 of 183 camera-days touched), the DINO LoRA systems
  **0.5–0.8 %** (B4 49, B6 33). 516 frames (8.1 %) alert for >=1 system, only 5 for all four, 23 for both a YOLO and a DINO system.
* **The "no-smoke" candidate contains real smoke**: agent screening (full-res) of two 4-system frames shows a dense white plume
  (2020-04-21 17:01 local) and a thin grey column (2020-04-20 17:01) above the far-left farm buildings — field/slash burning.
  Human confirmation needed; such frames are not FPs and must leave the negative pool.
* BAMBI (aerial nadir, 96 s at 1 fps): expB 1, E_ET 2, D_curr 3 alerting frames (frames 15/37 and 90–92; 90–95 are over
  houses, not forest); B4/B6 output **no box at all** (silence, not evidence). One 96-s segment cannot carry an episode rate.
* MONIMET's 30-min cadence makes it a **frame-FP source, not an episode source** (§2); with the production engine every alert
  there is a single-frame provisional and every alert frame is its own "episode".
* Acceptance (§6: >=2 h valid, >=3 independent units, human-verified) is NOT met and cannot be met from these two downloads:
  MONIMET is ONE camera-year (one independent unit), BAMBI is 1.6 min of one flight.

## 1. Material (cards: data/v5/raw/<set>/CARD.md; ingest record data/v5/manifests/newfootage_day1_ingest.csv + 2 rows appended to DATA_INVENTORY_V5.csv BEFORE scoring, PROTOCOL §2)
| set | what | frames | units | resolution | licence | assignment at ingest |
|---|---|---|---|---|---|---|
| monimet_mc124 | Lammi (FI) mixed-stand landscape phenocam MC124, 2020-01-10 → 2020-11-02, 04–22 h local (UTC+2 as written in file names) | 6,375 JPG | 183 camera-days (card said 168: 183 distinct dates incl. partial days) | 2592x1944 | CC BY 4.0 (Zenodo 10.5281/zenodo.7500295) | fixed-camera CALIBRATION/VAL candidate, whole camera-year |
| bambi_flight98 | BAMBI raw flight_98 "point1", Purkersdorf (AT) 2023-04-26 17:30 local, Mavic 3T RGB, gimbal −90°, 63–67 m AGL | 96 (1 fps from 2,873 @ 29.97 fps, 95.9 s) | 1 segment | 3840x2160 | CC BY 4.0 (Zenodo 10.5281/zenodo.19177615) | aerial CALIBRATION/VAL candidate, whole flight |
Not scored (per cards): commons_usgov_burn_videos (edited B-roll, review material only), zenodo_forest_smoke_video (smoke-bomb clips, tool testing only).

## 2. Cadence and VALID observed time (PROTOCOL §3)
* MONIMET: median gap 30.0 min (p10 = p90 = 30.0); 200 gaps > 90 min (= 3x cadence) excluded (182 day boundaries + 18 intra-day);
  §3 covered time = **3,217.97 h** (`review_common.valid_observed_seconds`: a frame covers its following gap if <= 90 min, else one
  cadence; last frame one cadence). Units for episode counting = camera-day (`data/v5/newfootage/units_monimet_mc124.csv`).
* **Sampled-rate statement**: each MONIMET frame is a ~1-s snapshot standing for 30 min. "Episodes per valid hour" below is a
  30-min SAMPLED rate — it is NOT comparable to the 29-s pyro-sdis figure (29.4/h for expB) or to any continuous rate, and it
  cannot resolve episode duration, K-of-N confirmation or cooldown. The honest comparator is the frame-level alert fraction.
* Engine fact: cadence 1,800 s > `stale_after_s` 600 s, so the service engine resets history on every MONIMET frame; the K-of-N
  (3/5) confirmed path can never fire, only the single-frame provisional path at conf_single; cooldown 60 s << cadence, so
  episodes == alert frames by construction (verified: 344 == 344 etc.).
* BAMBI: 1 fps, 96 covered seconds = **0.0267 h**, one device, fresh engine; a clip this short holds at most ~1 episode per system,
  so the rate is 0 or 37.5/h with an exact-Poisson interval of [0.9, 209]/h — reported for completeness only.

## 3. Systems and operating points (Day-1 calibrations; weights as in scripts/v5/calibrate_aerial.py SYSTEMS / CALIBRATION_FIXED §1)
Fixed (CALIBRATION_FIXED_V5 §5 CHOSEN; engine 3/5, cooldown 60 s, stale 600 s, floor 0.05): expB 0.30/0.23 · expE 0.32/0.20 ·
B4 0.91/0.89 · B6 0.94/0.88. Aerial (CALIBRATION_AERIAL_V5 §3 conf*, single = seq, floor 0.02): expB 0.1070 · E_ET 0.2160 ·
D_curr 0.2120 · B4 0.9890 · B6 0.9934. Inference identical to the calibrations: YOLO ultralytics predict imgsz 1280 conf 0.02
on CPU (23 shard processes x 8 threads, 15 min); DINO B4/B6 on the ONE allowed GPU node via `cluster/v3/submit_dino_eval.py
--frames-tar` (4.0 GB tgz of all frames, `--rootmap-override data/v5/frames=/tmp/frames`, `--infer-extra "--lora 16"`,
1280x736), registry `res-v5score` reserved 01:50Z → released 02:16Z, 2 jobs, **0.31 node-h**, 6,471 rows each, 0 errors.
Preds: `models/v5/evals/<system>__<set>.jsonl` (`{"image","w","h","preds":[[x1n,y1n,x2n,y2n,score]..],"source_tag":"historical_replay"}`) + `.meta.json`.

## 4. Per-system results — candidate FP on UNVERIFIED footage: detector output is not evidence of smoke or of its absence
### 4.1 MONIMET MC124 (fixed-camera systems; 30 min SAMPLED; episode == alert frame)
| system | conf_single/conf_seq | alert frames / 6,375 | frame alert rate | camera-days with >=1 alert / 183 | episodes | episodes per valid h (SAMPLED) [exact 95 %] | frac. covered time in alert |
|---|---|---|---|---|---|---|---|
| expB (deployed default, YOLO11s@1280) | 0.30/0.23 | 344 | **5.40 %** | 100 | 344 | 0.107 [0.096, 0.119] | 0.0536 |
| expE (v3 S0, YOLO11m@1280) | 0.32/0.20 | 343 | **5.38 %** | 92 | 343 | 0.107 [0.096, 0.118] | 0.0539 |
| DINOv3-L LoRA r16 (B4) | 0.91/0.89 | 49 | **0.77 %** | 33 | 49 | 0.015 [0.011, 0.020] | 0.0076 |
| DINOv3-H+ LoRA r16 (B6) | 0.94/0.88 | 33 | **0.52 %** | 22 | 33 | 0.010 [0.007, 0.014] | 0.0051 |
Union: 516 frames alert for >=1 system (287 for exactly one, 210 for two, 14 for three, 5 for four); 118/183 camera-days touched;
YOLO-any 475, DINO-any 64, overlap 23. Combos: expB+expE 201, expE only 130, expB only 121, B4 only 28, B6 only 8.
Camera-days with >=1 / >=3 / >=5 / >=10 union alerts: 118 / 71 / 43 / 10 (max 18 of 38 frames: 2020-03-09, 2020-10-31).
By month (any-system alerts / frames): Jan 101/818, Feb 110/1098, Mar 87/1174, Apr 50/805, May 0/27, Jun 4/207, Jul 3/550,
Aug 0/377, Sep 0/94, Oct 127/1154, Nov 34/71 — i.e. the leafless / low-sun / overcast seasons; DINO alerts (B4: Jan 13, Feb 12,
Mar 7, Apr 4, Jun 2, Oct 11) follow the same pattern. Alerts are spread over the whole daylight range (3-h local buckets
03: 24, 06: 90, 09: 110, 12: 106, 15: 99, 18: 58, 21: 29).
Context only (different domain, different cadence — not an equivalence claim): on pyro-sdis VAL embedded negatives the same
operating points flag 37–39 % of negative frames for all four systems (CALIBRATION_FIXED §5); on this Finnish farm/forest
landscape the YOLO systems flag ~7x more frames than the DINO systems, with little overlap.

### 4.2 BAMBI flight_98 point1 (aerial systems; 1 fps; 0.0267 valid h; one unit)
| system | conf* | alert frames / 96 | frame alert rate | episodes | episodes per valid h [exact 95 %] | frac. time in alert | frames with any box >= 0.02 |
|---|---|---|---|---|---|---|---|
| expB (deployed default) | 0.1070 | 1 (#90) | 1.04 % | 1 | 37.5 [0.9, 208.7] | 0.0104 | 4 (max 0.324) |
| E_ET (v4e_s1) | 0.2160 | 2 (#91, #92) | 2.08 % | 1 | 37.5 [0.9, 208.7] | 0.0208 | 4 (max 0.352) |
| D_curr (v4dada_s1) | 0.2120 | 3 (#15, #37, #92) | 3.12 % | 1 | 37.5 [0.9, 208.7] | 0.0312 | 93 (max 0.293) |
| DINOv3-L LoRA r16 (B4) | 0.9890 | 0 | 0.00 % | 0 | 0 [0, 138.2] | 0 | **0** |
| DINOv3-H+ LoRA r16 (B6) | 0.9934 | 0 | 0.00 % | 0 | 0 [0, 138.2] | 0 | **0** |
Frames 90–95 (the last 6 s) are over a residential edge (roofs, gardens, blossoming trees), not forest; 3 of the 5 alerting frames
are there. B4/B6 produce no detection >= 0.02 on any nadir frame — consistent with the aerial calibration's AuSmoke finding that
the heinola-selected DINO points do not transfer; here it is also "no box", which says nothing about smoke absence.

## 5. What a human must look at (package: data/v5/review/newfootage_day1/, README.md there)
| set | to verify the detector outputs (candidate FPs) | to verify the SET as no-smoke (§6) |
|---|---|---|
| MONIMET | **516 alert-any frames** + 365 spot-check frames (2 per camera-day) = **881 frames on 42 sheets in 11 month-units**; per-frame template `frame_verdicts_TEMPLATE_monimet_mc124.csv` (881 rows); priority = the 19 frames alerting for >=3 systems | all **6,375 frames / 183 camera-days (266 sheets at 24/sheet)** — detector silence is not evidence; and still only ONE independent unit |
| BAMBI | 5 alert frames (#15, #37, #90, #91, #92) | all **96 frames on 4 sheets** (1 unit; ~2 min of viewing) |
The 19 >=3-system MONIMET frames (local time; expB/expE/B4/B6 scores): 01-16 07:01 (.12/.35/.94/.97), 01-22 15:01, 01-29 06:31,
02-04 10:31, 02-06 14:31, 02-20 07:01, **02-28 07:01 (4 systems)**, 02-28 07:31, 02-28 22:01, 03-03 10:31, 03-03 11:31, 03-08 14:01,
03-12 04:01, **04-20 17:01 (4)**, **04-21 17:01 (4)**, **06-08 18:31 (4)**, 06-09 09:31, 10-14 07:31, **10-15 07:31 (4)**.
Agent screening (NOT a verdict; 480-px sheets + two full-res frames): 04-21 17:01 = dense white plume from a field at far left,
04-21 17:01 and 04-20 17:01 = smoke above the far-left farm; the heavy YOLO-only days (03-09, 10-30 → 11-02: 12–18 alerts of 38,
B4/B6 = 0) are flat overcast grey scenes with no visible plume (candidate FP: dull light); October sheet 1 shows low-sun lens
flare / dusk frames as the other recurrent pattern. Sheets with the 4-system frames: 202002/sheet_006, 202004/sheet_003 (#77,
#84), 202006/sheet_000 (#1), 202010/sheet_002 (#60).

## 6. Caveats (state, do not soften)
* MONIMET: a 30-min cadence is a frame-FP source, not an episode source (§2). Even fully verified it is one camera-year = one
  independent unit; §6 needs >=3 (other MONIMET cameras/years: ACQUISITION_LEADS_V5 §2.3, 15–48 GB each). The scene is rural
  farmland + forest edge with a farm and a village in view: chimneys and field burning are expected, so "no-smoke" will be
  per-frame, not per-set; a human must decide which smoke frames (if confirmed) move to the onset/positive side.
* BAMBI: 1.6 min, evening light, nadir only, partly non-forest; ~75+ such segments from different flights/sites would be needed.
* Operating points were chosen on pyro-sdis VAL (fixed) / heinola (aerial); the DINO points sit in the saturated tail
  (0.91–0.9934) and are fragile (CALIBRATION_AERIAL §8). Frame-FP fractions here are at those points only; no new threshold was
  tuned on this footage and none should be before the human verdicts exist.
* `negative_review.py` `units.csv`/`verdicts.csv` in the package cover the SELECTED frames (subset), and its displayed
  "candidate hours" (1,756 h) is the hold-rule arithmetic on that subset — not a verified or authoritative quantity.
* Agent screening covered 4 of 46 sheets and 2 full-res frames; it prioritises, it does not verify.

## 7. Artifacts
`scripts/v5/newfootage_extract_bambi.py` (1-fps extraction, SRT wall-clock), `scripts/v5/newfootage_candidate_fp.py`
(lists / score-yolo / score-dino / merge / metrics / review-lists), `scripts/v5/newfootage_report_tables.py`;
`scripts/v5/review_common.py` gained `_ISO2_RE` so `YYYY_MM_DD_HHMMSS` phenocam names parse (backward-compatible).
Frames `data/v5/frames/{monimet_mc124,bambi_flight98/frames}`, lists `data/v5/frames/lists/*.txt`, tar `data/v5/stage/newfootage_frames.tgz`;
tables `data/v5/newfootage/{cadence_summary.json, units_*.csv, frames_*.csv, frame_scores_*.csv, metrics_*_units.csv, metrics_*_summary.json}`;
preds `models/v5/evals/*__{monimet_mc124,bambi_flight98}.jsonl` (+ raw cluster outputs `models/v3/evals/v4B{4,6}_*_newfootage/`);
logs `logs/v5/{yolo,dino}/`. Not touched: STATUS.md, reports/v4/, TEST splits, lockboxes.
