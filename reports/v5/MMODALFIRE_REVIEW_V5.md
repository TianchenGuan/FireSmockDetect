# MmodalFire interference / early-fire review — V5 Day-1 (ai_agent_proposal)

Date 2026-10-01. CPU only, PIL contact sheets, no model outputs used. Nothing
here is a human label; every artefact is a PROPOSAL for the human labelling
pass required by PROTOCOL_V5 section 2 ("AUXILIARY TRAIN candidate after
human visual labelling; never a forest/UAV eval").

## 1. Material and counts
- Source: 23,571 frames (~1 fps, 1920x1080) in 36 sessions + 28 raw DVR MP4s;
  one indoor lab room, one fixed camera, 2022-08-02..10 (card:
  data/v4/manifests/mmodalfire_CARD.md). Labels are session-level fire /
  interference / blank only; smoke is never annotated in the deposit.
- Reviewed here: 47 contact/zoom sheets, 820 index rows, 710 distinct
  frames from 16 sessions (Mist1-4, DryIce1-4, BlankControl1-2, Cotton7,
  NHeptane1, Polyurethane2/3, Wood1/3). Not reviewed: the other 20 sessions.
- Outputs (all under data/v5/mmodalfire/): sheets/ (+ sheet_index.csv),
  hard_negative_candidates.csv (100 rows), onset_candidates.json (4 events),
  AUX_LABEL_PACKAGE/ (300 frames, 71.8 MB, proposed_labels.csv, README.md),
  scripts/sheets_lib.py. Leftovers of the earlier attempt (_review_sheets/,
  sensor_aid/, presession_frames/, plume_proxy.csv) were reused, not deleted.

## 2. Interference and blank sessions — what I saw
- Mist1, Mist2 (floor humidifier): a white, ground-hugging plume leaves the
  nozzle and drifts right/down in EVERY sampled frame -> excellent smoke
  look-alike hard negatives (visible_mist=yes, smoke_lookalike=yes, 20/20).
- Mist4 (humidifier on rack, tall tube): diffuse white cloud at the tube
  outlet, dense by i260-i330; operator in frame at times; last sampled frame
  (i359) shows switch-off (9 yes / 1 uncertain).
- Mist3: humidifier on rack with tall tube, LED on, but NO plume discernible
  at 1080p in either the centre or the nozzle crop (10/10 uncertain, not a
  look-alike). Treat as "labelled interference, visually blank" until a human
  checks full-res frames.
- DryIce1-4: no fog anywhere (full frame and centre crop); room identical to
  blank control; people walk through in 8/40 frames. Sensor-only confounder.
- BlankControl1-2: empty room, people in 5/20 frames, no haze seen. Caveat:
  they start 61 s after Mist4 ended, and a global +20 grey-level brightness
  step appears at Mist4 i320 and reverts at BlankControl2 i~45 (one-frame
  steps -> most likely a lamp/exposure change, not haze).
- Confounders everywhere: pedestrians, coloured lamps (pink/purple in
  polyurethane sessions), fixed pale streaks on the left wall, burned-in OSD
  clock and watermark.

## 3. Early-fire segments — onset candidates (idx = 0-based 1 fps position in session)
| event | last clear no-smoke | first clear smoke | uncertain span | confidence | key facts |
|---|---|---|---|---|---|
| Cotton7 | 84 (11:38:24) | 124 (11:39:04) | [88,124] | LOW | flame on cotton pile from i88; plume barely resolvable; room haze only from ~i440; pre-session minute clear |
| NHeptane1 | 42 (16:03:42) | 61 (16:04:01) | [43,61] | MED-LOW | pan ignited i43; flame dominant; thin soot wisp from i61; no whiteout |
| Polyurethane2 | 28 (15:37:28) | 68 (15:38:08) | [32,68] | MEDIUM | lighter at i28, flame i32, wisps i44-60, definite puff i68; dense smoke by i170 |
| Wood1 | 620 (10:00:29) | 760 (10:02:49) | [640,760] | MEDIUM | no plume at all, diffuse haze; contrast std 33 -> 29 (i680) -> 24 (i760) -> 17 (i1080); NOT in first 90 s |
All four: starts_with_smoke=false, camera_turn_before_smoke=false.
Sensor first-rise times (audit aid only) sit inside or just after each
proposed span (Cotton7 i92, NHeptane1 i62, Polyurethane2 i54) except Wood1,
where the sensor rises ~7 min before the camera sees haze (i320 vs i760).
Protocol acceptance needs >=10 independent events with reviewed intervals;
these 4 are a start, and 3 of 4 are flame-first events where "first clear
smoke" is genuinely ambiguous at 1080p.

## 4. AUX_LABEL_PACKAGE (300 frames, 75 per class)
- faint_smoke: Wood1 20, Wood3 15 (haze, no flame); Cotton7 15, NHeptane1 10,
  Polyurethane2 15 (flame with thin plume; confidence low/medium).
- dense_smoke: Wood1 25, Wood3 20 (white whiteout); Polyurethane2 15,
  Polyurethane3 15 (dense smoke under coloured lamps).
- mist: Mist1 25, Mist2 20, Mist4 30 (Mist3 excluded: no visible plume).
- blank: BlankControl1 20, BlankControl2 20, DryIce1-3 20, pre-ignition
  frames of Cotton7/NHeptane1/Wood1 15 (fire-labelled sessions, clear room).
- proposed_labels.csv: origin=ai_agent_proposal, empty human_confirmed /
  human_label columns, supporting_sheets per row. README states: indoor, one
  room, one fixed camera, one week; auxiliary training candidate only; never
  a forest/UAV evaluation set; sensor CSVs are audit aids, not inputs; CC BY 4.0.

## 5. What a human must confirm (exactly)
1. Mist3: is there any plume at full resolution? If not, re-class the session
   as visually blank (sensor-only interference) rather than look-alike.
2. BlankControl1/2: no residual humidifier haze despite following Mist4 by
   61 s; interpret the brightness step at Mist4 i320 / BlankControl2 i~45.
3. For each onset event, accept or move first_clear_smoke_idx, and rule
   whether flame-with-thin-plume (Cotton7, NHeptane1, early Polyurethane2)
   counts as "clear smoke" for V5 onset truth; otherwise mark the event
   flame-only and exclude it from the early-warning denominator.
4. Wood1: confirm the diffuse contrast loss at i640-i760 is smoke and not
   exposure drift (the hot plate ignites nothing visible; sensor rose at i320).
5. Spot-check every proposed_labels.csv row, especially confidence=low rows
   and all frames with people, then fill human_confirmed/human_label.
6. Decide the OSD/watermark masking rule before any training use.

## 6. Limits
- Labels were extrapolated between inspected frames (1-20 s spacing); the
  Wood1 span carries +/-20 s index slop; XLSX session bounds carry +/-30 s.
- Only 4 of 26 fire sessions were walked for onset; Polyurethane1's sensor
  rises 9 s after session start (likely starts_with_smoke) but was not viewed.
- No Wood2/Wood4/Cotton1-6,8,9/NHeptane2-8 frames were inspected; none are in
  the package.
- Test/lockbox data, STATUS.md and reports/v4/ were not touched.
