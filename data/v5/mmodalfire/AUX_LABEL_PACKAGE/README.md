# MmodalFire auxiliary labelling package (V5, Day-1 proposal)

Status: PROPOSED LABELS ONLY. Every label in `proposed_labels.csv` has
`origin=ai_agent_proposal`; none is a human label. Nothing here may be used
for training, calibration or evaluation until a human has filled
`human_confirmed` / `human_label` for each frame.

## What this is
- 300 JPEG frames (1920x1080, ~1 fps extracts) copied from
  `data/v4/raw/mmodalfire/frames_1fps/` into `frames/<proposed_class>/`.
- Source: MmodalFire (Zhang et al., Scientific Data 2026; figshare 28804448).
  Licence: CC BY 4.0 (record: data/v4/raw/mmodalfire/LICENSE_TERMS.txt).
  Attribution required in any derived artefact.
- Scene: INDOOR, ONE ROOM, ONE FIXED CAMERA, ONE WEEK (2022-08-02..10).
  All 300 frames come from 15 sessions of the same lab; adjacent frames are
  near-duplicates. For any grouped split treat the WHOLE SOURCE as one site.

## Intended use and hard limits
- AUXILIARY TRAINING CANDIDATE ONLY (protocol V5 section 2), after human
  visual labelling.
- NEVER a forest / UAV / fixed-camera-outdoor evaluation set, never used for
  calibration or model selection, never counted toward V5 negative hours.
- The MmodalFire sensor CSVs (smoke density, temperature, IR, UV) were used
  only as LABEL-AUDIT AIDS to cross-check onset times. They MUST NOT become
  model inputs or features.
- Burned-in Chinese OSD clock (top-left) and lab watermark (bottom-right) are
  on every frame: crop or mask them at training time to avoid shortcuts.

## Classes (balanced 75 / 75 / 75 / 75)
| proposed_class | meaning | subtypes |
|---|---|---|
| faint_smoke | thin plume or early diffuse haze; flame may be present | wood_haze_early, flame_with_thin_plume |
| dense_smoke | room-filling smoke / near whiteout, often under pink/purple lamps | room_filling_white_smoke, dense_dark_smoke_coloured_light |
| mist | humidifier plume (smoke look-alike), labelled non-fire | humidifier_plume_floor_unit / _rack_unit |
| blank | no smoke, no mist visible | blank_control, dry_ice_no_visible_fog, pre_ignition_fire_session |

`proposed_smoke_visible` is the binary target proposal (yes for the two smoke
classes, no otherwise). `confidence` is the agent's own confidence; `low`
rows (cotton / n-heptane flame-with-thin-plume) are the ones most likely to be
overturned.

## How the proposals were made
Frames were inspected on 4x5 contact sheets and 4x3 zoom crops
(`data/v5/mmodalfire/sheets/`, index `sheet_index.csv`); labels were then
extrapolated to evenly spaced frames inside each inspected interval
(`supporting_sheets` column). No model output was used.

## What a human must do
1. Confirm or overturn `proposed_class` / `proposed_smoke_visible` per frame.
2. Decide whether flame-with-thin-plume frames count as smoke-visible.
3. Check `blank` frames from BlankControl1/2 for residual humidifier haze
   (they start 61 s after Mist4) and note the lamp/exposure step at
   BlankControl2 i~45.
4. Record people-in-frame where relevant (common in all classes).
