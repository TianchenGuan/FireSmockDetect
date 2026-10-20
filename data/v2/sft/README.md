# SFT dataset v2 - Qwen3.5 smoke detection (JSON output)

Built by `scripts/build_sft_v2.py` (seed 0, deterministic). Verified by
`scripts/test_sft_roundtrip.py` (round-trip IoU + contact sheets in `verify/`).
Stats: `DATASET.md` / `stats.json`.

## Files
- `train.jsonl` - from `data/yolo/lists/local/expB_train.txt` ONLY (frozen train split:
  pyro-sdis train sites + D-Fire train supplement). Shuffled with seed 0.
- `val_mini.jsonl` - 600 full-image examples derived from `sdis_val.txt` (FROZEN VAL SPLIT).
  For quick TRAINING-TIME eval only - NEVER train on it. Every id is prefixed `fromval:`.
- `overfit_16.jsonl` - 16 diverse train examples (8 pos / 8 neg) for tiny-overfit smoke test.

## JSONL schema (one example per line)
- `id`: unique, `<source>:<image-stem>:<full|cropP|cropN>` (val_mini adds `fromval:` prefix)
- `image_path`: absolute path to the ORIGINAL image (crops are never materialized)
- `crop_box`: `null` for full-image examples, else `[x1,y1,x2,y2]` in PIXELS of the
  original image, x2/y2 EXCLUSIVE (crop width = x2-x1). The training collator must crop
  `img[y1:y2, x1:x2]` on the fly and feed the crop as the model input.
- `instruction`: one of the 4 fixed phrasings below (assignment = md5 hash of id, i.e. seeded/deterministic)
- `answer`: target string, COMPACT JSON:
  `{"smoke_present":true,"boxes":[[x1,y1,x2,y2],...]}` or `{"smoke_present":false,"boxes":[]}`
- `n_boxes`: number of boxes in `answer`
- `source`: `pyro-sdis` | `dfire`
- `tiny`: bool - any box with sqrt(area_frac) < 0.03, area_frac relative to the MODEL INPUT
  (the crop for crop examples, the full image otherwise)

## Coordinate convention (exact)
Boxes are `[x1, y1, x2, y2]`, top-left origin, `x1<x2`, `y1<y2`, each value an INTEGER in
0..1000 = absolute-normalized thousandths of the model-input image:
`x_t = round(1000 * x_px / input_width)`, `y_t = round(1000 * y_px / input_height)`
(clamped to [0,1000]; degenerate boxes expanded to >=1 thousandth). For crop examples the
input is the crop window, so boxes are re-normalized to crop coordinates. Decoding:
`x_px = x_t / 1000 * input_width`.

Derived from YOLO labels (`class cx cy w h`, normalized): `x1 = cx - w/2`, etc., clamped
to [0,1], then quantized as above. Empty label file = verified negative (audited) ->
`{"smoke_present":false,"boxes":[]}`. No fabricated boxes or explanations anywhere.

## Local crops (small-smoke preservation)
- For every train positive with any box `area_frac = w*h < 0.01`: ONE 640x640 crop
  (clipped at image borders, so <=640 on small images). A random qualifying box is the
  anchor; the window is placed UNIFORMLY at random among all positions that fully contain
  the anchor (no center bias). Up to 20 placements are tried to avoid partially cutting
  other boxes; if unavoidable, cut boxes are clipped to the window and kept when >=2px on
  each side and >=2% of their area is visible (occurrences counted in stats.json).
- For a seeded random 20% of train negatives: ONE random 640x640 crop, empty boxes.

## Instruction phrasings (fixed list)
1. `Detect wildfire smoke in this image. Reply with compact JSON only: {"smoke_present": true|false, "boxes": [[x1,y1,x2,y2],...]}. Coordinates are integers from 0 to 1000, normalized to the image width and height. If there is no smoke, answer {"smoke_present": false, "boxes": []}.`
2. `Is there smoke in this image? Answer with a single JSON object {"smoke_present": ..., "boxes": ...} where each box is [x1,y1,x2,y2] corner coordinates on a 0-1000 scale relative to the full image; use an empty boxes list when no smoke is visible.`
3. `You are a wildfire early-warning system. Inspect the image and output JSON {"smoke_present": bool, "boxes": list} - each box [x1,y1,x2,y2] with integer coordinates in thousandths (0-1000) of the image dimensions. Output nothing besides the JSON.`
4. `Look for smoke plumes. Respond only with compact JSON of the form {"smoke_present": true|false, "boxes": [[x1,y1,x2,y2], ...]}; coordinates are integers scaled 0-1000 across the image width and height. Report every smoke region; boxes must be [] if smoke_present is false.`
