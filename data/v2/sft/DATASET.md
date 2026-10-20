# DATASET.md - SFT v2 (Qwen3.5 smoke detection, JSON output)

Generated 2026-09-11 by `scripts/build_sft_v2.py` + `scripts/test_sft_roundtrip.py` (seed 0).
Schema and conventions: `README.md`. Machine-readable stats: `stats.json`.

## Counts
| file | examples |
|---|---|
| train.jsonl | 59999 |
| val_mini.jsonl | 600 (400 pos incl. 200 tiny, 200 neg; full-image; FROM FROZEN VAL - training-time eval only) |
| overfit_16.jsonl | 16 (8 pos / 8 neg, both sources, full+crop, incl. tiny + multi-box) |

train.jsonl composition: 38454 full-image + 19294 positive local crops (any box area_frac<0.01) + 2251 negative random crops (20% of negatives).
Cap 70000: total 59999 <= cap -> **0 full-image negatives dropped** (dropped ids recorded in stats.json).

Breakdowns (pyro pos | pyro neg | dfire pos | dfire neg | pos | neg | tiny):
| subset | counts |
|---|---|
| train (all) | 36356 | 4141 | 10135 | 9367 | 46491 | 13508 | 9690 |
| train full-image | 18753 | 3424 | 8444 | 7833 | 27197 | 11257 | 7236 |
| train pos crops | 17603 | 0 | 1691 | 0 | 19294 | 0 | 2454 |
| train neg crops | 0 | 717 | 0 | 1534 | 0 | 2251 | 0 |
| val_mini | 400 | 200 | 0 | 0 | 400 | 200 | 200 |

`tiny` = any box sqrt(area_frac) < 0.03 relative to the model input (crop frame for crops).
Note how cropping rescues small smoke: 7236 of 38454 full-image examples are tiny, but only 2454 of 19294 crop examples remain tiny in the crop frame.

## Box size histogram (sqrt(area_frac), per box)
| bucket | full-image frame | crop frame |
|---|---|---|
| <0.01 | 73 | 1 |
| 0.01-0.02 | 2685 | 441 |
| 0.02-0.03 | 5030 | 2075 |
| 0.03-0.05 | 8191 | 6349 |
| 0.05-0.10 | 6284 | 9063 |
| 0.10-0.20 | 2748 | 2833 |
| >=0.20 | 6476 | 204 |

## Coordinate convention (exact)
Boxes `[x1,y1,x2,y2]`, top-left origin, x1<x2, y1<y2; each value an INTEGER in 0..1000
(absolute-normalized thousandths of the model-input image): `t = round(1000 * v_norm)`,
clamped to [0,1000], degenerate boxes expanded to >=1 thousandth (0 occurrences).
YOLO source: `x1_norm = cx - w/2` etc., clamped to [0,1]. Crop examples are re-normalized
to the crop window (`crop_box` in pixels on the original image, x2/y2 exclusive; collator
crops on the fly - no crop files are written). Decode: `v_px = t/1000 * input_dim`.
Target string is compact JSON: `{"smoke_present":true,"boxes":[[x1,y1,x2,y2],...]}` or
`{"smoke_present":false,"boxes":[]}`. Answers derive ONLY from audited YOLO labels
(empty label file = verified negative); no fabricated boxes or free-text explanations.

## Instruction phrasings (seeded assignment via md5(id))
1. `Detect wildfire smoke in this image. Reply with compact JSON only: {"smoke_present": true|false, "boxes": [[x1,y1,x2,y2],...]}. Coordinates are integers from 0 to 1000, normalized to the image width and height. If there is no smoke, answer {"smoke_present": false, "boxes": []}.`
2. `Is there smoke in this image? Answer with a single JSON object {"smoke_present": ..., "boxes": ...} where each box is [x1,y1,x2,y2] corner coordinates on a 0-1000 scale relative to the full image; use an empty boxes list when no smoke is visible.`
3. `You are a wildfire early-warning system. Inspect the image and output JSON {"smoke_present": bool, "boxes": list} - each box [x1,y1,x2,y2] with integer coordinates in thousandths (0-1000) of the image dimensions. Output nothing besides the JSON.`
4. `Look for smoke plumes. Respond only with compact JSON of the form {"smoke_present": true|false, "boxes": [[x1,y1,x2,y2], ...]}; coordinates are integers scaled 0-1000 across the image width and height. Report every smoke region; boxes must be [] if smoke_present is false.`

## Verification (round-trip, 50 random positive examples = 56 boxes, seed 0)
- Answer re-derivation from raw labels: EXACT string match for all samples.
- thousandths -> normalized -> thousandths identity: exact.
- Continuous IoU (decoded box vs original pre-quantization box, input frame):
  mean 0.9750, min 0.9352, 18% >= 0.99.
- Pixel-grid IoU (boxes drawn back on the image as integer pixel rects):
  mean 0.9733, min 0.8745, 46% >= 0.99.
- Boxes with min side >= 100 thousandths: 6, 100% IoU >= 0.99 (hard-asserted).
- Every box IoU >= its quantization-error lower bound ((dx-1)(dy-1))/((dx+1)(dy+1)) (hard-asserted)
  -> all deviation is pure 1/1000 rounding; 0.99 for arbitrarily tiny boxes is mathematically
  unattainable at this resolution (e.g. a 15x36-thousandth plume bounds at ~0.92).
- crop_box within actual image bounds and manifest dims == actual dims (PIL) for all samples.
- Contact sheets: `verify/contact_sheet_full.jpg`, `verify/contact_sheet_crops.jpg`
  (green = original label box, red = decoded answer box, yellow-framed = zoom inset).
- Crops where another box was unavoidably cut by the window (clipped + kept if >=2px/>=2% visible): 72 of 19294.

## Provenance / safety rails
- train.jsonl: ONLY `data/yolo/lists/local/expB_train.txt` (frozen site-disjoint train + D-Fire train supplement).
- val_mini.jsonl: derived from frozen VAL list `sdis_val.txt`, ids prefixed `fromval:` - quick training-time eval ONLY, never train on it, never report as final metric.
- Frozen test / FIgLib: untouched.
