#!/usr/bin/env python
"""Round-trip verification for data/v2/sft (see build_sft_v2.py).

For 50 seeded-random positive train examples:
  YOLO label -> [builder pipeline] -> answer JSON (integer thousandths)
  -> decode back into the model-input frame -> IoU vs the original (pre-quantization) box.

Checks:
  A. box counts match n_boxes and the answer JSON re-derived from labels is IDENTICAL
     to the stored answer (exact string equality) -> conversion is deterministic.
  B. decode(encode(x)) == encode-space identity: thousandths -> norm -> thousandths exact.
  C. per-box continuous IoU >= quantization-error lower bound
     ((dx-1)(dy-1))/((dx+1)(dy+1)) with dx,dy = box size in thousandths -> any deviation
     beyond rounding would fail.
  D. per-box IoU >= 0.99 for boxes with min side >= 100 thousandths (0.99 is
     mathematically unattainable at 1/1000 quantization for very small boxes; the
     pass-rate for all sizes is reported).
  E. crop_box lies within the actual image (PIL-verified), image dims match manifest.

Also writes 2 contact sheets (full-image + crop examples; green = original label box,
red = decoded answer box) to data/v2/sft/verify/, writes verify/roundtrip_results.json,
and generates data/v2/sft/DATASET.md from stats.json + results.

Run: python scripts/test_sft_roundtrip.py   (also pytest-compatible)
"""
import json
import math
import os
import random
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sft_v2 import (OUT, boxes_in_crop, img_dims, load_dims, read_yolo,
                          to_thousandths, answer_json)

from PIL import Image, ImageDraw

VERIFY = os.path.join(OUT, "verify")
N_SAMPLES = 50
SEED = 0


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def orig_boxes_input_frame(ex, dims):
    """Original label boxes mapped into the model-input frame, normalized [0,1] floats.
    Reproduces the builder's crop filtering exactly (deterministic given crop_box)."""
    boxes = read_yolo(ex["image_path"])
    if ex["crop_box"] is None:
        return boxes
    W, H = img_dims(dims, ex["image_path"])
    boxes_px = [(x1 * W, y1 * H, x2 * W, y2 * H) for x1, y1, x2, y2 in boxes]
    kept, _ = boxes_in_crop(boxes_px, tuple(ex["crop_box"]))
    return kept


def input_frame_size(ex, dims):
    if ex["crop_box"] is None:
        return img_dims(dims, ex["image_path"])
    x1, y1, x2, y2 = ex["crop_box"]
    return x2 - x1, y2 - y1


def test_roundtrip():
    dims = load_dims()
    train = [json.loads(l) for l in open(os.path.join(OUT, "train.jsonl"))]
    pos = [e for e in train if e["n_boxes"] > 0]
    rng = random.Random(SEED)
    sample = rng.sample(pos, N_SAMPLES)

    ious, ious_px, per_sample = [], [], []
    n_ge99 = n_boxes = n_big = n_big_ge99 = 0
    for ex in sample:
        orig = orig_boxes_input_frame(ex, dims)
        ans = json.loads(ex["answer"])
        assert ans["smoke_present"] is True
        dec_t = ans["boxes"]
        # A: exact re-derivation of the stored answer from the raw labels
        assert answer_json([to_thousandths(b) for b in orig]) == ex["answer"], ex["id"]
        assert len(dec_t) == len(orig) == ex["n_boxes"], ex["id"]
        # B: thousandths -> normalized -> thousandths identity
        for q in dec_t:
            q2 = [min(max(int(round(v / 1000 * 1000)), 0), 1000) for v in q]
            assert q2 == q, ex["id"]
        # E: crop_box within actual image; dims match manifest
        with Image.open(ex["image_path"]) as im:
            aw, ah = im.size
        W, H = img_dims(dims, ex["image_path"])
        assert (aw, ah) == (W, H), f"{ex['id']}: manifest {W}x{H} vs actual {aw}x{ah}"
        if ex["crop_box"] is not None:
            cx1, cy1, cx2, cy2 = ex["crop_box"]
            assert 0 <= cx1 < cx2 <= aw and 0 <= cy1 < cy2 <= ah, ex["id"]
        iw, ih = input_frame_size(ex, dims)
        s_min = 1.0
        for o, q in zip(orig, dec_t):  # answer preserves label order -> pair by index
            d = [v / 1000 for v in q]
            v = iou(o, d)
            # pixel-grid IoU: both boxes drawn as integer pixel rects on the input
            opx = [round(o[0] * iw), round(o[1] * ih), round(o[2] * iw), round(o[3] * ih)]
            dpx = [round(d[0] * iw), round(d[1] * ih), round(d[2] * iw), round(d[3] * ih)]
            vpx = iou(opx, dpx)
            ious.append(v)
            ious_px.append(vpx)
            s_min = min(s_min, v)
            n_boxes += 1
            if v >= 0.99:
                n_ge99 += 1
            # C: quantization lower bound
            dx = (o[2] - o[0]) * 1000
            dy = (o[3] - o[1]) * 1000
            bound = max(0.0, (dx - 1.0) * (dy - 1.0)) / ((dx + 1.0) * (dy + 1.0))
            assert v >= bound - 1e-9, f"{ex['id']}: IoU {v:.4f} < quantization bound {bound:.4f}"
            # D: hard 0.99 gate where quantization permits it
            if min(dx, dy) >= 100:
                n_big += 1
                if v >= 0.99:
                    n_big_ge99 += 1
                assert v >= 0.99, f"{ex['id']}: large box IoU {v:.4f} < 0.99"
        per_sample.append({"id": ex["id"], "min_iou": round(s_min, 4)})

    res = {
        "n_samples": N_SAMPLES,
        "n_boxes": n_boxes,
        "iou_continuous": {"mean": sum(ious) / len(ious), "min": min(ious),
                           "frac_ge_0.99": n_ge99 / n_boxes},
        "iou_pixel_grid": {"mean": sum(ious_px) / len(ious_px), "min": min(ious_px),
                           "frac_ge_0.99": sum(v >= 0.99 for v in ious_px) / n_boxes},
        "boxes_min_side_ge_100thou": {"n": n_big, "frac_ge_0.99": (n_big_ge99 / n_big) if n_big else None},
        "checks": {
            "answer_rederivation_exact": True,
            "thousandths_identity_exact": True,
            "iou_ge_quantization_bound_all": True,
            "large_box_iou_ge_0.99_all": True,
            "crop_box_in_bounds_and_dims_match": True,
        },
        "note": ("IoU >= 0.99 for EVERY box is mathematically impossible at 1/1000 "
                 "quantization for tiny boxes: worst-case bound is "
                 "((dx-1)(dy-1))/((dx+1)(dy+1)) for a box of dx x dy thousandths. "
                 "All observed IoUs meet that bound, i.e. error is pure rounding."),
        "per_sample_min_iou": per_sample,
    }
    os.makedirs(VERIFY, exist_ok=True)
    with open(os.path.join(VERIFY, "roundtrip_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


# ---------------- contact sheets ----------------

def draw_tile(ex, dims, tile_w=480, tile_h=330, cap_h=26):
    with Image.open(ex["image_path"]) as im:
        im = im.convert("RGB")
        if ex["crop_box"] is not None:
            cx1, cy1, cx2, cy2 = ex["crop_box"]
            im = im.crop((cx1, cy1, cx2, cy2))
        iw, ih = im.size
        sc = min(tile_w / iw, (tile_h - cap_h) / ih)
        im = im.resize((max(1, int(iw * sc)), max(1, int(ih * sc))))
    tile = Image.new("RGB", (tile_w, tile_h), (24, 24, 24))
    tile.paste(im, ((tile_w - im.width) // 2, 0))
    ox, oy = (tile_w - im.width) // 2, 0
    d = ImageDraw.Draw(tile)

    orig = orig_boxes_input_frame(ex, dims)
    dec = [[v / 1000 for v in b] for b in json.loads(ex["answer"])["boxes"]]

    def rect(b, color, w):
        d.rectangle([ox + b[0] * im.width, oy + b[1] * im.height,
                     ox + b[2] * im.width, oy + b[3] * im.height], outline=color, width=w)

    for b in orig:
        rect(b, (0, 255, 0), 3)
    for b in dec:
        rect(b, (255, 40, 40), 1)
    # zoom inset for the first box if it is small on the tile
    if dec:
        b = dec[0]
        if (b[2] - b[0]) * im.width < 60:
            zx1 = max(0, b[0] * im.width - 2.5 * (b[2] - b[0]) * im.width - 4)
            zy1 = max(0, b[1] * im.height - 2.5 * (b[3] - b[1]) * im.height - 4)
            zx2 = min(im.width, b[2] * im.width + 2.5 * (b[2] - b[0]) * im.width + 4)
            zy2 = min(im.height, b[3] * im.height + 2.5 * (b[3] - b[1]) * im.height + 4)
            if zx2 - zx1 > 2 and zy2 - zy1 > 2:
                z = tile.crop((int(ox + zx1), int(oy + zy1), int(ox + zx2), int(oy + zy2)))
                zs = 128
                z = z.resize((zs, int(zs * z.height / max(z.width, 1))) if z.width >= z.height
                             else (int(zs * z.width / max(z.height, 1)), zs))
                tile.paste(z, (tile_w - z.width - 4, 4))
                d.rectangle([tile_w - z.width - 5, 3, tile_w - 4, 4 + z.height], outline=(255, 255, 0), width=1)
    cap = f"{ex['id'][:56]}  n={ex['n_boxes']}{' TINY' if ex['tiny'] else ''}{' NEG' if ex['n_boxes'] == 0 else ''}"
    d.text((4, tile_h - cap_h + 6), cap, fill=(230, 230, 230))
    return tile


def contact_sheets(dims):
    train = [json.loads(l) for l in open(os.path.join(OUT, "train.jsonl"))]
    rng = random.Random(SEED + 7)
    full_pos_tiny = [e for e in train if e["crop_box"] is None and e["n_boxes"] > 0 and e["tiny"]]
    full_pos = [e for e in train if e["crop_box"] is None and e["n_boxes"] > 0 and not e["tiny"]]
    full_neg = [e for e in train if e["crop_box"] is None and e["n_boxes"] == 0]
    crop_pos = [e for e in train if e["crop_box"] is not None and e["n_boxes"] > 0]
    crop_neg = [e for e in train if e["crop_box"] is not None and e["n_boxes"] == 0]
    sheets = {
        "contact_sheet_full.jpg": rng.sample(full_pos_tiny, 10) + rng.sample(full_pos, 3) + rng.sample(full_neg, 3),
        "contact_sheet_crops.jpg": rng.sample(crop_pos, 12) + rng.sample(crop_neg, 4),
    }
    for name, exs in sheets.items():
        tw, th, cols = 480, 330, 4
        rows = math.ceil(len(exs) / cols)
        sheet = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
        for i, ex in enumerate(exs):
            sheet.paste(draw_tile(ex, dims, tw, th), ((i % cols) * tw, (i // cols) * th))
        p = os.path.join(VERIFY, name)
        sheet.save(p, quality=90)
        print("wrote", p)


# ---------------- DATASET.md ----------------

def write_dataset_md(res):
    st = json.load(open(os.path.join(OUT, "stats.json")))
    c = st["counts"]

    def bd(d):
        keys = ["pyro-sdis|pos", "pyro-sdis|neg", "dfire|pos", "dfire|neg", "pos", "neg", "tiny"]
        return " | ".join(str(d.get(k, 0)) for k in keys)

    md = f"""# DATASET.md - SFT v2 (Qwen3.5 smoke detection, JSON output)

Generated {date.today().isoformat()} by `scripts/build_sft_v2.py` + `scripts/test_sft_roundtrip.py` (seed {st['seed']}).
Schema and conventions: `README.md`. Machine-readable stats: `stats.json`.

## Counts
| file | examples |
|---|---|
| train.jsonl | {c['train_total']} |
| val_mini.jsonl | {c['val_mini']} (400 pos incl. {c['val_mini_tiny_pos']} tiny, 200 neg; full-image; FROM FROZEN VAL - training-time eval only) |
| overfit_16.jsonl | {c['overfit_16']} (8 pos / 8 neg, both sources, full+crop, incl. tiny + multi-box) |

train.jsonl composition: {c['train_full']} full-image + {c['train_pos_crops']} positive local crops (any box area_frac<0.01) + {c['train_neg_crops']} negative random crops (20% of negatives).
Cap {st['cap']}: total {c['train_total']} <= cap -> **{c['dropped_full_negatives']} full-image negatives dropped** (dropped ids recorded in stats.json).

Breakdowns (pyro pos | pyro neg | dfire pos | dfire neg | pos | neg | tiny):
| subset | counts |
|---|---|
| train (all) | {bd(st['train_breakdown'])} |
| train full-image | {bd(st['train_full_breakdown'])} |
| train pos crops | {bd(st['train_pos_crop_breakdown'])} |
| train neg crops | {bd(st['train_neg_crop_breakdown'])} |
| val_mini | {bd(st['val_mini_breakdown'])} |

`tiny` = any box sqrt(area_frac) < 0.03 relative to the model input (crop frame for crops).
Note how cropping rescues small smoke: {st['train_full_breakdown'].get('tiny',0)} of {c['train_full']} full-image examples are tiny, but only {st['train_pos_crop_breakdown'].get('tiny',0)} of {c['train_pos_crops']} crop examples remain tiny in the crop frame.

## Box size histogram (sqrt(area_frac), per box)
| bucket | full-image frame | crop frame |
|---|---|---|
""" + "\n".join(
        f"| {k} | {st['size_hist_full_sqrt_area_frac'][k]} | {st['size_hist_crop_sqrt_area_frac'][k]} |"
        for k in st["size_hist_full_sqrt_area_frac"]) + f"""

## Coordinate convention (exact)
Boxes `[x1,y1,x2,y2]`, top-left origin, x1<x2, y1<y2; each value an INTEGER in 0..1000
(absolute-normalized thousandths of the model-input image): `t = round(1000 * v_norm)`,
clamped to [0,1000], degenerate boxes expanded to >=1 thousandth ({st['degenerate_rounding_fixes']} occurrences).
YOLO source: `x1_norm = cx - w/2` etc., clamped to [0,1]. Crop examples are re-normalized
to the crop window (`crop_box` in pixels on the original image, x2/y2 exclusive; collator
crops on the fly - no crop files are written). Decode: `v_px = t/1000 * input_dim`.
Target string is compact JSON: `{{"smoke_present":true,"boxes":[[x1,y1,x2,y2],...]}}` or
`{{"smoke_present":false,"boxes":[]}}`. Answers derive ONLY from audited YOLO labels
(empty label file = verified negative); no fabricated boxes or free-text explanations.

## Instruction phrasings (seeded assignment via md5(id))
""" + "\n".join(f"{i+1}. `{s}`" for i, s in enumerate(st["instructions"])) + f"""

## Verification (round-trip, {res['n_samples']} random positive examples = {res['n_boxes']} boxes, seed {SEED})
- Answer re-derivation from raw labels: EXACT string match for all samples.
- thousandths -> normalized -> thousandths identity: exact.
- Continuous IoU (decoded box vs original pre-quantization box, input frame):
  mean {res['iou_continuous']['mean']:.4f}, min {res['iou_continuous']['min']:.4f}, {100*res['iou_continuous']['frac_ge_0.99']:.0f}% >= 0.99.
- Pixel-grid IoU (boxes drawn back on the image as integer pixel rects):
  mean {res['iou_pixel_grid']['mean']:.4f}, min {res['iou_pixel_grid']['min']:.4f}, {100*res['iou_pixel_grid']['frac_ge_0.99']:.0f}% >= 0.99.
- Boxes with min side >= 100 thousandths: {res['boxes_min_side_ge_100thou']['n']}, 100% IoU >= 0.99 (hard-asserted).
- Every box IoU >= its quantization-error lower bound ((dx-1)(dy-1))/((dx+1)(dy+1)) (hard-asserted)
  -> all deviation is pure 1/1000 rounding; 0.99 for arbitrarily tiny boxes is mathematically
  unattainable at this resolution (e.g. a 15x36-thousandth plume bounds at ~0.92).
- crop_box within actual image bounds and manifest dims == actual dims (PIL) for all samples.
- Contact sheets: `verify/contact_sheet_full.jpg`, `verify/contact_sheet_crops.jpg`
  (green = original label box, red = decoded answer box, yellow-framed = zoom inset).
- Crops where another box was unavoidably cut by the window (clipped + kept if >=2px/>=2% visible): {st['crops_with_partial_boxes']} of {c['train_pos_crops']}.

## Provenance / safety rails
- train.jsonl: ONLY `data/yolo/lists/local/expB_train.txt` (frozen site-disjoint train + D-Fire train supplement).
- val_mini.jsonl: derived from frozen VAL list `sdis_val.txt`, ids prefixed `fromval:` - quick training-time eval ONLY, never train on it, never report as final metric.
- Frozen test / FIgLib: untouched.
"""
    with open(os.path.join(OUT, "DATASET.md"), "w") as f:
        f.write(md)
    print("wrote", os.path.join(OUT, "DATASET.md"))


if __name__ == "__main__":
    res = test_roundtrip()
    print(json.dumps({k: v for k, v in res.items() if k != "per_sample_min_iou"}, indent=2))
    dims = load_dims()
    contact_sheets(dims)
    write_dataset_md(res)
    print("ALL CHECKS PASSED")
