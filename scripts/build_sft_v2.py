#!/usr/bin/env python
"""Build multimodal SFT dataset (v2) for Qwen3.5 JSON-output smoke detection.

Outputs under data/v2/sft/:
  train.jsonl      - full-image + local-crop examples from expB_train.txt (shuffled seed 0)
  val_mini.jsonl   - 600 full-image examples from sdis_val.txt (training-time eval ONLY)
  overfit_16.jsonl - 16 diverse train examples (8 pos / 8 neg)
  README.md        - schema, coordinate convention, instruction phrasings
  stats.json       - machine-readable stats (consumed by test_sft_roundtrip.py -> DATASET.md)

Coordinate convention (answer JSON): boxes as [x1,y1,x2,y2], integer thousandths
(0-1000) of the FULL input image the model sees (the crop, for crop examples).
Compact JSON: {"smoke_present":true,"boxes":[[x1,y1,x2,y2],...]} / {"smoke_present":false,"boxes":[]}

No fabricated boxes: answers come only from the audited YOLO label files.
Crops are NOT written to disk; JSONL records crop_box = [x1,y1,x2,y2] pixels
(x2/y2 EXCLUSIVE) on the original image for on-the-fly cropping by the collator.
"""
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict

ROOT = "/local/home/tianrgua/cvproject"
OUT = os.path.join(ROOT, "data/v2/sft")
TRAIN_LIST = os.path.join(ROOT, "data/yolo/lists/local/expB_train.txt")
VAL_LIST = os.path.join(ROOT, "data/yolo/lists/local/sdis_val.txt")
CROP = 640
CAP = 70000
TINY_SQRT = 0.03          # "tiny" flag: any box sqrt(area_frac) < 0.03 (relative to model input)
CROP_AREA_FRAC = 0.01     # positives with any box area_frac < 0.01 get one local crop
NEG_CROP_FRAC = 0.20      # fraction of negatives that get one random crop
SEED = 0

INSTRUCTIONS = [
    ('Detect wildfire smoke in this image. Reply with compact JSON only: '
     '{"smoke_present": true|false, "boxes": [[x1,y1,x2,y2],...]}. Coordinates are '
     'integers from 0 to 1000, normalized to the image width and height. If there is '
     'no smoke, answer {"smoke_present": false, "boxes": []}.'),
    ('Is there smoke in this image? Answer with a single JSON object '
     '{"smoke_present": ..., "boxes": ...} where each box is [x1,y1,x2,y2] corner '
     'coordinates on a 0-1000 scale relative to the full image; use an empty boxes '
     'list when no smoke is visible.'),
    ('You are a wildfire early-warning system. Inspect the image and output JSON '
     '{"smoke_present": bool, "boxes": list} - each box [x1,y1,x2,y2] with integer '
     'coordinates in thousandths (0-1000) of the image dimensions. Output nothing '
     'besides the JSON.'),
    ('Look for smoke plumes. Respond only with compact JSON of the form '
     '{"smoke_present": true|false, "boxes": [[x1,y1,x2,y2], ...]}; coordinates are '
     'integers scaled 0-1000 across the image width and height. Report every smoke '
     'region; boxes must be [] if smoke_present is false.'),
]


def stable_int(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest(), 16)


def pick_instruction(ex_id: str) -> str:
    return INSTRUCTIONS[stable_int("instr0:" + ex_id) % len(INSTRUCTIONS)]


def label_path(img: str) -> str:
    return img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"


def read_yolo(img: str):
    """Return list of (x1,y1,x2,y2) normalized floats, clamped to [0,1]."""
    boxes = []
    with open(label_path(img)) as f:
        for ln in f:
            t = ln.split()
            if not t:
                continue
            assert t[0] == "0", f"unexpected class {t[0]} in {img}"
            cx, cy, w, h = map(float, t[1:5])
            x1 = min(max(cx - w / 2, 0.0), 1.0)
            y1 = min(max(cy - h / 2, 0.0), 1.0)
            x2 = min(max(cx + w / 2, 0.0), 1.0)
            y2 = min(max(cy + h / 2, 0.0), 1.0)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


DEGENERATE_FIXES = 0


def to_thousandths(b):
    """(x1,y1,x2,y2) in [0,1] floats -> integer thousandths, non-degenerate."""
    global DEGENERATE_FIXES
    q = [min(max(int(round(v * 1000)), 0), 1000) for v in b]
    for lo, hi in ((0, 2), (1, 3)):
        if q[hi] <= q[lo]:
            DEGENERATE_FIXES += 1
            if q[hi] < 1000:
                q[hi] = q[lo] + 1
            else:
                q[lo] = q[hi] - 1
    return q


def answer_json(boxes_t):
    if boxes_t:
        return json.dumps({"smoke_present": True, "boxes": boxes_t}, separators=(",", ":"))
    return json.dumps({"smoke_present": False, "boxes": []}, separators=(",", ":"))


def tiny_flag(boxes_norm):
    return any(math.sqrt((x2 - x1) * (y2 - y1)) < TINY_SQRT for x1, y1, x2, y2 in boxes_norm)


def load_dims():
    import pandas as pd
    pm = pd.read_parquet(os.path.join(ROOT, "data/manifests/pyro_sdis_manifest.parquet"))
    dm = pd.read_parquet(os.path.join(ROOT, "data/manifests/dfire_manifest.parquet"))
    dims = {r.image_path: (int(r.width), int(r.height)) for r in pm.itertuples()}
    for r in dm.itertuples():
        dims["dfire_bn:" + os.path.basename(r.image_path)] = (int(r.width), int(r.height))
    return dims


def img_dims(dims, img):
    if "pyro_sdis" in img:
        return dims[img]
    return dims["dfire_bn:" + os.path.basename(img)]


def src_of(img):
    return "pyro-sdis" if "pyro_sdis" in img else "dfire"


def base_id(img):
    return f"{src_of(img)}:{os.path.splitext(os.path.basename(img))[0]}"


def make_example(ex_id, img, crop_box, boxes_t, boxes_norm_input, source):
    return {
        "id": ex_id,
        "image_path": img,
        "crop_box": crop_box,
        "instruction": pick_instruction(ex_id),
        "answer": answer_json(boxes_t),
        "n_boxes": len(boxes_t),
        "source": source,
        "tiny": tiny_flag(boxes_norm_input) if boxes_norm_input else False,
    }


def full_example(img, dims, id_prefix=""):
    boxes = read_yolo(img)
    boxes_t = [to_thousandths(b) for b in boxes]
    ex_id = f"{id_prefix}{base_id(img)}:full"
    return make_example(ex_id, img, None, boxes_t, boxes, src_of(img)), boxes


def crop_window_for_box(rng, W, H, box_px):
    """Random 640x640 window (clipped at borders) fully containing box_px.
    Box lands at a RANDOM position inside the window (uniform over feasible
    offsets -> no center bias). Returns (x1,y1,x2,y2) ints, x2/y2 exclusive."""
    bx1, by1, bx2, by2 = box_px

    def axis(lo_b, hi_b, size):
        c = min(CROP, size)
        lo = max(0, int(math.ceil(hi_b)) - c)
        hi = min(size - c, int(math.floor(lo_b)))
        hi = max(hi, lo)
        o = rng.randint(lo, hi)
        return o, o + c

    x1, x2 = axis(bx1, bx2, W)
    y1, y2 = axis(by1, by2, H)
    return x1, y1, x2, y2


def boxes_in_crop(boxes_px, win, min_px=2.0, min_vis=0.02):
    """Clip pixel boxes to window; return (kept_boxes_crop_norm, n_partial).
    A box is 'partial' if 0 < visible_fraction < 0.999. Kept if visible and the
    clipped part is at least min_px on each side and min_vis of original area."""
    cx1, cy1, cx2, cy2 = win
    cw, ch = cx2 - cx1, cy2 - cy1
    kept, n_partial = [], 0
    for bx1, by1, bx2, by2 in boxes_px:
        ix1, iy1 = max(bx1, cx1), max(by1, cy1)
        ix2, iy2 = min(bx2, cx2), min(by2, cy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        vis = ((ix2 - ix1) * (iy2 - iy1)) / max((bx2 - bx1) * (by2 - by1), 1e-9)
        if vis < 0.999:
            n_partial += 1
            if vis < min_vis or (ix2 - ix1) < min_px or (iy2 - iy1) < min_px:
                continue
        kept.append(((ix1 - cx1) / cw, (iy1 - cy1) / ch, (ix2 - cx1) / cw, (iy2 - cy1) / ch))
    return kept, n_partial


def pos_crop_example(img, dims, boxes_norm, stats):
    W, H = img_dims(dims, img)
    small = [b for b in boxes_norm if (b[2] - b[0]) * (b[3] - b[1]) < CROP_AREA_FRAC]
    if not small:
        return None
    rng = random.Random(stable_int(f"crop0:{base_id(img)}") % (2 ** 32))
    anchor = rng.choice(small)
    boxes_px = [(x1 * W, y1 * H, x2 * W, y2 * H) for x1, y1, x2, y2 in boxes_norm]
    anchor_px = (anchor[0] * W, anchor[1] * H, anchor[2] * W, anchor[3] * H)
    # try up to 20 placements avoiding partial cuts of OTHER boxes
    win = kept = None
    for attempt in range(20):
        w = crop_window_for_box(rng, W, H, anchor_px)
        k, n_part = boxes_in_crop(boxes_px, w)
        win, kept = w, k
        if n_part == 0:
            break
    else:
        stats["crops_with_partial_boxes"] += 1
    boxes_t = [to_thousandths(b) for b in kept]
    ex_id = f"{base_id(img)}:cropP"
    ex = make_example(ex_id, img, list(win), boxes_t, kept, src_of(img))
    return ex


def neg_crop_example(img, dims):
    W, H = img_dims(dims, img)
    rng = random.Random(stable_int(f"cropN0:{base_id(img)}") % (2 ** 32))
    cw, ch = min(CROP, W), min(CROP, H)
    x1 = rng.randint(0, W - cw)
    y1 = rng.randint(0, H - ch)
    ex_id = f"{base_id(img)}:cropN"
    return make_example(ex_id, img, [x1, y1, x1 + cw, y1 + ch], [], [], src_of(img))


def build_train(dims, stats):
    imgs = [l.strip() for l in open(TRAIN_LIST) if l.strip()]
    full_ex, pos_crop_ex, neg_imgs, boxes_by_img = [], [], [], {}
    for img in imgs:
        ex, boxes = full_example(img, dims)
        full_ex.append(ex)
        boxes_by_img[img] = boxes
        if not boxes:
            neg_imgs.append(img)
    for img in imgs:
        boxes = boxes_by_img[img]
        if boxes:
            ex = pos_crop_example(img, dims, boxes, stats)
            if ex is not None:
                pos_crop_ex.append(ex)
    rng = random.Random(SEED)
    neg_sample = sorted(rng.sample(range(len(neg_imgs)), int(round(NEG_CROP_FRAC * len(neg_imgs)))))
    neg_crop_ex = [neg_crop_example(neg_imgs[i], dims) for i in neg_sample]

    examples = full_ex + pos_crop_ex + neg_crop_ex
    dropped = []
    if len(examples) > CAP:
        overshoot = len(examples) - CAP
        full_neg = [e for e in full_ex if e["n_boxes"] == 0]
        rng2 = random.Random(SEED + 1)
        drop_ids = set(e["id"] for e in rng2.sample(full_neg, min(overshoot, len(full_neg))))
        dropped = sorted(drop_ids)
        examples = [e for e in examples if e["id"] not in drop_ids]
    random.Random(SEED).shuffle(examples)
    return examples, full_ex, pos_crop_ex, neg_crop_ex, dropped, boxes_by_img


def build_val_mini(dims):
    imgs = [l.strip() for l in open(VAL_LIST) if l.strip()]
    tiny_pos, other_pos, neg = [], [], []
    for img in imgs:
        ex, boxes = full_example(img, dims, id_prefix="fromval:")
        if not boxes:
            neg.append(ex)
        elif ex["tiny"]:
            tiny_pos.append(ex)
        else:
            other_pos.append(ex)
    rng = random.Random(SEED)
    rng.shuffle(tiny_pos), rng.shuffle(other_pos), rng.shuffle(neg)
    sel = tiny_pos[:200] + other_pos[:200] + neg[:200]
    rng.shuffle(sel)
    assert len(sel) == 600
    return sel, len(tiny_pos[:200])


def build_overfit16(train_examples):
    by = defaultdict(list)
    for e in train_examples:  # already shuffled seed 0 -> deterministic pick order
        pos = e["n_boxes"] > 0
        kind = "full" if e["crop_box"] is None else "crop"
        by[(e["source"], kind, pos, e["tiny"])].append(e)
        by[(e["source"], kind, pos)].append(e)
    want_pos = [("pyro-sdis", "full", True, True), ("pyro-sdis", "full", True, True),
                ("pyro-sdis", "full", True, False), ("pyro-sdis", "crop", True),
                ("dfire", "full", True, True), ("dfire", "full", True, False),
                ("dfire", "full", True), ("dfire", "crop", True)]
    want_neg = [("pyro-sdis", "full", False), ("pyro-sdis", "full", False),
                ("pyro-sdis", "crop", False), ("pyro-sdis", "crop", False),
                ("dfire", "full", False), ("dfire", "full", False),
                ("dfire", "crop", False), ("dfire", "crop", False)]
    chosen, used = [], set()
    # prefer one multi-box positive for diversity
    multi = next((e for e in train_examples if e["n_boxes"] >= 2 and e["crop_box"] is None), None)
    if multi:
        chosen.append(multi), used.add(multi["id"])
    for want in (want_pos[1:] if multi else want_pos) + want_neg:
        pool = [e for e in by.get(want, []) if e["id"] not in used]
        if not pool:  # fallback: same source+polarity, any kind
            pool = [e for e in by.get((want[0], "full", want[2]), []) if e["id"] not in used]
        chosen.append(pool[0]), used.add(pool[0]["id"])
    assert len(chosen) == 16
    assert sum(e["n_boxes"] > 0 for e in chosen) == 8
    return chosen


def size_hist(boxes_norm_lists):
    edges = [0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 1.01]
    names = ["<0.01", "0.01-0.02", "0.02-0.03", "0.03-0.05", "0.05-0.10", "0.10-0.20", ">=0.20"]
    hist = Counter()
    for boxes in boxes_norm_lists:
        for x1, y1, x2, y2 in boxes:
            s = math.sqrt((x2 - x1) * (y2 - y1))
            for i in range(len(names)):
                if edges[i] <= s < edges[i + 1]:
                    hist[names[i]] += 1
                    break
    return {n: hist.get(n, 0) for n in names}


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "verify"), exist_ok=True)
    dims = load_dims()
    stats = Counter()

    train, full_ex, pos_crop_ex, neg_crop_ex, dropped, boxes_by_img = build_train(dims, stats)
    val_mini, n_val_tiny = build_val_mini(dims)
    overfit = build_overfit16(train)

    write_jsonl(os.path.join(OUT, "train.jsonl"), train)
    write_jsonl(os.path.join(OUT, "val_mini.jsonl"), val_mini)
    write_jsonl(os.path.join(OUT, "overfit_16.jsonl"), overfit)

    def cnt(exs):
        c = Counter()
        for e in exs:
            pos = e["n_boxes"] > 0
            c[f"{e['source']}|{'pos' if pos else 'neg'}"] += 1
            c["pos" if pos else "neg"] += 1
            c[e["source"]] += 1
            if e["tiny"]:
                c["tiny"] += 1
        return dict(c)

    # crop-frame box sizes (normalized to crop)
    crop_boxes_norm = []
    for e in pos_crop_ex:
        bx = json.loads(e["answer"])["boxes"]
        crop_boxes_norm.append([(x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000) for x1, y1, x2, y2 in bx])

    report = {
        "seed": SEED,
        "cap": CAP,
        "counts": {
            "train_total": len(train),
            "train_full": len(full_ex) - len(dropped),
            "train_pos_crops": len(pos_crop_ex),
            "train_neg_crops": len(neg_crop_ex),
            "dropped_full_negatives": len(dropped),
            "val_mini": len(val_mini),
            "val_mini_tiny_pos": n_val_tiny,
            "overfit_16": len(overfit),
        },
        "train_breakdown": cnt(train),
        "train_full_breakdown": cnt(full_ex),
        "train_pos_crop_breakdown": cnt(pos_crop_ex),
        "train_neg_crop_breakdown": cnt(neg_crop_ex),
        "val_mini_breakdown": cnt(val_mini),
        "size_hist_full_sqrt_area_frac": size_hist(boxes_by_img.values()),
        "size_hist_crop_sqrt_area_frac": size_hist(crop_boxes_norm),
        "crops_with_partial_boxes": stats["crops_with_partial_boxes"],
        "degenerate_rounding_fixes": DEGENERATE_FIXES,
        "instructions": INSTRUCTIONS,
        "dropped_ids": dropped,
    }
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write(README_TMPL.format(instr="\n".join(f"{i+1}. `{s}`" for i, s in enumerate(INSTRUCTIONS))))

    print(json.dumps({k: v for k, v in report.items() if k not in ("instructions", "dropped_ids")}, indent=2))


README_TMPL = """# SFT dataset v2 - Qwen3.5 smoke detection (JSON output)

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
  `{{"smoke_present":true,"boxes":[[x1,y1,x2,y2],...]}}` or `{{"smoke_present":false,"boxes":[]}}`
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
`{{"smoke_present":false,"boxes":[]}}`. No fabricated boxes or explanations anywhere.

## Local crops (small-smoke preservation)
- For every train positive with any box `area_frac = w*h < 0.01`: ONE 640x640 crop
  (clipped at image borders, so <=640 on small images). A random qualifying box is the
  anchor; the window is placed UNIFORMLY at random among all positions that fully contain
  the anchor (no center bias). Up to 20 placements are tried to avoid partially cutting
  other boxes; if unavoidable, cut boxes are clipped to the window and kept when >=2px on
  each side and >=2% of their area is visible (occurrences counted in stats.json).
- For a seeded random 20% of train negatives: ONE random 640x640 crop, empty boxes.

## Instruction phrasings (fixed list)
{instr}
"""

if __name__ == "__main__":
    main()
