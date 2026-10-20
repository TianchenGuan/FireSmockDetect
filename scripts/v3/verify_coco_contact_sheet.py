"""Draw-back verification of the v3 COCO conversion.

Deterministic 20-sample draw (seed 0) per split: 14 images WITH boxes +
6 negatives. Draws the COCO xywh boxes back onto the images and tiles a
contact sheet (reports/v3/coco_verify_{train,val}.jpg) for human inspection.
Also cross-checks every drawn box against the original YOLO label line
(cxcywh-norm -> xyxy) with an exact-arithmetic tolerance.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/local/home/tianrgua/cvproject")
DATA_ROOT = ROOT / "data" / "yolo"
OUT = ROOT / "reports" / "v3"

TILE_W, TILE_H = 480, 300
COLS = 5


def yolo_boxes(img_path: Path, w: int, h: int):
    lp = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    out = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                _, cx, cy, bw, bh = (float(x) for x in p)
                x1 = max(0.0, (cx - bw / 2) * w)
                y1 = max(0.0, (cy - bh / 2) * h)
                x2 = min(float(w), (cx + bw / 2) * w)
                y2 = min(float(h), (cy + bh / 2) * h)
                if x2 - x1 >= 1 and y2 - y1 >= 1:
                    out.append((x1, y1, x2, y2))
    return out


def verify(split: str) -> None:
    coco = json.loads((ROOT / "data" / "v3" / "coco" / f"instances_{split}.json").read_text())
    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a["bbox"])
    imgs = coco["images"]
    pos = [im for im in imgs if im["id"] in anns_by_img]
    neg = [im for im in imgs if im["id"] not in anns_by_img]
    rng = random.Random(0)
    sample = rng.sample(pos, 14) + rng.sample(neg, 6)

    # cross-check against original YOLO labels
    mismatches = 0
    for im in sample:
        p = DATA_ROOT / im["file_name"]
        ref = yolo_boxes(p, im["width"], im["height"])
        got = [(b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in anns_by_img.get(im["id"], [])]
        if len(ref) != len(got):
            mismatches += 1
            print(f"COUNT MISMATCH {im['file_name']}: yolo={len(ref)} coco={len(got)}")
            continue
        for r, g in zip(sorted(ref), sorted(got)):
            if any(abs(a - b) > 0.51 for a, b in zip(r, g)):  # 0.01 rounding + clamp
                mismatches += 1
                print(f"BOX MISMATCH {im['file_name']}: {r} vs {g}")
                break

    rows = (len(sample) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TILE_W, rows * TILE_H), (20, 20, 20))
    font = ImageFont.load_default()
    for i, im in enumerate(sample):
        p = DATA_ROOT / im["file_name"]
        img = Image.open(p).convert("RGB")
        assert img.size == (im["width"], im["height"]), f"size mismatch {p}"
        d = ImageDraw.Draw(img)
        for b in anns_by_img.get(im["id"], []):
            x1, y1, x2, y2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
            d.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=max(2, im["width"] // 400))
        tag = f"id={im['id']} n={len(anns_by_img.get(im['id'], []))} {Path(im['file_name']).name[:40]}"
        img = img.resize((TILE_W, TILE_H))
        d2 = ImageDraw.Draw(img)
        d2.rectangle([0, 0, TILE_W, 14], fill=(0, 0, 0))
        d2.text((2, 1), tag, fill=(255, 255, 0), font=font)
        sheet.paste(img, ((i % COLS) * TILE_W, (i // COLS) * TILE_H))
    out = OUT / f"coco_verify_{split}.jpg"
    sheet.save(out, quality=88)
    print(f"{split}: sheet -> {out}; box cross-check mismatches: {mismatches} "
          f"(sample 14 pos + 6 neg of {len(pos)} pos/{len(neg)} neg)")
    assert mismatches == 0, "conversion mismatch — DO NOT TRAIN"


for s in ("train", "val"):
    verify(s)
print("VERIFY_OK")
