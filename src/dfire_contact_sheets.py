#!/usr/bin/env python3
"""Track C: contact sheets for D-Fire — smoke positives, fire-only, negatives.
Draws GT boxes (smoke=green, fire=red) on thumbnails, 8x8 grid per sheet.
"""
import json
import os

import pandas as pd
from PIL import Image, ImageDraw

ROOT = "/local/home/tianrgua/cvproject"
OUT = os.path.join(ROOT, "data/contact_sheets/dfire")
THUMB = 256
GRID = 8
SEED = 42

os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet(os.path.join(ROOT, "data/manifests/dfire_manifest.parquet"))

def cat(cj):
    s = set(json.loads(cj))
    if not s: return "none"
    if s == {1}: return "fire_only"
    return "smoke_pos"  # any image containing smoke boxes

df["cat3"] = df["class_ids"].map(cat)

def tile(rec):
    im = Image.open(os.path.join(ROOT, rec["image_path"])).convert("RGB")
    w, h = im.size
    scale = THUMB / max(w, h)
    im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    dr = ImageDraw.Draw(im)
    for c, b in zip(json.loads(rec["class_ids"]), json.loads(rec["boxes"])):
        xc, yc, bw, bh = b
        x0 = (xc - bw / 2) * w * scale; x1 = (xc + bw / 2) * w * scale
        y0 = (yc - bh / 2) * h * scale; y1 = (yc + bh / 2) * h * scale
        dr.rectangle([max(0, x0), max(0, y0), min(im.width - 1, x1), min(im.height - 1, y1)],
                     outline=(0, 220, 0) if c == 0 else (255, 40, 40), width=2)
    canvas = Image.new("RGB", (THUMB, THUMB + 14), (24, 24, 24))
    canvas.paste(im, ((THUMB - im.width) // 2, (THUMB - im.height) // 2))
    name = os.path.basename(rec["image_path"])
    dr2 = ImageDraw.Draw(canvas)
    dr2.text((3, THUMB + 1), f"{name} [{rec['split_official']}]", fill=(220, 220, 220))
    return canvas

for catname, fname in [("smoke_pos", "dfire_smoke_positives.jpg"),
                       ("fire_only", "dfire_fire_only.jpg"),
                       ("none", "dfire_negatives.jpg")]:
    sub = df[df["cat3"] == catname].sample(n=min(GRID * GRID, (df["cat3"] == catname).sum()),
                                           random_state=SEED)
    sheet = Image.new("RGB", (GRID * THUMB, GRID * (THUMB + 14)), (24, 24, 24))
    for i, (_, rec) in enumerate(sub.iterrows()):
        sheet.paste(tile(rec), ((i % GRID) * THUMB, (i // GRID) * (THUMB + 14)))
    out = os.path.join(OUT, fname)
    sheet.save(out, quality=88)
    print("wrote", out, f"({len(sub)} tiles, pool={int((df['cat3']==catname).sum())})")
print("DONE")
