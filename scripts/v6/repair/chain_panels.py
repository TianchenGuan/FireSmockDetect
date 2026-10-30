"""Generation-chain recovery panels for the sampled V6 images (prereg A3 §1).

For each arm-B composite/negative from the sample zip: background -> asset
(with its derived alpha) -> final encoded composite with trained boxes ->
EXACT detector-style input (1280 letterbox) with the same boxes.
For each GLM sample: raw RGB output -> the luma-derived pseudo-alpha the
filter would extract (shown explicitly as DERIVED, not native) -> prompt and
seed from the sidecar.

Writes PNG panels + a JSON lineage dump per sample to reports/v6/repair/panels/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports/v6/repair/panels"
OUT.mkdir(parents=True, exist_ok=True)

ARMB = ["p001179", "p024945", "c002380", "c001420"]
GLM_RAW = ["6272e4875676c035", "beda7feb8407c3fd"]
GLM_ACC = ["glm_1ae38d532c1a7756", "glm_38acfcaa2dbcd81f"]


def letterbox(im: Image.Image, size=1280, color=(114, 114, 114)):
    s = size / max(im.size)
    nw, nh = round(im.size[0] * s), round(im.size[1] * s)
    im2 = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), color)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas.paste(im2, (ox, oy))
    return canvas, (s, ox, oy, nw, nh)


def draw_boxes(im: Image.Image, boxes, geom=None, color=(255, 60, 60)):
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for cx, cy, w, h in boxes:
        if geom:  # boxes are normalized to the pre-letterbox image
            s, ox, oy, nw, nh = geom
            x0, y0 = (cx - w / 2) * nw + ox, (cy - h / 2) * nh + oy
            x1, y1 = (cx + w / 2) * nw + ox, (cy + h / 2) * nh + oy
        else:
            W, H = im.size
            x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
            x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
    return im


def tile(images, labels, out_path, tile_w=640):
    tiles = []
    for im, lab in zip(images, labels):
        im = im.convert("RGB")
        s = tile_w / im.size[0]
        im = im.resize((tile_w, max(1, round(im.size[1] * s))), Image.LANCZOS)
        header = Image.new("RGB", (tile_w, 28), (20, 20, 20))
        hd = ImageDraw.Draw(header)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 15)
        except Exception:
            font = ImageFont.load_default()
        hd.text((6, 6), lab[:90], fill=(240, 240, 240), font=font)
        col = Image.new("RGB", (tile_w, im.size[1] + 28), (0, 0, 0))
        col.paste(header, (0, 0))
        col.paste(im, (0, 28))
        tiles.append(col)
    h = max(t.size[1] for t in tiles)
    panel = Image.new("RGB", (tile_w * len(tiles), h), (0, 0, 0))
    for i, t in enumerate(tiles):
        panel.paste(t, (i * tile_w, 0))
    panel.save(out_path)
    print("wrote", out_path)


def luma_pseudo_alpha(p: Path) -> Image.Image:
    rgb = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
    luma = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    floor = float(np.percentile(luma, 8))
    a = np.clip((luma - floor) * 1.8, 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))


def main():
    recs = {}
    for line in (ROOT / "data/v6/synthB2/manifest.jsonl").open():
        r = json.loads(line)
        stem = Path(r.get("out", r.get("img", ""))).stem
        recs.setdefault(stem, r)

    for stem in ARMB:
        r = recs.get(stem)
        if not r:
            print("MISSING manifest record:", stem)
            continue
        (OUT / f"{stem}_lineage.json").write_text(json.dumps(r, indent=1))
        comp = ROOT / "data/v6/synthB2/images" / f"{stem}.jpg"
        if not comp.exists():
            hits = list((ROOT / "data/v6/synthB2").rglob(f"{stem}.jpg"))
            comp = hits[0] if hits else None
        lbl = Path(str(comp).replace("/images/", "/labels/")
                   ).with_suffix(".txt") if comp else None
        boxes = []
        if lbl and lbl.exists() and lbl.stat().st_size:
            boxes = [[float(x) for x in l.split()[1:]]
                     for l in lbl.read_text().splitlines() if l.strip()]
        imgs, labs = [], []
        bgp = r.get("bg") or r.get("bg_path")
        if bgp and Path(bgp).exists():
            imgs.append(Image.open(bgp))
            labs.append(f"1. REAL TRAIN background ({Path(bgp).name})")
        ap = r.get("asset")
        if ap and Path(ap).exists():
            a = Image.open(ap).convert("RGB")
            imgs.append(a)
            labs.append(f"2. asset {Path(ap).stem} (luma mode)")
        if comp and comp.exists():
            ci = Image.open(comp)
            imgs.append(draw_boxes(ci, boxes))
            labs.append(f"3. final encoded composite + {len(boxes)} trained box(es)")
            lb, geom = letterbox(ci)
            imgs.append(draw_boxes(lb, boxes, geom))
            labs.append("4. detector input 1280 letterbox + boxes")
        if imgs:
            tile(imgs, labs, OUT / f"chain_{stem}.png")

    for stem in GLM_RAW:
        hits = list((ROOT / "data/v6/gen").rglob(f"{stem}.jpg"))
        if not hits:
            print("missing glm raw", stem)
            continue
        p = hits[0]
        meta = json.loads(p.with_suffix(".json").read_text())
        (OUT / f"{stem}_lineage.json").write_text(json.dumps(meta, indent=1))
        tile([Image.open(p), luma_pseudo_alpha(p)],
             [f"GLM raw {stem} (spec {meta.get('spec_id')}, seed {meta.get('seed')})",
              "DERIVED luma pseudo-alpha (filter view) - NOT a native matte"],
             OUT / f"chain_glmraw_{stem}.png")

    for stem in GLM_ACC:
        p = ROOT / "data/v6/assets/C_glm/png" / f"{stem}.png"
        if not p.exists():
            hits = list((ROOT / "data/v6/assets").rglob(f"{stem}.png"))
            p = hits[0] if hits else None
        if not p:
            print("missing accepted", stem)
            continue
        man = None
        for mf in (ROOT / "data/v6/assets/C_glm/manifest.jsonl",
                   ROOT / "data/v6/assets/C_glm_b200/manifest.jsonl"):
            if mf.exists():
                for line in mf.open():
                    rr = json.loads(line)
                    if stem in rr.get("asset", ""):
                        man = rr
                        break
            if man:
                break
        if man:
            (OUT / f"{stem}_lineage.json").write_text(json.dumps(man, indent=1))
        tile([Image.open(p), luma_pseudo_alpha(p)],
             [f"accepted GLM asset {stem} (RGB, desaturated crop)",
              "DERIVED luma pseudo-alpha (compositor opacity) - NOT native"],
             OUT / f"chain_glmacc_{stem}.png")


if __name__ == "__main__":
    main()
