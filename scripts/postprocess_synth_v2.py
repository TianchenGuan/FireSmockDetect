"""Unpack synthetic-pilot segment tars and build contact sheets.

- data/v2/synth/segments/<seg>/results.tar -> data/v2/synth/raw/
  (PNG + per-image provenance JSON, worker logs kept per segment)
- contact sheets under data/v2/synth/contact_sheets/:
    triplet_XX.png   — rows of [background | control | smoke | confounder]
                       with the requested region rectangle drawn on
                       smoke (red) and confounder (orange). Requested regions
                       are generation REQUESTS, not verified labels.
    smokezoom_XX.png — 4x-upscaled crops around each smoke job region
                       (3x region padding) for faint-plume screening.
- data/v2/synth/generation_stats.json — per-segment + total rates.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "data" / "v2" / "synth"
RAW = SYNTH / "raw"
SHEETS = SYNTH / "contact_sheets"
TH_W, TH_H = 400, 225
ROWS_PER_SHEET = 12
ZOOM_COLS, ZOOM_ROWS = 5, 8


def unpack() -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    stats = {}
    for seg in sorted((SYNTH / "segments").glob("*/")):
        t = seg / "results.tar"
        if not t.exists():
            continue
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                n = Path(m.name).name
                if n.endswith((".png", ".json")) and not n.startswith(
                        ("worker", "status")):
                    tf.extract(m, RAW, filter="data")
        st = seg / "status.json"
        if st.exists():
            stats[seg.name] = json.loads(st.read_text())
    # flatten any ./ prefix dirs
    for p in RAW.rglob("*"):
        if p.is_file() and p.parent != RAW:
            p.rename(RAW / p.name)
    return stats


def load_meta():
    metas = {}
    for j in RAW.glob("bg*.json"):
        metas[j.stem] = json.loads(j.read_text())
    return metas


def thumb_with_box(img_path, meta, color):
    im = Image.open(img_path).convert("RGB").resize((TH_W, TH_H))
    if meta and meta.get("region_norm") and color:
        d = ImageDraw.Draw(im)
        x1, y1, x2, y2 = meta["region_norm"]
        d.rectangle([x1 * TH_W, y1 * TH_H, x2 * TH_W, y2 * TH_H],
                    outline=color, width=2)
    return im


def triplet_sheets(metas):
    SHEETS.mkdir(parents=True, exist_ok=True)
    bgs = sorted({k[:5] for k in metas})  # bgNNN
    label_h, pad = 14, 4
    cols = ["background", "control", "smoke", "confounder"]
    for si in range(0, len(bgs), ROWS_PER_SHEET):
        chunk = bgs[si:si + ROWS_PER_SHEET]
        W = 4 * TH_W + 5 * pad
        H = len(chunk) * (TH_H + label_h + pad) + pad
        sheet = Image.new("RGB", (W, H), (24, 24, 24))
        d = ImageDraw.Draw(sheet)
        for ri, bg in enumerate(chunk):
            y = pad + ri * (TH_H + label_h + pad)
            sm = metas.get(f"{bg}_smoke")
            any_meta = sm or metas.get(f"{bg}_ctrl") or metas.get(f"{bg}_conf")
            srcs = [
                (Path(any_meta["background_src"]) if any_meta else None,
                 None, None),
                (RAW / f"{bg}_ctrl.png", metas.get(f"{bg}_ctrl"), None),
                (RAW / f"{bg}_smoke.png", sm, (255, 40, 40)),
                (RAW / f"{bg}_conf.png", metas.get(f"{bg}_conf"),
                 (255, 160, 20)),
            ]
            for ci, (p, meta, color) in enumerate(srcs):
                x = pad + ci * (TH_W + pad)
                if p and p.exists():
                    sheet.paste(thumb_with_box(p, meta, color), (x, y))
                else:
                    d.rectangle([x, y, x + TH_W, y + TH_H], outline=(90,) * 3)
                    d.text((x + 8, y + 8), "missing", fill=(160,) * 3)
            lab = bg
            if sm:
                lab += (f"  {sm.get('scale','')}  seed {sm.get('seed','')}"
                        f"  {sm.get('site','')}/{sm.get('camera_id','')}")
            cf = metas.get(f"{bg}_conf")
            if cf:
                lab += f"  conf={cf.get('confounder_type','')}"
            d.text((pad, y + TH_H + 1), lab, fill=(220,) * 3)
        for ci, c in enumerate(cols):
            d.text((pad + ci * (TH_W + pad) + 4, 2), c, fill=(255, 255, 0))
        out = SHEETS / f"triplet_{si // ROWS_PER_SHEET:02d}.png"
        sheet.save(out)
        print("wrote", out)


def zoom_sheets(metas):
    smoke = sorted(k for k in metas if k.endswith("_smoke")
                   and (RAW / f"{k}.png").exists())
    cell_w, cell_h, label_h, pad = 384, 288, 14, 4
    per = ZOOM_COLS * ZOOM_ROWS
    for si in range(0, len(smoke), per):
        chunk = smoke[si:si + per]
        W = ZOOM_COLS * (cell_w + pad) + pad
        H = ZOOM_ROWS * (cell_h + label_h + pad) + pad
        sheet = Image.new("RGB", (W, H), (24, 24, 24))
        d = ImageDraw.Draw(sheet)
        for i, k in enumerate(chunk):
            m = metas[k]
            im = Image.open(RAW / f"{k}.png").convert("RGB")
            x1, y1, x2, y2 = m["region_norm"]
            cx, cy = (x1 + x2) / 2 * im.width, (y1 + y2) / 2 * im.height
            rw, rh = (x2 - x1) * im.width, (y2 - y1) * im.height
            # 3x padded crop around region, clamped, 4:3
            cw = max(rw * 3, cell_w / 4)
            ch = max(rh * 3, cell_h / 4)
            if cw / ch > cell_w / cell_h:
                ch = cw * cell_h / cell_w
            else:
                cw = ch * cell_w / cell_h
            bx1 = max(0, min(im.width - cw, cx - cw / 2))
            by1 = max(0, min(im.height - ch, cy - ch / 2))
            crop = im.crop((int(bx1), int(by1), int(bx1 + cw),
                            int(by1 + ch))).resize((cell_w, cell_h))
            dc = ImageDraw.Draw(crop)
            sx, sy = cell_w / cw, cell_h / ch
            dc.rectangle([(x1 * im.width - bx1) * sx,
                          (y1 * im.height - by1) * sy,
                          (x2 * im.width - bx1) * sx,
                          (y2 * im.height - by1) * sy],
                         outline=(255, 40, 40), width=2)
            gx = pad + (i % ZOOM_COLS) * (cell_w + pad)
            gy = pad + (i // ZOOM_COLS) * (cell_h + label_h + pad)
            sheet.paste(crop, (gx, gy))
            d.text((gx, gy + cell_h + 1),
                   f"{k} {m.get('scale')} seed {m.get('seed')}",
                   fill=(220,) * 3)
        out = SHEETS / f"smokezoom_{si // per:02d}.png"
        sheet.save(out)
        print("wrote", out)


def main():
    stats = unpack()
    metas = load_meta()
    pngs = sorted(RAW.glob("bg*.png"))
    kinds = {}
    for p in pngs:
        kinds[p.stem.split("_")[-1]] = kinds.get(p.stem.split("_")[-1], 0) + 1
    summary = dict(segments=stats, n_png=len(pngs), per_kind=kinds,
                   n_meta=len(metas))
    (SYNTH / "generation_stats.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    triplet_sheets(metas)
    zoom_sheets(metas)


if __name__ == "__main__":
    main()
