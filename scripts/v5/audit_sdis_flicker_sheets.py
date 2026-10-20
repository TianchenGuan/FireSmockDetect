#!/usr/bin/env python
"""V5 Day-1 audit helper (B), v2: per-flip comparison rows for the top-20 flicker
sequences (pyro-sdis TRAIN sites only). For every run of unlabeled frames ('island')
bounded by labeled frames, crop the SAME region (around the union of the bounding
labeled boxes, >=3x box size, >=200 px) from labeled-before / island / labeled-after
frames and tile them in a row (green = labeled box; red border = unlabeled).
Also a full-frame context image. No model is involved."""
import json, os, pandas as pd
from PIL import Image, ImageDraw, ImageFont
ROOT = '/local/home/tianrgua/cvproject'
OUT = f'{ROOT}/data/v5/audit/contact_sheets/sdis'; os.makedirs(OUT, exist_ok=True)
top = pd.read_csv(f'{ROOT}/data/v5/audit/sdis_flicker_top20.csv')
S = {s['seq_id']: s for s in json.load(open(f'{ROOT}/data/v3/sequences/sequences_train.json'))}
font = ImageFont.load_default()
T = 360; MAXROWS = 14; MAXTILES = 8
meta = []
def boxes_px(f, W, H):
    return [((cx - w/2)*W, (cy - h/2)*H, (cx + w/2)*W, (cy + h/2)*H) for cx, cy, w, h in f['boxes_xywhn']]
for sid in top.seq_id:
    s = S[sid]; fr = s['frames']; n = len(fr)
    W, H = Image.open(fr[0]['image_path']).size
    lab = [bool(f['smoke']) for f in fr]
    islands = []; i = 0
    while i < n:
        if not lab[i]:
            j = i
            while j < n and not lab[j]: j += 1
            if i > 0 and j < n: islands.append((i - 1, list(range(i, j)), j))   # bounded both sides
            i = j
        else: i += 1
    rows_img = []
    for (pb, isl, pa) in islands[:MAXROWS]:
        bb = boxes_px(fr[pb], W, H) + boxes_px(fr[pa], W, H)
        x0 = min(b[0] for b in bb); y0 = min(b[1] for b in bb); x1 = max(b[2] for b in bb); y1 = max(b[3] for b in bb)
        side = max(x1 - x0, y1 - y0) * 3; side = max(side, 200); side = min(side, H)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cx0 = int(max(0, min(W - side, cx - side / 2))); cy0 = int(max(0, min(H - side, cy - side / 2))); side = int(side)
        crop = (cx0, cy0, cx0 + side, cy0 + side)
        idxs = [pb] + isl + [pa]
        if len(idxs) > MAXTILES: idxs = [pb] + isl[:MAXTILES - 3] + [isl[-1], pa]
        row = Image.new('RGB', (len(idxs) * T, T + 14), 'black'); d = ImageDraw.Draw(row)
        for k, fi in enumerate(idxs):
            f = fr[fi]; im = Image.open(f['image_path']).convert('RGB').crop(crop)
            dc = ImageDraw.Draw(im)
            for b in boxes_px(f, W, H): dc.rectangle([b[0]-cx0, b[1]-cy0, b[2]-cx0, b[3]-cy0], outline='lime', width=2)
            im = im.resize((T, T))
            if not f['smoke']: ImageDraw.Draw(im).rectangle([0, 0, T-1, T-1], outline='red', width=6)
            row.paste(im, (k * T, 14))
            dt = f['ts'] - fr[idxs[k-1]]['ts'] if k else 0
            d.text((k*T + 3, 1), f"f{fi:03d} {'P' if f['smoke'] else 'n'} +{dt}s {f['image_name'][-12:-4]} crop{crop}", fill='lime' if f['smoke'] else 'red', font=font)
        rows_img.append(row)
    if rows_img:
        Wm = max(r.width for r in rows_img); sheet = Image.new('RGB', (Wm, sum(r.height + 6 for r in rows_img)), (40, 40, 40)); y = 0
        for r in rows_img: sheet.paste(r, (0, y)); y += r.height + 6
        sheet.save(f'{OUT}/{sid}.jpg', quality=85)
    pf = next(f for f in fr if f['smoke']); ctx = Image.open(pf['image_path']).convert('RGB'); dc = ImageDraw.Draw(ctx)
    for f in fr:
        for b in boxes_px(f, W, H): dc.rectangle(b, outline='lime', width=1)
    ctx.save(f'{OUT}/{sid}_context.jpg', quality=80)
    meta.append(dict(seq_id=sid, n_frames=n, n_islands_bounded=len(islands), n_island_frames=sum(len(x[1]) for x in islands), rows_shown=min(len(islands), MAXROWS),
                     leading_neg=next((k for k, l in enumerate(lab) if l), n), trailing_neg=next((k for k, l in enumerate(reversed(lab)) if l), n)))
    print(sid, 'islands', len(islands), 'island frames', sum(len(x[1]) for x in islands), 'lead/trail neg', meta[-1]['leading_neg'], meta[-1]['trailing_neg'])
pd.DataFrame(meta).to_csv(f'{ROOT}/data/v5/audit/sdis_flicker_islands.csv', index=False)
