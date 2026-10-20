#!/usr/bin/env python
"""V5 Day-1 audit helper (A): deterministic seed-0 sample of 30 Boreal TRAIN-burn
Subset-B clips (evo + karkkila ONLY), contact sheets for human-style viewing,
and a camera-motion cue (global frame shift via phase correlation + mean abs diff).
The motion cue is NOT a verdict and is NOT detector output; verdicts are made by
looking at the frames. heinola (VAL) and ruokolahti (TEST) are never touched."""
import json, random, sys, os
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = '/local/home/tianrgua/cvproject'
OUT = f'{ROOT}/data/v5/audit'
SHEETS = f'{OUT}/contact_sheets/boreal'
os.makedirs(SHEETS, exist_ok=True)

df = pd.read_parquet(f'{ROOT}/data/v4/sequences/sequences_v4.parquet')
b = df[(df.source == 'boreal') & (df.split == 'train') & (df.parent_unit.isin(['burn_evo', 'burn_karkkila']))]
assert not b.abs_path.str.contains('heinola|ruokolahti').any()
clips = sorted(b.seq_id.unique())
assert len(clips) == 105, len(clips)
rng = random.Random(0)
sample = sorted(rng.sample(clips, 30))
meta = b.groupby('seq_id').agg(parent_unit=('parent_unit', 'first'), seq_label=('seq_label', 'first'), n=('frame_idx', 'count')).loc[sample]
print(meta.groupby(['parent_unit', 'seq_label']).size())
json.dump({'seed': 0, 'population': clips, 'sample': sample}, open(f'{OUT}/boreal_sample_seed0.json', 'w'), indent=1)

def motion_cue(paths, small=(256, 135)):
    """Per-step global shift (px at 256-wide) via phase correlation on grayscale + MAD."""
    prev = None; out = []
    for p in paths:
        g = np.asarray(Image.open(p).convert('L').resize(small), dtype=np.float32)
        g = g - g.mean()
        if prev is not None:
            F = np.fft.fft2(prev) * np.conj(np.fft.fft2(g))
            R = F / (np.abs(F) + 1e-6)
            r = np.fft.ifft2(R).real
            dy, dx = np.unravel_index(np.argmax(r), r.shape)
            dy = dy - small[1] if dy > small[1] // 2 else dy
            dx = dx - small[0] if dx > small[0] // 2 else dx
            out.append((int(dx), int(dy), float(np.abs(prev - g).mean())))
        prev = g
    return out

font = ImageFont.load_default()
rows = []
for cid in sample:
    fr = b[b.seq_id == cid].sort_values('frame_idx')
    paths = fr.abs_path.tolist()
    mc = motion_cue(paths)
    shifts = [abs(dx) + abs(dy) for dx, dy, _ in mc]
    # contact sheet: 6 cols, thumbnails 512x270 with frame index label
    tw, th = 512, 270; cols = 6; n = len(paths); rws = (n + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tw, rws * (th + 14)), 'black')
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(paths):
        im = Image.open(p).convert('RGB').resize((tw, th))
        x, y = (i % cols) * tw, (i // cols) * (th + 14)
        sheet.paste(im, (x, y + 14))
        s = shifts[i - 1] if i > 0 else 0
        d.text((x + 2, y), f't{i:02d}  shift={s}', fill='yellow' if s > 8 else 'white', font=font)
    sheet.save(f'{SHEETS}/{cid}.jpg', quality=82)
    rows.append(dict(seq_id=cid, parent_unit=fr.parent_unit.iloc[0], seq_label=fr.seq_label.iloc[0], n_frames=n,
                     max_shift_px256=max(shifts) if shifts else 0, n_steps_shift_gt8=sum(s > 8 for s in shifts),
                     mean_mad=float(np.mean([m for _, _, m in mc])) if mc else 0.0,
                     shifts=' '.join(map(str, shifts))))
pd.DataFrame(rows).to_csv(f'{OUT}/boreal_sample_motion_cue.csv', index=False)
print(pd.DataFrame(rows)[['seq_id', 'seq_label', 'n_frames', 'max_shift_px256', 'n_steps_shift_gt8', 'mean_mad']].to_string())
