"""Contact sheets for per-video burn identification. RGB: 3 frames per video (10/50/90%). Thermal: overlay crops (top-left) of first/last frame."""
import os, sys
from PIL import Image, ImageDraw
RAW = '/local/home/tianrgua/cvproject/data/v5b/raw/fireman/Multiclass/Unimodal'
OUT = '/local/home/tianrgua/cvproject/data/v5b/manifests/_inspect'; os.makedirs(OUT, exist_ok=True)
def vids(mod):
    out = []
    for split in ('train', 'val', 'test'):
        d = f'{RAW}/{mod}/{split}'
        if os.path.isdir(d):
            for v in sorted(os.listdir(d)): out.append((f'{split}/{v}', f'{d}/{v}/images'))
    return out
def rgb_sheets(per_sheet=8):
    vs = vids('rgb'); TW, TH = 480, 270
    for si in range(0, len(vs), per_sheet):
        chunk = vs[si:si + per_sheet]; sheet = Image.new('RGB', (3 * TW, len(chunk) * TH), 'black')
        for r, (name, d) in enumerate(chunk):
            fs = sorted(os.listdir(d))
            for c, frac in enumerate((0.1, 0.5, 0.9)):
                f = fs[min(len(fs) - 1, int(frac * len(fs)))]
                im = Image.open(f'{d}/{f}').convert('RGB'); w, h = im.size; im = im.resize((TW, TH)); dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, TW, 14], fill='black'); dr.text((3, 2), f'{name} {f} {w}x{h} n={len(fs)}', fill='yellow')
                sheet.paste(im, (c * TW, r * TH))
        p = f'{OUT}/fireman_rgb_sheet_{si // per_sheet:02d}.jpg'; sheet.save(p, quality=80); print(p)
def thermal_overlays():
    vs = vids('thermal'); rows = []
    for name, d in vs:
        fs = sorted(os.listdir(d))
        for f in (fs[0], fs[len(fs) // 2], fs[-1]):
            im = Image.open(f'{d}/{f}').convert('RGB'); w, h = im.size
            crop = im.crop((0, 0, min(w, int(w * 0.6)), min(h, int(h * 0.2)))); crop = crop.resize((int(crop.width * 2), int(crop.height * 2)))
            dr = ImageDraw.Draw(crop); dr.text((3, crop.height - 12), f'{name} {f} {w}x{h} n={len(fs)}', fill='yellow'); rows.append(crop)
    W = max(r.width for r in rows); H = sum(r.height for r in rows); sheet = Image.new('RGB', (W, H), 'black'); y = 0
    for r in rows: sheet.paste(r, (0, y)); y += r.height
    p = f'{OUT}/fireman_thermal_overlays.jpg'; sheet.save(p, quality=85); print(p, sheet.size)
    # also full first frames of thermal videos
    TW, TH = 480, 384; sheet = Image.new('RGB', (2 * TW, ((len(vs) + 1) // 2) * TH), 'black')
    for i, (name, d) in enumerate(vs):
        fs = sorted(os.listdir(d)); im = Image.open(f'{d}/{fs[len(fs) // 2]}').convert('RGB').resize((TW, TH)); dr = ImageDraw.Draw(im); dr.rectangle([0, TH - 14, TW, TH], fill='black'); dr.text((3, TH - 12), f'{name} n={len(fs)}', fill='yellow')
        sheet.paste(im, ((i % 2) * TW, (i // 2) * TH))
    p = f'{OUT}/fireman_thermal_frames.jpg'; sheet.save(p, quality=80); print(p)
if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('all', 'rgb'): rgb_sheets()
    if what in ('all', 'thermal'): thermal_overlays()
