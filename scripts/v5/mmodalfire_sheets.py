#!/usr/bin/env python
"""Contact-sheet builder for MmodalFire visual review (V5 Day-1, CPU only).

Usage:
  python scripts/v5/mmodalfire_sheets.py --session Mist1 --stride 10 --crop 700,200,1450,950 \
      --tile 480 --cols 4 --rows 3 --out data/v5/mmodalfire/_review_sheets/mist1
  python scripts/v5/mmodalfire_sheets.py --session Cotton7 --indices 0,120,240,360 --tile 960 --cols 2 --rows 2 --out ...
Frames are read from data/v4/raw/mmodalfire/frames_1fps/<session>/ (1 fps, 1920x1080, wall-clock in filename).
Each tile is labelled with HHMMSS and frame index within the session.
"""
import argparse, glob, os, cv2, numpy as np
FR_ROOT = "data/v4/raw/mmodalfire/frames_1fps"

def build(session, frames, crop, tile_w, cols, rows, out_prefix, label_extra=None):
    x0,y0,x1,y1 = crop
    scale = tile_w/float(x1-x0); tile_h = int(round((y1-y0)*scale))
    per = cols*rows; paths=[]
    for s in range(0, len(frames), per):
        chunk = frames[s:s+per]; tiles=[]
        for (i,p) in chunk:
            im = cv2.imread(p)
            if im is None: im = np.zeros((1080,1920,3),np.uint8)
            c = im[y0:y1, x0:x1]; c = cv2.resize(c, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            hhmmss = os.path.basename(p)[-10:-4]
            txt = f"{hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:]} #{i}"
            if label_extra: txt += " " + label_extra(i,p)
            cv2.putText(c, txt, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 4)
            cv2.putText(c, txt, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
            cv2.rectangle(c,(0,0),(tile_w-1,tile_h-1),(80,80,80),1)
            tiles.append(c)
        while len(tiles) < per: tiles.append(np.zeros((tile_h,tile_w,3),np.uint8))
        grid = np.vstack([np.hstack(tiles[r*cols:(r+1)*cols]) for r in range(rows)])
        outp = f"{out_prefix}_{s//per:02d}.jpg"
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        cv2.imwrite(outp, grid, [cv2.IMWRITE_JPEG_QUALITY, 88]); paths.append(outp)
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--indices", default=None, help="comma list of frame indices within session")
    ap.add_argument("--range", default=None, help="start:end indices (end exclusive)")
    ap.add_argument("--crop", default="0,0,1920,1080")
    ap.add_argument("--tile", type=int, default=480)
    ap.add_argument("--cols", type=int, default=4); ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fr = sorted(glob.glob(f"{FR_ROOT}/{a.session}/*.jpg"))
    lo,hi = 0,len(fr)
    if a.range: lo,hi = [int(v) for v in a.range.split(":")]
    if a.indices: idx=[int(v) for v in a.indices.split(",")]
    else: idx=list(range(lo,hi,a.stride or 1))
    frames=[(i,fr[i]) for i in idx if 0<=i<len(fr)]
    crop=tuple(int(v) for v in a.crop.split(","))
    paths = build(a.session, frames, crop, a.tile, a.cols, a.rows, a.out)
    print(f"{a.session}: {len(fr)} frames, {len(frames)} tiles -> {len(paths)} sheets"); [print(" ",p) for p in paths]

if __name__=="__main__": main()
