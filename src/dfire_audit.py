#!/usr/bin/env python3
"""Track C: audit D-Fire dataset and build manifest.

Columns: image_path (relative to project root), split_official, n_boxes,
boxes (JSON list of [xc,yc,w,h] normalized YOLO), class_ids (JSON list),
width, height, source='dfire', license='CC0-1.0', filename_prefix.
Class map (verified against README box counts): 0=smoke, 1=fire.
"""
import json
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

ROOT = "/local/home/tianrgua/cvproject"
EXTRACTED = os.path.join(ROOT, "data/raw/dfire/extracted")
LICENSE = "CC0-1.0"

def process_one(args):
    split, fname = args
    img_path = os.path.join(EXTRACTED, split, "images", fname)
    lbl_path = os.path.join(EXTRACTED, split, "labels", os.path.splitext(fname)[0] + ".txt")
    rec = {
        "image_path": os.path.relpath(img_path, ROOT),
        "split_official": split,
        "source": "dfire",
        "license": LICENSE,
    }
    # image integrity + dims
    try:
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            im.load()
            rec["width"], rec["height"] = im.size
            rec["mode"] = im.mode
        rec["corrupt"] = False
    except Exception as e:
        rec["width"], rec["height"] = None, None
        rec["mode"] = None
        rec["corrupt"] = True
        rec["corrupt_err"] = str(e)[:200]
    # labels
    boxes, cids, bad_lines = [], [], 0
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                try:
                    c = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])
                    boxes.append([round(xc, 6), round(yc, 6), round(w, 6), round(h, 6)])
                    cids.append(c)
                except Exception:
                    bad_lines += 1
        rec["has_label_file"] = True
    else:
        rec["has_label_file"] = False
    rec["n_boxes"] = len(boxes)
    rec["boxes"] = json.dumps(boxes)
    rec["class_ids"] = json.dumps(cids)
    rec["bad_label_lines"] = bad_lines
    # coordinate sanity
    oob = sum(1 for b in boxes if not all(-1e-6 <= v <= 1.0 + 1e-6 for v in b))
    rec["n_boxes_out_of_range"] = oob
    rec["filename_prefix"] = "".join(ch for ch in fname if not ch.isdigit()).split(".")[0]
    return rec


def main():
    tasks = []
    for split in ("train", "test"):
        for fname in sorted(os.listdir(os.path.join(EXTRACTED, split, "images"))):
            tasks.append((split, fname))
    print(f"{len(tasks)} images to process")
    with Pool(96) as p:
        recs = p.map(process_one, tasks, chunksize=64)
    df = pd.DataFrame(recs)

    out_parquet = os.path.join(ROOT, "data/manifests/dfire_manifest.parquet")
    out_csv = os.path.join(ROOT, "data/manifests/dfire_manifest.csv.gz")
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False, compression="gzip")
    print("wrote", out_parquet, "and", out_csv)

    # ---------- audit report ----------
    print("\n=== BASIC ===")
    print(df.groupby("split_official").size())
    print("corrupt images:", int(df["corrupt"].sum()))
    if df["corrupt"].any():
        print(df.loc[df["corrupt"], ["image_path", "corrupt_err"]].to_string())
    print("missing label files:", int((~df["has_label_file"]).sum()))
    print("bad label lines total:", int(df["bad_label_lines"].sum()))
    print("boxes out of [0,1] range:", int(df["n_boxes_out_of_range"].sum()))

    def cat(cids_json):
        s = set(json.loads(cids_json))
        if s == {0}: return "smoke_only"
        if s == {1}: return "fire_only"
        if s == {0, 1}: return "smoke_and_fire"
        if not s: return "none"
        return "other:" + str(sorted(s))

    df["category"] = df["class_ids"].map(cat)
    print("\n=== CATEGORY x SPLIT ===")
    print(df.pivot_table(index="category", columns="split_official", values="image_path", aggfunc="count", fill_value=0, margins=True))

    print("\n=== BOX COUNTS PER CLASS x SPLIT ===")
    rows = []
    for split, g in df.groupby("split_official"):
        cnt = Counter()
        for cj in g["class_ids"]:
            cnt.update(json.loads(cj))
        rows.append({"split": split, "smoke(0)": cnt.get(0, 0), "fire(1)": cnt.get(1, 0)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== FILENAME PREFIX x SPLIT ===")
    print(df.pivot_table(index="filename_prefix", columns="split_official", values="image_path", aggfunc="count", fill_value=0, margins=True))

    print("\n=== IMAGE DIMS ===")
    ok = df[~df["corrupt"]]
    print("width:  min", ok["width"].min(), "median", ok["width"].median(), "max", ok["width"].max())
    print("height: min", ok["height"].min(), "median", ok["height"].median(), "max", ok["height"].max())
    print("top-10 resolutions:")
    print((ok["width"].astype(str) + "x" + ok["height"].astype(str)).value_counts().head(10).to_string())
    print("modes:", ok["mode"].value_counts().to_dict())

    # smoke box size distribution
    print("\n=== SMOKE BOX SIZE DISTRIBUTION ===")
    areas_norm, areas_px = [], []
    for _, r in ok.iterrows():
        cids = json.loads(r["class_ids"])
        boxes = json.loads(r["boxes"])
        for c, b in zip(cids, boxes):
            if c == 0:
                areas_norm.append(b[2] * b[3])
                areas_px.append(b[2] * r["width"] * b[3] * r["height"])
    areas_norm = np.array(areas_norm); areas_px = np.array(areas_px)
    print(f"n smoke boxes: {len(areas_norm)}")
    qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    print("normalized area quantiles:")
    for q, v in zip(qs, np.quantile(areas_norm, qs)):
        print(f"  q{q:0.2f}: {v:.5f}  (sqrt={np.sqrt(v):.4f})")
    print("pixel-area quantiles (sqrt = box side px):")
    for q, v in zip(qs, np.quantile(areas_px, qs)):
        print(f"  q{q:0.2f}: {v:10.0f} px^2  (sqrt={np.sqrt(v):6.1f} px)")
    for thr, name in [(32*32, "<32x32px (COCO-small)"), (96*96, "<96x96px (COCO-medium cutoff)")]:
        print(f"  smoke boxes {name}: {(areas_px < thr).sum()} ({100*(areas_px < thr).mean():.1f}%)")
    print(f"  smoke boxes covering >50% of image (norm area>0.5): {(areas_norm > 0.5).sum()} ({100*(areas_norm>0.5).mean():.1f}%)")

    # save distribution for plotting
    np.save(os.path.join(ROOT, "data/manifests/dfire_smoke_areas_px.npy"), areas_px)
    print("\nDONE")


if __name__ == "__main__":
    main()
