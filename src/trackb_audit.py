"""Track B: audit pyro-sdis manifest + generate contact sheets."""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

ROOT = "/local/home/tianrgua/cvproject"
MAN = os.path.join(ROOT, "data/manifests/pyro_sdis_manifest.parquet")
SHEET_DIR = os.path.join(ROOT, "data/contact_sheets/pyro_sdis")

df = pd.read_parquet(MAN)
print("=== AUDIT ===")
print("total rows:", len(df))
print(df.groupby("split_official").size().to_string())
print("\npositives (>=1 box) per split:")
print(df[df.n_boxes > 0].groupby("split_official").size().to_string())
print("\nnegatives (0 box) per split:")
print(df[df.n_boxes == 0].groupby("split_official").size().to_string())
print("\nbox count distribution (all):")
print(df.n_boxes.value_counts().sort_index().to_string())
print("\ntotal boxes:", int(df.n_boxes.sum()))
print("\nclass ids seen (orig):", sorted({c for cl in df.class_ids_orig for c in json.loads(cl)}))
print("decode failures:", int((~df.decode_ok).sum()))
if (~df.decode_ok).any():
    print(df[~df.decode_ok][["image_name", "decode_err"]].to_string())
print("coord violations:", int(df.coord_violation.sum()))
print("duplicate image_name:", int(df.image_name.duplicated().sum()))
print("name/meta mismatch:", int((~df.name_meta_match).sum()))
print("\nimage size distribution (width x height):")
print(df.groupby(["width", "height"]).size().to_string())
print("\npartner distribution:")
print(df.groupby(["split_official", "partner"]).size().to_string())
print("n distinct camera_id:", df.camera_id.nunique(), "| n distinct sites:", df.site.nunique())
print("sites:", sorted(df.site.unique()))
tr = set(df[df.split_official == "train"].camera_id.unique())
vl = set(df[df.split_official == "val"].camera_id.unique())
print("camera_id in BOTH train and val:", len(tr & vl), "of", len(tr | vl), "total")
tr_s = set(df[df.split_official == "train"].site.unique())
vl_s = set(df[df.split_official == "val"].site.unique())
print("site in BOTH train and val:", len(tr_s & vl_s), "of", len(tr_s | vl_s), "total")
print("date range:", df.datetime.min(), "to", df.datetime.max())

# per-box area stats
rows = []
for name, split, boxes in zip(df.image_name, df.split_official, df.boxes):
    for b in json.loads(boxes):
        rows.append((name, split, b[2] * b[3]))
bx = pd.DataFrame(rows, columns=["image_name", "split", "area_frac"])
bx["sqrt_area"] = np.sqrt(bx.area_frac)
print("\n=== BOX SIZE (area fraction of image) ===")
print("n boxes:", len(bx))
print(bx.area_frac.describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]).to_string())
print("\nsqrt(area_frac) quantiles:")
print(bx.sqrt_area.describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]).to_string())
tiny = bx.sqrt_area < 0.03
print(f"\nTINY boxes sqrt(area_frac)<0.03: {int(tiny.sum())} / {len(bx)} = {tiny.mean():.4f}")
print("  per-split tiny fraction:",
      {s: round(float(bx[bx.split == s].sqrt_area.lt(0.03).mean()), 4) for s in ("train", "val")})

# smallest-box positives: min box area per image
min_area = bx.groupby("image_name").area_frac.min().rename("min_area_frac").reset_index()
df2 = df.merge(min_area, on="image_name", how="left")

os.makedirs(SHEET_DIR, exist_ok=True)


def contact_sheet(sub, out_path, title, lw=1.3):
    ncols, nrows = 8, 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.1))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, sub.iterrows()):
        try:
            im = Image.open(r.image_path)
            im.thumbnail((512, 512))
            W, H = im.size
            ax.imshow(im)
            for b in json.loads(r.boxes):
                xc, yc, w, h = b
                rect = patches.Rectangle(((xc - w / 2) * W, (yc - h / 2) * H),
                                         w * W, h * H, fill=False,
                                         edgecolor="red", linewidth=lw)
                ax.add_patch(rect)
            ax.set_title(f"{r.camera_id} {r.datetime}", fontsize=5)
        except Exception as e:  # noqa: BLE001
            ax.set_title(f"ERR {e}", fontsize=5)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print("saved:", out_path)


pos_train = df2[(df2.split_official == "train") & (df2.n_boxes > 0)]
neg_all = df2[df2.n_boxes == 0]
sel_pos = pos_train.sample(48, random_state=42)
sel_neg = neg_all.sample(48, random_state=42)
sel_tiny = df2[df2.n_boxes > 0].nsmallest(48, "min_area_frac")
print("\nsmallest 48 min_area_frac range:",
      float(sel_tiny.min_area_frac.min()), "-", float(sel_tiny.min_area_frac.max()))

contact_sheet(sel_pos, os.path.join(SHEET_DIR, "train_positives_random48.jpg"),
              "pyro-sdis: 48 random TRAIN positives (red = smoke boxes)")
contact_sheet(sel_neg, os.path.join(SHEET_DIR, "negatives_random48.jpg"),
              "pyro-sdis: 48 random zero-box negatives (train+val)")
contact_sheet(sel_tiny, os.path.join(SHEET_DIR, "smallest_box_positives48.jpg"),
              "pyro-sdis: 48 positives with SMALLEST box area fraction", lw=1.0)
print("DONE")
