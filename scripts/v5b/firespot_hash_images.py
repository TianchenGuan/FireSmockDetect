"""Hash every FireSpot-CNX-r1 image: sha256, imagehash.phash (64-bit hex), width, height, bytes.
Output: data/v5b/manifests/_cache/firespot_hashes.parquet"""
import os, hashlib, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import imagehash
ROOT = "/local/home/tianrgua/cvproject"
IMG = f"{ROOT}/data/v5b/raw/firespot/dataset/images"
def work(fn):
    p = os.path.join(IMG, fn)
    try:
        b = open(p, "rb").read()
        with Image.open(p) as im:
            w, h = im.size; ph = str(imagehash.phash(im)); mode = im.mode
        return (fn, hashlib.sha256(b).hexdigest(), ph, w, h, len(b), mode, "")
    except Exception as e:
        return (fn, "", "", 0, 0, 0, "", f"{type(e).__name__}: {e}")
fns = sorted(f for f in os.listdir(IMG) if f.lower().endswith((".jpg", ".jpeg", ".png")))
with ProcessPoolExecutor(96) as ex:
    rows = list(ex.map(work, fns, chunksize=16))
df = pd.DataFrame(rows, columns=["file", "sha256", "phash", "width", "height", "bytes", "mode", "err"])
df.to_parquet(f"{ROOT}/data/v5b/manifests/_cache/firespot_hashes.parquet", index=False)
print(df.shape, "errors:", (df.err != "").sum()); print(df.err[df.err != ""].head().tolist())
print("dims:", df.groupby(["width", "height"]).size().sort_values(ascending=False).head(8).to_dict())
print("modes:", df["mode"].value_counts().to_dict(), "bytes total GB:", round(df.bytes.sum() / 1e9, 2))
print("unique sha:", df.sha256.nunique(), "unique phash:", df.phash.nunique())
