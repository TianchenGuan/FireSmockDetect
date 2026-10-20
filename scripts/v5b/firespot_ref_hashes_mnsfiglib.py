"""Add MultiNatSmoke FIgLib-family / dup rows (origin_class != usable, never phashed in v4) to the
FireSpot private reference table as role=quarantine so FIgLib-derived material is fully covered."""
import os, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import imagehash
ROOT = "/local/home/tianrgua/cvproject"
OUT = f"{ROOT}/data/v5b/manifests/_cache/firespot_ref_hashes.parquet"
def ph(p):
    try:
        with Image.open(p) as im: return (p, str(imagehash.phash(im)))
    except Exception: return (p, "")
ref = pd.read_parquet(OUT)
mns = pd.read_parquet(f"{ROOT}/data/v4/manifests/multinatsmoke_manifest.parquet")
mns = mns[mns.image_present.astype(bool)]
mns["path"] = f"{ROOT}/data/v4/raw/multinatsmoke/" + mns.rel_path
add = mns[~mns.path.isin(set(ref.path))]
print("to add:", len(add), add.origin_class.value_counts().to_dict(), flush=True)
with ProcessPoolExecutor(96) as ex:
    res = dict(ex.map(ph, add.path.tolist(), chunksize=64))
add = add.assign(phash=add.path.map(res), ref_source="multinatsmoke", ref_role="quarantine",
                 ref_unit=add.parent_unit.astype(str))[["path", "sha256", "phash", "ref_source", "ref_role", "ref_unit"]]
out = pd.concat([ref, add], ignore_index=True).drop_duplicates("path")
out.to_parquet(OUT, index=False)
print(out.shape, "phash empty:", (out.phash == "").sum(), out.groupby(["ref_source", "ref_role"]).size().to_dict())
