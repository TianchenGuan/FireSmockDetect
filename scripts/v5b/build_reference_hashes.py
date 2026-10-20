#!/usr/bin/env python
"""Build data/v5b/manifests/_cache/reference_hashes.parquet

One row per EXISTING image in the project's reference material with
[ref_source, ref_path, sha256, phash, parent_unit, role].  Used by the v5b data
agents to dedup new acquisitions (exact sha256 + phash hamming<=6, same
convention as scripts/v4/build_corpus.py: imagehash.phash, 64-bit hex).
Roles follow PROTOCOL_V5 §2 / splits_frozen / aerial_split_v4 / forest_holdout_v4:
  train | val | quarantine_final_test | quarantine_holdout | quarantine_figlib
Reuses data/v4/manifests/_cache/hash_cache.parquet; computes only missing phash.
"""
import json, os, sys, hashlib
import pandas as pd, numpy as np
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import imagehash

ROOT = "/local/home/tianrgua/cvproject"
OUT = f"{ROOT}/data/v5b/manifests/_cache/reference_hashes.parquet"

def _ph(p):
    try:
        with Image.open(p) as im:
            return p, str(imagehash.phash(im))
    except Exception as e:  # noqa
        return p, ""

def _sha(p):
    try:
        return p, hashlib.sha256(open(p, "rb").read()).hexdigest()
    except Exception:
        return p, ""

def main():
    os.chdir(ROOT)
    cache = pd.read_parquet("data/v4/manifests/_cache/hash_cache.parquet")
    cache["path"] = cache["path"].str.replace(ROOT + "/", "", regex=False)
    c_sha = dict(zip(cache.path, cache.sha256)); c_ph = dict(zip(cache.path, cache.phash))
    rows = []
    # boreal
    b = pd.read_parquet("data/v4/manifests/boreal_manifest.parquet")
    role = {"burn_heinola": "val", "burn_ruokolahti": "quarantine_final_test"}
    for r in b.itertuples():
        rows.append(("boreal", f"data/v4/raw/boreal/{r.rel_path}", r.sha256, r.parent_unit, role.get(r.parent_unit, "train")))
    # multinatsmoke
    fh = json.load(open("data/v4/manifests/forest_holdout_v4.json"))
    aus_test, aus_val = set(fh["test"]), set(fh["val"])
    m = pd.read_parquet("data/v4/manifests/multinatsmoke_manifest.parquet")
    for r in m.itertuples():
        rp = r.rel_path or ""
        if r.origin_class == "quarantine_figlib" or "FIgLib" in rp: ro = "quarantine_figlib"
        elif r.parent_unit in aus_test: ro = "quarantine_holdout"
        elif r.parent_unit in aus_val: ro = "val"
        elif "heinola" in rp: ro = "val"
        elif "ruokolahti" in rp: ro = "quarantine_final_test"
        else: ro = "train"
        rows.append(("multinatsmoke", f"data/v4/raw/multinatsmoke/{rp}", r.sha256 or "", f"multinat/{r.parent_unit}", ro))
    # firesense frames (from cache) + mmodalfire
    for p in cache.path[cache.path.str.startswith("data/v4/raw/firesense/")]:
        rows.append(("firesense", p, c_sha.get(p, ""), "firesense/" + p.split("/")[-2], "train"))
    mm = pd.read_parquet("data/v4/manifests/mmodalfire_manifest.parquet")
    for r in mm.itertuples():
        rows.append(("mmodalfire", r.rel_path, r.sha256, f"mmodalfire/{r.parent_unit}", "train"))
    # pyro_sdis by site split
    sp = json.load(open("data/manifests/splits_frozen.json"))["site_split"]
    site_role = {s: ("quarantine_final_test" if k == "test" else k) for k, v in sp.items() for s in v}
    ps = pd.read_parquet("data/manifests/pyro_sdis_manifest.parquet")
    for r in ps.itertuples():
        p = r.image_path.replace(ROOT + "/", "")
        rows.append(("pyro_sdis", p, c_sha.get(p, ""), f"sdis/{r.site}", site_role.get(r.site, "train")))
    # dfire
    d = pd.read_parquet("data/manifests/dfire_manifest.parquet")
    for r in d.itertuples():
        rows.append(("dfire", r.image_path, c_sha.get(r.image_path, ""), f"dfire/{r.split_official}", "quarantine_holdout" if r.split_official == "test" else "train"))
    # figlib subset + lockbox
    f = pd.read_parquet("data/manifests/figlib_subset_manifest.parquet")
    for r in f.itertuples():
        rows.append(("figlib_subset", r.frame_path, c_sha.get(r.frame_path, ""), f"figlib/{r.event_id}", "quarantine_figlib"))
    for p in cache.path[cache.path.str.startswith("data/v3/figlib_lockbox/")]:
        rows.append(("figlib_lockbox", p, c_sha.get(p, ""), "figlib_lockbox/" + p.split("/")[3], "quarantine_figlib"))
    df = pd.DataFrame(rows, columns=["ref_source", "ref_path", "sha256", "parent_unit", "role"])
    df["phash"] = df.ref_path.map(c_ph).fillna("")
    df["exists"] = df.ref_path.map(os.path.exists)
    need_ph = df.ref_path[(df.phash == "") & df.exists].tolist()
    need_sha = df.ref_path[(df.sha256 == "") & df.exists].tolist()
    print(f"rows={len(df)} missing_files={(~df.exists).sum()} need_phash={len(need_ph)} need_sha={len(need_sha)}", flush=True)
    with ProcessPoolExecutor(96) as ex:
        ph = dict(ex.map(_ph, need_ph, chunksize=64))
        sh = dict(ex.map(_sha, need_sha, chunksize=64))
    df.loc[df.phash == "", "phash"] = df.ref_path[df.phash == ""].map(ph).fillna("")
    df.loc[df.sha256 == "", "sha256"] = df.ref_path[df.sha256 == ""].map(sh).fillna("")
    tmp = OUT + ".tmp"; df.to_parquet(tmp, index=False); os.replace(tmp, OUT)
    print(df.groupby(["ref_source", "role"]).size().to_string())
    print("phash empty:", (df.phash == "").sum(), "sha empty:", (df.sha256 == "").sum())

if __name__ == "__main__":
    main()
