"""FireSpot agent: private snapshot of the dedup REFERENCE table (sha256 + imagehash.phash hex)
for all existing project material, with sha256 filled from the v4 source manifests (the v4 hash
cache stores phash-only for data/v4/raw rows) and roles refined per PROTOCOL_V5 s2.
Input : data/v5b/manifests/_cache/ref_hashes.parquet (shared, built by a sibling agent; read-only here)
Output: data/v5b/manifests/_cache/firespot_ref_hashes.parquet [path, sha256, phash, ref_source, ref_role, ref_unit]
ref_role: train | val | quarantine | dup_multinat (MNS rows that are themselves usable/pending; MNS FIgLib-family,
          Boreal heinola/ruokolahti copies and AuSmoke VAL/TEST units are re-labelled val/quarantine)."""
import json, pandas as pd
ROOT = "/local/home/tianrgua/cvproject"
r = pd.read_parquet(f"{ROOT}/data/v5b/manifests/_cache/ref_hashes.parquet")
fill = []
for src in ["multinatsmoke", "boreal", "firesense", "mmodalfire"]:
    m = pd.read_parquet(f"{ROOT}/data/v4/manifests/{src}_manifest.parquet")[["rel_path", "sha256"]]
    m["path"] = f"{ROOT}/data/v4/raw/{src}/" + m.rel_path
    fill.append(m[["path", "sha256"]])
fill = pd.concat(fill).drop_duplicates("path").set_index("path").sha256
r["sha256"] = r.sha256.where(r.sha256 != "", r.path.map(fill).fillna(""))
# refine MultiNatSmoke roles
mns = pd.read_parquet(f"{ROOT}/data/v4/manifests/multinatsmoke_manifest.parquet")
mns["path"] = f"{ROOT}/data/v4/raw/multinatsmoke/" + mns.rel_path
hold = json.load(open(f"{ROOT}/data/v4/manifests/forest_holdout_v4.json"))
aus_role = {}
for role, groups in hold["assignment"].items():
    for g in groups:
        for u in hold["groups"][g]:
            aus_role[u] = {"train": "dup_multinat", "val": "val", "test": "quarantine"}[role]
def mns_role(row):
    oc = row.origin_class; pu = str(row.parent_unit); sub = str(row.subset).lower()
    if "figlib" in sub or "figlib" in pu or oc in ("quarantine_figlib",): return "quarantine"
    if "boreal" in sub or pu.startswith("boreal"):
        if "ruokolahti" in row.rel_path.lower(): return "quarantine"
        if "heinola" in row.rel_path.lower(): return "val"
        return "train"
    if pu in aus_role: return aus_role[pu]
    if oc == "usable": return "dup_multinat"
    return "quarantine"  # dup_dfire etc. -> already in E; quarantine-as-reference is conservative
mns["role2"] = mns.apply(mns_role, axis=1)
mns["unit2"] = mns.parent_unit.astype(str)
role_map = dict(zip(mns.path, mns.role2)); unit_map = dict(zip(mns.path, mns.unit2))
is_mns = r.ref_source == "multinatsmoke"
r.loc[is_mns, "ref_role"] = r.loc[is_mns, "path"].map(role_map).fillna("dup_multinat")
r.loc[is_mns, "ref_unit"] = r.loc[is_mns, "path"].map(unit_map).fillna(r.loc[is_mns, "ref_unit"])
r.to_parquet(f"{ROOT}/data/v5b/manifests/_cache/firespot_ref_hashes.parquet", index=False)
print(r.shape, "sha empty:", (r.sha256 == "").sum(), "phash empty:", (r.phash == "").sum())
print(r.groupby(["ref_source", "ref_role"]).size().to_string())
