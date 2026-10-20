"""FireSpot vs existing material: exact sha256 + phash hamming<=6 against firespot_ref_hashes.parquet,
plus within-dataset exact/near duplicates. Output: data/v5b/manifests/_cache/firespot_dedup_hits.parquet
and firespot_dedup_summary.json."""
import json, numpy as np, pandas as pd
ROOT = "/local/home/tianrgua/cvproject"; MAXD = 6
fs = pd.read_parquet(f"{ROOT}/data/v5b/manifests/_cache/firespot_hashes.parquet")
ref = pd.read_parquet(f"{ROOT}/data/v5b/manifests/_cache/firespot_ref_hashes.parquet")
ref = ref[(ref.sha256 != "") & (ref.phash != "")].reset_index(drop=True)
mns = pd.read_parquet(f"{ROOT}/data/v4/manifests/multinatsmoke_manifest.parquet")
mns_fs_sha = set(mns.loc[mns.subset.astype(str).str.lower().str.contains("firespot"), "sha256"])
def u64(hexes): return np.array([int(h, 16) for h in hexes], dtype=np.uint64)
def popcnt(x):
    v = np.ascontiguousarray(x).view(np.uint8); return np.unpackbits(v.reshape(x.shape + (8,)), axis=-1).sum(-1)
q = u64(fs.phash); r = u64(ref.phash)
hits = []
chunk = 200
for s in range(0, len(q), chunk):
    d = popcnt(q[s:s + chunk, None] ^ r[None, :])
    qi, ri = np.nonzero(d <= MAXD)
    for a, b in zip(qi, ri):
        hits.append((s + int(a), int(b), int(d[a, b]), "phash"))
sha_ix = {}
for i, sh in enumerate(ref.sha256): sha_ix.setdefault(sh, []).append(i)
for a, sh in enumerate(fs.sha256):
    for b in sha_ix.get(sh, []): hits.append((a, b, 0, "sha256"))
H = pd.DataFrame(hits, columns=["fs_ix", "ref_ix", "hamming", "kind"])
H["fs_file"] = fs.file.values[H.fs_ix]; H["fs_sha"] = fs.sha256.values[H.fs_ix]
for c in ["path", "sha256", "ref_source", "ref_role", "ref_unit"]: H[f"ref_{c}" if not c.startswith("ref_") else c] = ref[c].values[H.ref_ix]
H["ref_is_mns_firespot"] = H.ref_path.str.contains("/FireSpot/")
H.to_parquet(f"{ROOT}/data/v5b/manifests/_cache/firespot_dedup_hits.parquet", index=False)
# per-file verdict
per = {}
for f, g in H.groupby("fs_file"):
    roles = set(g.ref_role); srcs = set(g.ref_source)
    nonfs = g[~g.ref_is_mns_firespot]
    per[f] = {"n_hits": len(g), "exact_any": bool((g.kind == "sha256").any()),
              "exact_mns_firespot": bool(((g.kind == "sha256") & g.ref_is_mns_firespot).any()),
              "phash_mns_firespot": bool(((g.kind == "phash") & g.ref_is_mns_firespot).any()),
              "hit_quarantine_or_val": bool(nonfs.ref_role.isin(["quarantine", "val"]).any()),
              "hit_other_train": bool(nonfs.ref_role.isin(["train", "dup_multinat"]).any()),
              "other_sources": ",".join(sorted(set(nonfs.ref_source))), "min_hamming": int(g.hamming.min()),
              "ref_examples": ";".join(nonfs.ref_path.str.replace(ROOT + "/", "", regex=False).head(3))}
P = pd.DataFrame.from_dict(per, orient="index")
P.to_parquet(f"{ROOT}/data/v5b/manifests/_cache/firespot_dedup_perfile.parquet")
summ = {"n_firespot": len(fs), "n_with_any_hit": len(P), "exact_mns_firespot": int(P.exact_mns_firespot.sum()),
        "phash_only_mns_firespot": int((P.phash_mns_firespot & ~P.exact_mns_firespot).sum()),
        "hit_quarantine_or_val_nonfirespot": int(P.hit_quarantine_or_val.sum()), "hit_other_train_nonfirespot": int(P.hit_other_train.sum()),
        "mns_firespot_rows_total": len(mns_fs_sha), "mns_firespot_sha_found_in_r1": int(len(mns_fs_sha & set(fs.sha256))),
        "nonfirespot_hit_sources": H[~H.ref_is_mns_firespot].groupby(["ref_source", "ref_role", "kind"]).size().to_dict()}
summ = {k: ({str(a): int(b) for a, b in v.items()} if isinstance(v, dict) else v) for k, v in summ.items()}
json.dump(summ, open(f"{ROOT}/data/v5b/manifests/_cache/firespot_dedup_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
print(P[P.hit_quarantine_or_val | P.hit_other_train][["min_hamming", "other_sources", "ref_examples"]].head(12).to_string())
