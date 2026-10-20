"""Build data/v5b/manifests/firespot_manifest.parquet for FireSpot-CNX-r1 (DATA_COMMON schema + extras).
Grouping: parent_unit = site (first filename token, 5 sites) — verified by phash: 0 cross-site near-dups,
76 same-site cross-session near-dup pairs and 9 exact-dup groups spanning sessions => sessions are NOT independent.
proposed_split by SITE: 01,02,05 -> train; 03 -> val; 04 -> holdout (author-labelled final holdout).
permission_status = pending for every row (no licence text in repo or archive)."""
import os, json, re, pandas as pd, numpy as np
ROOT = "/local/home/tianrgua/cvproject"; RAW = f"{ROOT}/data/v5b/raw/firespot"
C = f"{ROOT}/data/v5b/manifests/_cache"
fs = pd.read_parquet(f"{C}/firespot_hashes.parquet"); fs["stem"] = fs.file.str[:-4]
H = pd.read_parquet(f"{C}/firespot_dedup_hits.parquet")
mns = pd.read_parquet(f"{ROOT}/data/v4/manifests/multinatsmoke_manifest.parquet")
m = mns[mns.subset.astype(str).str.lower().str.contains("firespot")].copy()
m["stem"] = m.rel_path.str.split("/").str[-1].str.replace(r"\.(jpg|png|jpeg)$", "", regex=True)
mns_by_stem = m.set_index("stem")
# labels
nbox, boxes_oob = {}, {}
for s in fs.stem:
    n = 0; oob = 0
    for ln in open(f"{RAW}/dataset/labels/{s}.txt"):
        p = ln.split()
        if len(p) == 5:
            n += 1; xc, yc, w, h = map(float, p[1:])
            if xc - w / 2 < 0 or yc - h / 2 < 0 or xc + w / 2 > 1.0001 or yc + h / 2 > 1.0001: oob += 1
    nbox[s] = n; boxes_oob[s] = oob
fs["n_smoke_boxes"] = fs.stem.map(nbox); fs["n_boxes_out_of_range"] = fs.stem.map(boxes_oob)
fs["label_suffix"] = fs.stem.str[-1]
assert ((fs.label_suffix == "1") == (fs.n_smoke_boxes > 0)).all(), "suffix/label mismatch"
tok = fs.stem.str.split("-")
fs["site"] = tok.str[0]; fs["session_unit"] = tok.str[0] + "-" + tok.str[1]; fs["camera_pos"] = tok.str[2]; fs["seq_no"] = tok.str[3]
fs["camera_pos"] = fs.camera_pos.replace({"2": "02"})
# MNS overlap (same-stem phash match <=6 to the MNS FireSpot copy)
hm = H[H.ref_is_mns_firespot].copy(); hm["ref_stem"] = hm.ref_path.str.split("/").str[-1].str.replace(r"\.(jpg|png|jpeg)$", "", regex=True); hm["fs_stem"] = hm.fs_file.str[:-4]
same = hm[hm.ref_stem == hm.fs_stem]
in_mns = set(same.fs_stem)
# MNS stems absent from r1 (renamed between releases): an r1 image at hamming 0 to such a stem is the same picture
absent = set(m.stem) - set(fs.stem)
renamed = hm[(hm.ref_stem.isin(absent)) & (hm.hamming == 0) & (~hm.fs_stem.isin(in_mns))]
renamed_map = dict(zip(renamed.fs_stem, renamed.ref_stem)); in_mns |= set(renamed_map)
print("MNS stems absent from r1:", len(absent), " r1 images matched at hamming 0 to them (renamed copies):", len(renamed_map), " absent stems still unmatched:", len(absent - set(renamed.ref_stem)))
scene_mns = set(hm[hm.ref_stem != hm.fs_stem].fs_stem) - in_mns
# internal dups
fs = fs.sort_values("stem").reset_index(drop=True)
first_sha = fs.drop_duplicates("sha256").set_index("sha256").stem
fs["canonical_stem"] = fs.sha256.map(first_sha)
ph_lab = fs.groupby("phash").label_suffix.agg(lambda s: len(set(s)))
def dup(r):
    flags = []
    if r.stem != r.canonical_stem: flags.append("internal_exact_dup")
    if ph_lab[r.phash] > 1: flags.append("internal_neardup_label_conflict")
    if r.stem in in_mns:
        flags.append("dup_multinat_conflict" if r.label_suffix == "0" else "dup_multinat")
        if r.stem in renamed_map: flags.append("mns_renamed:" + renamed_map[r.stem])
    elif r.stem in scene_mns: flags.append("scene_neardup_mns")
    primary = "internal_exact_dup" if "internal_exact_dup" in flags else ("dup_multinat_conflict" if "dup_multinat_conflict" in flags else ("dup_multinat" if "dup_multinat" in flags else "none"))
    return primary, ";".join(flags)
dd = fs.apply(dup, axis=1, result_type="expand"); fs["dup_class"] = dd[0]; fs["dup_flags"] = dd[1]
fs.loc[fs.stem == "03-02-04-017-1", "dup_flags"] = (fs.loc[fs.stem == "03-02-04-017-1", "dup_flags"] + ";phash_false_positive_vs_pyro_sdis_train(visually_distinct)").str.strip(";")
split_map = {"01": "train", "02": "train", "05": "train", "03": "val", "04": "holdout"}
fs["proposed_split"] = fs.site.map(split_map)
fs["mns_rel_path"] = fs.stem.map(lambda s: mns_by_stem.rel_path.get(s, "")); fs["mns_split_official"] = fs.stem.map(lambda s: mns_by_stem.split_official.get(s, ""))
insp = open(f"{C}/../_inspect/firespot_inspected_list.txt").read().split(); insp = set(x for x in insp if x not in ("POS", "NEG"))
diag = pd.read_parquet(f"{C}/firespot_sanity_preds.parquet")
fs["visually_inspected_sheet"] = fs.stem.isin(insp); fs["in_sanity_diag_sample"] = fs.stem.isin(set(diag.stem))
out = pd.DataFrame({
    "source": "firespot", "rel_path": "dataset/images/" + fs.file, "sha256": fs.sha256, "phash": fs.phash,
    "parent_unit": "firespot_site_" + fs.site, "label_type": "box", "has_smoke_label": fs.n_smoke_boxes > 0, "has_fire_label": False,
    "is_explicit_negative": fs.n_smoke_boxes == 0, "n_smoke_boxes": fs.n_smoke_boxes.astype(int), "annotation_origin": "author",
    "licence": "UNSPECIFIED — no LICENSE in github.com/Biometrix-4/FireSpot-CNX nor in FireSpot-CNX-r1.zip; README only asks for citation (iSAI-NLP 2023, doi:10.1109/iSAI-NLP60301.2023.10354807); 'CC BY 4.0' appears only in the MultiNatSmoke README (aggregator claim, not evidence)",
    "permission_status": "pending", "dup_class": fs.dup_class, "proposed_split": fs.proposed_split,
    # extras
    "session_unit": "firespot_sess_" + fs.session_unit, "camera_pos": fs.camera_pos, "seq_no": fs.seq_no, "width": fs.width, "height": fs.height, "bytes": fs.bytes,
    "n_boxes_out_of_range": fs.n_boxes_out_of_range, "dup_flags": fs.dup_flags, "canonical_stem": fs.canonical_stem,
    "mns_rel_path": fs.stem.map(lambda x: mns_by_stem.rel_path.get(renamed_map.get(x, x), "")), "mns_split_official": fs.mns_split_official, "visually_inspected_sheet": fs.visually_inspected_sheet, "in_sanity_diag_sample": fs.in_sanity_diag_sample,
    "archive": "FireSpot-CNX-r1.zip sha256=143fefded0860a94f943652dbe8dc70d9f6624794df98038d3655fc616062e67 (Google Drive id 1s6tF48CL87TCYJQMyD6cvE0wW9UkPkjQ, linked from repo README)",
})
out.to_parquet(f"{ROOT}/data/v5b/manifests/firespot_manifest.parquet", index=False)
out.drop(columns=["licence", "archive"]).to_csv(f"{ROOT}/data/v5b/manifests/firespot_manifest.csv.gz", index=False)
print(out.shape)
print("split x label:\n", pd.crosstab(out.proposed_split, out.is_explicit_negative.map({True: "neg", False: "pos"}), margins=True).to_string())
print("units per split:", out.groupby("proposed_split").parent_unit.nunique().to_dict(), " sessions per split:", out.groupby("proposed_split").session_unit.nunique().to_dict())
print("dup_class:", out.dup_class.value_counts().to_dict())
print("dup_flags:", out.dup_flags.replace("", "none").str.split(";").explode().value_counts().to_dict())
print("boxes total:", int(out.n_smoke_boxes.sum()), "images w/ OOB boxes:", int((out.n_boxes_out_of_range > 0).sum()))
print("camera_pos:", out.camera_pos.value_counts().to_dict())
print("holdout site images in diag sample:", int((out.in_sanity_diag_sample & (out.proposed_split == "holdout")).sum()))
