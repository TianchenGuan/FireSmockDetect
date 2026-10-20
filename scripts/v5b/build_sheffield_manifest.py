#!/usr/bin/env python
"""Build data/v5b/manifests/sheffield_uav_manifest.parquet + YOLO export (class 0 = smoke).

Source: Sheffield ORDA 33080681 "Aerial Imagery Dataset for Wildfire Detection using UAVs"
(Roboflow export, 1,336 DJI H20-style video frames, classes 0=fire 1=smoke).
Grouping: parent_unit = DJI recording (W and Z streams of the same recording are
co-temporal views of one scene -> same unit). All 6 recordings are one burn session
(2022-07-21 14:07-14:19 local) -> see CARD limitation.
Dedup: exact sha256 + phash hamming<=6 vs data/v5b/manifests/_cache/reference_hashes.parquet.
"""
import os, re, json, glob, collections
import numpy as np, pandas as pd

ROOT = "/local/home/tianrgua/cvproject"
RAW = f"{ROOT}/data/v5b/raw/sheffield_uav/extracted"
REF = f"{ROOT}/data/v5b/manifests/_cache/reference_hashes.parquet"
HASHES = f"{ROOT}/data/v5b/manifests/_cache/sheffield_uav_hashes.parquet"
OUT = f"{ROOT}/data/v5b/manifests/sheffield_uav_manifest.parquet"
YOLO = f"{ROOT}/data/v5b/yolo/sheffield_uav"
PAT = re.compile(r"^(DJI_(\d{14})_(\d{4})_([WZ]))_MP4-(\d+)_jpg\.rf\.[A-Za-z0-9]+\.jpg$")
SMOKE, FIRE = 1, 0
# agent (automated) visual inspection of the 30 empty-label frames, 2026-10-02: smoke visible in these 6
AGENT_SMOKE_VISIBLE = {"DJI_20220721141130_0005_W_MP4-0029", "DJI_20220721140720_0003_Z_MP4-0042",
                       "DJI_20220721140720_0003_Z_MP4-0128", "DJI_20220721141011_0004_Z_MP4-0027",
                       "DJI_20220721141419_0006_W_MP4-0132", "DJI_20220721141419_0006_Z_MP4-0084"}
# proposed split by WHOLE recording (6 units): 4 train / 1 val / 1 holdout (last recording, largest)
UNIT_SPLIT = {"rec_0003": "train", "rec_0004": "train", "rec_0005": "train", "rec_0006": "train",
              "rec_0007": "val", "rec_0008": "holdout"}

def popcount_hits(q_hex, r_hex, maxd=6):
    q = np.array([int(h, 16) for h in q_hex], dtype=np.uint64); r = np.array([int(h, 16) for h in r_hex], dtype=np.uint64)
    hits = {}
    for s in range(0, len(q), 256):
        d = np.bitwise_count(q[s:s + 256, None] ^ r[None, :]) if hasattr(np, "bitwise_count") else None
        if d is None:
            x = (q[s:s + 256, None] ^ r[None, :]).view(np.uint8).reshape(-1, len(r), 8)
            d = np.unpackbits(x, axis=-1).sum(-1)
        qi, ri = np.nonzero(d <= maxd)
        for a, b in zip(qi, ri):
            k = s + int(a)
            if k not in hits or d[a, b] < hits[k][1]: hits[k] = (int(b), int(d[a, b]))
    return hits

def main():
    h = pd.read_parquet(HASHES); ref = pd.read_parquet(REF)
    ref = ref[(ref.sha256 != "")]
    rows = []
    for r in h.itertuples():
        fn = os.path.basename(r.rel_path); m = PAT.match(fn); assert m, fn
        stream, ts, rec, cam, idx = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
        split_off = r.rel_path.split("/")[0]
        lf = f"{RAW}/{split_off}/labels/{fn[:-4]}.txt"
        ns = nf = 0; lines = []
        for l in open(lf).read().splitlines():
            p = l.split()
            if len(p) < 5: continue
            k = int(float(p[0])); lines.append((k, *map(float, p[1:5])))
            ns += k == SMOKE; nf += k == FIRE
        stem = fn.split("_jpg.rf.")[0]
        rows.append(dict(source="sheffield_uav", rel_path=r.rel_path, sha256=r.sha256, phash=r.phash,
                         parent_unit=f"sheffield_uav/rec_{rec}", session_unit="sheffield_uav/burn_20220721",
                         video_stream=stream, recording_start_local=f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}",
                         camera=("wide" if cam == "W" else "zoom"), frame_idx=idx, official_split=split_off,
                         width=r.width, height=r.height, label_type="box", has_smoke_label=ns > 0, has_fire_label=nf > 0,
                         is_explicit_negative=(ns == 0 and nf == 0), n_smoke_boxes=ns, n_fire_boxes=nf,
                         ignore_region_candidate=(ns == 0 and nf > 0),
                         agent_visual_flag=("smoke_visible_in_empty_label" if stem in AGENT_SMOKE_VISIBLE else ""),
                         annotation_origin="author", licence="CC BY 4.0", permission_status="confirmed",
                         boxes=json.dumps(lines)))
    df = pd.DataFrame(rows)
    # ---- dedup vs reference
    sha_map = ref.drop_duplicates("sha256").set_index("sha256")
    exact = df.sha256.map(sha_map.ref_source); exact_role = df.sha256.map(sha_map.role); exact_path = df.sha256.map(sha_map.ref_path)
    refp = ref[ref.phash != ""].reset_index(drop=True)
    hits = popcount_hits(df.phash.tolist(), refp.phash.tolist(), 6)
    near_src = pd.Series([refp.ref_source.iloc[hits[i][0]] if i in hits else None for i in range(len(df))])
    near_role = pd.Series([refp.role.iloc[hits[i][0]] if i in hits else None for i in range(len(df))])
    near_path = pd.Series([refp.ref_path.iloc[hits[i][0]] if i in hits else None for i in range(len(df))])
    near_d = pd.Series([hits[i][1] if i in hits else None for i in range(len(df))])
    def dup_class(i):
        if isinstance(exact.iloc[i], str): return "dup_multinat" if exact.iloc[i] == "multinatsmoke" else f"dup_{exact.iloc[i]}"
        if isinstance(near_src.iloc[i], str): return "dup_multinat" if near_src.iloc[i] == "multinatsmoke" else f"neardup_{near_src.iloc[i]}"
        return "unique"
    df["dup_class"] = [dup_class(i) for i in range(len(df))]
    df["dup_detail"] = [(f"exact:{exact_path.iloc[i]}" if isinstance(exact.iloc[i], str) else (f"phash<= {near_d.iloc[i]}:{near_path.iloc[i]}" if isinstance(near_src.iloc[i], str) else "")) for i in range(len(df))]
    inherited = [(exact_role.iloc[i] if isinstance(exact.iloc[i], str) else (near_role.iloc[i] if isinstance(near_src.iloc[i], str) else None)) for i in range(len(df))]
    df["inherited_role"] = [x or "" for x in inherited]
    # within-source near-dups (same phash as an earlier frame)
    seen = set(); w = []
    for ph in df.phash: w.append(ph in seen); seen.add(ph)
    df["within_source_neardup"] = w
    # ---- proposed split
    def split(i):
        ir = df.inherited_role.iloc[i]
        if ir.startswith("quarantine"): return "quarantine"
        if ir == "val": return "val"
        return UNIT_SPLIT[df.parent_unit.iloc[i].split("/")[-1]]
    df["proposed_split"] = [split(i) for i in range(len(df))]
    cols = ["source", "rel_path", "sha256", "phash", "parent_unit", "label_type", "has_smoke_label", "has_fire_label",
            "is_explicit_negative", "n_smoke_boxes", "annotation_origin", "licence", "permission_status", "dup_class", "proposed_split",
            "session_unit", "video_stream", "recording_start_local", "camera", "frame_idx", "official_split", "width", "height",
            "n_fire_boxes", "ignore_region_candidate", "agent_visual_flag", "dup_detail", "inherited_role", "within_source_neardup", "boxes"]
    df = df[cols].sort_values(["video_stream", "frame_idx"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    # ---- YOLO export (class 0 = smoke only); fire-only + agent-flagged empties EXCLUDED, listed in sidecars
    import shutil
    if os.path.exists(YOLO): shutil.rmtree(YOLO)
    side = {"ignore_region_candidates_fire_only": [], "needs_human_review_empty_label_smoke_visible": [], "excluded_quarantine": []}
    n_exp = collections.Counter()
    for r in df.itertuples():
        if r.proposed_split == "quarantine": side["excluded_quarantine"].append(r.rel_path); continue
        if r.ignore_region_candidate: side["ignore_region_candidates_fire_only"].append(r.rel_path); continue
        if r.agent_visual_flag: side["needs_human_review_empty_label_smoke_visible"].append(r.rel_path); continue
        sp = r.proposed_split; os.makedirs(f"{YOLO}/images/{sp}", exist_ok=True); os.makedirs(f"{YOLO}/labels/{sp}", exist_ok=True)
        fn = os.path.basename(r.rel_path)
        os.symlink(f"{RAW}/{r.rel_path}", f"{YOLO}/images/{sp}/{fn}")
        with open(f"{YOLO}/labels/{sp}/{fn[:-4]}.txt", "w") as f:
            for k, x, y, w_, h_ in json.loads(r.boxes):
                if k == SMOKE: f.write(f"0 {x:.6f} {y:.6f} {w_:.6f} {h_:.6f}\n")
        n_exp[(sp, "neg" if r.is_explicit_negative else "pos")] += 1
    for k, v in side.items():
        open(f"{YOLO}/{k}.txt", "w").write("\n".join(v) + ("\n" if v else ""))
    open(f"{YOLO}/data.yaml", "w").write(f"path: {YOLO}\ntrain: images/train\nval: images/val\ntest: images/holdout\nnc: 1\nnames: ['smoke']\n")
    print("manifest rows", len(df)); print(df.dup_class.value_counts().to_dict()); print(df.proposed_split.value_counts().to_dict())
    print("inherited roles:", df.inherited_role.value_counts().to_dict()); print("yolo export:", dict(n_exp)); print({k: len(v) for k, v in side.items()})
    print(df.groupby("parent_unit").agg(n=("rel_path", "size"), smoke=("has_smoke_label", "sum"), neg=("is_explicit_negative", "sum"), split=("proposed_split", "first")).to_string())

if __name__ == "__main__":
    main()
