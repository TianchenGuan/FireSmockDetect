"""Build FROZEN leakage-safe splits for smoke detection experiments.

pyro-sdis: site-disjoint split (8 sites; no event ids exist, and the official
random split shares 23/40 cameras across train/val). Grouping by SITE is the
conservative choice: all cameras at a site see the same incidents.
Residual risk (documented): two sites of the same partner could in principle
observe the same large incident; not verifiable without geolocation work.

D-Fire: TRAINING-ONLY supplement. No event/camera metadata and adjacent
sequential indices are near-duplicate frames, so no leakage-safe test split
can be constructed from it. Smoke boxes kept, fire boxes dropped (single-class
model), fire-only images EXCLUDED (possible unlabeled faint smoke), 'none'
images kept as negatives. Official D-Fire test kept only as a FLAGGED
secondary eval (its own train/test split is not verified duplicate-safe).

FIgLib: evaluation only (sequence replay). Never trained on.

Outputs:
  data/yolo/dfire_smoke/            hardlinked images + rewritten labels
  data/yolo/lists/{local,pod}/*.txt frozen image lists (absolute paths)
  data/yolo/yamls/{local,pod}/*.yaml
  data/manifests/splits_frozen.json site map, counts, sha256 of every list
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path("/local/home/tianrgua/cvproject")
POD_DATA_ROOT = "/mnt/local/smoke_data"          # layout inside cluster pod
LOCAL_DATA_ROOT = str(ROOT / "data" / "yolo")    # same layout on devbox

SITE_SPLIT = {
    "train": ["brison", "serre-de-barre", "courmettes", "valbonne"],
    "val": ["cabanelle", "marguerite"],
    "test": ["ferion", "croix-augas"],
}
RATIONALE = (
    "Site-disjoint; both major partners (sdis-07, force-06) present in every "
    "split; test adds sdis-77 (croix-augas) and winter-only ferion for domain "
    "shift; brison (38% of data) must be train to keep enough positives."
)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel_from_yolo_root(abs_path: str) -> str:
    """image path relative to data/yolo/ (shared layout root)."""
    return str(Path(abs_path).resolve().relative_to((ROOT / "data" / "yolo").resolve()))


def main():
    man = pd.read_parquet(ROOT / "data/manifests/pyro_sdis_manifest.parquet")
    site_to_split = {s: k for k, v in SITE_SPLIT.items() for s in v}
    man["split_frozen"] = man["site"].map(site_to_split)
    assert man.split_frozen.notna().all(), "unmapped site!"

    # ---- D-Fire curation (train-only) ----
    dfire_raw = ROOT / "data/raw/dfire/extracted"
    out = ROOT / "data/yolo/dfire_smoke"
    (out / "images/train").mkdir(parents=True, exist_ok=True)
    (out / "labels/train").mkdir(parents=True, exist_ok=True)
    kept, excluded_fire_only, neg = [], 0, 0
    for lp in sorted((dfire_raw / "train/labels").glob("*.txt")):
        lines = [l.split() for l in lp.read_text().splitlines() if l.strip()]
        smoke = [l for l in lines if int(float(l[0])) == 0]
        fire = [l for l in lines if int(float(l[0])) == 1]
        if fire and not smoke:           # fire-only: exclude
            excluded_fire_only += 1
            continue
        ip = dfire_raw / "train/images" / (lp.stem + ".jpg")
        if not ip.exists():
            matches = list((dfire_raw / "train/images").glob(lp.stem + ".*"))
            if not matches:
                continue
            ip = matches[0]
        dst_img = out / "images/train" / ip.name
        if not dst_img.exists():
            os.link(ip, dst_img)
        # rewrite labels: smoke lines only, clip coords to [0,1]
        txt = []
        for l in smoke:
            c, x, y, w, h = 0, *map(float, l[1:5])
            x, y = min(max(x, 0), 1), min(max(y, 0), 1)
            w, h = min(w, 1.0), min(h, 1.0)
            txt.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        (out / "labels/train" / lp.name).write_text("\n".join(txt) + ("\n" if txt else ""))
        if not smoke:
            neg += 1
        kept.append(str(dst_img))
    print(f"dfire_smoke: kept {len(kept)} (neg {neg}), excluded fire-only {excluded_fire_only}")

    # ---- list files ----
    lists = {}
    for split in ["train", "val", "test"]:
        sub = man[man.split_frozen == split]
        lists[f"sdis_{split}"] = [rel_from_yolo_root(p) for p in sub.image_path]
    lists["expB_train"] = lists["sdis_train"] + [rel_from_yolo_root(p) for p in kept]
    # flagged secondary: official dfire test (not duplicate-safe vs dfire train)
    dfire_test = sorted(str(p) for p in (dfire_raw / "test/images").iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    # note: dfire test used raw (2-class) — evaluated smoke-only via eval script.

    meta = {"site_split": SITE_SPLIT, "rationale": RATIONALE, "lists": {}}
    for root_name, prefix in [("local", LOCAL_DATA_ROOT), ("pod", POD_DATA_ROOT)]:
        d = ROOT / "data/yolo/lists" / root_name
        d.mkdir(parents=True, exist_ok=True)
        for name, rels in lists.items():
            f = d / f"{name}.txt"
            f.write_text("\n".join(f"{prefix}/{r}" for r in rels) + "\n")
            if root_name == "local":
                meta["lists"][name] = {"n": len(rels), "sha256": sha256(f)}
    (ROOT / "data/yolo/lists/local/dfire_test_raw.txt").write_text("\n".join(dfire_test) + "\n")

    # ---- dataset yamls ----
    for root_name, prefix in [("local", LOCAL_DATA_ROOT), ("pod", POD_DATA_ROOT)]:
        yd = ROOT / "data/yolo/yamls" / root_name
        yd.mkdir(parents=True, exist_ok=True)
        for exp, train_list in [("expA", "sdis_train"), ("expB", "expB_train")]:
            (yd / f"{exp}.yaml").write_text(
                f"path: {prefix}\n"
                f"train: {prefix}/lists/{root_name}/{train_list}.txt\n"
                f"val: {prefix}/lists/{root_name}/sdis_val.txt\n"
                f"test: {prefix}/lists/{root_name}/sdis_test.txt\n"
                "nc: 1\nnames: [smoke]\n")

    # ---- per-split sanity ----
    stats = man.groupby("split_frozen").agg(
        n=("image_name", "count"), pos=("n_boxes", lambda s: int((s > 0).sum())),
        neg=("n_boxes", lambda s: int((s == 0).sum())),
        sites=("site", "nunique"), cameras=("camera_id", "nunique")).to_dict("index")
    meta["pyro_sdis_stats"] = stats
    meta["dfire_train_supplement"] = {"kept": len(kept), "negatives": neg,
                                      "excluded_fire_only": excluded_fire_only}
    meta["frozen"] = True
    meta["note"] = ("val = threshold/model selection ONLY; test = final report ONLY; "
                    "FIgLib replay = early-detection eval ONLY (never trained on).")
    (ROOT / "data/manifests/splits_frozen.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(stats, indent=2))
    print("FROZEN ->", ROOT / "data/manifests/splits_frozen.json")


if __name__ == "__main__":
    main()
