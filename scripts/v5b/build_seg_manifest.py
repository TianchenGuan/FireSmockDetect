"""V5b thin-smoke segmentation: build the training/validation manifest and the
one-shot staging tar for the cluster (DATA_TARS pattern).

Eligibility (PROTOCOL_V5 §2 + V5b brief):
  TRAIN  = Boreal burns evo + karkkila only:
           Subset-C SAM masks (model-assisted, train+valid splits of the authors),
           Subset-C manual masks (human, authors' test split),
           Subset-A Empty-Images karkkila (author-labelled smoke-free -> all-zero mask).
  VAL    = Boreal heinola only (aerial VAL, never training): Subset-C SAM masks,
           manual masks, and the 46 author-negative frames (+ their boxes for
           mask->box recall).
  NEVER  = ruokolahti (historical final test), FIgLib / SmokeSeg / FoSp material.
The 5 unlabeled evo frames (no mask, smoke visible) are EXCLUDED (not negatives).
Provenance is recorded per row; nothing is converted between label types except
the explicit all-zero mask for author-labelled negative frames (named below).
"""
from __future__ import annotations
import hashlib, json, os, re, sys, tarfile, time
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/v4/raw/boreal/Boreal-Forest-Fire"
C = RAW / "Boreal-Forest-Fire-Subset-C"
A = RAW / "Boreal-Forest-Fire-Subset-A"
OUT = ROOT / "data/v5b/manifests"
STAGE = ROOT / "data/v5b/stage/seg_v5b"          # tar root (dir name inside tar)
TAR = ROOT / "data/v5b/stage/seg_v5b.tar"
R50 = ROOT / "models/v5b/seg/pretrained/resnet50-11ad3fa6.pth"
TRAIN_BURNS = {"evo": "burn_evo", "karkkila": "burn_karkkila"}
VAL_BURNS = {"heinola": "burn_heinola"}


def burn_of(stem):
    m = re.match(r"(evo|heinola|karkkila|ruokolahti)", stem)
    return m.group(1) if m else None


def flight_of(stem):
    return re.sub(r"_frame\d+$", "", stem)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_boxes(p):
    out = []
    if not p.exists():
        return out
    for ln in p.read_text().splitlines():
        t = ln.split()
        if len(t) >= 5:
            cx, cy, w, h = map(float, t[1:5])
            out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return out


def main():
    man = pd.read_parquet(ROOT / "data/v4/manifests/boreal_manifest.parquet")
    sha_by_rel = dict(zip(man.rel_path, man.sha256))
    rows = []
    for split in ("train", "valid", "test"):
        for f in sorted(os.listdir(C / "images" / split)):
            stem = Path(f).stem
            b = burn_of(stem)
            if b in TRAIN_BURNS:
                role = "train"
            elif b in VAL_BURNS:
                role = "val"
            else:
                continue  # ruokolahti: never shipped
            rel = f"Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-C/images/{split}/{f}"
            sam = C / "sam_masks" / split / f"{stem}.png"
            manual = C / "manual_masks" / split / f"{stem}.png"
            lab = C / "labels" / split / f"{stem}.txt"
            boxes = read_boxes(lab)
            if manual.exists():
                mask, prov = manual, "manual_human"
            elif sam.exists():
                mask, prov = sam, "sam_model_assisted"
            elif lab.exists() and not boxes:
                mask, prov = None, "author_negative_empty"   # verified smoke-free frame
            else:
                # unlabeled positive (no label file, no mask): exclude, never a negative
                rows.append(dict(id=stem, role="excluded", burn=b, flight=flight_of(stem),
                                 src_split=split, provenance="unlabeled_positive",
                                 note="smoke visible, no label/mask (card) -> excluded"))
                continue
            with Image.open(C / "images" / split / f) as im:
                w, h = im.size
            rows.append(dict(id=stem, role=role, burn=b, flight=flight_of(stem), src_split=split,
                             provenance=prov, has_mask=mask is not None,
                             is_negative=mask is None, boxes_xyxy_norm=boxes,
                             image=f"images/{f}", mask=(f"masks/{stem}.png" if mask else None),
                             orig_w=w, orig_h=h, src_rel_path=rel, src_sha256=sha_by_rel.get(rel),
                             src_image_path=str(C / "images" / split / f),
                             src_mask_path=(str(mask) if mask else None)))
    # Subset-A Empty-Images: karkkila (train burn) -> train negatives; heinola -> val negatives
    # already covered by Subset-C's 46 heinola empties (same frames, downsampled) -> skip A heinola
    for f in sorted(os.listdir(A / "Empty-Images")):
        stem = Path(f).stem
        b = burn_of(stem)
        if b != "karkkila":
            continue
        rel = f"Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-A/Empty-Images/{f}"
        rows.append(dict(id=stem, role="train", burn=b, flight=flight_of(stem), src_split="A_empty",
                         provenance="author_negative_empty", has_mask=False, is_negative=True,
                         boxes_xyxy_norm=[], image=f"images/{f}", mask=None,
                         orig_w=1920, orig_h=1080, src_rel_path=rel, src_sha256=sha_by_rel.get(rel),
                         src_image_path=str(A / "Empty-Images" / f), src_mask_path=None,
                         note="Subset-A 3840x2160 downsampled to 1920x1080 at staging"))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    kept = df[df.role != "excluded"].copy()
    assert not kept.burn.isin(["ruokolahti"]).any()
    assert (kept[kept.role == "train"].burn.isin(["evo", "karkkila"])).all()
    assert (kept[kept.role == "val"].burn == "heinola").all()
    # stage files
    (STAGE / "images").mkdir(parents=True, exist_ok=True)
    (STAGE / "masks").mkdir(parents=True, exist_ok=True)
    for r in kept.itertuples():
        dst = STAGE / r.image
        if not dst.exists():
            if r.src_split == "A_empty":
                with Image.open(r.src_image_path) as im:
                    im.convert("RGB").resize((1920, 1080), Image.LANCZOS).save(dst, quality=92)
            else:
                os.link(r.src_image_path, dst) if False else Path(dst).write_bytes(Path(r.src_image_path).read_bytes())
        if r.mask:
            dm = STAGE / r.mask
            if not dm.exists():
                dm.write_bytes(Path(r.src_mask_path).read_bytes())
    (STAGE / "resnet50-11ad3fa6.pth").write_bytes(R50.read_bytes())
    kept.to_json(STAGE / "manifest.json", orient="records", indent=0)
    kept.to_json(OUT / "seg_v5b_manifest.json", orient="records", indent=0)
    df.to_csv(OUT / "seg_v5b_rows_all.csv", index=False)
    prov = kept.groupby(["role", "burn", "provenance"]).size().rename("n_frames").reset_index()
    prov.to_csv(OUT / "seg_v5b_provenance_table.csv", index=False)
    print(prov.to_string())
    print("excluded:", (df.role == "excluded").sum())
    with tarfile.open(TAR, "w") as tf:
        tf.add(STAGE, arcname="seg_v5b")
    info = dict(tar=str(TAR), sha256=sha(TAR), bytes=TAR.stat().st_size,
                n_train=int((kept.role == "train").sum()), n_val=int((kept.role == "val").sum()),
                resnet50_sha256=sha(R50), built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (OUT / "seg_v5b_stage_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
