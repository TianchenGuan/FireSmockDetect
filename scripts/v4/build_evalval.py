"""Build the iteration-4 VAL-ONLY eval package (v4_evalval): boreal val burn
(burn_heinola) + AuSmoke forest-holdout VAL units. NEVER touches test units
(burn_ruokolahti, forest_holdout "test") -- asserted below.

Outputs
  data/v4/eval/boreal_val/{images,labels,lists/{local,pod}}   (symlinked images)
  data/v4/eval/forest_val/{images,labels,lists/{local,pod}}
  data/v4/eval/yamls/{boreal_val,forest_val}.yaml              (local, for
      src.eval.eval_detection: path/val/nc/names)
  data/v4/coco/instances_{boreal_val,forest_val}.json          (file_name =
      ABSOLUTE pod path under /scratch/.../smoke/v4data/v4_evalval/)
  data/v4/stage/v4_evalval.tar   (dereferenced; top dir v4_evalval/ so it
      extracts under /scratch/.../smoke/v4data/ like the corpus tars)
  data/v4/eval/EVALVAL_MANIFEST.json  (counts, exclusions, shas)

Label policy (mirrors scripts/v4/build_corpus.py converters, imported):
  boreal_val: Subset-A box rows only. has_smoke box rows -> yolo_txt_to_lines;
      empty-image rows (has_smoke=False, explicit Empty-Labels) -> negatives
      (empty label txt). Subset-C mask rows are per-basename duplicates of
      Subset-A frames (336/336, different encode) -> SKIPPED, counted.
      videolabel rows are .mp4 (non-image) -> skipped, counted.
  forest_val: all mask rows -> mask_to_lines (same converter as training
      corpus). Zero negatives exist in these units (documented: matched-FP
      anchoring is impossible within forest_val).

Run: .venv/bin/python scripts/v4/build_evalval.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path("/local/home/tianrgua/cvproject")
sys.path.insert(0, str(ROOT / "scripts/v4"))
from build_corpus import (LabelResolver, flat_name, mask_to_lines,  # noqa: E402
                          yolo_txt_to_lines)
from yolo_to_coco_eet import image_size  # noqa: E402

EVAL_ROOT = ROOT / "data/v4/eval"
POD_ROOT = "/scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evalval"
TEST_BANNED_BOREAL = {"burn_ruokolahti"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_boreal(excl: Counter):
    """returns list of (abs_img, label_lines) -- label_lines [] => negative."""
    b = pd.read_parquet(ROOT / "data/v4/manifests/boreal_manifest.parquet")
    h = b[b.parent_unit == "burn_heinola"]
    res = LabelResolver("boreal", ROOT / "data/v4/raw/boreal")
    entries = []
    for r in h.itertuples(index=False):
        if os.path.splitext(r.rel_path)[1].lower() not in IMG_EXTS:
            excl["boreal:non_image(videolabel_mp4)"] += 1
            continue
        if "Subset-C" in r.rel_path:
            excl["boreal:subsetC_duplicate_of_subsetA"] += 1
            continue
        img = ROOT / "data/v4/raw/boreal" / r.rel_path
        if not img.exists():
            excl["boreal:missing_file"] += 1
            continue
        if str(r.label_type) != "box":
            excl[f"boreal:unexpected_label_type_{r.label_type}"] += 1
            continue
        found = res.find_box_label(img)
        if found is None:
            excl["boreal:label_not_found"] += 1
            continue
        fmt, ref = found
        assert fmt == "yolo", f"unexpected label fmt {fmt} for {img}"
        lines, _ = yolo_txt_to_lines(ref, {0})
        if bool(r.has_smoke_label):
            if not lines:
                excl["boreal:smoke_pos_converted_to_zero_boxes"] += 1
                continue
            entries.append((str(img), lines))
        else:
            if lines:
                excl["boreal:hs_false_but_boxes_found"] += 1
                continue
            entries.append((str(img), []))  # verified-empty negative
    return entries


def build_forest(excl: Counter):
    m = pd.read_parquet(ROOT / "data/v4/manifests/multinatsmoke_manifest.parquet")
    hold = json.loads((ROOT / "data/v4/manifests/forest_holdout_v4.json").read_text())
    val_units = set(hold["val"])
    assert not val_units & set(hold["test"]), "val/test unit overlap"
    mv = m[m.parent_unit.isin(val_units)]
    raw = ROOT / "data/v4/raw/multinatsmoke"
    entries = []
    for r in mv.itertuples(index=False):
        img = raw / r.rel_path
        mask = raw / r.mask_rel_path
        if not img.exists() or not mask.exists():
            excl["forest:missing_file"] += 1
            continue
        lines = mask_to_lines(mask, img)
        if not lines:
            excl["forest:mask_converted_to_zero_boxes"] += 1
            continue
        entries.append((str(img), lines))
    return entries


def materialize(name: str, source: str, entries):
    d = EVAL_ROOT / name
    for sub in ("images", "labels", "lists/local", "lists/pod"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    raw_root = ROOT / "data/v4/raw" / source
    local_lines, pod_lines, coco_images, coco_anns = [], [], [], []
    ann_id, n_pos, n_neg, n_boxes = 1, 0, 0, 0
    used = set()
    for i, (abs_img, lines) in enumerate(sorted(entries)):
        rel = os.path.relpath(abs_img, raw_root)
        fname = flat_name(source, rel)
        assert fname not in used, f"flat name collision: {fname}"
        used.add(fname)
        link = d / "images" / fname
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(abs_img, link)
        (d / "labels" / (os.path.splitext(fname)[0] + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        local_lines.append(str(link))
        pod_lines.append(f"{POD_ROOT}/{name}/images/{fname}")
        W, H = image_size(abs_img)
        img_id = i + 1
        coco_images.append({"id": img_id,
                            "file_name": f"{POD_ROOT}/{name}/images/{fname}",
                            "flat_file_name": fname, "width": W, "height": H})
        if lines:
            n_pos += 1
        else:
            n_neg += 1
        for ln in lines:
            _, cx, cy, w, h = (float(v) for v in ln.split())
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            bw, bh = w * W, h * H
            coco_anns.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                              "bbox": [round(x1, 2), round(y1, 2),
                                       round(bw, 2), round(bh, 2)],
                              "area": round(bw * bh, 2), "iscrowd": 0,
                              "segmentation": []})
            ann_id += 1
            n_boxes += 1
    (d / "lists/local/val.txt").write_text("\n".join(local_lines) + "\n")
    (d / "lists/pod/val.txt").write_text("\n".join(pod_lines) + "\n")
    coco = {"info": {"description": f"smoke-detect v4 evalval -- {name}",
                     "conversion": "scripts/v4/build_evalval.py",
                     "path_convention": f"file_name = ABSOLUTE pod path under {POD_ROOT}"},
            "licenses": [],
            "categories": [{"id": 1, "name": "smoke", "supercategory": "none"}],
            "images": coco_images, "annotations": coco_anns}
    cj = ROOT / "data/v4/coco" / f"instances_{name}.json"
    cj.write_text(json.dumps(coco))
    yd = EVAL_ROOT / "yamls"
    yd.mkdir(exist_ok=True)
    (yd / f"{name}.yaml").write_text(
        f"# v4 evalval {name} -- generated by scripts/v4/build_evalval.py\n"
        f"path: {d}\n"
        f"val: {d / 'lists/local/val.txt'}\n"
        f"nc: 1\nnames: [smoke]\n")
    return {"images": len(local_lines), "positives": n_pos, "negatives": n_neg,
            "boxes": n_boxes, "coco": str(cj), "coco_sha256": sha256_file(cj)}


def main():
    excl = Counter()
    boreal = build_boreal(excl)
    forest = build_forest(excl)
    # hard test-set bans
    for abs_img, _ in boreal + forest:
        assert "ruokolahti" not in abs_img.lower(), f"TEST LEAK: {abs_img}"
    hold = json.loads((ROOT / "data/v4/manifests/forest_holdout_v4.json").read_text())
    for u in hold["test"]:
        for abs_img, _ in forest:
            assert u not in abs_img, f"TEST LEAK: {u} in {abs_img}"

    stats = {"boreal_val": materialize("boreal_val", "boreal", boreal),
             "forest_val": materialize("forest_val", "multinatsmoke", forest),
             "excluded": dict(excl)}

    # stage tar: v4_evalval/{boreal_val,forest_val}/{images,labels,lists} + coco
    stage = ROOT / "data/v4/stage"
    stage.mkdir(parents=True, exist_ok=True)
    tar_path = stage / "v4_evalval.tar"
    with tarfile.open(tar_path, "w", dereference=True) as tf:
        for name in ("boreal_val", "forest_val"):
            tf.add(EVAL_ROOT / name, arcname=f"v4_evalval/{name}")
            tf.add(ROOT / "data/v4/coco" / f"instances_{name}.json",
                   arcname=f"v4_evalval/instances_{name}.json")
    stats["tar"] = {"path": str(tar_path), "bytes": tar_path.stat().st_size,
                    "sha256": sha256_file(tar_path)}
    (EVAL_ROOT / "EVALVAL_MANIFEST.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    sys.exit(main())
