"""Convert the frozen v4 E_ET_v1 pod train list to a COCO detection JSON for
the DINOv3 track (iteration 4).

Adapted from scripts/v3/yolo_to_coco.py with one deliberate difference:
file_name entries are ABSOLUTE POD PATHS (src/v3/det_data.py joins
root/file_name with pathlib, so absolute paths override the DATA_ROOT):

  E lines  /mnt/local/smoke_data/<rel>
           -> file_name /scratch/scratch/tianrgua-sandbox/smoke_data_v2/<rel>
           (v3 FSx copy; 1:1 by prefix swap)
           local image+label: data/yolo/<rel> (labels via /images/->/labels/)
  T lines  /scratch/scratch/tianrgua-sandbox/smoke/v4data/E_ET_v1/images/<name>
           -> file_name unchanged
           local image: data/v4/yolo/E_ET_v1/images/<name> (symlink)
           local label: data/v4/yolo/E_ET_v1/labels/<stem>.txt

Image sizes are read from the LOCAL files (header-fast, PIL fallback).
Missing label files are kept as negatives (v3 convention for pyro-sdis/D-Fire
empty frames) but counted per component and reported — nothing is skipped
silently. Missing local IMAGE files are a hard error.

Run: .venv/bin/python scripts/v4/yolo_to_coco_eet.py
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/local/home/tianrgua/cvproject")
LIST = ROOT / "data/v4/yolo/E_ET_v1/lists/pod/train.txt"
OUT = ROOT / "data/v4/coco/instances_train_eet.json"

E_POD_PREFIX = "/mnt/local/smoke_data/"
E_LOCAL_ROOT = ROOT / "data/yolo"
E_SCRATCH_PREFIX = "/scratch/scratch/tianrgua-sandbox/smoke_data_v2/"
T_POD_PREFIX = "/scratch/scratch/tianrgua-sandbox/smoke/v4data/E_ET_v1/images/"
T_LOCAL_DIR = ROOT / "data/v4/yolo/E_ET_v1/images"
T_LABEL_DIR = ROOT / "data/v4/yolo/E_ET_v1/labels"


def image_size(path: str) -> tuple[int, int]:
    """Fast JPEG/PNG header-only size read (falls back to PIL)."""
    with open(path, "rb") as f:
        head = f.read(32)
        if head[:2] == b"\xff\xd8":  # JPEG
            f.seek(2)
            while True:
                b = f.read(4)
                if len(b) < 4:
                    break
                marker, size = struct.unpack(">HH", b)
                if 0xFFC0 <= marker <= 0xFFCF and marker not in (0xFFC4, 0xFFC8, 0xFFCC):
                    f.read(1)
                    h, w = struct.unpack(">HH", f.read(4))
                    return w, h
                f.seek(size - 2, 1)
        elif head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return w, h
    from PIL import Image  # fallback

    with Image.open(path) as im:
        return im.size


def route(pod_line: str):
    """pod list line -> (component, local_img, local_label, file_name, flat)."""
    if pod_line.startswith(E_POD_PREFIX):
        rel = pod_line[len(E_POD_PREFIX):]
        local = E_LOCAL_ROOT / rel
        label = Path(str(local).replace("/images/", "/labels/")).with_suffix(".txt")
        return ("E", local, label, E_SCRATCH_PREFIX + rel, rel.replace("/", "__"))
    if pod_line.startswith(T_POD_PREFIX):
        name = pod_line[len(T_POD_PREFIX):]
        if "/" in name:
            raise ValueError(f"unexpected nested T path: {pod_line}")
        local = T_LOCAL_DIR / name
        label = T_LABEL_DIR / (Path(name).stem + ".txt")
        return ("T", local, label, pod_line, name)
    raise ValueError(f"unroutable pod line: {pod_line}")


def load_labels(label_path: Path):
    """returns (boxes, label_file_exists)."""
    if not label_path.exists():
        return [], False
    out = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            if parts:
                raise ValueError(f"malformed label line in {label_path}: {line!r}")
            continue
        cls = int(parts[0])
        if cls != 0:
            raise ValueError(f"unexpected class {cls} in {label_path}")
        out.append(tuple(float(x) for x in parts[1:]))
    return out, True


def process_one(item):
    idx, pod_line = item
    comp, local, label, file_name, flat = route(pod_line)
    if not local.exists():
        return {"idx": idx, "error": f"missing local image: {local}"}
    w, h = image_size(str(local))
    raw, has_label = load_labels(label)
    anns, dropped = [], 0
    for cx, cy, bw, bh in raw:
        x1 = max(0.0, (cx - bw / 2.0) * w)
        y1 = max(0.0, (cy - bh / 2.0) * h)
        x2 = min(float(w), (cx + bw / 2.0) * w)
        y2 = min(float(h), (cy + bh / 2.0) * h)
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            dropped += 1
            continue
        anns.append([round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)])
    return {"idx": idx, "component": comp, "file_name": file_name, "flat": flat,
            "w": w, "h": h, "boxes": anns, "dropped": dropped,
            "has_label": has_label}


def main():
    pod_lines = [l for l in LIST.read_text().splitlines() if l.strip()]
    print(f"{len(pod_lines)} pod list lines")
    with ProcessPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(process_one, list(enumerate(pod_lines)), chunksize=128))
    errors = [r for r in results if "error" in r]
    if errors:
        for e in errors[:10]:
            print("ERROR:", e["error"])
        raise SystemExit(f"{len(errors)} images missing locally — aborting")
    results.sort(key=lambda r: r["idx"])

    images, annotations = [], []
    ann_id = 1
    stats = {c: {"images": 0, "boxes": 0, "neg_no_label_file": 0,
                 "neg_empty_label": 0, "dropped_lt1px": 0} for c in ("E", "T")}
    for r in results:
        img_id = r["idx"] + 1
        images.append({"id": img_id, "file_name": r["file_name"],
                       "flat_file_name": r["flat"],
                       "width": r["w"], "height": r["h"]})
        s = stats[r["component"]]
        s["images"] += 1
        s["dropped_lt1px"] += r["dropped"]
        if not r["boxes"]:
            if r["has_label"]:
                s["neg_empty_label"] += 1
            else:
                s["neg_no_label_file"] += 1
        for b in r["boxes"]:
            s["boxes"] += 1
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                "bbox": b, "area": round(b[2] * b[3], 2),
                                "iscrowd": 0, "segmentation": []})
            ann_id += 1

    coco = {
        "info": {
            "description": "smoke-detect v4 — E_ET_v1 pod train list, absolute pod paths",
            "source_list": str(LIST),
            "conversion": "scripts/v4/yolo_to_coco_eet.py (deterministic, list order)",
            "path_convention": "file_name = ABSOLUTE pod path "
                               "(E remapped /mnt/local/smoke_data -> "
                               "/scratch/scratch/tianrgua-sandbox/smoke_data_v2)",
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "smoke", "supercategory": "none"}],
        "images": images,
        "annotations": annotations,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(coco))
    summary = {
        "out": str(OUT),
        "images": len(images),
        "annotations": len(annotations),
        "per_component": stats,
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }
    (OUT.parent / "CONVERSION_STATS_eet.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
