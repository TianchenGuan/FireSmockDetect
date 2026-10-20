"""Convert frozen YOLO-format train/val lists to COCO detection JSON.

Deterministic: images sorted by their order in the frozen list files; ids
assigned sequentially. Single category 'smoke' (category_id=1; id 0 is
reserved-as-background by several DETR pipelines, and Roboflow-style COCO
exports use a supercategory stub at id 0 — we emit category id 1 only).

YOLO label line: cls cx cy w h (all normalized to [0,1], cx/cy = box center).
COCO bbox: [x_min, y_min, width, height] in absolute pixels (xywh, top-left).
We convert cx,cy,w,h -> x1 = (cx - w/2)*W, y1 = (cy - h/2)*H, then clamp the
xyxy box to the image and re-derive w,h. Boxes that collapse (<1px side after
clamping) are dropped and counted.

Images with no label file or an empty label file are kept as negatives
(present in images[], no annotations) — matching how ultralytics treated them
in expB/expE training.

file_name in the emitted JSON = path RELATIVE to the data root
(data/yolo/ locally == /scratch/scratch/tianrgua-sandbox/smoke_data_v2/ in
pods), e.g. "pyro_sdis/images/train/foo.jpg". Loaders that require a flat
directory get a symlink tree built pod-side (see cluster/v3/det_entry.sh);
the flat name is the relative path with '/' -> '__' (also recorded here as
'flat_file_name' for that purpose; harmless extra key for other loaders).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/local/home/tianrgua/cvproject")
DATA_ROOT = ROOT / "data" / "yolo"


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


def load_labels(img_path: str) -> list[tuple[int, float, float, float, float]]:
    lp = Path(img_path.replace("/images/", "/labels/")).with_suffix(".txt")
    if not lp.exists():
        return []
    out = []
    for line in lp.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            if parts:  # non-empty malformed line: fail loudly
                raise ValueError(f"malformed label line in {lp}: {line!r}")
            continue
        out.append((int(parts[0]), *(float(x) for x in parts[1:])))
    return out


def process_one(item: tuple[int, str]) -> dict:
    idx, img_path = item
    w, h = image_size(img_path)
    anns = []
    dropped = 0
    for cls, cx, cy, bw, bh in load_labels(img_path):
        if cls != 0:
            raise ValueError(f"unexpected class {cls} in labels of {img_path}")
        x1 = max(0.0, (cx - bw / 2.0) * w)
        y1 = max(0.0, (cy - bh / 2.0) * h)
        x2 = min(float(w), (cx + bw / 2.0) * w)
        y2 = min(float(h), (cy + bh / 2.0) * h)
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            dropped += 1
            continue
        anns.append([round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)])
    return {"idx": idx, "path": img_path, "w": w, "h": h, "boxes": anns, "dropped": dropped}


def convert(list_file: Path, out_json: Path) -> dict:
    paths = [p for p in list_file.read_text().splitlines() if p.strip()]
    with ProcessPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(process_one, list(enumerate(paths)), chunksize=256))
    results.sort(key=lambda r: r["idx"])

    images, annotations = [], []
    ann_id = 1
    n_neg = n_dropped = 0
    for r in results:
        rel = str(Path(r["path"]).relative_to(DATA_ROOT))
        img_id = r["idx"] + 1
        images.append(
            {
                "id": img_id,
                "file_name": rel,
                "flat_file_name": rel.replace("/", "__"),
                "width": r["w"],
                "height": r["h"],
            }
        )
        n_dropped += r["dropped"]
        if not r["boxes"]:
            n_neg += 1
        for b in r["boxes"]:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": b,
                    "area": round(b[2] * b[3], 2),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

    coco = {
        "info": {
            "description": f"smoke-detect v3 — converted from {list_file.name}",
            "source_list": str(list_file),
            "conversion": "scripts/v3/yolo_to_coco.py (deterministic, list order)",
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "smoke", "supercategory": "none"}],
        "images": images,
        "annotations": annotations,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco))
    sha = hashlib.sha256(out_json.read_bytes()).hexdigest()
    stats = {
        "out": str(out_json),
        "images": len(images),
        "annotations": len(annotations),
        "negatives_no_boxes": n_neg,
        "boxes_dropped_lt1px": n_dropped,
        "sha256": sha,
    }
    print(json.dumps(stats, indent=1))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "data" / "v3" / "coco"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    lists = ROOT / "data" / "yolo" / "lists" / "local"
    s1 = convert(lists / "expB_train.txt", outdir / "instances_train.json")
    s2 = convert(lists / "sdis_val.txt", outdir / "instances_val.json")
    (outdir / "CONVERSION_STATS.json").write_text(json.dumps([s1, s2], indent=1))


if __name__ == "__main__":
    sys.exit(main())
