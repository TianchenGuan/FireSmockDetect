"""Track B: convert pyro-sdis parquet shards to ultralytics YOLO layout + build manifest.

- Images: raw embedded JPEG bytes written as-is (no re-encode) to images/{split}/{image_name}.
- Labels: labels/{split}/{stem}.txt ; class id REMAPPED 1 -> 0 to match data.yaml (nc=1, names=['smoke']).
  Original class ids preserved in manifest column `class_ids_orig`.
- Zero-box images get an empty .txt (explicit negative, mirrors upstream README export code).
- Integrity: every image decoded fully with PIL to detect corruption; width/height recorded.
"""
import glob
import io
import json
import multiprocessing as mp
import os
import re
import sys

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

RAW = "/local/home/tianrgua/cvproject/data/raw/pyro_sdis/data"
OUT = "/local/home/tianrgua/cvproject/data/yolo/pyro_sdis"
MANIFEST_DIR = "/local/home/tianrgua/cvproject/data/manifests"

NAME_RE = re.compile(
    r"^(?P<partner>.+?)_(?P<camera>.+?)_(?P<date>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.jpg$"
)
CAM_RE = re.compile(r"^(?P<site>.+)-(?P<azimuth>\d{1,3})$")


def process_row_group(task):
    fp, rg = task
    split = "train" if "/train-" in fp else "val"
    img_dir = os.path.join(OUT, "images", split)
    lbl_dir = os.path.join(OUT, "labels", split)
    pf = pq.ParquetFile(fp)
    t = pf.read_row_group(rg)
    records = []
    for i in range(t.num_rows):
        img_struct = t.column("image")[i].as_py()
        raw = img_struct["bytes"]
        ann = t.column("annotations")[i].as_py() or ""
        name = t.column("image_name")[i].as_py()
        partner = t.column("partner")[i].as_py()
        camera = t.column("camera")[i].as_py()
        date = t.column("date")[i].as_py()

        # decode check + dims
        width = height = None
        decode_ok = True
        err = ""
        try:
            im = Image.open(io.BytesIO(raw))
            im.load()  # full decode to catch truncation
            width, height = im.size
        except Exception as e:  # noqa: BLE001
            decode_ok = False
            err = repr(e)[:200]

        img_path = os.path.join(img_dir, name)
        with open(img_path, "wb") as f:
            f.write(raw)

        boxes = []
        class_ids = []
        coord_violation = False
        for line in ann.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cid = int(parts[0])
            xc, yc, w, h = (float(x) for x in parts[1:5])
            class_ids.append(cid)
            boxes.append([xc, yc, w, h])
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1
                    and xc - w / 2 >= -1e-6 and xc + w / 2 <= 1 + 1e-6
                    and yc - h / 2 >= -1e-6 and yc + h / 2 <= 1 + 1e-6):
                coord_violation = True

        stem = os.path.splitext(name)[0]
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            for (xc, yc, w, h) in boxes:
                f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        m = NAME_RE.match(name)
        parsed_partner = m.group("partner") if m else None
        parsed_camera = m.group("camera") if m else None
        parsed_date = m.group("date") if m else None
        cm = CAM_RE.match(camera) if camera else None
        site = cm.group("site") if cm else camera
        azimuth = int(cm.group("azimuth")) if cm else None
        dt_iso = None
        if parsed_date:
            d, tm = parsed_date.split("T")
            dt_iso = d + " " + tm.replace("-", ":")

        records.append({
            "image_path": img_path,
            "image_name": name,
            "split_official": split,
            "n_boxes": len(boxes),
            "boxes": json.dumps(boxes),
            "class_ids_orig": json.dumps(class_ids),
            "class_ids_yolo": json.dumps([0] * len(class_ids)),
            "partner": partner,
            "camera_id": camera,
            "site": site,
            "azimuth_deg": azimuth,
            "datetime": dt_iso,
            "event_id": None,  # not encoded in pyro-sdis filenames
            "name_meta_match": bool(m) and parsed_partner == partner
                               and parsed_camera == camera and parsed_date == date,
            "width": width,
            "height": height,
            "bytes_len": len(raw),
            "decode_ok": decode_ok,
            "decode_err": err,
            "coord_violation": coord_violation,
            "source": "pyro-sdis",
            "license": "Apache-2.0",
        })
    return records


def main():
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "*.parquet")))
    tasks = []
    for fp in files:
        pf = pq.ParquetFile(fp)
        tasks.extend((fp, rg) for rg in range(pf.metadata.num_row_groups))
    print(f"{len(files)} shards, {len(tasks)} row-group tasks", flush=True)

    all_records = []
    with mp.Pool(64) as pool:
        for i, recs in enumerate(pool.imap_unordered(process_row_group, tasks)):
            all_records.extend(recs)
            if (i + 1) % 40 == 0:
                print(f"row groups done: {i+1}/{len(tasks)}, rows: {len(all_records)}", flush=True)

    df = pd.DataFrame(all_records).sort_values(["split_official", "image_name"]).reset_index(drop=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    pq_path = os.path.join(MANIFEST_DIR, "pyro_sdis_manifest.parquet")
    csv_path = os.path.join(MANIFEST_DIR, "pyro_sdis_manifest.csv.gz")
    df.to_parquet(pq_path, index=False)
    df.to_csv(csv_path, index=False, compression="gzip")
    print("manifest rows:", len(df))
    print("decode failures:", int((~df.decode_ok).sum()))
    print("coord violations:", int(df.coord_violation.sum()))
    print("name/meta mismatches:", int((~df.name_meta_match).sum()))
    print("saved:", pq_path, "and", csv_path)


if __name__ == "__main__":
    sys.exit(main())
