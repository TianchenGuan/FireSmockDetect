"""Generate the arm-B (procedural/copy-paste) asset bank.

- up to --n-cp copy-paste RGBA crops from REAL TRAIN positives (box >=24px)
- --n-param parametric plumes (luma mode)
- --n-conf parametric confounders (fog-band / cloud-blob param ranges)
Writes data/v6/assets/B/{cp,param,conf}/*.png + manifest.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v6"))
import procedural_assets as P  # noqa: E402

TRAIN_LIST = ROOT / "data/yolo/lists/local/expB_train.txt"
OUT = ROOT / "data/v6/assets/B"


def try_cp(task):
    img_path, box, out = task
    rng = random.Random(hash(out) & 0xFFFFFFF)
    try:
        return P.copy_paste_asset(img_path, box, Path(out), rng)
    except Exception:
        return None


def main(n_cp=0, n_param=2400, n_conf=600):
    rng = random.Random(7)
    imgs = [l.strip() for l in TRAIN_LIST.read_text().splitlines() if l.strip()]
    rng.shuffle(imgs)
    # candidate (img, box) with pixel side >= 24 using header-only size reads
    tasks = []
    for ip in imgs:
        lp = ip.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        p = Path(lp)
        if not p.exists() or p.stat().st_size == 0:
            continue
        try:
            with Image.open(ip) as im:
                W, H = im.size
        except Exception:
            continue
        for line in p.read_text().splitlines():
            f = line.split()
            if len(f) != 5:
                continue
            _, cx, cy, w, h = map(float, f)
            if w * W >= 24 and h * H >= 24:
                tasks.append((ip, [cx, cy, w, h],
                              str(OUT / "cp" / f"cp{len(tasks):05d}.png")))
                break  # one crop per image for diversity
        if len(tasks) >= n_cp * 2:
            break
    print(f"{len(tasks)} crop candidates")
    recs = []
    with ProcessPoolExecutor(max_workers=32) as ex:
        for r in ex.map(try_cp, tasks, chunksize=8):
            if r:
                r["family_id"] = "cp_" + Path(r["src_image"]).stem
                recs.append(r)
            if len(recs) >= n_cp:
                break
    print(f"{len(recs)} copy-paste assets")

    for i in range(n_param):
        r = P.parametric_plume(OUT / "param" / f"pp{i:05d}.png", seed=1000 + i)
        r.update(kind="parametric", family_id=f"param{1000+i}")
        recs.append(r)
    for i in range(n_conf):
        # fog band: low density, strong wind, wide; cloud blob: dense top
        wind = random.Random(i).uniform(0.6, 1.0) * (1 if i % 2 else -1)
        dens = random.Random(i + 1).uniform(0.25, 0.55)
        r = P.parametric_plume(OUT / "conf" / f"cf{i:05d}.png",
                               seed=5000 + i, wind=wind, density=dens)
        r.update(kind="confounder", family_id=f"conf{5000+i}")
        recs.append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "manifest.jsonl").open("w") as f:
        for r in recs:
            r.setdefault("kind", r.get("kind", "copy_paste"))
            f.write(json.dumps(r) + "\n")
    print("manifest:", len(recs), "assets ->", OUT / "manifest.jsonl")


if __name__ == "__main__":
    main()
