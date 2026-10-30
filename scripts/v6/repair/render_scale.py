"""Render the SCALE i2i queue for the repaired corpus (post-pilot, rule v3).

Route A only (pilot winner: 53% pos / 57% ctrl acceptance vs 16%/19% t2i).
Bank v6-repair-1c. Eligible backgrounds: TRAIN-role author-negative pool,
dfire restricted to AoF-prefix fixed-camera frames (WEB*/PublicDataset* web
scrapes excluded after non-wildland scenes were found), mean luma >= 0.25.
Unique-background counts are reported; same-site repetition (sdis cameras,
fireman burn-A video frames, dfire AoF site) is recorded, not hidden.

Output: data/v6/repair/q_scale_s{0..NSHARDS-1}.jsonl + bgs_scale.tar
E01 x2 seeds per bg (positives, visibility 50/35/15), E02 x1 on a prefix
subset (matched-style controls).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "data/v6/repair"
BANK = Path(__file__).resolve().parent / "prompt_bank_v6_repair_1c.yaml"
BGN = {"pyro_sdis_train": 1100, "dfire_train": 250,
       "fireman_burnA_train": 480, "boreal_train_karkkila_empty": 37,
       "sheffield_rec0005_train": 24}


def jhash(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def luma_ok(path, lo=0.25):
    try:
        im = Image.open(path).convert("L")
        im.thumbnail((96, 96))
        return float(np.asarray(im).mean()) / 255 >= lo
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nshards", type=int, default=6)
    ap.add_argument("--n-ctrl", type=int, default=1500)
    args = ap.parse_args()
    bank = yaml.safe_load(BANK.read_text())
    T = {t["id"]: t for t in bank["templates"]}
    F = bank["factors"]
    VIS = F["visibility"]
    base = bank["base_style"]
    e01 = T["E01_local_smoke"]["template"].replace("{base_style}", base)
    e02 = T["E02_matched_control"]["template"].replace("{base_style}", base)

    pool = json.load((ROOT / "data/v6/background_pool_v1.json").open())["images"]
    bysrc = {}
    for p in pool:
        if p["source"] == "dfire_train" and \
                not p["path"].rsplit("/", 1)[-1].startswith("AoF"):
            continue
        bysrc.setdefault(p["source"], []).append(p)
    rng = random.Random("v6rp:scale-bgs")
    bgs = []
    for src, n in BGN.items():
        cand = bysrc.get(src, [])
        rng.shuffle(cand)
        keep = []
        for c in cand:
            if luma_ok(c["path"]):
                keep.append(c)
            if len(keep) >= n:
                break
        if len(keep) < n:
            print(f"WARN {src}: {len(keep)}/{n}")
        bgs += keep
    bgdir = OUTD / "bgs_scale"
    bgdir.mkdir(parents=True, exist_ok=True)
    for b in bgs:
        p = Path(b["path"])
        b["bgname"] = f"{b['source']}__{p.name}"
        if not (bgdir / b["bgname"]).exists():
            shutil.copy2(p, bgdir / b["bgname"])

    nvis = len(bgs) * 2
    vq = (["clearly_visible"] * round(nvis * 0.5)
          + ["small_discernible"] * round(nvis * 0.35))
    vq += ["faint_discernible"] * (nvis - len(vq))
    rng.shuffle(vq)

    jobs = []

    def render(tmpl, **f):
        out = tmpl
        for k, v in f.items():
            out = out.replace("{%s}" % k, v)
        return out

    k = 0
    for bi, b in enumerate(bgs):
        for s in range(2):
            f = {"position": F["position"][(bi + s) % 4],
                 "wind_direction": F["wind_direction"][(bi * 2 + s) % 3],
                 "visibility": VIS[vq[k]]}
            rec = {"route": "i2i", "spec_id": "E01_local_smoke",
                   "kind": "smoke", "prompt": render(e01, **f),
                   "seed": 60000 + bi * 10 + s, "visibility": vq[k],
                   "factors": {x: f[x] for x in ("position", "wind_direction")},
                   "bg": b["bgname"], "bg_src_path": b["path"],
                   "pair": b["bgname"], "bank_version": bank["version"]}
            rec["hash"] = jhash({x: rec[x] for x in
                                 ("route", "spec_id", "prompt", "seed", "bg")})
            jobs.append(rec)
            k += 1
        if bi < args.n_ctrl:
            rec = {"route": "i2i", "spec_id": "E02_matched_control",
                   "kind": "ctrl", "prompt": e02, "seed": 61000 + bi,
                   "visibility": None, "factors": None, "bg": b["bgname"],
                   "bg_src_path": b["path"], "pair": b["bgname"],
                   "bank_version": bank["version"]}
            rec["hash"] = jhash({x: rec[x] for x in
                                 ("route", "spec_id", "prompt", "seed", "bg")})
            jobs.append(rec)

    rng.shuffle(jobs)  # balance shard runtimes
    for s in range(args.nshards):
        with (OUTD / f"q_scale_s{s}.jsonl").open("w") as f:
            for j in jobs[s::args.nshards]:
                f.write(json.dumps(j) + "\n")
    with tarfile.open(OUTD / "bgs_scale.tar", "w") as tf:
        for p in sorted(bgdir.glob("*")):
            tf.add(p, arcname=p.name)

    stats = {"total": len(jobs), "bgs": len(bgs),
             "by_kind": dict(Counter(j["kind"] for j in jobs)),
             "by_src": dict(Counter(b["source"] for b in bgs)),
             "by_visibility": dict(Counter(j["visibility"] for j in jobs
                                           if j["visibility"])),
             "nshards": args.nshards, "bank_version": bank["version"]}
    (OUTD / "SCALE_RENDER_MANIFEST.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
