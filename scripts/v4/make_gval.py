"""Split a built G corpus's train list into unit-grouped train/val (90/10).

Arm B (generic-only transfer) must be STRICTLY source-only: model selection
(ultralytics best.pt) and thresholds may not touch target-domain data, so the
training yaml's val split must come from G itself. Split is by parent_unit
(video/camera/subsource unit), deterministic (md5 of unit), not by frame.

Writes into the existing corpus dir:
  lists/{local,pod}/train90.txt, gval.txt
  yamls/{local,pod}/<corpus>_srconly.yaml   (train=train90, val=gval)
  data/v4/manifests/CORPUS_<corpus>_gval_addendum.json  (counts + list sha256s)
Then re-tars the corpus (same tar path, dereferenced) so pods see the lists.

Usage: python scripts/v4/make_gval.py --corpus B_G_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POD_PREFIX = "/scratch/scratch/tianrgua-sandbox/smoke/v4data"


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--val-frac", type=float, default=0.10)
    args = ap.parse_args()

    cdir = ROOT / "data/v4/yolo" / args.corpus
    stage_tar = ROOT / "data/v4/stage" / f"{args.corpus}.tar"
    man_dir = ROOT / "data/v4/manifests"

    # staged filename (source__rel__path.jpg) -> parent_unit
    unit_of = {}
    for src in ["multinatsmoke", "firesense", "mmodalfire", "dfs", "boreal"]:
        mp = man_dir / f"{src}_manifest.parquet"
        if not mp.exists():
            continue
        df = pd.read_parquet(mp)
        for r in df.itertuples():
            staged = f"{src}__{str(r.rel_path).replace('/', '__')}"
            unit_of[staged] = f"{src}::{r.parent_unit}"

    train_local = cdir / "lists/local/train.txt"
    lines = [l for l in train_local.read_text().splitlines() if l.strip()]
    n_units, unresolved = set(), 0
    buckets = {"train": [], "val": []}
    for l in lines:
        name = Path(l).name
        unit = unit_of.get(name)
        if unit is None:
            unresolved += 1
            buckets["train"].append(l)
            continue
        n_units.add(unit)
        h = int(hashlib.md5(unit.encode()).hexdigest(), 16) % 1000
        buckets["val" if h < args.val_frac * 1000 else "train"].append(l)

    pod_train = (cdir / "lists/pod/train.txt").read_text().splitlines()
    pod_by_name = {Path(l).name: l for l in pod_train if l.strip()}
    for flavor in ["local", "pod"]:
        ldir = cdir / "lists" / flavor
        if flavor == "local":
            out_lines = buckets
        else:
            out_lines = {k: [pod_by_name[Path(l).name] for l in v
                             if Path(l).name in pod_by_name]
                         for k, v in buckets.items()}
        (ldir / "train90.txt").write_text("\n".join(out_lines["train"]) + "\n")
        (ldir / "gval.txt").write_text("\n".join(out_lines["val"]) + "\n")

    for flavor, root in [("local", str(cdir)),
                         ("pod", f"{POD_PREFIX}/{args.corpus}")]:
        ydir = cdir / "yamls" / flavor
        ydir.mkdir(parents=True, exist_ok=True)
        (ydir / f"{args.corpus}_srconly.yaml").write_text(
            f"path: {root}\n"
            f"train: {root}/lists/{flavor}/train90.txt\n"
            f"val: {root}/lists/{flavor}/gval.txt\n"
            f"nc: 1\nnames: [smoke]\n")

    add = {
        "corpus": args.corpus,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "strict source-only val for arm B / D_pre model selection",
        "grouping": "parent_unit (md5 bucket)",
        "val_frac_target": args.val_frac,
        "n_units": len(n_units), "unresolved_to_train": unresolved,
        "n_train90": len(buckets["train"]), "n_gval": len(buckets["val"]),
        "list_sha256": {
            "local/train90": sha(cdir / "lists/local/train90.txt"),
            "local/gval": sha(cdir / "lists/local/gval.txt"),
            "pod/train90": sha(cdir / "lists/pod/train90.txt"),
            "pod/gval": sha(cdir / "lists/pod/gval.txt"),
        },
    }
    out = man_dir / f"CORPUS_{args.corpus}_gval_addendum.json"
    out.write_text(json.dumps(add, indent=2))
    print(json.dumps(add, indent=2))

    print(f"re-tarring {stage_tar} ...")
    with tarfile.open(stage_tar, "w", dereference=True) as tf:
        tf.add(cdir, arcname=args.corpus)
    print("tar bytes:", stage_tar.stat().st_size)


if __name__ == "__main__":
    main()
