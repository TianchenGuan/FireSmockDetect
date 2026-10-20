"""Package a V6 synthetic corpus for pod training: combined list, yaml, tar.

- combined train list = real expB_train (pod paths) x real_mult
  + synth list_pod.txt x synth_mult
- yaml: /scratch corpus paths + baked sdis val/test (v5b convention)
- tar contains <name>/{images,labels,lists}; extracted at V6 data root
  /scratch/scratch/tianrgua-sandbox/smoke/v6data by train_entry_v6.sh

Usage:
  .venv/bin/python scripts/v6/pack_arm.py --corpus data/v6/synthB --name synthB \
      --arm B [--real-mult 1 --synth-mult 1]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_REAL = ROOT / "data/yolo/lists/local/expB_train.txt"
LOCAL_PREFIX = str(ROOT / "data/yolo")
POD_PREFIX = "/mnt/local/smoke_data"
V6ROOT = "/scratch/scratch/tianrgua-sandbox/smoke/v6data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--real-mult", type=int, default=1)
    ap.add_argument("--synth-mult", type=int, default=1)
    args = ap.parse_args()

    corpus = ROOT / args.corpus
    real = [l.strip().replace(LOCAL_PREFIX, POD_PREFIX)
            for l in LOCAL_REAL.read_text().splitlines() if l.strip()]
    synth = [l.strip() for l in (corpus / "list_pod.txt").read_text().splitlines()
             if l.strip()]
    combined = real * args.real_mult + synth * args.synth_mult
    (corpus / "lists").mkdir(exist_ok=True)
    train_list = corpus / "lists" / f"train{args.arm}.txt"
    train_list.write_text("\n".join(combined) + "\n")

    yaml = ROOT / f"data/v6/yamls/v6{args.arm}.yaml"
    yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml.write_text(f"""# V6 arm {args.arm} ({args.name}) -- generated {datetime.now(timezone.utc).isoformat()}
# real x{args.real_mult} ({len(real)}) + synth x{args.synth_mult} ({len(synth)}) = {len(combined)} per epoch
path: {V6ROOT}/{args.name}
train: {V6ROOT}/{args.name}/lists/train{args.arm}.txt
val: {POD_PREFIX}/lists/pod/sdis_val.txt
test: {POD_PREFIX}/lists/pod/sdis_test.txt
nc: 1
names: [smoke]
""")

    stage = ROOT / "data/v6/stage"
    stage.mkdir(parents=True, exist_ok=True)
    tar = stage / f"v6{args.name}.tar"
    subprocess.run(["tar", "-cf", str(tar), "-C", str(corpus.parent),
                    f"{corpus.name}/images", f"{corpus.name}/labels",
                    f"{corpus.name}/lists"], check=True)
    sz = tar.stat().st_size / 1e9
    man = {"arm": args.arm, "name": args.name, "real": len(real),
           "real_mult": args.real_mult, "synth": len(synth),
           "synth_mult": args.synth_mult, "epoch_images": len(combined),
           "tar_gb": round(sz, 2), "corpus_stats": json.loads(
               (corpus / "stats.json").read_text())}
    mpath = ROOT / f"data/v6/manifests/CORPUS_{args.arm}_v1.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))
    print(f"\ntar: {tar} ({sz:.1f} GB)\nyaml: {yaml}\nlaunch:\n"
          f"  cluster/v6/run_yolo.sh v6{args.arm.lower()}-s0 1 nano{args.arm} "
          f"'arm {args.arm}' -- --exp expV6{args.arm}_s0 --job-tag v6{args.arm.lower()}0 "
          f"--data-yaml-file {yaml.relative_to(ROOT)} "
          f"--data-tar {args.name}={tar.relative_to(ROOT)} "
          f"--pod-data-root {V6ROOT} --epochs 30 --imgsz 1280 --batch 48 "
          f"--base-weights yolo11n.pt --seed 0 --init-key v6/weights/yolo11n.pt")


if __name__ == "__main__":
    main()
