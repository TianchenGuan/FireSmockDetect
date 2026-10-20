"""Parallel CPU frame scoring on the devbox for Track B/C DINOv3 detectors.

Same pattern as score_frames_cpu.py (which drives rfdetr_infer.py) but spawns
cluster/v3/dino_infer.py shards, so FIgLib replay scoring uses the identical
inference implementation as the cluster eval jobs.

Usage:
  python scripts/v3/score_frames_cpu_dino.py \
      --backbone-dir models/v3/hf/dinov3-vitl16 --ckpt <best.pt> \
      --list data/v3/figlib39_frames.txt --workers 22 --threads 8 \
      --out models/v3/evals/<name>/figlib39_preds.jsonl
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=736)
    ap.add_argument("--conf-floor", type=float, default=0.02)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / (out.stem + "_shards")
    tmp.mkdir(exist_ok=True)

    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": str(args.threads),
                "MKL_NUM_THREADS": str(args.threads),
                "PYTHONPATH": str(ROOT)})
    procs = []
    t0 = time.time()
    for i in range(args.workers):
        cmd = [args.python, str(ROOT / "cluster/v3/dino_infer.py"),
               "--backbone-dir", args.backbone_dir, "--ckpt", args.ckpt,
               "--list", args.list, "--shard", f"{i}/{args.workers}",
               "--device", "cpu",
               "--width", str(args.width), "--height", str(args.height),
               "--conf-floor", str(args.conf_floor),
               "--out", str(tmp / f"shard{i}.jsonl")]
        if args.max_images:
            cmd += ["--max-images", str(args.max_images)]
        log = (tmp / f"shard{i}.log").open("w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
    rc = 0
    for p in procs:
        rc |= p.wait()
    rows = 0
    with out.open("w") as fh:
        for i in range(args.workers):
            f = tmp / f"shard{i}.jsonl"
            if f.exists():
                txt = f.read_text()
                rows += txt.count("\n")
                fh.write(txt)
    el = time.time() - t0
    print(f"workers={args.workers} threads={args.threads} rc={rc} rows={rows} "
          f"wall={el:.0f}s ({rows/max(el,1):.2f} img/s) -> {out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
