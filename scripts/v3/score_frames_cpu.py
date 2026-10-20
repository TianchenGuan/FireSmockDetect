"""Parallel CPU frame scoring on the devbox, reusing the pod inference driver.

Spawns N copies of cluster/v3/rfdetr_infer.py (--device cpu, one shard each,
OMP/MKL threads pinned per worker) and concatenates the JSONL shards, so devbox
and cluster runs execute byte-identical inference code.

Used for the FIgLib causal replay (3,056 frames / 39 events): those frames are
NOT staged on the cluster (they live only on the devbox), and the AlertEngine
replay is a deterministic function of per-frame max scores, so CPU scoring here
+ offline replay gives exactly the result a GPU scoring job would.

Usage:
  python scripts/v3/score_frames_cpu.py --weights <ckpt> --list frames.txt \
      --workers 24 --threads 8 --out models/v3/evals/<name>/figlib_preds.jsonl
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
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads", type=int, default=8, help="torch threads per worker")
    ap.add_argument("--imgsz", type=int, default=1280)
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
        cmd = [args.python, str(ROOT / "cluster/v3/rfdetr_infer.py"),
               "--weights", args.weights, "--list", args.list,
               "--shard", f"{i}/{args.workers}", "--device", "cpu",
               "--imgsz", str(args.imgsz), "--conf-floor", str(args.conf_floor),
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
