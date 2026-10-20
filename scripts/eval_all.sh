#!/bin/bash
# One-shot post-training evaluation for a checkpoint.
# Usage: bash scripts_eval_all.sh <best.pt> <tag> [imgsz]
set -euo pipefail
cd /local/home/tianrgua/cvproject && source .venv/bin/activate
W=$1; TAG=$2; SZ=${3:-1280}
echo "== $TAG: val sweep (threshold selection) =="
python -m src.eval.eval_detection --data-yaml data/yolo/yamls/local/expA.yaml \
  --split val --weights "$W" --imgsz "$SZ" --device cpu \
  --out "reports/eval_${TAG}_val.json"
echo "== $TAG: frozen test =="
python -m src.eval.eval_detection --data-yaml data/yolo/yamls/local/expA.yaml \
  --split test --weights "$W" --imgsz "$SZ" --device cpu \
  --out "reports/eval_${TAG}_test.json"
echo "== $TAG: FIgLib causal replay (clean for our models) =="
python -m src.eval.replay_eval --manifest data/manifests/figlib_subset_manifest.parquet \
  --weights "$W" --imgsz "$SZ" --device cpu --out "reports/replay_${TAG}_figlib.json"
echo "== $TAG: FIgLib replay with tiling (expC config) =="
python -m src.eval.replay_eval --manifest data/manifests/figlib_subset_manifest.parquet \
  --weights "$W" --imgsz "$SZ" --tile-size 1024 --device cpu \
  --out "reports/replay_${TAG}_figlib_tiled.json"
echo "done: reports/*_${TAG}_*.json"
