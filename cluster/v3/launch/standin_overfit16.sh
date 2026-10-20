#!/bin/bash
# Plumbing validation on UNGATED dinov2-large (patch 14 -> 1288x728, grid
# 92x52). Overfits 16 train images (12 pos / 4 neg) for 300 steps and evals
# on the SAME 16. Success = loss collapse + high train-mAP + resumable ckpts.
# THE NUMBERS ARE NOT DINOv3 RESULTS (stand-in validates plumbing only).
set -euo pipefail
cd "$(dirname "$0")/../../.."
source .venv/bin/activate
exec python cluster/v3/orchestrate_dino.py \
  --run standin_d2l_overfit16 \
  --model-dir /scratch/scratch/tianrgua-sandbox/models/dinov2-large \
  --mode overfit \
  --width 1288 --height 728 \
  --bs 2 --lr 2e-4 --workers 4 \
  --max-segments 2
