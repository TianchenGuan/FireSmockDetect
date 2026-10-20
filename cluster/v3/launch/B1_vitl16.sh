#!/bin/bash
# B1: frozen dinov3-vitl16 + SFP neck + Faster R-CNN head, 12 epochs @1280x736.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source .venv/bin/activate
exec python cluster/v3/orchestrate_dino.py \
  --run B1_dino3vitl16_frozen_1280 \
  --model-dir /scratch/scratch/tianrgua-sandbox/models/dinov3-vitl16 \
  --width 1280 --height 736 \
  --epochs 12 --bs 4 --lr 1e-4 --warmup 500 \
  --eval-max 1500 --save-steps 400 --workers 6
