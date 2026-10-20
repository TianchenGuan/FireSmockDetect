#!/bin/bash
# B2: frozen dinov3-vith16plus (840M, 1280-dim) — identical head/data/exposure
# to B1. Forced differences vs B1: SFP input projection dim (1280 vs 1024);
# bs 4 kept (H+ activations still fit: backbone is no-grad).
set -euo pipefail
cd "$(dirname "$0")/../../.."
source .venv/bin/activate
exec python cluster/v3/orchestrate_dino.py \
  --run B2_dino3vith16plus_frozen_1280 \
  --model-dir /scratch/scratch/tianrgua-sandbox/models/dinov3-vith16plus \
  --width 1280 --height 736 \
  --epochs 12 --bs 4 --lr 1e-4 --warmup 500 \
  --eval-max 1500 --save-steps 400 --workers 6
