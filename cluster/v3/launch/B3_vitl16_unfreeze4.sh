#!/bin/bash
# B3: dinov3-vitl16 with the TOP 4 (of 24) blocks unfrozen at 0.1x LR
# (~50M/304M backbone params trainable; HF gradient checkpointing on).
# Chosen over LoRA — justification in reports/v3/trackBC_prep.md §B3.
# Same data/exposure as B1; only the unfreezing differs.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source .venv/bin/activate
exec python cluster/v3/orchestrate_dino.py \
  --run B3_dino3vitl16_top4_1280 \
  --model-dir /scratch/scratch/tianrgua-sandbox/models/dinov3-vitl16 \
  --unfreeze-top 4 \
  --width 1280 --height 736 \
  --epochs 12 --bs 2 --lr 1e-4 --warmup 500 \
  --eval-max 1500 --save-steps 400 --workers 6
