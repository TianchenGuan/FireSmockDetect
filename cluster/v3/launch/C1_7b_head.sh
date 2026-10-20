#!/bin/bash
# C1 step 2: train the SAME SFP+FRCNN head on cached 7B features (embed 4096).
# NOTE: cached features freeze augmentation (no flip/jitter) — run the
# B1-cached-noaug control before comparing C1 to B1/B2 (trackBC_prep.md §C1).
set -euo pipefail
cd "$(dirname "$0")/../../.."
source .venv/bin/activate
exec python cluster/v3/orchestrate_dino.py \
  --run C1_7b_head_cached_1280 \
  --feat-root /scratch/scratch/tianrgua-sandbox/smoke/feats_v3/dino3_7b_1280x736 \
  --embed-dim 4096 \
  --width 1280 --height 736 \
  --epochs 12 --bs 4 --lr 1e-4 --warmup 500 \
  --eval-max 1500 --save-steps 400 --workers 8
