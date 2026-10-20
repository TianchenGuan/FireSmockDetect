#!/bin/bash
# C1 step 1: run frozen dinov3-vit7b16 ONCE over train+val, cache bf16
# last-layer feature maps to /scratch (30.1 MB/img -> ~1.35 TB total).
# BEFORE running: check scratch_free_gb in any recent status.json (the entry
# reports it) >= ~1600 GB; fallbacks in trackBC_prep.md §C1 (fp8 / 1024x576).
# 7B bf16 = ~13.4 GB weights -> fits ONE L40S for inference; 8 GPU-replicas
# shard the image list; resumable (skips existing .pt files).
set -euo pipefail
cd "$(dirname "$0")/../../.."
source .venv/bin/activate
exec python cluster/v3/orchestrate_dino.py \
  --run C1_7b_precompute_1280 \
  --job-mode precompute \
  --model-dir /scratch/scratch/tianrgua-sandbox/models/dinov3-vit7b16 \
  --feat-out /scratch/scratch/tianrgua-sandbox/smoke/feats_v3/dino3_7b_1280x736 \
  --width 1280 --height 736 --bs 2 --workers 6 \
  --max-segments 8
