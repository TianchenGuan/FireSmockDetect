#!/bin/bash
# v4 YOLO arm runner: registry reservation around the segmented orchestrator.
# Usage: bash cluster/v4/run_yolo_arm.sh <exp_name> <job_tag> <data_yaml_pod_path> <epochs> <base_weights> [extra orchestrator args...]
set -u
# NOTE (v5b R1): this launcher cannot pass --seed (v6 entry) -> every run is seed 0.
# Use cluster/v4/run_v4_arm.sh --seed <S> for seeded runs.
[ "${ALLOW_SEED0_REPLICATE:-0}" = "1" ] || { echo "refusing: seed-0-only launcher; set ALLOW_SEED0_REPLICATE=1 or use run_v4_arm.sh --seed"; exit 7; }
cd /local/home/tianrgua/cvproject
source .venv/bin/activate
EXP="$1"; TAG="$2"; DATA="$3"; EPOCHS="$4"; BASE="$5"; shift 5

python cluster/v4/registry.py reserve --job "res-$TAG" --nodes 1 --track yolo-matrix \
  --note "$EXP" || exit 3

python cluster/orchestrate_training.py --exp "$EXP" --job-tag "$TAG" \
  --data-yaml "$DATA" --epochs "$EPOCHS" --imgsz 1280 --batch 48 \
  --base-weights "$BASE" \
  --image-uri 000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-detect:v6 \
  "$@"
RC=$?

python cluster/v4/registry.py release --job "res-$TAG" \
  --outcome "$([ $RC -eq 0 ] && echo SUCCEEDED || echo "RC$RC")"
exit $RC
