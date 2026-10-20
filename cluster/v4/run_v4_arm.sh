#!/bin/bash
# v4 arm runner (new-data capable): registry reservation around orchestrate_v4.
# Usage: bash cluster/v4/run_v4_arm.sh <exp> <tag> [orchestrate_v4 args...]
set -u
cd /local/home/tianrgua/cvproject
source .venv/bin/activate
EXP="$1"; TAG="$2"; shift 2

python cluster/v4/registry.py reserve --job "res-$TAG" --nodes 1 --track yolo-matrix \
  --note "$EXP" || exit 3

python cluster/v4/orchestrate_v4.py --exp "$EXP" --job-tag "$TAG" "$@"
RC=$?

python cluster/v4/registry.py release --job "res-$TAG" \
  --outcome "$([ $RC -eq 0 ] && echo SUCCEEDED || echo "RC$RC")"
exit $RC
