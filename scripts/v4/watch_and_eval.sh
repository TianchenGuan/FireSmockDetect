#!/bin/bash
# Poll the four v4 DINO orchestrator logs; when a run prints RUN COMPLETE,
# fire scripts/v4/eval_dino_run.sh for it exactly once (marker file).
# Exits when all runs have either been evaluated or their orchestrator died.
set -u
cd "$(dirname "$0")/../.."
MARK=logs/v4/.evalfired
mkdir -p "$MARK"
declare -A BB=(
  [v4B1_frozen_L]=/scratch/scratch/tianrgua-sandbox/models/dinov3-vitl16
  [v4B4_lora_L]=/scratch/scratch/tianrgua-sandbox/models/dinov3-vitl16
  [v4B5_full_L]=/scratch/scratch/tianrgua-sandbox/models/dinov3-vitl16
  [v4B6_lora_Hplus]=/scratch/scratch/tianrgua-sandbox/models/dinov3-vith16plus
)
while true; do
  pending=0
  for r in "${!BB[@]}"; do
    [ -f "$MARK/$r" ] && continue
    LOG=logs/v4/dino_$r.log
    if grep -aq "RUN COMPLETE" "$LOG" 2>/dev/null; then
      touch "$MARK/$r"
      echo "$(date -u +%FT%TZ) $r complete -> launching eval" >> logs/v4/watcher.log
      nohup bash scripts/v4/eval_dino_run.sh "$r" "${BB[$r]}" \
        > "logs/v4/eval_$r.log" 2>&1 &
    elif grep -aqE "max segments without done|fatal staging problem" "$LOG" 2>/dev/null; then
      touch "$MARK/$r"
      echo "$(date -u +%FT%TZ) $r ABORTED (no eval)" >> logs/v4/watcher.log
    else
      pending=$((pending+1))
    fi
  done
  [ "$pending" -eq 0 ] && { echo "$(date -u +%FT%TZ) watcher done" >> logs/v4/watcher.log; break; }
  sleep 120
done
