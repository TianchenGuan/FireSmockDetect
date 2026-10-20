#!/bin/bash
# Gate watch for the three manually-gated DINOv3 checkpoints (Track B/C).
# One call tests token access to ALL 3 models; exit 0 iff all three grant
# file access (a gated repo returns metadata with HTTP 200 but 403 on
# /resolve/, so we probe an actual file, not the model API).
#
# Usage: bash scripts/v3/dinov3_gate_check.sh [--loop [interval_s]]
#   --loop: poll until all 3 are accessible, then run is complete; pair with
#           scripts/v3/stage_dinov3.sh to start staging the moment access lands.
set -u
TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
[ -f "$TOKEN_FILE" ] || { echo "no HF token at $TOKEN_FILE"; exit 2; }
TOKEN=$(cat "$TOKEN_FILE")
MODELS=(
  facebook/dinov3-vitl16-pretrain-lvd1689m
  facebook/dinov3-vith16plus-pretrain-lvd1689m
  facebook/dinov3-vit7b16-pretrain-lvd1689m
)
check_all() {
  local ok=0
  echo "=== $(date -u +%FT%TZ) ==="
  for m in "${MODELS[@]}"; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
      -H "Authorization: Bearer $TOKEN" \
      "https://huggingface.co/$m/resolve/main/config.json")
    if [ "$code" = "200" ]; then
      echo "  OK   $m (config.json HTTP $code)"
    else
      echo "  GATED $m (config.json HTTP $code)"
      ok=1
    fi
  done
  return $ok
}
if [ "${1:-}" = "--loop" ]; then
  INTERVAL="${2:-300}"
  while true; do
    if check_all; then
      echo "ALL 3 DINOv3 MODELS ACCESSIBLE — run scripts/v3/stage_dinov3.sh now"
      exit 0
    fi
    sleep "$INTERVAL"
  done
else
  if check_all; then
    echo "ALL 3 DINOv3 MODELS ACCESSIBLE — run scripts/v3/stage_dinov3.sh now"
    exit 0
  fi
  echo "access still gated (user must accept terms on the three model pages)"
  exit 1
fi
