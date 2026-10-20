#!/usr/bin/env bash
# V5b FAST / ACCURATE demo on a SPARE port (default 8013). Production :8000 is never touched.
#   FAST     = file defaults (yolo expB@1280, 0.30/0.15)
#   ACCURATE = yolo expV4A_ctrl_s2 (0.30/0.15) + dino3 v4B4_lora_L (0.93/0.82, reports/v4/BACKEND_DINO3_LORA.md), fusion "or"
# Steps: start server -> wait ready -> replay one sdis VAL sequence + one Boreal heinola (aerial VAL) clip in
# both modes through /detect (source=historical_replay) -> screenshots with headless chrome (if present) -> stop.
#   bash scripts/v5b/run_demo_v5b.sh [--keep]      (--keep leaves the server running for interactive use)
set -euo pipefail
ROOT=/local/home/tianrgua/cvproject
cd "$ROOT"; source .venv/bin/activate
PORT=${PORT:-8013}
OUT=${OUT:-$ROOT/data/v5b/demo}
SHOTS=${SHOTS:-$ROOT/reports/v5b/demo}
SDIS_SEQ=${SDIS_SEQ:-val_force-06_cabanelle-125_20240403T101630}
BOREAL_SEQ=${BOREAL_SEQ:-boreal_heinola_103}
CHROME=${CHROME:-$(ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome 2>/dev/null | sort | tail -1 || true)}
KEEP=0; [ "${1:-}" = "--keep" ] && KEEP=1
[ "$PORT" = "8000" ] && { echo "refusing the production port 8000"; exit 2; }
if ss -ltn | grep -q ":${PORT} "; then echo "port $PORT busy; abort" >&2; exit 2; fi
mkdir -p "$OUT/runs" "$SHOTS"

# ---- exact env for the ACCURATE set (FAST stays at file defaults: no SMOKE_BACKEND / SMOKE_WEIGHTS_PATH)
export SMOKE_ACCURATE_MODELS="yolo_A=yolo:models/trained/expV4A_ctrl_s2/seg05/best.pt,version=expV4A_ctrl_s2@seg05;dino_B4=dino3:models/v3/runs/v4B4_lora_L/seg05/best.pt,version=v4B4_lora_L@seg05-step15026,conf_single=0.93,conf_seq=0.82"
export SMOKE_DEMO_BOREAL_CLIPS=${SMOKE_DEMO_BOREAL_CLIPS:-heinola_103}
export SMOKE_DEMO_RUNS_DIR="$OUT/runs"
# SMOKE_MODE is deliberately NOT set -> default "fast"; the UI / clients pick accurate per request.

SRVLOG="$OUT/server_${PORT}.log"; PIDF="$OUT/server_${PORT}.pid"
nohup uvicorn src.serve.app:app --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
echo $! > "$PIDF"; echo "server pid $(cat $PIDF) on :$PORT; log $SRVLOG"
cleanup() { if [ "$KEEP" = 0 ] && [ -f "$PIDF" ]; then kill "$(cat $PIDF)" 2>/dev/null || true; sleep 2; kill -9 "$(cat $PIDF)" 2>/dev/null || true; rm -f "$PIDF"; echo "demo server :$PORT stopped"; fi; }
trap cleanup EXIT
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/healthz" 2>/dev/null | python -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d['status']=='ok' and d['modes']['accurate'].get('loaded') else 1)" 2>/dev/null; then break; fi
  kill -0 "$(cat $PIDF)" 2>/dev/null || { echo "server exited early"; tail -30 "$SRVLOG"; exit 3; }
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/healthz" > "$OUT/healthz_start.json" || { echo "server never became ready"; tail -30 "$SRVLOG"; exit 3; }
python -c "import json; d=json.load(open('$OUT/healthz_start.json'))['modes']; print('FAST', d['fast']['model_version'], '| ACCURATE', [(m['name'], m['model_version'], m['conf_single'], m['conf_seq']) for m in d['accurate']['models']], d['accurate']['fusion']['method'])"

# ---- replays (historical_replay) : sdis VAL sequence + heinola clip, each in both modes
T0=$(date +%s)
for spec in "fast:$SDIS_SEQ:fast_sdis" "accurate:$SDIS_SEQ:acc_sdis" "fast:$BOREAL_SEQ:fast_heinola" "accurate:$BOREAL_SEQ:acc_heinola"; do
  IFS=: read -r MODE SEQ RID <<< "$spec"
  python scripts/v5b/demo_replay.py --server "http://127.0.0.1:$PORT" --seq "$SEQ" --mode "$MODE" --run-id "$RID" --runs-dir "$OUT/runs" | tail -3
done
echo "replays wall: $(( $(date +%s) - T0 )) s"
curl -sf "http://127.0.0.1:$PORT/healthz" > "$OUT/healthz_end.json"

# ---- screenshots (headless chrome; the page loads the saved runs through /demo/runs and frames through /demo/frame)
if [ -n "$CHROME" ] && [ -x "$CHROME" ]; then
  shot() { "$CHROME" --headless=new --no-sandbox --disable-gpu --hide-scrollbars --window-size="${3:-1500,2200}" \
             --virtual-time-budget=30000 --screenshot="$SHOTS/$1.png" "http://127.0.0.1:$PORT/demo?$2" >/dev/null 2>&1 && echo "shot $SHOTS/$1.png" || echo "screenshot $1 FAILED"; }
  shot demo_accurate_sdis   "run=acc_sdis&frame=${SDIS_FRAME:-10}"
  shot demo_fast_sdis       "run=fast_sdis&frame=${SDIS_FRAME:-10}"
  shot demo_accurate_heinola "run=acc_heinola&frame=${HEINOLA_FRAME:-3}"
  shot demo_fast_heinola    "run=fast_heinola&frame=${HEINOLA_FRAME:-3}"
  shot demo_landing         "mode=accurate" "1500,1400"
else
  echo "no headless chrome found; JSON timelines in $OUT/runs are the artifact"
fi
echo "runs: $(ls $OUT/runs)"
curl -sf http://127.0.0.1:8000/healthz | python -c "import sys,json; d=json.load(sys.stdin); print('production :8000 still', d['status'], d['model_version'], 'uptime_s', d['uptime_s'])" || echo "production :8000 not reachable (unchanged by this script)"
