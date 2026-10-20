#!/usr/bin/env bash
# Shadow-mode DEMO on a spare port (default 8012). Production :8000 is never touched.
# primary = file defaults (yolo expB@1280, 0.30/0.15); shadow = dino3 B4 LoRA (0.93/0.82).
# Inputs are STORED frames replayed with source=historical_replay (plus 2 posts
# with the default tag to show the tag differs) -- this is not a live run.
set -euo pipefail
ROOT=/local/home/tianrgua/cvproject
cd "$ROOT"; source .venv/bin/activate
PORT=${PORT:-8012}
OUT=${OUT:-$ROOT/data/v5/shadow}
RUN=${RUN:-demo_$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
LOG="$OUT/${RUN}_shadow.jsonl"; SRVLOG="$OUT/${RUN}_server_${PORT}.log"; PIDF="$OUT/${RUN}_server.pid"

if ss -ltn | grep -q ":${PORT} "; then echo "port $PORT busy; abort" >&2; exit 2; fi

# ---- exact variables that enable the same shadow on :8000 later (SMOKE_PORT not needed; uvicorn --port)
export SMOKE_SHADOW_BACKEND=dino3
export SMOKE_SHADOW_WEIGHTS_PATH=models/v3/runs/v4B4_lora_L/seg05/best.pt
export SMOKE_SHADOW_MODEL_VERSION=v4B4_lora_L@seg05-step15026
export SMOKE_SHADOW_CONF_SINGLE=0.93
export SMOKE_SHADOW_CONF_SEQ=0.82
export SMOKE_SHADOW_LOG_PATH="$LOG"
export SMOKE_SHADOW_DEVICE=cpu
# primary deliberately left at file defaults (expB@1280) -- no SMOKE_BACKEND/WEIGHTS set.

nohup uvicorn src.serve.app:app --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
echo $! > "$PIDF"
echo "server pid $(cat $PIDF) on :$PORT; log $SRVLOG"
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/healthz" 2>/dev/null | python -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d['status']=='ok' and d['shadow'].get('loaded') else 1)" 2>/dev/null; then break; fi
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/healthz" > "$OUT/${RUN}_healthz_start.json" || { echo "server never became ready"; tail -30 "$SRVLOG"; kill $(cat $PIDF) || true; exit 3; }
python -c "import json; d=json.load(open('$OUT/${RUN}_healthz_start.json')); print('primary', d['backend'], d['model_version'], '| shadow', d['shadow']['backend'], d['shadow']['model_version'], d['shadow']['conf_single'], d['shadow']['conf_seq'])"

T0=$(date +%s)
# ---- 1) one pyro-sdis VAL sequence (cabanelle, VAL site) with real per-frame ts
python -m src.serve.replay_client --sequence-json data/v3/sequences/sequences_val.json \
  --seq-id val_force-06_cabanelle-125_20240403T101630 --server "http://127.0.0.1:$PORT" \
  --device-id replay-sdis-cabanelle-125 --out "$OUT/${RUN}_replay_sdis_val.jsonl" \
  --run-tag "${RUN}-sdis-val-cabanelle-125" | tail -6
# ---- 2) one Boreal heinola NON-SMOKE clip (aerial VAL; 1 fps frames)
python -m src.serve.replay_client --frames data/v4/sequences/boreal_frames/heinola_103 \
  --fps 1 --server "http://127.0.0.1:$PORT" --device-id replay-boreal-heinola_103 \
  --out "$OUT/${RUN}_replay_boreal_heinola_103.jsonl" --run-tag "${RUN}-boreal-heinola_103" | tail -6
# ---- 3) two posts WITHOUT a source field -> default tag live_upload (still stored frames!)
for f in data/v4/sequences/boreal_frames/heinola_103/heinola_103_t0000.jpg \
         "$(python -c "import json; s=[x for x in json.load(open('data/v3/sequences/sequences_val.json')) if x['seq_id']=='val_force-06_cabanelle-125_20240403T101630'][0]; print(sorted(s['frames'],key=lambda f:f['ts'])[1]['image_path'])")"; do
  curl -sf -F "image=@$f" -F device_id=demo-default-source-curl -F capture_ts=1700000000 \
       -F 'metadata={"note":"default source demo - stored frame, not a field upload"}' \
       "http://127.0.0.1:$PORT/detect" | python -c "import sys,json; d=json.load(sys.stdin); print('default-source post:', d['device_id'], d['max_smoke_score'], d['alert']['state'], d['latency_ms'], 'ms')"
done
T1=$(date +%s)

# ---- wait for the shadow worker to drain, then snapshot /healthz
for i in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$PORT/healthz" | python -c "import sys,json; s=json.load(sys.stdin)['shadow']; sys.exit(0 if s['queue_depth']==0 and s['records_written']>=61 else 1)"; then break; fi
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/healthz" > "$OUT/${RUN}_healthz_end.json"
echo "wall for 61 posts: $((T1-T0)) s; shadow drained at +$(( $(date +%s) - T1 )) s after last post"

# ---- stop the demo server (only this pid), prove :8000 is untouched
kill "$(cat $PIDF)"; sleep 2; (kill -0 "$(cat $PIDF)" 2>/dev/null && kill -9 "$(cat $PIDF)") || true
ss -ltn | grep -q ":${PORT} " && echo "WARNING port $PORT still bound" || echo "demo server on :$PORT stopped"
curl -sf http://127.0.0.1:8000/healthz | python -c "import sys,json; d=json.load(sys.stdin); print('production :8000 still', d['status'], d['model_version'], 'uptime_s', d['uptime_s'], '| shadow:', d.get('shadow','(pre-v5 build, no field)'))"
echo "RUN=$RUN"; echo "SHADOW_LOG=$LOG"
