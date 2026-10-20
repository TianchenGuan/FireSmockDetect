#!/bin/bash
# Integration test for the dino3 (LoRA) serving backend: start the FastAPI app
# on a SPARE port with env-var configuration only, POST real images to
# /detect, check the JSON contract + /healthz, then stop the server we started.
# Never touches the production service on port 8000 (hard refusal below).
#
#   bash scripts/v4/integration_dino3_service.sh [img1 img2 img3 ...]
# env: PORT (default 8011), WEIGHTS (B4 seg05 best.pt), OUT (logs/v4/integration_dino3),
#      CONF_SINGLE / CONF_SEQ (val-selected thresholds; NOT the YOLO 0.30/0.15)
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

PORT="${PORT:-8011}"
WEIGHTS="${WEIGHTS:-models/v3/runs/v4B4_lora_L/seg05/best.pt}"
MODEL_VERSION="${MODEL_VERSION:-v4B4_lora_L@seg05-step15026}"
# val-selected by the v3 rule (reports/v3/trackBC_results.md 2c): B4 conf at which its
# sdis_val neg-image FP equals the deployed expB@1280 model's at 0.30 / 0.15
# (34.0% / 45.3%) -> 0.93 / 0.82. Low-FP alternatives (5% / 10% FP): 0.9934 / 0.9881.
CONF_SINGLE="${CONF_SINGLE:-0.93}"
CONF_SEQ="${CONF_SEQ:-0.82}"
OUT="${OUT:-logs/v4/integration_dino3}"
mkdir -p "$OUT"

[ "$PORT" = "8000" ] && { echo "refusing to use the production port 8000"; exit 2; }
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then echo "port ${PORT} already in use"; exit 2; fi

if [ "$#" -ge 1 ]; then IMGS=("$@"); else
  mapfile -t IMGS < <(head -n 3 data/yolo/lists/local/sdis_val.txt); fi
echo "images: ${IMGS[*]}"

echo "== starting server on :$PORT (backend=dino3, weights=$WEIGHTS) =="
SMOKE_BACKEND=dino3 SMOKE_WEIGHTS_PATH="$WEIGHTS" SMOKE_MODEL_VERSION="$MODEL_VERSION" \
SMOKE_SINGLE_FRAME_ALERT_CONF="$CONF_SINGLE" SMOKE_SEQ_FRAME_CONF="$CONF_SEQ" \
SMOKE_PORT="$PORT" \
  python -m uvicorn src.serve.app:app --host 127.0.0.1 --port "$PORT" \
  > "$OUT/server.log" 2>&1 &
PID=$!
cleanup() { kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; }
trap cleanup EXIT

T0=$(date +%s)
for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:${PORT}/healthz" -o "$OUT/healthz.json" 2>/dev/null; then break; fi
  kill -0 "$PID" 2>/dev/null || { echo "server exited early"; tail -40 "$OUT/server.log"; exit 1; }
  sleep 2
done
echo "ready after $(( $(date +%s) - T0 ))s"
echo "== GET /healthz =="; cat "$OUT/healthz.json"; echo

TS=$(python -c "import time; print(time.time())")
i=0
for img in "${IMGS[@]}"; do
  i=$((i+1))
  TS=$(python -c "print($TS + 10)")   # monotonically increasing capture_ts, same device
  S=$(date +%s.%N)
  curl -sf -F "image=@${img}" -F device_id=itest-dino3 -F "capture_ts=${TS}" \
       "http://127.0.0.1:${PORT}/detect" -o "$OUT/detect_${i}.json"
  E=$(date +%s.%N)
  echo "== POST /detect #$i ($img) wall $(python -c "print(round(($E-$S)*1000,1))") ms =="
  cat "$OUT/detect_${i}.json"; echo
done

echo "== contract check =="
python - "$OUT" "${#IMGS[@]}" <<'PY'
import json, sys
out, n = sys.argv[1], int(sys.argv[2])
h = json.load(open(f"{out}/healthz.json"))
assert h["status"] == "ok" and h["backend"] == "dino3", h
assert h.get("backend_info", {}).get("lora_r", 0) > 0, h
need = {"detections", "n_smoke", "max_smoke_score", "alert", "model_version", "image_size",
        "device_id", "capture_ts", "client_metadata_echo", "score_semantics",
        "latency_ms", "preprocess_ms", "infer_ms", "postprocess_ms", "full_pass_ms",
        "tile_pass_ms", "n_tiles"}
alert_need = {"state", "reason", "window_hits", "window_size", "history_reset",
              "cooldown_active", "note"}
det_need = {"bbox_xyxy", "score", "cls_id", "cls_name", "source"}
rows = []
for i in range(1, n + 1):
    d = json.load(open(f"{out}/detect_{i}.json"))
    missing = need - set(d)
    assert not missing, f"detect_{i} missing {missing}"
    assert alert_need <= set(d["alert"]), d["alert"]
    for det in d["detections"]:
        assert det_need <= set(det), det
        assert det["cls_name"] == "smoke" and det["source"] == "full"
        assert len(det["bbox_xyxy"]) == 4 and 0.0 <= det["score"] <= 1.0
    assert d["alert"]["state"] in ("none", "watch", "provisional", "confirmed")
    rows.append({"i": i, "n_dets": len(d["detections"]), "n_smoke": d["n_smoke"],
                 "max_smoke_score": d["max_smoke_score"], "alert_state": d["alert"]["state"],
                 "latency_ms": d["latency_ms"], "infer_ms": d["infer_ms"],
                 "image_size": d["image_size"], "model_version": d["model_version"]})
print("healthz backend_info:", json.dumps(h["backend_info"]))
for r in rows: print(json.dumps(r))
print("CONTRACT OK")
PY
echo "== stopping server pid $PID =="
cleanup
trap - EXIT
sleep 1
ss -ltn 2>/dev/null | grep -q ":${PORT} " && { echo "port still bound!"; exit 1; } || echo "port ${PORT} released"
echo "server log tail:"; tail -n 8 "$OUT/server.log"
