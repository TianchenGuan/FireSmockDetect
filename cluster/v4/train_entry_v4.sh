#!/bin/bash
# v4 segmented training entrypoint. Fork of cluster/train_entry.sh (v6) plus:
#   - DATA_TARS: comma-sep "name=presignedGET" tarballs staged ONCE to
#     $V4_DATA_ROOT on /scratch (FSx, persists across jobs); .done markers skip.
#   - DATA_YAML_B64: base64 dataset yaml written pod-side (arbitrary mixtures
#     without image rebuilds); overrides DATA_YAML when set.
#   - SEED passed through to train_yolo.py.
# Fetched at job start via presigned URL_ENTRY (no rebuild needed).
# Exit codes identical to v6: 0 ok, 90/91 budget-skip, 100+bitmask else,
# 95 = data staging failed.
set -u +e
# self-log: pods have no log channel, so the entry captures its own
# stdout/stderr and ships it through the args channel (see upload()).
exec > /tmp/entry.log 2>&1
BITS=0
NOW=$(date +%s)
BUDGET=$(( ${URL_EXPIRY_EPOCH:-$((NOW+2700))} - NOW - 360 ))
echo "segment budget: ${BUDGET}s"

RUN_ROOT=""
for d in "${RUN_ROOT_PREF:-/scratch/scratch/tianrgua-sandbox/smoke/runs}" /mnt/local/runs /tmp/runs; do
  if mkdir -p "$d" 2>/dev/null && touch "$d/.w" 2>/dev/null; then RUN_ROOT="$d"; rm -f "$d/.w"; break; fi
done
RUN_DIR=$RUN_ROOT/$EXP_NAME
mkdir -p "$RUN_DIR/weights"
CONSOLE=/tmp/segment_console.log
touch "$CONSOLE"
STATUS=/tmp/status.json

put() { [ -f "$2" ] && curl -sf --max-time 180 -T "$2" "$1"; }

upload() {
  ok=1
  put "$URL_LAST"    "$RUN_DIR/weights/last.pt"  || ok=0
  put "$URL_BEST"    "$RUN_DIR/weights/best.pt"
  put "$URL_RESULTS" "$RUN_DIR/results.csv"
  { echo "=== entry.log ==="; tail -c 6000 /tmp/entry.log;
    echo "=== run dir ==="; ls -la "$RUN_DIR" "$RUN_DIR/weights" 2>&1;
    echo "=== ultralytics console ==="; tail -c 8000 "$CONSOLE" 2>&1;
  } > /tmp/console_tail.txt
  put "$URL_CONSOLE" /tmp/console_tail.txt
  put "$URL_ARGS" "$RUN_DIR/args.yaml"
  curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS" || ok=0
  return $((1-ok))
}

write_status() {
  EPOCH_LINE=$(tail -1 "$RUN_DIR/results.csv" 2>/dev/null | cut -c1-200)
  GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  cat > $STATUS <<EOF
{"exp":"$EXP_NAME","phase":"$1","train_rc":"$2","budget_s":$BUDGET,"gpus":$GPUS,
 "run_root":"$RUN_ROOT","resumed_from":"${RESUMED_FROM:-none}","entry":"v4",
 "last_results_line":"${EPOCH_LINE//\"/}","host":"$(hostname)","ts":"$(date -u +%FT%TZ)"}
EOF
}

write_status "starting" "-"
curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS" && echo beacon=1 || echo beacon=0

if [ "$BUDGET" -lt 600 ]; then
  write_status "skipped_budget_too_small" "-"
  upload && exit 90
  exit 91
fi

# baked E data (pyro-sdis + D-Fire) same as v6
if [ -d /mnt/local/smoke_data ] || cp -r /opt/smoke_data /mnt/local/smoke_data 2>/dev/null || { mkdir -p /tmp/sd && cp -r /opt/smoke_data /tmp/sd/smoke_data && ln -sfn /tmp/sd/smoke_data /mnt/local/smoke_data 2>/dev/null; }; then
  BITS=$((BITS|1))
fi

# v4 new-data tars -> /scratch, once per cluster (marker files)
V4_DATA_ROOT="${V4_DATA_ROOT:-/scratch/scratch/tianrgua-sandbox/smoke/v4data}"
if [ -n "${DATA_TARS:-}" ]; then
  mkdir -p "$V4_DATA_ROOT" || { write_status "v4data_root_unwritable" "-"; upload; exit 95; }
  IFS=',' read -ra TARS <<< "$DATA_TARS"
  for spec in "${TARS[@]}"; do
    name="${spec%%=*}"; url="${spec#*=}"
    if [ -f "$V4_DATA_ROOT/.${name}.done" ]; then echo "tar $name already staged"; continue; fi
    echo "staging tar $name ..."
    if curl -sf --max-time 2400 -o "$V4_DATA_ROOT/${name}.tar" "$url" \
       && tar -xf "$V4_DATA_ROOT/${name}.tar" -C "$V4_DATA_ROOT"; then
      rm -f "$V4_DATA_ROOT/${name}.tar"
      touch "$V4_DATA_ROOT/.${name}.done"
      echo "tar $name staged ok"
    else
      rm -f "$V4_DATA_ROOT/${name}.tar"
      write_status "v4data_stage_failed_$name" "-"
      upload
      exit 95
    fi
  done
fi

# dataset yaml delivered inline (base64) -> arbitrary mixtures, no rebuild
if [ -n "${DATA_YAML_B64:-}" ]; then
  echo "$DATA_YAML_B64" | base64 -d > /tmp/v4data.yaml || { write_status "yaml_decode_failed" "-"; upload; exit 95; }
  DATA_YAML=/tmp/v4data.yaml
  echo "--- v4 data yaml ---"; cat /tmp/v4data.yaml; echo "--------------------"
fi

# recompute the segment budget AFTER staging: large first-segment tars can eat
# minutes, and the presigned upload URLs expire at URL_EXPIRY_EPOCH regardless
# (v4/v5b first segments lost their uploads this way; training itself resumed
# from the persistent /scratch run dir).
NOW=$(date +%s)
BUDGET=$(( ${URL_EXPIRY_EPOCH:-$((NOW+2700))} - NOW - 360 ))
echo "segment budget after staging: ${BUDGET}s"
if [ "$BUDGET" -lt 600 ]; then
  write_status "skipped_budget_too_small_after_staging" "-"; upload && exit 90; exit 91
fi

MODEL="/opt/weights/${BASE_WEIGHTS:-yolo11s.pt}"
# curriculum stage-2: initialize from a prior run's best.pt (fresh training,
# not resume) — only when no resume checkpoint exists for THIS run
if [ -n "${URL_INIT:-}" ] && [ ! -f "$RUN_DIR/weights/last.pt" ]; then
  if curl -sf --max-time 300 -o /tmp/init.pt "$URL_INIT"; then
    MODEL=/tmp/init.pt; RESUMED_FROM=init_ckpt
  else
    write_status "init_ckpt_fetch_failed" "-"; upload; exit 95
  fi
fi
RESUME=""
if [ -f "$RUN_DIR/weights/last.pt" ]; then
  MODEL="$RUN_DIR/weights/last.pt"; RESUME="--resume"; RESUMED_FROM=local
elif [ -n "${URL_RESUME:-}" ]; then
  if curl -sf --max-time 300 -o "$RUN_DIR/weights/last.pt" "$URL_RESUME"; then
    MODEL="$RUN_DIR/weights/last.pt"; RESUME="--resume"; RESUMED_FROM=s3
  else
    rm -f "$RUN_DIR/weights/last.pt"
  fi
fi

write_status "training" "-"
curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS"

cd /opt/weights
timeout --signal=INT --kill-after=180 "$BUDGET" \
  python /opt/train_yolo.py --data "$DATA_YAML" --model "$MODEL" \
    --epochs "${EPOCHS:-60}" --imgsz "${IMGSZ:-1280}" --batch "${BATCH:-64}" \
    --device "${DEVICES:-0,1,2,3,4,5,6,7}" --project "$RUN_ROOT" \
    --name "$EXP_NAME" --workers "${WORKERS:-8}" --seed "${SEED:-0}" $RESUME \
    > "$CONSOLE" 2>&1
RC=$?
echo "train rc=$RC"
LAST_EPOCH=$(tail -1 "$RUN_DIR/results.csv" 2>/dev/null | cut -d, -f1 | tr -dc 0-9)
DONE=0
grep -q TRAIN_DONE "$CONSOLE" && DONE=1
[ -n "$LAST_EPOCH" ] && [ "$LAST_EPOCH" -ge "${EPOCHS:-60}" ] && DONE=1
{ [ $RC -eq 0 ] || [ $RC -eq 124 ]; } && BITS=$((BITS|2))
[ $DONE -eq 1 ] && BITS=$((BITS|4))
write_status "$([ $DONE -eq 1 ] && echo done || echo segment_end)" "$RC"
upload && BITS=$((BITS|8))
if [ $((BITS & 2)) -ne 0 ] && [ $((BITS & 8)) -ne 0 ]; then exit 0; fi
exit $((100+BITS))
