#!/bin/bash
# v6 segmented training entrypoint. Fork of cluster/v4/train_entry_v4.sh plus:
#   - TRAIN_SCRIPT: pod path of the training script (default /opt/train_yolo.py);
#     lets arms ship custom trainers (e.g. KD) via DATA_TARS without rebuilds.
#   - EXTRA_ARGS: extra CLI args appended to the trainer invocation.
# Exit codes: 0 ok, 90/91 budget-skip, 95 staging failed, 100+bitmask else.
set -u +e
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
    echo "=== trainer console ==="; tail -c 8000 "$CONSOLE" 2>&1;
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
 "run_root":"$RUN_ROOT","resumed_from":"${RESUMED_FROM:-none}","entry":"v6",
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

# baked recipe data (pyro-sdis + D-Fire)
if [ -d /mnt/local/smoke_data ] || cp -r /opt/smoke_data /mnt/local/smoke_data 2>/dev/null || { mkdir -p /tmp/sd && cp -r /opt/smoke_data /tmp/sd/smoke_data && ln -sfn /tmp/sd/smoke_data /mnt/local/smoke_data 2>/dev/null; }; then
  BITS=$((BITS|1))
fi

# staged tars -> /scratch, once per cluster (marker files); name@/abs/root overrides root
V4_DATA_ROOT="${V4_DATA_ROOT:-/scratch/scratch/tianrgua-sandbox/smoke/v6data}"
if [ -n "${DATA_TARS:-}" ]; then
  IFS=',' read -ra TARS <<< "$DATA_TARS"
  for spec in "${TARS[@]}"; do
    name="${spec%%=*}"; url="${spec#*=}"
    root="$V4_DATA_ROOT"
    case "$name" in *@*) root="${name#*@}"; name="${name%%@*}";; esac
    mkdir -p "$root" || { write_status "v6data_root_unwritable" "-"; upload; exit 95; }
    if [ -f "$root/.${name}.done" ]; then echo "tar $name already staged"; continue; fi
    echo "staging tar $name ..."
    if curl -sf --max-time 2400 -o "$root/${name}.tar" "$url" \
       && tar -xf "$root/${name}.tar" -C "$root"; then
      rm -f "$root/${name}.tar"
      touch "$root/.${name}.done"
      echo "tar $name staged ok"
    else
      rm -f "$root/${name}.tar"
      write_status "v6data_stage_failed_$name" "-"
      upload
      exit 95
    fi
  done
fi

# dataset yaml delivered inline (base64)
if [ -n "${DATA_YAML_B64:-}" ]; then
  echo "$DATA_YAML_B64" | base64 -d > /tmp/v6data.yaml || { write_status "yaml_decode_failed" "-"; upload; exit 95; }
  DATA_YAML=/tmp/v6data.yaml
  echo "--- v6 data yaml ---"; cat /tmp/v6data.yaml; echo "--------------------"
fi

# recompute the segment budget AFTER staging (presigned PUT URLs expire at
# URL_EXPIRY_EPOCH regardless of how long staging took)
NOW=$(date +%s)
BUDGET=$(( ${URL_EXPIRY_EPOCH:-$((NOW+2700))} - NOW - 360 ))
echo "segment budget after staging: ${BUDGET}s"
if [ "$BUDGET" -lt 600 ]; then
  write_status "skipped_budget_too_small_after_staging" "-"; upload && exit 90; exit 91
fi

MODEL="/opt/weights/${BASE_WEIGHTS:-yolo11s.pt}"
# initialize from a shipped checkpoint (fresh training) when no resume exists
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

TRAIN_SCRIPT="${TRAIN_SCRIPT:-/opt/train_yolo.py}"
if [ ! -f "$TRAIN_SCRIPT" ]; then
  write_status "train_script_missing" "-"; upload; exit 95
fi
cd /opt/weights
timeout --signal=INT --kill-after=180 "$BUDGET" \
  python "$TRAIN_SCRIPT" --data "$DATA_YAML" --model "$MODEL" \
    --epochs "${EPOCHS:-60}" --imgsz "${IMGSZ:-1280}" --batch "${BATCH:-64}" \
    --device "${DEVICES:-0,1,2,3,4,5,6,7}" --project "$RUN_ROOT" \
    --name "$EXP_NAME" --workers "${WORKERS:-8}" --seed "${SEED:-0}" $RESUME \
    ${EXTRA_ARGS:-} \
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
