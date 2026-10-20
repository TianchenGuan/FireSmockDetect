#!/bin/bash
# VLM SFT segment entrypoint (green g6e pods, offline).
# Fetches latest code/config/data via presigned GETs, trains with a time
# budget, persists run state on /scratch, returns artifacts via presigned PUT.
# exit 0 = clean segment (train exited 0/124 + uploads ok); 90/91 = skipped;
# else 100+bits (bit0 fetch ok, bit1 clean train exit, bit2 DONE, bit3 uploads).
set -u +e
# pod runs as uid 1001 with no passwd entry -> getpass.getuser() crashes in
# torch inductor cache-dir resolution; pin every cache/home to /tmp instead.
export HOME=/tmp USER=worker LOGNAME=worker
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache TRITON_CACHE_DIR=/tmp/triton_cache
export HF_HOME=/tmp/hf DS_BUILD_OPS=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/inductor_cache /tmp/triton_cache /tmp/hf
BITS=0
NOW=$(date +%s)
BUDGET=$(( ${URL_EXPIRY_EPOCH:-$((NOW+2700))} - NOW - 420 ))
echo "budget ${BUDGET}s"
RUN_ROOT=/scratch/scratch/tianrgua-sandbox/smoke/runs_v2
RUN_DIR=$RUN_ROOT/$RUN_NAME
mkdir -p "$RUN_DIR"
STATUS=/tmp/status.json

put() { [ -f "$2" ] && curl -sf --max-time 300 -T "$2" "$1"; }
upload() {
  ok=1
  write_status "$1" "$2"
  [ -d "$RUN_DIR/final_adapter" ] && tar czf /tmp/final_adapter.tgz -C "$RUN_DIR" final_adapter params.json 2>/dev/null
  put "$URL_ADAPTER" /tmp/final_adapter.tgz
  if [ -f "$RUN_DIR/console.log" ]; then
    grep -m4 -B3 -A25 -E "Traceback|OutOfMemoryError|RuntimeError|ValueError|AssertionError" "$RUN_DIR/console.log" | head -c 20000 > /tmp/ct.txt
    echo "===== TAIL =====" >> /tmp/ct.txt
    tail -c 12000 "$RUN_DIR/console.log" >> /tmp/ct.txt
    put "$URL_CONSOLE" /tmp/ct.txt
  fi
  [ -f "$RUN_DIR/trainer_log.jsonl" ] && put "$URL_TLOG" "$RUN_DIR/trainer_log.jsonl"
  curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS" || ok=0
  return $((1-ok))
}
write_status() {
  LOSS=$(grep -oE "loss[\":' ]+[0-9.]+" "$RUN_DIR/console.log" 2>/dev/null | tail -1)
  STEP=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sed "s/.*checkpoint-//" | sort -n | tail -1)
  GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  cat > $STATUS <<EOF
{"run":"$RUN_NAME","phase":"$1","rc":"$2","budget_s":$BUDGET,"gpus":$GPUS,
 "last_ckpt_step":"${STEP:-none}","last_loss":"${LOSS:-none}",
 "host":"$(hostname)","ts":"$(date -u +%FT%TZ)"}
EOF
}

write_status starting -
curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS" && echo beacon=1 || echo beacon=0
[ "$BUDGET" -lt 900 ] && { write_status skipped_budget -; curl -sf -T "$STATUS" "$URL_STATUS" && exit 90; exit 91; }

mkdir -p /tmp/job
F=1
curl -sf --max-time 120 -o /tmp/job/train_vlm.py "$URL_CODE" || F=0
curl -sf --max-time 120 -o /tmp/job/ds.json "$URL_DS" || F=0
curl -sf --max-time 300 -o /tmp/job/train.jsonl "$URL_DATA" || F=0
[ -n "${URL_VAL:-}" ] && curl -sf --max-time 300 -o /tmp/job/val.jsonl "$URL_VAL"
[ $F -eq 1 ] && BITS=$((BITS|1))
[ -d "$MODEL_DIR" ] || { echo "MODEL_DIR missing"; write_status model_missing -; upload model_missing -; exit $((100+BITS)); }
[ -d /scratch/scratch/tianrgua-sandbox/smoke_data_v2 ] || { echo "data root missing"; write_status data_missing -; upload data_missing -; exit $((100+BITS)); }

VAL_ARG=""
[ -f /tmp/job/val.jsonl ] && VAL_ARG="--val /tmp/job/val.jsonl"
write_status training -
curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS"

DS_ARG="--deepspeed /tmp/job/ds.json"
[ "${NO_DS:-0}" = "1" ] && DS_ARG=""
timeout --signal=TERM --kill-after=240 "$BUDGET" \
  deepspeed --num_gpus "${NGPUS:-8}" /tmp/job/train_vlm.py \
    --model-path "$MODEL_DIR" \
    --data /tmp/job/train.jsonl $VAL_ARG \
    --image-root-map "/local/home/tianrgua/cvproject/data/yolo=/scratch/scratch/tianrgua-sandbox/smoke_data_v2" \
    --output-dir "$RUN_DIR" \
    --max-steps "${MAX_STEPS:--1}" --epochs "${EPOCHS:-1}" \
    --per-device-bs "${BS:-1}" --grad-accum "${GA:-8}" --lr "${LR:-1e-4}" \
    --lora-r "${LORA_R:-16}" --lora-alpha "${LORA_ALPHA:-32}" \
    --max-pixels "${MAX_PIXELS:-1048576}" \
    $DS_ARG --save-steps "${SAVE_STEPS:-200}" \
    --workers "${WORKERS:-4}" ${EXTRA_ARGS:-} \
    > "$RUN_DIR/console.log" 2>&1
RC=$?
echo "train rc=$RC"
grep -q VLM_TRAIN_DONE "$RUN_DIR/console.log" && DONE=1 || DONE=0
{ [ $RC -eq 0 ] || [ $RC -eq 124 ]; } && BITS=$((BITS|2))
[ $DONE -eq 1 ] && BITS=$((BITS|4))
upload "$([ $DONE -eq 1 ] && echo done || echo segment_end)" "$RC" && BITS=$((BITS|8))
{ [ $((BITS & 2)) -ne 0 ] && [ $((BITS & 8)) -ne 0 ]; } && exit 0
exit $((100+BITS))
