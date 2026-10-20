#!/bin/bash
# Track B/C detection segment entrypoint (green g6e pods, offline).
# Fetches code tarball + COCO jsons via presigned GET, trains (or precomputes
# 7B features) with a time budget, persists run state on /scratch, uploads
# artifacts via presigned PUT. Exit codes mirror the v2 pattern:
#   0 = clean segment; 90/91 = skipped (low budget); else 100+bits
#   (bit0 fetch ok, bit1 clean train exit, bit2 DONE, bit3 uploads ok).
set -u +e
# uid 1001 has no passwd entry -> pin every cache/home to /tmp (v2 lesson)
export HOME=/tmp USER=worker LOGNAME=worker
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache TRITON_CACHE_DIR=/tmp/triton_cache
export HF_HOME=/tmp/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/inductor_cache /tmp/triton_cache /tmp/hf
BITS=0
NOW=$(date +%s)
BUDGET=$(( ${URL_EXPIRY_EPOCH:-$((NOW+2700))} - NOW - 420 ))
echo "budget ${BUDGET}s"
RUN_ROOT=/scratch/scratch/tianrgua-sandbox/smoke/runs_v3
RUN_DIR=$RUN_ROOT/$RUN_NAME
mkdir -p "$RUN_DIR"
STATUS=/tmp/status.json

put() { [ -f "$2" ] && curl -sf --max-time 600 -T "$2" "$1"; }
write_status() {
  # bare numbers only — embedded quotes would corrupt the JSON below
  LOSS=$(grep -oE '"loss": [0-9.]+' "$RUN_DIR/metrics.jsonl" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  STEP=$(grep -oE '"step": [0-9]+' "$RUN_DIR/metrics.jsonl" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
  GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  DFREE=$(df -B1G /scratch 2>/dev/null | awk 'NR==2{print $4}')
  cat > $STATUS <<EOF
{"run":"$RUN_NAME","phase":"$1","rc":"$2","budget_s":$BUDGET,"gpus":$GPUS,
 "last":"${STEP:-none} ${LOSS:-none}","scratch_free_gb":"${DFREE:-na}",
 "host":"$(hostname)","ts":"$(date -u +%FT%TZ)"}
EOF
}
upload() {
  ok=1
  write_status "$1" "$2"
  put "$URL_LAST" "$RUN_DIR/last.pt"
  put "$URL_BEST" "$RUN_DIR/best.pt"
  put "$URL_METRICS" "$RUN_DIR/metrics.jsonl"
  if [ -f "$RUN_DIR/console.log" ]; then
    grep -m4 -B3 -A25 -E "Traceback|OutOfMemoryError|RuntimeError|ValueError|AssertionError" \
      "$RUN_DIR/console.log" | head -c 20000 > /tmp/ct.txt
    echo "===== TAIL =====" >> /tmp/ct.txt
    tail -c 12000 "$RUN_DIR/console.log" >> /tmp/ct.txt
    put "$URL_CONSOLE" /tmp/ct.txt
  fi
  curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS" || ok=0
  return $((1-ok))
}

write_status starting -
curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS" && echo beacon=1 || echo beacon=0
[ "$BUDGET" -lt 900 ] && { write_status skipped_budget -; curl -sf -T "$STATUS" "$URL_STATUS" && exit 90; exit 91; }

mkdir -p /tmp/job
F=1
curl -sf --max-time 180 -o /tmp/job/code.tgz "$URL_CODE" || F=0
tar xzf /tmp/job/code.tgz -C /tmp/job || F=0
curl -sf --max-time 180 -o /tmp/job/instances_train.json "$URL_TRAIN_JSON" || F=0
[ -n "${URL_VAL_JSON:-}" ] && { curl -sf --max-time 180 -o /tmp/job/instances_val.json "$URL_VAL_JSON" || F=0; }
[ $F -eq 1 ] && BITS=$((BITS|1))

DATA_ROOT=/scratch/scratch/tianrgua-sandbox/smoke_data_v2
[ -d "$DATA_ROOT" ] || { echo "data root missing"; write_status data_missing -; upload data_missing -; exit $((100+BITS)); }
if [ -n "${MODEL_DIR:-}" ]; then
  [ -d "$MODEL_DIR" ] || { echo "MODEL_DIR missing"; write_status model_missing -; upload model_missing -; exit $((100+BITS)); }
fi
# resume: /scratch run dir survives across segments; S3 fallback if scrubbed
RESUME_ARG=""
if [ -f "$RUN_DIR/last.pt" ]; then
  RESUME_ARG="--resume $RUN_DIR/last.pt"
elif [ -n "${URL_RESUME:-}" ]; then
  curl -sf --max-time 300 -o "$RUN_DIR/last.pt" "$URL_RESUME" && RESUME_ARG="--resume $RUN_DIR/last.pt"
fi

VAL_ARG=""
[ -f /tmp/job/instances_val.json ] && VAL_ARG="--val-json /tmp/job/instances_val.json"
BB_ARG=""
[ -n "${MODEL_DIR:-}" ] && BB_ARG="--backbone-dir $MODEL_DIR"
[ -n "${RANDOM_SPEC:-}" ] && BB_ARG="--random-spec $RANDOM_SPEC"

write_status running -
curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS"
TRAIN_BUDGET=$((BUDGET-300))

if [ "${MODE:-train}" = "precompute" ]; then
  timeout --signal=TERM --kill-after=240 "$BUDGET" \
    torchrun --nproc_per_node "${NGPUS:-8}" /tmp/job/precompute_feats.py \
      --backbone-dir "$MODEL_DIR" \
      --coco-json /tmp/job/instances_train.json ${VAL_ARG:+/tmp/job/instances_val.json} \
      --image-root "$DATA_ROOT" \
      --out-root "${FEAT_OUT:?}" \
      --width "${WIDTH:-1280}" --height "${HEIGHT:-736}" \
      --bs "${BS:-2}" --workers "${WORKERS:-4}" ${EXTRA_ARGS:-} \
      --time-budget-s "$TRAIN_BUDGET" \
      > "$RUN_DIR/console.log" 2>&1
  RC=$?
  DONE=$(ls "${FEAT_OUT}"/.done_rank* 2>/dev/null | wc -l)
  [ "$DONE" -ge "${NGPUS:-8}" ] && DONE=1 || DONE=0
else
  timeout --signal=TERM --kill-after=240 "$BUDGET" \
    torchrun --nproc_per_node "${NGPUS:-8}" /tmp/job/train_det.py \
      $BB_ARG $RESUME_ARG $VAL_ARG \
      ${FEAT_ROOT:+--feat-root "$FEAT_ROOT"} ${EMBED_DIM:+--embed-dim "$EMBED_DIM"} \
      --unfreeze-top "${UNFREEZE_TOP:-0}" \
      --train-json /tmp/job/instances_train.json \
      --image-root "$DATA_ROOT" \
      --width "${WIDTH:-1280}" --height "${HEIGHT:-736}" \
      --epochs "${EPOCHS:-12}" --max-steps "${MAX_STEPS:--1}" \
      --bs "${BS:-4}" --lr "${LR:-1e-4}" --wd "${WD:-1e-4}" \
      --warmup-steps "${WARMUP:-500}" --workers "${WORKERS:-4}" \
      --overfit-n "${OVERFIT_N:-0}" \
      --save-every-steps "${SAVE_STEPS:-500}" \
      --eval-max-images "${EVAL_MAX:-1500}" \
      --eval-every-epochs "${EVAL_EVERY:-1}" \
      --output-dir "$RUN_DIR" ${EXTRA_ARGS:-} \
      --time-budget-s "$TRAIN_BUDGET" \
      > "$RUN_DIR/console.log" 2>&1
  RC=$?
  [ -f "$RUN_DIR/DONE" ] && DONE=1 || DONE=0
fi
echo "rc=$RC done=$DONE"
{ [ $RC -eq 0 ] || [ $RC -eq 124 ]; } && BITS=$((BITS|2))
[ $DONE -eq 1 ] && BITS=$((BITS|4))
upload "$([ $DONE -eq 1 ] && echo done || echo segment_end)" "$RC" && BITS=$((BITS|8))
{ [ $((BITS & 2)) -ne 0 ] && [ $((BITS & 8)) -ne 0 ]; } && exit 0
exit $((100+BITS))
