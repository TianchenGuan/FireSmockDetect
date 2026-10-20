#!/bin/bash
# GRPO e2e feasibility probe entrypoint (green g6e pods, offline).
# One-shot (no segment resume): fetches code + probe data + SFT adapter via
# presigned GETs, runs 20 GRPO steps on 8 GPUs, persists to /scratch and
# returns artifacts via presigned PUT.
# exit 0 = clean (train rc 0/124 + uploads ok); 90/91 = budget skip;
# else 100+bits (bit0 fetch ok, bit1 clean train exit, bit2 DONE, bit3 uploads).
set -u +e
# uid 1001 has no passwd entry -> pin HOME/caches (iteration-2 lesson)
export HOME=/tmp USER=worker LOGNAME=worker
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache TRITON_CACHE_DIR=/tmp/triton_cache
export HF_HOME=/tmp/hf DS_BUILD_OPS=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/inductor_cache /tmp/triton_cache /tmp/hf
BITS=0
NOW=$(date +%s)
# decoupled from URL expiry: results persist on /scratch and are retrieved
# by a fetch job, so the training budget can exceed the presigned window.
BUDGET=${TRAIN_BUDGET:-12600}
echo "budget ${BUDGET}s"
RUN_ROOT=/scratch/scratch/tianrgua-sandbox/smoke/runs_v2
RUN_DIR=$RUN_ROOT/${RUN_NAME:-grpo_e2e}/attempt_$NOW
mkdir -p "$RUN_DIR"
STATUS=/tmp/status.json

put() { [ -f "$2" ] && curl -sf --max-time 300 -T "$2" "$1"; }
write_status() {
  GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  STEPS=$(grep -c '"step"' "$RUN_DIR/trainer_log.jsonl" 2>/dev/null || echo 0)
  cat > $STATUS <<EOF
{"run":"${RUN_NAME:-grpo_e2e}","phase":"$1","rc":"$2","budget_s":$BUDGET,
 "gpus":$GPUS,"log_lines":"$STEPS","run_dir":"$RUN_DIR",
 "host":"$(hostname)","ts":"$(date -u +%FT%TZ)"}
EOF
}
upload() {
  ok=1
  write_status "$1" "$2"
  [ -d "$RUN_DIR/final_adapter" ] && tar czf /tmp/final_adapter.tgz -C "$RUN_DIR" final_adapter 2>/dev/null
  put "$URL_ADAPTER_OUT" /tmp/final_adapter.tgz
  if [ -f "$RUN_DIR/console.log" ]; then
    grep -m4 -B3 -A25 -E "Traceback|OutOfMemoryError|RuntimeError|ValueError|AssertionError" "$RUN_DIR/console.log" | head -c 20000 > /tmp/ct.txt
    echo "===== TAIL =====" >> /tmp/ct.txt
    tail -c 12000 "$RUN_DIR/console.log" >> /tmp/ct.txt
    put "$URL_CONSOLE" /tmp/ct.txt
  fi
  [ -f "$RUN_DIR/trainer_log.jsonl" ] && put "$URL_TLOG" "$RUN_DIR/trainer_log.jsonl"
  cat "$RUN_DIR"/reward_log.jsonl.rank* > /tmp/reward_log.jsonl 2>/dev/null
  put "$URL_RLOG" /tmp/reward_log.jsonl
  curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS" || ok=0
  return $((1-ok))
}

write_status starting -
curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS" && echo beacon=1 || echo beacon=0
[ "$BUDGET" -lt 900 ] && { write_status skipped_budget -; curl -sf -T "$STATUS" "$URL_STATUS" && exit 90; exit 91; }

mkdir -p /tmp/job
F=1
curl -sf --max-time 120 -o /tmp/job/grpo_train.py "$URL_CODE" || F=0
curl -sf --max-time 120 -o /tmp/job/grpo_reward.py "$URL_REWARD" || F=0
curl -sf --max-time 120 -o /tmp/job/eval_vlm.py "$URL_EVALVLM" || F=0
curl -sf --max-time 300 -o /tmp/job/probe.jsonl "$URL_DATA" || F=0
curl -sf --max-time 300 -o /tmp/job/adapter.tgz "$URL_ADAPTER_IN" || F=0
mkdir -p /tmp/job/adapter && tar xzf /tmp/job/adapter.tgz -C /tmp/job/adapter || F=0
[ -f /tmp/job/adapter/final_adapter/adapter_model.safetensors ] || F=0
[ $F -eq 1 ] && BITS=$((BITS|1))
[ -d "$MODEL_DIR" ] || { echo "MODEL_DIR missing"; write_status model_missing -; upload model_missing -; exit $((100+BITS)); }
[ -d /scratch/scratch/tianrgua-sandbox/smoke_data_v2 ] || { echo "data root missing"; write_status data_missing -; upload data_missing -; exit $((100+BITS)); }

export GRPO_REWARD_LOG="$RUN_DIR/reward_log.jsonl"
write_status training -
curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS"

timeout --signal=TERM --kill-after=240 "$BUDGET" \
  torchrun --standalone --nproc-per-node "${NGPUS:-8}" /tmp/job/grpo_train.py \
    --model-path "$MODEL_DIR" \
    --adapter /tmp/job/adapter/final_adapter \
    --data /tmp/job/probe.jsonl \
    --image-root-map "/local/home/tianrgua/cvproject/data/yolo=/scratch/scratch/tianrgua-sandbox/smoke_data_v2" \
    --output-dir "$RUN_DIR" \
    --max-steps "${MAX_STEPS:-20}" --num-generations "${NUM_GEN:-4}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-128}" --per-device-bs "${BS:-1}" \
    --lr "${LR:-5e-6}" --temperature "${TEMP:-1.0}" \
    --max-pixels "${MAX_PIXELS:-1048576}" \
    > "$RUN_DIR/console.log" 2>&1
RC=$?
echo "train rc=$RC"
grep -q GRPO_TRAIN_DONE "$RUN_DIR/console.log" && DONE=1 || DONE=0
{ [ $RC -eq 0 ] || [ $RC -eq 124 ]; } && BITS=$((BITS|2))
[ $DONE -eq 1 ] && BITS=$((BITS|4))
upload "$([ $DONE -eq 1 ] && echo done || echo segment_end)" "$RC" && BITS=$((BITS|8))
{ [ $((BITS & 2)) -ne 0 ] && [ $((BITS & 8)) -ne 0 ]; } && exit 0
exit $((100+BITS))
