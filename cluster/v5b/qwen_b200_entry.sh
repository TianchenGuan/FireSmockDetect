#!/bin/bash
# cluster/v5b/qwen_b200_entry.sh — V5b Qwen3.5-27B bf16 LoRA SFT **smoke test** (NOT a campaign)
# on the obsidian-pu B200 node. Shipped inline (ENTRY_B64) by cluster/v5b/orchestrate_b200_qwen.py;
# image smoke-detect-vlm-b200:v1 (weights, SFT data, code baked under /opt). Same pod rules as
# dino_b200_entry.sh: exit 0 always, status via `kraken jobs connect`, PET_* unset, HOME=/tmp.
# Measures: image->GPU load time of 55.6 GB, step time / GPU memory at bf16 (no nf4), a
# MAX_STEPS-step LoRA run at the v2 effective batch (bs*ga*8 = 64), optional small eval.
set -u +e
export HOME=/tmp USER=worker LOGNAME=worker PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PET_RDZV_BACKEND PET_RDZV_ENDPOINT PET_RDZV_ID PET_NNODES PET_NPROC_PER_NODE PET_MAX_RESTARTS PET_NODE_RANK
mkdir -p /tmp/hf /tmp/inductor_cache /tmp/triton_cache
RUN="${RUN_NAME:-v5bB200_qwen27b_smoke}"
ROOT=""; for d in /mnt/local/v5b_qwen /tmp/v5b_qwen; do mkdir -p "$d" 2>/dev/null && touch "$d/.w" 2>/dev/null && { ROOT=$d; break; }; done
RUN_DIR=$ROOT/$RUN; mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/entry.log; exec > >(tee -a "$LOG") 2>&1
ST=/tmp/status.json; T0=$(date +%s); NG=$(nvidia-smi -L 2>/dev/null | wc -l)
MODEL=/opt/models/qwen3.5-27b; CODE=/opt/smoke_vlm; SFT=/opt/smoke_data/sft
RMAP="/local/home/tianrgua/cvproject/data/yolo=/opt/smoke_data"
st() {
  L=$(grep -oE "'loss': '[0-9.]+'" "$RUN_DIR/train/console.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+' | tail -1)
  S=$(grep -oE "[0-9]+/[0-9]+ \[" "$RUN_DIR/train/console.log" 2>/dev/null | tail -1 | cut -d/ -f1)
  echo "{\"run\":\"$RUN\",\"phase\":\"$1\",\"t\":$(( $(date +%s)-T0 )),\"gpus\":$NG,\"step\":\"${S:-none}\",\"loss\":\"${L:-none}\",\"root\":\"$ROOT\",\"host\":\"$(hostname)\",\"ts\":\"$(date -u +%FT%TZ)\"${2:+,$2}}" > $ST
  echo "== [$(( $(date +%s)-T0 ))s] phase=$1 ${2:-}"
}
st start
if [ -f "$RUN_DIR/EXITED" ]; then st restarted_idle; for i in $(seq 1 $(( ${RETAIN_S:-1800}/10 ))); do [ -f /tmp/ack ] && break; sleep 10; done; exit 0; fi
echo "host=$(hostname) uid=$(id -u) nproc=$(nproc) gpus=$NG root=$ROOT"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -2
python -c "import torch,transformers,peft;print('torch',torch.__version__,'transformers',transformers.__version__,'peft',peft.__version__,'sm',torch.cuda.get_device_capability(0))"
# background GPU sampler (memory/util every 20 s) -> gpu_samples.csv
( while true; do echo "$(date -u +%T),$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ',' )" >> $RUN_DIR/gpu_samples.csv; sleep 20; done ) & SAMPLER=$!

# ---- phase 1: how fast can the baked 55.6 GB be read from the image layer (cold)?
st load_check
T1=$(date +%s); BYTES=$(cat $MODEL/*.safetensors | wc -c); T2=$(date +%s)
echo "{\"shard_bytes\":$BYTES,\"read_s\":$((T2-T1)),\"mb_per_s\":$(( BYTES/1000000/((T2-T1)>0?(T2-T1):1) ))}" | tee $RUN_DIR/load_check.json

# ---- phase 2: MAX_STEPS-step bf16 LoRA SFT, eff. batch BS*GA*8 (v2: 64), lr 1e-4, r16 a32, 1 MP
st train
mkdir -p $RUN_DIR/train
timeout --signal=TERM --kill-after=120 ${TRAIN_TIMEOUT:-3600} torchrun --standalone --nproc_per_node $NG $CODE/train_vlm.py \
  --model-path $MODEL --data $SFT/train.jsonl --image-root-map "$RMAP" --output-dir $RUN_DIR/train \
  --max-steps ${MAX_STEPS:-200} --epochs 1 --per-device-bs ${BS:-2} --grad-accum ${GA:-4} --lr ${LR:-1e-4} \
  --lora-r 16 --lora-alpha 32 --max-pixels ${MAX_PIXELS:-1048576} --save-steps ${SAVE_STEPS:-100} \
  --workers ${WORKERS:-4} --logging-steps 5 --quant none ${EXTRA_ARGS:-} > $RUN_DIR/train/console.log 2>&1
TRC=$?; DONE=$(grep -c VLM_TRAIN_DONE $RUN_DIR/train/console.log)
echo "train rc=$TRC done=$DONE"; grep -m1 "^PARAMS" $RUN_DIR/train/console.log
grep -m3 -A14 -E "Traceback|OutOfMemoryError|RuntimeError" $RUN_DIR/train/console.log | head -60 > $RUN_DIR/console_errors.txt
tr '\r' '\n' < $RUN_DIR/train/console.log | grep -E "'loss'|it/s|s/it" | tail -n 6
CK=$(ls -d $RUN_DIR/train/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); [ -n "$CK" ] && cp $CK/trainer_state.json $RUN_DIR/trainer_state.json
cp $RUN_DIR/train/params.json $RUN_DIR/ 2>/dev/null
[ -d $RUN_DIR/train/final_adapter ] && tar czf $RUN_DIR/final_adapter.tgz -C $RUN_DIR/train final_adapter && ls -la $RUN_DIR/final_adapter.tgz
tr '\r' '\n' < $RUN_DIR/train/console.log | tail -c 30000 > $RUN_DIR/console_tail.txt
st trained "\"train_rc\":$TRC,\"done\":$DONE"

# ---- phase 3 (optional): small eval with the adapter to time bf16 27B inference on B200
if [ "${EVAL_N:-0}" -gt 0 ] && [ -d $RUN_DIR/train/final_adapter ]; then
  st eval; mkdir -p $RUN_DIR/eval; T3=$(date +%s); PIDS=""; PER=$(( (${EVAL_N} + NG - 1) / NG ))  # eval_vlm.py applies --max-images PER SHARD
  for i in $(seq 0 $((NG-1))); do
    python $CODE/eval_vlm.py --model-path $MODEL --adapter $RUN_DIR/train/final_adapter --list $SFT/val_eval_2000.txt \
      --root-map "$RMAP" --max-images $PER --shard $i/$NG --device-index $i --max-pixels ${MAX_PIXELS:-1048576} \
      --max-new-tokens 192 --out $RUN_DIR/eval/shard$i.jsonl > $RUN_DIR/eval/shard$i.log 2>&1 &
    PIDS="$PIDS $!"
  done; wait $PIDS   # never bare `wait`: it would also wait for the tee procsub and the GPU sampler
  cat $RUN_DIR/eval/shard*.jsonl > $RUN_DIR/eval_preds.jsonl 2>/dev/null; gzip -kf $RUN_DIR/eval_preds.jsonl
  echo "eval rows $(wc -l < $RUN_DIR/eval_preds.jsonl) in $(( $(date +%s)-T3 ))s"; tail -n 2 $RUN_DIR/eval/shard0.log
  st evaluated "\"eval_rows\":$(wc -l < $RUN_DIR/eval_preds.jsonl),\"eval_s\":$(( $(date +%s)-T3 ))"
fi
kill $SAMPLER 2>/dev/null
echo "$(date -u +%FT%TZ) rc=$TRC" > $RUN_DIR/EXITED
st retain
for i in $(seq 1 $(( ${RETAIN_S:-3600}/10 ))); do [ -f /tmp/ack ] && break; sleep 10; done
st exit; exit 0
