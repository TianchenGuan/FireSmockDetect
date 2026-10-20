#!/bin/bash
# cluster/v6/glm_b200_entry.sh — V6 GLM-Image generation VALIDATION on the
# obsidian-pu B200 node. Shipped inline (ENTRY_B64) by orchestrate_b200_glm.py;
# image smoke-diffusion-b200:v6glm (weights/worker/queue baked under /opt).
# B200 pod rules: exit 0 always, no internet/S3, retrieval via jobs connect,
# PET_* unset, HOME=/tmp. Measures load time + s/img, generates the baked
# queue with 8 one-GPU workers, tars outputs, retains for ack.
set -u +e
export HOME=/tmp USER=worker LOGNAME=worker PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf
unset PET_RDZV_BACKEND PET_RDZV_ENDPOINT PET_RDZV_ID PET_NNODES PET_NPROC_PER_NODE PET_MAX_RESTARTS PET_NODE_RANK
mkdir -p /tmp/hf
RUN="${RUN_NAME:-v6B200_glm}"
ROOT=""; for d in /mnt/local/v6_glm /tmp/v6_glm; do mkdir -p "$d" 2>/dev/null && touch "$d/.w" 2>/dev/null && { ROOT=$d; break; }; done
RUN_DIR=$ROOT/$RUN; mkdir -p "$RUN_DIR/out"
LOG=$RUN_DIR/entry.log; exec > >(tee -a "$LOG") 2>&1
ST=/tmp/status.json; T0=$(date +%s); NG=$(nvidia-smi -L 2>/dev/null | wc -l)
st() {
  N=$(ls $RUN_DIR/out 2>/dev/null | grep -c '\.json$')
  echo "{\"run\":\"$RUN\",\"phase\":\"$1\",\"t\":$(( $(date +%s)-T0 )),\"gpus\":$NG,\"done_imgs\":$N,\"root\":\"$ROOT\",\"host\":\"$(hostname)\",\"ts\":\"$(date -u +%FT%TZ)\"${2:+,$2}}" > $ST
  echo "== [$(( $(date +%s)-T0 ))s] phase=$1 imgs=$N ${2:-}"
}
st start
if [ -f "$RUN_DIR/EXITED" ]; then st restarted_idle; for i in $(seq 1 $(( ${RETAIN_S:-1800}/10 ))); do [ -f /tmp/ack ] && break; sleep 10; done; exit 0; fi
echo "host=$(hostname) gpus=$NG root=$ROOT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -2
python -c "import torch,diffusers;print('torch',torch.__version__,'diffusers',diffusers.__version__,'sm',torch.cuda.get_device_capability(0))"
( while true; do echo "$(date -u +%T),$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ',')" >> $RUN_DIR/gpu_samples.csv; sleep 20; done ) & SAMPLER=$!

st load_check
T1=$(date +%s); BYTES=$(cat /opt/glm-image/*/*.safetensors 2>/dev/null | wc -c); T2=$(date +%s)
echo "{\"weight_bytes\":$BYTES,\"read_s\":$((T2-T1))}" | tee $RUN_DIR/load_check.json

st generate
PIDS=""
for i in $(seq 0 $((NG-1))); do
  CUDA_VISIBLE_DEVICES=$i SHARD=$i NSHARDS=$NG QUEUE=/opt/q_glm_b200.jsonl \
  OUTDIR=$RUN_DIR/out WEIGHTS=/opt/glm-image IMGSZ=${IMGSZ:-1024} STEPS=${STEPS:-} \
  DEADLINE_EPOCH=$(( $(date +%s) + ${GEN_TIMEOUT:-5400} )) \
  python /opt/glm_gen_worker.py > $RUN_DIR/w$i.log 2>&1 &
  PIDS="$PIDS $!"
  st generate   # refresh status between staggered starts
done
wait $PIDS      # never bare wait (tee procsub + sampler)
for i in $(seq 0 $((NG-1))); do echo "=== worker $i ==="; tail -c 1200 $RUN_DIR/w$i.log; done
N=$(ls $RUN_DIR/out | grep -c '\.json$'); J=$(ls $RUN_DIR/out | grep -c '\.jpg$')
echo "generated: $N meta, $J jpgs"
grep -h '"latency_s"' $RUN_DIR/out/*.json 2>/dev/null | grep -oE '"latency_s": [0-9.]+' | awk '{s+=$2; n++} END {if(n)printf "mean s/img: %.1f over %d\n", s/n, n}' | tee $RUN_DIR/latency.txt
tar -czf $RUN_DIR/glmout.tgz -C $RUN_DIR out
ls -la $RUN_DIR/glmout.tgz
st generated "\"jpgs\":$J"
kill $SAMPLER 2>/dev/null
echo "$(date -u +%FT%TZ)" > $RUN_DIR/EXITED
st retain
for i in $(seq 1 $(( ${RETAIN_S:-5400}/10 ))); do [ -f /tmp/ack ] && break; sleep 10; done
st exit; exit 0
