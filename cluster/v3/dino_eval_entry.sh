#!/bin/bash
# Track B/C: DINOv3-detector inference-only eval job (green g6e, offline).
# Same phases as rfdetr_eval_entry.sh: (1) single-stream batch-1 GPU latency,
# (2) 8 single-GPU shards per requested image list. Predictions only; sweeps
# run offline on the devbox against the frozen YOLO labels.
set -u +e
export HOME=/tmp USER=worker LOGNAME=worker
export TORCHINDUCTOR_CACHE_DIR=/tmp/ic TRITON_CACHE_DIR=/tmp/tc HF_HOME=/tmp/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/ic /tmp/tc /tmp/hf /tmp/job /tmp/out

SANDBOX=/scratch/scratch/tianrgua-sandbox
DATA_ROOT=$SANDBOX/smoke_data_v2
PERSIST=$SANDBOX/smoke/evals_v3/${EVAL_NAME:-unnamed}
mkdir -p "$PERSIST"

curl -sf --max-time 300 -o /tmp/job/code.tgz "$URL_CODE" || { echo "code fetch failed"; exit 101; }
tar xzf /tmp/job/code.tgz -C /tmp/job || exit 102
curl -sf --max-time 300 -o /tmp/job/lists.tgz "$URL_LISTS" || exit 103
tar xzf /tmp/job/lists.tgz -C /tmp/job || exit 104
[ -f "${CKPT:?}" ] || { echo "ckpt missing: $CKPT"; exit 105; }
[ -d "$DATA_ROOT" ] || { echo "data root missing"; exit 106; }
if [ -n "${URL_FRAMES:-}" ]; then
  # optional image tarball (e.g. FIgLib frames, not staged on /scratch)
  mkdir -p /tmp/frames
  curl -sf --max-time 900 -o /tmp/job/frames.tgz "$URL_FRAMES" || exit 107
  tar xzf /tmp/job/frames.tgz -C /tmp/frames || exit 108
fi
ls -la /tmp/job /tmp/job/lists

ROOTMAP="${ROOTMAP_OVERRIDE:-/local/home/tianrgua/cvproject/data/yolo=$DATA_ROOT}"
NGPUS=${NGPUS:-8}
MODE_ARGS=""
if [ -n "${FEAT_ROOT:-}" ]; then
  MODE_ARGS="--feat-root $FEAT_ROOT --embed-dim ${EMBED_DIM:?}"
else
  MODE_ARGS="--backbone-dir ${BACKBONE_DIR:?}"
  [ "${BF16:-0}" = "1" ] && MODE_ARGS="$MODE_ARGS --bf16-backbone"
fi
# v4: extra dino_infer.py flags (e.g. "--lora 16" so LoRA checkpoints load)
MODE_ARGS="$MODE_ARGS ${INFER_EXTRA:-}"
COMMON="--ckpt $CKPT --width ${WIDTH:-1280} --height ${HEIGHT:-736} \
  --conf-floor ${CONF_FLOOR:-0.02} --root-map $ROOTMAP"
RC=0

# ---- phase 1: single-stream batch-1 GPU latency -----------------------------
if [ "${LATENCY_N:-0}" -gt 0 ]; then
  echo "== LATENCY phase (n=${LATENCY_N}) =="
  python /tmp/job/dino_infer.py $MODE_ARGS $COMMON \
    --list "/tmp/job/lists/${LATENCY_LIST:-sdis_val}.txt" --shard 0/1 \
    --device cuda --device-index 0 --max-images "${LATENCY_N}" \
    --latency-n "${LATENCY_N}" --latency-out /tmp/out/latency_gpu.json \
    --out /tmp/out/latency_probe.jsonl > /tmp/out/latency.log 2>&1
  echo "latency rc=$?"
  cat /tmp/out/latency_gpu.json 2>/dev/null
  cp -f /tmp/out/latency_gpu.json /tmp/out/latency.log "$PERSIST"/ 2>/dev/null
  curl -sf --max-time 60 -T /tmp/out/latency_gpu.json "${URL_LATENCY:-}" 2>/dev/null
fi

# ---- phase 1b (v4): extra latency variants in the same job -------------------
# LATENCY_VARIANTS = "tag|extra flags;tag2|extra flags" -> latency_gpu_<tag>.json
# each measured with the SAME protocol as phase 1 (n=LATENCY_N, batch 1, GPU 0).
if [ -n "${LATENCY_VARIANTS:-}" ] && [ "${LATENCY_N:-0}" -gt 0 ]; then
  IFS=';' read -ra VARS <<< "$LATENCY_VARIANTS"
  for spec in "${VARS[@]}"; do
    tag="${spec%%|*}"; flags="${spec#*|}"
    echo "== LATENCY variant $tag (flags: $flags) =="
    python /tmp/job/dino_infer.py --backbone-dir "${BACKBONE_DIR:?}" $flags $COMMON \
      --list "/tmp/job/lists/${LATENCY_LIST:-sdis_val}.txt" --shard 0/1 \
      --device cuda --device-index 0 --max-images "${LATENCY_N}" \
      --latency-n "${LATENCY_N}" --latency-out "/tmp/out/latency_gpu_${tag}.json" \
      --out "/tmp/out/latency_probe_${tag}.jsonl" > "/tmp/out/latency_${tag}.log" 2>&1
    echo "latency variant $tag rc=$?"
    cat "/tmp/out/latency_gpu_${tag}.json" 2>/dev/null
    { echo "===== latency variant $tag log tail ====="; tail -c 3000 "/tmp/out/latency_${tag}.log"; } >> /tmp/out/shardlogs.txt
    cp -f "/tmp/out/latency_gpu_${tag}.json" "/tmp/out/latency_${tag}.log" "$PERSIST"/ 2>/dev/null
    V="URL_LATENCY_$(echo "$tag" | tr '[:lower:]-' '[:upper:]_')"
    [ -n "${!V:-}" ] && curl -sf --max-time 60 -T "/tmp/out/latency_gpu_${tag}.json" "${!V}" 2>/dev/null
  done
fi

# ---- phase 2: sharded inference over each list -----------------------------
for SPLIT in ${SPLITS:-sdis_val}; do
  echo "== SPLIT $SPLIT =="
  PIDS=()
  for g in $(seq 0 $((NGPUS-1))); do
    python /tmp/job/dino_infer.py $MODE_ARGS $COMMON \
      --list "/tmp/job/lists/${SPLIT}.txt" --shard "$g/$NGPUS" \
      --device cuda --device-index "$g" \
      ${MAX_IMAGES:+--max-images $MAX_IMAGES} \
      --out "/tmp/out/${SPLIT}_shard${g}.jsonl" > "/tmp/out/${SPLIT}_shard${g}.log" 2>&1 &
    PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
  cat /tmp/out/${SPLIT}_shard*.jsonl > "/tmp/out/${SPLIT}_preds.jsonl"
  N=$(wc -l < "/tmp/out/${SPLIT}_preds.jsonl")
  echo "$SPLIT rows=$N rc=$RC"
  gzip -cf "/tmp/out/${SPLIT}_preds.jsonl" > "/tmp/out/${SPLIT}_preds.jsonl.gz"
  cp -f "/tmp/out/${SPLIT}_preds.jsonl.gz" "$PERSIST"/ 2>/dev/null
  for g in $(seq 0 $((NGPUS-1))); do
    tail -c 2000 "/tmp/out/${SPLIT}_shard${g}.log" >> /tmp/out/shardlogs.txt
  done
done
cp -f /tmp/out/shardlogs.txt "$PERSIST"/ 2>/dev/null

# best-effort uploads (/scratch copy is authoritative)
for SPLIT in ${SPLITS:-sdis_val}; do
  V="URL_PREDS_$(echo "$SPLIT" | tr '[:lower:]-' '[:upper:]_')"
  U="${!V:-}"
  [ -n "$U" ] && curl -sf --max-time 600 -T "/tmp/out/${SPLIT}_preds.jsonl.gz" "$U" \
    && echo "uploaded $SPLIT" || echo "upload skipped/failed $SPLIT"
done
curl -sf --max-time 120 -T /tmp/out/shardlogs.txt "${URL_LOG:-}" 2>/dev/null
echo "DONE rc=$RC persist=$PERSIST"
exit $RC
