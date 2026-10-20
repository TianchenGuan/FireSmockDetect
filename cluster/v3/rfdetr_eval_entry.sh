#!/bin/bash
# Iteration-3 Track A: RF-DETR inference-only eval job (green g6e, offline).
#
# Phases:
#   1. batch-1 GPU latency on ONE process / ONE GPU (nothing else running), so
#      the number in reports/v3/rfdetr_eval.md is a clean single-stream figure.
#   2. 8 single-GPU shards over each requested image list (val, test, ...).
# Predictions only (no GT on the pod): the conf sweep is computed offline on
# the devbox from the frozen YOLO labels.
#
# Everything persists to /scratch (evals_v3/$EVAL_NAME) because presigned URLs
# expire in ~1h; uploads at the end are best-effort and a fetch job
# (cluster/v3/fetch_run.py --run ../evals_v3/<name>) can retrieve them later.
set -u +e
export HOME=/tmp USER=worker LOGNAME=worker
export TORCHINDUCTOR_CACHE_DIR=/tmp/ic TRITON_CACHE_DIR=/tmp/tc HF_HOME=/tmp/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/ic /tmp/tc /tmp/hf /tmp/job /tmp/out

SANDBOX=/scratch/scratch/tianrgua-sandbox
DATA_ROOT=$SANDBOX/smoke_data_v2
PERSIST=$SANDBOX/smoke/evals_v3/${EVAL_NAME:-unnamed}
WEIGHTS=${WEIGHTS:-$SANDBOX/smoke/runs_v3/rfdetrL_1280_e48/checkpoint_best_ema.pth}
mkdir -p "$PERSIST"

curl -sf --max-time 300 -o /tmp/job/code.tgz "$URL_CODE" || { echo "code fetch failed"; exit 101; }
tar xzf /tmp/job/code.tgz -C /tmp/job || exit 102
curl -sf --max-time 300 -o /tmp/job/lists.tgz "$URL_LISTS" || exit 103
tar xzf /tmp/job/lists.tgz -C /tmp/job || exit 104
[ -f "$WEIGHTS" ] || { echo "weights missing: $WEIGHTS"; exit 105; }
[ -d "$DATA_ROOT" ] || { echo "data root missing"; exit 106; }
ls -la /tmp/job /tmp/job/lists

ROOTMAP="/local/home/tianrgua/cvproject/data/yolo=$DATA_ROOT"
IMGSZ=${IMGSZ:-1280}
CONF_FLOOR=${CONF_FLOOR:-0.02}
NGPUS=${NGPUS:-8}
RC=0

# ---- phase 1: single-stream batch-1 GPU latency -----------------------------
if [ "${LATENCY_N:-50}" -gt 0 ]; then
  echo "== LATENCY phase (n=${LATENCY_N}) =="
  python /tmp/job/rfdetr_infer.py --weights "$WEIGHTS" \
    --list "/tmp/job/lists/${LATENCY_LIST:-sdis_val}.txt" --shard 0/1 \
    --device cuda --device-index 0 --imgsz "$IMGSZ" --conf-floor "$CONF_FLOOR" \
    --root-map "$ROOTMAP" --max-images "${LATENCY_N}" \
    --latency-n "${LATENCY_N}" --latency-out /tmp/out/latency_gpu.json \
    --out /tmp/out/latency_probe.jsonl > /tmp/out/latency.log 2>&1
  echo "latency rc=$?"
  cat /tmp/out/latency_gpu.json 2>/dev/null
  cp -f /tmp/out/latency_gpu.json /tmp/out/latency.log "$PERSIST"/ 2>/dev/null
  curl -sf --max-time 60 -T /tmp/out/latency_gpu.json "$URL_LATENCY"
fi

# ---- phase 2: sharded inference over each list -----------------------------
for SPLIT in ${SPLITS:-sdis_val sdis_test}; do
  echo "== SPLIT $SPLIT =="
  PIDS=()
  for g in $(seq 0 $((NGPUS-1))); do
    python /tmp/job/rfdetr_infer.py --weights "$WEIGHTS" \
      --list "/tmp/job/lists/${SPLIT}.txt" --shard "$g/$NGPUS" \
      --device cuda --device-index "$g" --imgsz "$IMGSZ" \
      --conf-floor "$CONF_FLOOR" --root-map "$ROOTMAP" \
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
  tail -c 4000 "/tmp/out/${SPLIT}_shard0.log" >> /tmp/out/shardlogs.txt
done
cp -f /tmp/out/shardlogs.txt "$PERSIST"/ 2>/dev/null

# best-effort uploads (URLs may have expired; /scratch copy is authoritative)
for SPLIT in ${SPLITS:-sdis_val sdis_test}; do
  V="URL_PREDS_$(echo "$SPLIT" | tr '[:lower:]-' '[:upper:]_')"
  U="${!V:-}"
  [ -n "$U" ] && curl -sf --max-time 600 -T "/tmp/out/${SPLIT}_preds.jsonl.gz" "$U" \
    && echo "uploaded $SPLIT" || echo "upload skipped/failed $SPLIT"
done
curl -sf --max-time 120 -T /tmp/out/shardlogs.txt "${URL_LOG:-}" 2>/dev/null
echo "DONE rc=$RC persist=$PERSIST"
exit $RC
