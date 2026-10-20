#!/bin/bash
# V5b TEACHER-POOL inference entry (green g6e, offline). One job = one image
# list, NGPUS single-GPU shards (same phase-2 pattern as cluster/v3/dino_eval_entry.sh,
# no latency phase). FAMILY=dino -> dino_infer.py (image smoke-dino3:v1);
# FAMILY=rfdetr -> rfdetr_infer.py (image smoke-det3:v1). Predictions only, no GT.
# Pods have NO log channel: everything is self-logged to /tmp/out/entry.log,
# shipped via presigned PUT (URL_LOG) on exit and persisted on /scratch
# (authoritative copy if the presigned URLs expired; fetch with cluster/v3/fetch_run.py).
# exit: 0 ok | 101-104 code/list fetch | 105 ckpt missing | 106 data root missing
#       | 107 DATA_TARS staging failed | 109 ckpt sha256 mismatch vs EXPECT_SHA | 110 bad FAMILY | 1 a shard failed
set -u +e
export HOME=/tmp USER=worker LOGNAME=worker
export TORCHINDUCTOR_CACHE_DIR=/tmp/ic TRITON_CACHE_DIR=/tmp/tc HF_HOME=/tmp/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/ic /tmp/tc /tmp/hf /tmp/job /tmp/out
exec > /tmp/out/entry.log 2>&1
SANDBOX=/scratch/scratch/tianrgua-sandbox
PERSIST=$SANDBOX/smoke/v5b/teacher/${EVAL_NAME:-unnamed}
mkdir -p "$PERSIST"
ship_log() { tail -c 60000 /tmp/out/entry.log > /tmp/out/entry_log_tail.txt
  cp -f /tmp/out/entry_log_tail.txt "$PERSIST"/ 2>/dev/null
  [ -n "${URL_LOG:-}" ] && curl -sf --max-time 120 -T /tmp/out/entry_log_tail.txt "$URL_LOG"; }
trap ship_log EXIT
echo "START $(date -u +%FT%TZ) EVAL_NAME=${EVAL_NAME:-} FAMILY=${FAMILY:-} host=$(hostname)"
nvidia-smi -L
curl -sf --max-time 300 -o /tmp/job/code.tgz "$URL_CODE" || { echo "code fetch failed"; exit 101; }
tar xzf /tmp/job/code.tgz -C /tmp/job || exit 102
curl -sf --max-time 300 -o /tmp/job/lists.tgz "$URL_LISTS" || exit 103
tar xzf /tmp/job/lists.tgz -C /tmp/job || exit 104
[ -f "${CKPT:?}" ] || { echo "ckpt missing: $CKPT"; exit 105; }
[ -d "$SANDBOX/smoke_data_v2" ] || { echo "data root missing"; exit 106; }
[ -d "$SANDBOX/smoke/v4data/E_ET_v1/images" ] || echo "WARN: E_ET_v1 images not staged on /scratch"
# optional DATA_TARS (comma-sep name=presignedGET): staged ONCE under $DATA_ROOT on /scratch (same pattern + markers as
# cluster/v4/train_entry_v4.sh, so a later training job with the same tar name skips the download). exit 107 on failure.
DATA_ROOT="${DATA_ROOT:-$SANDBOX/smoke/v5bdata}"
if [ -n "${DATA_TARS:-}" ]; then
  mkdir -p "$DATA_ROOT" || { echo "data root unwritable: $DATA_ROOT"; exit 107; }
  IFS=',' read -ra TARS <<< "$DATA_TARS"
  for spec in "${TARS[@]}"; do
    name="${spec%%=*}"; url="${spec#*=}"
    if [ -f "$DATA_ROOT/.${name}.done" ]; then echo "tar $name already staged"; continue; fi
    echo "staging tar $name -> $DATA_ROOT ... $(date -u +%T)"
    if curl -sf --max-time 2400 -o "$DATA_ROOT/${name}.tar" "$url" && tar -xf "$DATA_ROOT/${name}.tar" -C "$DATA_ROOT"; then
      rm -f "$DATA_ROOT/${name}.tar"; touch "$DATA_ROOT/.${name}.done"; echo "tar $name staged ok $(date -u +%T)"
    else
      rm -f "$DATA_ROOT/${name}.tar"; echo "FATAL tar $name staging failed"; exit 107
    fi
  done
fi
SHA=$(sha256sum "$CKPT" | cut -c1-64); echo "ckpt $CKPT sha256=$SHA size=$(stat -c %s "$CKPT")"
if [ -n "${EXPECT_SHA:-}" ] && [[ "$SHA" != ${EXPECT_SHA}* ]]; then
  echo "FATAL ckpt sha256 mismatch (expected ${EXPECT_SHA}*)"; exit 109; fi
ls -la /tmp/job /tmp/job/lists
NGPUS=${NGPUS:-8}; RC=0
case "${FAMILY:?}" in
  dino)   BASE="python /tmp/job/dino_infer.py --backbone-dir ${BACKBONE_DIR:?} ${INFER_EXTRA:-} --ckpt $CKPT --width ${WIDTH:-1280} --height ${HEIGHT:-736} --conf-floor ${CONF_FLOOR:-0.02} ${ROOTMAP:+--root-map $ROOTMAP}";;
  rfdetr) BASE="python /tmp/job/rfdetr_infer.py --weights $CKPT --imgsz ${IMGSZ:-1280} --conf-floor ${CONF_FLOOR:-0.02} ${ROOTMAP:+--root-map $ROOTMAP}";;
  *) echo "bad FAMILY ${FAMILY}"; exit 110;;
esac
echo "BASE: $BASE"
for SPLIT in ${SPLITS:?}; do
  L=/tmp/job/lists/${SPLIT}.txt; NL=$(grep -c . "$L"); echo "== SPLIT $SPLIT ($NL images) $(date -u +%T)"
  PIDS=()
  for g in $(seq 0 $((NGPUS-1))); do
    $BASE --list "$L" --shard "$g/$NGPUS" --device cuda --device-index "$g" ${MAX_IMAGES:+--max-images $MAX_IMAGES} \
      --out "/tmp/out/${SPLIT}_shard${g}.jsonl" > "/tmp/out/${SPLIT}_shard${g}.log" 2>&1 &
    PIDS+=($!)
  done
  for g in $(seq 0 $((NGPUS-1))); do wait "${PIDS[$g]}"; r=$?; [ $r = 0 ] || { echo "shard $g rc=$r"; RC=1; }; done
  cat /tmp/out/${SPLIT}_shard*.jsonl > "/tmp/out/${SPLIT}_preds.jsonl"
  N=$(wc -l < "/tmp/out/${SPLIT}_preds.jsonl"); NE=$(grep -c '"error"' "/tmp/out/${SPLIT}_preds.jsonl")
  echo "$SPLIT rows=$N errors=$NE list=$NL rc=$RC $(date -u +%T)"
  gzip -cf "/tmp/out/${SPLIT}_preds.jsonl" > "/tmp/out/${SPLIT}_preds.jsonl.gz"
  cp -f "/tmp/out/${SPLIT}_preds.jsonl.gz" "$PERSIST"/
  for g in $(seq 0 $((NGPUS-1))); do { echo "----- $SPLIT shard $g"; tail -c 1500 "/tmp/out/${SPLIT}_shard${g}.log"; } >> /tmp/out/shardlogs.txt; done
  printf '{"split":"%s","list_n":%s,"rows":%s,"errors":%s,"rc":%s,"ckpt_sha256":"%s"}\n' "$SPLIT" "$NL" "$N" "$NE" "$RC" "$SHA" >> /tmp/out/summary.jsonl
  V="URL_PREDS_$(echo "$SPLIT" | tr '[:lower:]-' '[:upper:]_')"; U="${!V:-}"
  if [ -n "$U" ] && curl -sf --max-time 600 -T "/tmp/out/${SPLIT}_preds.jsonl.gz" "$U"; then echo "uploaded $SPLIT"
  else echo "upload skipped/failed $SPLIT (persisted at $PERSIST)"; fi
done
cp -f /tmp/out/shardlogs.txt /tmp/out/summary.jsonl "$PERSIST"/
[ -n "${URL_SHARDLOGS:-}" ] && curl -sf --max-time 120 -T /tmp/out/shardlogs.txt "$URL_SHARDLOGS"
[ -n "${URL_SUMMARY:-}" ] && curl -sf --max-time 60 -T /tmp/out/summary.jsonl "$URL_SUMMARY"
echo "DONE rc=$RC persist=$PERSIST $(date -u +%FT%TZ)"
exit $RC
