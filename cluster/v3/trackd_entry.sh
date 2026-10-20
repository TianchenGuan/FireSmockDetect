#!/bin/bash
# Track D entrypoint (green g6e pods, offline). MODE=extract | train.
# extract: 8-way GPU-sharded frozen-feature extraction -> /scratch, then
#          tar+presigned-PUT feature tars (train/val) back to S3.
# train:   temporal-head training on /scratch features, artifacts via PUT.
# exit 0 = success; 90 = skipped (budget); else 100+bitmask
# (bit0 fetch ok, bit1 work done, bit2 uploads ok).
set -u +e
export HOME=/tmp USER=worker LOGNAME=worker
export HF_HOME=/tmp/hf YOLO_CONFIG_DIR=/tmp/yolo_cfg
mkdir -p /tmp/hf /tmp/yolo_cfg
BITS=0
NOW=$(date +%s)
BUDGET=$(( ${URL_EXPIRY_EPOCH:-$((NOW+2700))} - NOW - 420 ))
echo "mode=$MODE budget=${BUDGET}s"
[ "$BUDGET" -lt 600 ] && exit 90

SBX=/scratch/scratch/tianrgua-sandbox/smoke/trackd_v3
FEAT=$SBX/features
RUN=$SBX/head_runs/${RUN_NAME:-run}
mkdir -p "$FEAT" "$RUN" /tmp/job
DATA_ROOT=/scratch/scratch/tianrgua-sandbox/smoke_data_v2
[ -d "$DATA_ROOT" ] || { echo "data root missing"; exit 101; }

put() { [ -f "$2" ] && curl -sf --max-time "${3:-300}" -T "$2" "$1"; }
status() {
  cat > /tmp/status.json <<EOF
{"mode":"$MODE","phase":"$1","detail":"$2","gpus":$(nvidia-smi -L 2>/dev/null | wc -l),
 "host":"$(hostname)","ts":"$(date -u +%FT%TZ)"}
EOF
  curl -sf --max-time 60 -T /tmp/status.json "$URL_STATUS"
}
status starting -

F=1
curl -sf --max-time 120 -o /tmp/job/trackd_features.py "$URL_FEAT_CODE" || F=0
curl -sf --max-time 120 -o /tmp/job/trackd_temporal.py "$URL_HEAD_CODE" || F=0
curl -sf --max-time 300 -o /tmp/job/weights.pt "$URL_WEIGHTS" || F=0
curl -sf --max-time 120 -o /tmp/job/sequences_train.json "$URL_SEQ_TRAIN" || F=0
curl -sf --max-time 120 -o /tmp/job/sequences_val.json "$URL_SEQ_VAL" || F=0
[ $F -eq 1 ] && BITS=$((BITS|1)) || { status fetch_failed -; exit $((100+BITS)); }

ROOTMAP="/local/home/tianrgua/cvproject/data/yolo=$DATA_ROOT"

if [ "$MODE" = "extract" ]; then
  status extracting -
  NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || NG=1
  PIDS=()
  for i in $(seq 0 $((NG-1))); do
    CUDA_VISIBLE_DEVICES=$i python /tmp/job/trackd_features.py \
      --weights /tmp/job/weights.pt \
      --sequences /tmp/job/sequences_train.json /tmp/job/sequences_val.json \
      --out "$FEAT" --device cuda:0 --imgsz 1280 \
      --image-root-map "$ROOTMAP" \
      --shard-idx "$i" --shard-n "$NG" > "/tmp/shard$i.log" 2>&1 &
    PIDS+=($!)
  done
  RC=0
  for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
  cat /tmp/shard*.log | tail -c 20000 > /tmp/console.txt
  NTR=$(ls "$FEAT/train" 2>/dev/null | grep -c '\.npz$')
  NVA=$(ls "$FEAT/val" 2>/dev/null | grep -c '\.npz$')
  echo "rc=$RC train_npz=$NTR val_npz=$NVA" >> /tmp/console.txt
  status extracted "rc=$RC train=$NTR val=$NVA"
  put "$URL_CONSOLE" /tmp/console.txt
  if [ "$NTR" -eq "${EXPECT_TRAIN:-915}" ] && [ "$NVA" -eq "${EXPECT_VAL:-266}" ]; then
    BITS=$((BITS|2))
    U=1
    tar czf /tmp/feats_val.tgz -C "$FEAT" val || U=0
    put "$URL_FEATS_VAL" /tmp/feats_val.tgz 900 || U=0
    tar czf /tmp/feats_train.tgz -C "$FEAT" train || U=0
    put "$URL_FEATS_TRAIN" /tmp/feats_train.tgz 1500 || U=0
    [ $U -eq 1 ] && BITS=$((BITS|4))
    status done "train=$NTR val=$NVA up=$U"
  else
    status incomplete "train=$NTR val=$NVA"
  fi
elif [ "$MODE" = "train" ]; then
  [ -d "$FEAT/train" ] && [ -d "$FEAT/val" ] || { status features_missing -; exit 102; }
  status training -
  timeout --signal=TERM --kill-after=120 "$BUDGET" \
    python /tmp/job/trackd_temporal.py \
      --features "$FEAT" --out "$RUN" --resume \
      --epochs "${EPOCHS:-8}" --bs "${BS:-256}" --lr "${LR:-1e-3}" \
      --device cuda:0 --workers "${WORKERS:-6}" ${EXTRA_ARGS:-} \
      > "$RUN/console.log" 2>&1
  RC=$?
  echo "train rc=$RC"
  tail -c 20000 "$RUN/console.log" > /tmp/console.txt
  put "$URL_CONSOLE" /tmp/console.txt
  grep -q TEMPORAL_TRAIN_DONE "$RUN/console.log" && BITS=$((BITS|2))
  U=1
  tar czf /tmp/head.tgz -C "$RUN" . || U=0
  put "$URL_HEAD" /tmp/head.tgz 600 || U=0
  [ $U -eq 1 ] && BITS=$((BITS|4))
  if [ $((BITS & 2)) -ne 0 ]; then status done "rc=$RC"; else status segment_end "rc=$RC"; fi
else
  echo "bad MODE"; exit 99
fi

[ $((BITS & 2)) -ne 0 ] && [ $((BITS & 4)) -ne 0 ] && exit 0
exit $((100+BITS))
