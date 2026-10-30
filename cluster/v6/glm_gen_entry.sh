#!/bin/bash
# V6 GLM-Image generation entry (green L40S node, smoke-diffusion image).
# Stages model weights to /scratch once (marker), fetches worker+queue,
# runs NSHARDS one-GPU workers, tars whatever was produced, PUTs it.
# exit 0 = outputs uploaded; 95 staging failed; 97 upload failed.
set -u +e
exec > /tmp/entry.log 2>&1
NOW=$(date +%s)
BUDGET=$(( ${URL_EXPIRY_EPOCH:-$((NOW+2700))} - NOW - 420 ))
echo "budget: ${BUDGET}s"
ship_log() { tail -c 14000 /tmp/entry.log > /tmp/log.txt; \
  curl -sf --max-time 120 -T /tmp/log.txt "$URL_LOG"; }
trap ship_log EXIT

WROOT=/scratch/scratch/tianrgua-sandbox/smoke/v6data
mkdir -p "$WROOT" || exit 95
if [ ! -f "$WROOT/.glmimage.done" ]; then
  echo "staging GLM-Image weights ..."
  curl -sf --max-time 3000 -o "$WROOT/glm-image.tar" "$URL_WEIGHTS" \
    && tar -xf "$WROOT/glm-image.tar" -C "$WROOT" \
    && rm -f "$WROOT/glm-image.tar" \
    && touch "$WROOT/.glmimage.done" || { echo "weights staging FAILED"; exit 95; }
fi
echo "weights ready: $(du -sh $WROOT/glm-image 2>/dev/null)"

curl -sf --max-time 120 -o /tmp/worker.py "$URL_WORKER" || exit 95
curl -sf --max-time 300 -o /tmp/queue.jsonl "$URL_QUEUE" || exit 95
wc -l /tmp/queue.jsonl
if [ -n "${URL_BGS:-}" ]; then
  mkdir -p /tmp/bgs
  curl -sf --max-time 600 -o /tmp/bgs.tar "$URL_BGS" \
    && tar -xf /tmp/bgs.tar -C /tmp/bgs || exit 95
  echo "bgs staged: $(ls /tmp/bgs | wc -l)"
fi

NOW=$(date +%s)
BUDGET=$(( ${URL_EXPIRY_EPOCH:-$((NOW+2700))} - NOW - 420 ))
echo "budget after staging: ${BUDGET}s"
[ "$BUDGET" -lt 300 ] && { echo "budget too small"; exit 90; }
DEADLINE=$(( NOW + BUDGET ))

OUT="$WROOT/glmout/${OUTTAG:-default}"   # /scratch persists -> resume across jobs
mkdir -p "$OUT"
echo "existing outputs: $(ls $OUT 2>/dev/null | grep -c json)"
N=${NSHARDS:-8}
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i SHARD=$i NSHARDS=$N QUEUE=/tmp/queue.jsonl \
  OUTDIR=$OUT WEIGHTS=$WROOT/glm-image IMGSZ=${IMGSZ:-1024} STEPS=${STEPS:-} \
  BGDIR=/tmp/bgs \
  DEADLINE_EPOCH=$DEADLINE python /tmp/worker.py > /tmp/w$i.log 2>&1 &
done
wait
for i in $(seq 0 $((N-1))); do echo "=== worker $i ==="; tail -c 2200 /tmp/w$i.log; done

cd "$OUT" || exit 97
ls | head -5; ls | wc -l
tar -czf /tmp/glmout.tgz .
curl -sf --max-time 900 -T /tmp/glmout.tgz "$URL_OUT" || exit 97
echo UPLOAD_OK
exit 0
