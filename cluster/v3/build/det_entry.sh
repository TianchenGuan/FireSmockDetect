#!/bin/bash
# Iteration-3 Track A detection entrypoint (green g6e pods, offline).
#
# PATTERN CHOICE (documented): single LONG job, not 55-min segments.
# RF-DETR (Lightning) checkpoints at epoch granularity, so 55-min segments
# would lose partial epochs; /scratch persists across jobs, so the run dir
# ($RUN_DIR) holds console.log, metrics, last.ckpt and survives the job.
# Presigned URLs only live ~55 min -> status/console beacons are best-effort
# for the first ~50 min (background loop), after which observability =
# kraken job state + on-demand cluster/v3/fetch_run.py jobs reading /scratch.
# Wall clock bounded by WALL_S (default 43200 = 12h) via timeout; if the job
# dies or hits the wall, resubmit with the same RUN_NAME -> auto-resume from
# last.ckpt.
#
# exit 0 = train rc 0 (or 124 timeout with DONE-less clean wall-stop);
# 90 = skipped (budget); else 100+bits (bit0 code fetch ok, bit1 clean train
# exit, bit2 RFDETR_TRAIN_DONE, bit3 status upload ok).
set -u +e
# uid 1001 has no passwd entry -> pin HOME/caches (iteration-2 lesson)
export HOME=/tmp USER=worker LOGNAME=worker
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache TRITON_CACHE_DIR=/tmp/triton_cache
export HF_HOME=/tmp/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p /tmp/inductor_cache /tmp/triton_cache /tmp/hf

SANDBOX=/scratch/scratch/tianrgua-sandbox
DATA_ROOT=$SANDBOX/smoke_data_v2
RUN_ROOT=$SANDBOX/smoke/runs_v3
RUN_DIR=$RUN_ROOT/$RUN_NAME
DS_DIR=$SANDBOX/smoke/det3/rf_ds
mkdir -p "$RUN_DIR"
STATUS=/tmp/status.json
BITS=0
WALL_S=${WALL_S:-43200}

write_status() {
  EP=$(grep -oE "Epoch [0-9]+" "$RUN_DIR/console.log" 2>/dev/null | tail -1)
  IT=$(grep -oE "[0-9.]+it/s" "$RUN_DIR/console.log" 2>/dev/null | tail -1)
  LO=$(grep -oE "loss=[0-9.]+" "$RUN_DIR/console.log" 2>/dev/null | tail -1)
  GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  cat > $STATUS <<EOF
{"run":"$RUN_NAME","mode":"${MODE:-train}","phase":"$1","rc":"$2","gpus":$GPUS,
 "epoch":"${EP:-}","rate":"${IT:-}","loss":"${LO:-}",
 "host":"$(hostname)","ts":"$(date -u +%FT%TZ)"}
EOF
}
upload() {
  write_status "$1" "$2"
  if [ -f "$RUN_DIR/console.log" ]; then
    grep -m4 -B3 -A25 -E "Traceback|OutOfMemoryError|RuntimeError|ValueError|AssertionError|PARAMS" "$RUN_DIR/console.log" 2>/dev/null | head -c 20000 > /tmp/ct.txt
    echo "===== TAIL =====" >> /tmp/ct.txt
    tail -c 14000 "$RUN_DIR/console.log" | tr '\r' '\n' | tail -n 120 >> /tmp/ct.txt
    curl -sf --max-time 120 -T /tmp/ct.txt "$URL_CONSOLE"
  fi
  curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS"
}

write_status starting -
curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS" && echo beacon=1 || echo beacon=0

[ -d "$DATA_ROOT" ] || { echo "data root missing"; upload data_missing -; exit 101; }

# dataset dir: annotation jsons baked in image; 'data' symlink -> image tree
for sp in train valid; do
  mkdir -p "$DS_DIR/$sp"
  cp -f /opt/data/rf/$sp/_annotations.coco.json "$DS_DIR/$sp/_annotations.coco.json"
  ln -sfn "$DATA_ROOT" "$DS_DIR/$sp/data"
done

# fetch freshest driver (fallback = baked copy)
mkdir -p /tmp/job
if curl -sf --max-time 120 -o /tmp/job/train_rfdetr.py "${URL_CODE:-}"; then BITS=$((BITS|1));
else cp /opt/train_rfdetr.py /tmp/job/train_rfdetr.py; fi

# best-effort beacon loop while presigned URLs are alive (~50 min)
( END=$(( $(date +%s) + 2900 ))
  while [ "$(date +%s)" -lt "$END" ]; do sleep 240; upload running -; done ) &
BEACON_PID=$!

run_train() { # $1=mode $2=outdir $3=epochs $4=bs $5=ga $6=timeout $7=extra
  timeout --signal=TERM --kill-after=300 "$6" \
    python /tmp/job/train_rfdetr.py \
      --mode "$1" \
      --dataset-dir "$DS_DIR" --data-root "$DATA_ROOT" \
      --output-dir "$2" \
      --resolution "${RES:-1280}" --epochs "$3" \
      --batch-size "$4" --grad-accum "$5" \
      --lr "${LR:-1e-4}" --workers "${WORKERS:-4}" --devices "${NGPUS:-8}" \
      --checkpoint-interval "${CKPT_INT:-5}" --eval-interval "${EVAL_INT:-1}" \
      $7 >> "$RUN_DIR/console.log" 2>&1
}

run_deimv2() { # $1=epochs-override(0=none) $2=timeout $3=extra update args
  UPD="output_dir=$RUN_DIR/out"
  [ "$1" != "0" ] && UPD="$UPD epoches=$1"
  RES_ARG=""
  [ -f "$RUN_DIR/out/last.pth" ] && RES_ARG="-r $RUN_DIR/out/last.pth" \
    || RES_ARG="-t ${DEIMV2_TUNE:-/opt/DEIMv2/ckpts/deimv2_dinov3_m_coco.pth}"
  ( cd /opt/DEIMv2 && timeout --signal=TERM --kill-after=300 "$2" \
      torchrun --master_port=7777 --nproc_per_node="${NGPUS:-8}" train.py \
        -c /tmp/job/deimv2_cfg.yml --use-amp --seed=0 $RES_ARG \
        -u $UPD ${3:-} ) >> "$RUN_DIR/console.log" 2>&1
}

RC=0
if [ "${ARCH:-rfdetr}" = "deimv2" ]; then
  curl -sf --max-time 120 -o /tmp/job/deimv2_cfg.yml "${URL_CFG:-}" \
    || { echo "deimv2 cfg fetch failed"; upload cfg_missing -; exit 102; }
  mkdir -p "$RUN_DIR/out"
  if [ "${MODE:-train}" = "sanity" ]; then
    echo "== DEIMV2 PROFILE ==" > "$RUN_DIR/console.log"
    upload deimv2_profile -
    run_deimv2 2 "${PROFILE_TIMEOUT:-1500}" "${DEIMV2_UPDATES:-}"
    RC=$?
    echo "== DEIMV2 PROFILE rc=$RC ==" >> "$RUN_DIR/console.log"
  else
    BUDGET=$(( WALL_S - 900 ))
    echo "== DEIMV2 TRAIN budget=${BUDGET}s ==" > "$RUN_DIR/console.log"
    upload deimv2_training -
    run_deimv2 0 "$BUDGET" "${DEIMV2_UPDATES:-}"
    RC=$?
    echo "== DEIMV2 TRAIN rc=$RC ==" >> "$RUN_DIR/console.log"
  fi
  kill $BEACON_PID 2>/dev/null
  { [ $RC -eq 0 ] || [ $RC -eq 124 ]; } && BITS=$((BITS|2))
  upload "deimv2_end" "$RC" && BITS=$((BITS|8))
  [ $((BITS & 2)) -ne 0 ] && exit 0
  exit $((100+BITS))
fi
if [ "${MODE:-train}" = "sanity" ]; then
  echo "== OVERFIT-16 ==" > "$RUN_DIR/console.log"
  upload overfit -
  run_train overfit "$RUN_DIR" "${OVERFIT_EPOCHS:-50}" 2 1 1800 "--eval-interval 50"
  RC=$?
  echo "== OVERFIT rc=$RC ==" >> "$RUN_DIR/console.log"
  upload overfit_end "$RC"
  for cfg in ${PROFILE_LIST:-"2:1 4:1"}; do
    bs="${cfg%%:*}"; ga="${cfg##*:}"
    echo "== PROFILE bs=$bs ga=$ga ==" >> "$RUN_DIR/console.log"
    upload "profile_${bs}_${ga}" -
    run_train profile "$RUN_DIR/prof_${bs}_${ga}" 1 "$bs" "$ga" "${PROFILE_TIMEOUT:-720}" "--eval-interval 999"
    echo "== PROFILE bs=$bs ga=$ga rc=$? ==" >> "$RUN_DIR/console.log"
    upload "profile_${bs}_${ga}_end" -
  done
  RC=0
else
  NOW=$(date +%s); BUDGET=$(( WALL_S - 900 ))
  echo "== TRAIN budget=${BUDGET}s ==" >> "$RUN_DIR/console.log"
  upload training -
  run_train train "$RUN_DIR" "${EPOCHS:-50}" "${BS:-2}" "${GA:-1}" "$BUDGET" ""
  RC=$?
  echo "== TRAIN rc=$RC ==" >> "$RUN_DIR/console.log"
fi

kill $BEACON_PID 2>/dev/null
grep -q RFDETR_TRAIN_DONE "$RUN_DIR/console.log" && DONE=1 || DONE=0
{ [ $RC -eq 0 ] || [ $RC -eq 124 ]; } && BITS=$((BITS|2))
[ $DONE -eq 1 ] && BITS=$((BITS|4))
upload "$([ $DONE -eq 1 ] && echo done || echo ended)" "$RC" && BITS=$((BITS|8))
{ [ $((BITS & 2)) -ne 0 ]; } && exit 0
exit $((100+BITS))
