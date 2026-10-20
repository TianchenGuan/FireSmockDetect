#!/bin/bash
# V5b segmentation-branch pod entry (green g6e, 8x L40S, offline pods).
# Runs several trainer configs CONCURRENTLY on disjoint GPU sets on ONE node
# (each with its own --standalone rendezvous: the pod presets PET_RDZV_ENDPOINT /
# MASTER_PORT, which made 3 torchruns share one rendezvous in segV5b_a seg01):
#   CONFIGS="lora:0,1,2,3:2,frozen:4,5:4,r50:6,7:8"   (name:gpus:per-gpu-bs)
# Stages the data tar once (DATA_TARS name=presignedGET -> $V5B_DATA_ROOT, .done
# marker), fetches the code tgz, self-logs (no pod log channel) and uploads every
# artifact via presigned PUT. Resume: run dir on /scratch survives segments.
# exit 0 = all configs DONE + uploaded; 95 = staging failed; else 100+bits
#   (bit0 fetch ok, bit1 all trainers exited 0/124, bit2 all DONE, bit3 uploads ok)
set -u +e
export HOME=/tmp USER=worker LOGNAME=worker HF_HOME=/tmp/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/inductor_cache /tmp/triton_cache /tmp/hf
exec > /tmp/entry.log 2>&1
put() { [ -f "$2" ] && curl -sf --max-time 900 -T "$2" "$1"; }
ship_log() { [ -n "${URL_LOG:-}" ] && tail -c 60000 /tmp/entry.log > /tmp/log.txt \
  && curl -sf --max-time 120 -T /tmp/log.txt "$URL_LOG"; }
trap ship_log EXIT
NOW=$(date +%s)
BUDGET=$(( ${URL_EXPIRY_EPOCH:-$((NOW+2700))} - NOW - 420 ))
echo "budget ${BUDGET}s host $(hostname) gpus $(nvidia-smi -L 2>/dev/null | wc -l)"
RUN_ROOT=/scratch/scratch/tianrgua-sandbox/smoke/runs_v5b
RUN_DIR=$RUN_ROOT/$RUN_NAME
mkdir -p "$RUN_DIR" || { RUN_DIR=/tmp/runs/$RUN_NAME; mkdir -p "$RUN_DIR"; }
STATUS=/tmp/status.json
write_status() {
  local per=""
  for spec in ${CONFIGS//,/ }; do :; done
  for c in $CFG_NAMES; do
    local st="none"; [ -f "$RUN_DIR/$c/DONE" ] && st="done"
    local last=$(grep -oE '"step": [0-9]+' "$RUN_DIR/$c/metrics.jsonl" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
    local ev=$(grep -c '"eval"' "$RUN_DIR/$c/metrics.jsonl" 2>/dev/null)
    per="$per\"$c\":{\"state\":\"$st\",\"step\":\"${last:-none}\",\"evals\":\"${ev:-0}\"},"
  done
  cat > $STATUS <<EOS
{"run":"$RUN_NAME","phase":"$1","rc":"$2","budget_s":$BUDGET,"host":"$(hostname)",
 "ts":"$(date -u +%FT%TZ)","configs":{${per%,}}}
EOS
  curl -sf --max-time 60 -T "$STATUS" "$URL_STATUS"
}
# CONFIGS parse: name:gpus:bs  (gpus comma-separated -> use ';' between fields' entries)
CFG_NAMES=""; declare -A CFG_GPUS CFG_BS
IFS=' ' read -ra SPECS <<< "${CONFIGS// /}"
for spec in $(echo "$CONFIGS" | tr ';' ' '); do
  name="${spec%%:*}"; rest="${spec#*:}"; gpus="${rest%%:*}"; bs="${rest#*:}"
  CFG_NAMES="$CFG_NAMES $name"; CFG_GPUS[$name]="$gpus"; CFG_BS[$name]="$bs"
done
echo "configs:$CFG_NAMES"
write_status starting -
[ "$BUDGET" -lt 900 ] && { write_status skipped_budget -; exit 90; }

BITS=0
# ---- data staging (once per cluster)
V5B_DATA_ROOT="${V5B_DATA_ROOT:-/scratch/scratch/tianrgua-sandbox/smoke/v5bdata}"
if [ -n "${DATA_TARS:-}" ]; then
  mkdir -p "$V5B_DATA_ROOT" || exit 95
  IFS=',' read -ra TARS <<< "$DATA_TARS"
  for spec in "${TARS[@]}"; do
    name="${spec%%=*}"; url="${spec#*=}"
    [ -f "$V5B_DATA_ROOT/.${name}.done" ] && { echo "staged: $name"; continue; }
    echo "staging $name ..."
    curl -sf --max-time 2400 -o "$V5B_DATA_ROOT/${name}.tar" "$url" \
      && tar -xf "$V5B_DATA_ROOT/${name}.tar" -C "$V5B_DATA_ROOT" \
      && rm -f "$V5B_DATA_ROOT/${name}.tar" \
      && touch "$V5B_DATA_ROOT/.${name}.done" || { write_status staging_failed -; exit 95; }
  done
fi
DATA=$V5B_DATA_ROOT/seg_v5b
[ -f "$DATA/manifest.json" ] || { echo "manifest missing"; write_status data_missing -; exit 95; }
[ -d "${MODEL_DIR:?}" ] || { echo "MODEL_DIR missing"; write_status model_missing -; exit 95; }
# ---- code
mkdir -p /tmp/job; F=1
curl -sf --max-time 180 -o /tmp/job/code.tgz "$URL_CODE" && tar xzf /tmp/job/code.tgz -C /tmp/job || F=0
[ $F -eq 1 ] && BITS=$((BITS|1))
ls /tmp/job
python -c "import torch, torchvision, transformers; print(torch.__version__, torchvision.__version__, transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -8

TRAIN_BUDGET=$((BUDGET-780))   # leave time for final eval+dump+uploads
write_status running -
# heartbeat: status + console tails every 4 min
( while true; do sleep 240; write_status running -; for c in $CFG_NAMES; do
    v="URL_${c^^}_CONSOLE"; [ -f "$RUN_DIR/$c/console.log" ] && tail -c 12000 "$RUN_DIR/$c/console.log" > /tmp/ct_$c.txt && put "${!v}" /tmp/ct_$c.txt; done; done ) &
HB=$!

i=0; declare -A PIDS
for c in $CFG_NAMES; do
  gpus="${CFG_GPUS[$c]}"; bs="${CFG_BS[$c]}"; n=$(echo "$gpus" | tr ',' '\n' | wc -l)
  mkdir -p "$RUN_DIR/$c"
  case "$c" in
    lora)   KARGS="--kind dino_lora --backbone-dir $MODEL_DIR --lora 16 --grad-ckpt --backbone-lr ${LORA_LR:-1e-4}" ;;
    frozen) KARGS="--kind dino_frozen --backbone-dir $MODEL_DIR" ;;
    r50)    KARGS="--kind r50 --r50-weights $DATA/resnet50-11ad3fa6.pth" ;;
    *) echo "unknown config $c"; continue ;;
  esac
  RES=""; [ -f "$RUN_DIR/$c/last.pt" ] && [ ! -f "$RUN_DIR/$c/DONE" ] && RES="--resume $RUN_DIR/$c/last.pt"
  if [ -f "$RUN_DIR/$c/DONE" ]; then echo "$c already DONE"; continue; fi
  echo "launch $c on gpus $gpus (n=$n bs=$bs) $RES"
  CUDA_VISIBLE_DEVICES=$gpus timeout --signal=TERM --kill-after=240 "$BUDGET" \
    env -u MASTER_ADDR -u MASTER_PORT -u PET_RDZV_ENDPOINT -u PET_RDZV_BACKEND -u PET_RDZV_ID -u PET_MASTER_ADDR -u PET_MASTER_PORT \
    torchrun --standalone --nproc_per_node "$n" /tmp/job/train_seg.py $KARGS $RES \
      --manifest "$DATA/manifest.json" --data-root "$DATA" \
      --width "${WIDTH:-1280}" --height "${HEIGHT:-736}" --epochs "${EPOCHS:-10}" --bs "$bs" \
      --lr "${LR:-2e-4}" --wd 1e-4 --warmup-steps "${WARMUP:-50}" --workers "${WORKERS:-4}" \
      --output-dir "$RUN_DIR/$c" --time-budget-s "$TRAIN_BUDGET" --dump-probs ${EXTRA_ARGS:-} \
      > "$RUN_DIR/$c/console.log" 2>&1 &
  PIDS[$c]=$!; i=$((i+1))
done
ALL_OK=1
for c in "${!PIDS[@]}"; do wait "${PIDS[$c]}"; rc=$?; echo "$c rc=$rc"; { [ $rc -eq 0 ] || [ $rc -eq 124 ]; } || ALL_OK=0; done
kill $HB 2>/dev/null
[ $ALL_OK -eq 1 ] && BITS=$((BITS|2))
ALL_DONE=1; for c in $CFG_NAMES; do [ -f "$RUN_DIR/$c/DONE" ] || ALL_DONE=0; done
[ $ALL_DONE -eq 1 ] && BITS=$((BITS|4))
# ---- uploads
UP=1
for c in $CFG_NAMES; do
  d="$RUN_DIR/$c"; C="${c^^}"
  grep -m4 -B3 -A25 -E "Traceback|OutOfMemoryError|Error" "$d/console.log" 2>/dev/null | head -c 20000 > /tmp/ct_$c.txt
  echo "===== TAIL =====" >> /tmp/ct_$c.txt; tail -c 20000 "$d/console.log" >> /tmp/ct_$c.txt 2>/dev/null
  for pair in "BEST:best.pt" "LAST:last.pt" "METRICS:metrics.jsonl" "VALMETRICS:val_metrics_best.json" \
              "PERIMG:val_per_image.json" "PROBS:val_probs.tar" "PARAMS:param_counts.json"; do
    k="${pair%%:*}"; f="${pair#*:}"; v="URL_${C}_$k"; url="${!v:-}"
    [ -z "$url" ] && continue
    [ -f "$d/$f" ] && { put "$url" "$d/$f" || { echo "upload failed $c/$f"; UP=0; }; }
  done
  v="URL_${C}_CONSOLE"; put "${!v:-}" /tmp/ct_$c.txt
done
[ $UP -eq 1 ] && BITS=$((BITS|8))
write_status "$([ $ALL_DONE -eq 1 ] && echo done || echo segment_end)" "$BITS"
echo "BITS=$BITS"
[ $ALL_DONE -eq 1 ] && [ $UP -eq 1 ] && exit 0
exit $((100+BITS))
