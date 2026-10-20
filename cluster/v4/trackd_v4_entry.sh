#!/bin/bash
# Track D v4 entrypoint (green g6e pods, offline). MODE=extract | train.
#
# extract: stage the v4 frame tar onto /scratch, then 8-way GPU-sharded frozen
#          expE feature extraction for the v4 train-extension + val-replay
#          sequences, using the UNCHANGED src/v3/trackd_features.py so the
#          temporal head input space (194ch 12x20 fp16) is identical to v3.
# train:   build a merged feature dir (v3 train + v4 train_ext -> train/,
#          v3 val -> val/) by symlink, then train the four head variants
#          (temporal/t1-only x seed 0/1) and PUT the artifacts back.
#
# exit 0 = success; 90 = skipped (no budget); else 100+bitmask
# (bit0 inputs fetched, bit1 work done, bit2 uploads ok).
# No pod log access -> everything is self-logged to /tmp/entry.log and PUT.
set -u +e
export HOME=/tmp USER=worker LOGNAME=worker
export HF_HOME=/tmp/hf YOLO_CONFIG_DIR=/tmp/yolo_cfg
mkdir -p /tmp/hf /tmp/yolo_cfg /tmp/job
exec > /tmp/entry.log 2>&1

BITS=0
NOW=$(date +%s)
BUDGET=$(( ${URL_EXPIRY_EPOCH:-$((NOW+2700))} - NOW - 420 ))
echo "mode=$MODE budget=${BUDGET}s host=$(hostname) gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"
[ "$BUDGET" -lt 600 ] && exit 90

SBX=/scratch/scratch/tianrgua-sandbox/smoke
FRAMES=$SBX/v4data/trackd_v4
F3=$SBX/trackd_v3/features
F4=$SBX/trackd_v4/features
MERGED=$SBX/trackd_v4/merged
RUNROOT=$SBX/trackd_v4/head_runs
mkdir -p "$F4" "$RUNROOT" "$SBX/v4data" || { echo "scratch unwritable"; exit 95; }

put() { [ -f "$2" ] && curl -sf --max-time "${3:-300}" -T "$2" "$1"; }
finish() { tail -c 60000 /tmp/entry.log > /tmp/entry_tail.txt
           put "${URL_ENTRYLOG:-}" /tmp/entry_tail.txt 120; }
status() {
  cat > /tmp/status.json <<EOF
{"mode":"$MODE","phase":"$1","detail":"$2","gpus":$(nvidia-smi -L 2>/dev/null | wc -l),
 "host":"$(hostname)","ts":"$(date -u +%FT%TZ)"}
EOF
  curl -sf --max-time 60 -T /tmp/status.json "$URL_STATUS"
}
status starting -

# ---------------------------------------------------------------- inputs
F=1
curl -sf --max-time 120 -o /tmp/job/trackd_features.py "$URL_FEAT_CODE" || F=0
curl -sf --max-time 120 -o /tmp/job/trackd_temporal.py "$URL_HEAD_CODE" || F=0
curl -sf --max-time 300 -o /tmp/job/weights.pt "$URL_WEIGHTS" || F=0
curl -sf --max-time 180 -o /tmp/job/seq_train_ext.json "$URL_SEQ_TRAIN_EXT" || F=0
curl -sf --max-time 180 -o /tmp/job/seq_val_replay.json "$URL_SEQ_VAL_REPLAY" || F=0
[ $F -eq 1 ] || { status fetch_failed code_or_weights; finish; exit $((100+BITS)); }

# v4 frames onto /scratch (idempotent; ~16 GB)
if [ -f "$SBX/v4data/.trackd_v4.done" ]; then
  echo "frames already staged: $(find "$FRAMES" -name '*.jpg' | wc -l) jpg"
else
  status staging_frames -
  curl -sf --max-time 2400 -o "$SBX/v4data/trackd_v4.tar" "$URL_FRAMES_TAR" \
    && tar -xf "$SBX/v4data/trackd_v4.tar" -C "$SBX/v4data" \
    && rm -f "$SBX/v4data/trackd_v4.tar" && touch "$SBX/v4data/.trackd_v4.done" \
    || { rm -f "$SBX/v4data/trackd_v4.tar"; status stage_failed frames; finish; exit 96; }
  echo "staged frames: $(find "$FRAMES" -name '*.jpg' | wc -l) jpg"
fi

# v3 features onto /scratch (iteration-3 extract may have been evicted)
if [ -d "$F3/train" ] && [ -d "$F3/val" ]; then
  echo "v3 features present: train=$(ls "$F3/train" | grep -c '\.npz$') val=$(ls "$F3/val" | grep -c '\.npz$')"
else
  status staging_v3_features -
  mkdir -p "$F3"
  curl -sf --max-time 1800 -o /tmp/f3v.tgz "${URL_FEATS_V3_VAL:-}" \
    && tar -xzf /tmp/f3v.tgz -C "$F3" && rm -f /tmp/f3v.tgz || F=0
  curl -sf --max-time 2400 -o /tmp/f3t.tgz "${URL_FEATS_V3_TRAIN:-}" \
    && tar -xzf /tmp/f3t.tgz -C "$F3" && rm -f /tmp/f3t.tgz || F=0
  echo "v3 features restored ok=$F train=$(ls "$F3/train" 2>/dev/null | grep -c '\.npz$') val=$(ls "$F3/val" 2>/dev/null | grep -c '\.npz$')"
fi
BITS=$((BITS|1))

ROOTMAP="/local/home/tianrgua/cvproject/data/v4/stage/trackd_v4_frames=$FRAMES"

# ---------------------------------------------------------------- extract
if [ "$MODE" = "extract" ]; then
  status extracting -
  NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || NG=1
  PIDS=()
  for i in $(seq 0 $((NG-1))); do
    CUDA_VISIBLE_DEVICES=$i python /tmp/job/trackd_features.py \
      --weights /tmp/job/weights.pt \
      --sequences /tmp/job/seq_train_ext.json /tmp/job/seq_val_replay.json \
      --out "$F4" --device cuda:0 --imgsz 1280 \
      --image-root-map "$ROOTMAP" \
      --shard-idx "$i" --shard-n "$NG" > "/tmp/shard$i.log" 2>&1 &
    PIDS+=($!)
  done
  RC=0
  for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
  cat /tmp/shard*.log | tail -c 40000 > /tmp/console.txt
  NTR=$(ls "$F4/train_ext" 2>/dev/null | grep -c '\.npz$')
  NVA=$(ls "$F4/val_replay" 2>/dev/null | grep -c '\.npz$')
  echo "rc=$RC train_ext_npz=$NTR val_replay_npz=$NVA" | tee -a /tmp/console.txt
  status extracted "rc=$RC train_ext=$NTR val_replay=$NVA"
  put "$URL_CONSOLE" /tmp/console.txt
  if [ "$NTR" -eq "${EXPECT_TRAIN_EXT:-23}" ] && [ "$NVA" -eq "${EXPECT_VAL_REPLAY:-139}" ]; then
    BITS=$((BITS|2))
    U=1
    tar czf /tmp/feats_v4.tgz -C "$F4" train_ext val_replay || U=0
    put "$URL_FEATS_V4" /tmp/feats_v4.tgz 1800 || U=0
    [ $U -eq 1 ] && BITS=$((BITS|4))
    status done "train_ext=$NTR val_replay=$NVA up=$U bytes=$(stat -c%s /tmp/feats_v4.tgz 2>/dev/null)"
  else
    status incomplete "train_ext=$NTR val_replay=$NVA (want ${EXPECT_TRAIN_EXT:-23}/${EXPECT_VAL_REPLAY:-139})"
  fi

# ---------------------------------------------------------------- train
elif [ "$MODE" = "train" ]; then
  [ -d "$F4/train_ext" ] || { status features_missing v4; finish; exit 102; }
  [ -d "$F3/train" ] && [ -d "$F3/val" ] || { status features_missing v3; finish; exit 103; }
  # merged view: train = v3 train + v4 train_ext, val = v3 val (selection set)
  rm -rf "$MERGED"; mkdir -p "$MERGED/train" "$MERGED/val"
  ln -sf "$F3"/train/*.npz "$MERGED/train/" 2>/dev/null
  # AUSMOKE_KEEP caps how many AuSmoke sequences enter the extension. The full
  # extension is 10,905 all-positive frames vs 22,170 v3 frames, which is enough
  # to let the head learn a DOMAIN cue instead of a smoke cue; capping it is the
  # ablation that isolates that mechanism.
  if [ -n "${AUSMOKE_KEEP:-}" ]; then
    ls "$F4"/train_ext/*ausmoke*.npz 2>/dev/null | sort | head -n "$AUSMOKE_KEEP" \
      | while read -r f; do ln -sf "$f" "$MERGED/train/"; done
    ln -sf "$F4"/train_ext/*boreal*.npz "$MERGED/train/" 2>/dev/null
    echo "AUSMOKE_KEEP=$AUSMOKE_KEEP -> linked $(ls "$MERGED/train" | grep -c ausmoke) ausmoke npz"
  else
    ln -sf "$F4"/train_ext/*.npz "$MERGED/train/" 2>/dev/null
  fi
  ln -sf "$F3"/val/*.npz "$MERGED/val/" 2>/dev/null
  NTR=$(ls "$MERGED/train" | grep -c '\.npz$'); NVA=$(ls "$MERGED/val" | grep -c '\.npz$')
  echo "merged: train=$NTR val=$NVA"
  status training "merged train=$NTR val=$NVA"
  [ "$NTR" -ge 900 ] || { status merge_failed "train=$NTR"; finish; exit 104; }

  RCALL=0
  for spec in $RUN_SPECS; do
    # spec = name:seed:flags   (flags "-" means none)
    NAME=$(echo "$spec" | cut -d: -f1)
    SEED=$(echo "$spec" | cut -d: -f2)
    FLAG=$(echo "$spec" | cut -d: -f3); [ "$FLAG" = "-" ] && FLAG=""
    RD="$RUNROOT/$NAME"; mkdir -p "$RD"
    LEFT=$(( ${URL_EXPIRY_EPOCH:-$((NOW+2700))} - $(date +%s) - 420 ))
    [ "$LEFT" -lt 240 ] && { echo "out of budget before $NAME"; break; }
    echo "=== run $NAME seed=$SEED flags='$FLAG' left=${LEFT}s ==="
    timeout --signal=TERM --kill-after=120 "$LEFT" \
      python /tmp/job/trackd_temporal.py \
        --features "$MERGED" --out "$RD" --resume \
        --epochs "${EPOCHS:-12}" --bs "${BS:-256}" --lr "${LR:-1e-3}" \
        --seed "$SEED" --device cuda:0 --workers "${WORKERS:-8}" $FLAG \
        > "$RD/console.log" 2>&1
    RC=$?
    echo "run $NAME rc=$RC"; tail -n 25 "$RD/console.log"
    grep -q TEMPORAL_TRAIN_DONE "$RD/console.log" || RCALL=1
  done
  NDONE=$(ls "$RUNROOT"/*/done.json 2>/dev/null | wc -l)
  echo "runs with done.json: $NDONE"
  [ "$NDONE" -ge "${EXPECT_RUNS:-4}" ] && BITS=$((BITS|2))
  U=1
  tar czf /tmp/heads.tgz -C "$RUNROOT" . || U=0
  put "$URL_HEADS" /tmp/heads.tgz 900 || U=0
  [ $U -eq 1 ] && BITS=$((BITS|4))
  tail -c 60000 /tmp/entry.log > /tmp/console.txt
  put "$URL_CONSOLE" /tmp/console.txt
  if [ $((BITS & 2)) -ne 0 ]; then status done "runs=$NDONE rcall=$RCALL"
  else status segment_end "runs=$NDONE rcall=$RCALL"; fi
else
  echo "bad MODE"; finish; exit 99
fi

finish
[ $((BITS & 2)) -ne 0 ] && [ $((BITS & 4)) -ne 0 ] && exit 0
exit $((100+BITS))
