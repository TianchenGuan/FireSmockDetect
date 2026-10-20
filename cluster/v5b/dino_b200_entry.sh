#!/bin/bash
# cluster/v5b/dino_b200_entry.sh — V5b high-res DINOv3-L LoRA detection run on the
# obsidian-pu B200 node (ap-south-1). Delivered inline (ENTRY_B64 env) by
# cluster/v5b/orchestrate_b200_dino.py; image smoke-dino3-b200:v1 has code, data,
# backbone baked under /opt (reports/v5b/B200_PROBE.md: no S3 egress, pods are uid
# 1001, non-zero exit => automatic restart, exit code not surfaced). Rules:
#   * ALWAYS exit 0; status lives in $ST (json) and is read via `kraken jobs connect`.
#   * one segment, no URL expiry: train with a wall budget, then val inference,
#     then hold the pod RETAIN_S seconds (touch /tmp/ack to release early).
#   * torchrun --standalone with the operator PET_* rendezvous vars unset (probe §4).
set -u +e
export HOME=/tmp USER=worker LOGNAME=worker PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PET_RDZV_BACKEND PET_RDZV_ENDPOINT PET_RDZV_ID PET_NNODES PET_NPROC_PER_NODE PET_MAX_RESTARTS PET_NODE_RANK
mkdir -p /tmp/hf /tmp/inductor_cache /tmp/triton_cache

RUN="${RUN_NAME:-v5bB200_lora_L_1920}"
ROOT=""; for d in /mnt/local/v5b_dino /tmp/v5b_dino; do mkdir -p "$d" 2>/dev/null && touch "$d/.w" 2>/dev/null && { ROOT=$d; break; }; done
RUN_DIR=$ROOT/$RUN; mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/entry.log; exec > >(tee -a "$LOG") 2>&1
ST=/tmp/status.json; T0=$(date +%s)
CODE=/opt/smoke_dino; COCO=$CODE/coco; BB=/opt/models/dinov3-vitl16
NG=$(nvidia-smi -L 2>/dev/null | wc -l)
st() {  # phase, extra-json-fragment
  STEP=$(grep -oE '"step": [0-9]+' "$RUN_DIR/metrics.jsonl" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
  LOSS=$(grep -oE '"loss": [0-9.]+' "$RUN_DIR/metrics.jsonl" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  EV=$(grep -c '"eval"' "$RUN_DIR/metrics.jsonl" 2>/dev/null || echo 0)
  BEST=$( [ -f "$RUN_DIR/best.pt" ] && stat -c '%s %Y' "$RUN_DIR/best.pt" || echo "0 0")
  LAST=$( [ -f "$RUN_DIR/last.pt" ] && stat -c '%s %Y' "$RUN_DIR/last.pt" || echo "0 0")
  echo "{\"run\":\"$RUN\",\"phase\":\"$1\",\"t\":$(( $(date +%s)-T0 )),\"gpus\":$NG,\"step\":\"${STEP:-none}\",\"loss\":\"${LOSS:-none}\",\"evals\":$EV,\"best\":\"$BEST\",\"last\":\"$LAST\",\"root\":\"$ROOT\",\"host\":\"$(hostname)\",\"ts\":\"$(date -u +%FT%TZ)\"${2:+,$2}}" > $ST
  echo "== [$(( $(date +%s)-T0 ))s] phase=$1 ${2:-}"
}
st start
# restart guard: a restarted container (same node) finds the marker and just idles for retrieval
if [ -f "$RUN_DIR/EXITED" ]; then st restarted_idle; for i in $(seq 1 $(( ${RETAIN_S:-1800}/10 ))); do [ -f /tmp/ack ] && break; sleep 10; done; exit 0; fi
echo "host=$(hostname) uid=$(id -u) nproc=$(nproc) gpus=$NG root=$ROOT"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -8; df -h $ROOT /tmp | tail -2
python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__,'sm',torch.cuda.get_device_capability(0))"

W=${WIDTH:-1920}; H=${HEIGHT:-1088}; BS=${BS:-4}; EPOCHS=${EPOCHS:-10}; WORKERS=${WORKERS:-8}
COMMON="--backbone-dir $BB --train-json $COCO/instances_train_eb.json --image-root /opt/smoke_data --width $W --height $H --lr ${LR:-1e-4} --wd ${WD:-1e-4} --lora ${LORA_R:-16} --backbone-lr ${BACKBONE_LR:-1e-4} --grad-ckpt --seed ${SEED:-0}"

# ---- phase 1: memory/throughput probe at candidate batch sizes (short, no val)
if [ -n "${MEMPROBE_BS:-}" ]; then
  for b in $MEMPROBE_BS; do
    st memprobe "\"probe_bs\":$b"
    PD=$RUN_DIR/memprobe_bs$b; mkdir -p $PD
    timeout --signal=TERM --kill-after=60 ${MEMPROBE_TIMEOUT:-900} torchrun --standalone --nproc_per_node $NG $CODE/train_det.py $COMMON \
      --bs $b --max-steps ${MEMPROBE_STEPS:-40} --warmup-steps 10 --workers $WORKERS --save-every-steps 0 --output-dir $PD > $PD/console.log 2>&1
    rc=$?; oom=$(grep -c "OutOfMemoryError\|CUDA out of memory" $PD/console.log)
    echo "memprobe bs=$b rc=$rc oom=$oom $(grep -oE 'step [0-9]+/[0-9]+ .*' $PD/console.log | tail -1)"
    python - "$PD" "$b" "$rc" "$oom" "$NG" <<'PY'
import json,sys,re
pd,b,rc,oom,ng=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
rows=[json.loads(l) for l in open(f"{pd}/metrics.jsonl")] if rc==0 and oom==0 else []
rows=[r for r in rows if "s_per_it" in r]
tail=rows[1:] if len(rows)>1 else rows
spi=sum(r["s_per_it"] for r in tail)/len(tail) if tail else None
rec={"bs":b,"rc":rc,"oom":bool(oom),"peak_mem_gib":max([r["peak_mem_gib"] for r in rows],default=None),
     "s_per_it":spi,"img_per_s_node":(b*ng/spi if spi else None),"n_log_rows":len(rows)}
open(f"{pd}/../memprobe.jsonl","a").write(json.dumps(rec)+"\n"); print("MEMPROBE",json.dumps(rec))
PY
    rm -f $PD/last.pt
  done
fi

# ---- phase 2: training (single segment, wall budget; resume if a last.pt exists)
st train
RES=""; [ -f "$RUN_DIR/last.pt" ] && RES="--resume $RUN_DIR/last.pt"
TRAIN_BUDGET=${TRAIN_BUDGET_S:-21600}
timeout --signal=TERM --kill-after=240 $((TRAIN_BUDGET+900)) torchrun --standalone --nproc_per_node $NG $CODE/train_det.py $COMMON $RES \
  --val-json $COCO/instances_val.json --bs $BS --epochs $EPOCHS --warmup-steps ${WARMUP:-500} --workers $WORKERS \
  --save-every-steps ${SAVE_STEPS:-400} --eval-max-images ${EVAL_MAX:-1500} --eval-every-epochs 1 \
  --output-dir $RUN_DIR --time-budget-s $TRAIN_BUDGET > $RUN_DIR/console.log 2>&1
TRC=$?; DONE=$([ -f $RUN_DIR/DONE ] && echo 1 || echo 0)
echo "train rc=$TRC done=$DONE $(grep -oE 'step [0-9]+/[0-9]+ .*' $RUN_DIR/console.log | tail -1)"
grep -m3 -A12 -E "Traceback|OutOfMemoryError|RuntimeError" $RUN_DIR/console.log | head -60 > $RUN_DIR/console_errors.txt
# strip the optimizer from best.pt for a fast pull (model-only copy; full file kept)
python - "$RUN_DIR" <<'PY'
import torch,sys,os
rd=sys.argv[1]
for n in ("best.pt","last.pt"):
    p=f"{rd}/{n}"
    if os.path.exists(p):
        ck=torch.load(p,map_location="cpu",weights_only=False)
        torch.save({k:v for k,v in ck.items() if k!="opt"}, f"{rd}/{n[:-3]}_model.pt")
        print(n,"step",ck.get("step"),"best_map",ck.get("best_map"),"->",f"{n[:-3]}_model.pt",os.path.getsize(f"{rd}/{n[:-3]}_model.pt"),"B")
PY
st trained "\"train_rc\":$TRC,\"done\":$DONE"

# ---- phase 3: full sdis_val inference with best.pt at the training resolution (8 shards)
CK=$RUN_DIR/best.pt; [ -f "$CK" ] || CK=$RUN_DIR/last.pt
if [ -f "$CK" ] && [ "${SKIP_INFER:-0}" != 1 ]; then
  st infer
  PR=$RUN_DIR/preds; mkdir -p $PR
  PIDS=""  # NOTE: a bare `wait` also waits for the `exec > >(tee ...)` process substitution and never returns
  for i in $(seq 0 $((NG-1))); do
    EXTRA=""; [ $i -eq 0 ] && EXTRA="--latency-n 50 --latency-out $PR/latency_b200.json"
    python $CODE/dino_infer.py --backbone-dir $BB --ckpt $CK --list $COCO/sdis_val_local.txt --shard $i/$NG --device-index $i \
      --width $W --height $H --lora ${LORA_R:-16} --conf-floor 0.02 \
      --root-map /local/home/tianrgua/cvproject/data/yolo=/opt/smoke_data --out $PR/shard$i.jsonl $EXTRA > $PR/shard$i.log 2>&1 &
    PIDS="$PIDS $!"
  done
  wait $PIDS
  cat $PR/shard*.jsonl > $RUN_DIR/sdis_val_preds.jsonl; gzip -kf $RUN_DIR/sdis_val_preds.jsonl
  echo "preds rows $(wc -l < $RUN_DIR/sdis_val_preds.jsonl) gz $(stat -c%s $RUN_DIR/sdis_val_preds.jsonl.gz) B"; grep -h "DONE" $PR/shard*.log | head -8
  st inferred "\"preds_rows\":$(wc -l < $RUN_DIR/sdis_val_preds.jsonl)"
fi

# ---- phase 4: hold for retrieval (connect pulls), then exit 0
echo "$(date -u +%FT%TZ) rc=$TRC" > $RUN_DIR/EXITED
st retain
for i in $(seq 1 $(( ${RETAIN_S:-3600}/10 ))); do [ -f /tmp/ack ] && break; sleep 10; done
st exit; exit 0
