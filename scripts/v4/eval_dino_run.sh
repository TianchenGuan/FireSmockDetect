#!/bin/bash
# Iteration-4 DINOv3 val evaluation for ONE trained run, on all three val sets
# (sdis_val + boreal_val + forest_val) in a single cluster job, then the local
# offline sweep for each set.
#
#   bash scripts/v4/eval_dino_run.sh <run> <backbone-dir> [ckpt-basename]
#
# Cluster side: cluster/v3/submit_dino_eval.py ships dino_infer.py; sdis_val
# images are on /scratch/.../smoke_data_v2 and the two v4 sets on
# /scratch/.../smoke/v4data/v4_evalval (staged once by
# cluster/v4/stage_evalval.py) -> no frames tar needed. Lists shipped are the
# LOCAL lists, so preds keys are local paths and match the local GT lookup.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

RUN="${1:?run name}"
BB="${2:?backbone dir on /scratch}"
CKPT_NAME="${3:-best.pt}"
CKPT="/scratch/scratch/tianrgua-sandbox/smoke/runs_v3/${RUN}/${CKPT_NAME}"
EVAL_NAME="${RUN}_val"
ROOTMAP="/local/home/tianrgua/cvproject/data/yolo=/scratch/scratch/tianrgua-sandbox/smoke_data_v2,/local/home/tianrgua/cvproject/data/v4/eval=/scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evalval"

# LoRA-trained checkpoints carry lora_A/B keys -> the inference model must
# get the same injection (rank 16, alpha default 2r, as trained) or every
# shard dies with "ckpt mismatch ... unexpected". Full-unfreeze/frozen: none.
INFER_EXTRA="${INFER_EXTRA_OVERRIDE:-}"
if [ -z "$INFER_EXTRA" ] && [[ "$RUN" == *lora* ]]; then INFER_EXTRA="--lora 16"; fi
echo "infer extra flags: '${INFER_EXTRA}'"

RES="res-ev$(echo "$RUN" | tr -d '_' | tr '[:upper:]' '[:lower:]')"
python cluster/v4/registry.py reserve --job "$RES" --nodes 1 --track dino \
  --note "val eval $RUN"

set +e
python cluster/v3/submit_dino_eval.py \
  --eval-name "$EVAL_NAME" \
  --backbone-dir "$BB" \
  --ckpt "$CKPT" \
  --splits sdis_val \
  --extra-list "$(pwd)/data/v4/eval/boreal_val/lists/local/val.txt=boreal_val" \
  --extra-list "$(pwd)/data/v4/eval/forest_val/lists/local/val.txt=forest_val" \
  --rootmap-override "$ROOTMAP" \
  --width 1280 --height 736 --conf-floor 0.02 \
  ${INFER_EXTRA:+--infer-extra "$INFER_EXTRA"}
RC=$?
set -e
python cluster/v4/registry.py release --job "$RES" --outcome "eval_rc=$RC"

DEST="models/v4/evals/${RUN}"
mkdir -p "$DEST"
for s in sdis_val boreal_val forest_val; do
  SRC="models/v3/evals/${EVAL_NAME}/${s}_preds.jsonl.gz"
  if [ -f "$SRC" ]; then
    gunzip -cf "$SRC" > "${DEST}/${s}_preds.jsonl"
    echo "$s rows: $(wc -l < "${DEST}/${s}_preds.jsonl")"
  else
    echo "MISSING preds: $SRC"
  fi
done
cp -f "models/v3/evals/${EVAL_NAME}/shardlogs.txt" "$DEST/" 2>/dev/null || true

# hard guard: an empty preds file means the shards crashed (see shardlogs);
# never sweep it (an empty sweep looks like a valid report file).
BAD=0
for s in sdis_val boreal_val forest_val; do
  P="${DEST}/${s}_preds.jsonl"
  if [ ! -s "$P" ]; then echo "EMPTY PREDS for $s"; rm -f "$P"; BAD=1; fi
done
if [ "$BAD" = 1 ]; then
  echo "=== shardlogs error context ==="
  grep -a -m3 -A8 "Traceback" "$DEST/shardlogs.txt" 2>/dev/null | head -40
  echo "EVAL FAILED $RUN (empty predictions) -- no sweeps written"
  exit 2
fi

mkdir -p reports/v4
for spec in "sdis_val:data/yolo/yamls/local/expB.yaml" \
            "boreal_val:data/v4/eval/yamls/boreal_val.yaml" \
            "forest_val:data/v4/eval/yamls/forest_val.yaml"; do
  S="${spec%%:*}"; Y="${spec#*:}"
  P="${DEST}/${S}_preds.jsonl"
  [ -f "$P" ] || { echo "skip local sweep for $S (no preds)"; continue; }
  python -m src.eval.eval_detection --data-yaml "$Y" --split val \
    --preds-jsonl "$P" --out "reports/v4/eval_${RUN}_${S}.json" \
    > "logs/v4/localsweep_${RUN}_${S}.log" 2>&1 \
    && echo "swept $S -> reports/v4/eval_${RUN}_${S}.json" \
    || { echo "LOCAL SWEEP FAILED $S"; tail -5 "logs/v4/localsweep_${RUN}_${S}.log"; }
done
echo "EVAL DONE $RUN (submit rc=$RC)"
