#!/bin/bash
# V5b validation-set predictions for a set of YOLO checkpoints (one 1-node job):
# sdis VAL, boreal heinola VAL, FireMan val_s5 (v5b), then local eval_detection
# sweeps -> reports/v5b/val/eval_<sys>_<set>.json. Usage:
#   bash cluster/v5b/run_val_eval.sh <tag> <sys>=<ckpt> [<sys>=<ckpt> ...]
set -euo pipefail
cd /local/home/tianrgua/cvproject; source .venv/bin/activate
TAG="$1"; shift; CK=(); for s in "$@"; do CK+=(--ckpt "$s"); done
export SMOKE_REGISTRY_DIR=cluster/v5b
python cluster/v4/registry.py reserve --job "res-$TAG" --nodes 1 --gpu-type L40S --track eval --note "val preds $*" || exit 3
python cluster/v4/submit_yolo_eval.py --tag "$TAG" "${CK[@]}" \
  --list sdisval=/mnt/local/smoke_data/lists/pod/sdis_val.txt \
  --list borealval=/scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evalval/boreal_val/lists/pod/val.txt \
  --list firemanval=/scratch/scratch/tianrgua-sandbox/smoke/v5bdata/eval_v5b/fireman_val_s5/lists/pod/fireman_val_s5.txt \
  --data-tar v4_evalval=data/v4/stage/v4_evalval.tar \
  --data-tar eval_v5b@/scratch/scratch/tianrgua-sandbox/smoke/v5bdata=data/v5b/stage/eval_v5b.tar \
  --rootmap /mnt/local/smoke_data/pyro_sdis=/local/home/tianrgua/cvproject/data/yolo/pyro_sdis \
  --rootmap /scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evalval=/local/home/tianrgua/cvproject/data/v4/eval \
  --rootmap /scratch/scratch/tianrgua-sandbox/smoke/v5bdata/eval_v5b=/local/home/tianrgua/cvproject/data/v5b/eval \
  --out models/v5b/evals; RC=$?
python cluster/v4/registry.py release --job "res-$TAG" --outcome "RC$RC"
[ $RC -ne 0 ] && { echo "preds incomplete: models/v5b/evals/$TAG/entry.log"; exit 4; }
mkdir -p reports/v5b/val
for s in "$@"; do n=${s%%=*}
  for set in sdisval borealval firemanval; do
    case $set in sdisval) Y=data/yolo/yamls/local/expB.yaml; SP=val;; borealval) Y=data/v4/eval/yamls/boreal_val.yaml; SP=val;; firemanval) Y=data/v5b/eval/yamls/fireman_val_s5.yaml; SP=val;; esac
    python -m src.eval.eval_detection --data-yaml $Y --split $SP --preds-jsonl models/v5b/evals/$TAG/${n}__${set}.jsonl --out reports/v5b/val/eval_${n}_${set}.json > logs/v5b/val_${n}_${set}.log 2>&1 || echo "EVAL FAIL $n $set"
  done
done
echo "VAL EVAL DONE $TAG"
