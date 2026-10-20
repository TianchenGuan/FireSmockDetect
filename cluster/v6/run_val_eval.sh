#!/bin/bash
# V6 validation-set predictions for YOLO checkpoints (one 1-node job on use1):
# sdis VAL + boreal heinola VAL, then local eval_detection sweeps ->
# reports/v6/val/eval_<sys>_<set>.json. Usage:
#   bash cluster/v6/run_val_eval.sh <tag> <sys>=<ckpt> [...]
set -euo pipefail
cd /local/home/tianrgua/cvproject; source .venv/bin/activate
TAG="$1"; shift; CK=(); for s in "$@"; do CK+=(--ckpt "$s"); done
export SMOKE_REGISTRY_DIR=cluster/v6
python cluster/v4/registry.py reserve --job "res-$TAG" --nodes 1 --gpu-type L40S --track eval --note "v6 val preds $*" || exit 3
python cluster/v4/submit_yolo_eval.py --tag "$TAG" "${CK[@]}" \
  --list sdisval=/mnt/local/smoke_data/lists/pod/sdis_val.txt \
  --list borealval=/scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evalval/boreal_val/lists/pod/val.txt \
  --data-tar v4_evalval=data/v4/stage/v4_evalval.tar \
  --rootmap /mnt/local/smoke_data/pyro_sdis=/local/home/tianrgua/cvproject/data/yolo/pyro_sdis \
  --rootmap /scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evalval=/local/home/tianrgua/cvproject/data/v4/eval \
  --out models/v6/evals; RC=$?
python cluster/v4/registry.py release --job "res-$TAG" --outcome "RC$RC"
[ $RC -ne 0 ] && { echo "preds incomplete: models/v6/evals/$TAG/entry.log"; exit 4; }
mkdir -p reports/v6/val logs/v6
for s in "$@"; do n=${s%%=*}
  for set in sdisval borealval; do
    case $set in sdisval) Y=data/yolo/yamls/local/expB.yaml;; borealval) Y=data/v4/eval/yamls/boreal_val.yaml;; esac
    python -m src.eval.eval_detection --data-yaml $Y --split val --preds-jsonl models/v6/evals/$TAG/${n}__${set}.jsonl --out reports/v6/val/eval_${n}_${set}.json > logs/v6/val_${n}_${set}.log 2>&1 || echo "EVAL FAIL $n $set"
  done
done
echo "VAL EVAL DONE $TAG"
