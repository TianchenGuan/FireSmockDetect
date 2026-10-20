#!/bin/bash
# ONE-SHOT V5b final evaluation (PREREGISTRATION_V5B.md): H1 fireman_holdout_s10,
# H2 pyro-sdis TEST, H3 sheffield_holdout (+ historical boreal_test/forest_test).
# Usage: bash cluster/v5b/run_final_test_eval.sh <batch_tag> <sys>=<ckpt> ...  Each system is evaluated once (guard file).
set -euo pipefail
cd /local/home/tianrgua/cvproject; source .venv/bin/activate
TAG="$1"; shift   # batch tag; systems are guarded individually (each evaluated once)
mkdir -p reports/v5b/test; touch reports/v5b/test/.EVALUATED_SYSTEMS
for s in "$@"; do n=${s%%=*}; grep -qx "$n" reports/v5b/test/.EVALUATED_SYSTEMS && { echo "REFUSING: $n already evaluated once"; exit 2; }; done
CK=(); for s in "$@"; do CK+=(--ckpt "$s"); done
export SMOKE_REGISTRY_DIR=cluster/v5b
python cluster/v4/registry.py reserve --job "res-$TAG" --nodes 1 --gpu-type L40S --track eval --note "ONE-SHOT v5b final $TAG" || exit 3
python cluster/v4/submit_yolo_eval.py --tag "$TAG" "${CK[@]}" \
  --list sdistest=/mnt/local/smoke_data/lists/pod/sdis_test.txt \
  --list firemanholdout=/scratch/scratch/tianrgua-sandbox/smoke/v5bdata/eval_v5b/fireman_holdout_s10/lists/pod/fireman_holdout_s10.txt \
  --list sheffieldholdout=/scratch/scratch/tianrgua-sandbox/smoke/v5bdata/eval_v5b/sheffield_holdout/lists/pod/sheffield_holdout.txt \
  --list borealtest=/scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evaltest/boreal_test/lists/pod/test.txt \
  --list foresttest=/scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evaltest/forest_test/lists/pod/test.txt \
  --data-tar eval_v5b@/scratch/scratch/tianrgua-sandbox/smoke/v5bdata=data/v5b/stage/eval_v5b.tar \
  --data-tar v4_evaltest=data/v4/stage/v4_evaltest.tar \
  --rootmap /mnt/local/smoke_data/pyro_sdis=/local/home/tianrgua/cvproject/data/yolo/pyro_sdis \
  --rootmap /scratch/scratch/tianrgua-sandbox/smoke/v5bdata/eval_v5b=/local/home/tianrgua/cvproject/data/v5b/eval \
  --rootmap /scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evaltest=/local/home/tianrgua/cvproject/data/v4/eval \
  --out models/v5b/evals; RC=$?
python cluster/v4/registry.py release --job "res-$TAG" --outcome "RC$RC"
[ $RC -ne 0 ] && { echo "preds incomplete — see models/v5b/evals/$TAG/entry.log"; exit 4; }
for s in "$@"; do echo "${s%%=*}" >> reports/v5b/test/.EVALUATED_SYSTEMS; done; date -u +%FT%TZ >> reports/v5b/test/.EVALUATED_ONCE
for s in "$@"; do n=${s%%=*}
  for set in sdistest firemanholdout sheffieldholdout borealtest foresttest; do
    case $set in sdistest) Y=data/yolo/yamls/local/expB.yaml; SP=test;; firemanholdout) Y=data/v5b/eval/yamls/fireman_holdout_s10.yaml; SP=test;; sheffieldholdout) Y=data/v5b/eval/yamls/sheffield_holdout.yaml; SP=test;; borealtest) Y=data/v4/eval/yamls/boreal_test.yaml; SP=test;; foresttest) Y=data/v4/eval/yamls/forest_test.yaml; SP=test;; esac
    python -m src.eval.eval_detection --data-yaml $Y --split $SP --preds-jsonl models/v5b/evals/$TAG/${n}__${set}.jsonl --out reports/v5b/test/eval_${n}_${set}.json > logs/v5b/final_${n}_${set}.log 2>&1 || echo "EVAL FAIL $n $set"
  done
done
echo "V5B FINAL TEST EVAL DONE batch $TAG $(date -u +%FT%TZ)"
