#!/bin/bash
# ONE-SHOT V6 final evaluation (PREREGISTRATION_V6.md): frozen v5b sets
# (sdis TEST H2, fireman_holdout_s10, sheffield_holdout, boreal_test,
# forest_test — HISTORICAL, used once in v5b, disclosed) + H-NEW FireSpot
# (eval-only, licence pending, v4-G-lineage caveat does not affect v6 arms).
# Usage: bash cluster/v6/run_final_test_eval.sh <batch_tag> <sys>=<ckpt> ...
# Each system is evaluated once (guard file).
set -euo pipefail
cd /local/home/tianrgua/cvproject; source .venv/bin/activate
TAG="$1"; shift
mkdir -p reports/v6/test logs/v6/repair; touch reports/v6/test/.EVALUATED_SYSTEMS
for s in "$@"; do n=${s%%=*}; grep -qx "${n}@${BATCH_KIND:-std}" reports/v6/test/.EVALUATED_SYSTEMS && { echo "REFUSING: $n already evaluated once"; exit 2; }; done
CK=(); for s in "$@"; do CK+=(--ckpt "$s"); done
export SMOKE_REGISTRY_DIR=cluster/v6
python cluster/v4/registry.py reserve --job "res-$TAG" --nodes 1 --gpu-type L40S --track eval --note "ONE-SHOT v6 final $TAG" || exit 3
python cluster/v4/submit_yolo_eval.py --tag "$TAG" "${CK[@]}" \
  --list sdistest=/mnt/local/smoke_data/lists/pod/sdis_test.txt \
  --list firemanholdout=/scratch/scratch/tianrgua-sandbox/smoke/v5bdata/eval_v5b/fireman_holdout_s10/lists/pod/fireman_holdout_s10.txt \
  --list sheffieldholdout=/scratch/scratch/tianrgua-sandbox/smoke/v5bdata/eval_v5b/sheffield_holdout/lists/pod/sheffield_holdout.txt \
  --list borealtest=/scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evaltest/boreal_test/lists/pod/test.txt \
  --list foresttest=/scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evaltest/forest_test/lists/pod/test.txt \
  ${FIRESPOT_LIST:-} \
  --data-tar eval_v5b@/scratch/scratch/tianrgua-sandbox/smoke/v5bdata=data/v5b/stage/eval_v5b.tar \
  --data-tar v4_evaltest=data/v4/stage/v4_evaltest.tar \
  ${FIRESPOT_TAR:-} \
  --rootmap /mnt/local/smoke_data/pyro_sdis=/local/home/tianrgua/cvproject/data/yolo/pyro_sdis \
  --rootmap /scratch/scratch/tianrgua-sandbox/smoke/v5bdata/eval_v5b=/local/home/tianrgua/cvproject/data/v5b/eval \
  --rootmap /scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evaltest=/local/home/tianrgua/cvproject/data/v4/eval \
  --rootmap /scratch/scratch/tianrgua-sandbox/smoke/v6data/firespot_eval=/local/home/tianrgua/cvproject/data/v6/eval/firespot_eval \
  --out models/v6/evals; RC=$?
python cluster/v4/registry.py release --job "res-$TAG" --outcome "RC$RC"
[ $RC -ne 0 ] && { echo "preds incomplete — see models/v6/evals/$TAG/entry.log"; exit 4; }
for s in "$@"; do echo "${s%%=*}@${BATCH_KIND:-std}" >> reports/v6/test/.EVALUATED_SYSTEMS; done; date -u +%FT%TZ >> reports/v6/test/.EVALUATED_ONCE
for s in "$@"; do n=${s%%=*}
  for set in ${EVAL_SETS:-sdistest firemanholdout sheffieldholdout borealtest foresttest}; do
    case $set in
      sdistest) Y=data/yolo/yamls/local/expB.yaml; SP=test;;
      firemanholdout) Y=data/v5b/eval/yamls/fireman_holdout_s10.yaml; SP=test;;
      sheffieldholdout) Y=data/v5b/eval/yamls/sheffield_holdout.yaml; SP=test;;
      borealtest) Y=data/v4/eval/yamls/boreal_test.yaml; SP=test;;
      foresttest) Y=data/v4/eval/yamls/forest_test.yaml; SP=test;;
      firespot) Y=data/v6/eval/yamls/firespot.yaml; SP=test;;
    esac
    python -m src.eval.eval_detection --data-yaml $Y --split $SP --preds-jsonl models/v6/evals/$TAG/${n}__${set}.jsonl --out reports/v6/test/eval_${n}_${set}.json > logs/v6/repair/final_${n}_${set}.log 2>&1 || echo "EVAL FAIL $n $set"
  done
done
echo "V6 FINAL TEST EVAL DONE batch $TAG $(date -u +%FT%TZ)"
