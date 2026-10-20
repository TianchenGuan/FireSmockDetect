#!/bin/bash
# ONE-SHOT final test evaluation for the pre-registered YOLO systems
# (reports/v4/PREREGISTRATION_V4.md). Run exactly once at iteration close.
# Usage: bash cluster/v4/run_final_test_eval.sh <ckpt_spec>...   e.g.
#   bash cluster/v4/run_final_test_eval.sh S1_A=models/trained/expV4A_ctrl_s0/seg05/best.pt \
#        S2_E=models/v4/runs/expV4E_et_s1/seg05/best.pt ...
# Produces preds under models/v4/evals/v4finaltest/ then local eval JSONs
# reports/v4/test/eval_<sys>_<set>.json. Refuses to run twice.
set -euo pipefail
cd /local/home/tianrgua/cvproject
source .venv/bin/activate
[ -e reports/v4/test/.EVALUATED_ONCE ] && { echo "REFUSING: final test eval already ran"; exit 2; }
mkdir -p reports/v4/test
CK=(); for s in "$@"; do CK+=(--ckpt "$s"); done
python cluster/v4/registry.py reserve --job res-v4finaltest --nodes 1 --track eval --note "ONE-SHOT final test eval" || exit 3
python cluster/v4/submit_yolo_eval.py --tag v4finaltest "${CK[@]}" \
  --list sdistest=/mnt/local/smoke_data/lists/pod/sdis_test.txt \
  --list borealtest=/scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evaltest/boreal_test/lists/pod/test.txt \
  --list foresttest=/scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evaltest/forest_test/lists/pod/test.txt \
  --data-tar v4_evaltest=data/v4/stage/v4_evaltest.tar \
  --rootmap /mnt/local/smoke_data/pyro_sdis=/local/home/tianrgua/cvproject/data/yolo/pyro_sdis \
  --rootmap /scratch/scratch/tianrgua-sandbox/smoke/v4data/v4_evaltest=/local/home/tianrgua/cvproject/data/v4/eval
RC=$?
python cluster/v4/registry.py release --job res-v4finaltest --outcome "RC$RC"
[ $RC -ne 0 ] && { echo "preds incomplete — inspect models/v4/evals/v4finaltest/entry.log"; exit 4; }
date -u +%FT%TZ > reports/v4/test/.EVALUATED_ONCE
for s in "$@"; do name=${s%%=*}
  for set in sdistest borealtest foresttest; do
    case $set in sdistest) Y=data/yolo/yamls/local/expB.yaml; SP=test;; borealtest) Y=data/v4/eval/yamls/boreal_test.yaml; SP=test;; foresttest) Y=data/v4/eval/yamls/forest_test.yaml; SP=test;; esac
    python -m src.eval.eval_detection --data-yaml $Y --split $SP \
      --preds-jsonl models/v4/evals/v4finaltest/${name}__${set}.jsonl \
      --out reports/v4/test/eval_${name}_${set}.json > logs/v4_finaltest_${name}_${set}.log 2>&1 || echo "EVAL FAIL $name $set"
  done
done
echo "FINAL TEST EVAL DONE $(cat reports/v4/test/.EVALUATED_ONCE)"
