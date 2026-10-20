#!/bin/bash
# V5b: local offline sweep of the B200 high-res DINOv3 run's sdis_val predictions
# (computed pod-side by cluster/v3/dino_infer.py, pulled via kraken connect) and the
# matched-FP comparison against v4 B4 (1280x736) — identical matching code
# (src/eval/eval_detection, IoU>=0.4), GT from the frozen local YOLO labels.
#   bash scripts/v5b/eval_b200_dino.sh v5bB200_lora_L_1920
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN="${1:?run name}"
D="models/v5b/b200/$RUN"
P="$D/sdis_val_preds.jsonl"
[ -f "$P" ] || { gunzip -kf "$P.gz"; }
[ -s "$P" ] || { echo "EMPTY/MISSING preds $P"; exit 2; }
echo "preds rows: $(wc -l < "$P") (expect 7335)"
mkdir -p reports/v5b logs/v5b
.venv/bin/python -m src.eval.eval_detection --data-yaml data/yolo/yamls/local/expB.yaml --split val \
  --preds-jsonl "$P" --out "reports/v5b/eval_${RUN}_sdis_val.json" > "logs/v5b/localsweep_${RUN}_sdis_val.log" 2>&1 \
  && echo "swept -> reports/v5b/eval_${RUN}_sdis_val.json" || { echo "SWEEP FAILED"; tail -5 "logs/v5b/localsweep_${RUN}_sdis_val.log"; exit 3; }
.venv/bin/python scripts/v4/compare_matched_fp.py --top-fine \
  --model "${RUN}=reports/v5b/eval_${RUN}_sdis_val.raw.parquet" \
  --model "v4B4_lora_L_1280=reports/v4/eval_v4B4_lora_L_sdis_val.raw.parquet" \
  --model "v4B6_lora_Hplus_1280=reports/v4/eval_v4B6_lora_Hplus_sdis_val.raw.parquet" \
  --model "v4B1_frozen_L_1280=reports/v4/eval_v4B1_frozen_L_sdis_val.raw.parquet" \
  --out "reports/v5b/matched_fp_b200_${RUN}_sdis_val.json"
echo "matched-FP -> reports/v5b/matched_fp_b200_${RUN}_sdis_val.json"
