#!/bin/bash
# Tiny latency-only job (v3 protocol: n=50 sdis_val, batch 1, 1x L40S) for a
# LoRA run: unmerged (train form) + merged (deploy form). Default root map.
#   bash scripts/v4/latency_job.sh <run> <backbone-dir>
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
RUN="${1:?}"; BB="${2:?}"
CKPT="/scratch/scratch/tianrgua-sandbox/smoke/runs_v3/${RUN}/best.pt"
EV="${RUN}_lat"
RES="res-lt$(echo "$RUN" | tr -d '_' | tr '[:upper:]' '[:lower:]')"
python cluster/v4/registry.py reserve --job "$RES" --nodes 1 --track dino --note "latency $RUN" || exit 3
python cluster/v3/submit_dino_eval.py --eval-name "$EV" \
  --backbone-dir "$BB" --ckpt "$CKPT" \
  --splits sdis_val --max-images 8 \
  --infer-extra "--lora 16" \
  --latency-n 50 \
  --latency-variant "merged=--lora 16 --lora-merge" \
  --width 1280 --height 736 --conf-floor 0.02
RC=$?
python cluster/v4/registry.py release --job "$RES" --outcome "latency_rc=$RC"
D="models/v4/evals/${RUN}_figlib"; mkdir -p "$D"
cp -f models/v3/evals/${EV}/latency_gpu*.json "$D/" 2>/dev/null
for f in "$D"/latency_gpu*.json; do echo "== $f"; cat "$f"; echo; done
echo "LATENCY JOB DONE $RUN rc=$RC"
