#!/bin/bash
# One cluster job per run: FIgLib-39 frame scoring (shipped frames tar) +
# batch-1 latency (n=50, sdis_val first 50, GPU 0) unmerged and LoRA-merged.
#   bash scripts/v4/figlib_latency_job.sh <run> <backbone-dir>
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
RUN="${1:?}"; BB="${2:?}"
CKPT="/scratch/scratch/tianrgua-sandbox/smoke/runs_v3/${RUN}/best.pt"
EV="${RUN}_figlib"
RES="res-fg$(echo "$RUN" | tr -d '_' | tr '[:upper:]' '[:lower:]')"
python cluster/v4/registry.py reserve --job "$RES" --nodes 1 --track dino --note "figlib39+latency $RUN" || exit 3
python cluster/v3/submit_dino_eval.py --eval-name "$EV" \
  --backbone-dir "$BB" --ckpt "$CKPT" \
  --extra-list "$(pwd)/data/v3/figlib39_frames.txt=figlib39" \
  --frames-tar data/v4/stage/figlib39_frames.tgz \
  --rootmap-override "/local/home/tianrgua/cvproject/data/raw/figlib=/tmp/frames" \
  --infer-extra "--lora 16" \
  --latency-n 50 --latency-list sdis_val \
  --latency-variant "merged=--lora 16 --lora-merge" \
  --width 1280 --height 736 --conf-floor 0.02
RC=$?
python cluster/v4/registry.py release --job "$RES" --outcome "figlib_rc=$RC"
D="models/v4/evals/${EV}"; mkdir -p "$D"
SRC="models/v3/evals/${EV}"
[ -f "$SRC/figlib39_preds.jsonl.gz" ] && gunzip -cf "$SRC/figlib39_preds.jsonl.gz" > "$D/figlib39_preds.jsonl"
cp -f "$SRC"/latency_gpu*.json "$SRC"/shardlogs.txt "$D/" 2>/dev/null
echo "figlib rows: $(wc -l < "$D/figlib39_preds.jsonl" 2>/dev/null || echo 0)"
for f in "$D"/latency_gpu*.json; do echo "== $f"; cat "$f"; echo; done
echo "JOB DONE $RUN rc=$RC"
