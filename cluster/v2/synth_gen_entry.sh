#!/bin/bash
# Synthetic-smoke generation entrypoint (green g6e, smoke-diffusion:v1).
# 8 single-GPU workers of gen_synth_pod.py over a spec slice; results tar
# uploaded via presigned PUT. Pods: no internet, no logs -> exit codes +
# uploaded status.json/log tail are the only signals.
set -u +e
# uid-1001 fixes: no passwd entry in the image -> pin HOME + cache dirs
export HOME=/tmp USER=worker LOGNAME=worker
export TORCHINDUCTOR_CACHE_DIR=/tmp/ic TRITON_CACHE_DIR=/tmp/tc HF_HOME=/tmp/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/ic /tmp/tc /tmp/hf /tmp/job /tmp/out
curl -sf --max-time 120 -o /tmp/job/gen_synth_pod.py "$URL_CODE" || exit 101
curl -sf --max-time 120 -o /tmp/job/spec.jsonl "$URL_SPEC" || exit 101
curl -sf --max-time 300 -o /tmp/job/bg.tar "$URL_BG" || exit 102
mkdir -p /tmp/job/bgroot && tar xf /tmp/job/bg.tar -C /tmp/job/bgroot || exit 102
DONE_ARG=""
if [ -n "${URL_DONE:-}" ]; then
  curl -sf --max-time 120 -o /tmp/job/done.txt "$URL_DONE" || exit 101
  DONE_ARG="--done-file /tmp/job/done.txt"
fi
PIDS=()
for i in 0 1 2 3 4 5 6 7; do
  python /tmp/job/gen_synth_pod.py --spec /tmp/job/spec.jsonl \
    --backgrounds /tmp/job/bgroot --out /tmp/out --model-path "$MODEL_DIR" \
    --shard $i/8 --device-index $i --jobs-from "${JOBS_FROM:-0}" \
    --jobs-to "${JOBS_TO:-100000}" --budget-s "${BUDGET_S:-2100}" $DONE_ARG \
    > /tmp/out/worker$i.log 2>&1 &
  PIDS+=($!)
done
RC=0
for p in "${PIDS[@]}"; do wait $p || RC=1; done
NPNG=$(ls /tmp/out/*.png 2>/dev/null | wc -l)
NSMOKE=$(ls /tmp/out/*_smoke.png 2>/dev/null | wc -l)
NCTRL=$(ls /tmp/out/*_ctrl.png 2>/dev/null | wc -l)
NCONF=$(ls /tmp/out/*_conf.png 2>/dev/null | wc -l)
python - <<PY
import json, glob
ws = [json.load(open(p)) for p in sorted(glob.glob("/tmp/out/worker*_done.json"))]
gt = [t for w in ws for t in w.get("timings", [])]
s = dict(n_png=$NPNG, n_smoke=$NSMOKE, n_ctrl=$NCTRL, n_conf=$NCONF,
         workers=len(ws), assigned=sum(w["assigned"] for w in ws),
         done=sum(w["done"] for w in ws),
         mean_load_s=round(sum(w["load_s"] for w in ws)/max(len(ws),1),1),
         mean_gen_s=round(sum(gt)/max(len(gt),1),1),
         node_imgs_per_min=round(len(gt)/max(sum(gt)/8/60,1e-9),2) if gt else 0,
         workers_rc=$RC)
open("/tmp/out/status.json","w").write(json.dumps(s))
print(json.dumps(s))
PY
tar cf /tmp/results.tar -C /tmp/out .
[ "$NPNG" -gt 0 ] || exit 103
curl -sf --retry 3 --max-time 900 -T /tmp/results.tar "$URL_RESULTS" || exit 104
curl -sf --retry 3 --max-time 60 -T /tmp/out/status.json "$URL_STATUS" || exit 104
tail -c 6000 /tmp/out/worker0.log > /tmp/out/log0.txt
curl -sf --max-time 60 -T /tmp/out/log0.txt "$URL_LOG"
exit 0
