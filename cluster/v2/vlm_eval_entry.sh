#!/bin/bash
# VLM eval entrypoint: 8 single-GPU shards of eval_vlm.py over an image list,
# optional LoRA adapter fetched from S3, merged results via presigned PUT.
set -u +e
export HOME=/tmp USER=worker LOGNAME=worker
export TORCHINDUCTOR_CACHE_DIR=/tmp/ic TRITON_CACHE_DIR=/tmp/tc HF_HOME=/tmp/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PERSIST=/scratch/scratch/tianrgua-sandbox/smoke/evals_v2/${EVAL_NAME:-unnamed}
mkdir -p /tmp/ic /tmp/tc /tmp/hf /tmp/job /tmp/out "$PERSIST"
curl -sf --max-time 120 -o /tmp/job/eval_vlm.py "$URL_CODE" || exit 101
curl -sf --max-time 120 -o /tmp/job/list.txt "$URL_LIST" || exit 101
ADAPTER_ARG=""
if [ -n "${URL_ADAPTER_GET:-}" ]; then
  curl -sf --max-time 600 -o /tmp/job/adapter.tgz "$URL_ADAPTER_GET" || exit 102
  mkdir -p /tmp/job/adapter && tar xzf /tmp/job/adapter.tgz -C /tmp/job/adapter
  AD=$(find /tmp/job/adapter -name adapter_config.json | head -1 | xargs dirname)
  ADAPTER_ARG="--adapter $AD"
fi
PIDS=()
for i in 0 1 2 3 4 5 6 7; do
  python /tmp/job/eval_vlm.py --model-path "$MODEL_DIR" $ADAPTER_ARG \
    ${QUANT:+--quant $QUANT} --max-new-tokens "${MAX_NEW_TOKENS:-192}" \
    --list /tmp/job/list.txt --shard $i/8 --device-index $i \
    --root-map "/local/home/tianrgua/cvproject/data/yolo=/scratch/scratch/tianrgua-sandbox/smoke_data_v2" \
    --max-pixels "${MAX_PIXELS:-1048576}" ${MAX_IMAGES:+--max-images $MAX_IMAGES} \
    --out /tmp/out/shard$i.jsonl > /tmp/out/shard$i.log 2>&1 &
  PIDS+=($!)
done
RC=0
for p in "${PIDS[@]}"; do wait $p || RC=1; done
cat /tmp/out/shard*.jsonl > /tmp/out/all.jsonl
python - <<'PY'
import json, glob
tot = {"n_images":0,"n_pos":0,"n_neg":0,"tiny_n":0}
acc = {"tp_img":0,"fp_img":0,"box_tp":0,"box_fp":0,"box_fn":0,"tiny_tp":0,"lat":[],"parse_fail":0}
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0
for line in open("/tmp/out/all.jsonl"):
    r=json.loads(line); gt=r["gt_boxes"]; pr=r["pred"]
    tot["n_images"]+=1
    has=len(gt)>0; tot["n_pos"]+=has; tot["n_neg"]+= (not has)
    pos=bool(pr.get("smoke_present")) or bool(pr.get("boxes"))
    acc["tp_img"]+= has and pos; acc["fp_img"]+= (not has) and pos
    acc["parse_fail"]+= not pr.get("parse_ok",False)
    m=set()
    for pb in pr.get("boxes",[]):
        bj,bv=-1,0.0
        for j,g in enumerate(gt):
            if j in m: continue
            v=iou(pb,g[:4])
            if v>bv: bj,bv=j,v
        if bv>=0.4: m.add(bj); acc["box_tp"]+=1
        else: acc["box_fp"]+=1
    acc["box_fn"]+=len(gt)-len(m)
    for j,g in enumerate(gt):
        if g[4]<0.001:
            tot["tiny_n"]+=1
            acc["tiny_tp"]+= j in m
    acc["lat"].append(r["latency_s"])
s={**tot,
 "img_recall":round(acc["tp_img"]/max(tot["n_pos"],1),4),
 "img_fp_rate_neg":round(acc["fp_img"]/max(tot["n_neg"],1),4),
 "box_precision":round(acc["box_tp"]/max(acc["box_tp"]+acc["box_fp"],1),4),
 "box_recall":round(acc["box_tp"]/max(acc["box_tp"]+acc["box_fn"],1),4),
 "tiny_recall":round(acc["tiny_tp"]/max(tot["tiny_n"],1),4),
 "parse_fail":acc["parse_fail"],
 "mean_latency_s":round(sum(acc["lat"])/max(len(acc["lat"]),1),3)}
open("/tmp/out/summary.json","w").write(json.dumps(s,indent=2))
print(json.dumps(s))
PY
tail -c 6000 /tmp/out/shard0.log > /tmp/out/log0.txt
cp /tmp/out/all.jsonl /tmp/out/summary.json /tmp/out/log0.txt "$PERSIST"/ 2>/dev/null
# presigned PUTs may have expired if load+inference exceeded the URL window;
# results persist on /scratch either way (fetch job retrieves them).
curl -sf --max-time 300 -T /tmp/out/all.jsonl "$URL_PREDS"
curl -sf --max-time 60 -T /tmp/out/summary.json "$URL_SUMMARY"
curl -sf -T /tmp/out/log0.txt "$URL_LOG"
[ -f "$PERSIST/summary.json" ] && exit 0
exit $RC
