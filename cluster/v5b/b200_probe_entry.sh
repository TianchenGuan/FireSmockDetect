#!/bin/bash
# cluster/v5b/b200_probe_entry.sh -- V5b B200 end-to-end probe (obsidian-pu, ap-south-1).
# Delivered inline (base64 in env ENTRY_B64). Pods have no log channel: self-log to
# /tmp/probe.log, mirror to /scratch sandbox + presigned PUT (if egress works), hold the
# pod RETAIN_S seconds at the end so files can be pulled with `kraken jobs connect -c`
# (touch /tmp/ack to end early). Exit code = 100 + bitmask:
#   1 = 8 GPUs visible AND torch capability sm_100 (Blackwell)
#   2 = /opt/gpu_test.py conv fwd+bwd + bf16 matmul PASS
#   4 = presigned PUT to the ap-south-1 bucket returned HTTP 200
#   8 = toy 8-GPU DDP training (yolo11m, imgsz 640) reached epoch 2 (results.csv)
#  16 = --resume from the killed run completed epoch 3
#  32 = final last.pt reload + predict OK
#  64 = presigned PUT to the us-east-1 bucket (regional endpoint) returned HTTP 200
set -u +e
export HOME=/tmp USER=worker
# obsidian-pu pods run as a non-root uid: the baked /root/.config/Ultralytics (Arial.ttf) is unreadable and
# ultralytics check_font() -> is_url(github) hangs forever (VPC accepts TCP, blackholes HTTPS). Stage a TTF at
# the RESOLVED config dir ($YOLO_CONFIG_DIR/Ultralytics) and force offline mode.
export YOLO_CONFIG_DIR=/tmp/yolo_cfg YOLO_OFFLINE=true; mkdir -p /tmp/yolo_cfg/Ultralytics
F=$(python -c "import matplotlib,os; print(os.path.join(os.path.dirname(matplotlib.__file__),'mpl-data/fonts/ttf/DejaVuSans.ttf'))" 2>/dev/null)
[ -r "$F" ] && cp "$F" /tmp/yolo_cfg/Ultralytics/Arial.ttf
LOG=/tmp/probe.log
exec > >(tee -a "$LOG") 2>&1
OUT=/tmp/probe_out; mkdir -p $OUT
BITS=0; T0=$(date +%s)
st() { echo "{\"phase\":\"$1\",\"bits\":$BITS,\"t\":$(( $(date +%s)-T0 )),\"ts\":\"$(date -u +%FT%TZ)\"}" > /tmp/probe_status; echo "== [$(( $(date +%s)-T0 ))s] $1 bits=$BITS"; }
st start
# restart guard (kraken forces maxRestarts>=1): a restarted container finds the marker and exits at once
for M in /scratch/scratch/tianrgua-sandbox/smoke/v5b_b200probe/exit_${RUN_ID:-default} /mnt/local/v5b_b200probe/exit_${RUN_ID:-default}; do
  if [ -f "$M" ]; then echo "RESTART DETECTED: previous exit code $(cat $M) at $M -> exiting immediately"; exit $(cat $M); fi
done
echo "host=$(hostname) uid=$(id -u) nproc=$(nproc)"; free -g | head -2; df -h /tmp /mnt/local /scratch 2>&1 | head -6

st hardware
nvidia-smi > $OUT/nvidia_smi.txt 2>&1; head -16 $OUT/nvidia_smi.txt
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv > $OUT/gpus.csv 2>&1; cat $OUT/gpus.csv
NG=$(nvidia-smi -L 2>/dev/null | wc -l)
python - > $OUT/torch_env.json 2>$OUT/torch_env.err <<'PY'
import json, torch
d = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
     "n_gpu": torch.cuda.device_count(), "arch_list": torch.cuda.get_arch_list()}
if d["n_gpu"]:
    d["names"] = sorted(set(torch.cuda.get_device_name(i) for i in range(d["n_gpu"])))
    d["cap0"] = list(torch.cuda.get_device_capability(0))
    d["mem_total_gib"] = [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1) for i in range(d["n_gpu"])]
try:
    import ultralytics; d["ultralytics"] = ultralytics.__version__
except Exception as e: d["ultralytics"] = str(e)
print(json.dumps(d))
PY
cat $OUT/torch_env.json; cat $OUT/torch_env.err | tail -3
grep -q '"cap0": \[10, 0\]' $OUT/torch_env.json && [ "$NG" -eq 8 ] && BITS=$((BITS|1))

st gpu_test
python /opt/gpu_test.py > $OUT/gpu_test.txt 2>&1; tail -6 $OUT/gpu_test.txt
grep -q GPU_TEST_PASS $OUT/gpu_test.txt && BITS=$((BITS|2))

st egress
for h in s3.ap-south-1.amazonaws.com ${BUCKET_APS1}.s3.ap-south-1.amazonaws.com s3.us-east-1.amazonaws.com ${BUCKET_USE1}.s3.us-east-1.amazonaws.com sts.ap-south-1.amazonaws.com 000000000001.dkr.ecr.ap-south-1.amazonaws.com 000000000001.dkr.ecr.us-east-1.amazonaws.com pypi.org huggingface.co; do
  r=$(timeout 10 getent hosts $h | head -1 | awk '{print $1}'); echo "dns $h -> ${r:-NXDOMAIN/timeout}"; done | tee $OUT/dns.txt
echo "hello from $(hostname) $(date -u +%FT%TZ)" > /tmp/hello.txt
C_APS1=$(curl -s -o /tmp/put_aps1.resp -w '%{http_code}' --max-time 60 -T /tmp/hello.txt "$URL_PUT_HELLO_APS1"); echo "presigned PUT aps1: http=$C_APS1 $(tr -d '\n' < /tmp/put_aps1.resp | cut -c1-200)"
[ "$C_APS1" = "200" ] && BITS=$((BITS|4))
C_USE1=$(curl -s -o /tmp/put_use1.resp -w '%{http_code}' --max-time 60 -T /tmp/hello.txt "$URL_PUT_HELLO_USE1"); echo "presigned PUT use1: http=$C_USE1 $(tr -d '\n' < /tmp/put_use1.resp | cut -c1-200)"
[ "$C_USE1" = "200" ] && BITS=$((BITS|64))
G_APS1=$(curl -s -o /tmp/get_aps1.txt -w '%{http_code}' --max-time 60 "$URL_GET_APS1"); echo "presigned GET aps1: http=$G_APS1 $(tr -d '\n' < /tmp/get_aps1.txt | cut -c1-120)"
G_USE1=$(curl -s -o /tmp/get_use1.txt -w '%{http_code}' --max-time 60 "$URL_GET_USE1"); echo "presigned GET use1: http=$G_USE1 $(tr -d '\n' < /tmp/get_use1.txt | cut -c1-120)"
timeout 45 aws sts get-caller-identity --region ap-south-1 > $OUT/sts.txt 2>&1; echo "sts rc=$? $(tr -d '\n ' < $OUT/sts.txt | cut -c1-300)"
timeout 60 aws s3 cp /tmp/hello.txt s3://$BUCKET_APS1/probes/v5b/podid_hello.txt --region ap-south-1 > $OUT/s3cp_aps1.txt 2>&1; echo "pod-identity s3 cp aps1 rc=$? $(tr -d '\n' < $OUT/s3cp_aps1.txt | cut -c1-200)"
timeout 60 aws s3 cp /tmp/hello.txt s3://$BUCKET_USE1/probes/v5b/podid_hello.txt --region us-east-1 > $OUT/s3cp_use1.txt 2>&1; echo "pod-identity s3 cp use1 rc=$? $(tr -d '\n' < $OUT/s3cp_use1.txt | cut -c1-200)"
timeout 60 aws s3 ls s3://$BUCKET_APS1/ --region ap-south-1 > $OUT/s3ls_aps1.txt 2>&1; echo "pod-identity s3 ls aps1 rc=$? $(tr -d '\n' < $OUT/s3ls_aps1.txt | cut -c1-200)"
SCR=/scratch/scratch/tianrgua-sandbox/smoke/v5b_b200probe
if mkdir -p $SCR 2>/dev/null && touch $SCR/.w 2>/dev/null; then echo "scratch writable: $SCR"; else echo "scratch NOT writable"; SCR=""; fi
echo "scratch sandbox listing:"; ls /scratch/scratch/tianrgua-sandbox/ 2>&1 | head -8; ls /scratch/scratch/tianrgua-sandbox/models 2>&1 | head -5

st dataset
RUN_ROOT=""; for d in /mnt/local/v5b_b200probe /tmp/v5b_b200probe; do mkdir -p $d 2>/dev/null && touch $d/.w 2>/dev/null && { RUN_ROOT=$d; break; }; done; echo "run root: $RUN_ROOT"
N_TRAIN=${N_TRAIN:-2048}; N_VAL=${N_VAL:-128}
head -n $N_TRAIN /opt/smoke_data/lists/pod/expB_train.txt | sed 's#^/mnt/local/smoke_data/#/opt/smoke_data/#' > /tmp/toy_train.txt
head -n $N_VAL /opt/smoke_data/lists/pod/sdis_val.txt | sed 's#^/mnt/local/smoke_data/#/opt/smoke_data/#' > /tmp/toy_val.txt
printf 'path: /tmp\ntrain: /tmp/toy_train.txt\nval: /tmp/toy_val.txt\nnc: 1\nnames: [smoke]\n' > /tmp/toy.yaml
ok=0; while read f; do [ -f "$f" ] && ok=$((ok+1)); done < /tmp/toy_train.txt; echo "train imgs present: $ok/$N_TRAIN (pyro-sdis TRAIN sites; val = first $N_VAL of sdis VAL list)"
echo "$TRAIN_PY_B64" | base64 -d > /tmp/train_yolo.py; python -c "import ast; ast.parse(open('/tmp/train_yolo.py').read()); print('train_yolo.py (src/train/train_yolo.py, shipped inline) syntax ok')"

st train_run1
cd /opt/weights
RUN1=$RUN_ROOT/run1; mkdir -p $RUN1
setsid python /tmp/train_yolo.py --data /tmp/toy.yaml --model /opt/weights/${BASE_WEIGHTS:-yolo11m.pt} --epochs 3 --imgsz ${IMGSZ:-640} --batch ${BATCH:-64} --device 0,1,2,3,4,5,6,7 --project $RUN_ROOT --name run1 --workers 8 --patience 100 > $OUT/run1_console.log 2>&1 &
PG=$!; TSTART=$(date +%s); KILLED=0
while kill -0 $PG 2>/dev/null; do
  if [ -f $RUN1/results.csv ]; then
    rows=$(( $(wc -l < $RUN1/results.csv) - 1 ))
    if [ $rows -ge 2 ] && [ -f $RUN1/weights/last.pt ] && [ $RUN1/weights/last.pt -nt $RUN1/results.csv ]; then
      s1=$(stat -c%s $RUN1/weights/last.pt); sleep 3; s2=$(stat -c%s $RUN1/weights/last.pt)
      if [ "$s1" = "$s2" ] && [ "$s1" -gt 1000000 ]; then
        echo "epoch 2 checkpoint saved (last.pt $s1 bytes) -> killing run1 process group $PG (simulated preemption)"
        kill -TERM -- -$PG; sleep 5; kill -KILL -- -$PG 2>/dev/null; KILLED=1; break
      fi
    fi
  fi
  if [ $(( $(date +%s) - TSTART )) -gt ${TRAIN_TIMEOUT:-1200} ]; then echo "run1 TIMEOUT"; kill -KILL -- -$PG; break; fi
  sleep 2
done
wait $PG 2>/dev/null; echo "run1 rc=$? killed=$KILLED wall=$(( $(date +%s) - TSTART ))s"
pkill -9 -f "_temp_" 2>/dev/null; pkill -9 -f torch.distributed 2>/dev/null; sleep 3
echo "gpu mem after kill: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
echo "--- run1 results.csv"; cat $RUN1/results.csv 2>/dev/null
rows=$(( $(wc -l < $RUN1/results.csv 2>/dev/null || echo 1) - 1 )); [ $rows -ge 2 ] && BITS=$((BITS|8))
echo "run1 it/s samples: $(grep -o '[0-9.]*it/s' $OUT/run1_console.log | tail -6 | tr '\n' ' ') s/it: $(grep -o '[0-9.]*s/it' $OUT/run1_console.log | tail -3 | tr '\n' ' ')"
grep -i "error\|Traceback" $OUT/run1_console.log | head -5

st train_run2_resume
python - <<PY
import torch; ck = torch.load("$RUN1/weights/last.pt", map_location="cpu", weights_only=False)
print("resume ckpt: epoch idx", ck.get("epoch"), "| optimizer present:", ck.get("optimizer") is not None, "| train_args.epochs:", ck["train_args"].get("epochs"))
PY
T2=$(date +%s)
timeout --signal=INT --kill-after=120 ${TRAIN_TIMEOUT:-1200} python /tmp/train_yolo.py --data /tmp/toy.yaml --model $RUN1/weights/last.pt --resume --epochs 3 --imgsz ${IMGSZ:-640} --batch ${BATCH:-64} --device 0,1,2,3,4,5,6,7 --project $RUN_ROOT --name run1 --workers 8 --patience 100 > $OUT/run2_console.log 2>&1
echo "run2 (resume) rc=$? wall=$(( $(date +%s) - T2 ))s"
echo "--- results.csv after resume"; cat $RUN1/results.csv 2>/dev/null
grep -q TRAIN_DONE $OUT/run2_console.log && echo "TRAIN_DONE seen"
grep -o "Resuming training.*" $OUT/run2_console.log | head -2
rows=$(( $(wc -l < $RUN1/results.csv 2>/dev/null || echo 1) - 1 )); [ $rows -ge 3 ] && BITS=$((BITS|16))
echo "run2 it/s samples: $(grep -o '[0-9.]*it/s' $OUT/run2_console.log | tail -6 | tr '\n' ' ')"
grep -i "error\|Traceback" $OUT/run2_console.log | head -5

st reload
python - > $OUT/reload.txt 2>&1 <<PY
import torch, time
from ultralytics import YOLO
p = "$RUN1/weights/last.pt"
ck = torch.load(p, map_location="cpu", weights_only=False)
print("final ckpt epoch:", ck.get("epoch"), "| optimizer stripped:", ck.get("optimizer") is None)
m = YOLO(p)
img = open("/tmp/toy_val.txt").readline().strip()
t = time.time(); r = m.predict([img], imgsz=640, device=0, verbose=False); dt = time.time() - t
print("predict ok, boxes:", tuple(r[0].boxes.xyxy.shape), "dt=%.3fs" % dt)
print("RELOAD_OK")
PY
tail -4 $OUT/reload.txt; grep -q RELOAD_OK $OUT/reload.txt && BITS=$((BITS|32))

st upload
cp $RUN1/results.csv $RUN1/args.yaml $RUN1/env.json $OUT/ 2>/dev/null
ls -la $RUN1/weights > $OUT/weights_ls.txt 2>&1; sha256sum $RUN1/weights/*.pt > $OUT/weights_sha256.txt 2>&1; cat $OUT/weights_ls.txt
echo "FINAL_BITS=$BITS exit=$((100+BITS))"; echo $((100+BITS)) > /tmp/probe_exit
cp "$LOG" $OUT/probe.log
tar czf /tmp/probe_out.tgz -C /tmp probe_out; ls -la /tmp/probe_out.tgz
U1=$(curl -s -o /dev/null -w '%{http_code}' --max-time 120 -T /tmp/probe_out.tgz "$URL_PUT_TGZ_APS1"); echo "PUT probe_out.tgz aps1: http=$U1"
U2=$(curl -s -o /dev/null -w '%{http_code}' --max-time 120 -T /tmp/probe_out.tgz "$URL_PUT_TGZ_USE1"); echo "PUT probe_out.tgz use1: http=$U2"
U3=$(curl -s -o /dev/null -w '%{http_code}' --max-time 300 -T $RUN1/weights/last.pt "$URL_PUT_LAST_APS1"); echo "PUT last.pt aps1: http=$U3"
[ -n "$SCR" ] && cp /tmp/probe_out.tgz $RUN1/weights/last.pt $SCR/ 2>/dev/null && echo "mirrored to $SCR"

st retain
for i in $(seq 1 $(( ${RETAIN_S:-900} / 10 ))); do [ -f /tmp/ack ] && break; sleep 10; done
for M in /scratch/scratch/tianrgua-sandbox/smoke/v5b_b200probe /mnt/local/v5b_b200probe; do mkdir -p $M 2>/dev/null && echo $((100+BITS)) > $M/exit_${RUN_ID:-default} 2>/dev/null; done
st done
exit $((100+BITS))
