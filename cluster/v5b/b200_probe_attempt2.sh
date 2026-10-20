#!/bin/bash
# V5b B200 probe ATTEMPT 2 -- run INSIDE the already-running probe pod via `kraken jobs connect`
# (as the connecting corp uid). Same train/resume/reload sequence as b200_probe_entry.sh plus the
# fix for attempt 1's stall: obsidian-pu pods run as a non-root uid, so ultralytics' config dir is
# not the baked /root/.config/Ultralytics -> Arial.ttf missing -> check_det_dataset()->check_font()
# tries a GitHub download with no internet and blocks for minutes. We stage a TTF into YOLO_CONFIG_DIR.
set -u +e
M=/tmp/m; mkdir -p $M/Ultralytics/Ultralytics   # ultralytics resolves USER_CONFIG_DIR = $YOLO_CONFIG_DIR/Ultralytics
export HOME=$M YOLO_CONFIG_DIR=$M/Ultralytics PYTHONUNBUFFERED=1 YOLO_OFFLINE=true
LOG=$M/attempt2.log; exec > >(tee -a $LOG) 2>&1
T0=$(date +%s); BITS=0
st() { echo "{\"phase\":\"$1\",\"bits\":$BITS,\"t\":$(( $(date +%s)-T0 )),\"ts\":\"$(date -u +%FT%TZ)\"}" > $M/status; echo "== [$(( $(date +%s)-T0 ))s] $1 bits=$BITS"; }
st font
echo "uid=$(id -u) home=$HOME cwd=$(pwd)"; ls -la /root/.config/Ultralytics/ 2>&1 | head -3
F=""; for c in /root/.config/Ultralytics/Arial.ttf $(python -c "import matplotlib,os; print(os.path.join(os.path.dirname(matplotlib.__file__),'mpl-data/fonts/ttf/DejaVuSans.ttf'))" 2>/dev/null) $(find /opt/conda -name "*.ttf" 2>/dev/null | head -1); do [ -r "$c" ] && { F=$c; break; }; done
echo "font source: ${F:-NONE}"; [ -n "$F" ] && cp "$F" $M/Ultralytics/Ultralytics/Arial.ttf; ls -la $M/Ultralytics/Ultralytics/
st train_run1
cd /opt/weights; RUN1=$M/run1; mkdir -p $RUN1
setsid python /tmp/train_yolo.py --data /tmp/toy.yaml --model /opt/weights/yolo11m.pt --epochs 3 --imgsz 640 --batch 64 --device 0,1,2,3,4,5,6,7 --project $M --name run1 --workers 8 --patience 100 > $M/run1_console.log 2>&1 &
PG=$!; TSTART=$(date +%s); KILLED=0
while kill -0 $PG 2>/dev/null; do
  if [ -f $RUN1/results.csv ]; then
    rows=$(( $(wc -l < $RUN1/results.csv) - 1 ))
    if [ $rows -ge 2 ] && [ -f $RUN1/weights/last.pt ] && [ $RUN1/weights/last.pt -nt $RUN1/results.csv ]; then
      s1=$(stat -c%s $RUN1/weights/last.pt); sleep 3; s2=$(stat -c%s $RUN1/weights/last.pt)
      if [ "$s1" = "$s2" ] && [ "$s1" -gt 1000000 ]; then echo "epoch 2 checkpoint saved (last.pt $s1 B) -> kill process group $PG (simulated preemption)"; kill -TERM -- -$PG; sleep 5; kill -KILL -- -$PG 2>/dev/null; KILLED=1; break; fi
    fi
  fi
  [ $(( $(date +%s) - TSTART )) -gt 900 ] && { echo "run1 TIMEOUT"; kill -KILL -- -$PG; break; }
  sleep 2
done
wait $PG 2>/dev/null; echo "run1 rc=$? killed=$KILLED wall=$(( $(date +%s) - TSTART ))s"
pkill -u $(id -u) -9 -f "_temp_" 2>/dev/null; sleep 3
echo "gpu mem after kill: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
echo "--- run1 results.csv"; cat $RUN1/results.csv 2>/dev/null
rows=$(( $(wc -l < $RUN1/results.csv 2>/dev/null || echo 1) - 1 )); [ $rows -ge 2 ] && BITS=$((BITS|8))
echo "run1 it/s samples: $(tr '\r' '\n' < $M/run1_console.log | grep -o '[0-9.]*it/s' | tail -8 | tr '\n' ' ')"
grep -i "error\|Traceback" $M/run1_console.log | head -5
st resume
python -c "import torch; ck=torch.load('$RUN1/weights/last.pt',map_location='cpu',weights_only=False); print('resume ckpt epoch idx',ck.get('epoch'),'| optimizer present',ck.get('optimizer') is not None,'| train_args.epochs',ck['train_args'].get('epochs'))"
T2=$(date +%s)
timeout --signal=INT --kill-after=120 900 python /tmp/train_yolo.py --data /tmp/toy.yaml --model $RUN1/weights/last.pt --resume --epochs 3 --imgsz 640 --batch 64 --device 0,1,2,3,4,5,6,7 --project $M --name run1 --workers 8 --patience 100 > $M/run2_console.log 2>&1
echo "run2 (resume) rc=$? wall=$(( $(date +%s) - T2 ))s"
echo "--- results.csv after resume"; cat $RUN1/results.csv 2>/dev/null
grep -q TRAIN_DONE $M/run2_console.log && echo "TRAIN_DONE seen"; tr '\r' '\n' < $M/run2_console.log | grep -o "Resuming training.*" | head -1
rows=$(( $(wc -l < $RUN1/results.csv 2>/dev/null || echo 1) - 1 )); [ $rows -ge 3 ] && BITS=$((BITS|16))
echo "run2 it/s samples: $(tr '\r' '\n' < $M/run2_console.log | grep -o '[0-9.]*it/s' | tail -8 | tr '\n' ' ')"
grep -i "error\|Traceback" $M/run2_console.log | head -5
st reload
python - > $M/reload.txt 2>&1 <<PY
import torch, time
from ultralytics import YOLO
p = "$RUN1/weights/last.pt"
ck = torch.load(p, map_location="cpu", weights_only=False)
print("final ckpt epoch:", ck.get("epoch"), "| optimizer stripped:", ck.get("optimizer") is None)
m = YOLO(p); img = open("/tmp/toy_val.txt").readline().strip()
t = time.time(); r = m.predict([img], imgsz=640, device=0, verbose=False); dt = time.time() - t
print("predict ok, boxes:", tuple(r[0].boxes.xyxy.shape), "dt=%.3fs" % dt); print("RELOAD_OK")
PY
tail -3 $M/reload.txt; grep -q RELOAD_OK $M/reload.txt && BITS=$((BITS|32))
ls -la $RUN1/weights; sha256sum $RUN1/weights/*.pt > $M/weights_sha256.txt; cat $M/weights_sha256.txt
echo "ATTEMPT2_BITS=$BITS"; echo $BITS > $M/bits
st done
