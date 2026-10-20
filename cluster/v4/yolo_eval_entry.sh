#!/bin/bash
# v4 YOLO batch val-inference entry. Fetches N checkpoints via presigned URLs
# (CKPT_URLS: name=url,name=url,...), runs ultralytics prediction over one or
# more image lists (LISTS: name=podlistpath,...), writes one preds jsonl per
# (ckpt, list) and uploads via presigned PUTs (URL_PREDS_<CKPT>_<LIST>).
# Optional DATA_TARS staged first (same pattern as train_entry_v4.sh).
# preds format matches src/eval/eval_detection.load_preds_jsonl:
#   {"image": <pod path>, "preds": [[x1n,y1n,x2n,y2n,score], ...]}
# exit 0 = all preds written+uploaded; 95 = staging failed; 97 = inference err.
set -u +e
export HOME=/tmp USER=worker
exec > /tmp/entry.log 2>&1          # pods have no log channel; self-log
put() { [ -f "$2" ] && curl -sf --max-time 300 -T "$2" "$1"; }
ship_log() { [ -n "${URL_LOG:-}" ] && tail -c 12000 /tmp/entry.log > /tmp/log.txt \
  && curl -sf --max-time 120 -T /tmp/log.txt "$URL_LOG"; }
trap ship_log EXIT

# baked pyro-sdis/D-Fire (image ships them at /opt/smoke_data); val lists and
# their image paths are under /mnt/local/smoke_data, so stage it like training.
[ -d /mnt/local/smoke_data ] || cp -r /opt/smoke_data /mnt/local/smoke_data 2>/dev/null \
  || { mkdir -p /tmp/sd && cp -r /opt/smoke_data /tmp/sd/smoke_data \
       && ln -sfn /tmp/sd/smoke_data /mnt/local/smoke_data; }
ls /mnt/local/smoke_data >/dev/null 2>&1 || { echo "FATAL: no smoke_data"; exit 95; }

V4_DATA_ROOT="${V4_DATA_ROOT:-/scratch/scratch/tianrgua-sandbox/smoke/v4data}"
if [ -n "${DATA_TARS:-}" ]; then
  mkdir -p "$V4_DATA_ROOT" || exit 95
  IFS=',' read -ra TARS <<< "$DATA_TARS"
  for spec in "${TARS[@]}"; do
    name="${spec%%=*}"; url="${spec#*=}"
    root="$V4_DATA_ROOT"
    case "$name" in *@*) root="${name#*@}"; name="${name%%@*}";; esac   # name@/abs/root
    mkdir -p "$root" || exit 95
    [ -f "$root/.${name}.done" ] && continue
    curl -sf --max-time 2400 -o "$root/${name}.tar" "$url" \
      && tar -xf "$root/${name}.tar" -C "$root" \
      && rm -f "$root/${name}.tar" \
      && touch "$root/.${name}.done" || exit 95
  done
fi

mkdir -p /tmp/ckpts /tmp/preds
IFS=',' read -ra CK <<< "$CKPT_URLS"
for spec in "${CK[@]}"; do
  name="${spec%%=*}"; url="${spec#*=}"
  curl -sf --max-time 600 -o "/tmp/ckpts/${name}.pt" "$url" || exit 95
done

python - <<'PYEOF'
import json, os, sys
from pathlib import Path
from ultralytics import YOLO

lists = dict(kv.split("=", 1) for kv in os.environ["LISTS"].split(","))
ckpts = sorted(Path("/tmp/ckpts").glob("*.pt"))
imgsz = int(os.environ.get("IMGSZ", "1280"))
for ck in ckpts:
    model = YOLO(str(ck))
    for lname, lpath in lists.items():
        imgs = [l.strip() for l in Path(lpath).read_text().splitlines() if l.strip()]
        out = Path(f"/tmp/preds/{ck.stem}__{lname}.jsonl")
        with out.open("w") as f:
            for i in range(0, len(imgs), 64):
                batch = imgs[i:i+64]
                try:
                    results = model.predict(batch, imgsz=imgsz, conf=0.02,
                                            device=0, verbose=False)
                except Exception as e:
                    for p in batch:
                        f.write(json.dumps({"image": p, "error": str(e)[:200]}) + "\n")
                    continue
                for p, r in zip(batch, results):
                    h, w = r.orig_shape
                    preds = [[float(b[0]) / w, float(b[1]) / h,
                              float(b[2]) / w, float(b[3]) / h, float(c)]
                             for b, c in zip(r.boxes.xyxy.cpu().tolist(),
                                             r.boxes.conf.cpu().tolist())]
                    f.write(json.dumps({"image": p, "preds": preds}) + "\n")
                if (i // 64) % 20 == 0:
                    print(f"{ck.stem}/{lname}: {i}/{len(imgs)}", flush=True)
        print(f"WROTE {out} ({len(imgs)} images)", flush=True)
print("INFER_DONE")
PYEOF
RC=$?
[ $RC -ne 0 ] && exit 97

ok=1
for f in /tmp/preds/*.jsonl; do
  base=$(basename "$f" .jsonl)
  var="URL_PREDS_$(echo "$base" | tr '[:lower:]-' '[:upper:]_' | tr -cd 'A-Z0-9_')"
  url="${!var:-}"
  if [ -n "$url" ]; then
    gzip -kf "$f"
    put "$url" "$f.gz" || ok=0
  else
    echo "no URL for $base ($var)"; ok=0
  fi
done
[ $ok -eq 1 ] && exit 0
exit 97
