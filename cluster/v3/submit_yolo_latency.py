"""Tiny job: batch-1 GPU latency for the YOLO11m control (expE) on one L40S.

Needed because the RF-DETR eval container has no ultralytics; this runs the
same measurement (cluster/v3/yolo_latency.py) in smoke-detect:v6, which does.
Code + weights + list go up as one tarball, result comes back over a presigned
PUT (small, well inside the URL lifetime).

Usage: python cluster/v3/submit_yolo_latency.py [--n 50] [--imgsz 1280]
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = "000000000001"
BUCKET = "smoke-detect-data-000000000001-use1"
REGION = "us-east-1"
KRAKEN = "kraken"
ADA = "ada"
IMAGE = f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/smoke-detect:v6"
QUEUE = "green-hyperpod-g6e.48xlarge-us-east-1-tas"

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--weights", default=str(ROOT / "models/trained/expE_mix_y11m_1280_v2/seg05/best.pt"))
ap.add_argument("--list", default=str(ROOT / "data/yolo/lists/local/sdis_val.txt"))
args = ap.parse_args()

p = subprocess.run([ADA, "credentials", "print", "--account", ACCOUNT, "--provider",
                    "INTERNAL-PROVIDER", "--role", "INTERNAL-ADMIN-ROLE",
                    "--format", "json"], capture_output=True, text=True, check=True)
c = json.loads(p.stdout)
sess = boto3.Session(aws_access_key_id=c["AccessKeyId"],
                     aws_secret_access_key=c["SecretAccessKey"],
                     aws_session_token=c["SessionToken"])
s3 = sess.client("s3", region_name=REGION,
                 endpoint_url=f"https://s3.{REGION}.amazonaws.com",
                 config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
exp = int(datetime.strptime(c["Expiration"], "%Y-%m-%dT%H:%M:%SZ")
          .replace(tzinfo=timezone.utc).timestamp())
ttl = max(600, exp - int(time.time()) - 60)

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tf:
    tf.add(ROOT / "cluster/v3/yolo_latency.py", arcname="yolo_latency.py")
    tf.add(ROOT / "src/__init__.py", arcname="src/__init__.py")
    tf.add(ROOT / "src/serve/__init__.py", arcname="src/serve/__init__.py")
    tf.add(ROOT / "src/serve/detector.py", arcname="src/serve/detector.py")
    tf.add(args.weights, arcname="expE_best.pt")
    tf.add(args.list, arcname="list.txt")
prefix = f"v3/evals/yololat/{int(time.time())}"
s3.put_object(Bucket=BUCKET, Key=f"{prefix}/bundle.tgz", Body=buf.getvalue())
gu = s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": f"{prefix}/bundle.tgz"}, ExpiresIn=ttl)
pu = s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": f"{prefix}/latency_gpu_yolo.json"}, ExpiresIn=ttl)

script = (
    'export HOME=/tmp USER=worker YOLO_CONFIG_DIR=/tmp/yolo_cfg HF_HOME=/tmp/hf; '
    'mkdir -p /tmp/job /tmp/yolo_cfg /tmp/hf; '
    f'curl -sf --max-time 300 -o /tmp/job/b.tgz "{gu}" || exit 11; '
    'tar xzf /tmp/job/b.tgz -C /tmp/job; cd /tmp/job; '
    f'python yolo_latency.py --weights /tmp/job/expE_best.pt --list /tmp/job/list.txt '
    f'--n {args.n} --device cuda:0 --imgsz {args.imgsz} --src-root /tmp/job '
    '--root-map "/local/home/tianrgua/cvproject/data/yolo=/scratch/scratch/tianrgua-sandbox/smoke_data_v2" '
    '--out /tmp/job/lat.json > /tmp/job/log.txt 2>&1; '
    'echo rc=$?; tail -5 /tmp/job/log.txt; '
    f'curl -sf --max-time 60 -T /tmp/job/lat.json "{pu}" && echo uploaded'
)
yaml_text = f"""jobNamePrefix: tianrgua-yololat
jobPriority: LOW
containers:
  - imageUri: {IMAGE}
    command: /bin/bash
    args:
      - -c
      - '{script}'
resourceConfig:
  nodeCount: 1
  queue: {QUEUE}
storageConfig:
  enableHomeWriteOnly: true
restartConfig:
  maxRestarts: 1
  restartPolicy: All
workloadTaxonomyConfig:
  workloadType: evaluation
  subtype: general
  role: general
  runType: dev
"""
yf = ROOT / "cluster/generated/v3_yolo_latency.yaml"
yf.write_text(yaml_text)
r = subprocess.run([KRAKEN, "-p", "green", "jobs", "create", "-i", str(yf)],
                   capture_output=True, text=True)
out = (r.stdout + r.stderr).strip()
print(out)
job = next((t.strip(".:;") for t in out.replace('"', " ").replace(",", " ").split()
            if t.startswith("tianrgua-yololat")), None)
print("job:", job, flush=True)
for _ in range(40):
    time.sleep(30)
    rr = subprocess.run([KRAKEN, "-p", "green", "jobs", "list", "-r", "tianrgua"],
                        capture_output=True, text=True)
    st = ""
    for line in (rr.stdout + rr.stderr).splitlines():
        if job and line.startswith(job):
            st = line.split()[1]
    if st.split(":")[0] in ("SUCCEEDED", "FAILED", "STOPPED"):
        print("terminal:", st, flush=True)
        break
dest = ROOT / "models/v3/evals/yolo_latency"
dest.mkdir(parents=True, exist_ok=True)
s3b = sess.client("s3", region_name=REGION)
for o in s3b.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", []):
    name = o["Key"].split("/")[-1]
    if name == "bundle.tgz":
        continue
    s3b.download_file(BUCKET, o["Key"], str(dest / name))
    print("got", name, o["Size"])
    print((dest / name).read_text()[:800])
