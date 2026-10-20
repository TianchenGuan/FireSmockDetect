"""One-shot job: stage a v4 data tar onto /scratch (FSx) so later eval jobs can
read it directly, instead of shipping a frames tarball per job.

Idempotent pod-side via a .<name>.done marker (same convention as
cluster/v4/train_entry_v4.sh). Upload of the tar to S3 is skipped when the
local marker cluster/v4/.uploaded_<name> exists.

Usage:
  python cluster/v4/stage_evalval.py --name v4_evalval \
      --tar data/v4/stage/v4_evalval.tar --probe v4_evalval/boreal_val/lists/pod/val.txt
"""
from __future__ import annotations

import argparse
import json
import subprocess
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
IMAGE = f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/smoke-dino3:v1"
QUEUE = "green-hyperpod-g6e.48xlarge-us-east-1-tas"
PROJECT = "green"

ENTRY = r'''#!/bin/bash
set -u +e
export HOME=/tmp
V4=/scratch/scratch/tianrgua-sandbox/smoke/v4data
NAME="${TAR_NAME:?}"
mkdir -p "$V4" || { echo "v4data unwritable"; exit 95; }
if [ -f "$V4/.${NAME}.done" ]; then
  echo "already staged"
else
  curl -sf --max-time 2400 -o "$V4/${NAME}.tar" "$URL_TAR" \
    && tar -xf "$V4/${NAME}.tar" -C "$V4" \
    && rm -f "$V4/${NAME}.tar" && touch "$V4/.${NAME}.done" \
    || { rm -f "$V4/${NAME}.tar"; echo "stage failed"; exit 95; }
fi
{ echo "=== listing ==="; ls -la "$V4"; ls "$V4/$NAME" 2>/dev/null;
  for s in boreal_val forest_val; do
    echo "$s images: $(ls "$V4/$NAME/$s/images" 2>/dev/null | wc -l)";
    echo "$s labels: $(ls "$V4/$NAME/$s/labels" 2>/dev/null | wc -l)";
  done
  echo "probe ${PROBE:-none}:"; head -2 "$V4/${PROBE:-/dev/null}" 2>/dev/null;
  df -B1G /scratch | tail -1; } > /tmp/report.txt 2>&1
cat /tmp/report.txt
curl -sf --max-time 120 -T /tmp/report.txt "$URL_REPORT" && echo uploaded
exit 0
'''

YAML_TMPL = """jobNamePrefix: tianrgua-{job}
jobPriority: LOW
containers:
  - imageUri: {image_uri}
    command: /bin/bash
    args:
      - -c
      - >-
        curl -sf --max-time 120 -o /tmp/entry.sh "$URL_ENTRY" && bash /tmp/entry.sh
    environmentVariables:
{envs}
resourceConfig:
  nodeCount: 1
  queue: {queue}
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


def ada_session():
    p = subprocess.run([ADA, "credentials", "print", "--account", ACCOUNT,
                        "--provider INTERNAL-PROVIDER", "--role",
                        "INTERNAL-ADMIN-ROLE", "--format", "json"],
                       capture_output=True, text=True, timeout=90, check=True)
    c = json.loads(p.stdout)
    return boto3.Session(aws_access_key_id=c["AccessKeyId"],
                         aws_secret_access_key=c["SecretAccessKey"],
                         aws_session_token=c["SessionToken"]), c["Expiration"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--probe", default="")
    ap.add_argument("--poll-s", type=int, default=30)
    args = ap.parse_args()

    sess, exp = ada_session()
    s3 = sess.client("s3", region_name=REGION,
                     endpoint_url=f"https://s3.{REGION}.amazonaws.com",
                     config=Config(signature_version="s3v4",
                                   s3={"addressing_style": "virtual"}))
    exp_epoch = int(datetime.strptime(exp, "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=timezone.utc).timestamp())
    ttl = max(600, exp_epoch - int(time.time()) - 60)

    prefix = f"v4/stage/{args.name}"
    key_tar = f"v4/data/{args.name}.tar"
    marker = ROOT / "cluster/v4" / f".uploaded_{args.name}"
    if not marker.exists():
        print(f"uploading {args.tar} -> s3://{BUCKET}/{key_tar} ...", flush=True)
        s3.upload_file(str(ROOT / args.tar), BUCKET, key_tar)
        marker.write_text(datetime.now(timezone.utc).isoformat())
    else:
        print("tar already uploaded (marker present)")
    s3.put_object(Bucket=BUCKET, Key=f"{prefix}/entry.sh", Body=ENTRY.encode())

    gu = lambda k: s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=ttl)
    pu = lambda k: s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=ttl)
    envs = {"TAR_NAME": args.name, "PROBE": args.probe,
            "URL_ENTRY": gu(f"{prefix}/entry.sh"), "URL_TAR": gu(key_tar),
            "URL_REPORT": pu(f"{prefix}/report.txt")}
    env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\"" for k, v in envs.items())
    tag = ("stg" + args.name.replace("_", ""))[:14].lower()
    ydir = ROOT / "cluster/generated"
    ydir.mkdir(exist_ok=True)
    yfile = ydir / f"v4_stage_{args.name}.yaml"
    yfile.write_text(YAML_TMPL.format(job=tag, image_uri=IMAGE, envs=env_yaml, queue=QUEUE))
    r = subprocess.run([KRAKEN, "-p", PROJECT, "jobs", "create", "-i", str(yfile)],
                       capture_output=True, text=True, timeout=180)
    out = (r.stdout + r.stderr).strip()
    print(out)
    job = None
    for tok in out.replace('"', " ").replace(",", " ").split():
        tok = tok.strip(".:;")
        if tok.startswith(f"tianrgua-{tag}"):
            job = tok
    print("job:", job, flush=True)
    if not job:
        return 2
    t0, last = time.time(), ""
    while time.time() - t0 < 3600:
        rr = subprocess.run([KRAKEN, "-p", PROJECT, "jobs", "list", "-r", "tianrgua"],
                            capture_output=True, text=True, timeout=180)
        state = ""
        for line in (rr.stdout + rr.stderr).splitlines():
            if line.startswith(job):
                state = line.split()[1]
        if state != last:
            print(f"  [{int(time.time()-t0)}s] {state}", flush=True)
            last = state
        if state.split(":")[0] in ("SUCCEEDED", "FAILED", "STOPPED"):
            break
        time.sleep(args.poll_s)
    sess2, _ = ada_session()
    s3b = sess2.client("s3", region_name=REGION)
    dest = ROOT / "models/v4/stage" / args.name
    dest.mkdir(parents=True, exist_ok=True)
    for o in s3b.list_objects_v2(Bucket=BUCKET, Prefix=f"{prefix}/report.txt").get("Contents", []):
        s3b.download_file(BUCKET, o["Key"], str(dest / "report.txt"))
        print((dest / "report.txt").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
