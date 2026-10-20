"""Fetch files from a /scratch runs_v3 run dir via a tiny cluster job.

(Artifacts must survive presigned-URL expiry -> everything lives in
/scratch; this job re-uploads requested files with fresh URLs. Pattern from
cluster/v2/fetch_eval.py.)

Usage:
  python cluster/v3/fetch_run.py --run rfdetrL_sanity \
      --files console.log --files last.ckpt
Then poll S3 v3/fetched/<run>/ (this script waits and downloads).
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = "000000000001"
BUCKET = "smoke-detect-data-000000000001-use1"
KRAKEN = "kraken"
ADA = "ada"

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--files", action="append", default=None,
                help="paths relative to the run dir (repeatable); default console.log")
ap.add_argument("--no-wait", action="store_true")
args = ap.parse_args()
files = args.files or ["console.log"]

p = subprocess.run([ADA, "credentials", "print", "--account", ACCOUNT,
                    "--provider INTERNAL-PROVIDER", "--role",
                    "INTERNAL-ADMIN-ROLE", "--format", "json"],
                   capture_output=True, text=True, check=True)
c = json.loads(p.stdout)
sess = boto3.Session(aws_access_key_id=c["AccessKeyId"],
                     aws_secret_access_key=c["SecretAccessKey"],
                     aws_session_token=c["SessionToken"])
s3 = sess.client("s3", region_name="us-east-1",
                 endpoint_url="https://s3.us-east-1.amazonaws.com",
                 config=Config(signature_version="s3v4",
                               s3={"addressing_style": "virtual"}))
pu = lambda k: s3.generate_presigned_url("put_object",
    Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=3500)

run_dir = f"/scratch/scratch/tianrgua-sandbox/smoke/runs_v3/{args.run}"
prefix = f"v3/fetched/{args.run}/{int(time.time())}"
cmds = [f'cd "{run_dir}" || exit 11', "ls -laR . > /tmp/listing.txt",
        f'curl -sf --max-time 60 -T /tmp/listing.txt "{pu(prefix + "/listing.txt")}"']
for f in files:
    safe = f.replace("/", "__")
    cmds.append(f'[ -f "{f}" ] && curl -sf --max-time 900 -T "{f}" "{pu(prefix + "/" + safe)}" || echo "missing {f}"')
cmds.append("echo done")
script = "; ".join(cmds)

envs_yaml = ""
yaml_text = f"""jobNamePrefix: tianrgua-f3{args.run.replace('_', '')[:10].lower()}
jobPriority: LOW
containers:
  - imageUri: 000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-det3:v1
    command: /bin/bash
    args:
      - -c
      - '{script}'
resourceConfig:
  nodeCount: 1
  queue: green-hyperpod-g6e.48xlarge-us-east-1-tas
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
yf = ROOT / "cluster/generated" / f"fetch3_{args.run}.yaml"
yf.write_text(yaml_text)
r = subprocess.run([KRAKEN, "-p", "green", "jobs", "create", "-i", str(yf)],
                   capture_output=True, text=True)
out = (r.stdout + r.stderr).strip()
print(out)
job = None
for tok in out.replace('"', " ").replace(",", " ").split():
    tok = tok.strip(".:;")
    if tok.startswith("tianrgua-f3"):
        job = tok
print("job:", job)
if args.no_wait or not job:
    print("prefix:", prefix)
    raise SystemExit(0)

for _ in range(60):
    time.sleep(30)
    r = subprocess.run([KRAKEN, "-p", "green", "jobs", "list", "-r", "tianrgua"],
                       capture_output=True, text=True)
    state = ""
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith(job):
            state = line.split()[1]
    if state.split(":")[0] in ("SUCCEEDED", "FAILED", "STOPPED"):
        print("terminal:", state)
        break
dest = ROOT / "models" / "v3" / "fetched" / args.run
dest.mkdir(parents=True, exist_ok=True)
s3b = sess.client("s3", region_name="us-east-1")
for o in s3b.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", []):
    name = o["Key"].split("/")[-1]
    s3b.download_file(BUCKET, o["Key"], str(dest / name))
    print("got", name, o["Size"])
print("->", dest)
