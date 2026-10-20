"""Fetch persisted eval results from /scratch via a tiny cluster job."""
import argparse, json, subprocess, time
from pathlib import Path
import boto3
from botocore.config import Config
ROOT = Path(__file__).resolve().parents[2]
BUCKET = "smoke-detect-data-000000000001-use1"
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True)
args = ap.parse_args()
p = subprocess.run(["ada","credentials","print","--account","000000000001","--provider INTERNAL-PROVIDER","--role","INTERNAL-ADMIN-ROLE","--format","json"], capture_output=True, text=True, check=True)
c = json.loads(p.stdout)
sess = boto3.Session(aws_access_key_id=c["AccessKeyId"], aws_secret_access_key=c["SecretAccessKey"], aws_session_token=c["SessionToken"])
s3 = sess.client("s3", region_name="us-east-1", endpoint_url="https://s3.us-east-1.amazonaws.com", config=Config(signature_version="s3v4", s3={"addressing_style":"virtual"}))
pu = lambda k: s3.generate_presigned_url("put_object", Params={"Bucket":BUCKET,"Key":k}, ExpiresIn=3500)
envs = {"P": f"/scratch/scratch/tianrgua-sandbox/smoke/evals_v2/{args.name}",
        "U1": pu(f"v2/evals/{args.name}/preds.jsonl"),
        "U2": pu(f"v2/evals/{args.name}/summary.json"),
        "U3": pu(f"v2/evals/{args.name}/log0.txt")}
env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\"" for k,v in envs.items())
yaml_text = f"""jobNamePrefix: tianrgua-fetch{args.name.replace('_','')[:10].lower()}
jobPriority: LOW
containers:
  - imageUri: 000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-detect:v6
    command: /bin/bash
    args:
      - -c
      - >-
        set -e;
        curl -sf --max-time 300 -T "$P/all.jsonl" "$U1";
        curl -sf --max-time 60 -T "$P/summary.json" "$U2";
        curl -sf -T "$P/log0.txt" "$U3" || true;
        echo done
    environmentVariables:
{env_yaml}
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
yf = ROOT/"cluster/generated"/f"fetch_{args.name}.yaml"; yf.write_text(yaml_text)
r = subprocess.run(["kraken","-p","green","jobs","create","-i",str(yf)], capture_output=True, text=True)
print((r.stdout+r.stderr).strip())
