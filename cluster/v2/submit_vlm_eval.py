"""Submit a VLM eval job (base or +adapter) on green g6e."""
import argparse, json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[2]
BUCKET = "smoke-detect-data-000000000001-use1"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--adapter-key", default=None, help="S3 key of final_adapter.tgz")
    ap.add_argument("--list", default="data/v2/sft/val_eval_2000.txt")
    ap.add_argument("--max-images", default="")
    ap.add_argument("--max-pixels", default="1048576")
    ap.add_argument("--quant", default="")
    ap.add_argument("--max-new-tokens", default="192")
    args = ap.parse_args()

    p = subprocess.run(["ada","credentials","print","--account","000000000001","--provider INTERNAL-PROVIDER","--role","INTERNAL-ADMIN-ROLE","--format","json"], capture_output=True, text=True, check=True)
    c = json.loads(p.stdout)
    sess = boto3.Session(aws_access_key_id=c["AccessKeyId"], aws_secret_access_key=c["SecretAccessKey"], aws_session_token=c["SessionToken"])
    s3 = sess.client("s3", region_name="us-east-1", endpoint_url="https://s3.us-east-1.amazonaws.com", config=Config(signature_version="s3v4", s3={"addressing_style":"virtual"}))
    kp = f"v2/evaljobs/{args.name}"
    s3.upload_file(str(ROOT/"src/eval/eval_vlm.py"), BUCKET, f"{kp}/eval_vlm.py")
    s3.upload_file(str(ROOT/"cluster/v2/vlm_eval_entry.sh"), BUCKET, f"{kp}/entry.sh")
    s3.upload_file(str(ROOT/args.list), BUCKET, f"{kp}/list.txt")
    g = lambda k: s3.generate_presigned_url("get_object", Params={"Bucket":BUCKET,"Key":k}, ExpiresIn=3500)
    pu = lambda k: s3.generate_presigned_url("put_object", Params={"Bucket":BUCKET,"Key":k}, ExpiresIn=3500)
    envs = {
        "URL_ENTRY": g(f"{kp}/entry.sh"), "URL_CODE": g(f"{kp}/eval_vlm.py"),
        "URL_LIST": g(f"{kp}/list.txt"), "MODEL_DIR": args.model_dir,
        "EVAL_NAME": args.name,
        "MAX_PIXELS": args.max_pixels,
        "URL_PREDS": pu(f"v2/evals/{args.name}/preds.jsonl"),
        "URL_SUMMARY": pu(f"v2/evals/{args.name}/summary.json"),
        "URL_LOG": pu(f"v2/evals/{args.name}/log0.txt"),
    }
    if args.max_images: envs["MAX_IMAGES"] = args.max_images
    if args.quant: envs["QUANT"] = args.quant
    envs["MAX_NEW_TOKENS"] = args.max_new_tokens
    if args.adapter_key:
        envs["URL_ADAPTER_GET"] = g(args.adapter_key)
    env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\"" for k,v in envs.items())
    yaml_text = f"""jobNamePrefix: tianrgua-{args.name.replace('_','')[:14].lower()}
jobPriority: LOW
containers:
  - imageUri: 000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-detect-vlm:v2
    command: /bin/bash
    args:
      - -c
      - >-
        curl -sf --max-time 120 -o /tmp/entry.sh "$URL_ENTRY" && bash /tmp/entry.sh
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
  subtype: benchmark
  role: general
  runType: dev
"""
    yf = ROOT/"cluster/generated"/f"vlmeval_{args.name}.yaml"
    yf.write_text(yaml_text)
    r = subprocess.run(["kraken","-p","green","jobs","create","-i",str(yf)], capture_output=True, text=True)
    print((r.stdout+r.stderr).strip())

if __name__ == "__main__":
    main()
