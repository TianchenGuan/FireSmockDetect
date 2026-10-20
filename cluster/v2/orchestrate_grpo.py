"""GRPO feasibility-probe submitter (iteration-2 §6). One-shot variant of
orchestrate_vlm.py: uploads code/probe, presigns GET/PUT (incl. GET of the
frozen SFT adapter tgz already in S3), submits ONE job on green g6e, polls,
fetches artifacts.

Usage:
  python cluster/v2/orchestrate_grpo.py --run grpo_e2e --attempt 1 \
    --model-dir /scratch/scratch/tianrgua-sandbox/models/qwen3.5-9b
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
BUCKET = "smoke-detect-data-000000000001-use1"
REGION = "us-east-1"
QUEUE = "green-hyperpod-g6e.48xlarge-us-east-1-tas"
PROJECT = "green"
KRAKEN = "kraken"
ADAPTER_KEY = "v2/runs/trkA_9b_lora/seg05/final_adapter.tgz"

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
  workloadType: training
  subtype: rl
  role: trainer
  runType: dev
"""


def ada_session():
    p = subprocess.run(["ada", "credentials",
                        "print", "--account", "000000000001", "--provider",
                        "INTERNAL-PROVIDER", "--role", "INTERNAL-ADMIN-ROLE",
                        "--format", "json"],
                       capture_output=True, text=True, timeout=90, check=True)
    c = json.loads(p.stdout)
    return (boto3.Session(aws_access_key_id=c["AccessKeyId"],
                          aws_secret_access_key=c["SecretAccessKey"],
                          aws_session_token=c["SessionToken"]),
            c["Expiration"])


def s3c(sess):
    return sess.client("s3", region_name=REGION,
                       endpoint_url=f"https://s3.{REGION}.amazonaws.com",
                       config=Config(signature_version="s3v4",
                                     s3={"addressing_style": "virtual"}))


def kraken(args_list):
    p = subprocess.run([KRAKEN, "-p", PROJECT, "jobs"] + args_list,
                       capture_output=True, text=True, timeout=180)
    return p.returncode, p.stdout + p.stderr


def submit(args):
    sess, exp = ada_session()
    s3 = s3c(sess)
    exp_epoch = int(datetime.strptime(exp, "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=timezone.utc).timestamp())
    ttl = max(600, exp_epoch - int(time.time()) - 60)
    kp = f"v2/jobs/{args.run}"
    up = {f"{kp}/grpo_train.py": ROOT / "cluster/v2/grpo_train.py",
          f"{kp}/grpo_reward.py": ROOT / "src/train/grpo_reward.py",
          f"{kp}/eval_vlm.py": ROOT / "src/eval/eval_vlm.py",
          f"{kp}/entry.sh": ROOT / "cluster/v2/grpo_entry.sh",
          f"{kp}/probe.jsonl": ROOT / "data/v2/sft/grpo_probe_32.jsonl"}
    for key, path in up.items():
        s3.upload_file(str(path), BUCKET, key)
    s3.head_object(Bucket=BUCKET, Key=ADAPTER_KEY)  # fail fast if missing

    def get(key):
        return s3.generate_presigned_url("get_object",
            Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

    def put(key):
        return s3.generate_presigned_url("put_object",
            Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

    op = f"v2/runs/{args.run}/attempt{args.attempt:02d}"
    envs = {
        "URL_ENTRY": get(f"{kp}/entry.sh"),
        "URL_CODE": get(f"{kp}/grpo_train.py"),
        "URL_REWARD": get(f"{kp}/grpo_reward.py"),
        "URL_EVALVLM": get(f"{kp}/eval_vlm.py"),
        "URL_DATA": get(f"{kp}/probe.jsonl"),
        "URL_ADAPTER_IN": get(ADAPTER_KEY),
        "RUN_NAME": args.run, "MODEL_DIR": args.model_dir,
        "MAX_STEPS": str(args.max_steps), "NUM_GEN": str(args.num_gen),
        "MAX_NEW_TOKENS": str(args.max_new_tokens), "BS": str(args.bs),
        "LR": args.lr, "TEMP": args.temp,
        "MAX_PIXELS": str(args.max_pixels), "NGPUS": str(args.ngpus),
        "URL_EXPIRY_EPOCH": str(exp_epoch),
        "URL_STATUS": put(f"{op}/status.json"),
        "URL_CONSOLE": put(f"{op}/console_tail.txt"),
        "URL_TLOG": put(f"{op}/trainer_log.jsonl"),
        "URL_RLOG": put(f"{op}/reward_log.jsonl"),
        "URL_ADAPTER_OUT": put(f"{op}/final_adapter.tgz"),
    }
    env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\""
                         for k, v in envs.items())
    tag = f"grpoe2ea{args.attempt}"
    yfile = ROOT / "cluster/generated" / f"grpo_{args.run}_a{args.attempt:02d}.yaml"
    yfile.write_text(YAML_TMPL.format(job=tag, image_uri=args.image_uri,
                                      envs=env_yaml, queue=QUEUE))
    rc, out = kraken(["create", "-i", str(yfile)])
    if rc != 0:
        raise RuntimeError(out)
    job = None
    for tok in out.replace('"', ' ').split():
        tok = tok.strip(".:;,")
        if tok.startswith(f"tianrgua-{tag}"):
            job = tok
    print(f"submitted {job} ttl={ttl}s out_prefix={op}", flush=True)
    return job, op, exp_epoch


def wait(job, stale_deadline):
    t0, last = time.time(), ""
    while time.time() - t0 < 10800:
        rc, out = kraken(["list", "-r", "tianrgua"])
        st = ""
        for line in out.splitlines():
            if line.startswith(job):
                st = line.split()[1]
        if st != last:
            print(f"  [{int(time.time()-t0)}s] {st}", flush=True)
            last = st
        if st.split(":")[0] in ("SUCCEEDED", "FAILED", "STOPPED"):
            rc, d = kraken(["describe", "--job-name", job])
            ec = [l.strip() for l in d.splitlines() if "exit code" in l]
            return st + " | " + (ec[0] if ec else "")
        if stale_deadline and time.time() > stale_deadline and \
                st.split(":")[0] in ("PENDING", "QUEUED", "CREATED", ""):
            kraken(["stop", "--job-name", job])
            return "STALE"
        time.sleep(60)
    return "TIMEOUT"


def fetch(op, dest):
    sess, _ = ada_session()
    s3 = sess.client("s3", region_name=REGION)
    Path(dest).mkdir(parents=True, exist_ok=True)
    got = {}
    for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=op).get("Contents", []):
        n = o["Key"].split("/")[-1]
        s3.download_file(BUCKET, o["Key"], str(Path(dest) / n))
        got[n] = o["Size"]
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="grpo_e2e")
    ap.add_argument("--attempt", type=int, required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--num-gen", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--lr", default="5e-6")
    ap.add_argument("--temp", default="1.0")
    ap.add_argument("--max-pixels", type=int, default=1048576)
    ap.add_argument("--ngpus", type=int, default=8)
    ap.add_argument("--image-uri",
        default="000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-detect-vlm:v3")
    args = ap.parse_args()

    job, op, exp_epoch = submit(args)
    st = wait(job, exp_epoch - 1500)
    print(f"final: {st}", flush=True)
    dest = ROOT / "models/v2/runs" / args.run / f"attempt{args.attempt:02d}"
    got = fetch(op, dest)
    print(f"fetched {got} -> {dest}", flush=True)
    sf = dest / "status.json"
    if sf.exists():
        print("status " + sf.read_text(), flush=True)
    return 0 if "SUCCEEDED" in st else 1


if __name__ == "__main__":
    raise SystemExit(main())
