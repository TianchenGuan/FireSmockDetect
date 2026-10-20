"""VLM SFT orchestrator (v2): segmented resumable Qwen3.5 LoRA jobs on
green g6e. Per segment: upload latest code/config/data to S3, mint presigned
URLs, submit, poll, fetch artifacts, resubmit until done.

Usage (after container+weights+data staging are verified):
  python cluster/v2/orchestrate_vlm.py --run trkA_9b_lora \
    --model-dir /scratch/scratch/tianrgua-sandbox/models/qwen3.5-9b \
    --data data/v2/sft/train.jsonl --val data/v2/sft/val_mini.jsonl \
    --ds cluster/v2/zero2.json --epochs 1 [--max-steps N] [--mode overfit]
Modes: overfit (16-sample sanity, 60 steps), profile (100 steps), train.
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
  subtype: sft
  role: trainer
  runType: dev
"""


def ada_session():
    p = subprocess.run(["ada", "credentials", "print",
                        "--account", "000000000001", "--provider INTERNAL-PROVIDER",
                        "--role", "INTERNAL-ADMIN-ROLE", "--format", "json"],
                       capture_output=True, text=True, timeout=90, check=True)
    c = json.loads(p.stdout)
    return (boto3.Session(aws_access_key_id=c["AccessKeyId"],
                          aws_secret_access_key=c["SecretAccessKey"],
                          aws_session_token=c["SessionToken"]), c["Expiration"])


def s3c(sess):
    return sess.client("s3", region_name=REGION,
                       endpoint_url=f"https://s3.{REGION}.amazonaws.com",
                       config=Config(signature_version="s3v4",
                                     s3={"addressing_style": "virtual"}))


def kraken(args_list):
    p = subprocess.run([KRAKEN, "-p", PROJECT, "jobs"] + args_list,
                       capture_output=True, text=True, timeout=180)
    return p.returncode, p.stdout + p.stderr


def submit(seg, args):
    sess, exp = ada_session()
    s3 = s3c(sess)
    exp_epoch = int(datetime.strptime(exp, "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=timezone.utc).timestamp())
    ttl = max(600, exp_epoch - int(time.time()) - 60)
    kp = f"v2/jobs/{args.run}"
    # upload current code/config/data
    up = {f"{kp}/train_vlm.py": ROOT / "src/train/train_vlm.py",
          f"{kp}/entry.sh": ROOT / "cluster/v2/vlm_entry.sh",
          f"{kp}/ds.json": Path(args.ds),
          f"{kp}/train.jsonl": Path(args.data)}
    if args.val:
        up[f"{kp}/val.jsonl"] = Path(args.val)
    for key, path in up.items():
        s3.upload_file(str(path), BUCKET, key)

    def get(key):
        return s3.generate_presigned_url("get_object",
            Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

    def put(key):
        return s3.generate_presigned_url("put_object",
            Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

    op = f"v2/runs/{args.run}/seg{seg:02d}"
    envs = {
        "URL_ENTRY": get(f"{kp}/entry.sh"), "URL_CODE": get(f"{kp}/train_vlm.py"),
        "URL_DS": get(f"{kp}/ds.json"), "URL_DATA": get(f"{kp}/train.jsonl"),
        "RUN_NAME": args.run, "MODEL_DIR": args.model_dir,
        "MAX_STEPS": str(args.max_steps), "EPOCHS": str(args.epochs),
        "BS": str(args.bs), "GA": str(args.ga), "LR": args.lr,
        "LORA_R": str(args.lora_r), "LORA_ALPHA": str(args.lora_alpha),
        "MAX_PIXELS": str(args.max_pixels), "SAVE_STEPS": str(args.save_steps),
        "NGPUS": str(args.ngpus), "WORKERS": "4",
        "EXTRA_ARGS": args.extra_args,
        "NO_DS": "1" if args.no_ds else "0",
        "URL_EXPIRY_EPOCH": str(exp_epoch),
        "URL_STATUS": put(f"{op}/status.json"),
        "URL_CONSOLE": put(f"{op}/console_tail.txt"),
        "URL_TLOG": put(f"{op}/trainer_log.jsonl"),
        "URL_ADAPTER": put(f"{op}/final_adapter.tgz"),
    }
    if args.val:
        envs["URL_VAL"] = get(f"{kp}/val.jsonl")
    env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\""
                         for k, v in envs.items())
    yfile = ROOT / "cluster/generated" / f"vlm_{args.run}_s{seg:02d}.yaml"
    yfile.write_text(YAML_TMPL.format(job=f"{args.job_tag}s{seg}",
                                      image_uri=args.image_uri,
                                      envs=env_yaml, queue=QUEUE))
    rc, out = kraken(["create", "-i", str(yfile)])
    if rc != 0:
        raise RuntimeError(out)
    job = None
    for tok in out.replace('"', ' ').split():
        tok = tok.strip(".:;,")
        if tok.startswith(f"tianrgua-{args.job_tag}s{seg}"):
            job = tok
    print(f"[seg {seg}] submitted {job} ttl={ttl}s", flush=True)
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


def fetch(op, dest, args):
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
    ap.add_argument("--run", required=True)
    ap.add_argument("--job-tag", default=None)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--ds", default="cluster/v2/zero2.json")
    ap.add_argument("--mode", default="train", choices=["overfit", "profile", "train"])
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--ga", type=int, default=8)
    ap.add_argument("--lr", default="1e-4")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=1048576)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--ngpus", type=int, default=8)
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--no-ds", action="store_true",
                    help="pure DDP (QLoRA path): omit --deepspeed entirely")
    ap.add_argument("--image-uri",
        default="000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-detect-vlm:v1")
    ap.add_argument("--max-segments", type=int, default=12)
    ap.add_argument("--start-seg", type=int, default=1)
    args = ap.parse_args()
    if args.mode == "overfit":
        args.max_steps, args.save_steps, args.ga = 60, 50, 1
    elif args.mode == "profile":
        args.max_steps, args.save_steps = 100, 90
    if not args.job_tag:
        args.job_tag = ("vlm" + args.run.replace("_", ""))[:12].lower()
    args.job_tag = args.job_tag.lower()

    outdir = ROOT / "models/v2/runs" / args.run
    for seg in range(args.start_seg, args.max_segments + 1):
        job, op, exp_epoch = submit(seg, args)
        st = wait(job, exp_epoch - 1500)
        print(f"[seg {seg}] {st}", flush=True)
        if st == "STALE":
            continue
        got = fetch(op, outdir / f"seg{seg:02d}", args)
        print(f"[seg {seg}] fetched {got}", flush=True)
        sf = outdir / f"seg{seg:02d}/status.json"
        if sf.exists():
            s = json.loads(sf.read_text())
            print(f"[seg {seg}] status {s}", flush=True)
            if s.get("phase") == "done":
                print(f"RUN COMPLETE ({seg} segs) -> {outdir}", flush=True)
                return 0
            if s.get("phase") in ("model_missing", "data_missing"):
                print("fatal staging problem — aborting", flush=True)
                return 3
    print("max segments without done", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
