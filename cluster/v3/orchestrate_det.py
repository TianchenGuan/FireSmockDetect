"""Devbox-side orchestrator for iteration-3 Track A detection jobs (green g6e).

Long-job pattern (documented in det_entry.sh): one kraken job per attempt,
wall-bounded by WALL_S (default 12h), run state persisted in
/scratch/scratch/tianrgua-sandbox/smoke/runs_v3/<run>; resume = resubmit with
the same --run (driver auto-resumes from last.ckpt). Presigned status/console
beacons cover only the first ~50 min; afterwards use cluster/v3/fetch_run.py.

Usage:
  # sanity (overfit-16 + profile trials, ~30-40 min)
  python cluster/v3/orchestrate_det.py --run rfdetrL_sanity --mode sanity \
      --profile-list "2:1 4:1"
  # real run
  python cluster/v3/orchestrate_det.py --run rfdetrL_1280_e40 --mode train \
      --epochs 40 --bs 2 --ga 1 --wall-s 43200
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
IMAGE = f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/smoke-det3:v1"
QUEUE = "green-hyperpod-g6e.48xlarge-us-east-1-tas"
PROJECT = "green"

YAML_TMPL = """jobNamePrefix: tianrgua-{job}
jobPriority: LOW
containers:
  - imageUri: {image_uri}
    command: /bin/bash
    args:
      - -c
      - >-
        if curl -sf --max-time 120 -o /tmp/entry.sh "$URL_ENTRY";
        then bash /tmp/entry.sh; else bash /opt/det_entry.sh; fi
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
  subtype: general
  role: trainer
  runType: dev
"""


def ada_session():
    p = subprocess.run(
        [ADA, "credentials", "print", "--account", ACCOUNT, "--provider",
         "INTERNAL-PROVIDER", "--role", "INTERNAL-ADMIN-ROLE", "--format", "json"],
        capture_output=True, text=True, timeout=90, check=True)
    c = json.loads(p.stdout)
    sess = boto3.Session(aws_access_key_id=c["AccessKeyId"],
                         aws_secret_access_key=c["SecretAccessKey"],
                         aws_session_token=c["SessionToken"])
    return sess, c.get("Expiration")


def s3_client(sess):
    return sess.client("s3", region_name=REGION,
                       endpoint_url=f"https://s3.{REGION}.amazonaws.com",
                       config=Config(signature_version="s3v4",
                                     s3={"addressing_style": "virtual"}))


def kraken_jobs(args_list):
    p = subprocess.run([KRAKEN, "-p", PROJECT, "jobs"] + args_list,
                       capture_output=True, text=True, timeout=180)
    return p.returncode, p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", choices=["sanity", "train"], default="train")
    ap.add_argument("--arch", choices=["rfdetr", "deimv2"], default="rfdetr")
    ap.add_argument("--deimv2-cfg", default=str(ROOT / "cluster/v3/deimv2_m_smoke_1280.yml"))
    ap.add_argument("--deimv2-updates", default="")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--res", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--ga", type=int, default=1)
    ap.add_argument("--lr", default="1e-4")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt-int", type=int, default=5)
    ap.add_argument("--eval-int", type=int, default=1)
    ap.add_argument("--overfit-epochs", type=int, default=50)
    ap.add_argument("--profile-list", default="2:1 4:1")
    ap.add_argument("--profile-timeout", type=int, default=720)
    ap.add_argument("--wall-s", type=int, default=43200)
    ap.add_argument("--image-uri", default=IMAGE)
    ap.add_argument("--poll-s", type=int, default=120)
    args = ap.parse_args()

    sess, cred_exp = ada_session()
    s3 = s3_client(sess)
    exp_epoch = int(datetime.strptime(cred_exp, "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=timezone.utc).timestamp())
    ttl = max(600, exp_epoch - int(time.time()) - 60)

    # upload freshest entry + driver, presign GET
    code_prefix = f"v3/code/{args.run}_a{args.attempt}"
    s3.upload_file(str(ROOT / "cluster/v3/det_entry.sh"), BUCKET, f"{code_prefix}/det_entry.sh")
    s3.upload_file(str(ROOT / "scripts/v3/train_rfdetr.py"), BUCKET, f"{code_prefix}/train_rfdetr.py")
    gu = lambda k: s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=ttl)
    pu = lambda k: s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=ttl)
    run_prefix = f"v3/runs/{args.run}/a{args.attempt:02d}"

    envs = {
        "RUN_NAME": args.run, "MODE": args.mode,
        "RES": str(args.res), "EPOCHS": str(args.epochs),
        "BS": str(args.bs), "GA": str(args.ga), "LR": args.lr,
        "WORKERS": str(args.workers), "CKPT_INT": str(args.ckpt_int),
        "EVAL_INT": str(args.eval_int), "WALL_S": str(args.wall_s),
        "OVERFIT_EPOCHS": str(args.overfit_epochs),
        "PROFILE_LIST": args.profile_list,
        "PROFILE_TIMEOUT": str(args.profile_timeout),
        "URL_ENTRY": gu(f"{code_prefix}/det_entry.sh"),
        "URL_CODE": gu(f"{code_prefix}/train_rfdetr.py"),
        "URL_STATUS": pu(f"{run_prefix}/status.json"),
        "URL_CONSOLE": pu(f"{run_prefix}/console_tail.txt"),
        "URL_EXPIRY_EPOCH": str(exp_epoch),
    }
    if args.arch == "deimv2":
        s3.upload_file(args.deimv2_cfg, BUCKET, f"{code_prefix}/deimv2_cfg.yml")
        envs["ARCH"] = "deimv2"
        envs["URL_CFG"] = gu(f"{code_prefix}/deimv2_cfg.yml")
        if args.deimv2_updates:
            envs["DEIMV2_UPDATES"] = args.deimv2_updates
    env_yaml = "\n".join(
        f"      - name: {k}\n        value: \"{v}\"" for k, v in envs.items())
    tag = (args.run.replace("_", "") + f"a{args.attempt}")[:16].lower()
    ydir = ROOT / "cluster" / "generated"
    ydir.mkdir(exist_ok=True)
    yfile = ydir / f"v3_{args.run}_a{args.attempt:02d}.yaml"
    yfile.write_text(YAML_TMPL.format(job=tag, image_uri=args.image_uri,
                                      envs=env_yaml, queue=QUEUE))
    rc, out = kraken_jobs(["create", "-i", str(yfile)])
    if rc != 0:
        raise RuntimeError(f"submit failed: {out}")
    job_name = None
    for tok in out.replace('"', " ").replace(",", " ").split():
        tok = tok.strip(".:;")
        if tok.startswith(f"tianrgua-{tag}"):
            job_name = tok
    print(f"submitted {job_name} (mode={args.mode}, run={args.run}, "
          f"attempt={args.attempt}, url ttl {ttl}s)", flush=True)
    if not job_name:
        return 2

    t0 = time.time()
    last = ""
    max_wait = args.wall_s + 7200
    while time.time() - t0 < max_wait:
        rc, out = kraken_jobs(["list", "-r", "tianrgua"])
        state = ""
        for line in out.splitlines():
            if line.startswith(job_name):
                state = line.split()[1]
        if state != last:
            print(f"  [{int(time.time()-t0)}s] {state}", flush=True)
            last = state
        if state.split(":")[0] in ("SUCCEEDED", "FAILED", "STOPPED"):
            rc, desc = kraken_jobs(["describe", "--job-name", job_name])
            info = [l for l in desc.splitlines() if "exit code" in l]
            print(f"terminal: {state} | {info[0].strip() if info else ''}", flush=True)
            break
        time.sleep(args.poll_s)

    # pull whatever beacons made it to S3
    dest = ROOT / "models" / "v3" / "runs" / args.run / f"a{args.attempt:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    sess2, _ = ada_session()
    s3b = sess2.client("s3", region_name=REGION)
    got = {}
    for o in s3b.list_objects_v2(Bucket=BUCKET, Prefix=run_prefix).get("Contents", []):
        name = o["Key"].split("/")[-1]
        s3b.download_file(BUCKET, o["Key"], str(dest / name))
        got[name] = o["Size"]
    print(f"fetched beacons: {got} -> {dest}", flush=True)
    st = dest / "status.json"
    if st.exists():
        print("status:", st.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
