"""Track D orchestrator: segmented resumable jobs on green g6e (pattern from
cluster/v2/orchestrate_vlm.py — presigned regional URLs, /scratch persistence,
exit-code + status.json observability).

  python cluster/v3/orchestrate_trackd.py extract
  python cluster/v3/orchestrate_trackd.py train --run trackd_head_v1 \
      [--epochs 8] [--extra-args "--max-steps 20 --overfit-seqs 8"]
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
IMAGE = "000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-detect:v6"
WEIGHTS = ROOT / "models/trained/expE_mix_y11m_1280_v2/seg05/best.pt"

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
  subtype: general
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
    kp = "v3/jobs/trackd"
    up = {f"{kp}/entry.sh": ROOT / "cluster/v3/trackd_entry.sh",
          f"{kp}/trackd_features.py": ROOT / "src/v3/trackd_features.py",
          f"{kp}/trackd_temporal.py": ROOT / "src/v3/trackd_temporal.py",
          f"{kp}/sequences_train.json": ROOT / "data/v3/sequences/sequences_train.json",
          f"{kp}/sequences_val.json": ROOT / "data/v3/sequences/sequences_val.json"}
    wkey = f"{kp}/expE_best.pt"
    if not args.skip_weights_upload:
        up[wkey] = WEIGHTS
    for key, path in up.items():
        s3.upload_file(str(path), BUCKET, key)

    def get(key):
        return s3.generate_presigned_url("get_object",
            Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

    def put(key):
        return s3.generate_presigned_url("put_object",
            Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

    op = f"v3/runs/trackd/{args.run}/seg{seg:02d}"
    envs = {
        "MODE": args.mode, "RUN_NAME": args.run,
        "URL_ENTRY": get(f"{kp}/entry.sh"),
        "URL_FEAT_CODE": get(f"{kp}/trackd_features.py"),
        "URL_HEAD_CODE": get(f"{kp}/trackd_temporal.py"),
        "URL_WEIGHTS": get(wkey),
        "URL_SEQ_TRAIN": get(f"{kp}/sequences_train.json"),
        "URL_SEQ_VAL": get(f"{kp}/sequences_val.json"),
        "URL_EXPIRY_EPOCH": str(exp_epoch),
        "URL_STATUS": put(f"{op}/status.json"),
        "URL_CONSOLE": put(f"{op}/console_tail.txt"),
        "EPOCHS": str(args.epochs), "BS": str(args.bs), "LR": args.lr,
        "EXTRA_ARGS": args.extra_args,
    }
    if args.mode == "extract":
        envs["URL_FEATS_TRAIN"] = put("v3/runs/trackd/feats_train.tgz")
        envs["URL_FEATS_VAL"] = put("v3/runs/trackd/feats_val.tgz")
    else:
        envs["URL_HEAD"] = put(f"{op}/head.tgz")
    env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\""
                         for k, v in envs.items())
    tag = (args.mode[:3] + args.run.replace("_", ""))[:12].lower()
    yfile = ROOT / "cluster/generated" / f"trackd_{args.mode}_{args.run}_s{seg:02d}.yaml"
    yfile.write_text(YAML_TMPL.format(job=f"{tag}s{seg}", image_uri=IMAGE,
                                      envs=env_yaml, queue=QUEUE))
    rc, out = kraken(["create", "-i", str(yfile)])
    if rc != 0:
        raise RuntimeError(out)
    job = None
    for tok in out.replace('"', ' ').split():
        tok = tok.strip(".:;,")
        if tok.startswith(f"tianrgua-{tag}s{seg}"):
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
        time.sleep(45)
    return "TIMEOUT"


def fetch(prefix, dest):
    sess, _ = ada_session()
    s3 = sess.client("s3", region_name=REGION)
    Path(dest).mkdir(parents=True, exist_ok=True)
    got = {}
    for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", []):
        n = o["Key"].split("/")[-1]
        s3.download_file(BUCKET, o["Key"], str(Path(dest) / n))
        got[n] = o["Size"]
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "train"])
    ap.add_argument("--run", default="extract")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", default="1e-3")
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--max-segments", type=int, default=6)
    ap.add_argument("--start-seg", type=int, default=1)
    ap.add_argument("--skip-weights-upload", action="store_true")
    ap.add_argument("--fetch-features", action="store_true",
                    help="after extract succeeds, download feature tars")
    args = ap.parse_args()

    outdir = ROOT / "models/v3/trackd" / args.run
    for seg in range(args.start_seg, args.max_segments + 1):
        job, op, exp_epoch = submit(seg, args)
        st = wait(job, exp_epoch - 1500)
        print(f"[seg {seg}] {st}", flush=True)
        if st == "STALE":
            continue
        got = fetch(op, outdir / f"seg{seg:02d}")
        print(f"[seg {seg}] fetched {got}", flush=True)
        sf = outdir / f"seg{seg:02d}/status.json"
        if sf.exists():
            s = json.loads(sf.read_text())
            print(f"[seg {seg}] status {s}", flush=True)
            if s.get("phase") == "done":
                if args.mode == "extract" and args.fetch_features:
                    print("fetching feature tars...", flush=True)
                    got = fetch("v3/runs/trackd/feats_", ROOT / "data/v3/features_tars")
                    print(f"feature tars: {got}", flush=True)
                print(f"RUN COMPLETE ({seg} segs) -> {outdir}", flush=True)
                return 0
    print("max segments without done", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
