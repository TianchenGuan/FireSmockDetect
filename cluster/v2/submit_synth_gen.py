"""Submit a synthetic-smoke generation segment on green g6e.

Uploads entry/code/spec/backgrounds-tar to S3, presigns everything (regional
endpoint — pods only reach S3 via the regional VPC endpoint), renders a
kraken YAML (jobPriority LOW, 1 node, taxonomy data-gen/x-image-edit/general/
dev, NO single quotes anywhere in args) and submits. Optionally polls to a
terminal state and downloads results.tar + status.json.

Segmenting: presigned URLs die with the ada creds (~55 min). Each segment
takes a [--jobs-from, --jobs-to) slice of data/v2/synth/spec.jsonl and a
--budget-s generation cutoff; gen_synth_pod.py stops cleanly at the budget
and the entry uploads whatever finished.

Usage:
  python cluster/v2/submit_synth_gen.py --name synthseg1 --jobs-from 0 \
      --jobs-to 48 --budget-s 1900
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
ADA = "ada"
KRAKEN = "kraken"
IMAGE = "000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-diffusion:v1"
MODEL_DIR = "/scratch/scratch/tianrgua-sandbox/models/qwen-image-edit-2511"

YAML_TMPL = """jobNamePrefix: tianrgua-{tag}
jobPriority: LOW
containers:
  - imageUri: {image}
    command: /bin/bash
    args:
      - -c
      - >-
        curl -sf --max-time 120 -o /tmp/entry.sh "$URL_ENTRY" && bash /tmp/entry.sh
    environmentVariables:
{envs}
resourceConfig:
  nodeCount: 1
  queue: green-hyperpod-g6e.48xlarge-us-east-1-tas
storageConfig:
  enableHomeWriteOnly: true
restartConfig:
  maxRestarts: 1
  restartPolicy: All
workloadTaxonomyConfig:
  workloadType: data-gen
  subtype: x-image-edit
  role: general
  runType: dev
"""


def ada_session():
    p = subprocess.run(
        [ADA, "credentials", "print", "--account", "000000000001",
         "--provider INTERNAL-PROVIDER", "--role", "INTERNAL-ADMIN-ROLE",
         "--format", "json"],
        capture_output=True, text=True, timeout=90, check=True)
    c = json.loads(p.stdout)
    sess = boto3.Session(aws_access_key_id=c["AccessKeyId"],
                         aws_secret_access_key=c["SecretAccessKey"],
                         aws_session_token=c["SessionToken"])
    exp = int(datetime.strptime(c["Expiration"], "%Y-%m-%dT%H:%M:%SZ")
              .replace(tzinfo=timezone.utc).timestamp())
    return sess, exp


def kraken(args_list):
    p = subprocess.run([KRAKEN, "-p", "green", "jobs"] + args_list,
                       capture_output=True, text=True, timeout=180)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--jobs-from", type=int, default=0)
    ap.add_argument("--jobs-to", type=int, default=100000)
    ap.add_argument("--budget-s", type=int, default=2100)
    ap.add_argument("--submit-only", action="store_true")
    ap.add_argument("--done-from-segments", action="store_true",
                    help="skip job_ids whose PNG already exists in any "
                         "fetched data/v2/synth/segments/*/results.tar")
    args = ap.parse_args()

    done_ids = set()
    if args.done_from_segments:
        import tarfile
        for t in sorted((ROOT / "data/v2/synth/segments").glob(
                "*/results.tar")):
            with tarfile.open(t) as tf:
                for n in tf.getnames():
                    b = Path(n).name
                    if b.startswith("bg") and b.endswith(".png"):
                        done_ids.add(b[:-4])
        print(f"done so far: {len(done_ids)} job_ids", flush=True)

    sess, exp = ada_session()
    ttl = max(600, exp - int(time.time()) - 60)
    s3 = sess.client("s3", region_name="us-east-1",
                     endpoint_url="https://s3.us-east-1.amazonaws.com",
                     config=Config(signature_version="s3v4",
                                   s3={"addressing_style": "virtual"}))
    kp = f"v2/synthjobs/{args.name}"
    s3.upload_file(str(ROOT / "cluster/v2/synth_gen_entry.sh"), BUCKET,
                   f"{kp}/entry.sh")
    s3.upload_file(str(ROOT / "cluster/v2/gen_synth_pod.py"), BUCKET,
                   f"{kp}/gen_synth_pod.py")
    s3.upload_file(str(ROOT / "data/v2/synth/spec.jsonl"), BUCKET,
                   f"{kp}/spec.jsonl")
    bg_tar = ROOT / "data/v2/synth/backgrounds.tar"
    bg_key = "v2/synthjobs/backgrounds.tar"
    try:
        s3.head_object(Bucket=BUCKET, Key=bg_key)
    except Exception:  # noqa: BLE001
        s3.upload_file(str(bg_tar), BUCKET, bg_key)
        print("uploaded backgrounds.tar")
    g = lambda k: s3.generate_presigned_url(  # noqa: E731
        "get_object", Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=ttl)
    pu = lambda k: s3.generate_presigned_url(  # noqa: E731
        "put_object", Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=ttl)
    envs = {
        "URL_ENTRY": g(f"{kp}/entry.sh"),
        "URL_CODE": g(f"{kp}/gen_synth_pod.py"),
        "URL_SPEC": g(f"{kp}/spec.jsonl"),
        "URL_BG": g(bg_key),
        "MODEL_DIR": MODEL_DIR,
        "JOBS_FROM": str(args.jobs_from),
        "JOBS_TO": str(args.jobs_to),
        "BUDGET_S": str(args.budget_s),
        "URL_RESULTS": pu(f"v2/synth/{args.name}/results.tar"),
        "URL_STATUS": pu(f"v2/synth/{args.name}/status.json"),
        "URL_LOG": pu(f"v2/synth/{args.name}/log0.txt"),
    }
    if done_ids:
        s3.put_object(Bucket=BUCKET, Key=f"{kp}/done.txt",
                      Body=("\n".join(sorted(done_ids)) + "\n").encode())
        envs["URL_DONE"] = g(f"{kp}/done.txt")
    env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\""
                         for k, v in envs.items())
    tag = args.name.replace("_", "")[:14].lower()
    yf = ROOT / "cluster/v2/generated" / f"synthgen_{args.name}.yaml"
    yf.write_text(YAML_TMPL.format(tag=tag, image=IMAGE, envs=env_yaml))
    rc, out = kraken(["create", "-i", str(yf)])
    if rc != 0:
        raise SystemExit(f"submit failed: {out}")
    job_name = None
    for tok in out.replace('"', " ").replace(",", " ").split():
        tok = tok.strip(".:;")
        if tok.startswith(f"tianrgua-{tag}"):
            job_name = tok
    print(f"submitted {job_name} (results -> s3://{BUCKET}/v2/synth/"
          f"{args.name}/)", flush=True)
    if not job_name:
        raise SystemExit("could not parse job name:\n" + out)
    if args.submit_only:
        return 0

    t0, last = time.time(), ""
    while time.time() - t0 < 7200:
        rc, out = kraken(["list", "-r", "tianrgua"])
        state = ""
        for line in out.splitlines():
            if line.startswith(job_name):
                state = line.split()[1]
        if state != last:
            print(f"  [{int(time.time() - t0)}s] {state}", flush=True)
            last = state
        if state.split(":")[0] in ("SUCCEEDED", "FAILED", "STOPPED"):
            rc, desc = kraken(["describe", "--job-name", job_name])
            info = [l for l in desc.splitlines() if "exit code" in l]
            print("terminal:", state, "|",
                  info[0].strip() if info else "", flush=True)
            break
        if (time.time() > exp - 900
                and state.split(":")[0] in ("PENDING", "QUEUED", "CREATED", "")):
            print("still pending near URL expiry -> stop + resubmit",
                  flush=True)
            kraken(["stop", "--job-name", job_name])
            return 2
        time.sleep(45)

    sess2, _ = ada_session()
    s3v = sess2.client("s3", region_name="us-east-1")
    dst = ROOT / "data/v2/synth/segments" / args.name
    dst.mkdir(parents=True, exist_ok=True)
    for fn in ("results.tar", "status.json", "log0.txt"):
        try:
            s3v.download_file(BUCKET, f"v2/synth/{args.name}/{fn}",
                              str(dst / fn))
            print("fetched", dst / fn, flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"missing {fn}: {e}", flush=True)
    st = dst / "status.json"
    if st.exists():
        print("status:", st.read_text(), flush=True)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
