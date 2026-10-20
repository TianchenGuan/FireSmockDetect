"""Submit the RF-DETR inference-only eval job (green g6e) and collect predictions.

Reuses the iteration-2/3 presigned pattern: code + image lists go up as
tarballs (presigned GET), predictions come back as gzipped JSONL (presigned
PUT) and also persist on /scratch. Weights are read directly from the training
run dir on /scratch -- nothing large crosses the wire.

Usage:
  python cluster/v3/submit_rfdetr_eval.py --eval-name rfdetrL_e48_valtest \
      --splits sdis_val --splits sdis_test [--latency-n 50] [--max-images 0]
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

CODE_FILES = [
    ("rfdetr_infer.py", "cluster/v3/rfdetr_infer.py"),
    ("src/__init__.py", "src/__init__.py"),
    ("src/serve/__init__.py", "src/serve/__init__.py"),
    ("src/serve/detector.py", "src/serve/detector.py"),
    ("src/serve/backend_rfdetr.py", "src/serve/backend_rfdetr.py"),
]


def ada_session():
    p = subprocess.run(
        [ADA, "credentials", "print", "--account", ACCOUNT, "--provider INTERNAL-PROVIDER",
         "--role", "INTERNAL-ADMIN-ROLE", "--format", "json"],
        capture_output=True, text=True, timeout=90, check=True)
    c = json.loads(p.stdout)
    return boto3.Session(aws_access_key_id=c["AccessKeyId"],
                         aws_secret_access_key=c["SecretAccessKey"],
                         aws_session_token=c["SessionToken"]), c.get("Expiration")


def make_tar(entries: list[tuple[str, Path]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, path in entries:
            tf.add(str(path), arcname=name)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-name", required=True)
    ap.add_argument("--splits", action="append", default=None,
                    help="list basenames under data/yolo/lists/local (repeatable)")
    ap.add_argument("--extra-list", action="append", default=None,
                    help="path=name for an extra image list to ship")
    ap.add_argument("--weights", default=None,
                    help="pod path; default = runs_v3/rfdetrL_1280_e48/checkpoint_best_ema.pth")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf-floor", type=float, default=0.02)
    ap.add_argument("--latency-n", type=int, default=50)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--ngpus", type=int, default=8)
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()
    splits = args.splits or ["sdis_val", "sdis_test"]

    sess, cred_exp = ada_session()
    s3 = sess.client("s3", region_name=REGION,
                     endpoint_url=f"https://s3.{REGION}.amazonaws.com",
                     config=Config(signature_version="s3v4",
                                   s3={"addressing_style": "virtual"}))
    exp_epoch = int(datetime.strptime(cred_exp, "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=timezone.utc).timestamp())
    ttl = max(600, exp_epoch - int(time.time()) - 60)

    prefix = f"v3/evals/{args.eval_name}"
    # code tarball (single inference implementation, shared with the devbox)
    s3.put_object(Bucket=BUCKET, Key=f"{prefix}/code.tgz",
                  Body=make_tar([(n, ROOT / p) for n, p in CODE_FILES]))
    # list tarball
    entries = []
    for sp in splits:
        entries.append((f"lists/{sp}.txt", ROOT / "data/yolo/lists/local" / f"{sp}.txt"))
    for spec in (args.extra_list or []):
        path, name = spec.split("=", 1)
        entries.append((f"lists/{name}.txt", Path(path)))
    s3.put_object(Bucket=BUCKET, Key=f"{prefix}/lists.tgz", Body=make_tar(entries))
    s3.upload_file(str(ROOT / "cluster/v3/rfdetr_eval_entry.sh"), BUCKET,
                   f"{prefix}/rfdetr_eval_entry.sh")

    gu = lambda k: s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=ttl)
    pu = lambda k: s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=ttl)

    all_splits = splits + [s.split("=", 1)[1] for s in (args.extra_list or [])]
    envs = {
        "EVAL_NAME": args.eval_name,
        "SPLITS": " ".join(all_splits),
        "IMGSZ": str(args.imgsz), "CONF_FLOOR": str(args.conf_floor),
        "LATENCY_N": str(args.latency_n), "LATENCY_LIST": all_splits[0],
        "NGPUS": str(args.ngpus),
        "URL_ENTRY": gu(f"{prefix}/rfdetr_eval_entry.sh"),
        "URL_CODE": gu(f"{prefix}/code.tgz"),
        "URL_LISTS": gu(f"{prefix}/lists.tgz"),
        "URL_LATENCY": pu(f"{prefix}/latency_gpu.json"),
        "URL_LOG": pu(f"{prefix}/shardlogs.txt"),
    }
    if args.weights:
        envs["WEIGHTS"] = args.weights
    if args.max_images:
        envs["MAX_IMAGES"] = str(args.max_images)
    for sp in all_splits:
        envs[f"URL_PREDS_{sp.upper().replace('-', '_')}"] = pu(f"{prefix}/{sp}_preds.jsonl.gz")

    env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\"" for k, v in envs.items())
    tag = ("re" + args.eval_name.replace("_", "").replace("-", ""))[:16].lower()
    ydir = ROOT / "cluster" / "generated"
    ydir.mkdir(exist_ok=True)
    yfile = ydir / f"v3_eval_{args.eval_name}.yaml"
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
    print("job:", job, "| ttl", ttl, "s", flush=True)
    if args.no_wait or not job:
        return 0

    t0 = time.time()
    last = ""
    while time.time() - t0 < 7200:
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

    dest = ROOT / "models" / "v3" / "evals" / args.eval_name
    dest.mkdir(parents=True, exist_ok=True)
    sess2, _ = ada_session()
    s3b = sess2.client("s3", region_name=REGION)
    for o in s3b.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", []):
        name = o["Key"].split("/")[-1]
        if name in ("code.tgz", "lists.tgz", "rfdetr_eval_entry.sh"):
            continue
        s3b.download_file(BUCKET, o["Key"], str(dest / name))
        print("got", name, o["Size"], flush=True)
    print("->", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
