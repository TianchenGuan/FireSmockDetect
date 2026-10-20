"""Submit a v4 YOLO batch val-inference job and fetch preds.

Ships local checkpoints to S3, presigns GETs for the pod, presigns PUTs for
the preds jsonl.gz outputs, submits one 1-node job (registry-reserved by the
CALLER via run wrapper or manually), waits, downloads, gunzips, and remaps
pod image paths back to local paths so src/eval/eval_detection.py
--preds-jsonl works against local lists.

Usage:
  python cluster/v4/submit_yolo_eval.py --tag v4evalr1 \
    --ckpt v4a=models/trained/expV4A_ctrl_s0/seg05/best.pt \
    --ckpt v4b=models/v4/runs/expV4B_gonly_s0/seg01/best.pt \
    --list sdisval=/mnt/local/smoke_data/lists/pod/sdis_val.txt \
    --rootmap /mnt/local/smoke_data/pyro_sdis=/local/home/tianrgua/cvproject/data/yolo/pyro_sdis \
    [--data-tar name=localtar] --out models/v4/evals
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster"))
import orchestrate_training as ot  # noqa: E402

BUCKET = "smoke-detect-data-000000000001-use1"
REGION = "us-east-1"
QUEUE = "green-hyperpod-g6e.48xlarge-us-east-1-tas"
IMAGE = "000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-detect:v6"

YAML_TMPL = """jobNamePrefix: tianrgua-{job}
jobPriority: LOW
containers:
  - imageUri: {image_uri}
    command: /bin/bash
    args:
      - -c
      - curl -sf --max-time 120 -o /tmp/e.sh "$URL_ENTRY" && bash /tmp/e.sh; ec=$?; [ -f /tmp/e.sh ] || ec=96; exit $ec
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
  role: inference
  runType: dev
"""


def env_var_for(base: str) -> str:
    return "URL_PREDS_" + re.sub(r"[^A-Z0-9_]", "", base.upper().replace("-", "_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="name=localpath")
    ap.add_argument("--list", action="append", required=True, dest="lists",
                    help="name=podlistpath")
    ap.add_argument("--data-tar", action="append", default=[])
    ap.add_argument("--rootmap", action="append", default=[],
                    help="podprefix=localprefix (applied to preds image paths)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", default="models/v4/evals")
    args = ap.parse_args()

    sess, cred_exp = ot.ada_session()
    s3 = ot.s3_client(sess, REGION)
    exp_epoch = int(datetime.strptime(cred_exp, "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=timezone.utc).timestamp())
    ttl = max(600, exp_epoch - int(time.time()) - 60)
    kp = f"v4/evals/{args.tag}"

    def put_url(key):
        return s3.generate_presigned_url("put_object",
            Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

    def get_url(key):
        return s3.generate_presigned_url("get_object",
            Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

    entry_key = f"{kp}/entry.sh"
    s3.upload_file(str(ROOT / "cluster/v4/yolo_eval_entry.sh"), BUCKET, entry_key)

    ckpts = {}
    for spec in args.ckpt:
        name, local = spec.split("=", 1)
        key = f"{kp}/ckpt_{name}.pt"
        s3.upload_file(str(ROOT / local), BUCKET, key)
        ckpts[name] = get_url(key)

    envs = {
        "URL_ENTRY": get_url(entry_key),
        "CKPT_URLS": ",".join(f"{k}={v}" for k, v in ckpts.items()),
        "LISTS": ",".join(args.lists),
        "IMGSZ": str(args.imgsz),
        "URL_LOG": put_url(f"{kp}/entry.log"),
    }
    if args.data_tar:
        specs = []
        for spec in args.data_tar:
            name, local = spec.split("=", 1)
            base = name.split("@", 1)[0]          # "name@/pod/root" -> per-tar root
            key = f"v4/data/{base}.tar"
            marker = ROOT / "cluster/v4" / f".uploaded_{base}"
            if not marker.exists():
                s3.upload_file(str(ROOT / local), BUCKET, key)
                marker.write_text(datetime.now(timezone.utc).isoformat())
            specs.append(f"{name}={get_url(key)}")
        envs["DATA_TARS"] = ",".join(specs)

    expected = []
    for cname in ckpts:
        for lspec in args.lists:
            lname = lspec.split("=", 1)[0]
            base = f"{cname}__{lname}"
            envs[env_var_for(base)] = put_url(f"{kp}/{base}.jsonl.gz")
            expected.append(base)

    env_yaml = "\n".join(
        f"      - name: {k}\n        value: \"{v}\"" for k, v in envs.items())
    yfile = ROOT / "cluster/generated" / f"yoloeval_{args.tag}.yaml"
    yfile.write_text(YAML_TMPL.format(job=args.tag, image_uri=IMAGE,
                                      envs=env_yaml, queue=QUEUE))
    rc, out = ot.kraken_jobs("green", ["create", "-i", str(yfile)])
    job = None
    for tok in out.split():
        if tok.startswith(f"tianrgua-{args.tag}"):
            job = tok.strip().strip("'\",.")
            break
    print(f"submitted {job}")
    st = ot.wait_terminal("green", job, stale_deadline=exp_epoch - 1500)
    print(f"terminal: {st}")

    outdir = ROOT / args.out / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    maps = [m.split("=", 1) for m in args.rootmap]
    n_ok = 0
    for base in expected:
        key = f"{kp}/{base}.jsonl.gz"
        gz = outdir / f"{base}.jsonl.gz"
        try:
            s3.download_file(BUCKET, key, str(gz))
        except Exception as e:
            print(f"MISSING {base}: {e}")
            continue
        text = gzip.decompress(gz.read_bytes()).decode()
        for pod, local in maps:
            text = text.replace(f'"image": "{pod}', f'"image": "{local}')
        (outdir / f"{base}.jsonl").write_text(text)
        n_ok += 1
        print(f"fetched {base}.jsonl")
    try:
        s3.download_file(BUCKET, f"{kp}/entry.log", str(outdir / "entry.log"))
        print(f"pod log -> {outdir / 'entry.log'}")
    except Exception as e:
        print(f"no pod log: {e}")
    print(f"{n_ok}/{len(expected)} preds files fetched -> {outdir}")
    return 0 if n_ok == len(expected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
