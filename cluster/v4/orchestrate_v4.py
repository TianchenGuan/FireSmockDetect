"""v4 YOLO transfer-matrix orchestrator.

Fork of cluster/orchestrate_training.py (reuses its helpers via import) with:
  - entry script fetched at job start via presigned URL_ENTRY (edit
    cluster/v4/train_entry_v4.sh locally; no image rebuild)
  - --data-tar name=localpath (repeatable): uploaded to s3 once, presigned
    per segment; pod stages to /scratch with .done markers
  - --data-yaml-file: local yaml shipped inline via DATA_YAML_B64
  - --seed
Same segmented resume, exit-code channel, results.csv completion logic.

Usage:
  python cluster/v4/orchestrate_v4.py --exp expV4C_mix_s0 --job-tag v4cmixs0 \
    --data-yaml-file data/v4/yolo/yamls/pod/v4C.yaml --epochs 38 \
    --base-weights yolo11m.pt --seed 0 \
    --data-tar v4G=data/v4/stage/v4G.tar --data-tar v4T=data/v4/stage/v4T.tar
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster"))
import orchestrate_training as ot  # noqa: E402

YAML_TMPL = """jobNamePrefix: tianrgua-{job}
jobPriority: LOW
containers:
  - imageUri: {image_uri}
    command: /bin/bash
    args:
      - -c
      - curl -sf --max-time 120 -o /tmp/entry_v4.sh "$URL_ENTRY" && bash /tmp/entry_v4.sh; ec=$?; [ -f /tmp/entry_v4.sh ] || ec=96; exit $ec
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


def upload_once(s3, bucket, key, localpath):
    s3.upload_file(str(localpath), bucket, key)


def submit_segment(seg, args, resume_key=None):
    sess, cred_exp = ot.ada_session()
    s3 = ot.s3_client(sess, args.s3_region)
    exp_epoch = int(datetime.strptime(cred_exp, "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=timezone.utc).timestamp())
    ttl = max(600, exp_epoch - int(time.time()) - 60)
    prefix = f"runs/{args.exp}/seg{seg:02d}"

    # entry script: re-upload every segment so local edits propagate
    entry_key = f"v4/entry/{args.exp}_entry_v4.sh"
    upload_once(s3, args.bucket, entry_key, ROOT / "cluster/v4/train_entry_v4.sh")

    def put_url(key):
        return s3.generate_presigned_url("put_object",
            Params={"Bucket": args.bucket, "Key": key}, ExpiresIn=ttl)

    def get_url(key):
        return s3.generate_presigned_url("get_object",
            Params={"Bucket": args.bucket, "Key": key}, ExpiresIn=ttl)

    envs = {
        "EXP_NAME": args.exp,
        "EPOCHS": str(args.epochs), "IMGSZ": str(args.imgsz),
        "BATCH": str(args.batch), "DEVICES": args.devices,
        "WORKERS": str(args.workers), "BASE_WEIGHTS": args.base_weights,
        "SEED": str(args.seed),
        "URL_EXPIRY_EPOCH": str(exp_epoch),
        "URL_ENTRY": get_url(entry_key),
        "URL_LAST": put_url(f"{prefix}/last.pt"),
        "URL_BEST": put_url(f"{prefix}/best.pt"),
        "URL_RESULTS": put_url(f"{prefix}/results.csv"),
        "URL_ARGS": put_url(f"{prefix}/args.yaml"),
        "URL_STATUS": put_url(f"{prefix}/status.json"),
        "URL_CONSOLE": put_url(f"{prefix}/console_tail.txt"),
    }
    if args.data_yaml_file:
        envs["DATA_YAML_B64"] = base64.b64encode(
            Path(args.data_yaml_file).read_bytes()).decode()
    else:
        envs["DATA_YAML"] = args.data_yaml

    if resume_key:
        envs["URL_RESUME"] = get_url(resume_key)
    if args.init_key:
        envs["URL_INIT"] = get_url(args.init_key)
    if args.pod_data_root:
        # v5b: stage --data-tar archives under a different /scratch root
        # (train_entry_v4.sh honours $V4_DATA_ROOT; default unchanged = v4data)
        envs["V4_DATA_ROOT"] = args.pod_data_root

    if args.data_tar:
        specs = []
        for spec in args.data_tar:
            name, local = spec.split("=", 1)
            key = f"v4/data/{name}.tar"
            marker = ROOT / "cluster/v4" / f".uploaded_{name}"
            if not marker.exists():
                print(f"uploading {local} -> s3://{args.bucket}/{key} ...",
                      flush=True)
                upload_once(s3, args.bucket, key, ROOT / local)
                marker.write_text(datetime.now(timezone.utc).isoformat())
            specs.append(f"{name}={get_url(key)}")
        envs["DATA_TARS"] = ",".join(specs)

    env_yaml = "\n".join(
        f"      - name: {k}\n        value: \"{v}\"" for k, v in envs.items())
    ydir = ROOT / "cluster" / "generated"
    ydir.mkdir(exist_ok=True)
    yfile = ydir / f"{args.exp}_seg{seg:02d}.yaml"
    yfile.write_text(YAML_TMPL.format(job=f"{args.job_tag}s{seg}",
                                      image_uri=args.image_uri,
                                      envs=env_yaml, queue=args.queue))
    rc, out = ot.kraken_jobs(args.project, ["create", "-i", str(yfile)])
    job = None
    for tok in out.split():
        if tok.startswith(f"tianrgua-{args.job_tag}s{seg}"):
            job = tok.strip().strip("'\",.")
            break
    print(f"[seg {seg}] submitted {job} (url ttl {ttl}s, "
          f"resume={'yes' if resume_key else 'no'})", flush=True)
    return job, prefix, exp_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--job-tag", default=None)
    ap.add_argument("--data-yaml", default=None)
    ap.add_argument("--data-yaml-file", default=None,
                    help="local yaml shipped inline (DATA_YAML_B64)")
    ap.add_argument("--data-tar", action="append", default=[],
                    help="name=localpath, staged to /scratch on pods")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-weights", default="yolo11m.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="green")
    ap.add_argument("--queue", default="green-hyperpod-g6e.48xlarge-us-east-1-tas")
    ap.add_argument("--bucket", default="smoke-detect-data-000000000001-use1")
    ap.add_argument("--s3-region", default="us-east-1")
    ap.add_argument("--image-uri",
        default="000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-detect:v6")
    ap.add_argument("--max-segments", type=int, default=14)
    ap.add_argument("--start-seg", type=int, default=1)
    ap.add_argument("--resume-key", default=None)
    ap.add_argument("--init-key", default=None,
                    help="S3 key of a checkpoint to initialize from "
                         "(fresh training; curriculum stage-2)")
    ap.add_argument("--pod-data-root", default=None,
                    help="pod-side root where --data-tar archives are extracted "
                         "(sets V4_DATA_ROOT; default /scratch/.../smoke/v4data)")
    args = ap.parse_args()
    if not args.data_yaml and not args.data_yaml_file:
        ap.error("need --data-yaml or --data-yaml-file")
    if not args.job_tag:
        args.job_tag = args.exp.replace("_", "")[:12]
    args.job_tag = args.job_tag.lower()

    outdir = ROOT / "models" / "v4" / "runs" / args.exp
    resume_key = args.resume_key
    for seg in range(args.start_seg, args.max_segments + 1):
        job, prefix, exp_epoch = submit_segment(seg, args, resume_key)
        if not job:
            print("could not parse job name"); return 2
        desc = ot.wait_terminal(args.project, job,
                                stale_deadline=exp_epoch - 1500)
        print(f"[seg {seg}] terminal: {desc}", flush=True)
        if desc == "STALE":
            continue
        got = ot.fetch(prefix, outdir / f"seg{seg:02d}", args)
        print(f"[seg {seg}] fetched: {got}", flush=True)
        if "last.pt" in got:
            resume_key = f"{prefix}/last.pt"
        st_file = outdir / f"seg{seg:02d}" / "status.json"
        if st_file.exists():
            st = json.loads(st_file.read_text())
            print(f"[seg {seg}] status: {st}", flush=True)
            if st.get("phase") == "done":
                print(f"TRAINING COMPLETE after {seg} segment(s) -> {outdir}",
                      flush=True)
                return 0
            if st.get("phase", "").startswith(("v4data_stage_failed",
                                               "yaml_decode_failed",
                                               "v4data_root_unwritable")):
                print(f"FATAL pod-side staging: {st['phase']} — aborting")
                return 3
        else:
            print(f"[seg {seg}] WARNING: no status.json — check exit code",
                  flush=True)
    print("max segments reached without done — investigate", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
