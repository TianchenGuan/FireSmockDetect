"""Track B/C orchestrator: segmented resumable DINOv3-detection jobs on
green g6e (v2 pattern: per segment mint ada creds -> upload code/data ->
presign URLs -> kraken submit -> poll -> fetch artifacts -> resubmit until
DONE). Code is fetched per-job (src/v3 tarball) so iterations need no image
rebuild.

Examples:
  # stand-in overfit-16 plumbing check (dinov2-large staged on /scratch):
  python cluster/v3/orchestrate_dino.py --run standin_overfit16 \
      --model-dir /scratch/scratch/tianrgua-sandbox/models/dinov2-large \
      --width 1288 --height 728 --mode overfit

  # B1 (after scripts/v3/stage_dinov3.sh):
  bash cluster/v3/launch/B1_vitl16.sh
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
BUCKET = "smoke-detect-data-000000000001-use1"
REGION = "us-east-1"
QUEUE = "green-hyperpod-g6e.48xlarge-us-east-1-tas"
PROJECT = "green"
KRAKEN = "kraken"
CODE_FILES = ["det_modeling.py", "det_data.py", "train_det.py", "precompute_feats.py"]

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


def code_tgz() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in CODE_FILES:
            tf.add(ROOT / "src/v3" / f, arcname=f)
    return buf.getvalue()


def submit(seg, args, resume_key=None):
    sess, exp = ada_session()
    s3 = s3c(sess)
    exp_epoch = int(datetime.strptime(exp, "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=timezone.utc).timestamp())
    ttl = max(600, exp_epoch - int(time.time()) - 60)
    kp = f"v3/jobs/{args.run}"
    s3.put_object(Bucket=BUCKET, Key=f"{kp}/code.tgz", Body=code_tgz())
    s3.upload_file(str(ROOT / "cluster/v3/dino_entry.sh"), BUCKET, f"{kp}/entry.sh")
    s3.upload_file(args.train_json, BUCKET, f"{kp}/instances_train.json")
    if args.val_json:
        s3.upload_file(args.val_json, BUCKET, f"{kp}/instances_val.json")

    def get(key):
        return s3.generate_presigned_url("get_object",
            Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

    def put(key):
        return s3.generate_presigned_url("put_object",
            Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

    op = f"v3/runs/{args.run}/seg{seg:02d}"
    envs = {
        "URL_ENTRY": get(f"{kp}/entry.sh"), "URL_CODE": get(f"{kp}/code.tgz"),
        "URL_TRAIN_JSON": get(f"{kp}/instances_train.json"),
        "RUN_NAME": args.run, "MODE": args.job_mode,
        "MODEL_DIR": args.model_dir or "", "RANDOM_SPEC": args.random_spec or "",
        "FEAT_ROOT": args.feat_root or "", "EMBED_DIM": args.embed_dim or "",
        "FEAT_OUT": args.feat_out or "",
        "UNFREEZE_TOP": str(args.unfreeze_top),
        "WIDTH": str(args.width), "HEIGHT": str(args.height),
        "EPOCHS": str(args.epochs), "MAX_STEPS": str(args.max_steps),
        "BS": str(args.bs), "LR": args.lr, "WD": args.wd,
        "WARMUP": str(args.warmup), "WORKERS": str(args.workers),
        "OVERFIT_N": str(args.overfit_n), "SAVE_STEPS": str(args.save_steps),
        "EVAL_MAX": str(args.eval_max), "EVAL_EVERY": str(args.eval_every),
        "NGPUS": str(args.ngpus),
        "EXTRA_ARGS": args.extra_args,
        "URL_EXPIRY_EPOCH": str(exp_epoch),
        "URL_STATUS": put(f"{op}/status.json"),
        "URL_CONSOLE": put(f"{op}/console_tail.txt"),
        "URL_METRICS": put(f"{op}/metrics.jsonl"),
        "URL_LAST": put(f"{op}/last.pt"),
        "URL_BEST": put(f"{op}/best.pt"),
    }
    if args.val_json:
        envs["URL_VAL_JSON"] = get(f"{kp}/instances_val.json")
    if resume_key:
        envs["URL_RESUME"] = get(resume_key)
    env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\""
                         for k, v in envs.items())
    yfile = ROOT / "cluster/generated" / f"dino_{args.run}_s{seg:02d}.yaml"
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
    print(f"[seg {seg}] submitted {job} ttl={ttl}s resume={'y' if resume_key else 'n'}",
          flush=True)
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
    ap.add_argument("--run", required=True)
    ap.add_argument("--job-tag", default=None)
    ap.add_argument("--model-dir", default=None,
                    help="/scratch HF snapshot dir (staged backbone)")
    ap.add_argument("--random-spec", default=None)
    ap.add_argument("--feat-root", default=None, help="C1: cached-feature root")
    ap.add_argument("--embed-dim", default=None)
    ap.add_argument("--feat-out", default=None, help="precompute output root")
    ap.add_argument("--job-mode", default="train", choices=["train", "precompute"])
    ap.add_argument("--mode", default="train",
                    choices=["overfit", "train"],
                    help="overfit: 16-image sanity (300 steps, eval on train)")
    ap.add_argument("--unfreeze-top", type=int, default=0)
    ap.add_argument("--train-json", default=str(ROOT / "data/v3/coco/instances_train.json"))
    ap.add_argument("--val-json", default=str(ROOT / "data/v3/coco/instances_val.json"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=736)
    ap.add_argument("--epochs", type=float, default=12)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", default="1e-4")
    ap.add_argument("--wd", default="1e-4")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overfit-n", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-max", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=1,
                    help="eval every N epochs (overfit mode: N=100 -> every "
                         "100 passes over the tiny set)")
    ap.add_argument("--ngpus", type=int, default=8)
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--image-uri",
        default="000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-dino3:v1")
    ap.add_argument("--max-segments", type=int, default=24)
    ap.add_argument("--start-seg", type=int, default=1)
    ap.add_argument("--resume-key", default=None)
    args = ap.parse_args()
    if args.mode == "overfit":
        args.overfit_n = args.overfit_n or 16
        args.max_steps = args.max_steps if args.max_steps > 0 else 300
        args.val_json = None
        args.warmup = 50
        args.save_steps = 100
        args.eval_every = 100  # overfit-16: 1 step/epoch -> eval every ~100 steps
    if not args.job_tag:
        args.job_tag = ("d3" + args.run.replace("_", ""))[:12].lower()
    args.job_tag = args.job_tag.lower()

    outdir = ROOT / "models/v3/runs" / args.run
    resume_key = args.resume_key
    for seg in range(args.start_seg, args.max_segments + 1):
        job, op, exp_epoch = submit(seg, args, resume_key)
        st = wait(job, exp_epoch - 1500)
        print(f"[seg {seg}] {st}", flush=True)
        if st == "STALE":
            continue
        got = fetch(op, outdir / f"seg{seg:02d}")
        print(f"[seg {seg}] fetched {got}", flush=True)
        if "last.pt" in got:
            resume_key = f"{op}/last.pt"
        sf = outdir / f"seg{seg:02d}/status.json"
        if sf.exists():
            try:
                s = json.loads(sf.read_text())
            except json.JSONDecodeError:
                # salvage the phase field rather than dying post-success
                txt = sf.read_text()
                s = {"phase": txt.split('"phase":"', 1)[1].split('"', 1)[0]
                     if '"phase":"' in txt else "unparseable", "raw": txt[:200]}
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
