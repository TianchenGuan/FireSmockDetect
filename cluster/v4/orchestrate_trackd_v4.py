"""Track D v4 orchestrator (green g6e, presigned regional S3 + /scratch, exit-code
observability). Same pattern as cluster/v3/orchestrate_trackd.py, extended with:
  - staging the 16 GB v4 frame tar onto /scratch (idempotent, marker file),
  - restoring the iteration-3 feature tars if /scratch was evicted,
  - training several head variants inside one job (RUN_SPECS).

Reservation is NOT done here: every submission must be reserved first via
  python cluster/v4/registry.py reserve --job res-<tag> --nodes 1 --track temporal
and released afterwards (see reports/v4/EARLY_WARNING_V4.md).

Usage:
  python cluster/v4/orchestrate_trackd_v4.py extract --run xv4
  python cluster/v4/orchestrate_trackd_v4.py train --run heads_v4 \
      --run-specs "v4_temporal_s0:0:- v4_temporal_s1:1:- v4_t1only_s0:0:--t1-only v4_t1only_s1:1:--t1-only"

argparse gotcha inherited from v3: pass extra flags as --extra-args=--flag
(space form eats the value).
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
from boto3.s3.transfer import TransferConfig

ROOT = Path(__file__).resolve().parents[2]
BUCKET = "smoke-detect-data-000000000001-use1"
REGION = "us-east-1"
QUEUE = "green-hyperpod-g6e.48xlarge-us-east-1-tas"
PROJECT = "green"
KRAKEN = "kraken"
ADA = "ada"
IMAGE = "000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-detect:v6"
WEIGHTS = ROOT / "models/trained/expE_mix_y11m_1280_v2/seg05/best.pt"
KP = "v4/jobs/trackd"
XFER = TransferConfig(multipart_threshold=8 << 20, max_concurrency=24,
                      multipart_chunksize=32 << 20)

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
    p = subprocess.run([ADA, "credentials", "print", "--account", "000000000001",
                        "--provider INTERNAL-PROVIDER", "--role",
                        "INTERNAL-ADMIN-ROLE", "--format", "json"],
                       capture_output=True, text=True, timeout=90, check=True)
    c = json.loads(p.stdout)
    return (boto3.Session(aws_access_key_id=c["AccessKeyId"],
                          aws_secret_access_key=c["SecretAccessKey"],
                          aws_session_token=c["SessionToken"]), c["Expiration"])


def s3c(sess):
    return sess.client("s3", region_name=REGION,
                       endpoint_url="https://s3.%s.amazonaws.com" % REGION,
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

    up = {
        KP + "/entry.sh": ROOT / "cluster/v4/trackd_v4_entry.sh",
        KP + "/trackd_features.py": ROOT / "src/v3/trackd_features.py",
        KP + "/trackd_temporal.py": ROOT / "src/v3/trackd_temporal.py",
        KP + "/seq_train_ext.json":
            ROOT / "data/v4/sequences/trackd/sequences_v4_train_ext.json",
        KP + "/seq_val_replay.json":
            ROOT / "data/v4/sequences/trackd/sequences_v4_val_replay.json",
    }
    if not args.skip_weights_upload:
        up[KP + "/expE_best.pt"] = WEIGHTS
    big = {}
    if not args.skip_frames_upload:
        big[KP + "/trackd_v4.tar"] = ROOT / "data/v4/stage/trackd_v4.tar"
    if not args.skip_v3_features_upload:
        big[KP + "/feats_v3_train.tgz"] = ROOT / "data/v3/features_tars/feats_train.tgz"
        big[KP + "/feats_v3_val.tgz"] = ROOT / "data/v3/features_tars/feats_val.tgz"
    for key, path in up.items():
        s3.upload_file(str(path), BUCKET, key)
    for key, path in big.items():
        t0 = time.time()
        s3.upload_file(str(path), BUCKET, key, Config=XFER)
        mb = path.stat().st_size / 1e6
        print("uploaded %s %.0f MB in %.0fs (%.0f MB/s)"
              % (key, mb, time.time() - t0, mb / max(time.time() - t0, 1e-9)),
              flush=True)

    def get(key):
        return s3.generate_presigned_url("get_object",
            Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

    def put(key):
        return s3.generate_presigned_url("put_object",
            Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

    op = "v4/runs/trackd/%s/seg%02d" % (args.run, seg)
    envs = {
        "MODE": args.mode, "RUN_NAME": args.run,
        "URL_ENTRY": get(KP + "/entry.sh"),
        "URL_FEAT_CODE": get(KP + "/trackd_features.py"),
        "URL_HEAD_CODE": get(KP + "/trackd_temporal.py"),
        "URL_WEIGHTS": get(KP + "/expE_best.pt"),
        "URL_SEQ_TRAIN_EXT": get(KP + "/seq_train_ext.json"),
        "URL_SEQ_VAL_REPLAY": get(KP + "/seq_val_replay.json"),
        "URL_FRAMES_TAR": get(KP + "/trackd_v4.tar"),
        "URL_FEATS_V3_TRAIN": get(KP + "/feats_v3_train.tgz"),
        "URL_FEATS_V3_VAL": get(KP + "/feats_v3_val.tgz"),
        "URL_EXPIRY_EPOCH": str(exp_epoch),
        "URL_STATUS": put(op + "/status.json"),
        "URL_CONSOLE": put(op + "/console_tail.txt"),
        "URL_ENTRYLOG": put(op + "/entry.log"),
        "EPOCHS": str(args.epochs), "BS": str(args.bs), "LR": args.lr,
        "WORKERS": str(args.workers),
    }
    if args.mode == "extract":
        envs["URL_FEATS_V4"] = put("v4/runs/trackd/feats_v4.tgz")
        envs["EXPECT_TRAIN_EXT"] = str(args.expect_train_ext)
        envs["EXPECT_VAL_REPLAY"] = str(args.expect_val_replay)
    else:
        envs["URL_HEADS"] = put(op + "/heads.tgz")
        envs["RUN_SPECS"] = args.run_specs
        envs["EXPECT_RUNS"] = str(len(args.run_specs.split()))
        if args.ausmoke_keep:
            envs["AUSMOKE_KEEP"] = str(args.ausmoke_keep)

    env_yaml = "\n".join('      - name: %s\n        value: "%s"' % (k, v)
                         for k, v in envs.items())
    tag = (args.mode[:3] + args.run.replace("_", ""))[:12].lower()
    yfile = ROOT / "cluster/generated" / ("trackdv4_%s_%s_s%02d.yaml"
                                         % (args.mode, args.run, seg))
    yfile.parent.mkdir(parents=True, exist_ok=True)
    yfile.write_text(YAML_TMPL.format(job="%ss%d" % (tag, seg), image_uri=IMAGE,
                                      envs=env_yaml, queue=QUEUE))
    rc, out = kraken(["create", "-i", str(yfile)])
    if rc != 0:
        raise RuntimeError(out)
    job = None
    for tok in out.replace('"', " ").split():
        tok = tok.strip(".:;,")
        if tok.startswith("tianrgua-%ss%d" % (tag, seg)):
            job = tok
    print("[seg %d] submitted %s ttl=%ss" % (seg, job, ttl), flush=True)
    return job, op, exp_epoch


def wait(job, stale_deadline, limit=10800):
    t0, last = time.time(), ""
    while time.time() - t0 < limit:
        rc, out = kraken(["list", "-r", "tianrgua"])
        st = ""
        for line in out.splitlines():
            if line.startswith(job):
                st = line.split()[1]
        if st != last:
            print("  [%ds] %s" % (int(time.time() - t0), st), flush=True)
            last = st
        if st.split(":")[0] in ("SUCCEEDED", "FAILED", "STOPPED"):
            rc, d = kraken(["describe", "--job-name", job])
            ec = [l.strip() for l in d.splitlines() if "exit code" in l]
            return st + " | " + (ec[0] if ec else "")
        if stale_deadline and time.time() > stale_deadline and \
                st.split(":")[0] in ("PENDING", "QUEUED", "CREATED", ""):
            kraken(["stop", "--job-name", job])
            return "STALE"
        time.sleep(40)
    return "TIMEOUT"


def fetch(prefix, dest):
    sess, _ = ada_session()
    s3 = sess.client("s3", region_name=REGION)
    Path(dest).mkdir(parents=True, exist_ok=True)
    got = {}
    for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", []):
        n = o["Key"].split("/")[-1]
        s3.download_file(BUCKET, o["Key"], str(Path(dest) / n), Config=XFER)
        got[n] = o["Size"]
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "train"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", default="1e-3")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--run-specs", default="")
    ap.add_argument("--ausmoke-keep", type=int, default=0,
                    help="cap the AuSmoke extension to N sequences (0 = all); "
                         "ablation that isolates extension dominance")
    ap.add_argument("--expect-train-ext", type=int, default=23)
    ap.add_argument("--expect-val-replay", type=int, default=139)
    ap.add_argument("--max-segments", type=int, default=4)
    ap.add_argument("--start-seg", type=int, default=1)
    ap.add_argument("--skip-weights-upload", action="store_true")
    ap.add_argument("--skip-frames-upload", action="store_true")
    ap.add_argument("--skip-v3-features-upload", action="store_true")
    ap.add_argument("--fetch-features", action="store_true")
    args = ap.parse_args()

    outdir = ROOT / "models/v4/trackd" / args.run
    for seg in range(args.start_seg, args.max_segments + 1):
        job, op, exp_epoch = submit(seg, args)
        st = wait(job, exp_epoch - 1500)
        print("[seg %d] %s" % (seg, st), flush=True)
        got = fetch(op, outdir / ("seg%02d" % seg))
        print("[seg %d] fetched %s" % (seg, got), flush=True)
        sf = outdir / ("seg%02d/status.json" % seg)
        if sf.exists():
            s = json.loads(sf.read_text())
            print("[seg %d] status %s" % (seg, s), flush=True)
            if s.get("phase") == "done":
                if args.mode == "extract" and args.fetch_features:
                    print("fetching v4 feature tar...", flush=True)
                    got = fetch("v4/runs/trackd/feats_v4.tgz",
                                ROOT / "data/v4/features_tars")
                    print("feature tar: %s" % got, flush=True)
                print("RUN COMPLETE (%d segs) -> %s" % (seg, outdir), flush=True)
                return 0
        # after seg1 the big artifacts are already in S3
        args.skip_weights_upload = True
        args.skip_frames_upload = True
        args.skip_v3_features_upload = True
    print("max segments without done", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
