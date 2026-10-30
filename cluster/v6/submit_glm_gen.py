"""Submit a GLM-Image generation job on green L40S and fetch outputs.

Reserves nothing itself — caller must reserve in the v6 registry.

Usage:
  .venv/bin/python cluster/v6/submit_glm_gen.py --tag glmsmk1 \
      --queue data/v6/gen/q_glm_probe.jsonl --route glm_probe \
      [--queue-name green-hyperpod-g6e.48xlarge-us-east-1-tas] \
      [--imgsz 1024] [--steps ''] [--nshards 8] [--skip-weights-upload]
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster"))
import orchestrate_training as ot  # noqa: E402

BUCKET = "smoke-detect-data-000000000001-use1"
REGION = "us-east-1"
IMAGE = "000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-diffusion:v1"

YAML_TMPL = """jobNamePrefix: tianrgua-{job}
jobPriority: LOW
containers:
  - imageUri: {image}
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
  workloadType: data-gen
  subtype: general
  role: general
  runType: dev
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--queue-name",
                    default="green-hyperpod-g6e.48xlarge-us-east-1-tas")
    ap.add_argument("--imgsz", default="1024")
    ap.add_argument("--steps", default="")
    ap.add_argument("--nshards", default="8")
    ap.add_argument("--skip-weights-upload", action="store_true")
    ap.add_argument("--bgs-tar", default="",
                    help="optional tar of background images for i2i items")
    ap.add_argument("--bgs-key", default="",
                    help="reuse an already-uploaded S3 key for the bgs tar")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--segments", type=int, default=1,
                    help="resubmit until the queue is complete (outputs "
                         "persist on /scratch), at most N jobs")
    args = ap.parse_args()
    for seg in range(1, args.segments + 1):
        rc = run_one(args, seg)
        if rc == 0:
            return 0
        if rc == 2:
            return 2
    print("segments exhausted; queue incomplete")
    return 1


def run_one(args, seg):

    sess, cred_exp = ot.ada_session()
    s3 = ot.s3_client(sess, REGION)
    exp_epoch = int(datetime.strptime(cred_exp, "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=timezone.utc).timestamp())
    ttl = max(600, exp_epoch - int(time.time()) - 60)
    kp = f"v6/glm/{args.tag}"

    wkey = "v6/weights/glm-image.tar"
    marker = ROOT / "cluster/v6/.uploaded_glmimage"
    if not args.skip_weights_upload and not marker.exists():
        print("uploading GLM weights tar (34GB) ...", flush=True)
        s3.upload_file(str(ROOT / "data/v6/weights/glm-image.tar"), BUCKET, wkey)
        marker.write_text(datetime.now(timezone.utc).isoformat())
        print("weights uploaded")

    s3.upload_file(str(ROOT / "cluster/v6/glm_gen_entry.sh"), BUCKET, f"{kp}/entry.sh")
    s3.upload_file(str(ROOT / "scripts/v6/glm_gen_worker.py"), BUCKET, f"{kp}/worker.py")
    s3.upload_file(str(ROOT / args.queue), BUCKET, f"{kp}/queue.jsonl")
    bgs_key = args.bgs_key
    if args.bgs_tar and not bgs_key:
        bgs_key = f"{kp}/bgs.tar"
        s3.upload_file(str(ROOT / args.bgs_tar), BUCKET, bgs_key)

    def put(k):
        return s3.generate_presigned_url("put_object",
            Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=ttl)

    def get(k):
        return s3.generate_presigned_url("get_object",
            Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=ttl)

    envs = {
        "URL_ENTRY": get(f"{kp}/entry.sh"),
        "URL_WORKER": get(f"{kp}/worker.py"),
        "URL_QUEUE": get(f"{kp}/queue.jsonl"),
        "URL_WEIGHTS": get(wkey),
        "URL_OUT": put(f"{kp}/glmout.tgz"),
        "URL_LOG": put(f"{kp}/entry.log"),
        "URL_EXPIRY_EPOCH": str(exp_epoch),
        "IMGSZ": args.imgsz, "STEPS": args.steps, "NSHARDS": args.nshards,
        "OUTTAG": args.route,
    }
    if bgs_key:
        envs["URL_BGS"] = get(bgs_key)
    env_yaml = "\n".join(
        f"      - name: {k}\n        value: \"{v}\"" for k, v in envs.items())
    yfile = ROOT / "cluster/generated" / f"glmgen_{args.tag}{seg}.yaml"
    yfile.write_text(YAML_TMPL.format(job=f"{args.tag}{seg}", image=args.image,
                                      envs=env_yaml, queue=args.queue_name))
    rc, out = ot.kraken_jobs("green", ["create", "-i", str(yfile)])
    job = None
    for tok in out.split():
        if tok.startswith(f"tianrgua-{args.tag}{seg}"):
            job = tok.strip().strip("'\",.")
    print(f"[seg {seg}] submitted {job}")
    if not job:
        print(out[:800]); return 2
    st = ot.wait_terminal("green", job, stale_deadline=exp_epoch - 1500)
    print("terminal:", st)

    outdir = ROOT / "data/v6/gen" / args.route
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(BUCKET, f"{kp}/entry.log", str(outdir / f"entry_{args.tag}.log"))
        print("log  ->", outdir / f"entry_{args.tag}.log")
    except Exception as e:
        print("no log:", e)
    try:
        tgz = outdir / f"glmout_{args.tag}.tgz"
        s3.download_file(BUCKET, f"{kp}/glmout.tgz", str(tgz))
        with tarfile.open(tgz) as tf:
            tf.extractall(outdir, filter="data")
        tgz.unlink()
    except Exception as e:
        print("no outputs:", e)
        return 1
    import json as _json
    qhashes = [_json.loads(l)["hash"] for l in
               (ROOT / args.queue).read_text().splitlines() if l.strip()]

    def _ok(h):
        p = outdir / f"{h}.json"
        if not p.exists():
            return False
        try:  # error sidecars count as remaining (worker retries them)
            return not _json.loads(p.read_text()).get("error")
        except Exception:
            return False

    remaining = [h for h in qhashes if not _ok(h)]
    n = len(list(outdir.glob("*.jpg")))
    print(f"[seg {seg}] extracted -> {outdir} ({n} jpgs; "
          f"{len(remaining)}/{len(qhashes)} remaining)")
    return 0 if not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
