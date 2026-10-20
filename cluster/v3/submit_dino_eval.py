"""Submit a Track B/C DINOv3-detector inference-only eval job (green g6e).

Same pattern as submit_rfdetr_eval.py; ships dino_infer.py + src/v3 modeling
files, reads weights straight from /scratch, returns preds jsonl(.gz).

Usage:
  python cluster/v3/submit_dino_eval.py --eval-name B1_val \
      --backbone-dir /scratch/scratch/tianrgua-sandbox/models/dinov3-vitl16 \
      --ckpt /scratch/scratch/tianrgua-sandbox/smoke/runs_v3/B1_dino3vitl16_frozen_1280/best.pt \
      --splits sdis_val [--splits sdis_test] [--latency-n 50]
  # C1 cached-feature route (val only; features must exist):
  python cluster/v3/submit_dino_eval.py --eval-name C1_val \
      --feat-root /scratch/.../feats_v3/dino3_7b_1280x736 --embed-dim 4096 \
      --ckpt /scratch/.../runs_v3/C1_7b_head_cached_1280/best.pt --splits sdis_val
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
IMAGE = f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/smoke-dino3:v1"
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
    ("dino_infer.py", "cluster/v3/dino_infer.py"),
    ("det_modeling.py", "src/v3/det_modeling.py"),
    ("det_data.py", "src/v3/det_data.py"),
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
    ap.add_argument("--splits", action="append", default=None)
    ap.add_argument("--extra-list", action="append", default=None,
                    help="path=name for an extra image list to ship")
    ap.add_argument("--backbone-dir", default=None)
    ap.add_argument("--feat-root", default=None)
    ap.add_argument("--embed-dim", default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--ckpt", required=True, help="pod path to best.pt")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=736)
    ap.add_argument("--conf-floor", type=float, default=0.02)
    ap.add_argument("--latency-n", type=int, default=0)
    ap.add_argument("--frames-tar", default=None,
                    help="local .tgz of images to ship (extracted to /tmp/frames)")
    ap.add_argument("--rootmap-override", default=None,
                    help="localprefix=podprefix for the shipped frames")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--latency-list", default=None,
                    help="split name (data/yolo/lists/local/<name>.txt) shipped ONLY "
                         "for the latency probe, not inferred (v3 protocol: sdis_val)")
    ap.add_argument("--latency-variant", action="append", default=[],
                    help="tag=<dino_infer flags>: extra batch-1 latency measurements "
                         "in the same job (v4: LoRA unmerged vs --lora-merge)")
    ap.add_argument("--infer-extra", default="",
                    help="extra dino_infer.py flags passed verbatim on the pod "
                         "(v4 LoRA checkpoints need e.g. '--lora 16')")
    ap.add_argument("--ngpus", type=int, default=8)
    ap.add_argument("--poll-s", type=int, default=30)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate inputs, write+print the kraken YAML with placeholder "
                         "URLs; NO S3 upload, NO kraken submit (v4 final-test prep)")
    args = ap.parse_args()
    splits = args.splits or ([] if args.extra_list else ["sdis_val"])

    prefix = f"v3/evals/{args.eval_name}"
    if args.dry_run:
        class _S3Dry:  # no network: URLs are placeholders, nothing is uploaded
            def put_object(self, **kw): print(f"  [dry-run] would put s3://{BUCKET}/{kw['Key']} ({len(kw['Body'])} B)")
            def upload_file(self, f, b, k): print(f"  [dry-run] would upload {f} -> s3://{b}/{k} ({Path(f).stat().st_size} B)")
            def generate_presigned_url(self, op, Params, ExpiresIn): return f"<presigned-{op}:{Params['Key']}>"
        s3, ttl = _S3Dry(), 0
        for n, pth in CODE_FILES:
            assert (ROOT / pth).exists(), f"missing code file {pth}"
    else:
        sess, cred_exp = ada_session()
        s3 = sess.client("s3", region_name=REGION,
                         endpoint_url=f"https://s3.{REGION}.amazonaws.com",
                         config=Config(signature_version="s3v4",
                                       s3={"addressing_style": "virtual"}))
        exp_epoch = int(datetime.strptime(cred_exp, "%Y-%m-%dT%H:%M:%SZ")
                        .replace(tzinfo=timezone.utc).timestamp())
        ttl = max(600, exp_epoch - int(time.time()) - 60)

    s3.put_object(Bucket=BUCKET, Key=f"{prefix}/code.tgz",
                  Body=make_tar([(n, ROOT / p) for n, p in CODE_FILES]))
    entries = []
    for sp in splits:
        entries.append((f"lists/{sp}.txt", ROOT / "data/yolo/lists/local" / f"{sp}.txt"))
    for spec in (args.extra_list or []):
        path, name = spec.split("=", 1)
        entries.append((f"lists/{name}.txt", Path(path)))
    if args.latency_list and args.latency_list not in splits:
        entries.append((f"lists/{args.latency_list}.txt",
                        ROOT / "data/yolo/lists/local" / f"{args.latency_list}.txt"))
    s3.put_object(Bucket=BUCKET, Key=f"{prefix}/lists.tgz", Body=make_tar(entries))
    s3.upload_file(str(ROOT / "cluster/v3/dino_eval_entry.sh"), BUCKET,
                   f"{prefix}/dino_eval_entry.sh")

    gu = lambda k: s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=ttl)
    pu = lambda k: s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=ttl)

    all_splits = splits + [s.split("=", 1)[1] for s in (args.extra_list or [])]
    envs = {
        "EVAL_NAME": args.eval_name,
        "SPLITS": " ".join(all_splits),
        "CKPT": args.ckpt,
        "WIDTH": str(args.width), "HEIGHT": str(args.height),
        "CONF_FLOOR": str(args.conf_floor),
        "LATENCY_N": str(args.latency_n),
        "LATENCY_LIST": args.latency_list or all_splits[0],
        "NGPUS": str(args.ngpus),
        "URL_ENTRY": gu(f"{prefix}/dino_eval_entry.sh"),
        "URL_CODE": gu(f"{prefix}/code.tgz"),
        "URL_LISTS": gu(f"{prefix}/lists.tgz"),
        "URL_LATENCY": pu(f"{prefix}/latency_gpu.json"),
        "URL_LOG": pu(f"{prefix}/shardlogs.txt"),
    }
    if args.backbone_dir:
        envs["BACKBONE_DIR"] = args.backbone_dir
    if args.feat_root:
        envs["FEAT_ROOT"] = args.feat_root
        envs["EMBED_DIM"] = str(args.embed_dim)
    if args.bf16:
        envs["BF16"] = "1"
    if args.frames_tar:
        s3.upload_file(args.frames_tar, BUCKET, f"{prefix}/frames.tgz")
        envs["URL_FRAMES"] = gu(f"{prefix}/frames.tgz")
    if args.rootmap_override:
        envs["ROOTMAP_OVERRIDE"] = args.rootmap_override
    if args.max_images:
        envs["MAX_IMAGES"] = str(args.max_images)
    if args.infer_extra:
        envs["INFER_EXTRA"] = args.infer_extra
    if args.latency_variant:
        specs, tags = [], []
        for spec in args.latency_variant:
            tag, flags = spec.split("=", 1)
            specs.append(f"{tag}|{flags}")
            tags.append(tag)
            envs[f"URL_LATENCY_{tag.upper().replace('-', '_')}"] = pu(f"{prefix}/latency_gpu_{tag}.json")
        envs["LATENCY_VARIANTS"] = ";".join(specs)
    for sp in all_splits:
        envs[f"URL_PREDS_{sp.upper().replace('-', '_')}"] = pu(f"{prefix}/{sp}_preds.jsonl.gz")

    env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\"" for k, v in envs.items())
    tag = ("de" + args.eval_name.replace("_", "").replace("-", ""))[:16].lower()
    ydir = ROOT / "cluster" / "generated"
    ydir.mkdir(exist_ok=True)
    yfile = ydir / (f"DRYRUN_v3_dinoeval_{args.eval_name}.yaml" if args.dry_run
                    else f"v3_dinoeval_{args.eval_name}.yaml")
    yfile.write_text(YAML_TMPL.format(job=tag, image_uri=IMAGE, envs=env_yaml, queue=QUEUE))
    if args.dry_run:
        print(f"[dry-run] kraken YAML written to {yfile} (NOT submitted):")
        print(yfile.read_text())
        return 0
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
        if name in ("code.tgz", "lists.tgz", "dino_eval_entry.sh", "frames.tgz"):
            continue
        s3b.download_file(BUCKET, o["Key"], str(dest / name))
        print("got", name, o["Size"], flush=True)
    print("->", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
