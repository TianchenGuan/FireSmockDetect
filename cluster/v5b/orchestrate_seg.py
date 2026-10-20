"""V5b segmentation branch orchestrator (1 green g6e node, 3 configs concurrently).
Pattern = cluster/v3/orchestrate_dino.py: mint ada creds -> upload code tgz +
entry + data tar (once, marker file) -> presign -> kraken submit -> poll ->
fetch artifacts -> resubmit with resume until every config has DONE.
Registry reservation is the CALLER's job (bash cluster/v5b/registry.sh reserve ...).

  python cluster/v5b/orchestrate_seg.py --run segV5b_a \
      --configs "lora:0,1,2,3:2;frozen:4,5:4;r50:6,7:8" --epochs 10
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster/v3"))
from orchestrate_dino import ada_session, s3c, kraken, wait, fetch, BUCKET, QUEUE, YAML_TMPL  # noqa: E402

CODE = [("src/v3/det_modeling.py", "det_modeling.py"), ("src/v5b/seg_head.py", "seg_head.py"),
        ("src/v5b/seg_data.py", "seg_data.py"), ("src/v5b/train_seg.py", "train_seg.py")]
ART = ["BEST:best.pt", "LAST:last.pt", "METRICS:metrics.jsonl", "VALMETRICS:val_metrics_best.json",
       "PERIMG:val_per_image.json", "PROBS:val_probs.tar", "PARAMS:param_counts.json",
       "CONSOLE:console_tail.txt"]


def code_tgz():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for src, arc in CODE:
            tf.add(ROOT / src, arcname=arc)
    return buf.getvalue()


def submit(seg, args):
    sess, exp = ada_session()
    s3 = s3c(sess)
    exp_epoch = int(datetime.strptime(exp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
    ttl = max(600, exp_epoch - int(time.time()) - 60)
    kp = f"v5b/jobs/{args.run}"
    s3.put_object(Bucket=BUCKET, Key=f"{kp}/code.tgz", Body=code_tgz())
    s3.upload_file(str(ROOT / "cluster/v5b/seg_entry.sh"), BUCKET, f"{kp}/entry.sh")
    name, local = args.data_tar.split("=", 1)
    dkey = f"v5b/data/{name}.tar"
    marker = ROOT / "cluster/v5b" / f".uploaded_{name}"
    if not marker.exists():
        print(f"uploading {local} -> s3://{BUCKET}/{dkey}", flush=True)
        s3.upload_file(str(ROOT / local), BUCKET, dkey)
        marker.write_text(datetime.now(timezone.utc).isoformat())

    def get(key):
        return s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

    def put(key):
        return s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

    op = f"v5b/runs/{args.run}/seg{seg:02d}"
    envs = {"URL_ENTRY": get(f"{kp}/entry.sh"), "URL_CODE": get(f"{kp}/code.tgz"),
            "DATA_TARS": f"{name}={get(dkey)}", "RUN_NAME": args.run, "MODEL_DIR": args.model_dir,
            "CONFIGS": args.configs, "EPOCHS": str(args.epochs), "WIDTH": str(args.width),
            "HEIGHT": str(args.height), "LR": args.lr, "LORA_LR": args.lora_lr, "WARMUP": str(args.warmup),
            "WORKERS": str(args.workers), "EXTRA_ARGS": args.extra_args,
            "URL_EXPIRY_EPOCH": str(exp_epoch), "URL_STATUS": put(f"{op}/status.json"),
            "URL_LOG": put(f"{op}/entry_log.txt")}
    cfgs = [c.split(":")[0] for c in args.configs.split(";") if c]
    for c in cfgs:
        for a in ART:
            k, f = a.split(":")
            envs[f"URL_{c.upper()}_{k}"] = put(f"{op}/{c}/{f}")
    env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\"" for k, v in envs.items())
    (ROOT / "cluster/generated").mkdir(exist_ok=True)
    yfile = ROOT / "cluster/generated" / f"seg_{args.run}_s{seg:02d}.yaml"
    yfile.write_text(YAML_TMPL.format(job=f"{args.job_tag}s{seg}", image_uri=args.image_uri,
                                      envs=env_yaml, queue=QUEUE))
    rc, out = kraken(["create", "-i", str(yfile)])
    if rc != 0:
        raise RuntimeError(out)
    job = None
    for tok in out.replace('"', " ").split():
        tok = tok.strip(".:;,")
        if tok.startswith(f"tianrgua-{args.job_tag}s{seg}"):
            job = tok
    print(f"[seg {seg}] submitted {job} ttl={ttl}s", flush=True)
    return job, op, exp_epoch, cfgs


def fetch_tree(op, dest, cfgs):
    sess, _ = ada_session()
    s3 = sess.client("s3", region_name="us-east-1")
    got = {}
    for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=op).get("Contents", []):
        rel = o["Key"][len(op) + 1:]
        p = Path(dest) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, o["Key"], str(p))
        got[rel] = o["Size"]
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--job-tag", default=None)
    ap.add_argument("--configs", default="lora:0,1,2,3:2;frozen:4,5:4;r50:6,7:8")
    ap.add_argument("--model-dir", default="/scratch/scratch/tianrgua-sandbox/models/dinov3-vitl16")
    ap.add_argument("--data-tar", default="seg_v5b=data/v5b/stage/seg_v5b.tar")
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=736)
    ap.add_argument("--lr", default="2e-4")
    ap.add_argument("--lora-lr", default="1e-4")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--image-uri", default="000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-dino3:v1")
    ap.add_argument("--max-segments", type=int, default=3)
    ap.add_argument("--start-seg", type=int, default=1)
    args = ap.parse_args()
    args.job_tag = (args.job_tag or ("sg" + args.run.replace("_", "")))[:12].lower()
    outdir = ROOT / "models/v5b/seg" / args.run
    for seg in range(args.start_seg, args.max_segments + 1):
        job, op, exp_epoch, cfgs = submit(seg, args)
        st = wait(job, exp_epoch - 1500)
        print(f"[seg {seg}] {st}", flush=True)
        got = fetch_tree(op, outdir / f"seg{seg:02d}", cfgs)
        print(f"[seg {seg}] fetched {json.dumps(got)[:1500]}", flush=True)
        sf = outdir / f"seg{seg:02d}/status.json"
        if sf.exists():
            try:
                s = json.loads(sf.read_text())
            except json.JSONDecodeError:  # salvage the phase field (v3 lesson)
                txt = sf.read_text()
                s = {"phase": txt.split('"phase":"', 1)[1].split('"', 1)[0]
                     if '"phase":"' in txt else "unparseable", "raw": txt[:300]}
            print(f"[seg {seg}] status {s}", flush=True)
            if s.get("phase") == "done":
                print(f"RUN COMPLETE ({seg} segs) -> {outdir}", flush=True)
                return 0
            if s.get("phase") in ("model_missing", "data_missing", "staging_failed"):
                print("fatal staging problem", flush=True)
                return 3
        if st == "STALE":
            continue
    print("max segments without done", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
