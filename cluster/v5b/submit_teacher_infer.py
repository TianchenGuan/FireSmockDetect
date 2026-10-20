"""V5b: submit ONE teacher-pool inference job (green g6e, 1 node, 8 L40S) and
collect its predictions. Generalises cluster/v3/submit_dino_eval.py /
submit_rfdetr_eval.py (same presigned-URL pattern, same images) so that
  * both families go through one entry (cluster/v5b/teacher_infer_entry.sh),
  * YAML / results stay under cluster/v5b/generated and models/v5b/teacher/raw,
  * the pod verifies the checkpoint sha256 (EXPECT_SHA) before scoring.
Callers MUST reserve a node via cluster/v5b/registry.sh first (scripts/v5b/teacher_job.sh does).

Usage (DINO LoRA, local-path list + multi-root map):
  python cluster/v5b/submit_teacher_infer.py --family dino --eval-name b4L_a \
     --list data/v5b/teacher/lists/dino_shard_a.txt=shard_a \
     --ckpt /scratch/scratch/tianrgua-sandbox/smoke/runs_v3/v4B4_lora_L/best.pt \
     --backbone-dir /scratch/scratch/tianrgua-sandbox/models/dinov3-vitl16 --infer-extra "--lora 16" \
     --rootmap "<local>/data/yolo=<pod>/smoke_data_v2,<local>/data/v4/yolo/E_ET_v1=<pod>/smoke/v4data/E_ET_v1" \
     --expect-sha 87a9f175 [--dry-run] [--no-wait]
Usage (RF-DETR, POD-path list, no root map):
  python cluster/v5b/submit_teacher_infer.py --family rfdetr --eval-name rfdetrL_all \
     --list data/v5b/teacher/lists/rfdetr_all_pod.txt=all \
     --ckpt /scratch/scratch/tianrgua-sandbox/smoke/runs_v3/rfdetrL_1280_e48/checkpoint_best_ema.pth --expect-sha 2cca1fe8
"""
from __future__ import annotations
import argparse, io, json, subprocess, tarfile, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = "000000000001"
BUCKET = "smoke-detect-data-000000000001-use1"
REGION = "us-east-1"
KRAKEN = "kraken"
ADA = "ada"
QUEUE = "green-hyperpod-g6e.48xlarge-us-east-1-tas"
PROJECT = "green"
IMAGES = {"dino": f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/smoke-dino3:v1",
          "rfdetr": f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/smoke-det3:v1"}
CODE_FILES = {
    "dino": [("dino_infer.py", "cluster/v3/dino_infer.py"), ("det_modeling.py", "src/v3/det_modeling.py"),
             ("det_data.py", "src/v3/det_data.py")],
    "rfdetr": [("rfdetr_infer.py", "cluster/v3/rfdetr_infer.py"), ("src/__init__.py", "src/__init__.py"),
               ("src/serve/__init__.py", "src/serve/__init__.py"), ("src/serve/detector.py", "src/serve/detector.py"),
               ("src/serve/backend_rfdetr.py", "src/serve/backend_rfdetr.py")],
}
ENTRY = "cluster/v5b/teacher_infer_entry.sh"
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


def ada_session():
    import boto3
    p = subprocess.run([ADA, "credentials", "print", "--account", ACCOUNT, "--provider INTERNAL-PROVIDER",
                        "--role", "INTERNAL-ADMIN-ROLE", "--format", "json"],
                       capture_output=True, text=True, timeout=90, check=True)
    c = json.loads(p.stdout)
    return boto3.Session(aws_access_key_id=c["AccessKeyId"], aws_secret_access_key=c["SecretAccessKey"],
                         aws_session_token=c["SessionToken"]), c.get("Expiration")


def make_tar(entries):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, path in entries:
            tf.add(str(path), arcname=name)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(IMAGES))
    ap.add_argument("--eval-name", required=True)
    ap.add_argument("--list", action="append", required=True, help="localpath=name (repeatable)")
    ap.add_argument("--ckpt", required=True, help="pod path of the checkpoint")
    ap.add_argument("--expect-sha", default="", help="sha256 prefix the pod must see on --ckpt")
    ap.add_argument("--backbone-dir", default=None)
    ap.add_argument("--infer-extra", default="")
    ap.add_argument("--rootmap", default="", help="comma-separated local=pod prefixes (dino: many; rfdetr: at most one)")
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=736)
    ap.add_argument("--imgsz", type=int, default=1280); ap.add_argument("--conf-floor", type=float, default=0.02)
    ap.add_argument("--max-images", type=int, default=0); ap.add_argument("--ngpus", type=int, default=8)
    ap.add_argument("--poll-s", type=int, default=30); ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--data-tar", action="append", default=[],
                    help="name=localpath (repeatable): uploaded ONCE to s3 v4/data/<name>.tar (marker cluster/v4/.uploaded_<name>, "
                         "shared with orchestrate_v4) and staged by the pod under /scratch/.../smoke/v5bdata (DATA_TARS, .<name>.done)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.family == "dino" and not a.backbone_dir:
        ap.error("--backbone-dir required for dino")
    if a.family == "rfdetr" and a.rootmap.count("=") > 1:
        ap.error("rfdetr_infer.py supports a single root-map pair; ship a pod-path list instead")
    lists = []
    for spec in a.list:
        p, n = spec.split("=", 1); p = Path(p)
        assert p.exists(), f"list missing {p}"
        lists.append((p, n))
    for _, pth in CODE_FILES[a.family]:
        assert (ROOT / pth).exists(), f"missing code file {pth}"
    assert (ROOT / ENTRY).exists()
    prefix = f"v5b/teacher/{a.eval_name}"
    outdir = ROOT / "models/v5b/teacher/raw" / a.eval_name; outdir.mkdir(parents=True, exist_ok=True)

    if a.dry_run:
        class _Dry:
            def put_object(self, **kw): print(f"  [dry-run] put s3://{BUCKET}/{kw['Key']} ({len(kw['Body'])} B)")
            def upload_file(self, f, b, k): print(f"  [dry-run] upload {f} -> s3://{b}/{k}")
            def generate_presigned_url(self, op, Params, ExpiresIn): return f"<presigned-{op}:{Params['Key']}>"
        s3, ttl = _Dry(), 0
    else:
        sess, exp = ada_session()
        from botocore.config import Config
        s3 = sess.client("s3", region_name=REGION, endpoint_url=f"https://s3.{REGION}.amazonaws.com",
                         config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
        exp_epoch = int(datetime.strptime(exp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
        ttl = max(600, exp_epoch - int(time.time()) - 60)
    s3.put_object(Bucket=BUCKET, Key=f"{prefix}/code.tgz", Body=make_tar([(n, ROOT / p) for n, p in CODE_FILES[a.family]]))
    s3.put_object(Bucket=BUCKET, Key=f"{prefix}/lists.tgz", Body=make_tar([(f"lists/{n}.txt", p) for p, n in lists]))
    s3.upload_file(str(ROOT / ENTRY), BUCKET, f"{prefix}/entry.sh")
    gu = lambda k: s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=ttl)
    pu = lambda k: s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=ttl)
    names = [n for _, n in lists]
    envs = {"EVAL_NAME": a.eval_name, "FAMILY": a.family, "SPLITS": " ".join(names), "CKPT": a.ckpt,
            "EXPECT_SHA": a.expect_sha, "NGPUS": str(a.ngpus), "CONF_FLOOR": str(a.conf_floor),
            "WIDTH": str(a.width), "HEIGHT": str(a.height), "IMGSZ": str(a.imgsz),
            "URL_ENTRY": gu(f"{prefix}/entry.sh"), "URL_CODE": gu(f"{prefix}/code.tgz"), "URL_LISTS": gu(f"{prefix}/lists.tgz"),
            "URL_LOG": pu(f"{prefix}/entry_log_tail.txt"), "URL_SHARDLOGS": pu(f"{prefix}/shardlogs.txt"),
            "URL_SUMMARY": pu(f"{prefix}/summary.jsonl")}
    if a.backbone_dir: envs["BACKBONE_DIR"] = a.backbone_dir
    if a.infer_extra: envs["INFER_EXTRA"] = a.infer_extra
    if a.rootmap: envs["ROOTMAP"] = a.rootmap
    if a.max_images: envs["MAX_IMAGES"] = str(a.max_images)
    if a.data_tar:
        specs = []
        for spec in a.data_tar:
            name, local = spec.split("=", 1)
            key = f"v4/data/{name}.tar"
            marker = ROOT / "cluster/v4" / f".uploaded_{name}"
            if not marker.exists():
                print(f"uploading {local} -> s3://{BUCKET}/{key} ...", flush=True)
                s3.upload_file(str(ROOT / local), BUCKET, key)
                if not a.dry_run:
                    marker.write_text(datetime.now(timezone.utc).isoformat())
            specs.append(f"{name}={gu(key)}")
        envs["DATA_TARS"] = ",".join(specs)
    for n in names:
        envs[f"URL_PREDS_{n.upper().replace('-', '_')}"] = pu(f"{prefix}/{n}_preds.jsonl.gz")
    env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\"" for k, v in envs.items())
    tag = ("tch" + a.eval_name.replace("_", "").replace("-", ""))[:16].lower()
    ydir = ROOT / "cluster/v5b/generated"; ydir.mkdir(exist_ok=True)
    yfile = ydir / (("DRYRUN_" if a.dry_run else "") + f"v5b_teacher_{a.eval_name}.yaml")
    yfile.write_text(YAML_TMPL.format(job=tag, image_uri=IMAGES[a.family], envs=env_yaml, queue=QUEUE))
    if a.dry_run:
        print(f"[dry-run] YAML -> {yfile} (NOT submitted); lists: {[(str(p), n, sum(1 for _ in open(p))) for p, n in lists]}")
        return 0
    r = subprocess.run([KRAKEN, "-p", PROJECT, "jobs", "create", "-i", str(yfile)], capture_output=True, text=True, timeout=180)
    out = (r.stdout + r.stderr).strip(); print(out)
    job = None
    for tok in out.replace('"', " ").replace(",", " ").split():
        tok = tok.strip(".:;")
        if tok.startswith(f"tianrgua-{tag}"): job = tok
    rec = {"eval_name": a.eval_name, "family": a.family, "job": job, "submitted_utc": datetime.now(timezone.utc).strftime("%FT%TZ"),
           "ckpt": a.ckpt, "expect_sha": a.expect_sha, "lists": {n: str(p) for p, n in lists}, "ttl_s": ttl, "yaml": str(yfile)}
    (outdir / "job.json").write_text(json.dumps(rec, indent=1))
    print("job:", job, "| ttl", ttl, "s", flush=True)
    if a.no_wait or not job:
        return 0 if job else 2
    t0, last, state = time.time(), "", ""
    while time.time() - t0 < 7200:
        rr = subprocess.run([KRAKEN, "-p", PROJECT, "jobs", "list", "-r", "tianrgua"], capture_output=True, text=True, timeout=180)
        for line in (rr.stdout + rr.stderr).splitlines():
            if line.startswith(job): state = line.split()[1]
        if state != last:
            print(f"  [{int(time.time()-t0)}s] {state}", flush=True); last = state
        if state.split(":")[0] in ("SUCCEEDED", "FAILED", "STOPPED"): break
        time.sleep(a.poll_s)
    rec.update(final_state=state, wall_s=int(time.time() - t0), finished_utc=datetime.now(timezone.utc).strftime("%FT%TZ"))
    sess2, _ = ada_session(); s3b = sess2.client("s3", region_name=REGION)
    got = []
    for o in s3b.list_objects_v2(Bucket=BUCKET, Prefix=prefix + "/").get("Contents", []):
        name = o["Key"].split("/")[-1]
        if name in ("code.tgz", "lists.tgz", "entry.sh"): continue
        s3b.download_file(BUCKET, o["Key"], str(outdir / name)); got.append((name, o["Size"]))
    rec["downloaded"] = got
    (outdir / "job.json").write_text(json.dumps(rec, indent=1))
    print("got:", got, "\n->", outdir, "| state", state)
    return 0 if state.startswith("SUCCEEDED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
