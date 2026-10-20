#!/usr/bin/env python
"""V5b B200 probe orchestrator (obsidian-pu, ap-south-1).

Mint ada creds -> stage GET-test objects + presigned PUT/GET URLs in BOTH regional
buckets -> render job YAML with cluster/v5b/b200_probe_entry.sh + src/train/train_yolo.py
inline (base64) -> kraken submit -> poll -> while RUNNING read /tmp/probe_status via
`kraken jobs connect -c` -> when the pod reaches the retain phase pull probe_out.tgz
(and optionally last.pt) through connect (base64) -> ack -> terminal state + exit code
-> list what landed in S3. Registry reserve/release is done by the caller (bash).
Never writes static credentials anywhere.
"""
import argparse, base64, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = "000000000001"
KRAKEN = "kraken"
ADA = "ada"
B_APS1 = "smoke-detect-data-000000000001-aps1"
B_USE1 = "smoke-detect-data-000000000001-use1"

YAML_TMPL = """jobNamePrefix: tianrgua-{job}
jobPriority: LOW
containers:
  - imageUri: {image_uri}
    command: /bin/bash
    args:
      - -c
      - {cmd}
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


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {msg}", flush=True)


def ada_session():
    p = subprocess.run([ADA, "credentials", "print", "--account", ACCOUNT, "--provider",
                        "INTERNAL-PROVIDER", "--role", "INTERNAL-ADMIN-ROLE", "--format", "json"],
                       capture_output=True, text=True, timeout=90, check=True)
    c = json.loads(p.stdout)
    return boto3.Session(aws_access_key_id=c["AccessKeyId"], aws_secret_access_key=c["SecretAccessKey"],
                         aws_session_token=c["SessionToken"]), c.get("Expiration")


def s3c(sess, region):
    return sess.client("s3", region_name=region, endpoint_url=f"https://s3.{region}.amazonaws.com",
                       config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))


def kraken(project, args, timeout=180):
    p = subprocess.run([KRAKEN, "-p", project, "jobs"] + args, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def connect(project, job, cmd, timeout=120):
    try:
        p = subprocess.run([KRAKEN, "-p", project, "jobs", "connect", "-j", job, "-c", cmd],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    return p.returncode, p.stdout, p.stderr


def job_state(project, job):
    rc, out = kraken(project, ["list", "-r", "tianrgua"])
    for line in out.splitlines():
        if line.startswith(job):
            return line.split()[1]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="job tag (short, lowercase)")
    ap.add_argument("--project", default="obsidian-pu")
    ap.add_argument("--queue", default="obsidian-pu-hyperpod-p6-b200.48xlarge-ap-south-1-tas")
    ap.add_argument("--image-uri", default=f"{ACCOUNT}.dkr.ecr.ap-south-1.amazonaws.com/smoke-detect:v4")
    ap.add_argument("--entry", default=str(ROOT / "cluster/v5b/b200_probe_entry.sh"))
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--retain-s", type=int, default=900)
    ap.add_argument("--pull-ckpt", action="store_true", help="also pull last.pt via connect (throughput test)")
    ap.add_argument("--max-wait-s", type=int, default=3600)
    ap.add_argument("--pending-abort-s", type=int, default=1500, help="stop job if not RUNNING by then")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--attach-job", default=None, help="skip submit; poll/pull/finalize an existing job name")
    args = ap.parse_args()

    out_dir = ROOT / "reports/v5b/b200_probe" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    sess, cred_exp = ada_session()
    exp_epoch = int(datetime.strptime(cred_exp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
    ttl = max(600, exp_epoch - int(time.time()) - 60)
    s3a, s3u = s3c(sess, "ap-south-1"), s3c(sess, "us-east-1")
    prefix = f"probes/v5b/{args.tag}"
    stamp = datetime.now(timezone.utc).isoformat()
    if args.attach_job:
        job = args.attach_job; log(f"attached to {job}")
    else:
        for c, b in ((s3a, B_APS1), (s3u, B_USE1)):
            c.put_object(Bucket=b, Key=f"{prefix}/hello_get.txt", Body=f"GET-test {b} {stamp}\n".encode())
        def put(c, b, key):
            return c.generate_presigned_url("put_object", Params={"Bucket": b, "Key": f"{prefix}/{key}"}, ExpiresIn=ttl)
        def get(c, b, key):
            return c.generate_presigned_url("get_object", Params={"Bucket": b, "Key": f"{prefix}/{key}"}, ExpiresIn=ttl)
        envs = {
            "ENTRY_B64": base64.b64encode(Path(args.entry).read_bytes()).decode(),
            "TRAIN_PY_B64": base64.b64encode((ROOT / "src/train/train_yolo.py").read_bytes()).decode(),
            "BUCKET_APS1": B_APS1, "BUCKET_USE1": B_USE1,
            "N_TRAIN": str(args.n_train), "N_VAL": str(args.n_val), "IMGSZ": str(args.imgsz),
            "RUN_ID": args.tag, "BATCH": str(args.batch), "RETAIN_S": str(args.retain_s), "BASE_WEIGHTS": "yolo11m.pt",
            "URL_PUT_HELLO_APS1": put(s3a, B_APS1, "hello_put.txt"),
            "URL_PUT_HELLO_USE1": put(s3u, B_USE1, "hello_put.txt"),
            "URL_GET_APS1": get(s3a, B_APS1, "hello_get.txt"),
            "URL_GET_USE1": get(s3u, B_USE1, "hello_get.txt"),
            "URL_PUT_TGZ_APS1": put(s3a, B_APS1, "probe_out.tgz"),
            "URL_PUT_TGZ_USE1": put(s3u, B_USE1, "probe_out.tgz"),
            "URL_PUT_LAST_APS1": put(s3a, B_APS1, "last.pt"),
        }
        env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\"" for k, v in envs.items())
        cmd = 'echo "$ENTRY_B64" | base64 -d > /tmp/entry.sh && bash /tmp/entry.sh'
        yfile = ROOT / "cluster/generated" / f"b200probe_{args.tag}.yaml"
        yfile.write_text(YAML_TMPL.format(job=args.tag, image_uri=args.image_uri, cmd=cmd, envs=env_yaml, queue=args.queue))
        log(f"yaml {yfile} (url ttl {ttl}s, image {args.image_uri}, queue {args.queue})")
        if args.dry_run:
            return
        rc, out = kraken(args.project, ["create", "-i", str(yfile)])
        (out_dir / "submit.txt").write_text(out)
        if rc != 0:
            log(f"SUBMIT FAILED rc={rc}: {out[:600]}"); sys.exit(2)
        job = None
        for tok in out.replace('"', " ").replace(",", " ").split():
            tok = tok.strip(".:;")
            if tok.startswith(f"tianrgua-{args.tag}"):
                job = tok
        log(f"submitted {job}")
    summary = {"job": job, "image": args.image_uri, "queue": args.queue, "submitted_utc": stamp,
               "states": [], "status_samples": [], "pulled": {}}
    t0 = time.time(); last = ""; pulled = False; first_running = None
    while time.time() - t0 < args.max_wait_s:
        st = job_state(args.project, job)
        if st != last:
            log(f"state {st}"); summary["states"].append([round(time.time() - t0), st]); last = st
        base = st.split(":")[0]
        if base in ("SUCCEEDED", "FAILED", "STOPPED"):
            break
        if base == "RUNNING":
            if first_running is None:
                first_running = time.time(); summary["running_after_s"] = round(first_running - t0)
            rc, so, se = connect(args.project, job, "cat /tmp/probe_status 2>/dev/null; echo; tail -c 600 /tmp/probe.log 2>/dev/null", 90)
            jl = [l for l in so.splitlines() if l.startswith("{")]
            line = jl[0] if jl else (so.strip().splitlines() or [""])[0]
            log(f"probe_status: {line[:160]} | connect rc={rc} {se.strip()[:120] if rc else ''}")
            summary["status_samples"].append([round(time.time() - t0), rc, line[:200]])
            if rc == 0 and so.strip():
                (out_dir / "live_tail.txt").write_text(so)
            if rc == 0 and '"phase":"retain"' in line and not pulled:
                pulled = True
                for fname in ["probe_out.tgz"] + (["run1/weights/last.pt"] if args.pull_ckpt else []):
                    src = f"/tmp/{fname}" if fname == "probe_out.tgz" else f"/mnt/local/v5b_b200probe/{fname}"
                    t1 = time.time()
                    rc2, so2, se2 = connect(args.project, job, f"base64 -w0 {src} 2>/dev/null || base64 -w0 /tmp/v5b_b200probe/{fname}", 900)
                    data = b""
                    try:
                        data = base64.b64decode("".join(so2.split()[-1:]) if so2 else "")
                    except Exception as e:
                        log(f"decode failed {fname}: {e}")
                    dest = out_dir / Path(fname).name
                    dest.write_bytes(data)
                    dt = time.time() - t1
                    summary["pulled"][Path(fname).name] = {"bytes": len(data), "seconds": round(dt, 1), "rc": rc2}
                    log(f"pulled {fname}: {len(data)} bytes in {dt:.1f}s via connect (rc={rc2})")
                connect(args.project, job, "touch /tmp/ack", 60)
                log("acked -> pod exits")
        elif first_running is None and time.time() - t0 > args.pending_abort_s:
            log("not RUNNING before pending-abort -> stopping job"); kraken(args.project, ["stop", "--job-name", job])
            summary["aborted_pending"] = True
        time.sleep(20 if base == "RUNNING" else 30)
    rc, desc = kraken(args.project, ["describe", "--job-name", job])
    (out_dir / "describe.txt").write_text(desc)
    info = [l.strip() for l in desc.splitlines() if "exit" in l.lower() or "reason" in l.lower() or "message" in l.lower()]
    summary["final_state"] = last; summary["describe_exit_lines"] = info[:8]
    log(f"terminal {last} | {info[:4]}")
    landed = {}
    for c, b in ((s3a, B_APS1), (s3u, B_USE1)):
        try:
            for o in c.list_objects_v2(Bucket=b, Prefix=prefix).get("Contents", []):
                landed[f"{b}/{o['Key'].split('/')[-1]}"] = o["Size"]
                if o["Key"].endswith(("probe_out.tgz", "hello_put.txt", "podid_hello.txt")) and o["Size"] < 50_000_000:
                    c.download_file(b, o["Key"], str(out_dir / ("s3_" + b[-4:] + "_" + o["Key"].split("/")[-1])))
        except Exception as e:
            landed[b] = f"list failed: {e}"
    summary["s3_landed"] = landed
    summary["node_seconds_running_est"] = round(time.time() - first_running) if first_running else 0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "status_samples"}, indent=1))


if __name__ == "__main__":
    main()
