"""Stage model weights from S3 onto cluster /scratch via a Leviathan job.

Devbox side: mint fresh ada creds -> list s3://<bucket>/weights/<name>/ ->
build a manifest of relpath|size|presigned-GET-url -> upload it + presign a
GET for it -> presign a PUT for a status.json -> render + submit a kraken job
(green g6e, smoke-detect:v6) that curls every file into
/scratch/scratch/tianrgua-sandbox/models/<name>/ with 8-way parallelism,
verifies per-file byte sizes against the manifest, and uploads status.json.
Job exits 0 only if every size matches (pods have no log access; exit code +
status.json are the only signals).

Presigned URLs die with the ada creds (~55 min) -> run this script right
before you want the job submitted. If a model's transfer would exceed ~45 min
(>= ~150 GB at observed same-region rates), split it: run twice with
--file-list listing half the relpaths each (status goes to status_<tag>.json).

Usage:
  python cluster/v2/stage_weights_job.py --name qwen3.5-9b
  python cluster/v2/stage_weights_job.py --name qwen3.5-27b --submit-only
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

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = "000000000001"
ADA = "ada"
KRAKEN = "kraken"
SCRATCH_BASE = "/scratch/scratch/tianrgua-sandbox/models"

YAML_TMPL = """jobNamePrefix: tianrgua-{job}
jobPriority: LOW
containers:
  - imageUri: {image_uri}
    command: /bin/bash
    args:
      - -c
      - >-
        set +e;
        D={dest};
        mkdir -p $D && cd $D || exit 101;
        curl -sf --retry 3 --max-time 180 -o /tmp/filelist.txt "$GET_LIST" || exit 102;
        N=$(grep -c . /tmp/filelist.txt);
        while IFS="|" read -r P S U || [ -n "$P" ]; do echo $P; echo $U; done < /tmp/filelist.txt > /tmp/pairs.txt;
        xargs -P8 -n2 curl -sf --retry 3 --create-dirs --max-time 2700 -o < /tmp/pairs.txt;
        RC=$?;
        FAIL=0; TOTE=0; TOTA=0; NV=0; SZ="";
        while IFS="|" read -r P S U || [ -n "$P" ]; do
        NV=$((NV+1));
        A=$(stat -c%s $D/$P 2>/dev/null); [ -z "$A" ] && A=-1;
        [ "$A" != "$S" ] && FAIL=$((FAIL+1));
        TOTE=$((TOTE+S)); [ "$A" -gt 0 ] && TOTA=$((TOTA+A));
        SZ="$SZ,\\"$P\\":[$S,$A]";
        done < /tmp/filelist.txt;
        SZ=${{SZ#,}};
        OK=false; [ "$FAIL" -eq 0 ] && [ "$N" -gt 0 ] && [ "$NV" -eq "$N" ] && OK=true;
        echo "{{\\"model\\":\\"$MODEL_NAME\\",\\"ok\\":$OK,\\"nfiles\\":$NV,\\"mismatches\\":$FAIL,\\"dl_rc\\":$RC,\\"bytes_expected\\":$TOTE,\\"bytes_actual\\":$TOTA,\\"dest\\":\\"$D\\",\\"sizes\\":{{$SZ}}}}" > /tmp/status.json;
        curl -sf --retry 3 -T /tmp/status.json "$PUT_STATUS";
        [ "$OK" = "true" ] && exit 0;
        exit 105
    environmentVariables:
      - name: MODEL_NAME
        value: "{name}"
      - name: GET_LIST
        value: "{get_list}"
      - name: PUT_STATUS
        value: "{put_status}"
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
    p = subprocess.run(
        [ADA, "credentials", "print", "--account", ACCOUNT, "--provider",
         "INTERNAL-PROVIDER", "--role", "INTERNAL-ADMIN-ROLE", "--format",
         "json"],
        capture_output=True, text=True, timeout=90, check=True)
    c = json.loads(p.stdout)
    sess = boto3.Session(aws_access_key_id=c["AccessKeyId"],
                         aws_secret_access_key=c["SecretAccessKey"],
                         aws_session_token=c["SessionToken"])
    return sess, c.get("Expiration")


def s3_client(sess, region):
    # presign against the regional endpoint: pods reach S3 only through a
    # regional VPC endpoint; the global hostname does not resolve there.
    return sess.client("s3", region_name=region,
                       endpoint_url=f"https://s3.{region}.amazonaws.com",
                       config=Config(signature_version="s3v4",
                                     s3={"addressing_style": "virtual"}))


def kraken_jobs(project, args_list):
    p = subprocess.run([KRAKEN, "-p", project, "jobs"] + args_list,
                       capture_output=True, text=True, timeout=180)
    return p.returncode, p.stdout + p.stderr


def wait_terminal(project, job_name, poll_s=45, max_wait_s=7200,
                  stale_deadline=None):
    t0 = time.time()
    last = ""
    while time.time() - t0 < max_wait_s:
        rc, out = kraken_jobs(project, ["list", "-r", "tianrgua"])
        state = ""
        for line in out.splitlines():
            if line.startswith(job_name):
                state = line.split()[1]
        if state != last:
            print(f"  [{int(time.time() - t0)}s] {state}", flush=True)
            last = state
        if state.split(":")[0] in ("SUCCEEDED", "FAILED", "STOPPED"):
            rc, desc = kraken_jobs(project, ["describe", "--job-name", job_name])
            info = [l for l in desc.splitlines() if "exit code" in l]
            return state + " | " + (info[0].strip() if info else "")
        if (stale_deadline and time.time() > stale_deadline
                and state.split(":")[0] in ("PENDING", "QUEUED", "CREATED", "")):
            print("  still pending near URL expiry -> stop (resubmit with "
                  "fresh URLs)", flush=True)
            kraken_jobs(project, ["stop", "--job-name", job_name])
            return "STALE"
        time.sleep(poll_s)
    return "TIMEOUT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    help="model dir name, e.g. qwen3.5-9b (S3 prefix "
                         "weights/<name>/, dest %s/<name>/)" % SCRATCH_BASE)
    ap.add_argument("--file-list", default=None,
                    help="optional file of relpaths (one per line) to stage "
                         "-- use to split a big model across two jobs")
    ap.add_argument("--job-tag", default=None)
    ap.add_argument("--project", default="green")
    ap.add_argument("--queue",
                    default="green-hyperpod-g6e.48xlarge-us-east-1-tas")
    ap.add_argument("--bucket", default="smoke-detect-data-000000000001-use1")
    ap.add_argument("--s3-region", default="us-east-1")
    ap.add_argument("--image-uri",
        default="000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-detect:v6")
    ap.add_argument("--submit-only", action="store_true",
                    help="submit and print job name; skip polling")
    args = ap.parse_args()

    tag = args.job_tag or ("ws" + "".join(c for c in args.name
                                          if c.isalnum()))[:16]
    tag = tag.lower()
    prefix = f"weights/{args.name}/"
    staging = f"v2/staging/{args.name}"
    status_key = (f"{staging}/status.json" if not args.file_list
                  else f"{staging}/status_{tag}.json")

    sess, cred_exp = ada_session()
    s3 = s3_client(sess, args.s3_region)
    exp_epoch = int(datetime.strptime(cred_exp, "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=timezone.utc).timestamp())
    ttl = max(600, exp_epoch - int(time.time()) - 60)

    # authoritative file list = what is actually in S3
    objs = []
    pag = s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=args.bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            rel = o["Key"][len(prefix):]
            if rel and not rel.endswith("/"):
                objs.append((rel, o["Size"]))
    if args.file_list:
        keep = {l.strip() for l in Path(args.file_list).read_text().splitlines()
                if l.strip()}
        objs = [o for o in objs if o[0] in keep]
        missing = keep - {o[0] for o in objs}
        if missing:
            raise SystemExit(f"--file-list entries not in S3: {sorted(missing)}")
    if not objs:
        raise SystemExit(f"no objects under s3://{args.bucket}/{prefix}")

    # cross-check against the local model dir when present
    local = ROOT / "models" / "v2" / args.name
    if local.is_dir():
        bad = [r for r, s in objs
               if not (local / r).exists() or (local / r).stat().st_size != s]
        if bad:
            raise SystemExit(f"S3/local size mismatch (re-sync first): {bad}")
        nloc = sum(1 for p in local.rglob("*")
                   if p.is_file() and ".cache" not in p.parts)
        if nloc != len(objs) and not args.file_list:
            print(f"WARNING: local has {nloc} files, S3 prefix has "
                  f"{len(objs)} -- sync incomplete?", flush=True)

    tot = sum(s for _, s in objs)
    est_min = tot / (8 * 100e6) / 60  # ~100 MB/s per curl stream, 8-way
    print(f"{args.name}: {len(objs)} files, {tot/1e9:.1f} GB, "
          f"rough est {est_min:.0f} min at 800 MB/s aggregate "
          f"(URL ttl {ttl}s)", flush=True)
    if est_min > 40:
        print("WARNING: estimated transfer near/over URL lifetime -- "
              "split via --file-list", flush=True)

    for rel, _ in objs:
        if "|" in rel or '"' in rel or "'" in rel or " " in rel:
            raise SystemExit(f"unsafe character in relpath: {rel!r}")

    lines = []
    for rel, size in objs:
        url = s3.generate_presigned_url("get_object",
            Params={"Bucket": args.bucket, "Key": prefix + rel},
            ExpiresIn=ttl)
        lines.append(f"{rel}|{size}|{url}")
    manifest_key = (f"{staging}/filelist.txt" if not args.file_list
                    else f"{staging}/filelist_{tag}.txt")
    # trailing newline is REQUIRED: the pod-side "while read" loop drops a
    # final line that lacks one (caused a silent skip of the last file once)
    s3.put_object(Bucket=args.bucket, Key=manifest_key,
                  Body=("\n".join(lines) + "\n").encode())
    get_list = s3.generate_presigned_url("get_object",
        Params={"Bucket": args.bucket, "Key": manifest_key}, ExpiresIn=ttl)
    put_status = s3.generate_presigned_url("put_object",
        Params={"Bucket": args.bucket, "Key": status_key}, ExpiresIn=ttl)

    ydir = ROOT / "cluster" / "v2" / "generated"
    ydir.mkdir(parents=True, exist_ok=True)
    yfile = ydir / f"wstage_{tag}.yaml"
    yfile.write_text(YAML_TMPL.format(
        job=tag, image_uri=args.image_uri, name=args.name,
        dest=f"{SCRATCH_BASE}/{args.name}", get_list=get_list,
        put_status=put_status, queue=args.queue))
    rc, out = kraken_jobs(args.project, ["create", "-i", str(yfile)])
    if rc != 0:
        raise SystemExit(f"submit failed: {out}")
    job_name = None
    for tok in out.replace('"', " ").replace(",", " ").split():
        tok = tok.strip(".:;")
        if tok.startswith(f"tianrgua-{tag}"):
            job_name = tok
    print(f"submitted {job_name} (status -> s3://{args.bucket}/{status_key})",
          flush=True)
    if not job_name:
        raise SystemExit("could not parse job name from kraken output:\n" + out)
    if args.submit_only:
        return 0

    desc = wait_terminal(args.project, job_name,
                         stale_deadline=exp_epoch - 900)
    print(f"terminal: {desc}", flush=True)

    sess2, _ = ada_session()
    s3v = sess2.client("s3", region_name=args.s3_region)
    try:
        body = s3v.get_object(Bucket=args.bucket, Key=status_key)["Body"].read()
        st = json.loads(body)
        print(f"status.json: ok={st.get('ok')} nfiles={st.get('nfiles')} "
              f"mismatches={st.get('mismatches')} "
              f"bytes={st.get('bytes_actual')}/{st.get('bytes_expected')}",
              flush=True)
        outp = ROOT / "cluster" / "v2" / "generated" / f"status_{tag}.json"
        outp.write_bytes(body)
        print(f"saved -> {outp}", flush=True)
        # devbox-side re-verification: the pod must have checked EVERY file
        # we listed, and the byte totals must match our own S3 listing
        good = (st.get("ok") is True and st.get("mismatches") == 0
                and st.get("nfiles") == len(objs)
                and len(st.get("sizes", {})) == len(objs)
                and st.get("bytes_expected") == tot
                and desc.startswith("SUCCEEDED"))
        if not good:
            print(f"VERIFICATION FAILED: expected {len(objs)} files / "
                  f"{tot} bytes, status says nfiles={st.get('nfiles')} "
                  f"sizes={len(st.get('sizes', {}))} "
                  f"bytes={st.get('bytes_expected')}", flush=True)
        return 0 if good else 1
    except Exception as e:  # noqa: BLE001
        print(f"no readable status.json ({e}) -- job desc: {desc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
