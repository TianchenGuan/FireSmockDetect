#!/usr/bin/env python
"""V6: GLM-Image generation validation on the obsidian-pu B200 node.
Reuses the connect/pull machinery of cluster/v5b/orchestrate_b200_dino.py;
entry = cluster/v6/glm_b200_entry.sh shipped as ENTRY_B64.

  .venv/bin/python cluster/v6/orchestrate_b200_glm.py --tag v6glmb200
"""
import argparse
import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v5b"))
from orchestrate_b200_dino import (QUEUE, ROOT, YAML_TMPL, connect, job_state,  # noqa: E402
                                   kraken, log, pull_file, status)

IMAGE = "000000000001.dkr.ecr.ap-south-1.amazonaws.com/smoke-diffusion-b200:v6glm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v6B200_glm")
    ap.add_argument("--tag", default="v6glmb200")
    ap.add_argument("--image-uri", default=IMAGE)
    ap.add_argument("--entry", default=str(ROOT / "cluster/v6/glm_b200_entry.sh"))
    ap.add_argument("--gen-timeout", type=int, default=5400)
    ap.add_argument("--retain-s", type=int, default=5400)
    ap.add_argument("--poll-s", type=int, default=90)
    ap.add_argument("--max-wall-s", type=int, default=4 * 3600)
    ap.add_argument("--pending-abort-s", type=int, default=3600)
    ap.add_argument("--attach-job", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "data/v6/gen/glm_b200"
    rep_dir = ROOT / "reports/v6/b200_glm"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    summary_f = rep_dir / "orchestrator_summary.json"
    summary = {"run": args.run, "image": args.image_uri, "queue": QUEUE,
               "states": [], "pulls": [], "status_samples": [],
               "submitted_utc": datetime.now(timezone.utc).isoformat()}
    if args.attach_job:
        job = args.attach_job
        log(f"attached to {job}")
    else:
        envs = {"ENTRY_B64": base64.b64encode(Path(args.entry).read_bytes()).decode(),
                "RUN_NAME": args.run, "GEN_TIMEOUT": str(args.gen_timeout),
                "RETAIN_S": str(args.retain_s), "IMGSZ": "1024"}
        env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\""
                             for k, v in envs.items())
        yfile = ROOT / "cluster/generated" / f"b200glm_{args.tag}.yaml"
        yfile.write_text(YAML_TMPL.format(job=args.tag, image_uri=args.image_uri,
                                          envs=env_yaml, queue=QUEUE))
        log(f"yaml {yfile}")
        if args.dry_run:
            return
        rc, out = kraken(["create", "-i", str(yfile)])
        (rep_dir / "submit.txt").write_text(out)
        if rc != 0:
            log(f"SUBMIT FAILED rc={rc}: {out[:800]}")
            sys.exit(2)
        job = None
        for tok in out.replace('"', " ").replace(",", " ").split():
            tok = tok.strip(".:;")
            if tok.startswith(f"tianrgua-{args.tag}"):
                job = tok
        log(f"submitted {job}")
    summary["job"] = job
    summary_f.write_text(json.dumps(summary, indent=1))

    t0 = time.time()
    last_state = ""
    first_running = None
    final_pulled = False
    while time.time() - t0 < args.max_wall_s:
        st = job_state(job)
        if st != last_state:
            log(f"state {st}")
            summary["states"].append([round(time.time() - t0), st])
            last_state = st
        base = st.split(":")[0]
        if base in ("SUCCEEDED", "FAILED", "STOPPED"):
            break
        if base == "RUNNING":
            if first_running is None:
                first_running = time.time()
                summary["running_after_s"] = round(first_running - t0)
            s = status(job)
            phase = s.get("phase", "?")
            summary["status_samples"].append([round(time.time() - t0),
                                              json.dumps(s)[:200]])
            log(f"status: {json.dumps(s)[:200]}")
            rd = f"{s.get('root', '/mnt/local/v6_glm')}/{args.run}"
            if phase == "retain" and not final_pulled:
                log("final retrieval")
                for rel, dest, to in (("entry.log", rep_dir / "entry.log", 300),
                                      ("load_check.json", rep_dir / "load_check.json", 90),
                                      ("latency.txt", rep_dir / "latency.txt", 90),
                                      ("gpu_samples.csv", rep_dir / "gpu_samples.csv", 180),
                                      ("glmout.tgz", out_dir / "glmout.tgz", 5400)):
                    ok, n, dt = pull_file(job, f"{rd}/{rel}", dest, to)
                    log(f"  {rel}: {n} B in {dt:.0f}s ok={ok}")
                    summary["pulls"].append({"file": rel, "bytes": n,
                                             "s": round(dt, 1), "ok": ok})
                final_pulled = True
                connect(job, "touch /tmp/ack", 60)
                log("acked -> pod exits 0")
            summary_f.write_text(json.dumps(summary, indent=1))
        elif first_running is None and time.time() - t0 > args.pending_abort_s:
            log("not RUNNING before pending-abort -> stopping job")
            kraken(["stop", "--job-name", job])
            summary["aborted_pending"] = True
        time.sleep(args.poll_s if base == "RUNNING" and not final_pulled else 30)
    rc, desc = kraken(["describe", "--job-name", job])
    (rep_dir / "describe.txt").write_text(desc)
    summary["final_state"] = last_state
    if first_running:
        summary["node_seconds_running_est"] = round(time.time() - first_running)
    summary_f.write_text(json.dumps(summary, indent=1))
    log(f"terminal {last_state}; summary -> {summary_f}")
    # extract outputs if pulled
    tgz = out_dir / "glmout.tgz"
    if tgz.exists():
        import tarfile
        with tarfile.open(tgz) as tf:
            tf.extractall(out_dir, filter="data")
        src = out_dir / "out"
        if src.exists():
            for p in src.iterdir():
                p.rename(out_dir / p.name)
            src.rmdir()
        tgz.unlink()
        n = len(list(out_dir.glob("*.jpg")))
        log(f"extracted {n} jpgs -> {out_dir}")


if __name__ == "__main__":
    main()
