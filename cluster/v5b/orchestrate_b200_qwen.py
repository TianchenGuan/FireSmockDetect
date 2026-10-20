#!/usr/bin/env python
"""V5b: Qwen3.5-27B bf16 LoRA **smoke test** on the obsidian-pu B200 node (plan:
reports/v5b/B200_QWEN27B_PLAN.md). Reuses the connect/pull machinery of
orchestrate_b200_dino.py; entry = cluster/v5b/qwen_b200_entry.sh shipped as ENTRY_B64.

  python cluster/v5b/orchestrate_b200_qwen.py --tag v5bqwensmk --max-steps 200 --eval-n 200
"""
import argparse, base64, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orchestrate_b200_dino import (QUEUE, ROOT, YAML_TMPL, connect, job_state, kraken, log,  # noqa: E402
                                   pull_file, status)

IMAGE = "000000000001.dkr.ecr.ap-south-1.amazonaws.com/smoke-detect-vlm-b200:v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v5bB200_qwen27b_smoke")
    ap.add_argument("--tag", default="v5bqwensmk")
    ap.add_argument("--image-uri", default=IMAGE)
    ap.add_argument("--entry", default=str(ROOT / "cluster/v5b/qwen_b200_entry.sh"))
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--ga", type=int, default=4)
    ap.add_argument("--max-pixels", type=int, default=1048576)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--train-timeout", type=int, default=3600)
    ap.add_argument("--retain-s", type=int, default=3600)
    ap.add_argument("--poll-s", type=int, default=90)
    ap.add_argument("--max-wall-s", type=int, default=4 * 3600)
    ap.add_argument("--pending-abort-s", type=int, default=3600)
    ap.add_argument("--attach-job", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "models/v5b/b200" / args.run
    rep_dir = ROOT / "reports/v5b/b200_qwen" / args.run
    out_dir.mkdir(parents=True, exist_ok=True); rep_dir.mkdir(parents=True, exist_ok=True)
    summary_f = rep_dir / "orchestrator_summary.json"
    summary = {"run": args.run, "image": args.image_uri, "queue": QUEUE, "states": [], "pulls": [],
               "status_samples": [], "submitted_utc": datetime.now(timezone.utc).isoformat(),
               "config": {k: v for k, v in vars(args).items() if k not in ("entry",)}}
    if args.attach_job:
        job = args.attach_job; log(f"attached to {job}")
    else:
        envs = {"ENTRY_B64": base64.b64encode(Path(args.entry).read_bytes()).decode(),
                "RUN_NAME": args.run, "MAX_STEPS": str(args.max_steps), "BS": str(args.bs), "GA": str(args.ga),
                "MAX_PIXELS": str(args.max_pixels), "EVAL_N": str(args.eval_n), "TRAIN_TIMEOUT": str(args.train_timeout),
                "RETAIN_S": str(args.retain_s), "LR": "1e-4", "SAVE_STEPS": "100", "WORKERS": "4"}
        env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\"" for k, v in envs.items())
        yfile = ROOT / "cluster/generated" / f"b200qwen_{args.tag}.yaml"
        yfile.write_text(YAML_TMPL.format(job=args.tag, image_uri=args.image_uri, envs=env_yaml, queue=QUEUE))
        log(f"yaml {yfile}")
        if args.dry_run:
            return
        rc, out = kraken(["create", "-i", str(yfile)])
        (rep_dir / "submit.txt").write_text(out)
        if rc != 0:
            log(f"SUBMIT FAILED rc={rc}: {out[:800]}"); sys.exit(2)
        job = None
        for tok in out.replace('"', " ").replace(",", " ").split():
            tok = tok.strip(".:;")
            if tok.startswith(f"tianrgua-{args.tag}"):
                job = tok
        log(f"submitted {job}")
    summary["job"] = job; summary_f.write_text(json.dumps(summary, indent=1))

    t0 = time.time(); last_state = ""; first_running = None; final_pulled = False; last_phase = ""
    while time.time() - t0 < args.max_wall_s:
        st = job_state(job)
        if st != last_state:
            log(f"state {st}"); summary["states"].append([round(time.time() - t0), st, datetime.now(timezone.utc).isoformat()]); last_state = st
        base = st.split(":")[0]
        if base in ("SUCCEEDED", "FAILED", "STOPPED"):
            break
        if base == "RUNNING":
            if first_running is None:
                first_running = time.time(); summary["running_after_s"] = round(first_running - t0)
                summary["running_utc"] = datetime.now(timezone.utc).isoformat()
            s = status(job); phase = s.get("phase", "?"); line = json.dumps(s)[:230]
            if phase != last_phase or len(summary["status_samples"]) % 4 == 0:
                log(f"status: {line}")
            summary["status_samples"].append([round(time.time() - t0), line]); last_phase = phase
            rd = f"{s.get('root', '/mnt/local/v5b_qwen')}/{args.run}"
            if phase == "train" and len(summary["status_samples"]) % 2 == 0:
                rc, tail, _ = connect(job, f"tr '\\r' '\\n' < {rd}/train/console.log 2>/dev/null | grep -E \"'loss'|it/s|s/it|PARAMS|Error\" | tail -n 3 | cut -c1-220", 90)
                (rep_dir / "console_live_tail.txt").write_text(tail)
                log("  " + (tail.strip().splitlines() or [""])[-1][:200])
            if phase == "retain" and not final_pulled:
                log("final retrieval")
                for rel, dest, to in (("entry.log", rep_dir / "entry.log", 300), ("load_check.json", rep_dir / "load_check.json", 90),
                                      ("params.json", rep_dir / "params.json", 90), ("trainer_state.json", rep_dir / "trainer_state.json", 120),
                                      ("gpu_samples.csv", rep_dir / "gpu_samples.csv", 120), ("console_errors.txt", rep_dir / "console_errors.txt", 120),
                                      ("console_tail.txt", rep_dir / "console_tail.txt", 300), ("eval_preds.jsonl.gz", out_dir / "eval_preds.jsonl.gz", 600),
                                      ("final_adapter.tgz", out_dir / "final_adapter.tgz", 2400)):
                    ok, n, dt = pull_file(job, f"{rd}/{rel}", dest, to)
                    log(f"  {rel}: {n} B in {dt:.0f}s ok={ok}")
                    summary["pulls"].append({"file": rel, "bytes": n, "s": round(dt, 1), "ok": ok})
                (rep_dir / "eval_shardlogs.txt").write_text(connect(job, f"tail -n 4 {rd}/eval/shard*.log 2>/dev/null", 120)[1])
                final_pulled = True
                connect(job, "touch /tmp/ack", 60); log("acked -> pod exits 0")
                summary["acked_utc"] = datetime.now(timezone.utc).isoformat()
            summary_f.write_text(json.dumps(summary, indent=1))
        elif first_running is None and time.time() - t0 > args.pending_abort_s:
            log("not RUNNING before pending-abort -> stopping job"); kraken(["stop", "--job-name", job]); summary["aborted_pending"] = True
        time.sleep(args.poll_s if base == "RUNNING" and not final_pulled else 30)
    rc, desc = kraken(["describe", "--job-name", job]); (rep_dir / "describe.txt").write_text(desc)
    summary["final_state"] = last_state; summary["ended_utc"] = datetime.now(timezone.utc).isoformat()
    if first_running:
        summary["node_seconds_running_est"] = round(time.time() - first_running)
    summary_f.write_text(json.dumps(summary, indent=1)); log(f"terminal {last_state}; summary -> {summary_f}")


if __name__ == "__main__":
    main()
