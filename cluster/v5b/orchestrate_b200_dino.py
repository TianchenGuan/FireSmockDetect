#!/usr/bin/env python
"""V5b: run the high-res DINOv3-L LoRA detection job on the obsidian-pu B200 node and
retrieve everything through `kraken jobs connect` (the only working channel —
reports/v5b/B200_PROBE.md §3/§5: no S3 egress, no writable /scratch, exit code not
surfaced). Single segment: image smoke-dino3-b200:v1 carries code+data+backbone, the
entry script (cluster/v5b/dino_b200_entry.sh) is shipped inline as ENTRY_B64.

  python cluster/v5b/orchestrate_b200_dino.py --run v5bB200_lora_L_1920 --tag v5bdinohr \
      --memprobe-bs "16 8" --bs 4 --epochs 10 --train-budget-s 23400
  python cluster/v5b/orchestrate_b200_dino.py --run ... --attach-job <job>   # re-attach

Registry reserve/release is done by the caller. Never writes credentials anywhere
(kraken auth = Midway; no AWS creds needed for this flow).
"""
import argparse, base64, hashlib, json, shlex, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KRAKEN = "kraken"
PROJECT = "obsidian-pu"
QUEUE = "obsidian-pu-hyperpod-p6-b200.48xlarge-ap-south-1-tas"
IMAGE = "000000000001.dkr.ecr.ap-south-1.amazonaws.com/smoke-dino3-b200:v1"
NOISE = ("Updated kubeconfig", "Unable to use a TTY")

YAML_TMPL = """jobNamePrefix: tianrgua-{job}
jobPriority: LOW
containers:
  - imageUri: {image_uri}
    command: /bin/bash
    args:
      - -c
      - echo "$ENTRY_B64" | base64 -d > /tmp/entry.sh && bash /tmp/entry.sh
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


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {msg}", flush=True)


def kraken(args, timeout=180):
    p = subprocess.run([KRAKEN, "-p", PROJECT, "jobs"] + args, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def job_state(job):
    rc, out = kraken(["list", "-r", "tianrgua"])
    for line in out.splitlines():
        if line.startswith(job):
            return line.split()[1]
    return ""


def connect(job, script, timeout=120):
    """run `sh -c <script>` in the pod; returns (rc, stdout-without-noise, stderr)."""
    cmd = "sh -c " + shlex.quote(script)
    try:
        p = subprocess.run([KRAKEN, "-p", PROJECT, "jobs", "connect", "-j", job, "-c", cmd],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    out = "\n".join(l for l in p.stdout.splitlines() if not l.startswith(NOISE))
    return p.returncode, out, p.stderr


def pull_file(job, remote, dest: Path, timeout=1500):
    """snapshot -> sha256 -> base64 in ONE connect call (files may be appended while we
    read, so hash the snapshot, not the live file); verify locally; 2 attempts.
    Returns (ok, bytes, seconds)."""
    for attempt in range(2):
        t0 = time.time()
        rc, out, err = connect(job, f"test -f {remote} && cp {remote} /tmp/.pull_snap && "
                                    f"sha256sum /tmp/.pull_snap | cut -d' ' -f1 && base64 -w0 /tmp/.pull_snap && echo && sleep 0.3", timeout)
        dt = time.time() - t0
        lines = [l for l in out.splitlines() if l.strip()]
        want = lines[0].strip() if lines else ""
        if rc != 0 or len(want) != 64:
            log(f"  pull {remote}: no file/sha (rc={rc} {err.strip()[:80]} {out[:60]!r})")
            return False, 0, dt
        try:
            data = base64.b64decode("".join("".join(lines[1:]).split()))
        except Exception:
            data = b""
        if hashlib.sha256(data).hexdigest() == want:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return True, len(data), dt
        log(f"  pull {remote}: sha mismatch ({len(data)} B) attempt {attempt+1}")
    return False, 0, 0.0


def status(job):
    rc, out, err = connect(job, "cat /tmp/status.json 2>/dev/null", 90)
    for l in out.splitlines():
        if l.startswith("{"):
            try:
                return json.loads(l)
            except json.JSONDecodeError:
                pass
    return {"phase": "unreadable", "rc": rc, "raw": (out or err)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v5bB200_lora_L_1920")
    ap.add_argument("--tag", default="v5bdinohr")
    ap.add_argument("--image-uri", default=IMAGE)
    ap.add_argument("--entry", default=str(ROOT / "cluster/v5b/dino_b200_entry.sh"))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1088)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--memprobe-bs", default="16 8")
    ap.add_argument("--memprobe-steps", type=int, default=40)
    ap.add_argument("--train-budget-s", type=int, default=23400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-steps", type=int, default=400)
    ap.add_argument("--eval-max", type=int, default=1500)
    ap.add_argument("--retain-s", type=int, default=5400)
    ap.add_argument("--poll-s", type=int, default=120)
    ap.add_argument("--metrics-every-s", type=int, default=600)
    ap.add_argument("--max-wall-s", type=int, default=9 * 3600)
    ap.add_argument("--pending-abort-s", type=int, default=2700)
    ap.add_argument("--attach-job", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "models/v5b/b200" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = ROOT / "reports/v5b/b200_dino" / args.run
    rep_dir.mkdir(parents=True, exist_ok=True)
    summary_f = rep_dir / "orchestrator_summary.json"
    summary = {"run": args.run, "image": args.image_uri, "queue": QUEUE, "states": [], "pulls": [],
               "status_samples": [], "submitted_utc": datetime.now(timezone.utc).isoformat()}

    if args.attach_job:
        job = args.attach_job
        log(f"attached to {job}")
    else:
        envs = {
            "ENTRY_B64": base64.b64encode(Path(args.entry).read_bytes()).decode(),
            "RUN_NAME": args.run, "WIDTH": str(args.width), "HEIGHT": str(args.height),
            "BS": str(args.bs), "EPOCHS": str(args.epochs), "WORKERS": str(args.workers),
            "MEMPROBE_BS": args.memprobe_bs, "MEMPROBE_STEPS": str(args.memprobe_steps),
            "TRAIN_BUDGET_S": str(args.train_budget_s), "SAVE_STEPS": str(args.save_steps),
            "EVAL_MAX": str(args.eval_max), "RETAIN_S": str(args.retain_s),
            "LR": "1e-4", "BACKBONE_LR": "1e-4", "WD": "1e-4", "WARMUP": "500", "LORA_R": "16", "SEED": "0",
        }
        env_yaml = "\n".join(f"      - name: {k}\n        value: \"{v}\"" for k, v in envs.items())
        yfile = ROOT / "cluster/generated" / f"b200dino_{args.tag}.yaml"
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
    summary["job"] = job
    summary_f.write_text(json.dumps(summary, indent=1))

    t0 = time.time(); last_state = ""; first_running = None; last_metrics = 0
    pulled_evals = -1; final_pulled = False; last_phase = ""
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
            s = status(job)
            phase = s.get("phase", "?")
            line = json.dumps(s)[:230]
            if phase != last_phase or len(summary["status_samples"]) % 5 == 0:
                log(f"status: {line}")
            summary["status_samples"].append([round(time.time() - t0), line]); last_phase = phase
            # periodic metrics pull (small) while training
            if phase in ("train", "memprobe") and time.time() - last_metrics > args.metrics_every_s // (3 if pulled_evals < 0 else 1):
                ok, n, dt = pull_file(job, f"{s.get('root','/mnt/local/v5b_dino')}/{args.run}/metrics.jsonl", out_dir / "metrics.jsonl", 300)
                rc, tail, _ = connect(job, f"tail -n 4 {s.get('root','/mnt/local/v5b_dino')}/{args.run}/console.log 2>/dev/null | cut -c1-200", 90)
                (rep_dir / "console_live_tail.txt").write_text(tail)
                log(f"  metrics.jsonl {n} B ok={ok} | " + (tail.strip().splitlines() or [""])[-1][:160])
                last_metrics = time.time()
            # a new best.pt after an epoch eval -> pull a model-only copy (fast).
            # The pod status.json is only rewritten at phase changes, so count evals
            # from the metrics.jsonl we just pulled (authoritative) instead.
            evals = 0
            mf = out_dir / "metrics.jsonl"
            if mf.exists():
                evals = sum(1 for l in mf.read_text().splitlines() if '"eval"' in l)
                if evals > max(pulled_evals, 0):
                    last_metrics = 0  # force a fresh metrics pull next loop too
            if phase == "train" and evals > max(pulled_evals, 0):
                rd = f"{s.get('root','/mnt/local/v5b_dino')}/{args.run}"
                rc, o, e = connect(job, f"cd {rd} && python -c \"import torch;ck=torch.load('best.pt',map_location='cpu',weights_only=False);torch.save({{k:v for k,v in ck.items() if k!='opt'}},'best_model.pt');print(ck['step'],ck['best_map'])\"", 300)
                ok, n, dt = pull_file(job, f"{rd}/best_model.pt", out_dir / "best_model.pt", 1200)
                log(f"  best_model.pt (evals={evals}, {o.strip()[:40]}) {n} B in {dt:.0f}s ok={ok}")
                summary["pulls"].append({"file": "best_model.pt", "evals": evals, "bytes": n, "s": round(dt, 1), "ok": ok, "meta": o.strip()[:60]})
                if ok:
                    pulled_evals = evals
            if phase in ("inferred", "retain", "trained") and not final_pulled and (phase != "trained" or s.get("done") is not None):
                if phase == "trained":
                    # inference still to come; wait for it unless training failed
                    pass
                else:
                    rd = f"{s.get('root','/mnt/local/v5b_dino')}/{args.run}"
                    log("final retrieval")
                    for rel, dest, to in (("metrics.jsonl", out_dir / "metrics.jsonl", 300),
                                          ("memprobe.jsonl", rep_dir / "memprobe.jsonl", 120),
                                          ("entry.log", rep_dir / "entry.log", 300),
                                          ("console_errors.txt", rep_dir / "console_errors.txt", 120),
                                          ("preds/latency_b200.json", rep_dir / "latency_b200.json", 120),
                                          ("sdis_val_preds.jsonl.gz", out_dir / "sdis_val_preds.jsonl.gz", 1500),
                                          ("best_model.pt", out_dir / "best_model.pt", 1500),
                                          ("last_model.pt", out_dir / "last_model.pt", 1500),
                                          ("best.pt", out_dir / "best.pt", 2400)):
                        ok, n, dt = pull_file(job, f"{rd}/{rel}", dest, to)
                        log(f"  {rel}: {n} B in {dt:.0f}s ok={ok}")
                        summary["pulls"].append({"file": rel, "bytes": n, "s": round(dt, 1), "ok": ok})
                    rc, tail, _ = connect(job, f"tail -c 20000 {rd}/console.log; echo; echo ==== ; tail -n 5 {rd}/preds/shard0.log 2>/dev/null; nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\\n' ' '", 120)
                    (rep_dir / "console_tail.txt").write_text(tail)
                    (rep_dir / "preds_shardlogs.txt").write_text(connect(job, f"cat {rd}/preds/shard*.log 2>/dev/null | tail -n 60", 120)[1])
                    final_pulled = True
                    connect(job, "touch /tmp/ack", 60)
                    log("acked -> pod exits 0")
                    summary["acked_utc"] = datetime.now(timezone.utc).isoformat()
            summary_f.write_text(json.dumps(summary, indent=1))
        elif first_running is None and time.time() - t0 > args.pending_abort_s:
            log("not RUNNING before pending-abort -> stopping job"); kraken(["stop", "--job-name", job])
            summary["aborted_pending"] = True
        time.sleep(args.poll_s if base == "RUNNING" and not final_pulled else 30)
    rc, desc = kraken(["describe", "--job-name", job])
    (rep_dir / "describe.txt").write_text(desc)
    summary["final_state"] = last_state
    summary["ended_utc"] = datetime.now(timezone.utc).isoformat()
    if first_running:
        summary["node_seconds_running_est"] = round(time.time() - first_running)
    summary_f.write_text(json.dumps(summary, indent=1))
    log(f"terminal {last_state}; summary -> {summary_f}")


if __name__ == "__main__":
    main()
