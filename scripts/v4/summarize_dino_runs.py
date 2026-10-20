"""Summarize iteration-4 DINOv3 runs from fetched segment artifacts.

Per run: segments, kraken durations (node-hours), steps reached, total steps,
epochs/exposures, best COCO mAP per eval, param counts from the checkpoint,
OOM/retry evidence from console tails.

  .venv/bin/python scripts/v4/summarize_dino_runs.py [--runs a b c]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path("/local/home/tianrgua/cvproject")
KRAKEN = "kraken"
RUNS = ["v4B1_frozen_L", "v4B4_lora_L", "v4B5_full_L", "v4B6_lora_Hplus"]
DUR = re.compile(r"(\d+)\s+(second|minute|hour)")


def kraken_durations():
    p = subprocess.run([KRAKEN, "-p", "green", "jobs", "list", "-r", "tianrgua"],
                       capture_output=True, text=True, timeout=180)
    out = {}
    for ln in p.stdout.splitlines():
        parts = ln.split()
        if len(parts) < 2 or not parts[0].startswith("tianrgua-"):
            continue
        m = DUR.search(ln)
        mins = 0.0
        if m:
            n, unit = int(m.group(1)), m.group(2)
            mins = n * {"second": 1 / 60, "minute": 1.0, "hour": 60.0}[unit]
        elif " a minute" in ln:
            mins = 1.0
        out[parts[0]] = {"status": parts[1], "minutes": round(mins, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--out", default=str(ROOT / "models/v4/dino_run_summary.json"))
    a = ap.parse_args()
    kd = kraken_durations()
    summary = {}
    for run in a.runs:
        log = ROOT / "logs/v4" / f"dino_{run}.log"
        rd = ROOT / "models/v3/runs" / run
        jobs, seg_status = [], {}
        if log.exists():
            txt = log.read_text(errors="replace")
            jobs = re.findall(r"submitted (tianrgua-\S+)", txt)
            complete = "RUN COMPLETE" in txt
        else:
            complete = False
        # kraken list ages out terminal jobs -> derive per-segment wall time
        # from the orchestrator's own "[Ns] STATE" poll lines (authoritative
        # for node-hours: 1 job = 1 whole node while it lives).
        seg_seconds = []
        if log.exists():
            for block in txt.split("[seg ")[1:]:
                secs = [int(m) for m in re.findall(r"\[(\d+)s\]", block)]
                if secs:
                    seg_seconds.append(max(secs))
        node_minutes = sum(seg_seconds) / 60.0
        live = sum(kd.get(j, {}).get("minutes", 0.0) for j in jobs
                   if kd.get(j, {}).get("status", "").startswith(("RUNNING", "PENDING")))
        segs = sorted(rd.glob("seg*/status.json"))
        for sf in segs:
            try:
                seg_status[sf.parent.name] = json.loads(sf.read_text())
            except json.JSONDecodeError:
                seg_status[sf.parent.name] = {"phase": "unparseable"}
        evals, last_step, total_steps = [], None, None
        seen_eval_steps = set()
        # metrics.jsonl lives on /scratch and is APPENDED across segments, so
        # each fetched copy is cumulative -> dedupe eval rows by step.
        for mf in sorted(rd.glob("seg*/metrics.jsonl")):
            for line in mf.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "eval" in r and r.get("step") not in seen_eval_steps:
                    seen_eval_steps.add(r.get("step"))
                    evals.append({"seg": mf.parent.name, "step": r.get("step"),
                                  **{k: round(v, 4) for k, v in r["eval"].items()
                                     if isinstance(v, (int, float))}})
                if "step" in r:
                    last_step = r["step"]
        oom = []
        for cf in sorted(rd.glob("seg*/console_tail.txt")):
            t = cf.read_text(errors="replace")
            if "OutOfMemoryError" in t or "CUDA out of memory" in t:
                oom.append(cf.parent.name)
            m = re.search(r"step \d+/(\d+)", t)
            if m:
                total_steps = int(m.group(1))
        summary[run] = {
            "complete": complete, "n_segments_submitted": len(jobs),
            "jobs": [{"name": j, **kd.get(j, {})} for j in jobs],
            "node_hours": round(node_minutes / 60.0, 2),
            "node_hours_live_jobs": round(live / 60.0, 2),
            "segment_wall_s": seg_seconds,
            "last_step": last_step, "total_steps": total_steps,
            "segment_phases": {k: v.get("phase") for k, v in seg_status.items()},
            "evals": evals, "best_mAP": (max((e.get("mAP", -1) for e in evals),
                                             default=None)),
            "oom_segments": oom,
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2))
    for run, d in summary.items():
        print(f"{run}: complete={d['complete']} segs={d['n_segments_submitted']} "
              f"node_h={d['node_hours']} step={d['last_step']}/{d['total_steps']} "
              f"best_mAP={d['best_mAP']} oom={d['oom_segments']}")
    print("saved:", a.out)


if __name__ == "__main__":
    raise SystemExit(main())
