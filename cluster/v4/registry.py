"""Shared v4 job registry + concurrency limiter (all subagents, one file).

Every v4 kraken submission MUST reserve nodes here first, so pending jobs
cannot later start together and exceed the 16-node (128 L40S) global cap.
Whole-node accounting: 1 job = N nodes = N*8 GPUs while not terminal.

Usage (from any shell/agent, safe under concurrency via flock):
  python cluster/v4/registry.py reserve --job <name> --nodes 1 --track yoloA \
      --note "arm A ctrl seed0"          # exit 0 = reserved, exit 3 = would exceed cap
  python cluster/v4/registry.py release --job <name> --outcome SUCCEEDED
  python cluster/v4/registry.py status
  python cluster/v4/registry.py sync    # cross-check active entries against kraken

Budget window: cluster/v4/BUDGET_V4.json (never reset window_start_utc).
Allocation GPU-hours are reconciled from kraken history at close (authoritative);
this registry enforces the concurrency ceiling and gives a live estimate.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import os
HERE = Path(__file__).resolve().parent
# SMOKE_REGISTRY_DIR selects the iteration (default v4 files). v5 prep uses
# SMOKE_REGISTRY_DIR=cluster/v5 with its own BUDGET_V5.json + registry.json.
HERE = Path(os.environ.get("SMOKE_REGISTRY_DIR", str(HERE))).resolve()
REG = HERE / "registry.json"
BUDGET = json.loads(next(HERE.glob("BUDGET_V*.json")).read_text())
KRAKEN = "kraken"


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load():
    if REG.exists():
        return json.loads(REG.read_text())
    return {"jobs": {}}


def _locked(fn):
    lock = HERE / ".registry.lock"
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _active_nodes(reg):
    return sum(j["nodes"] for j in reg["jobs"].values() if j["state"] == "active")


def _hours_spent(reg):
    tot = 0.0
    for j in reg["jobs"].values():
        end = j.get("released_utc")
        t0 = datetime.fromisoformat(j["reserved_utc"].replace("Z", "+00:00"))
        t1 = (datetime.fromisoformat(end.replace("Z", "+00:00")) if end
              else datetime.now(timezone.utc))
        tot += j["nodes"] * (t1 - t0).total_seconds() / 3600.0
    return tot


def cmd_reserve(a):
    def go():
        reg = _load()
        if a.job in reg["jobs"] and reg["jobs"][a.job]["state"] == "active":
            print(f"already active: {a.job}")
            return 0
        deadline = datetime.fromisoformat(
            BUDGET["window_deadline_utc"].replace("Z", "+00:00"))
        if datetime.now(timezone.utc) >= deadline:
            print("DENIED: past v4 window deadline")
            return 4
        used = _active_nodes(reg)
        caps = BUDGET.get("max_nodes_by_type")
        if caps:
            used_t = sum(j["nodes"] for j in reg["jobs"].values()
                         if j["state"] == "active" and j.get("gpu_type", "L40S") == a.gpu_type)
            if used_t + a.nodes > caps.get(a.gpu_type, 0):
                print(f"DENIED: {a.gpu_type} {used_t}+{a.nodes} nodes > type cap "
                      f"{caps.get(a.gpu_type, 0)}")
                return 3
        if used + a.nodes > BUDGET["max_concurrent_nodes"]:
            print(f"DENIED: {used}+{a.nodes} nodes > cap "
                  f"{BUDGET['max_concurrent_nodes']}")
            return 3
        spent = _hours_spent(reg)
        if spent > BUDGET["max_allocation_node_hours"]:
            print(f"DENIED: est {spent:.1f} node-h > cap "
                  f"{BUDGET['max_allocation_node_hours']}")
            return 5
        reg["jobs"][a.job] = {"nodes": a.nodes, "track": a.track,
                              "note": a.note, "state": "active",
                              "gpu_type": a.gpu_type,
                              "reserved_utc": _now()}
        REG.write_text(json.dumps(reg, indent=1))
        print(f"RESERVED {a.job} nodes={a.nodes} "
              f"(active {used + a.nodes}/{BUDGET['max_concurrent_nodes']}, "
              f"est spent {spent:.1f}/{BUDGET['max_allocation_node_hours']} node-h)")
        return 0
    raise SystemExit(_locked(go))


def cmd_release(a):
    def go():
        reg = _load()
        if a.job not in reg["jobs"]:
            print(f"unknown job: {a.job}")
            return 1
        j = reg["jobs"][a.job]
        if j["state"] == "active":
            j["state"] = "released"
            j["released_utc"] = _now()
            j["outcome"] = a.outcome
            REG.write_text(json.dumps(reg, indent=1))
        print(f"RELEASED {a.job} outcome={a.outcome} "
              f"(active now {_active_nodes(reg)})")
        return 0
    raise SystemExit(_locked(go))


def cmd_status(a):
    reg = _load()
    act = {k: v for k, v in reg["jobs"].items() if v["state"] == "active"}
    by_type = {}
    for j in reg["jobs"].values():
        t = j.get("gpu_type", "L40S")
        t0 = datetime.fromisoformat(j["reserved_utc"].replace("Z", "+00:00"))
        t1 = (datetime.fromisoformat(j["released_utc"].replace("Z", "+00:00"))
              if j.get("released_utc") else datetime.now(timezone.utc))
        by_type[t] = round(by_type.get(t, 0.0) + j["nodes"] * (t1 - t0).total_seconds() / 3600, 2)
    print(json.dumps({
        "active_jobs": act,
        "node_hours_by_gpu_type": by_type,
        "active_nodes_by_type": {t: sum(j["nodes"] for j in act.values() if j.get("gpu_type","L40S")==t) for t in set(j.get("gpu_type","L40S") for j in act.values())},
        "active_nodes": _active_nodes(reg),
        "node_cap": BUDGET["max_concurrent_nodes"],
        "est_node_hours_spent": round(_hours_spent(reg), 2),
        "node_hour_cap": BUDGET["max_allocation_node_hours"],
        "deadline_utc": BUDGET["window_deadline_utc"],
        "utc_now": _now(),
    }, indent=1))


def cmd_sync(a):
    """Release registry entries whose kraken job is terminal (crash safety)."""
    p = subprocess.run([KRAKEN, "-p", "green", "jobs", "list", "-r", "tianrgua"],
                       capture_output=True, text=True, timeout=120)
    lines = p.stdout.splitlines()
    term = {}
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 2 and parts[1] in {"SUCCEEDED", "FAILED", "STOPPED"}:
            term[parts[0]] = parts[1]

    def go():
        reg = _load()
        changed = []
        for name, j in reg["jobs"].items():
            if j["state"] == "active" and name in term:
                j["state"] = "released"
                j["released_utc"] = _now()
                j["outcome"] = term[name] + "(sync)"
                changed.append(name)
        if changed:
            REG.write_text(json.dumps(reg, indent=1))
        print(f"synced, released: {changed}")
        return 0
    _locked(go)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reserve")
    r.add_argument("--job", required=True)
    r.add_argument("--nodes", type=int, default=1)
    r.add_argument("--track", required=True)
    r.add_argument("--note", default="")
    r.add_argument("--gpu-type", default="L40S")
    r.set_defaults(fn=cmd_reserve)
    rl = sub.add_parser("release")
    rl.add_argument("--job", required=True)
    rl.add_argument("--outcome", default="UNKNOWN")
    rl.set_defaults(fn=cmd_release)
    st = sub.add_parser("status")
    st.set_defaults(fn=cmd_status)
    sy = sub.add_parser("sync")
    sy.set_defaults(fn=cmd_sync)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
