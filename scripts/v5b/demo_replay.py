#!/usr/bin/env python
"""Scripted HISTORICAL REPLAY of one demo sequence through a running service's
POST /detect (source=historical_replay, hard-wired) in FAST or ACCURATE mode,
saving a run file that GET /demo?run=<run_id> renders exactly like a live UI
replay. Frames come from the server's own /demo/sequences allow-list.

  python scripts/v5b/demo_replay.py --server http://127.0.0.1:8013 \
      --seq val_force-06_cabanelle-125_20240403T101630 --mode accurate --run-id acc_sdis_cab125

Writes <runs-dir>/<run_id>.json (default data/v5b/demo/runs = the server's
SMOKE_DEMO_RUNS_DIR default). Output is a replay of stored frames, never a field run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.serve.replay_client import SOURCE, build_form  # noqa: E402  (source=historical_replay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8013")
    ap.add_argument("--seq", required=True, help="seq id from GET /demo/sequences")
    ap.add_argument("--mode", default=None, choices=[None, "fast", "accurate"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-dir", default=str(ROOT / "data/v5b/demo/runs"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300)
    a = ap.parse_args()
    if a.server.rstrip("/").endswith(":8000"):
        raise SystemExit("refusing to replay against the production port 8000")

    hz = requests.get(f"{a.server}/healthz", timeout=30).json()
    seqs = requests.get(f"{a.server}/demo/sequences", timeout=30).json()["sequences"]
    seq = next((s for s in seqs if s["id"] == a.seq), None)
    if seq is None:
        raise SystemExit(f"{a.seq!r} not in /demo/sequences: {[s['id'] for s in seqs]}")
    frames = seq["frames"][: a.limit] if a.limit else seq["frames"]
    mode = a.mode or hz["modes"]["default"]
    run_tag = f"demo-{a.run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    device_id = f"demo-replay-{a.seq}-{mode}"
    print(f"HISTORICAL REPLAY (source={SOURCE}) seq={a.seq} mode={mode} n={len(frames)} device={device_id}")
    t0_all = time.time()
    out_frames, states, first_alert = [], {}, None
    t0 = frames[0]["ts"]
    for f in frames:
        img = requests.get(f"{a.server}{f['url']}", timeout=60)
        img.raise_for_status()
        form = build_form(device_id, f["ts"], run_tag, {"frame": f["name"], "seq_id": a.seq, "demo": "v5b"})
        if a.mode:
            form["mode"] = a.mode
        t1 = time.perf_counter()
        r = requests.post(f"{a.server}/detect", files={"image": (f["name"], img.content, "image/jpeg")},
                          data=form, timeout=a.timeout)
        wall = round((time.perf_counter() - t1) * 1000, 1)
        rec = {"idx": f["idx"], "name": f["name"], "ts": f["ts"], "offset_s": round(f["ts"] - t0, 1),
               "gt_smoke": f.get("gt_smoke"), "gt_boxes_xywhn": f.get("gt_boxes_xywhn", []),
               "status": r.status_code, "client_wall_ms": wall}
        if r.status_code == 200:
            j = r.json()
            rec["response"] = j
            st = j["alert"]["state"]
            states[st] = states.get(st, 0) + 1
            if first_alert is None and st in ("provisional", "confirmed"):
                first_alert = {"idx": f["idx"], "name": f["name"], "offset_s": rec["offset_s"], "state": st}
            per = ""
            if j.get("mode") == "accurate":
                per = " | " + " ".join(f"{n}:{m.get('max_smoke_score')}/{(m.get('alert') or {}).get('state')}/{m.get('latency_ms')}ms"
                                      for n, m in j["models"].items())
            print(f"[{f['idx']+1}/{len(frames)}] {f['name']} off={rec['offset_s']:>7.1f} "
                  f"max={j['max_smoke_score']:.3f} alert={st} lat={j['latency_ms']}ms wall={wall}ms{per}")
        else:
            rec["error"] = r.text[:300]
            print(f"[{f['idx']+1}/{len(frames)}] {f['name']} HTTP {r.status_code}: {r.text[:120]}")
        out_frames.append(rec)
    ok = [x["response"] for x in out_frames if "response" in x]
    lat = [x["latency_ms"] for x in ok]
    per_model = {}
    for j in ok:
        for n, m in (j.get("models") or {}).items():
            if m.get("latency_ms") is not None:
                per_model.setdefault(n, []).append(m["latency_ms"])
    summary = {
        "n_frames": len(out_frames), "n_ok": len(ok), "alerts_by_state": states, "first_alert": first_alert,
        "latency_ms": {"median": statistics.median(lat) if lat else None, "mean": round(statistics.fmean(lat), 1) if lat else None,
                       "max": max(lat) if lat else None},
        "per_model_latency_ms": {n: {"median": statistics.median(v), "mean": round(statistics.fmean(v), 1), "max": max(v)}
                                 for n, v in per_model.items()},
        "wall_s": round(time.time() - t0_all, 1)}
    run = {"run_id": a.run_id, "seq_id": a.seq, "mode": mode, "source": SOURCE, "run_tag": run_tag,
           "device_id": device_id, "server": a.server, "created_utc": datetime.now(timezone.utc).isoformat(),
           "model_version": ok[0]["model_version"] if ok else None,
           "healthz_modes": hz.get("modes"), "sequence": {k: v for k, v in seq.items() if k != "frames"},
           "summary": summary, "frames": out_frames,
           "note": "historical replay of stored frames through /detect; not a field run"}
    rd = Path(a.runs_dir); rd.mkdir(parents=True, exist_ok=True)
    p = rd / f"{a.run_id}.json"
    json.dump(run, open(p, "w"), indent=1)
    print("summary", json.dumps(summary))
    print("saved", p)


if __name__ == "__main__":
    main()
