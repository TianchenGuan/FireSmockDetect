#!/usr/bin/env python
"""Summarise saved demo replay runs (data/v5b/demo/runs/*.json) as markdown tables
for reports/v5b/DEMO_V5B.md. Read-only; numbers are historical-replay results."""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/v5b/demo/runs"
STATES = ("none", "watch", "provisional", "confirmed")

def med(v): return round(statistics.median(v), 1) if v else None
def q(v, p): 
    if not v: return None
    s = sorted(v); return round(s[min(len(s) - 1, int(p * len(s)))], 1)

rows = []
runs = {p.stem: json.load(open(p)) for p in sorted(RUNS.glob("*.json"))}
print("| run | seq | mode | frames | none/watch/prov/conf | first provisional+ (frame, offset s) | server latency ms median / p90 / max | per-model latency ms median | per-model alert counts (none/watch/prov/conf) |")
print("|---|---|---|---|---|---|---|---|---|")
for rid, r in runs.items():
    fr = [f for f in r["frames"] if "response" in f]
    st = {s: 0 for s in STATES}
    for f in fr: st[f["response"]["alert"]["state"]] += 1
    lat = [f["response"]["latency_ms"] for f in fr]
    fa = r["summary"].get("first_alert")
    per_lat, per_alert = {}, {}
    for f in fr:
        for n, m in (f["response"].get("models") or {}).items():
            per_lat.setdefault(n, []).append(m["latency_ms"])
            per_alert.setdefault(n, {s: 0 for s in STATES})[(m.get("alert") or {}).get("state", "none")] += 1
    pl = ", ".join(f"{n} {med(v)}" for n, v in per_lat.items()) or "-"
    pa = ", ".join(f"{n} {a['none']}/{a['watch']}/{a['provisional']}/{a['confirmed']}" for n, a in per_alert.items()) or "-"
    print(f"| {rid} | {r['seq_id']} | {r['mode']} | {len(fr)} | {st['none']}/{st['watch']}/{st['provisional']}/{st['confirmed']} | "
          f"{(str(fa['idx']) + ', ' + str(fa['offset_s'])) if fa else 'none'} | {med(lat)} / {q(lat, 0.9)} / {max(lat) if lat else None} | {pl} | {pa} |")

# frame-level comparison fast vs accurate on the same sequence
print()
print("| seq | frames | dataset label smoke frames | FAST frames in provisional/confirmed | ACCURATE frames in provisional/confirmed | both | FAST only | ACCURATE only |")
print("|---|---|---|---|---|---|---|---|")
by_seq = {}
for rid, r in runs.items(): by_seq.setdefault(r["seq_id"], {})[r["mode"]] = r
for sid, d in by_seq.items():
    if "fast" not in d or "accurate" not in d: continue
    fa = {f["idx"]: f["response"]["alert"]["state"] in ("provisional", "confirmed") for f in d["fast"]["frames"] if "response" in f}
    ac = {f["idx"]: f["response"]["alert"]["state"] in ("provisional", "confirmed") for f in d["accurate"]["frames"] if "response" in f}
    gt = [f.get("gt_smoke") for f in d["fast"]["frames"]]
    n_gt = sum(1 for g in gt if g is True); gt_s = str(n_gt) if any(g is not None for g in gt) else "n/a (clip label: " + str(d["fast"]["sequence"].get("label_pattern")) + ")"
    both = sum(1 for i in fa if fa[i] and ac.get(i)); fo = sum(1 for i in fa if fa[i] and not ac.get(i)); ao = sum(1 for i in ac if ac[i] and not fa.get(i))
    print(f"| {sid} | {len(fa)} | {gt_s} | {sum(fa.values())} | {sum(ac.values())} | {both} | {fo} | {ao} |")
