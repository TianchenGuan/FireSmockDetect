"""COMPUTE_LEDGER.csv from actual scheduler records (kraken jobs list).

Inputs (dumped by the coordinator, tokens scrubbed):
  reports/v4/ledger/kraken_<project>_raw.json   (--json: jobId, nodeCount,
                                                  queue, status, creation/
                                                  completion times)
  reports/v4/ledger/kraken_<project>_table.txt  (text table: START time,
                                                  which the JSON lacks)
Allocated hours = completion - start (queue wait excluded). Whole-node
accounting: GPU-hours = node-hours x GPUs/node from the nodepool name.
No utilization telemetry exists for these pods (no logs, no metrics
endpoint) -> utilization column is 'n/a'. Jobs still RUNNING are stamped
with the dump time and flagged.

Usage: python scripts/v4/make_compute_ledger.py --out reports/v4/COMPUTE_LEDGER.csv
"""
import argparse, csv, glob, json, re
from datetime import datetime, timezone, timedelta
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="reports/v4/COMPUTE_LEDGER.csv")
ap.add_argument("--dump-utc", default=None, help="when the tables were dumped (default: now)")
a = ap.parse_args()
DUMP = datetime.fromisoformat(a.dump_utc) if a.dump_utc else datetime.now(timezone.utc)
PDT = timezone(timedelta(hours=-7))

# jobs observed QUEUED for their whole life and then stopped by us: no node
# was ever allocated, so completion-creation is queue time, not allocation.
NEVER_ALLOCATED = {"tianrgua-sv4-dgnx"}
# same requester, different project (not smoke-detect): listed, flagged, excluded
# from iteration totals.
EXTERNAL_PREFIXES = ("probe-bindle",)

GPUS_PER_NODE = {"g6e.48xlarge": (8, "L40S"), "g6.48xlarge": (8, "L4"),
                 "g6.2xlarge": (1, "L4"), "p6-b200.48xlarge": (8, "B200"),
                 "p5en.48xlarge": (8, "H200"), "p5.48xlarge": (8, "H100")}

def gpu_info(nodepool: str):
    for k, v in GPUS_PER_NODE.items():
        if k in (nodepool or ""):
            return v
    return (8, "unknown-8?")

IT4_START = datetime(2026, 9, 14, 8, 33, 10, tzinfo=timezone.utc)   # BUDGET_V4.json window_start
IT3_START = datetime(2026, 9, 13, 8, 10, 0, tzinfo=timezone.utc)    # 24h budget closed 09-14 06:50Z w/ 80 min spare
def classify(name: str, start=None):
    n = name.replace("tianrgua-", "")
    # iteration: time-based for 3/4 (exact window starts); name-based for 1 vs 2
    # (no timestamped boundary recorded) -> attribution_method column says which
    if start is not None and start >= IT4_START:
        it, how = 4, "time(window_start)"
    elif start is not None and start >= IT3_START:
        it, how = 3, "time(window_start)"
    elif re.match(r"(vlm|grpo|sft|qwen|diff|synth|fetch|expe|trkd)", n):
        it, how = 2, "name-based(approx)"
    else:
        it, how = 1, "name-based(approx)"
    # track
    if re.match(r"d3v4", n): track = "dino-adapt"
    elif re.match(r"d3", n): track = "dino-v3"
    elif re.match(r"(tra|extxv4|trkd|trackd)", n): track = "temporal"
    elif re.match(r"v4eval|yoloeval|deb|.*val", n): track = "eval"
    elif re.match(r"v4(a|b|c|e|d)", n): track = "yolo-matrix"
    elif re.match(r"(grpo)", n): track = "grpo"
    elif re.match(r"(vlm|sft)", n): track = "vlm-sft"
    elif re.match(r"(diff|synth)", n): track = "diffusion"
    elif re.match(r"(rfd|det)", n): track = "rfdetr/deimv2"
    elif re.match(r"(probe|gputest|sv|s3|smoke|toy|inittst|diag|fetch)", n): track = "infra-probe"
    else: track = "yolo-train"
    # kind
    if re.match(r"(v4eval|yoloeval|deb|fetch|diag|lat)", n) or "eval" in n or n.endswith("val"):
        kind = "evaluation"
    elif re.match(r"(probe|gputest|sv|s3|toy|inittst|smoke)", n): kind = "staging/probe"
    elif "precom" in n or "feats" in n or "extx" in n: kind = "feature-extraction"
    else: kind = "training"
    return it, track, kind, how

def parse_table(path: Path):
    """name -> (start_utc, end_utc) from the text table."""
    out = {}
    tre = re.compile(r"(\d{1,2}/\d{1,2})\s+(\d{1,2}:\d{2}[AP]M) PDT")
    for ln in path.read_text().splitlines()[1:]:
        parts = ln.split()
        if not parts or not parts[0].startswith("tianrgua-"):
            continue
        ts = tre.findall(ln)
        def cv(md, hm):
            m, d = md.split("/")
            return datetime.strptime(f"2026-{int(m):02d}-{int(d):02d} {hm}", "%Y-%m-%d %I:%M%p").replace(tzinfo=PDT).astimezone(timezone.utc)
        start = cv(*ts[0]) if ts else None
        end = cv(*ts[1]) if len(ts) > 1 else None
        out[parts[0]] = (start, end)
    return out

rows = []
for rawf in sorted(glob.glob("reports/v4/ledger/kraken_*_raw.json")):
    proj = Path(rawf).stem.replace("kraken_", "").replace("_raw", "")
    tbl = parse_table(Path(rawf.replace("_raw.json", "_table.txt")))
    for j in json.load(open(rawf))["results"]:
        name = j["jobName"]; md = j["metadata"]
        status = md["status"]
        created = datetime.fromisoformat(md["creationTime"].replace("Z", "+00:00"))
        start, end = tbl.get(name, (None, None))
        if start is None:
            start = created; start_src = "creation(no start in table)"
        else:
            start_src = "table"
        if end is None and md.get("completionTime"):
            end = datetime.fromisoformat(md["completionTime"].replace("Z", "+00:00"))
        running = status in ("RUNNING", "RESTARTING")
        never_started = status in ("PENDING", "QUEUED") or name in NEVER_ALLOCATED
        if never_started:
            end = start   # requested but never allocated -> 0 hours
        elif end is None:
            end = DUMP if running else start
        nodes = (j.get("resourceConfig") or {}).get("nodeCount") or 1
        gpn, gtype = gpu_info(j.get("nodePool") or "")
        node_h = max(0.0, (end - start).total_seconds() / 3600.0)
        it, track, kind, how = classify(name, start)
        external = name.startswith(EXTERNAL_PREFIXES)
        if external:
            it, track, kind, how = 0, "external(other project)", "external", "name-prefix"
        outcome = (f"{status}(open at dump)" if running else
                   f"{status}(never allocated)" if never_started else status)
        if status == "FAILED" and kind == "training":
            kind = "training(failed/retry)"
        rows.append(dict(job_id=j["jobId"], job_name=name, project=proj, iteration=it,
                         track=track, kind=kind, gpu_type=gtype, gpus_per_node=gpn,
                         nodes=nodes, gpus=gpn * nodes,
                         created_utc=created.isoformat(timespec="seconds"),
                         alloc_start_utc=start.isoformat(timespec="seconds"),
                         alloc_end_utc=end.isoformat(timespec="seconds"),
                         queue_wait_min=round((start - created).total_seconds() / 60, 1),
                         node_hours=round(node_h, 3), gpu_hours=round(node_h * gpn * nodes, 2),
                         outcome=outcome, start_source=start_src,
                         utilization="n/a (no pod telemetry)",
                         attribution_method=how, in_scope=(not external),
                         record_type="exact(scheduler)" if not running else "open-at-dump"))
rows.sort(key=lambda r: r["alloc_start_utc"])
cols = list(rows[0].keys())
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

# summary
from collections import defaultdict
agg = defaultdict(lambda: [0, 0.0, 0.0])
for r in rows:
    if not r["in_scope"]:
        continue
    k = (r["iteration"], r["gpu_type"]); agg[k][0] += 1; agg[k][1] += r["node_hours"]; agg[k][2] += r["gpu_hours"]
print(f"{len(rows)} jobs -> {a.out}")
print(f"{'iter':>4} {'gpu':>8} {'jobs':>5} {'node-h':>8} {'gpu-h':>8}")
for k in sorted(agg):
    n, nh, gh = agg[k]; print(f"{k[0]:>4} {k[1]:>8} {n:>5} {nh:>8.1f} {gh:>8.1f}")
v4 = [r for r in rows if r["iteration"] == 4]
print(f"iteration 4 so far: {len(v4)} jobs, {sum(r['node_hours'] for r in v4):.1f} node-h, "
      f"{sum(r['gpu_hours'] for r in v4):.0f} GPU-h (cap 3072); by kind:",
      {k: round(sum(r['node_hours'] for r in v4 if r['kind'] == k), 1) for k in sorted({r['kind'] for r in v4})})
