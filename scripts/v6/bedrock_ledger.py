"""Shared Bedrock spend ledger for V6 (all accounts, all subagents, one file).

Caps (USD): total 2000.00 across generation + visual review + billable retries;
sub-cap 200.00 for the initial generator comparison (tag 'gencmp').
Reservation semantics: reserve estimated cost BEFORE submitting requests;
settle with observed/estimated-actual after; release on abort. Switching
accounts or restarting sessions never resets this file.

Usage (flock-safe):
  .venv/bin/python scripts/v6/bedrock_ledger.py reserve --id gencmp-sd35-01 \
      --tag gencmp --est-usd 1.60 --model stability.sd3-5-large-v1:0 \
      --account 000000000002 --region us-west-2 --n 20 --note "probe batch"
  .venv/bin/python scripts/v6/bedrock_ledger.py settle --id gencmp-sd35-01 \
      --actual-usd 1.44 [--request-ids f.jsonl]
  .venv/bin/python scripts/v6/bedrock_ledger.py release --id gencmp-sd35-01
  .venv/bin/python scripts/v6/bedrock_ledger.py status
Exit 0 ok; 3 = reservation would exceed a cap (do NOT submit).
"""
from __future__ import annotations

import argparse
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "cluster" / "v6"
LEDGER = HERE / "bedrock_ledger.json"
CAPS = {"total_usd": 2000.0, "gencmp_usd": 200.0}


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"caps": CAPS, "entries": {}}


def _spent(led, tag=None):
    """Committed actuals + outstanding reservations (conservative)."""
    s = 0.0
    for e in led["entries"].values():
        if tag and e.get("tag") != tag:
            continue
        if e["state"] == "reserved":
            s += e["est_usd"]
        elif e["state"] == "settled":
            s += e["actual_usd"]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=["reserve", "settle", "release", "status"])
    ap.add_argument("--id")
    ap.add_argument("--tag", default="gen")
    ap.add_argument("--est-usd", type=float)
    ap.add_argument("--actual-usd", type=float)
    ap.add_argument("--model", default="")
    ap.add_argument("--account", default="")
    ap.add_argument("--region", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--note", default="")
    ap.add_argument("--request-ids", default="")
    a = ap.parse_args()

    HERE.mkdir(parents=True, exist_ok=True)
    lockf = open(HERE / ".bedrock_ledger.lock", "w")
    fcntl.flock(lockf, fcntl.LOCK_EX)
    led = _load()

    if a.op == "status":
        tot = _spent(led)
        gc = _spent(led, "gencmp")
        n_res = sum(1 for e in led["entries"].values() if e["state"] == "reserved")
        print(json.dumps({"total_committed_plus_reserved_usd": round(tot, 2),
                          "gencmp_usd": round(gc, 2),
                          "caps": led["caps"], "open_reservations": n_res,
                          "entries": len(led["entries"])}, indent=2))
        return 0

    if not a.id:
        print("need --id"); return 2

    if a.op == "reserve":
        if a.est_usd is None:
            print("need --est-usd"); return 2
        if a.id in led["entries"] and led["entries"][a.id]["state"] != "released":
            print(f"DENIED: id {a.id} already exists "
                  f"({led['entries'][a.id]['state']}) — no double spend")
            return 3
        tot = _spent(led) + a.est_usd
        if tot > led["caps"]["total_usd"]:
            print(f"DENIED: total {tot:.2f} > cap {led['caps']['total_usd']}")
            return 3
        if a.tag == "gencmp":
            gc = _spent(led, "gencmp") + a.est_usd
            if gc > led["caps"]["gencmp_usd"]:
                print(f"DENIED: gencmp {gc:.2f} > cap {led['caps']['gencmp_usd']}")
                return 3
        led["entries"][a.id] = {
            "state": "reserved", "tag": a.tag, "est_usd": a.est_usd,
            "actual_usd": None, "model": a.model, "account": a.account,
            "region": a.region, "n": a.n, "note": a.note,
            "ts_reserved": _now(), "ts_settled": None, "request_ids": a.request_ids}
        print(f"RESERVED {a.id} est ${a.est_usd:.2f} "
              f"(total now ${_spent(led):.2f}/{led['caps']['total_usd']})")
    elif a.op == "settle":
        e = led["entries"].get(a.id)
        if not e or e["state"] != "reserved":
            print(f"cannot settle {a.id}: {e['state'] if e else 'missing'}")
            return 2
        if a.actual_usd is None:
            print("need --actual-usd"); return 2
        e.update(state="settled", actual_usd=a.actual_usd, ts_settled=_now())
        if a.request_ids:
            e["request_ids"] = a.request_ids
        if a.note:
            e["note"] = e.get("note", "") + " | " + a.note
        print(f"SETTLED {a.id} actual ${a.actual_usd:.2f} "
              f"(est was ${e['est_usd']:.2f}; total ${_spent(led):.2f})")
    elif a.op == "release":
        e = led["entries"].get(a.id)
        if not e or e["state"] != "reserved":
            print(f"cannot release {a.id}: {e['state'] if e else 'missing'}")
            return 2
        e.update(state="released", ts_settled=_now())
        print(f"RELEASED {a.id} (total ${_spent(led):.2f})")

    LEDGER.write_text(json.dumps(led, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
