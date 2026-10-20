#!/usr/bin/env python
"""Shadow-mode service test (iteration 5). Runs the FastAPI app IN PROCESS
twice -- without and with a shadow backend -- on the same stored frames and
asserts:

  1. response invariance: /detect JSON (schema, key order and values) is
     identical with and without the shadow, after dropping the *_ms timing
     fields (which legitimately vary run to run);
  2. source tagging: the optional form field `source` defaults to
     live_upload, accepts historical_replay, rejects anything else (422), and
     per-source counts appear in /healthz;
  3. the shadow JSONL log has exactly one record per request with the
     required fields, the right source, the SHA-256 of the uploaded bytes,
     and a primary block that matches the HTTP response;
  4. replay_client always sends source=historical_replay.

Default shadow = the yolo expB weights (fast, CPU) because the property under
test is the code path, not the model; --shadow-dino uses the B4 LoRA
checkpoint instead (slow on CPU, ~2 s/frame). All inputs are stored
historical frames (pyro-sdis VAL site cabanelle; Boreal heinola non-smoke
clip = aerial VAL) -- nothing here is a live upload even where the default
tag says live_upload; the tag only records what the client declared.

Run:  cd /local/home/tianrgua/cvproject && source .venv/bin/activate && \
      python scripts/v5/test_shadow_service.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SDIS_SEQ_JSON = ROOT / "data/v3/sequences/sequences_val.json"
SDIS_SEQ_ID = "val_force-06_cabanelle-125_20240403T101630"       # cabanelle = sdis VAL site
HEINOLA_CLIP = ROOT / "data/v4/sequences/boreal_frames/heinola_103"  # heinola = aerial VAL, no_smoke
TIMING_SUFFIX = "_ms"


def pick_inputs(n_sdis=2, n_boreal=2):
    seqs = json.load(open(SDIS_SEQ_JSON))
    seq = next(s for s in seqs if s["seq_id"] == SDIS_SEQ_ID)
    frames = sorted(seq["frames"], key=lambda f: f["ts"])[:n_sdis]
    items = [(Path(f["image_path"]), float(f["ts"]), "sdis-cabanelle-125-test") for f in frames]
    bf = sorted(p for p in HEINOLA_CLIP.iterdir() if p.suffix == ".jpg")[:n_boreal]
    items += [(p, 1_700_000_000.0 + i, "boreal-heinola_103-test") for i, p in enumerate(bf)]
    for p, _, _ in items:
        assert p.is_file(), p
    return items


def fresh_app(env: dict):
    """Re-instantiate settings + app with exactly these SMOKE_* variables."""
    for k in list(os.environ):
        if k.startswith("SMOKE_"):
            del os.environ[k]
    os.environ.update(env)
    import src.serve.config as cfg
    importlib.reload(cfg)
    import src.serve.app as appmod
    importlib.reload(appmod)
    return appmod


def strip_timing(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.endswith(TIMING_SUFFIX)}


def run_pass(appmod, items, sources):
    from fastapi.testclient import TestClient
    out = []
    with TestClient(appmod.app) as c:
        hz = c.get("/healthz").json()
        assert hz["status"] == "ok", hz
        for (p, ts, dev), src in zip(items, sources):
            raw = p.read_bytes()
            data = {"device_id": dev, "capture_ts": str(ts), "metadata": "t"}
            if src is not None:
                data["source"] = src
            r = c.post("/detect", files={"image": (p.name, raw, "image/jpeg")}, data=data)
            assert r.status_code == 200, r.text
            out.append({"path": str(p), "sha": hashlib.sha256(raw).hexdigest(),
                        "declared": src or "live_upload",
                        "text": r.text, "json": r.json(), "keys": list(r.json().keys())})
        # invalid source -> 422, and it must not be counted
        p, ts, dev = items[0]
        r = c.post("/detect", files={"image": (p.name, p.read_bytes(), "image/jpeg")},
                   data={"device_id": dev, "capture_ts": str(ts), "source": "field_test"})
        assert r.status_code == 422, (r.status_code, r.text)
        # let the shadow worker drain before shutdown
        if hz["shadow"]["enabled"]:
            for _ in range(600):
                s = c.get("/healthz").json()["shadow"]
                if s["queue_depth"] == 0 and s["records_written"] >= len(items):
                    break
                time.sleep(0.1)
        hz_end = c.get("/healthz").json()
    return out, hz_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-dino", action="store_true",
                    help="use dino3 B4 LoRA as the shadow (slow on CPU)")
    ap.add_argument("--log-dir", default=str(ROOT / "data/v5/shadow"))
    args = ap.parse_args()

    items = pick_inputs()
    sources = [None, "live_upload", "historical_replay", "historical_replay"]
    log = Path(args.log_dir) / f"test_shadow_{os.getpid()}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)

    # ---- pass A: shadow OFF (file defaults = deployed expB@1280)
    appA = fresh_app({})
    resA, hzA = run_pass(appA, items, sources)
    assert hzA["shadow"] == {"enabled": False}, hzA["shadow"]
    assert hzA["requests_by_source"] == {"live_upload": 2, "historical_replay": 2}, hzA["requests_by_source"]

    # ---- pass B: shadow ON, own thresholds, own log
    env = {"SMOKE_SHADOW_BACKEND": "yolo",
           "SMOKE_SHADOW_WEIGHTS_PATH": "models/expB_best.pt",
           "SMOKE_SHADOW_MODEL_VERSION": "expB-as-shadow-selftest",
           "SMOKE_SHADOW_CONF_SINGLE": "0.5", "SMOKE_SHADOW_CONF_SEQ": "0.3",
           "SMOKE_SHADOW_LOG_PATH": str(log)}
    if args.shadow_dino:
        env.update({"SMOKE_SHADOW_BACKEND": "dino3",
                    "SMOKE_SHADOW_WEIGHTS_PATH": "models/v3/runs/v4B4_lora_L/seg05/best.pt",
                    "SMOKE_SHADOW_MODEL_VERSION": "v4B4_lora_L@seg05-step15026",
                    "SMOKE_SHADOW_CONF_SINGLE": "0.93", "SMOKE_SHADOW_CONF_SEQ": "0.82"})
    appB = fresh_app(env)
    resB, hzB = run_pass(appB, items, sources)

    # ---- 1. response invariance
    n_bytes_identical = 0
    for a, b in zip(resA, resB):
        assert a["keys"] == b["keys"], ("response key set/order differs", a["keys"], b["keys"])
        assert strip_timing(a["json"]) == strip_timing(b["json"]), (
            "response values differ with shadow on:", a["path"], a["json"], b["json"])
        if a["text"] == b["text"]:
            n_bytes_identical += 1
    # the response must not leak the shadow or the source tag
    for b in resB:
        assert "shadow" not in b["text"].lower(), b["text"]
        assert "source" not in b["json"], b["keys"]

    # ---- 2. source counts
    assert hzB["sources_allowed"] == ["live_upload", "historical_replay"]
    assert hzB["requests_by_source"] == {"live_upload": 2, "historical_replay": 2}, hzB["requests_by_source"]
    sh = hzB["shadow"]
    assert sh["enabled"] and sh["loaded"] and sh["errors"] == 0 and sh["dropped"] == 0, sh
    assert sh["conf_single"] == 0.5 or args.shadow_dino, sh
    assert sh["records_by_source"] == {"live_upload": 2, "historical_replay": 2}, sh["records_by_source"]

    # ---- 3. JSONL records
    recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert len(recs) == len(items), (len(recs), len(items))
    required = {"ts", "device_id", "capture_ts", "source", "primary", "shadow", "image_sha256"}
    for rec, b in zip(recs, resB):   # FIFO worker -> same order as requests
        assert required <= set(rec), set(rec)
        assert rec["source"] == b["declared"], (rec["source"], b["declared"])
        assert rec["image_sha256"] == b["sha"]
        assert rec["device_id"] == b["json"]["device_id"]
        assert rec["capture_ts"] == b["json"]["capture_ts"]
        assert rec["primary"]["max_smoke"] == b["json"]["max_smoke_score"]
        assert rec["primary"]["alert"] == b["json"]["alert"]["state"]
        assert rec["primary"]["model_version"] == b["json"]["model_version"]
        for k in ("max_smoke", "alert", "n_dets", "latency_ms", "model_version", "conf_single", "conf_seq"):
            assert k in rec["shadow"], (k, rec["shadow"])
        assert rec["shadow"]["alert"] in ("none", "watch", "provisional", "confirmed")
    assert {r["source"] for r in recs} == {"live_upload", "historical_replay"}

    # ---- 4. replay client contract
    from src.serve import replay_client as rc
    form = rc.build_form("dev", 1.0, "tag")
    assert rc.SOURCE == "historical_replay" and form["source"] == "historical_replay"
    assert json.loads(form["metadata"])["source"] == "historical_replay"

    summary = {
        "n_requests": len(items),
        "responses_identical_after_timing_strip": len(items),
        "responses_byte_identical_incl_timing": n_bytes_identical,
        "requests_by_source": hzB["requests_by_source"],
        "shadow_records_by_source": sh["records_by_source"],
        "shadow_backend": sh["backend"], "shadow_model_version": sh["model_version"],
        "shadow_latency_ms_mean": sh["latency_ms_mean"],
        "primary_latency_ms": [b["json"]["latency_ms"] for b in resB],
        "shadow_alerts_by_state": sh["alerts_by_state"],
        "log": str(log),
    }
    print("PASS", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
