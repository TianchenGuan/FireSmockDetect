#!/usr/bin/env python
"""Capture the FAST (/detect, file-default settings) responses for the 4 fixed
test images used by scripts/v5b/test_modes.py, BEFORE the V5b mode change.

Golden = today's production contract (expB@1280, 0.30/0.15, no shadow),
timing fields (*_ms) stripped. Written once to data/v5b/demo/golden_fast_responses.json
and never regenerated with the new code (that would be circular); re-run only
with --force and say so in the report.

Run: cd /local/home/tianrgua/cvproject && source .venv/bin/activate && python scripts/v5b/make_golden_fast.py
"""
from __future__ import annotations
import argparse, importlib, json, os, sys, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/v5b/demo/golden_fast_responses.json"
SDIS_SEQ_JSON = ROOT / "data/v3/sequences/sequences_val.json"
SDIS_SEQ_ID = "val_force-06_cabanelle-125_20240403T101630"         # cabanelle = sdis VAL site
HEINOLA_CLIP = ROOT / "data/v4/sequences/boreal_frames/heinola_103"  # heinola = aerial VAL

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

def strip_timing(d): return {k: v for k, v in d.items() if not k.endswith("_ms")}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    if OUT.exists() and not a.force:
        raise SystemExit(f"{OUT} exists; refuse to overwrite without --force")
    for k in list(os.environ):
        if k.startswith("SMOKE_"): del os.environ[k]
    import src.serve.config as cfg; importlib.reload(cfg)
    import src.serve.app as appmod; importlib.reload(appmod)
    from fastapi.testclient import TestClient
    items = pick_inputs(); sources = [None, "live_upload", "historical_replay", "historical_replay"]
    recs = []
    with TestClient(appmod.app) as c:
        hz = c.get("/healthz").json(); assert hz["status"] == "ok"
        for (p, ts, dev), src in zip(items, sources):
            raw = p.read_bytes()
            data = {"device_id": dev, "capture_ts": str(ts), "metadata": "t"}
            if src is not None: data["source"] = src
            r = c.post("/detect", files={"image": (p.name, raw, "image/jpeg")}, data=data)
            assert r.status_code == 200, r.text
            recs.append({"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "device_id": dev,
                         "capture_ts": ts, "source": src, "keys": list(r.json().keys()),
                         "response_no_timing": strip_timing(r.json())})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"note": "FAST golden captured with pre-V5b src/serve code (file defaults, expB@1280 0.30/0.15, shadow off); *_ms stripped",
               "healthz_model_version": hz["model_version"], "healthz_keys": list(hz.keys()), "items": recs},
              open(OUT, "w"), indent=1)
    print("wrote", OUT, [r["response_no_timing"]["alert"]["state"] for r in recs])

if __name__ == "__main__":
    main()
