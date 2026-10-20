#!/usr/bin/env python
"""FAST / ACCURATE mode tests (iteration 5b). Runs the FastAPI app IN PROCESS
(fastapi TestClient) under several SMOKE_* configurations and asserts:

  A. file defaults: FAST /detect is identical to the pre-V5b golden responses
     (data/v5b/demo/golden_fast_responses.json, captured with the pre-change
     code; *_ms timing stripped, key ORDER compared) for 4 stored images;
     mode=accurate -> 422 (not configured); bad mode -> 422; /healthz lists modes.
  B. ACCURATE configured (default mode still fast): default requests stay
     golden-identical with the ensemble loaded; mode=accurate returns evidence
     for EVERY configured model (raw boxes, latency, version, own alert) plus
     fused boxes + one final alert; the component that shares weights with the
     FAST model reproduces the FAST boxes exactly (evidence is real, not a
     relabel); shadow recorder still writes one record per request in both
     modes; invalid inputs: unreadable -> 422, oversized -> 413, bad source ->
     422 (checked before mode), bad mode -> 422.
  C. SMOKE_MODE=accurate default: bare requests are ACCURATE; per-request
     mode=fast returns the unchanged FAST contract (golden-identical).
  D. fusion selection: recipe file -> wbf; explicit SMOKE_ACCURATE_FUSION wins
     over the recipe; "learned" without scripts/v5b/fusion.py falls back to "or"
     and says so.

Default components are two CPU-cheap YOLO checkpoints (the property under test
is the mode machinery, not the model). --dino swaps in the real ACCURATE demo
set (yolo v4a_s2 + dino3 B4 LoRA at 0.93/0.82; ~2 s/frame on CPU). All inputs
are stored historical frames (pyro-sdis VAL site cabanelle, Boreal heinola
aerial-VAL clip); nothing here is a live upload.

Run: cd /local/home/tianrgua/cvproject && source .venv/bin/activate && \
     python scripts/v5b/test_modes.py [--dino] [--skip-shadow-test]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GOLDEN = ROOT / "data/v5b/demo/golden_fast_responses.json"
STATES = ("none", "watch", "provisional", "confirmed")
SEV = {s: i for i, s in enumerate(STATES)}
YOLO_SET = ("expB_alt=yolo:models/expB_best.pt,version=expB-as-component;"
            "v4a_s2=yolo:models/trained/expV4A_ctrl_s2/seg05/best.pt,version=expV4A_ctrl_s2@seg05")
DINO_SET = ("v4a_s2=yolo:models/trained/expV4A_ctrl_s2/seg05/best.pt,version=expV4A_ctrl_s2@seg05;"
            "dino3_b4=dino3:models/v3/runs/v4B4_lora_L/seg05/best.pt,version=v4B4_lora_L@seg05-step15026,"
            "conf_single=0.93,conf_seq=0.82")


def strip_timing(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.endswith("_ms")}


def fresh_app(env: dict):
    for k in list(os.environ):
        if k.startswith("SMOKE_"):
            del os.environ[k]
    os.environ.update(env)
    import src.serve.config as cfg
    importlib.reload(cfg)
    import src.serve.app as appmod
    importlib.reload(appmod)
    return appmod


def post(c, item: dict, raw: bytes, **extra):
    data = {"device_id": extra.pop("device_id", item["device_id"]),
            "capture_ts": str(extra.pop("capture_ts", item["capture_ts"])), "metadata": "t"}
    if item.get("source") is not None:
        data["source"] = item["source"]
    data.update({k: v for k, v in extra.items() if v is not None})
    return c.post("/detect", files={"image": (Path(item["path"]).name, raw, "image/jpeg")}, data=data)


def assert_golden(resp, item, where):
    assert resp.status_code == 200, (where, resp.status_code, resp.text[:200])
    j = resp.json()
    assert list(j.keys()) == item["keys"], (where, "key set/order", list(j.keys()), item["keys"])
    assert strip_timing(j) == item["response_no_timing"], (where, "values differ", strip_timing(j), item["response_no_timing"])
    assert "mode" not in j and "models" not in j and "fusion" not in j, (where, list(j.keys()))


def check_accurate(j: dict, names: list[str], fusion: str, where: str):
    assert j["mode"] == "accurate", where
    assert set(j["models"]) == set(names), (where, list(j["models"]), names)
    assert j["model_version"].startswith(f"accurate:{fusion}["), (where, j["model_version"])
    assert j["fusion"]["method"] == fusion, (where, j["fusion"])
    assert "normalization" in j["fusion"] and "scale" in j["fusion"], where
    if j["fusion"].get("recipe_applied"):
        assert j["fusion"]["scale"] == "calibrated", (where, j["fusion"])
    else:
        assert j["fusion"]["scale"] == "alert_equivalent" and "alert_equivalent" in j["fusion"]["normalization"], (where, j["fusion"])
    for n in names:
        m = j["models"][n]
        assert "error" not in m, (where, n, m.get("error"))
        assert isinstance(m["detections"], list) and m["latency_ms"] is not None and m["latency_ms"] > 0, (where, n)
        assert m["model_version"] and m["backend"] and m["thresholds"], (where, n)
        assert m["alert"]["state"] in STATES, (where, n, m["alert"])
        for d in m["detections"]:
            assert set(d) >= {"bbox_xyxy", "score", "cls_name", "score_alert_equivalent"}, (where, n, d)
        assert m["max_smoke_score"] == (max([d["score"] for d in m["detections"] if d["cls_name"] == "smoke"], default=0.0)), (where, n)
    for d in j["detections"]:
        assert set(d) >= {"bbox_xyxy", "score", "cls_name", "source", "models", "raw_scores"}, (where, d)
        assert d["source"] == f"fused_{fusion}" and set(d["models"]) <= set(names) and d["models"], (where, d)
    if not j["fusion"].get("recipe_applied"):
        assert j["max_smoke_score"] == round(max([d["score"] for d in j["detections"] if d["cls_name"] == "smoke"], default=0.0), 4), where
    else:
        assert j["max_smoke_score"] >= round(max([d["score"] for d in j["detections"] if d["cls_name"] == "smoke"], default=0.0), 4) - 1e-4, where
    assert j["n_smoke"] == sum(d["cls_name"] == "smoke" for d in j["detections"]), where
    a = j["alert"]
    assert a["state"] in STATES and set(a["by_model"]) == set(names), (where, a)
    if fusion == "or":
        worst = max(a["by_model"].values(), key=lambda s: SEV[s])
        assert a["state"] == worst, (where, a)
        assert a["decided_by"].startswith("most severe per-model alert"), (where, a["decided_by"])
    else:
        assert "fused AlertEngine" in a["decided_by"], (where, a["decided_by"])
    assert j["latency_ms"] >= max(j["models"][n]["latency_ms"] for n in names) * 0.5, (where, "wall vs components")
    assert j["model_errors"] == {}, (where, j["model_errors"])


def same_boxes(a: list[dict], b: list[dict], tol=1e-3) -> bool:
    ka = sorted((tuple(d["bbox_xyxy"]), d["score"]) for d in a)
    kb = sorted((tuple(d["bbox_xyxy"]), d["score"]) for d in b)
    if len(ka) != len(kb):
        return False
    return all(abs(x[1] - y[1]) <= tol and all(abs(p - q) <= 0.2 for p, q in zip(x[0], y[0])) for x, y in zip(ka, kb))


def invalid_inputs(c, item, raw, max_mb: int, accurate_ok: bool, where: str):
    before = c.get("/healthz").json()
    r = c.post("/detect", files={"image": ("x.txt", b"not an image at all", "text/plain")},
               data={"device_id": "bad", "mode": "fast"})
    assert r.status_code == 422 and "decodable" in r.text, (where, r.status_code, r.text)
    if accurate_ok:
        r = c.post("/detect", files={"image": ("x.txt", b"not an image at all", "text/plain")},
                   data={"device_id": "bad", "mode": "accurate"})
        assert r.status_code == 422 and "decodable" in r.text, (where, r.status_code, r.text)
    big = b"\0" * (max_mb * 1024 * 1024 + 1024)
    r = c.post("/detect", files={"image": ("big.jpg", big, "image/jpeg")}, data={"device_id": "bad"})
    assert r.status_code == 413, (where, r.status_code, r.text)
    r = post(c, item, raw, device_id="bad", source="field_test", mode="accurate")
    assert r.status_code == 422 and "source" in r.text, (where, r.status_code, r.text)
    r = post(c, item, raw, device_id="bad", mode="turbo")
    assert r.status_code == 422 and "mode" in r.text, (where, r.status_code, r.text)
    r = post(c, item, raw, device_id="bad", mode="")          # empty -> server default, must work
    assert r.status_code == 200, (where, r.status_code, r.text)
    after = c.get("/healthz").json()
    assert after["requests_by_source"][item.get("source") or "live_upload"] == before["requests_by_source"][item.get("source") or "live_upload"] + 1, where
    assert sum(after["modes"]["requests_by_mode"].values()) == sum(before["modes"]["requests_by_mode"].values()) + 1, where


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino", action="store_true", help="ACCURATE = yolo v4a_s2 + dino3 B4 (slow on CPU)")
    ap.add_argument("--skip-shadow-test", action="store_true")
    ap.add_argument("--recipe-full", action="store_true",
                    help="also load ALL recipe components (yolo_A + dino B4/B6/B1; slow, CPU) and check calibrated WBF")
    args = ap.parse_args()
    g = json.load(open(GOLDEN))
    items = g["items"]
    raws = [Path(it["path"]).read_bytes() for it in items]
    for it, raw in zip(items, raws):
        import hashlib
        assert hashlib.sha256(raw).hexdigest() == it["sha256"], it["path"]
    comp = DINO_SET if args.dino else YOLO_SET
    names = [e.split("=", 1)[0] for e in comp.split(";")]
    tmp = Path(tempfile.mkdtemp(prefix="v5b_modes_"))
    summary = {"golden_items": len(items), "components": names, "passes": []}

    # ---------------------------------------------------------------- A: file defaults
    t0 = time.time()
    appA = fresh_app({})
    from fastapi.testclient import TestClient
    with TestClient(appA.app) as c:
        hz = c.get("/healthz").json()
        assert hz["status"] == "ok" and hz["model_version"] == g["healthz_model_version"], hz["model_version"]
        assert set(g["healthz_keys"]) <= set(hz), (set(g["healthz_keys"]) - set(hz))
        m = hz["modes"]
        assert m["default"] == "fast" and m["allowed"] == ["fast", "accurate"], m
        assert m["fast"]["model_version"] == hz["model_version"] and m["fast"]["conf_single"] == 0.30 and m["fast"]["conf_seq"] == 0.15, m["fast"]
        assert m["accurate"] == {"enabled": False, "models": []}, m["accurate"]
        for it, raw in zip(items, raws):
            assert_golden(post(c, it, raw), it, "A/default")
        r = post(c, items[0], raws[0], device_id="A-explicit", mode="fast")
        assert r.status_code == 200 and list(r.json().keys()) == items[0]["keys"], r.text[:200]
        r = post(c, items[0], raws[0], device_id="A-acc", mode="accurate")
        assert r.status_code == 422 and "not configured" in r.text, (r.status_code, r.text)
        r = post(c, items[0], raws[0], device_id="A-bad", mode="turbo")
        assert r.status_code == 422, r.text
        hz = c.get("/healthz").json()
        assert hz["modes"]["requests_by_mode"] == {"fast": 5, "accurate": 0}, hz["modes"]["requests_by_mode"]
    summary["passes"].append({"A_file_defaults": "PASS", "s": round(time.time() - t0, 1)})

    # ---------------------------------------------------------------- B: accurate configured, default fast
    t0 = time.time()
    log = tmp / "shadow_B.jsonl"
    appB = fresh_app({"SMOKE_ACCURATE_MODELS": comp, "SMOKE_MAX_UPLOAD_MB": "3",
                      "SMOKE_ACCURATE_RECIPE_PATH": str(tmp / "no_recipe.json"),
                      "SMOKE_SHADOW_BACKEND": "yolo", "SMOKE_SHADOW_WEIGHTS_PATH": "models/expB_best.pt",
                      "SMOKE_SHADOW_MODEL_VERSION": "expB-shadow-selftest", "SMOKE_SHADOW_LOG_PATH": str(log)})
    with TestClient(appB.app) as c:
        hz = c.get("/healthz").json()
        acc = hz["modes"]["accurate"]
        assert acc["enabled"] and acc["loaded"] and [x["name"] for x in acc["models"]] == names, acc
        assert all(x["loaded"] and x["model_version"] for x in acc["models"]), acc["models"]
        assert acc["fusion"]["method"] == "or" and acc["fusion"]["recipe_path"] is None and acc["fusion"]["scale"] == "alert_equivalent", acc["fusion"]
        if args.dino:
            d = next(x for x in acc["models"] if x["name"] == "dino3_b4")
            assert d["conf_single"] == 0.93 and d["conf_seq"] == 0.82 and d["backend_info"]["lora_r"] == 16, d
        for it, raw in zip(items, raws):                       # FAST unchanged with the ensemble loaded
            assert_golden(post(c, it, raw), it, "B/default-fast")
        acc_lat, acc_states = [], []
        for i, (it, raw) in enumerate(zip(items, raws)):
            r = post(c, it, raw, device_id="acc-" + it["device_id"], mode="accurate")
            assert r.status_code == 200, (r.status_code, r.text[:300])
            j = r.json()
            check_accurate(j, names, "or", f"B/accurate[{i}]")
            acc_lat.append(j["latency_ms"]); acc_states.append(j["alert"]["state"])
            if "expB_alt" in names:                            # same weights as FAST -> identical raw evidence
                assert same_boxes(j["models"]["expB_alt"]["detections"], it["response_no_timing"]["detections"]), (
                    "component evidence differs from the FAST model with the same weights",
                    j["models"]["expB_alt"]["detections"], it["response_no_timing"]["detections"])
                for d in j["models"]["expB_alt"]["detections"]:
                    assert abs(d["score_alert_equivalent"] - d["score"]) < 1e-4, d   # identity mapping at 0.30/0.15 (score is rounded to 4 dp)
            if "dino3_b4" in names:
                for d in j["models"]["dino3_b4"]["detections"]:   # threshold crossings map onto the reference pair
                    if abs(d["score"] - 0.93) > 1e-3:
                        assert (d["score"] >= 0.93) == (d["score_alert_equivalent"] >= 0.30), d
                    if abs(d["score"] - 0.82) > 1e-3:
                        assert (d["score"] >= 0.82) == (d["score_alert_equivalent"] >= 0.15), d
        invalid_inputs(c, items[0], raws[0], 3, accurate_ok=True, where="B/invalid")
        hz = c.get("/healthz").json()
        rb = hz["modes"]["requests_by_mode"]
        assert rb["fast"] == 5 and rb["accurate"] == 4, rb
        for _ in range(600):
            sh = c.get("/healthz").json()["shadow"]
            if sh["queue_depth"] == 0 and sh["records_written"] >= 9:
                break
            time.sleep(0.1)
        sh = c.get("/healthz").json()["shadow"]
        assert sh["records_written"] == 9 and sh["errors"] == 0 and sh["dropped"] == 0, sh
    recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert len(recs) == 9
    acc_recs = [r for r in recs if r["primary"].get("mode") == "accurate"]
    assert len(acc_recs) == 4 and all(r["primary"]["model_version"].startswith("accurate:or[") for r in acc_recs), [r["primary"] for r in recs][:3]
    assert all("mode" not in r["primary"] for r in recs if r["device_id"].startswith(("sdis", "boreal"))), "FAST shadow record changed"
    summary["passes"].append({"B_accurate_configured": "PASS", "s": round(time.time() - t0, 1),
                              "accurate_latency_ms": acc_lat, "accurate_alert_states": acc_states})

    # ---------------------------------------------------------------- C: default mode accurate, per-request fast
    t0 = time.time()
    appC = fresh_app({"SMOKE_ACCURATE_MODELS": comp, "SMOKE_MODE": "accurate",
                      "SMOKE_ACCURATE_RECIPE_PATH": str(tmp / "no_recipe.json")})
    with TestClient(appC.app) as c:
        hz = c.get("/healthz").json()
        assert hz["modes"]["default"] == "accurate", hz["modes"]["default"]
        for it, raw in zip(items, raws):                       # explicit fast override -> golden
            assert_golden(post(c, it, raw, mode="fast"), it, "C/override-fast")
        r = post(c, items[1], raws[1], device_id="C-bare")     # bare -> accurate
        assert r.status_code == 200, r.text[:200]
        check_accurate(r.json(), names, "or", "C/bare-accurate")
        rb = c.get("/healthz").json()["modes"]["requests_by_mode"]
        assert rb == {"fast": 4, "accurate": 1}, rb
    summary["passes"].append({"C_default_accurate": "PASS", "s": round(time.time() - t0, 1)})

    # ---------------------------------------------------------------- D: fusion selection (cheap yolo set)
    t0 = time.time()
    recipe = tmp / "accurate_recipe.json"
    recipe.write_text(json.dumps({"method": "wbf", "iou": 0.55, "weights": {n: 1.0 for n in names}}))
    appD = fresh_app({"SMOKE_ACCURATE_MODELS": comp, "SMOKE_ACCURATE_RECIPE_PATH": str(recipe),
                      "SMOKE_ACCURATE_FUSION": "wbf"})
    with TestClient(appD.app) as c:
        f = c.get("/healthz").json()["modes"]["accurate"]["fusion"]
        assert f["method"] == "wbf" and f["recipe_path"] == str(recipe) and f["iou"] == 0.55, f
        assert f["scale"] == "alert_equivalent" and not f["recipe_applied"] and "builtin wbf" in (f["fallback_reason"] or ""), f
        r = post(c, items[1], raws[1], device_id="D-wbf", mode="accurate")
        assert r.status_code == 200, r.text[:200]
        check_accurate(r.json(), names, "wbf", "D/wbf")
    appD2 = fresh_app({"SMOKE_ACCURATE_MODELS": comp, "SMOKE_ACCURATE_RECIPE_PATH": str(recipe),
                       "SMOKE_ACCURATE_FUSION": "or"})
    with TestClient(appD2.app) as c:
        f = c.get("/healthz").json()["modes"]["accurate"]["fusion"]
        assert f["method"] == "or" and f["recipe_path"] == str(recipe), f
    appD3 = fresh_app({"SMOKE_ACCURATE_MODELS": comp, "SMOKE_ACCURATE_FUSION": "learned",
                       "SMOKE_ACCURATE_RECIPE_PATH": str(tmp / "missing.json")})
    with TestClient(appD3.app) as c:
        f = c.get("/healthz").json()["modes"]["accurate"]["fusion"]
        assert f["method_requested"] == "learned", f
        assert f["method"] == "or" and "'learned' unavailable" in (f["fallback_reason"] or ""), f
        r = post(c, items[1], raws[1], device_id="D-learned", mode="accurate")
        check_accurate(r.json(), names, "or", "D/learned-fallback")
        summary["fusion_learned"] = f["fallback_reason"]
    summary["passes"].append({"D_fusion_selection": "PASS", "s": round(time.time() - t0, 1)})

    # ---------------------------------------------------------------- E: the real recipe (models/v5b/fusion/accurate_recipe.json)
    real = ROOT / "models/v5b/fusion/accurate_recipe.json"
    if real.is_file() and (ROOT / "scripts/v5b/fusion.py").is_file():
        t0 = time.time()
        rj = json.load(open(real))
        rec_models = list(rj["models"].keys())
        # E1: configured names != recipe model set -> recipe loaded but NOT applied; default stays "or"
        appE1 = fresh_app({"SMOKE_ACCURATE_MODELS": comp})
        with TestClient(appE1.app) as c:
            f = c.get("/healthz").json()["modes"]["accurate"]["fusion"]
            assert f["recipe_path"] == str(real) and f["recipe"]["models"] == rec_models, f
            assert f["method"] == "or" and not f["recipe_applied"] and "not applied" in (f["fallback_reason"] or ""), f
            r = post(c, items[1], raws[1], device_id="E1", mode="accurate")
            check_accurate(r.json(), names, "or", "E1/recipe-not-applicable")
        summary["passes"].append({"E1_recipe_not_applicable": "PASS", "s": round(time.time() - t0, 1)})
        if args.recipe_full:
            # E2: components named and weighted exactly as the recipe -> calibrated WBF via scripts/v5b/fusion.py
            t0 = time.time()
            spec = ";".join(f"{n}={'yolo' if n.startswith('yolo') else 'dino3'}:{rj['models'][n]['checkpoint']},version={n}"
                            + ("" if n.startswith("yolo") else ",conf_single=0.93,conf_seq=0.82") for n in rec_models)
            appE2 = fresh_app({"SMOKE_ACCURATE_MODELS": spec})
            with TestClient(appE2.app) as c:
                f = c.get("/healthz").json()["modes"]["accurate"]["fusion"]
                assert f["method"] == "wbf" and f["recipe_applied"] and f["scale"] == "calibrated", f
                assert f["final_alert_thresholds"]["source"] == "recipe thresholds_oof", f
                lat = []
                for i, (it, raw) in enumerate(zip(items, raws)):
                    r = post(c, it, raw, device_id="E2-" + it["device_id"], mode="accurate")
                    assert r.status_code == 200, r.text[:300]
                    j = r.json()
                    check_accurate(j, rec_models, "wbf", f"E2/recipe[{i}]")
                    assert j["fusion"]["impl"].endswith("scripts/v5b/fusion.py") and j["fusion"]["recipe_applied"], j["fusion"]
                    for n in rec_models:
                        m = j["models"][n]
                        assert "max_calibrated" in m and all("score_calibrated" in d for d in m["detections"]), (n, m.keys())
                    lat.append((j["latency_ms"], {n: j["models"][n]["latency_ms"] for n in rec_models}))
            summary["passes"].append({"E2_recipe_applied": "PASS", "s": round(time.time() - t0, 1), "latency": lat})

    # ---------------------------------------------------------------- existing shadow test unchanged
    if not args.skip_shadow_test:
        t0 = time.time()
        for k in list(os.environ):
            if k.startswith("SMOKE_"):
                del os.environ[k]
        out = subprocess.run([sys.executable, str(ROOT / "scripts/v5/test_shadow_service.py"), "--log-dir", str(tmp)],
                             cwd=ROOT, capture_output=True, text=True)
        assert out.returncode == 0 and "PASS" in out.stdout, out.stdout[-800:] + out.stderr[-800:]
        summary["passes"].append({"v5_test_shadow_service": "PASS", "s": round(time.time() - t0, 1)})

    summary["tmp"] = str(tmp)
    print("PASS", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
