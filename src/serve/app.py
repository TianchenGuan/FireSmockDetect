"""FastAPI smoke-detection service.

Run:  cd /local/home/tianrgua/cvproject && source .venv/bin/activate
      uvicorn src.serve.app:app --host 127.0.0.1 --port 8000
Private by default (localhost bind). POST /detect, GET /healthz, GET /.

Iteration 5 additions (all default-off / additive):
  * /detect form field `source` in {live_upload, historical_replay}
    (default live_upload). Replays MUST send historical_replay
    (src/serve/replay_client.py does); counts per source are in /healthz.
  * optional SHADOW backend (SMOKE_SHADOW_* settings, src/serve/shadow.py):
    runs a second model on the same bytes in a FIFO worker thread and appends
    a JSONL record per request. The HTTP response is produced by the primary
    model only and is byte-for-byte the same with or without a shadow
    (scripts/v5/test_shadow_service.py).

Iteration 5b additions (all default-off / additive; scripts/v5b/test_modes.py):
  * modes FAST | ACCURATE (SMOKE_MODE, default "fast" = exactly the response
    above). ACCURATE (SMOKE_ACCURATE_MODELS) runs N models in parallel threads
    and returns per-model evidence + fused boxes + one final alert
    (src/serve/modes.py). /detect form field `mode` overrides per request;
    /healthz["modes"] lists both modes' model versions.
  * demo UI at GET /demo (src/serve/demo.py, static/demo.html): upload or
    historical replay through /detect with source=historical_replay.
"""
from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import imagehash

from .config import settings
from .detector import SmokeDetector  # noqa: F401 (Detection re-export)
from .backends import load_backend
from .alerting import AlertEngine
from .shadow import DEFAULT_SOURCE, SOURCES, ShadowJob, ShadowRecorder, alert_summary
from .modes import MODES, AccurateEnsemble
from .demo import DemoRegistry, router as demo_router, install as demo_install

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="smoke-detect", docs_url=None, redoc_url=None)
_started = time.time()
detector: SmokeDetector | None = None
alerts = AlertEngine(settings)
shadow = ShadowRecorder(settings, ROOT)
_source_counts = {s: 0 for s in SOURCES}
_source_lock = threading.Lock()
# iteration 5b: FAST / ACCURATE modes + demo registry (default-off; see module doc)
if settings.mode not in MODES:
    raise ValueError(f"SMOKE_MODE must be one of {MODES}; got {settings.mode!r}")
ensemble = AccurateEnsemble(settings, ROOT)   # parses SMOKE_ACCURATE_MODELS; loud on bad specs
if settings.mode == "accurate" and not ensemble.enabled:
    raise ValueError("SMOKE_MODE=accurate requires SMOKE_ACCURATE_MODELS")
_mode_counts = {m: 0 for m in MODES}
demo = DemoRegistry(settings, ROOT)
demo_install(demo)
app.include_router(demo_router)


@app.on_event("startup")
def _load():
    global detector
    wp = Path(settings.weights_path)
    if not wp.is_absolute():
        wp = ROOT / wp
    detector = load_backend(settings.backend, weights_path=str(wp),
                            device=settings.device, imgsz=settings.imgsz,
                            tile_size=settings.tile_size,
                            tile_overlap=settings.tile_overlap,
                            merge_iou=settings.merge_iou,
                            conf_floor=settings.conf_floor,
                            # dino3-only (ignored by yolo; other backends take **kw)
                            backbone_dir=settings.dino_backbone_dir or None,
                            lora_r=settings.lora_r, lora_alpha=settings.lora_alpha,
                            lora_merge=settings.lora_merge,
                            width=settings.det_width or None,
                            height=settings.det_height or None)
    # ACCURATE components load after the primary (FAST readiness first); a bad
    # component spec raises loudly rather than serving a different ensemble.
    ensemble.start()
    try:
        demo.build()
    except Exception as e:   # the demo registry must never block the service
        demo.warnings.append(f"demo registry failed: {type(e).__name__}: {e}"[:200])
    # shadow model loads AFTER the primary so a shadow failure never delays
    # primary readiness semantics; it raises loudly (misconfiguration) rather
    # than silently serving without the requested shadow.
    shadow.start()


@app.on_event("shutdown")
def _unload():
    shadow.stop()
    ensemble.stop()


def _modes_status() -> dict:
    with _source_lock:
        by_mode = dict(_mode_counts)
    wp = Path(settings.weights_path)
    return {"default": settings.mode, "allowed": list(MODES), "requests_by_mode": by_mode,
            "fast": {"backend": settings.backend, "model_version": settings.model_version,
                     "weights_path": str(wp if wp.is_absolute() else ROOT / wp),
                     "conf_single": settings.single_frame_alert_conf, "conf_seq": settings.seq_frame_conf,
                     "conf_floor": settings.conf_floor, "imgsz": settings.imgsz, "device": settings.device},
            "accurate": ensemble.status()}


@app.get("/healthz")
def healthz():
    with _source_lock:
        by_source = dict(_source_counts)
    return {"status": "ok" if detector else "loading",
            "backend": settings.backend,
            "model_version": settings.model_version,
            "device": settings.device, "imgsz": settings.imgsz,
            "tile_size": settings.tile_size,
            "uptime_s": round(time.time() - _started, 1),
            # backends that expose info() (dino3: backbone, LoRA rank, ckpt step)
            **({"backend_info": detector.info()} if hasattr(detector, "info") else {}),
            # iteration 5: provenance tagging + shadow/bypass recorder
            "sources_allowed": list(SOURCES),
            "requests_by_source": by_source,
            "shadow": shadow.status(),
            # iteration 5b: FAST / ACCURATE model versions + demo registry
            "modes": _modes_status(),
            "demo": demo.status()}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/detect")
async def detect(image: UploadFile = File(...),
                 device_id: str = Form("anonymous"),
                 capture_ts: float = Form(None),
                 metadata: str = Form(None),
                 source: str = Form(DEFAULT_SOURCE),
                 mode: str = Form(None)):
    if detector is None:
        raise HTTPException(503, "model not loaded")
    if source not in SOURCES:
        raise HTTPException(422, f"source must be one of {list(SOURCES)}; got {source!r}")
    mode = (mode or settings.mode).lower()
    if mode not in MODES:
        raise HTTPException(422, f"mode must be one of {list(MODES)}; got {mode!r}")
    if mode == "accurate" and not ensemble.loaded:
        raise HTTPException(422, "accurate mode is not configured on this server "
                                 "(SMOKE_ACCURATE_MODELS is empty); use mode=fast")
    raw = await image.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"upload exceeds {settings.max_upload_mb} MB")
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(422, "not a decodable image")

    img = np.asarray(pil)[:, :, ::-1]  # RGB -> BGR for ultralytics numpy path
    if mode == "accurate":
        return _detect_accurate(raw, pil, img, device_id, capture_ts, metadata, source)

    # ---- FAST: the single primary backend; this path and its response are
    # unchanged from iteration 5 (byte-identical; scripts/v5b/test_modes.py).
    dets, timing = detector.detect(img)

    smoke = [d for d in dets if d.cls_name.lower() == "smoke" or len(detector.class_names) == 1]
    max_smoke = max((d.score for d in smoke), default=0.0)
    ts = capture_ts if capture_ts is not None else time.time()
    phash = imagehash.phash(pil.resize((256, 256)))
    alert = alerts.update(device_id, ts, max_smoke, phash=phash)

    with _source_lock:
        _source_counts[source] += 1
        _mode_counts["fast"] += 1

    # bypass: hand the SAME bytes + primary verdict to the shadow worker. Fully
    # non-blocking; nothing below the response depends on it.
    if shadow.enabled:
        shadow.submit(ShadowJob(
            raw=raw, phash=phash, device_id=device_id, capture_ts=ts, source=source,
            image_size=[pil.width, pil.height], client_metadata=metadata,
            primary={"model_version": settings.model_version,
                     "max_smoke": round(max_smoke, 4),
                     "alert": alert["state"],
                     "alert_info": alert_summary(alert),
                     "n_dets": len(dets),
                     "latency_ms": timing.get("latency_ms")},
            request_ts=time.time()))

    return JSONResponse({
        "detections": [d.as_dict() for d in dets],
        "n_smoke": len(smoke),
        "max_smoke_score": round(max_smoke, 4),
        "alert": alert,
        "model_version": settings.model_version,
        "image_size": [pil.width, pil.height],
        "device_id": device_id,
        "capture_ts": ts,
        "client_metadata_echo": metadata,
        "score_semantics": "raw detector confidence; NOT a calibrated probability",
        **timing,
    })


def _detect_accurate(raw: bytes, pil, img, device_id: str, capture_ts, metadata, source: str):
    """ACCURATE: all configured models in parallel on the same full frame; per-model
    evidence + fused boxes + one final alert. Additive keys only (mode, fusion,
    models, model_errors); the FAST keys keep their meaning on the fused result."""
    ts = capture_ts if capture_ts is not None else time.time()
    phash = imagehash.phash(pil.resize((256, 256)))
    res = ensemble.run(img, device_id=device_id, ts=ts, phash=phash)
    smoke = [d for d in res["detections"] if d["cls_name"].lower() == "smoke"]
    with _source_lock:
        _source_counts[source] += 1
        _mode_counts["accurate"] += 1
    if shadow.enabled:
        shadow.submit(ShadowJob(
            raw=raw, phash=phash, device_id=device_id, capture_ts=ts, source=source,
            image_size=[pil.width, pil.height], client_metadata=metadata,
            primary={"model_version": ensemble.label, "mode": "accurate",
                     "max_smoke": res["max_smoke_score"], "alert": res["alert"]["state"],
                     "alert_info": alert_summary(res["alert"]), "n_dets": len(res["detections"]),
                     "latency_ms": res["latency_ms"]},
            request_ts=time.time()))
    return JSONResponse({
        "detections": res["detections"],
        "n_smoke": len(smoke),
        "max_smoke_score": res["max_smoke_score"],
        "alert": res["alert"],
        "model_version": ensemble.label,
        "image_size": [pil.width, pil.height],
        "device_id": device_id,
        "capture_ts": ts,
        "client_metadata_echo": metadata,
        "score_semantics": ("fused alert-equivalent score (see fusion.normalization); per-model raw "
                            "confidences are in models[*].detections; NOT a calibrated probability"),
        "latency_ms": res["latency_ms"],          # wall time of the parallel section
        "full_pass_ms": res["full_pass_ms"],      # slowest component
        "tile_pass_ms": 0.0, "n_tiles": 0,
        "mode": "accurate",
        "fusion": res["fusion"],
        "models": res["models"],
        "model_errors": res["errors"],
    })


@app.get("/devices")
def devices():
    return alerts.stats()
