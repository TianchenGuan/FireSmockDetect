"""Shadow (bypass) recorder for the smoke-detection service (iteration 5).

A second detector ("shadow") sees exactly the same image bytes as the primary
model, runs in a dedicated FIFO worker thread, maintains its OWN AlertEngine
(own thresholds), and appends one JSONL record per /detect request to
`settings.shadow_log_path`. It never touches the HTTP response: the primary
result is captured before the job is queued and the response is sent without
waiting for the shadow model. Per-device alert history stays causal because
the single worker consumes jobs strictly in arrival order.

Record schema (one line per request, written when the shadow pass finishes):
  {ts, device_id, capture_ts, source, image_sha256, image_size,
   client_metadata,
   primary: {model_version, max_smoke, alert, alert_info, n_dets, latency_ms},
   shadow:  {backend, model_version, max_smoke, alert, alert_info, n_dets,
             latency_ms, conf_single, conf_seq}}
`source` is mandatory and is one of SOURCES: a historical_replay record is a
re-posted stored frame, never a field observation. Alert fields are detector
states, not confirmed wildfire reports; shadow silence is not evidence of
no-smoke.
"""
from __future__ import annotations

import hashlib
import io
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .alerting import AlertEngine
from .backends import load_backend

SOURCES = ("live_upload", "historical_replay")
DEFAULT_SOURCE = "live_upload"


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def alert_summary(alert: dict) -> dict:
    """Compact, history-relevant part of an AlertEngine.update() result."""
    return {k: alert.get(k) for k in ("window_hits", "window_size", "history_reset",
                                      "cooldown_active")}


@dataclass
class ShadowJob:
    raw: bytes                  # the exact upload bytes (decoded again in the worker)
    phash: object               # perceptual hash computed by the primary path
    device_id: str
    capture_ts: float
    source: str
    image_size: list
    client_metadata: str | None
    primary: dict               # {model_version, max_smoke, alert, alert_info, n_dets, latency_ms}
    request_ts: float


class ShadowRecorder:
    def __init__(self, settings, root: Path):
        self.cfg = settings
        self.root = Path(root)
        self.enabled = bool(settings.shadow_backend)
        self.detector = None
        self.alerts: AlertEngine | None = None
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(settings.shadow_queue_max)))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._fh = None
        self.records_written = 0
        self.records_by_source = {s: 0 for s in SOURCES}
        self.alerts_by_state = {}
        self.dropped = 0
        self.errors = 0
        self.last_error: str | None = None
        self._lat_sum = 0.0
        self._lat_max = 0.0
        self._lat_n = 0
        lp = Path(settings.shadow_log_path)
        self.log_path = lp if lp.is_absolute() else self.root / lp
        wp = settings.shadow_weights_path
        self.weights_path = (Path(wp) if Path(wp).is_absolute() else self.root / wp) if wp else None
        self.model_version = settings.shadow_model_version or (Path(wp).name if wp else "")
        self.device = settings.shadow_device or settings.device

    # ------------------------------------------------------------ lifecycle
    def start(self):
        if not self.enabled:
            return
        if not self.weights_path:
            raise ValueError("SMOKE_SHADOW_BACKEND is set but SMOKE_SHADOW_WEIGHTS_PATH is empty")
        cfg = self.cfg
        self.detector = load_backend(
            cfg.shadow_backend, weights_path=str(self.weights_path), device=self.device,
            imgsz=cfg.shadow_imgsz, tile_size=0, tile_overlap=cfg.tile_overlap,
            merge_iou=cfg.merge_iou, conf_floor=cfg.shadow_conf_floor,
            backbone_dir=cfg.shadow_dino_backbone_dir or None,
            lora_r=cfg.shadow_lora_r, lora_alpha=cfg.shadow_lora_alpha,
            lora_merge=cfg.shadow_lora_merge,
            width=cfg.shadow_det_width or None, height=cfg.shadow_det_height or None)
        # own AlertEngine: same persistence/cooldown/scene rules, own thresholds
        shadow_cfg = cfg.model_copy(update={"single_frame_alert_conf": cfg.shadow_conf_single,
                                            "seq_frame_conf": cfg.shadow_conf_seq})
        self.alerts = AlertEngine(shadow_cfg)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", buffering=1)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="shadow-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0):
        if not self._thread:
            return
        self._stop.set()
        self._thread.join(timeout)
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None

    # ------------------------------------------------------------ producer
    def submit(self, job: ShadowJob) -> bool:
        """Non-blocking; returns False (and counts a drop) when the queue is full."""
        if not self.enabled or self._thread is None:
            return False
        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
            with self._lock:
                self.dropped += 1
            return False

    # ------------------------------------------------------------ worker
    def _run(self):
        while not (self._stop.is_set() and self._q.empty()):
            try:
                job = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process(job)
            except Exception as e:  # never let the shadow kill the service
                with self._lock:
                    self.errors += 1
                    self.last_error = f"{type(e).__name__}: {e}"[:300]
            finally:
                self._q.task_done()

    def _process(self, job: ShadowJob):
        pil = Image.open(io.BytesIO(job.raw)).convert("RGB")
        img = np.asarray(pil)[:, :, ::-1]  # RGB -> BGR, same as the primary path
        dets, timing = self.detector.detect(img)
        smoke = [d for d in dets if d.cls_name.lower() == "smoke"
                 or len(self.detector.class_names) == 1]
        max_smoke = max((d.score for d in smoke), default=0.0)
        alert = self.alerts.update(job.device_id, job.capture_ts, max_smoke, phash=job.phash)
        rec = {
            "ts": round(time.time(), 3),
            "device_id": job.device_id,
            "capture_ts": job.capture_ts,
            "source": job.source,
            "image_sha256": sha256_hex(job.raw),
            "image_size": job.image_size,
            "client_metadata": job.client_metadata,
            "primary": job.primary,
            "shadow": {
                "backend": self.cfg.shadow_backend,
                "model_version": self.model_version,
                "max_smoke": round(max_smoke, 4),
                "alert": alert["state"],
                "alert_info": alert_summary(alert),
                "n_dets": len(dets),
                "latency_ms": timing.get("latency_ms"),
                "conf_single": self.cfg.shadow_conf_single,
                "conf_seq": self.cfg.shadow_conf_seq,
            },
            "queue_wait_ms": round((time.time() - job.request_ts) * 1000
                                   - float(timing.get("latency_ms") or 0.0), 1),
        }
        line = json.dumps(rec, separators=(",", ":"))
        with self._lock:
            if self._fh is None:
                return
            self._fh.write(line + "\n")
            self.records_written += 1
            self.records_by_source[job.source] = self.records_by_source.get(job.source, 0) + 1
            self.alerts_by_state[alert["state"]] = self.alerts_by_state.get(alert["state"], 0) + 1
            lat = float(timing.get("latency_ms") or 0.0)
            self._lat_sum += lat
            self._lat_max = max(self._lat_max, lat)
            self._lat_n += 1

    # ------------------------------------------------------------ status
    def status(self) -> dict:
        base = {"enabled": self.enabled}
        if not self.enabled:
            return base
        with self._lock:
            base.update({
                "backend": self.cfg.shadow_backend,
                "weights_path": str(self.weights_path),
                "model_version": self.model_version,
                "device": self.device,
                "imgsz": self.cfg.shadow_imgsz,
                "conf_floor": self.cfg.shadow_conf_floor,
                "conf_single": self.cfg.shadow_conf_single,
                "conf_seq": self.cfg.shadow_conf_seq,
                "log_path": str(self.log_path),
                "loaded": self.detector is not None,
                "worker_alive": bool(self._thread and self._thread.is_alive()),
                "queue_depth": self._q.qsize(),
                "queue_max": self._q.maxsize,
                "records_written": self.records_written,
                "records_by_source": dict(self.records_by_source),
                "alerts_by_state": dict(self.alerts_by_state),
                "dropped": self.dropped,
                "errors": self.errors,
                "last_error": self.last_error,
                "latency_ms_mean": round(self._lat_sum / self._lat_n, 1) if self._lat_n else None,
                "latency_ms_max": round(self._lat_max, 1) if self._lat_n else None,
            })
            if self.detector is not None and hasattr(self.detector, "info"):
                base["backend_info"] = self.detector.info()
        return base
