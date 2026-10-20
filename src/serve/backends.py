"""Selectable inference backends for the smoke-detection service.

Contract (frozen for v3): a backend exposes
    detect(img_bgr: np.ndarray) -> tuple[list[Detection], dict]   # timing dict
    filter_by_conf(dets, conf) -> list[Detection]
    class_names: dict[int, str]
Detection is src.serve.detector.Detection (pixel xyxy + score + cls + source).

The default 'yolo' backend is the existing SmokeDetector, unchanged.
New v3 backends register here as their tracks deliver checkpoints; a name
listed in AVAILABLE but not yet implemented raises a clear error instead of
silently falling back.
"""
from __future__ import annotations

from src.serve.detector import Detection, SmokeDetector

AVAILABLE = ("yolo", "deimv2", "rfdetr", "dino3", "yolo_temporal")


def load_backend(name: str, *, weights_path: str, device: str = "cpu",
                 imgsz: int = 1280, **kw):
    name = name.lower()
    if name == "yolo":
        return SmokeDetector(weights_path, device=device, imgsz=imgsz,
                             tile_size=kw.get("tile_size", 0),
                             tile_overlap=kw.get("tile_overlap", 0.2),
                             merge_iou=kw.get("merge_iou", 0.55),
                             conf_floor=kw.get("conf_floor", 0.05),
                             tile_conf_offset=kw.get("tile_conf_offset", 0.0))
    if name == "deimv2":
        from src.serve.backend_deimv2 import DeimV2Backend  # provided by track A
        return DeimV2Backend(weights_path, device=device, imgsz=imgsz, **kw)
    if name == "rfdetr":
        from src.serve.backend_rfdetr import RfDetrBackend  # provided by track A
        return RfDetrBackend(weights_path, device=device, imgsz=imgsz, **kw)
    if name == "dino3":
        from src.serve.backend_dino3 import Dino3Backend    # provided by track B/C
        return Dino3Backend(weights_path, device=device, imgsz=imgsz, **kw)
    if name == "yolo_temporal":
        from src.serve.backend_temporal import TemporalBackend  # track D
        return TemporalBackend(weights_path, device=device, imgsz=imgsz, **kw)
    raise ValueError(f"unknown backend {name!r}; available: {AVAILABLE}")
