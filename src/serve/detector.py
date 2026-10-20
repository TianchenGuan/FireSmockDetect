"""YOLO smoke detector wrapper: full-image and overlapping-tile inference.

Raw detector confidence is NOT a calibrated probability; scores are only
comparable against thresholds selected on validation data.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO


@dataclass
class Detection:
    # pixel xyxy in the original image
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls_id: int
    cls_name: str
    source: str  # "full" | "tile"

    def as_dict(self):
        d = asdict(self)
        d["bbox_xyxy"] = [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1)]
        for k in ("x1", "y1", "x2", "y2"):
            d.pop(k)
        d["score"] = round(self.score, 4)
        return d


class SmokeDetector:
    def __init__(self, weights_path: str, device: str = "cpu", imgsz: int = 1024,
                 tile_size: int = 0, tile_overlap: float = 0.2, merge_iou: float = 0.55,
                 conf_floor: float = 0.05, tile_conf_offset: float = 0.0):
        self.model = YOLO(weights_path)
        self.device = device
        self.imgsz = imgsz
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.merge_iou = merge_iou
        self.conf_floor = conf_floor
        # extra confidence demanded of tile-sourced detections at filter time
        # (tiles magnify faint texture -> more low-conf false alarms); 0.0 = off.
        self.tile_conf_offset = tile_conf_offset
        self.class_names = self.model.names  # {id: name}

    def filter_by_conf(self, dets: list[Detection], conf: float,
                       tile_conf_offset: float | None = None) -> list[Detection]:
        """Post-filter detections at a base threshold, with a stricter bar for tiles.

        Detections with source == "tile" must clear (conf + tile_conf_offset);
        all others just need conf. With the default offset of 0.0 this is a
        plain score threshold (existing behavior unchanged).
        """
        off = self.tile_conf_offset if tile_conf_offset is None else tile_conf_offset
        return [d for d in dets
                if d.score >= (conf + off if d.source == "tile" else conf)]

    def _predict_array(self, img: np.ndarray, source_tag: str, offset=(0, 0)) -> list[Detection]:
        res = self.model.predict(img, imgsz=self.imgsz, conf=self.conf_floor,
                                 device=self.device, verbose=False)[0]
        out = []
        ox, oy = offset
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cid = int(b.cls[0])
            out.append(Detection(x1 + ox, y1 + oy, x2 + ox, y2 + oy,
                                 float(b.conf[0]), cid,
                                 self.class_names.get(cid, str(cid)), source_tag))
        return out

    def _tiles(self, w: int, h: int):
        ts, ov = self.tile_size, self.tile_overlap
        stride = max(1, int(ts * (1 - ov)))
        xs = list(range(0, max(w - ts, 0) + 1, stride)) or [0]
        ys = list(range(0, max(h - ts, 0) + 1, stride)) or [0]
        if xs[-1] + ts < w:
            xs.append(w - ts)
        if ys[-1] + ts < h:
            ys.append(h - ts)
        for y in ys:
            for x in xs:
                yield x, y, min(x + ts, w), min(y + ts, h)

    def _merge(self, dets: list[Detection]) -> list[Detection]:
        if not dets:
            return []
        boxes = torch.tensor([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=torch.float32)
        scores = torch.tensor([d.score for d in dets])
        # class-aware NMS via coordinate offset trick
        cls = torch.tensor([d.cls_id for d in dets], dtype=torch.float32)
        offset = cls[:, None] * 1e5
        keep = nms(boxes + offset, scores, self.merge_iou)
        return [dets[i] for i in keep.tolist()]

    def detect(self, img: np.ndarray) -> tuple[list[Detection], dict]:
        """img: HWC BGR or RGB uint8 (ultralytics handles numpy as BGR)."""
        t0 = time.perf_counter()
        h, w = img.shape[:2]
        dets = self._predict_array(img, "full")
        t_full = time.perf_counter() - t0
        t_tile = 0.0
        n_tiles = 0
        if self.tile_size and (w > self.tile_size or h > self.tile_size):
            t1 = time.perf_counter()
            for x1, y1, x2, y2 in self._tiles(w, h):
                n_tiles += 1
                dets += self._predict_array(img[y1:y2, x1:x2], "tile", offset=(x1, y1))
            t_tile = time.perf_counter() - t1
            dets = self._merge(dets)
        timing = {"latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                  "full_pass_ms": round(t_full * 1000, 1),
                  "tile_pass_ms": round(t_tile * 1000, 1), "n_tiles": n_tiles}
        return dets, timing
