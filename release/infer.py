#!/usr/bin/env python3
"""Offline smoke-detection inference — single command, no network needed.

Examples:
  python infer.py --model model_int8.tflite --source image.jpg
  python infer.py --model model.onnx --source folder/ --conf 0.62
  python infer.py --model best.pt --source video.mp4 --out out/

Backends by file extension: .pt (ultralytics), .onnx (onnxruntime CPU),
.tflite (ai-edge-litert / tflite-runtime). Input letterboxed to 1280.
Default conf 0.62 = the sdis-val threshold at 10% negative-image FP for
model A; use 0.68 for ~5%. Latency printed per image is END-TO-END
(preprocess + inference + NMS) on this machine, batch=1.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

IMGSZ = 1280
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def letterbox(im, size=IMGSZ):
    import cv2
    h, w = im.shape[:2]
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    r = cv2.resize(im, (nw, nh))
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = r
    return out, s, left, top


def nms(boxes, scores, iou_t=0.7, max_det=100):
    idx = scores.argsort()[::-1][:1000]
    keep = []
    while len(idx):
        i = idx[0]
        keep.append(i)
        if len(keep) >= max_det:
            break
        xx0 = np.maximum(boxes[i, 0], boxes[idx[1:], 0])
        yy0 = np.maximum(boxes[i, 1], boxes[idx[1:], 1])
        xx1 = np.minimum(boxes[i, 2], boxes[idx[1:], 2])
        yy1 = np.minimum(boxes[i, 3], boxes[idx[1:], 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[idx[1:], 2] - boxes[idx[1:], 0]) * (boxes[idx[1:], 3] - boxes[idx[1:], 1])
        iou = inter / (a + b - inter + 1e-9)
        idx = idx[1:][iou < iou_t]
    return keep


def decode(raw, conf):
    # raw: (1, 5, N) -> cxcywh + score
    p = raw[0].T
    sc = p[:, 4]
    m = sc >= conf
    p, sc = p[m], sc[m]
    if not len(p):
        return np.zeros((0, 4)), np.zeros(0)
    xy = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2,
                   p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1)
    k = nms(xy, sc)
    return xy[k], sc[k]


class Runner:
    def __init__(self, path):
        self.path = Path(path)
        self.kind = self.path.suffix
        if self.kind == ".pt":
            from ultralytics import YOLO
            self.m = YOLO(str(path))
        elif self.kind == ".onnx":
            import onnxruntime as ort
            self.s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.inp = self.s.get_inputs()[0].name
        elif self.kind == ".tflite":
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                from tflite_runtime.interpreter import Interpreter
            self.it = Interpreter(model_path=str(path))
            self.it.allocate_tensors()
            self.ii = self.it.get_input_details()[0]
            self.oo = self.it.get_output_details()[0]

    def __call__(self, bgr, conf):
        if self.kind == ".pt":
            r = self.m.predict(bgr, imgsz=IMGSZ, conf=conf, verbose=False)[0]
            return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), None
        lb, s, dx, dy = letterbox(bgr)
        x = lb[:, :, ::-1].astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        if self.kind == ".onnx":
            raw = self.s.run(None, {self.inp: x})[0]
        else:
            xi = x.transpose(0, 2, 3, 1)
            if self.ii["dtype"] == np.int8:
                sc_, zp = self.ii["quantization"]
                xi = (xi / sc_ + zp).round().clip(-128, 127).astype(np.int8)
            self.it.set_tensor(self.ii["index"], xi)
            self.it.invoke()
            raw = self.it.get_tensor(self.oo["index"])
            if self.oo["dtype"] == np.int8:
                sc_, zp = self.oo["quantization"]
                raw = (raw.astype(np.float32) - zp) * sc_
            if raw.shape[1] != 5:
                raw = raw.transpose(0, 2, 1)
            raw = raw.copy()
            raw[:, :4] *= IMGSZ  # tflite head is normalized
        boxes, scores = decode(raw, conf)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / s
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / s
        return boxes, scores, None


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True, help="image / folder / video")
    ap.add_argument("--conf", type=float, default=0.62)
    ap.add_argument("--out", default="infer_out")
    args = ap.parse_args()
    run = Runner(args.model)
    outd = Path(args.out)
    outd.mkdir(parents=True, exist_ok=True)
    src = Path(args.source)
    imgs = []
    if src.is_dir():
        imgs = [p for p in sorted(src.iterdir()) if p.suffix.lower() in IMG_EXT]
    elif src.suffix.lower() in IMG_EXT:
        imgs = [src]

    results, lats = [], []
    if imgs:
        for p in imgs:
            im = cv2.imread(str(p))
            t0 = time.perf_counter()
            boxes, scores, _ = run(im, args.conf)
            lat = (time.perf_counter() - t0) * 1000
            lats.append(lat)
            for b, s_ in zip(boxes, scores):
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (0, 0, 255), 3)
                cv2.putText(im, f"smoke {s_:.2f}", (int(b[0]), int(b[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(str(outd / p.name), im)
            results.append({"image": str(p), "n_det": len(boxes),
                            "boxes": boxes.tolist(),
                            "scores": scores.tolist(), "latency_ms": round(lat, 1)})
            print(f"{p.name}: {len(boxes)} det, {lat:.0f} ms")
    elif src.suffix.lower() in VID_EXT:
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter(str(outd / (src.stem + "_det.mp4")),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            boxes, scores, _ = run(im, args.conf)
            lats.append((time.perf_counter() - t0) * 1000)
            for b, s_ in zip(boxes, scores):
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (0, 0, 255), 3)
            vw.write(im)
            n += 1
        vw.release()
        print(f"video: {n} frames -> {outd / (src.stem + '_det.mp4')}")
    else:
        raise SystemExit(f"unsupported source: {src}")

    if lats:
        rep = {"model": str(args.model),
               "model_mb": round(Path(args.model).stat().st_size / 1e6, 2),
               "imgsz": IMGSZ, "conf": args.conf, "n": len(lats),
               "latency_ms_median_e2e_batch1": round(float(np.median(lats)), 1),
               "note": "end-to-end (preprocess+inference+NMS) on THIS machine; "
                       "edge-device latency unmeasured unless run on the device"}
        (outd / "RUN_REPORT.json").write_text(json.dumps(
            {"report": rep, "results": results}, indent=1))
        print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
