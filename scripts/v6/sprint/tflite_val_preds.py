"""Full-val predictions from a TFLite file (deployment validation).

Runs the QUANTIZED artifact itself over a local image list (sharded across
processes), writing eval_detection-compatible jsonl: normalized xyxy+score,
conf floor 0.01 (threshold sweeps happen downstream).

Usage (single shard):
  .venv/bin/python scripts/v6/sprint/tflite_val_preds.py \
      --model <.tflite> --list data/yolo/lists/local/sdis_val.txt \
      --out <preds.jsonl> --shard 0 --nshards 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMGSZ = 1280


def letterbox(im: Image.Image):
    import cv2
    a = np.asarray(im)
    h, w = a.shape[:2]
    s = IMGSZ / max(w, h)
    nw, nh = round(w * s), round(h * s)
    r = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
    dx, dy = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = r
    return Image.fromarray(canvas), s, dx, dy, w, h


def nms(boxes, scores, iou_t=0.7, max_det=300):
    idx = scores.argsort()[::-1][:3000]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--conf-floor", type=float, default=0.01)
    args = ap.parse_args()
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
    it = Interpreter(model_path=args.model, num_threads=4)
    it.allocate_tensors()
    ii = it.get_input_details()[0]
    oo = it.get_output_details()[0]

    imgs = [l.strip() for l in Path(args.list).read_text().splitlines() if l.strip()]
    imgs = imgs[args.shard::args.nshards]
    outp = Path(args.out)
    done = set()
    if outp.exists():
        for l in outp.open():
            try:
                done.add(json.loads(l)["image"])
            except Exception:
                pass
    with outp.open("a") as f:
        for n, p in enumerate(imgs):
            if p in done:
                continue
            im = Image.open(p).convert("RGB")
            lb, s, dx, dy, W, H = letterbox(im)
            x = (np.asarray(lb, np.float32) / 255.0)[None]
            if ii["dtype"] == np.int8:
                sc, zp = ii["quantization"]
                x = (x / sc + zp).round().clip(-128, 127).astype(np.int8)
            it.set_tensor(ii["index"], x)
            it.invoke()
            raw = it.get_tensor(oo["index"])
            if oo["dtype"] == np.int8:
                sc, zp = oo["quantization"]
                raw = (raw.astype(np.float32) - zp) * sc
            if raw.shape[1] != 5:
                raw = raw.transpose(0, 2, 1)
            pr = raw[0].T  # (N,5) cxcywh+score, normalized coords
            m = pr[:, 4] >= args.conf_floor
            pr = pr[m]
            preds = []
            if len(pr):
                xy = np.stack([pr[:, 0] - pr[:, 2] / 2, pr[:, 1] - pr[:, 3] / 2,
                               pr[:, 0] + pr[:, 2] / 2, pr[:, 1] + pr[:, 3] / 2], 1) * IMGSZ
                k = nms(xy, pr[:, 4])
                for i in k:
                    b = xy[i]
                    preds.append([
                        round(float((b[0] - dx) / s / W), 6),
                        round(float((b[1] - dy) / s / H), 6),
                        round(float((b[2] - dx) / s / W), 6),
                        round(float((b[3] - dy) / s / H), 6),
                        round(float(pr[i, 4]), 6)])
            f.write(json.dumps({"image": p, "preds": preds}) + "\n")
            if n % 100 == 0:
                f.flush()
                print(f"shard {args.shard}: {n}/{len(imgs)}", flush=True)
    print("SHARD_DONE", args.shard)


if __name__ == "__main__":
    main()
