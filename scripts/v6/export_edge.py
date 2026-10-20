"""V6 edge export: YOLO11 checkpoint -> ONNX (+parity) -> TFLite FP16/INT8.

Parity gate: raw-output max|Δ| between torch and onnxruntime on real val
images, plus box-level agreement through ultralytics' own ONNX backend.
INT8 PTQ calibration uses TRAIN/VAL images only (local yaml) — never test.
Latency measured on this host's CPU and clearly labeled a proxy, not a
phone/drone number.

Usage:
  .venv/bin/python scripts/v6/export_edge.py --ckpt models/v6/.../best.pt \
      --imgsz 1280 --calib-yaml data/v6/yamls/calib_local.yaml \
      --n-parity 64 --out models/v6/edge/<name>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def letterbox(im, new=1280):
    import cv2
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im2 = cv2.resize(im, (nw, nh))
    top = (new - nh) // 2
    left = (new - nw) // 2
    out = np.full((new, new, 3), 114, im.dtype)
    out[top:top + nh, left:left + nw] = im2
    return out, r, left, top


def preprocess(path, imgsz):
    import cv2
    im = cv2.imread(str(path))[:, :, ::-1]
    lb, r, dx, dy = letterbox(im, imgsz)
    x = lb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, (im.shape[0], im.shape[1], r, dx, dy)


def decode(raw, meta, conf_t=0.01, iou_t=0.7, max_det=300):
    """raw (1, 4+nc, N) decoded xywh(px@imgsz)+scores -> norm xyxy+score."""
    import torch
    from torchvision.ops import nms
    p = torch.from_numpy(np.asarray(raw))[0]          # (5, N) for nc=1
    boxes, scores = p[:4].T, p[4:].max(0).values
    keep = scores > conf_t
    boxes, scores = boxes[keep], scores[keep]
    if not len(boxes):
        return []
    xy = boxes.clone()
    xy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    ki = nms(xy, scores, iou_t)[:max_det]
    xy, scores = xy[ki], scores[ki]
    H, W, r, dx, dy = meta
    out = []
    for (x1, y1, x2, y2), s in zip(xy.tolist(), scores.tolist()):
        out.append([max(0.0, (x1 - dx) / r / W), max(0.0, (y1 - dy) / r / H),
                    min(1.0, (x2 - dx) / r / W), min(1.0, (y2 - dy) / r / H),
                    float(s)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--calib-yaml", required=True)
    ap.add_argument("--parity-list", default="data/yolo/lists/local/sdis_val.txt")
    ap.add_argument("--n-parity", type=int, default=64)
    ap.add_argument("--n-latency", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-int8", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    report = {"ckpt": args.ckpt, "imgsz": args.imgsz}

    import torch
    from ultralytics import YOLO
    model = YOLO(str(ROOT / args.ckpt))
    report["params_m"] = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 3)

    onnx_path = model.export(format="onnx", imgsz=args.imgsz, simplify=True,
                             dynamic=False, nms=False)
    onnx_dst = out / "model.onnx"
    Path(onnx_path).rename(onnx_dst)
    report["onnx_mb"] = round(onnx_dst.stat().st_size / 1e6, 2)

    # ---- raw parity torch vs ORT on real val images
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_dst), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    imgs = [l.strip() for l in (ROOT / args.parity_list).read_text().splitlines()
            if l.strip()][:args.n_parity]
    tmodel = model.model.eval().float()
    maxd, n_box_match, n_boxes = 0.0, 0, 0
    lat_ort = []
    for p in imgs:
        x, meta = preprocess(p, args.imgsz)
        with torch.no_grad():
            t_raw = tmodel(torch.from_numpy(x))[0].numpy()
        t0 = time.time()
        o_raw = sess.run(None, {iname: x})[0]
        lat_ort.append(time.time() - t0)
        maxd = max(maxd, float(np.abs(t_raw - o_raw).max()))
        bt = decode(t_raw, meta)
        bo = decode(o_raw, meta)
        n_boxes += len(bt)
        n_box_match += sum(1 for a, b in zip(bt, bo)
                           if abs(a[4] - b[4]) < 1e-3)
    report["parity"] = {"n_images": len(imgs), "raw_max_abs_diff": maxd,
                        "boxes_torch": n_boxes, "boxes_score_matched": n_box_match}
    report["latency_cpu_onnx_ms"] = round(1000 * float(np.median(lat_ort)), 1)

    # ---- TFLite (fp16 + int8 PTQ on train/val calibration)
    try:
        fp16 = model.export(format="tflite", imgsz=args.imgsz, half=True)
        p16 = out / "model_fp16.tflite"
        Path(fp16).rename(p16)
        report["tflite_fp16_mb"] = round(p16.stat().st_size / 1e6, 2)
    except Exception as e:
        report["tflite_fp16_error"] = str(e)[:300]
    if not args.skip_int8:
        try:
            i8 = model.export(format="tflite", imgsz=args.imgsz, int8=True,
                              data=str(ROOT / args.calib_yaml), fraction=0.03)
            pi8 = out / "model_int8.tflite"
            Path(i8).rename(pi8)
            report["tflite_int8_mb"] = round(pi8.stat().st_size / 1e6, 2)
        except Exception as e:
            report["tflite_int8_error"] = str(e)[:300]

    # ---- TFLite latency + per-format preds for accuracy deltas
    def tflite_preds(tfl, img_list, out_jsonl, n=None):
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=str(tfl), num_threads=8)
        it.allocate_tensors()
        inp = it.get_input_details()[0]
        outd = it.get_output_details()[0]
        lats, recs = [], []
        for p in (img_list[:n] if n else img_list):
            x, meta = preprocess(p, args.imgsz)
            xin = x.transpose(0, 2, 3, 1) if inp["shape"][-1] == 3 else x
            if inp["dtype"] == np.int8:
                s, z = inp["quantization"]
                xin = (xin / s + z).round().clip(-128, 127).astype(np.int8)
            it.set_tensor(inp["index"], xin)
            t0 = time.time()
            it.invoke()
            lats.append(time.time() - t0)
            y = it.get_tensor(outd["index"])
            if outd["dtype"] == np.int8:
                s, z = outd["quantization"]
                y = (y.astype(np.float32) - z) * s
            if y.shape[-1] != len(imgs) and y.ndim == 3 and y.shape[1] > y.shape[2]:
                y = y.transpose(0, 2, 1)
            # ultralytics tflite output is normalized 0..1 xywh — rescale
            yy = np.array(y, np.float32)
            if yy[0, :4].max() <= 1.5:
                yy[0, :4] *= args.imgsz
            recs.append({"image": p, "preds": decode(yy, meta)})
        (out / out_jsonl).write_text("\n".join(json.dumps(r) for r in recs))
        return round(1000 * float(np.median(lats)), 1)

    for name, f in (("fp16", out / "model_fp16.tflite"),
                    ("int8", out / "model_int8.tflite")):
        if f.exists():
            try:
                ms = tflite_preds(f, imgs, f"parity_preds_{name}.jsonl",
                                  n=args.n_latency)
                report[f"latency_cpu_tflite_{name}_ms"] = ms
            except Exception as e:
                report[f"tflite_{name}_run_error"] = str(e)[:300]

    report["note"] = ("CPU x86 latency is a PROXY. Real phone/drone latency "
                      "and power are UNMEASURED (no device attached).")
    (out / "EXPORT_REPORT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
