"""Batch-1 latency probe for the YOLO backend (SmokeDetector), pod or devbox.

Same measurement protocol as cluster/v3/rfdetr_infer.py --latency-n: JPEG decode
excluded, preprocessing + forward + postprocess included, 3 warmup frames, then
one timed detect() per image. Exists so the RF-DETR vs YOLO11m latency table in
reports/v3/rfdetr_eval.md compares like with like on the same hardware.

Usage:
  python yolo_latency.py --weights best.pt --list list.txt --n 50 \
      --device cuda:0 --imgsz 1280 [--root-map local=pod] --out lat.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf-floor", type=float, default=0.02)
    ap.add_argument("--root-map", default=None)
    ap.add_argument("--src-root", default=None, help="dir containing src/ (default: cwd)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.src_root or str(Path(__file__).resolve().parents[2]))
    from src.serve.detector import SmokeDetector

    def resolve(p: str) -> str:
        if args.root_map:
            s, d = args.root_map.split("=", 1)
            if p.startswith(s):
                return d + p[len(s):]
        return p

    paths = [resolve(l.strip()) for l in Path(args.list).read_text().splitlines() if l.strip()]
    det = SmokeDetector(args.weights, device=args.device, imgsz=args.imgsz,
                        tile_size=0, conf_floor=args.conf_floor)
    warm = cv2.imread(paths[0])
    for _ in range(3):
        det.detect(warm)
    lat = []
    for p in paths[: args.n]:
        img = cv2.imread(p)
        if img is None:
            continue
        t0 = time.perf_counter()
        det.detect(img)
        lat.append((time.perf_counter() - t0) * 1000)
    res = {"model": "yolo/SmokeDetector", "weights": args.weights,
           "device": args.device, "imgsz": args.imgsz, "n": len(lat),
           "batch1_ms_mean": round(statistics.mean(lat), 2),
           "batch1_ms_median": round(statistics.median(lat), 2),
           "batch1_ms_p90": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 2),
           "batch1_ms_min": round(min(lat), 2), "batch1_ms_max": round(max(lat), 2)}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
