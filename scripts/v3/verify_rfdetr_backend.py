"""Verify src/serve/backend_rfdetr.py against upstream rfdetr predict().

Runs both paths on the same images (CPU by default) and reports, per image,
the number of detections, max |score| delta and max |box| delta after
matching rows in score order. Also prints batch-1 latency stats so the same
script doubles as the CPU latency measurement for reports/v3/rfdetr_eval.md.

Usage:
  python scripts/v3/verify_rfdetr_backend.py --weights <ckpt> --n 3 \
      [--device cpu] [--imgsz 1280] [--latency-n 10] [--skip-upstream]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.serve.backend_rfdetr import RfDetrBackend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list", default=str(ROOT / "data/yolo/lists/local/sdis_val.txt"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf-floor", type=float, default=0.02)
    ap.add_argument("--latency-n", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=default)")
    ap.add_argument("--skip-upstream", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    imgs = [Path(l.strip()) for l in Path(args.list).read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(0)
    pick = [imgs[i] for i in sorted(rng.choice(len(imgs), max(args.n, args.latency_n), replace=False))]

    be = RfDetrBackend(args.weights, device=args.device, imgsz=args.imgsz,
                       conf_floor=args.conf_floor)
    print("class_names:", be.class_names, flush=True)

    report = {"weights": args.weights, "device": args.device, "imgsz": args.imgsz,
              "images": [], "latency": {}}
    for p in pick[: args.n]:
        img = cv2.imread(str(p))
        dets, timing = be.detect(img)
        row = {"image": str(p), "n_dets": len(dets), "timing": timing,
               "top_scores": [round(d.score, 4) for d in
                              sorted(dets, key=lambda d: -d.score)[:5]]}
        if not args.skip_upstream:
            det_up = be._rf.predict(np.ascontiguousarray(img[:, :, ::-1]),
                                    threshold=args.conf_floor)
            ours = sorted(dets, key=lambda d: -d.score)
            up_order = np.argsort(-det_up.confidence)
            row["n_dets_upstream"] = int(len(det_up))
            n = min(len(ours), len(det_up))
            if n:
                ds = max(abs(ours[i].score - float(det_up.confidence[up_order[i]]))
                         for i in range(n))
                db = max(float(np.abs(np.array([ours[i].x1, ours[i].y1, ours[i].x2, ours[i].y2])
                                      - det_up.xyxy[up_order[i]]).max()) for i in range(n))
                row["max_score_delta"] = round(ds, 6)
                row["max_box_delta_px"] = round(db, 4)
        report["images"].append(row)
        print(json.dumps(row), flush=True)

    if args.latency_n:
        # warmup
        warm = cv2.imread(str(pick[0]))
        for _ in range(2):
            be.detect(warm)
        lat, pre, inf = [], [], []
        for p in pick[: args.latency_n]:
            img = cv2.imread(str(p))
            t0 = time.perf_counter()
            _, tm = be.detect(img)
            lat.append((time.perf_counter() - t0) * 1000)
            pre.append(tm["preprocess_ms"])
            inf.append(tm["infer_ms"])
        report["latency"] = {
            "n": len(lat), "device": args.device,
            "threads": args.threads or None,
            "batch1_ms_mean": round(statistics.mean(lat), 1),
            "batch1_ms_median": round(statistics.median(lat), 1),
            "batch1_ms_min": round(min(lat), 1), "batch1_ms_max": round(max(lat), 1),
            "preprocess_ms_median": round(statistics.median(pre), 1),
            "infer_ms_median": round(statistics.median(inf), 1),
        }
        print(json.dumps(report["latency"], indent=2), flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print("saved:", args.out)


if __name__ == "__main__":
    main()
