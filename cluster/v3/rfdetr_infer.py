"""Pod-side RF-DETR inference over an image list (one shard per GPU).

Runs the SAME code path as the serving backend (src/serve/backend_rfdetr.py,
fetched per-job) so cluster numbers and devbox numbers cannot drift. Emits one
JSONL row per image with predictions normalized to the ORIGINAL image size:

    {"image": "<path as listed>", "w": W, "h": H,
     "preds": [[x1n, y1n, x2n, y2n, score], ...], "latency_ms": ...}

Ground truth is NEVER read here: the sweep is computed offline on the devbox
from the frozen YOLO label files, so the pod only needs images + weights.

The container (smoke-det3:v1) has no `ultralytics`, but src/serve/detector.py
imports it at module scope and we only need its `Detection` dataclass -> a
stub module is registered before the import. This keeps ONE inference
implementation instead of a pod-side copy.

Usage (per shard):
  python rfdetr_infer.py --weights <ckpt> --list list.txt --shard 0/8 \
      --device-index 0 --root-map "<local prefix>=<pod prefix>" \
      --out /tmp/out/shard0.jsonl [--latency-n 50 --latency-out lat0.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import types
from pathlib import Path


def _stub_ultralytics():
    try:
        import ultralytics  # noqa: F401
    except ModuleNotFoundError:
        m = types.ModuleType("ultralytics")
        m.YOLO = None  # only Detection is needed from src.serve.detector
        sys.modules["ultralytics"] = m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--shard", default="0/1", help="i/N")
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf-floor", type=float, default=0.02)
    ap.add_argument("--root-map", default=None, help="localprefix=podprefix")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--latency-n", type=int, default=0,
                    help="batch-1 latency measurement on the first N images (after warmup)")
    ap.add_argument("--latency-out", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _stub_ultralytics()
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # /tmp/job (src/ next to it)
    from src.serve.backend_rfdetr import RfDetrBackend

    i, n = (int(x) for x in args.shard.split("/"))
    listed = [l.strip() for l in Path(args.list).read_text().splitlines() if l.strip()]
    if args.max_images:
        listed = listed[: args.max_images]
    mine = listed[i::n]

    def resolve(p: str) -> str:
        if args.root_map:
            src, dst = args.root_map.split("=", 1)
            if p.startswith(src):
                return dst + p[len(src):]
        return p

    dev = args.device if args.device == "cpu" else f"{args.device}:{args.device_index}"
    be = RfDetrBackend(args.weights, device=dev, imgsz=args.imgsz,
                       conf_floor=args.conf_floor)
    print(f"shard {i}/{n}: {len(mine)} images, device={dev}, "
          f"class_names={be.class_names}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    with out.open("w") as fh:
        for k, listed_path in enumerate(mine):
            ip = resolve(listed_path)
            img = cv2.imread(ip)
            if img is None:
                fh.write(json.dumps({"image": listed_path, "error": "unreadable"}) + "\n")
                continue
            h, w = img.shape[:2]
            dets, timing = be.detect(img)
            preds = [[round(d.x1 / w, 6), round(d.y1 / h, 6),
                      round(d.x2 / w, 6), round(d.y2 / h, 6), round(d.score, 6)]
                     for d in dets]
            fh.write(json.dumps({"image": listed_path, "w": w, "h": h,
                                 "preds": preds,
                                 "latency_ms": timing["latency_ms"]}) + "\n")
            if (k + 1) % 200 == 0:
                el = time.time() - t_start
                print(f"shard {i}: {k+1}/{len(mine)} ({el:.0f}s, "
                      f"{(k+1)/el:.2f} img/s)", flush=True)

    if args.latency_n and args.latency_out:
        # batch-1 latency INCLUDING preprocessing (decode excluded, as for the
        # devbox measurement): warm up twice, then time detect() per image.
        warm = cv2.imread(resolve(mine[0]))
        for _ in range(3):
            be.detect(warm)
        lat, pre, inf = [], [], []
        for listed_path in mine[: args.latency_n]:
            img = cv2.imread(resolve(listed_path))
            if img is None:
                continue
            t0 = time.perf_counter()
            _, tm = be.detect(img)
            lat.append((time.perf_counter() - t0) * 1000)
            pre.append(tm["preprocess_ms"])
            inf.append(tm["infer_ms"])
        Path(args.latency_out).write_text(json.dumps({
            "device": dev, "imgsz": args.imgsz, "n": len(lat),
            "batch1_ms_mean": round(statistics.mean(lat), 2),
            "batch1_ms_median": round(statistics.median(lat), 2),
            "batch1_ms_p90": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 2),
            "batch1_ms_min": round(min(lat), 2), "batch1_ms_max": round(max(lat), 2),
            "preprocess_ms_median": round(statistics.median(pre), 2),
            "infer_ms_median": round(statistics.median(inf), 2),
        }, indent=2))
    el = time.time() - t_start
    print(f"shard {i} DONE {len(mine)} imgs in {el:.0f}s ({len(mine)/max(el,1):.2f} img/s)",
          flush=True)


if __name__ == "__main__":
    main()
