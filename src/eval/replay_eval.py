"""Causal early-detection evaluation on FIgLib-style sequences.

For each event: iterate frames in chronological order (current + past frames
only — the detector is single-frame; the alert engine only sees history).
Detection delay = time_offset_s of the first alerting frame, measured from
FIgLib's first-visible-smoke frame (offset 0). Reports missed events
separately from delay among detected events.

FIgLib has no bounding boxes: this measures event-level alerting, not
localization quality. Frames before offset 0 that trigger alerts count as
pre-visible-smoke alerts (reported, not automatically 'false' — smoke may be
faintly visible before the labeled frame; they are listed for review).

Usage:
  python -m src.eval.replay_eval --manifest data/manifests/figlib_subset_manifest.parquet \
    --weights <best.pt> --imgsz 1152 [--tile-size 1024] --out reports/replay_<tag>.json

  # non-default backend (src/serve/backends.py):
  python -m src.eval.replay_eval ... --backend rfdetr --weights <ckpt.pth>

  # frame scores computed elsewhere (e.g. a GPU job): the detector is
  # single-frame and the AlertEngine is a deterministic function of the
  # per-frame max smoke score, so scoring and replay can be separated without
  # changing any result. --scores-jsonl accepts the per-image prediction rows
  # written by cluster/v3/rfdetr_infer.py (or {"image","max_smoke"} rows).
  python -m src.eval.replay_eval --manifest ... --scores-jsonl preds.jsonl \
      --conf-single 0.30 --conf-seq 0.15 --out reports/replay_<tag>.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.serve.detector import SmokeDetector  # noqa: E402
from src.serve.alerting import AlertEngine    # noqa: E402
from src.serve.config import Settings         # noqa: E402


def load_frame_scores(path: str, conf_floor: float) -> dict[str, float]:
    """Per-frame max smoke score from a prediction/score JSONL.

    Accepts either {"image", "preds": [[x1,y1,x2,y2,score], ...]} (the format
    written by cluster/v3/rfdetr_infer.py) or {"image", "max_smoke": float}.
    Keys are stored both as written and as resolved absolute paths so manifests
    with relative frame_path values still match.
    """
    scores: dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "max_smoke" in rec:
            mx = float(rec["max_smoke"])
        else:
            mx = max((float(p[4]) for p in rec.get("preds", [])
                      if float(p[4]) >= conf_floor), default=0.0)
        key = rec["image"]
        scores[key] = mx
        scores[str(Path(key).resolve())] = mx
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--weights", default=None,
                    help="required unless --scores-jsonl is given")
    ap.add_argument("--backend", default="yolo",
                    help="src/serve/backends.py backend name (default yolo)")
    ap.add_argument("--scores-jsonl", default=None,
                    help="precomputed per-frame predictions/scores (no model loaded)")
    ap.add_argument("--imgsz", type=int, default=1152)
    ap.add_argument("--tile-size", type=int, default=0)
    ap.add_argument("--tile-conf-offset", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf-single", type=float, default=None)
    ap.add_argument("--conf-seq", type=float, default=None)
    ap.add_argument("--events", default=None, help="comma list to subset")
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.weights and not args.scores_jsonl:
        ap.error("--weights is required unless --scores-jsonl is given")

    df = pd.read_parquet(args.manifest) if args.manifest.endswith(".parquet") \
        else pd.read_csv(args.manifest)
    if args.events:
        df = df[df.event_id.isin(args.events.split(","))]
    events = sorted(df.event_id.unique())
    if args.max_events:
        events = events[: args.max_events]

    cfg = Settings()
    if args.conf_single is not None:
        cfg.single_frame_alert_conf = args.conf_single
    if args.conf_seq is not None:
        cfg.seq_frame_conf = args.conf_seq

    det = None
    frame_scores = None
    if args.scores_jsonl:
        frame_scores = load_frame_scores(args.scores_jsonl, cfg.conf_floor)
        print(f"replay from precomputed scores: {len(frame_scores)} keys")
    elif args.backend == "yolo":
        det = SmokeDetector(args.weights, device=args.device, imgsz=args.imgsz,
                            tile_size=args.tile_size, conf_floor=cfg.conf_floor,
                            tile_conf_offset=args.tile_conf_offset)
    else:
        from src.serve.backends import load_backend  # noqa: PLC0415
        det = load_backend(args.backend, weights_path=args.weights,
                           device=args.device, imgsz=args.imgsz,
                           tile_size=args.tile_size, conf_floor=cfg.conf_floor,
                           tile_conf_offset=args.tile_conf_offset)

    results = {}
    n_missing_scores = 0
    for ev in events:
        sub = df[df.event_id == ev].sort_values("time_offset_s")
        engine = AlertEngine(cfg)  # fresh per event: no cross-event history
        first_alert, per_frame = None, []
        for _, row in sub.iterrows():
            if frame_scores is not None:
                fp = str(row.frame_path)
                if fp in frame_scores:
                    mx = frame_scores[fp]
                elif str(Path(fp).resolve()) in frame_scores:
                    mx = frame_scores[str(Path(fp).resolve())]
                else:
                    n_missing_scores += 1
                    per_frame.append({"offset": row.time_offset_s, "error": "no_score"})
                    continue
            else:
                img = cv2.imread(str(row.frame_path))
                if img is None:
                    per_frame.append({"offset": row.time_offset_s, "error": "unreadable"})
                    continue
                dets, timing = det.detect(img)
                dets = det.filter_by_conf(dets, cfg.conf_floor)
                smoke = [d for d in dets
                         if d.cls_name.lower() == "smoke" or len(det.class_names) == 1]
                mx = max((d.score for d in smoke), default=0.0)
            alert = engine.update(ev, 1_700_000_000 + row.time_offset_s, mx)
            per_frame.append({"offset": float(row.time_offset_s),
                              "max_smoke": round(mx, 4),
                              "state": alert["state"]})
            if first_alert is None and alert["state"] in ("provisional", "confirmed"):
                first_alert = float(row.time_offset_s)
        post = sub[sub.time_offset_s >= 0]
        results[ev] = {
            "n_frames": len(sub),
            "offset_range": [float(sub.time_offset_s.min()), float(sub.time_offset_s.max())],
            "first_alert_offset_s": first_alert,
            "detected": first_alert is not None,
            "delay_from_first_visible_s": (max(first_alert, 0.0)
                                           if first_alert is not None else None),
            "pre_visible_alert": bool(first_alert is not None and first_alert < 0),
            "has_post_smoke_frames": len(post) > 0,
            "per_frame": per_frame,
        }
        print(f"{ev}: detected={results[ev]['detected']} "
              f"first_alert_offset={first_alert}", flush=True)

    detected = [r for r in results.values() if r["detected"]]
    delays = sorted(r["delay_from_first_visible_s"] for r in detected)
    summary = {
        "n_events": len(results),
        "n_detected": len(detected),
        "n_missed": len(results) - len(detected),
        "n_pre_visible_alerts": sum(r["pre_visible_alert"] for r in detected),
        "median_delay_s_among_detected": delays[len(delays) // 2] if delays else None,
        "mean_delay_s_among_detected": (sum(delays) / len(delays)) if delays else None,
        "config": {"imgsz": args.imgsz, "tile_size": args.tile_size,
                   "single_frame_alert_conf": cfg.single_frame_alert_conf,
                   "seq_frame_conf": cfg.seq_frame_conf,
                   "seq_window": cfg.seq_window,
                   "seq_hits_required": cfg.seq_hits_required,
                   "weights": args.weights, "backend": args.backend,
                   "scores_jsonl": args.scores_jsonl,
                   "n_frames_without_score": n_missing_scores},
        "caveats": [
            "Delay measured from FIgLib first-visible-smoke frame (offset 0), not ignition.",
            "No bounding-box ground truth: event-level alerting only.",
            "Fixed-camera imagery; drone generalization NOT demonstrated by this eval.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "events": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
