"""Parity + CPU latency check for src/serve/backend_dino3.py against the
cluster predictions of the same checkpoint (models/v4/evals/<run>/*_preds.jsonl,
produced by cluster/v3/dino_infer.py on L40S, fp32, conf floor 0.02).

For N deterministic sdis_val images that have >=1 cluster prediction (plus
--n-empty images with none), the backend is run on CPU with the same conf
floor, boxes are normalized by the original W/H exactly as dino_infer.py does,
and rows are greedily matched by IoU in score order. Reports per image and
overall: max |score delta| and min / mean IoU over matched pairs, plus any
unmatched boxes on either side (expected only near the conf floor).
Also measures batch-1 latency with the same protocol as reports/v3
(includes preprocessing, excludes JPEG decode).

Usage:
  python scripts/v4/verify_dino3_backend.py \
      --weights models/v3/runs/v4B4_lora_L/seg05/best.pt \
      --preds models/v4/evals/v4B4_lora_L/sdis_val_preds.jsonl \
      --n 8 --n-empty 2 --latency-n 10 [--threads 8] [--merge-check] \
      --out reports/v4/dino3_backend_parity_B4.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.serve.backend_dino3 import Dino3Backend  # noqa: E402


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_bgr(path: str) -> np.ndarray:
    # PIL decode, like both dino_infer.py and the FastAPI app (not cv2)
    return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1]


def compare(ours_norm, cluster, match_iou=0.5):
    ours = sorted(ours_norm, key=lambda p: -p[4])
    theirs = list(cluster)
    used = set()
    pairs, unmatched_ours = [], []
    for o in ours:
        best, bi = -1.0, -1
        for j, c in enumerate(theirs):
            if j in used:
                continue
            v = iou(o[:4], c[:4])
            if v > best:
                best, bi = v, j
        if bi >= 0 and best >= match_iou:
            used.add(bi)
            pairs.append({"iou": round(best, 4), "score_ours": o[4],
                          "score_cluster": theirs[bi][4],
                          "abs_score_delta": round(abs(o[4] - theirs[bi][4]), 6),
                          "max_abs_box_delta_norm": round(
                              float(np.abs(np.array(o[:4]) - np.array(theirs[bi][:4])).max()), 6)})
        else:
            unmatched_ours.append(o)
    unmatched_theirs = [c for j, c in enumerate(theirs) if j not in used]
    return pairs, unmatched_ours, unmatched_theirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models/v3/runs/v4B4_lora_L/seg05/best.pt"))
    ap.add_argument("--preds", default=str(ROOT / "models/v4/evals/v4B4_lora_L/sdis_val_preds.jsonl"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n-empty", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf-floor", type=float, default=0.02, help="must equal the cluster run's")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--latency-n", type=int, default=0)
    ap.add_argument("--merge-check", action="store_true",
                    help="also load with lora_merge=True and compare on 2 images")
    ap.add_argument("--backbone-dir", default=None)
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    rows = [json.loads(l) for l in Path(args.preds).read_text().splitlines() if l.strip()]
    pos = [r for r in rows if r.get("preds")]
    neg = [r for r in rows if not r.get("preds") and "error" not in r]
    rng = np.random.default_rng(args.seed)
    pick = [pos[i] for i in sorted(rng.choice(len(pos), args.n, replace=False))]
    pick += [neg[i] for i in sorted(rng.choice(len(neg), args.n_empty, replace=False))]

    t0 = time.perf_counter()
    be = Dino3Backend(args.weights, device=args.device, imgsz=1280, conf_floor=args.conf_floor,
                      backbone_dir=args.backbone_dir, lora_r=args.lora_r)
    load_s = time.perf_counter() - t0
    info = be.info()
    print("backend info:", json.dumps(info), f"(load {load_s:.1f}s, torch threads "
          f"{torch.get_num_threads()})", flush=True)

    report = {"weights": args.weights, "preds": args.preds, "device": args.device,
              "torch_threads": torch.get_num_threads(), "conf_floor": args.conf_floor,
              "match_iou": args.match_iou, "backend_info": info, "load_s": round(load_s, 1),
              "images": [], "summary": {}, "latency": {}}
    all_pairs, n_unm_ours, n_unm_theirs = [], 0, 0
    unm_scores_ours, unm_scores_theirs = [], []
    for row in pick:
        img = load_bgr(row["image"])
        ih, iw = img.shape[:2]
        assert (iw, ih) == (row["w"], row["h"]), (row["image"], iw, ih, row["w"], row["h"])
        dets, timing = be.detect(img)
        ours = [[d.x1 / iw, d.y1 / ih, d.x2 / iw, d.y2 / ih, d.score] for d in dets]
        pairs, uo, ut = compare(ours, row["preds"], args.match_iou)
        all_pairs += pairs
        n_unm_ours += len(uo)
        n_unm_theirs += len(ut)
        unm_scores_ours += [round(p[4], 4) for p in uo]
        unm_scores_theirs += [round(p[4], 4) for p in ut]
        r = {"image": row["image"], "size": [iw, ih], "n_ours": len(ours),
             "n_cluster": len(row["preds"]), "n_matched": len(pairs),
             "max_abs_score_delta": max((p["abs_score_delta"] for p in pairs), default=None),
             "min_iou": min((p["iou"] for p in pairs), default=None),
             "max_abs_box_delta_norm": max((p["max_abs_box_delta_norm"] for p in pairs), default=None),
             "unmatched_ours_scores": [round(p[4], 4) for p in uo],
             "unmatched_cluster_scores": [round(p[4], 4) for p in ut],
             "top_ours": [round(p[4], 4) for p in sorted(ours, key=lambda p: -p[4])[:3]],
             "top_cluster": [round(p[4], 4) for p in sorted(row["preds"], key=lambda p: -p[4])[:3]],
             "timing": timing}
        report["images"].append(r)
        print(json.dumps(r), flush=True)

    ious = [p["iou"] for p in all_pairs]
    report["summary"] = {
        "n_images": len(pick), "n_matched_pairs": len(all_pairs),
        "max_abs_score_delta": max((p["abs_score_delta"] for p in all_pairs), default=None),
        "median_abs_score_delta": (statistics.median(p["abs_score_delta"] for p in all_pairs)
                                   if all_pairs else None),
        "min_iou": min(ious) if ious else None,
        "mean_iou": round(statistics.mean(ious), 5) if ious else None,
        "max_abs_box_delta_norm": max((p["max_abs_box_delta_norm"] for p in all_pairs), default=None),
        "n_unmatched_ours": n_unm_ours, "unmatched_ours_scores": unm_scores_ours,
        "n_unmatched_cluster": n_unm_theirs, "unmatched_cluster_scores": unm_scores_theirs,
        "images_with_identical_counts": sum(1 for r in report["images"]
                                            if r["n_ours"] == r["n_cluster"]),
    }
    print("SUMMARY", json.dumps(report["summary"]), flush=True)

    if args.latency_n:
        warm = load_bgr(pick[0]["image"])
        for _ in range(2):
            be.detect(warm)
        lat, pre, inf = [], [], []
        for row in (pick * ((args.latency_n // len(pick)) + 1))[: args.latency_n]:
            img = load_bgr(row["image"])
            t0 = time.perf_counter()
            _, tm = be.detect(img)
            lat.append((time.perf_counter() - t0) * 1000)
            pre.append(tm["preprocess_ms"])
            inf.append(tm["infer_ms"])
        report["latency"] = {
            "n": len(lat), "device": args.device, "torch_threads": torch.get_num_threads(),
            "letterbox": info["letterbox"], "lora_merged": False,
            "batch1_ms_mean": round(statistics.mean(lat), 1),
            "batch1_ms_median": round(statistics.median(lat), 1),
            "batch1_ms_p90": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 1),
            "batch1_ms_min": round(min(lat), 1), "batch1_ms_max": round(max(lat), 1),
            "preprocess_ms_median": round(statistics.median(pre), 1),
            "infer_ms_median": round(statistics.median(inf), 1),
        }
        print("LATENCY", json.dumps(report["latency"]), flush=True)

    if args.merge_check:
        bm = Dino3Backend(args.weights, device=args.device, imgsz=1280,
                          conf_floor=args.conf_floor, backbone_dir=args.backbone_dir,
                          lora_merge=True)
        mc = {"merged_linears": bm.info()["lora_merged_linears"],
              "lora_params_after_merge": bm.info()["lora_params"], "images": []}
        for row in pick[:2]:
            img = load_bgr(row["image"])
            d0, _ = be.detect(img)
            d1, tm = bm.detect(img)
            d0 = sorted(d0, key=lambda d: -d.score)
            d1 = sorted(d1, key=lambda d: -d.score)
            n = min(len(d0), len(d1))
            mc["images"].append({
                "image": row["image"], "n_unmerged": len(d0), "n_merged": len(d1),
                "max_abs_score_delta": max((abs(d0[i].score - d1[i].score) for i in range(n)), default=None),
                "max_abs_box_delta_px": max((max(abs(d0[i].x1 - d1[i].x1), abs(d0[i].y1 - d1[i].y1),
                                                 abs(d0[i].x2 - d1[i].x2), abs(d0[i].y2 - d1[i].y2))
                                             for i in range(n)), default=None),
                "merged_latency_ms": tm["latency_ms"]})
        report["merge_check"] = mc
        print("MERGE_CHECK", json.dumps(mc), flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print("saved:", args.out)


if __name__ == "__main__":
    main()
