"""Apply class-agnostic NMS to a predictions JSONL (offline, no inference).

DETR-style heads emit a fixed set of queries with NO NMS, so several queries can
cover the same plume. Our evaluation protocol (src/eval/eval_detection.py)
matches greedily one prediction per GT box and counts every other prediction as
a false positive, which charges those duplicates as FPs -- unlike the COCO mAP
protocol used inside RF-DETR's training loop, where extra lower-scored boxes
only cost precision at the tail of the ranking.

This script produces an NMS'd copy of a predictions file so the same sweep can
be recomputed and the size of that protocol interaction measured (val only;
reported as a sensitivity check, not as the headline number).

Usage:
  python scripts/v3/nms_preds.py --in preds.jsonl --out preds_nms.jsonl --iou 0.55
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.ops import nms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.55,
                    help="same default as SmokeDetector.merge_iou")
    args = ap.parse_args()

    n_in = n_out = 0
    with Path(args.out).open("w") as fh:
        for line in Path(args.inp).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            preds = rec.get("preds", [])
            n_in += len(preds)
            if len(preds) > 1:
                b = torch.tensor([p[:4] for p in preds], dtype=torch.float32)
                s = torch.tensor([p[4] for p in preds], dtype=torch.float32)
                keep = nms(b, s, args.iou).tolist()
                rec["preds"] = [preds[i] for i in sorted(keep)]
            n_out += len(rec.get("preds", []))
            fh.write(json.dumps(rec) + "\n")
    print(f"boxes {n_in} -> {n_out} (IoU {args.iou}) -> {args.out}")


if __name__ == "__main__":
    main()
