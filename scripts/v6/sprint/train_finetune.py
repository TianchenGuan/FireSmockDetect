"""Sprint fine-tune trainer (shipped to pod via data tar, TRAIN_SCRIPT env).

Same CLI as the baked /opt/train_yolo.py plus --lr0/--warmup for low-LR
fine-tuning from an existing checkpoint. Prints TRAIN_DONE on completion
(the entry script greps for it).
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="ft")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--warmup", type=float, default=1.0)
    args = ap.parse_args()

    from ultralytics import YOLO
    m = YOLO(args.model)
    m.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, device=args.device, project=args.project,
            name=args.name, workers=args.workers, seed=args.seed,
            resume=args.resume, lr0=args.lr0, warmup_epochs=args.warmup,
            exist_ok=True)
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
