"""Fine-tune a COCO-pretrained YOLO11 on smoke data (ultralytics).

Designed to run identically on devbox (CPU smoke-test) and cluster B200 pod.
The pod has NO internet: --model must point at a local .pt file (pretrained
weights are staged alongside the dataset), and dataset yaml must use absolute
local paths.

Example (cluster):
  python train_yolo.py --data /mnt/local/data/pyro_sdis.yaml \
    --model /mnt/local/weights/yolo11s.pt --epochs 60 --imgsz 1152 \
    --batch -1 --device 0 --project /scratch/scratch/tianrgua-sandbox/smoke/runs \
    --name expA_sdis_y11s_1152

Resume: same command + --resume (ultralytics picks up last.pt in the run dir).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="local .pt (COCO-pretrained or checkpoint)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1152)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto (60%% VRAM)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cos-lr", action="store_true", default=True)
    ap.add_argument("--fraction", type=float, default=1.0, help="dataset fraction (smoke tests)")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "argv": sys.argv,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "host": platform.node(),
        "pid": os.getpid(),
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2), flush=True)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        cos_lr=args.cos_lr,
        amp=True,                    # mixed precision
        fraction=args.fraction,
        # small/faint smoke: keep gentle photometric aug, avoid aggressive
        # spatial aug that can erase tiny plumes while keeping the label.
        mosaic=0.5,
        close_mosaic=10,             # disable mosaic for last N epochs
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        scale=0.3,
        translate=0.05,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.4,
        erasing=0.0,                 # random erasing can delete a faint plume
        val=True,
        plots=True,
        save_period=5,               # periodic checkpoints for resume safety
    )
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
