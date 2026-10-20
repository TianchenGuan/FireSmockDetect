"""V6 arm-E trainer: train_yolo.py-compatible CLI + KD via KDTrainer.

Identical hyperparameters to src/train/train_yolo.py (the matched recipe);
adds --teacher/--kd-* which are exported as env vars so ultralytics DDP
subprocesses inherit them. The file's directory is prepended to PYTHONPATH
so DDP can `from train_kd_lib import KDTrainer`.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["PYTHONPATH"] = f"{HERE}:{os.environ.get('PYTHONPATH', '')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--kd-lambda", type=float, default=1.0)
    ap.add_argument("--kd-cls-w", type=float, default=4.0)
    ap.add_argument("--kd-box-w", type=float, default=1.0)
    ap.add_argument("--kd-tau", type=float, default=2.0)
    ap.add_argument("--kd-fg-t", type=float, default=0.25)
    args = ap.parse_args()

    os.environ["KD_TEACHER"] = str(Path(args.teacher).resolve())
    os.environ["KD_LAMBDA"] = str(args.kd_lambda)
    os.environ["KD_CLS_W"] = str(args.kd_cls_w)
    os.environ["KD_BOX_W"] = str(args.kd_box_w)
    os.environ["KD_TAU"] = str(args.kd_tau)
    os.environ["KD_FG_T"] = str(args.kd_fg_t)

    import torch
    from ultralytics import YOLO
    from train_kd_lib import KDTrainer

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {"argv": sys.argv, "torch": torch.__version__,
           "cuda_available": torch.cuda.is_available(),
           "teacher": os.environ["KD_TEACHER"], "host": platform.node()}
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2), flush=True)

    model = YOLO(args.model)
    model.train(
        trainer=KDTrainer,
        data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, project=args.project,
        name=args.name, exist_ok=True, resume=args.resume,
        workers=args.workers, patience=args.patience, seed=args.seed,
        cos_lr=True, amp=True, fraction=args.fraction,
        mosaic=0.5, close_mosaic=10, mixup=0.0, copy_paste=0.0,
        degrees=0.0, shear=0.0, perspective=0.0, scale=0.3, translate=0.05,
        fliplr=0.5, flipud=0.0, hsv_h=0.01, hsv_s=0.4, hsv_v=0.4,
        erasing=0.0, val=True, plots=True, save_period=5,
    )
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
