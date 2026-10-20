"""RF-DETR training driver for green g6e pods (offline) and devbox CPU tests.

Modes:
  overfit  — 16-image (8 pos / 8 neg, deterministic) tiny-batch sanity run;
             valid split = the same 16 images so eval stays fast.
  profile  — real train split, 1 epoch, killed externally by timeout; steps/s
             parsed from console by the caller.
  train    — the real run (segmentless long job; resumable from last.ckpt in
             the /scratch output dir).

Lightning handles 8-GPU DDP itself (devices=auto).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def make_overfit_ds(src_ds: Path, dst_ds: Path, data_root: Path, n_pos=8, n_neg=8):
    # Lightning DDP re-execs this script once per rank -> only rank 0 builds
    # (8-way rmtree race hung job rfdetrlsanitya2 overfit stage); others wait.
    import os
    marker = dst_ds / ".ready"
    if os.environ.get("LOCAL_RANK", "0") != "0":
        for _ in range(600):
            if marker.exists():
                return
            time.sleep(1)
        raise RuntimeError("overfit ds not ready after 600s")
    if marker.exists():
        print(f"overfit ds already built -> {dst_ds}")
        return
    ann = json.loads((src_ds / "train" / "_annotations.coco.json").read_text())
    with_ann = {a["image_id"] for a in ann["annotations"]}
    pos = [im for im in ann["images"] if im["id"] in with_ann][:n_pos]
    neg = [im for im in ann["images"] if im["id"] not in with_ann][:n_neg]
    keep = pos + neg
    keep_ids = {im["id"] for im in keep}
    sub = {
        "info": {"description": "overfit-16 subset"},
        "licenses": [],
        "categories": ann["categories"],
        "images": keep,
        "annotations": [a for a in ann["annotations"] if a["image_id"] in keep_ids],
    }
    for split in ("train", "valid"):
        d = dst_ds / split
        d.mkdir(parents=True, exist_ok=True)
        (d / "_annotations.coco.json").write_text(json.dumps(sub))
        link = d / "data"
        if not link.exists():
            link.symlink_to(data_root)
    marker.touch()
    print(f"overfit ds: {len(pos)} pos / {len(neg)} neg -> {dst_ds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["overfit", "profile", "train"], required=True)
    ap.add_argument("--dataset-dir", required=True, help="rf-format ds root (train/valid with _annotations.coco.json + data symlink)")
    ap.add_argument("--data-root", required=True, help="image tree root the ds 'data' symlinks point to")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pretrain", default="/opt/rfhome/models/rf-detr-large-2026.pth")
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=2, help="per-device")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-interval", type=int, default=1)
    ap.add_argument("--checkpoint-interval", type=int, default=5)
    ap.add_argument("--device", default=None, help="e.g. cpu for devbox tests")
    ap.add_argument("--devices", type=int, default=8,
                    help="GPU count for Lightning DDP (TrainConfig.devices defaults to 1!)")
    ap.add_argument("--no-multi-scale", action="store_true")
    args = ap.parse_args()

    from rfdetr import RFDETRLarge

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds_dir = Path(args.dataset_dir)

    if args.mode == "overfit":
        ov = out / "overfit_ds"
        make_overfit_ds(ds_dir, ov, Path(args.data_root))
        ds_dir = ov

    model = RFDETRLarge(resolution=args.resolution, pretrain_weights=args.pretrain)
    core = model.model.model  # nn.Module
    n_total = sum(p.numel() for p in core.parameters())
    n_train = sum(p.numel() for p in core.parameters() if p.requires_grad)
    print(f"PARAMS total={n_total:,} trainable={n_train:,} resolution={args.resolution}")

    resume = out / "last.ckpt"
    kw = dict(
        dataset_dir=str(ds_dir),
        output_dir=str(out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        num_workers=args.workers,
        tensorboard=False,
        wandb=False,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping=False,
        devices=args.devices,
    )
    if args.no_multi_scale:
        kw["multi_scale"] = False
    if resume.exists():
        kw["resume"] = str(resume)
        print(f"RESUMING from {resume}")
    if args.device:
        kw["device"] = args.device
    t0 = time.time()
    model.train(**kw)
    print(f"RFDETR_TRAIN_DONE wall={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
