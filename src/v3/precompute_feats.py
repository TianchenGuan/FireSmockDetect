"""C1 route step 1: run the frozen DINOv3 backbone ONCE over train+val and
cache last-layer patch feature maps to disk (bf16 .pt per image, named by the
COCO json's flat_file_name). torchrun N-way: ranks shard the image list;
resumable (existing complete files are skipped; writes are tmp+rename).

Storage at 1280x736 / patch 16 (grid 80x46 = 3680 tokens):
  7B  (C=4096): 30.1 MB/img -> 38,454 train + 7,335 val = 1.35 TB
  L   (C=1024):  7.5 MB/img -> 345 GB (optional B1-noaug control)
Augmentation is FROZEN by caching (no flip/jitter possible at head-train time)
-- documented tradeoff; --flip also caches a mirrored copy (2x storage).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from det_data import CocoDetDataset, collate  # noqa: E402
from det_modeling import ViTBackbone  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-dir", required=True)
    ap.add_argument("--coco-json", required=True, nargs="+")
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=736)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--flip", action="store_true", help="also cache mirrored views (2x storage)")
    ap.add_argument("--time-budget-s", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    bb = ViTBackbone(args.backbone_dir, frozen=True,
                     dtype=torch.bfloat16 if device.type == "cuda" else None)
    bb.to(device).eval()
    gh, gw = args.height // bb.patch_size, args.width // bb.patch_size
    if rank == 0:
        print(f"backbone C={bb.embed_dim} patch={bb.patch_size} grid {gw}x{gh} "
              f"-> {bb.embed_dim*gh*gw*2/1e6:.1f} MB/img bf16", flush=True)

    items = []
    for cj in args.coco_json:
        ds = CocoDetDataset(cj, args.image_root, size=(args.width, args.height))
        items += ds.items
    items = items[rank::world]
    ds_all = CocoDetDataset(args.coco_json[0], args.image_root,
                            size=(args.width, args.height))
    ds_all.items = [it for it in items
                    if not (out / (it["flat"] + ".pt")).exists()
                    or (args.flip and not (out / (it["flat"] + ".flip.pt")).exists())]
    print(f"[rank {rank}] {len(items)} assigned, {len(ds_all.items)} to do", flush=True)
    ld = torch.utils.data.DataLoader(ds_all, batch_size=args.bs, shuffle=False,
                                     num_workers=args.workers, collate_fn=collate)
    n, t_report = 0, time.time()
    with torch.inference_mode():
        for bi, (xs, targets) in enumerate(ld):
            x = torch.stack(list(xs)).to(device)
            if device.type == "cuda":
                x = x.to(torch.bfloat16)
            variants = [("", x)] + ([(".flip", torch.flip(x, dims=[3]))] if args.flip else [])
            base = bi * args.bs
            for suffix, xv in variants:
                fm = bb(xv).to(torch.bfloat16).cpu()  # [B,C,gh,gw]
                for j in range(fm.shape[0]):
                    it = ds_all.items[base + j]
                    p = out / (it["flat"] + suffix + ".pt")
                    torch.save(fm[j].clone(), str(p) + f".tmp{rank}")
                    os.replace(str(p) + f".tmp{rank}", p)
            n += len(xs)
            if time.time() - t_report > 60:
                r = n / (time.time() - t0)
                print(f"[rank {rank}] {n}/{len(ds_all.items)} imgs, {r:.2f} img/s", flush=True)
                t_report = time.time()
            if args.time_budget_s and time.time() - t0 > args.time_budget_s:
                print(f"[rank {rank}] budget reached after {n} imgs (resumable)", flush=True)
                return 0
    print(f"[rank {rank}] FEATS_DONE {n} new / {len(items)} assigned", flush=True)
    (out / f".done_rank{rank}_of{world}").write_text(str(n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
