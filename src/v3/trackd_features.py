"""Track D: precompute frozen per-frame features from expE YOLO11m for
temporal-head training. One npz per sequence.

Features per frame (spec frozen in reports/v3/trackD_notes.md §3):
- neck maps hooked at layers 16/19/22 (Detect inputs), each spatially
  avg- and max-pooled to (12,20), channels group-mean-reduced to 32
  -> 64 ch/scale, 192 ch;
- 2 detection channels rasterized to (12,20): max-conf paint + count/4;
- scalar per frame: max det conf at floor 0.01 (single-frame baseline score).

Resumable: sequences whose output npz already exists (with matching T) are
skipped. Shardable: --shard-idx/--shard-n split sequences round-robin.

Runs on pod GPU (green g6e) or devbox CPU identically.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from ultralytics import YOLO

POOL_HW = (12, 20)
HOOK_LAYERS = [16, 19, 22]
GROUPS_OUT = 32
CONF_FLOOR = 0.01
N_CH = 3 * 2 * GROUPS_OUT + 2  # 194


def reduce_map(x: torch.Tensor) -> torch.Tensor:
    """(1,C,H,W) -> (2*GROUPS_OUT, 12, 20): spatial avg+max pool, then
    channel group-mean to GROUPS_OUT."""
    xa = F.adaptive_avg_pool2d(x, POOL_HW)
    xm = F.adaptive_max_pool2d(x, POOL_HW)
    out = []
    for t in (xa, xm):
        c = t.shape[1]
        assert c % GROUPS_OUT == 0, c
        out.append(t.view(1, GROUPS_OUT, c // GROUPS_OUT, *POOL_HW).mean(2)[0])
    return torch.cat(out, dim=0)  # (64,12,20)


def det_channels(boxes_xyxyn: np.ndarray, confs: np.ndarray) -> np.ndarray:
    """Rasterize detections onto the pooled grid. (2,12,20) float32."""
    H, W = POOL_HW
    conf_map = np.zeros((H, W), dtype=np.float32)
    cnt_map = np.zeros((H, W), dtype=np.float32)
    for (x1, y1, x2, y2), c in zip(boxes_xyxyn, confs):
        c0, c1 = int(np.floor(x1 * W)), max(int(np.ceil(x2 * W)), int(np.floor(x1 * W)) + 1)
        r0, r1 = int(np.floor(y1 * H)), max(int(np.ceil(y2 * H)), int(np.floor(y1 * H)) + 1)
        c0 = int(np.clip(c0, 0, W - 1)); c1 = int(np.clip(c1, c0 + 1, W))
        r0 = int(np.clip(r0, 0, H - 1)); r1 = int(np.clip(r1, r0 + 1, H))
        region = conf_map[r0:r1, c0:c1]
        np.maximum(region, c, out=region)
        cnt_map[r0:r1, c0:c1] += 1.0
    return np.stack([conf_map, np.clip(cnt_map / 4.0, 0, 1)])


class Extractor:
    def __init__(self, weights: str, device: str, imgsz: int):
        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.feats: dict[int, torch.Tensor] = {}
        mm = self.model.model.model
        for i in HOOK_LAYERS:
            mm[i].register_forward_hook(self._mk(i))

    def _mk(self, i):
        def hook(mod, inp, out):
            self.feats[i] = out.detach()
        return hook

    def frame(self, img: np.ndarray):
        self.feats.clear()
        r = self.model.predict(img, imgsz=self.imgsz, conf=CONF_FLOOR,
                               device=self.device, verbose=False)[0]
        maps = [reduce_map(self.feats[i].float()) for i in HOOK_LAYERS]
        if len(r.boxes):
            bxn = r.boxes.xyxyn.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
        else:
            bxn = np.zeros((0, 4), dtype=np.float32)
            cf = np.zeros((0,), dtype=np.float32)
        dch = torch.from_numpy(det_channels(bxn, cf)).to(maps[0].device)
        feat = torch.cat(maps + [dch], dim=0).cpu().numpy().astype(np.float16)
        assert feat.shape == (N_CH, *POOL_HW), feat.shape
        score = float(cf.max()) if len(cf) else 0.0
        dets = np.concatenate([bxn, cf[:, None]], axis=1).astype(np.float32) \
            if len(cf) else np.zeros((0, 5), dtype=np.float32)
        return feat, score, dets


def map_path(p: str, root_map: str | None) -> str:
    if root_map:
        src, dst = root_map.split("=", 1)
        if p.startswith(src):
            return dst + p[len(src):]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sequences", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output dir; per-split subdirs created")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--image-root-map", default=None)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--shard-n", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="max sequences (smoke test)")
    args = ap.parse_args()

    ex = Extractor(args.weights, args.device, args.imgsz)
    outroot = Path(args.out)
    t0, n_done, n_skip, n_frames = time.time(), 0, 0, 0
    for seq_file in args.sequences:
        seqs = json.loads(Path(seq_file).read_text())
        seqs = [s for i, s in enumerate(seqs) if i % args.shard_n == args.shard_idx]
        if args.limit:
            seqs = seqs[: args.limit]
        for s in seqs:
            od = outroot / s["split"]
            od.mkdir(parents=True, exist_ok=True)
            fp = od / f"{s['seq_id']}.npz"
            if fp.exists():
                try:
                    z = np.load(fp, allow_pickle=False)
                    if z["feat"].shape[0] == s["n_frames"]:
                        n_skip += 1
                        continue
                except Exception:
                    pass
            feats, scores, det_rows = [], [], []
            ok = True
            for fr in s["frames"]:
                img = cv2.imread(map_path(fr["image_path"], args.image_root_map))
                if img is None:
                    print(f"UNREADABLE {fr['image_path']}", flush=True)
                    ok = False
                    break
                f, sc, dets = ex.frame(img)
                feats.append(f)
                scores.append(sc)
                det_rows.append(dets.tolist())
            if not ok:
                continue
            gt_area = [max((b[2] * b[3] for b in fr["boxes_xywhn"]), default=0.0)
                       for fr in s["frames"]]
            tmp = fp.with_name(fp.stem + ".tmpnpz")  # must not match *.npz globs
            fh = open(tmp, "wb")  # file handle: numpy won't append ".npz"
            np.savez_compressed(
                fh,
                feat=np.stack(feats),
                score=np.array(scores, dtype=np.float32),
                ts=np.array([fr["ts"] for fr in s["frames"]], dtype=np.int64),
                smoke=np.array([fr["smoke"] for fr in s["frames"]], dtype=np.uint8),
                max_gt_area=np.array(gt_area, dtype=np.float32),
                names=np.array([fr["image_name"] for fr in s["frames"]]),
            )
            fh.close()
            tmp.rename(fp)
            (od / f"{s['seq_id']}.dets.json").write_text(json.dumps(det_rows))
            n_done += 1
            n_frames += s["n_frames"]
            if n_done % 20 == 0:
                dt = time.time() - t0
                print(f"[shard {args.shard_idx}/{args.shard_n}] {n_done} seqs "
                      f"{n_frames} frames {dt:.0f}s ({n_frames/max(dt,1e-9):.1f} f/s)",
                      flush=True)
    print(f"DONE shard {args.shard_idx}/{args.shard_n}: {n_done} extracted, "
          f"{n_skip} skipped, {n_frames} frames, {time.time()-t0:.0f}s", flush=True)
    print("EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
