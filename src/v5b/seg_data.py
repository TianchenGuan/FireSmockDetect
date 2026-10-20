"""Dataset for the V5b segmentation branch. Rows come from
data/v5b/manifests/seg_v5b_manifest.json (built by scripts/v5b/build_seg_manifest.py).
Letterbox to (W,H) exactly like src/v3/det_data (same mean/std, gray pad), mask
resized NEAREST; `valid` marks non-pad pixels; `has_mask` is True for every
training row by construction (positives: SAM/manual PNG; author-negatives:
all-zero mask) and False only if a row without a mask is ever added
(loss is masked out there)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance

try:
    from src.v3.det_modeling import IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    from det_modeling import IMAGENET_MEAN, IMAGENET_STD  # type: ignore

MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def letterbox_params(iw, ih, tw, th):
    r = min(tw / iw, th / ih)
    return r, round(iw * r), round(ih * r)


def load_rows(manifest, role=None):
    rows = json.loads(Path(manifest).read_text())
    if role:
        rows = [r for r in rows if r["role"] == role]
    return rows


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, data_root, role, width=1280, height=736, train=False,
                 max_items=0, seed=0, scale_jitter=(0.75, 1.0)):
        self.rows = load_rows(manifest, role)
        if max_items and len(self.rows) > max_items:
            self.rows = random.Random(seed).sample(self.rows, max_items)
        self.root, self.tw, self.th, self.train = Path(data_root), width, height, train
        self.scale_jitter = scale_jitter

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.root / r["image"]).convert("RGB")
        iw, ih = img.size
        s, nw, nh = letterbox_params(iw, ih, self.tw, self.th)
        ox = oy = 0
        if self.train and self.scale_jitter:
            j = random.uniform(*self.scale_jitter)
            s, nw, nh = s * j, max(16, round(nw * j)), max(16, round(nh * j))
            ox, oy = random.randint(0, self.tw - nw), random.randint(0, self.th - nh)
        img = img.resize((nw, nh), Image.BILINEAR)
        if r.get("mask"):
            m = Image.open(self.root / r["mask"]).convert("L").resize((nw, nh), Image.NEAREST)
            m = (np.asarray(m) > 127).astype(np.uint8)
        else:
            m = np.zeros((nh, nw), np.uint8)
        flip = self.train and random.random() < 0.5
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            m = m[:, ::-1]
        if self.train:
            for enh, lo, hi in ((ImageEnhance.Brightness, 0.8, 1.2),
                                (ImageEnhance.Contrast, 0.8, 1.2),
                                (ImageEnhance.Color, 0.8, 1.2)):
                if random.random() < 0.5:
                    img = enh(img).enhance(random.uniform(lo, hi))
        canvas = Image.new("RGB", (self.tw, self.th), (114, 114, 114))
        canvas.paste(img, (ox, oy))
        x = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255.0
        x = (x - MEAN) / STD
        tgt = torch.zeros(1, self.th, self.tw)
        tgt[0, oy:oy + nh, ox:ox + nw] = torch.from_numpy(np.ascontiguousarray(m)).float()
        valid = torch.zeros(1, self.th, self.tw)
        valid[0, oy:oy + nh, ox:ox + nw] = 1.0
        has_mask = bool(r.get("has_mask", True)) or bool(r.get("is_negative", False))
        boxes = r.get("boxes_xyxy_norm") or []
        if flip:
            boxes = [[1 - b[2], b[1], 1 - b[0], b[3]] for b in boxes]
        meta = {"id": r["id"], "burn": r["burn"], "flight": r["flight"],
                "provenance": r["provenance"], "is_negative": bool(r.get("is_negative", False)),
                "scale": s, "nw": nw, "nh": nh, "ox": ox, "oy": oy, "orig_w": iw, "orig_h": ih,
                "boxes_xyxy_norm": boxes, "has_mask": has_mask}
        return x, tgt, valid, has_mask, meta


def collate(batch):
    xs, ts, vs, hs, ms = zip(*batch)
    return (torch.stack(xs), torch.stack(ts), torch.stack(vs),
            torch.tensor(hs, dtype=torch.bool), list(ms))


def preprocess_image(img: Image.Image, width=1280, height=736):
    """Inference-time letterbox (no aug). Returns x[1,3,H,W] and meta for cropping back."""
    iw, ih = img.size
    s, nw, nh = letterbox_params(iw, ih, width, height)
    canvas = Image.new("RGB", (width, height), (114, 114, 114))
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), (0, 0))
    x = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255.0
    x = ((x - MEAN) / STD).unsqueeze(0)
    return x, {"scale": s, "nw": nw, "nh": nh, "orig_w": iw, "orig_h": ih}
