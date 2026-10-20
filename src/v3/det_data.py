"""COCO-json detection dataset for Track B/C (consumes Track A's conversion at
data/v3/coco/instances_{train,val}.json; file_name is relative to the yolo
data root, e.g. 'pyro_sdis/images/train/x.jpg').

Letterbox: scale to fit (W, H) preserving aspect ratio, pad bottom/right with
114-gray. Both dims of (W, H) must be multiples of the backbone patch size
(dinov3-*: 16 -> default 1280x736, grid 80x46; dinov2-large stand-in: 14 ->
1288x728, grid 92x52).

Augmentation (train): horizontal flip 0.5 + mild color jitter. Deliberately
minimal for the plumbing phase; documented in reports/v3/trackBC_prep.md.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance

try:
    from .det_modeling import IMAGENET_MEAN, IMAGENET_STD
except ImportError:  # flat per-job layout on pods (files fetched into one dir)
    from det_modeling import IMAGENET_MEAN, IMAGENET_STD

MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def letterbox_params(iw, ih, tw, th):
    r = min(tw / iw, th / ih)
    nw, nh = round(iw * r), round(ih * r)
    return r, nw, nh


class CocoDetDataset(torch.utils.data.Dataset):
    def __init__(self, coco_json: str, image_root: str, size=(1280, 736),
                 train: bool = False, subset_ids=None, max_images: int = 0,
                 seed: int = 0):
        d = json.loads(Path(coco_json).read_text())
        self.root = Path(image_root)
        self.tw, self.th = size
        self.train = train
        anns_by_img = {}
        for a in d["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        self.items = []
        for im in d["images"]:
            self.items.append(dict(
                id=im["id"], file_name=im["file_name"],
                flat=im.get("flat_file_name", im["file_name"].replace("/", "__")),
                w=im["width"], h=im["height"],
                boxes=[a["bbox"] for a in anns_by_img.get(im["id"], [])]))
        if subset_ids is not None:
            keep = set(subset_ids)
            self.items = [it for it in self.items if it["id"] in keep]
        if max_images and len(self.items) > max_images:
            rng = random.Random(seed)
            self.items = rng.sample(self.items, max_images)

    def __len__(self):
        return len(self.items)

    def load_image(self, it):
        img = Image.open(self.root / it["file_name"]).convert("RGB")
        return img

    def __getitem__(self, idx):
        it = self.items[idx]
        img = self.load_image(it)
        iw, ih = img.size
        r, nw, nh = letterbox_params(iw, ih, self.tw, self.th)
        img = img.resize((nw, nh), Image.BILINEAR)
        boxes = []
        for x, y, w, h in it["boxes"]:  # coco xywh absolute
            x1, y1 = x * r, y * r
            x2, y2 = (x + w) * r, (y + h) * r
            if x2 - x1 >= 1 and y2 - y1 >= 1:
                boxes.append([x1, y1, x2, y2])
        flip = self.train and random.random() < 0.5
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [[nw - x2, y1, nw - x1, y2] for x1, y1, x2, y2 in boxes]
        if self.train:
            for enh, lo, hi in ((ImageEnhance.Brightness, 0.85, 1.15),
                                (ImageEnhance.Contrast, 0.85, 1.15),
                                (ImageEnhance.Color, 0.85, 1.15)):
                if random.random() < 0.5:
                    img = enh(img).enhance(random.uniform(lo, hi))
        canvas = Image.new("RGB", (self.tw, self.th), (114, 114, 114))
        canvas.paste(img, (0, 0))
        x = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255.0
        x = (x - MEAN) / STD
        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": boxes_t,
            "labels": torch.ones((boxes_t.shape[0],), dtype=torch.int64),
            "image_id": torch.tensor([it["id"]]),
            # letterbox meta so eval can map detections back to original pixels
            "scale": torch.tensor([r]),
        }
        return x, target


class CachedFeatureDataset(CocoDetDataset):
    """C1 route: returns precomputed backbone feature maps instead of pixels.
    feat_root/<flat_file_name>.pt = bf16 tensor [C, gh, gw] computed at
    (tw, th) WITHOUT augmentation (cached features freeze augmentation --
    tradeoff documented in the prep report). train=True only shuffles."""

    def __init__(self, coco_json, image_root, feat_root, size=(1280, 736), **kw):
        super().__init__(coco_json, image_root, size=size, train=False, **kw)
        self.feat_root = Path(feat_root)

    def __getitem__(self, idx):
        it = self.items[idx]
        fmap = torch.load(self.feat_root / (it["flat"] + ".pt"),
                          map_location="cpu", weights_only=True).float()
        r, nw, nh = letterbox_params(it["w"], it["h"], self.tw, self.th)
        boxes = []
        for x, y, w, h in it["boxes"]:
            x1, y1, x2, y2 = x * r, y * r, (x + w) * r, (y + h) * r
            if x2 - x1 >= 1 and y2 - y1 >= 1:
                boxes.append([x1, y1, x2, y2])
        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {"boxes": boxes_t,
                  "labels": torch.ones((boxes_t.shape[0],), dtype=torch.int64),
                  "image_id": torch.tensor([it["id"]]),
                  "scale": torch.tensor([r])}
        return fmap, target


def collate(batch):
    return list(zip(*batch))
