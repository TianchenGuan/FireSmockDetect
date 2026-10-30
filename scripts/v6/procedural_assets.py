"""Arm-B asset sources: simple procedural + copy-paste smoke (NO generative model).

Two kinds, both cheap and fully deterministic given a seed:
  1. copy-paste crops: smoke boxes cut from REAL TRAIN positives, pasted with a
     feathered soft matte (classic copy-paste augmentation). The matte is a
     construction artifact, not a segmentation estimate.
  2. parametric plumes: perlin-ish buoyant-plume alpha on black (luma mode).

Arm B exists as the control that separates "any pasted smoke-like pixels help"
from "generative realism helps" (arm C/D).
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def copy_paste_asset(img_path: str, box_norm, out_path: Path,
                     rng: random.Random) -> dict | None:
    """Cut one labelled smoke box from a real train positive -> RGBA asset.

    Feathered rounded-rect matte; straight alpha. Returns provenance or None
    if the crop is degenerate (<24 px side).
    """
    im = np.asarray(Image.open(img_path).convert("RGB"))
    H, W = im.shape[:2]
    cx, cy, w, h = box_norm
    x0 = int((cx - w / 2) * W); x1 = int((cx + w / 2) * W)
    y0 = int((cy - h / 2) * H); y1 = int((cy + h / 2) * H)
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    crop = im[y0:y1, x0:x1]
    # reject crops containing visible flame (orange/red saturated pixels):
    # a flame blob labeled "smoke" teaches the wrong cue
    f = crop.astype(np.float32) / 255
    flame = (f[..., 0] > 0.55) & (f[..., 0] - f[..., 2] > 0.22) & \
            ((f.max(-1) - f.min(-1)) > 0.25)
    if flame.mean() > 0.004:
        return None
    ch, cw = crop.shape[:2]
    # feathered rounded-rect alpha
    a = np.zeros((ch, cw), np.float32)
    mx, my = int(cw * 0.08) + 1, int(ch * 0.08) + 1
    a[my:ch - my, mx:cw - mx] = 1.0
    sigma = max(2.0, 0.06 * max(ch, cw))
    a = cv2.GaussianBlur(a, (0, 0), sigma)
    a = np.clip(a / max(a.max(), 1e-6), 0, 1) * rng.uniform(0.75, 1.0)
    rgba = np.dstack([crop.astype(np.float32) / 255.0, a])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(out_path)
    return {"kind": "copy_paste", "src_image": img_path,
            "src_box_norm": list(box_norm), "asset": str(out_path),
            "mode": "straight"}


def parametric_plume(out_path: Path, seed: int, wind: float | None = None,
                     density: float | None = None) -> dict:
    """Buoyant plume alpha field on black (luma-mode asset)."""
    rng = np.random.default_rng(seed)
    pr = random.Random(seed)
    h, w = pr.randint(180, 420), pr.randint(120, 300)
    wind = pr.uniform(-0.5, 0.5) if wind is None else wind
    density = pr.uniform(0.35, 0.8) if density is None else density
    # multi-octave noise
    acc = np.zeros((h, w), np.float32)
    for oct_, amp in ((10, 1.0), (24, 0.5), (56, 0.12)):
        g = rng.random((max(2, h // oct_), max(2, w // oct_))).astype(np.float32)
        up = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        # smooth each octave by ~1/3 of its cell size: kills the value-noise
        # grid "checkerboard" that read as fabric texture in v1 plumes
        up = cv2.GaussianBlur(up, (0, 0), max(1.0, oct_ / 3.0))
        acc += up * amp
    acc /= 1.62
    # vertical buoyancy envelope: dense near source (bottom-center), spreading up
    yy = np.linspace(1.0, 0.0, h)[:, None]
    xx = np.linspace(-1, 1, w)[None, :]
    spread = 0.18 + (1 - yy) * 0.75
    center = wind * (1 - yy)
    env = np.exp(-((xx - center) ** 2) / (2 * spread ** 2)) * (yy ** 0.35)
    alpha = np.clip((acc - 0.35) * 2.2, 0, 1) * env * density
    # border taper: the silhouette must vanish inside the canvas, never be
    # cut by its rectangle (v1 produced square "patch" plumes at high density)
    bx = np.minimum(np.arange(w), np.arange(w)[::-1])[None, :] / (0.12 * w)
    by = np.minimum(np.arange(h), np.arange(h)[::-1])[None, :].T / (0.12 * h)
    alpha = alpha * np.clip(bx, 0, 1) * np.clip(by, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5)
    tone = pr.uniform(0.75, 0.95)
    rgb = np.stack([alpha * tone, alpha * tone, alpha * min(1, tone * 1.04)], -1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(out_path)
    return {"kind": "parametric", "seed": seed, "wind": wind,
            "density": density, "asset": str(out_path), "mode": "luma"}
