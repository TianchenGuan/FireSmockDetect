"""V6 Route-1 compositor: RGBA smoke assets onto real TRAIN backgrounds.

Produces, per composite: image (jpg), YOLO label (plume-level box[es]),
soft mask (png, construction metadata — NOT human segmentation GT),
and a JSON record with full provenance + transform params. A matched
no-smoke CONTROL of the same background goes through the identical
resize/degrade/JPEG pipeline (same sampled params) so classes cannot be
separated by processing artifacts.

Alpha conventions:
  mode="straight": asset is RGBA with straight (non-premultiplied) alpha.
  mode="luma":     asset is RGB on near-black; alpha := k*luminance (additive
                   scattering approximation), color := asset RGB (premult-like,
                   composited as C = bg*(1-a) + rgb).
Visibility is checked AFTER resizing to the student input size (default 1280
long side): plumes whose support or contrast vanish at train resolution are
labelled "invisible" -> caller drops or keeps as ignore-region, never as a
positive box.
"""
from __future__ import annotations

import io
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# support threshold on effective alpha for mask/box derivation
SUPPORT_T = 0.03
# a plume must keep >= this many support px at student resolution
MIN_SUPPORT_PX = 25
# and this much mean absolute luminance delta (0..255) over its support
MIN_CONTRAST = 1.5


@dataclass
class Placement:
    x_frac: float          # center x as fraction of bg width
    y_frac: float          # y position (meaning depends on y_mode)
    scale: float           # asset width as fraction of bg width
    opacity: float         # global alpha multiplier (faintness control)
    flip: bool
    rot_deg: float
    tint_strength: float   # 0..1 pull of smoke color toward bg ambient
    occlude_frac: float    # 0..0.5 bottom fraction erased by noise (canopy sim)
    y_mode: str = "center"  # "center" | "bottom" (bottom = plume base anchor)


def estimate_horizon(bg: np.ndarray) -> float:
    """Horizon row as a fraction of height: the split maximizing the
    texture-energy contrast between smooth sky (above) and terrain (below)."""
    g = cv2.cvtColor((bg * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (192, 144), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    grad = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) + \
        np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    smooth = grad < 45
    # top-down flood through smooth pixels with 1px lateral dilation per row:
    # sky is smooth and connected to the top; per-column skyline = flood depth.
    H2, W2 = smooth.shape
    sky_prev = smooth[0]
    depth = np.where(sky_prev, 1, 0).astype(np.int32)
    for r in range(1, H2):
        lat = sky_prev | np.roll(sky_prev, 1) | np.roll(sky_prev, -1)
        sky_prev = smooth[r] & lat
        depth += sky_prev.astype(np.int32)
    # columns blocked early by clouds/towers sit low; take the 70th percentile
    # of flood depths as the terrain skyline.
    return float(np.clip(np.percentile(depth, 70) / H2, 0.15, 0.9))


@dataclass
class Degrade:
    blur_sigma: float
    noise_sigma: float
    jpeg_q: int
    target_long: int       # student-facing long side for visibility check


def load_asset(path: str, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb float 0..1 straight color, alpha float 0..1)."""
    im = Image.open(path).convert("RGBA")
    arr = np.asarray(im).astype(np.float32) / 255.0
    rgb, a = arr[..., :3], arr[..., 3]
    if mode == "luma":
        # RGB-on-black: alpha from luminance; rgb kept as premultiplied light.
        luma = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        a = np.clip(luma * 1.6, 0.0, 1.0)
        return rgb, a            # composited additively: bg*(1-a) + rgb*est
    if mode == "straight":
        return rgb, a
    raise ValueError(mode)


def _perlinish(h, w, scale=48, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def place_asset(bg: np.ndarray, rgb: np.ndarray, a: np.ndarray, p: Placement,
                mode: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite one asset. Returns (out_bgr_float, effective_alpha_full)."""
    H, W = bg.shape[:2]
    ah, aw = a.shape
    new_w = max(8, int(p.scale * W))
    new_h = max(8, int(new_w * ah / aw))
    rgbr = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ar = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if p.flip:
        rgbr, ar = rgbr[:, ::-1], ar[:, ::-1]
    if abs(p.rot_deg) > 0.5:
        M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), p.rot_deg, 1.0)
        rgbr = cv2.warpAffine(rgbr, M, (new_w, new_h))
        ar = cv2.warpAffine(ar, M, (new_w, new_h))
    if p.occlude_frac > 0.01:
        occ = _perlinish(new_h, new_w, scale=24, seed=seed) > 0.45
        band = np.zeros((new_h, new_w), bool)
        band[int(new_h * (1 - p.occlude_frac)):, :] = True
        ar = ar * (1 - (occ & band).astype(np.float32) * 0.9)

    ar = ar * p.opacity
    cx = int(p.x_frac * W)
    if p.y_mode == "bottom":
        cy = int(p.y_frac * H) - new_h // 2    # y_frac marks the plume BASE
    else:
        cy = int(p.y_frac * H)
    x0, y0 = cx - new_w // 2, cy - new_h // 2
    # clip to frame
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + new_w), min(H, y0 + new_h)
    if dx1 <= dx0 or dy1 <= dy0:
        return bg, np.zeros((H, W), np.float32)
    rr = rgbr[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    aa = ar[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]

    # illumination/tint match: pull smoke color toward bg ambient
    patch = bg[dy0:dy1, dx0:dx1]
    amb = patch.reshape(-1, 3).mean(0)
    bright = np.percentile(patch.reshape(-1, 3).max(1), 90) + 1e-3
    out = bg.copy()
    if mode == "luma":
        # additive scattering: C = bg*(1-a) + S, S scaled by scene brightness
        # and tinted toward ambient so smoke inherits the scene illumination.
        rr = rr * np.clip(bright, 0.25, 1.0) * p.opacity
        tint = amb[None, None, :] / max(float(amb.mean()), 1e-3)
        rr = rr * ((1 - p.tint_strength) + p.tint_strength * tint)
        out[dy0:dy1, dx0:dx1] = np.clip(
            patch * (1 - aa[..., None]) + np.clip(rr, 0, 1), 0, 1)
    else:
        col = rr * (1 - p.tint_strength) + amb[None, None, :] * p.tint_strength
        col = col * np.clip(bright / max(col.mean(), 1e-3), 0.6, 1.4) ** 0.5
        out[dy0:dy1, dx0:dx1] = patch * (1 - aa[..., None]) + np.clip(col, 0, 1) * aa[..., None]

    full_a = np.zeros((H, W), np.float32)
    full_a[dy0:dy1, dx0:dx1] = aa
    return out, full_a


def degrade_pair(img: np.ndarray, d: Degrade, rng: random.Random) -> np.ndarray:
    """Shared degradation for composite AND its matched control (same d)."""
    out = img
    if d.blur_sigma > 0.05:
        out = cv2.GaussianBlur(out, (0, 0), d.blur_sigma)
    if d.noise_sigma > 1e-4:
        out = np.clip(out + np.random.default_rng(rng.randrange(1 << 30))
                      .normal(0, d.noise_sigma, out.shape).astype(np.float32), 0, 1)
    return out


def jpeg_roundtrip(img: np.ndarray, q: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray((img * 255).astype(np.uint8)).save(buf, "JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue()))).astype(np.float32) / 255.0


def boxes_from_alpha(alpha: np.ndarray, orig_hw, target_long: int):
    """Plume-level boxes + visibility stats at student resolution.

    Returns (boxes_norm, stats). boxes are merged connected components of
    support (dilated to bridge small gaps) — one box per plume, not per
    translucent fragment.
    """
    H, W = orig_hw
    s = target_long / max(H, W)
    th, tw = max(1, int(round(H * s))), max(1, int(round(W * s)))
    a_s = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_AREA)
    sup = (a_s > SUPPORT_T).astype(np.uint8)
    if sup.sum() == 0:
        return [], {"support_px": 0, "visible": False}
    k = max(3, int(0.012 * max(th, tw)) | 1)
    sup_d = cv2.dilate(sup, np.ones((k, k), np.uint8))
    n, lab = cv2.connectedComponents(sup_d)
    comps = []
    for i in range(1, n):
        comp = (lab == i) & (sup > 0)
        npx = int(comp.sum())
        if npx >= MIN_SUPPORT_PX:
            ys, xs = np.nonzero(comp)
            comps.append((npx, xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    if not comps:
        return [], {"support_px": 0, "n_plumes": 0, "visible": False}
    # One asset == one physical plume: wisps separated by simulated occlusion
    # are the SAME plume, so emit a single union box over the significant
    # components (drop sub-5%-of-max fragments entirely — translucent debris).
    biggest = max(c[0] for c in comps)
    keep = [c for c in comps if c[0] >= 0.05 * biggest]
    total_sup = sum(c[0] for c in keep)
    x0 = min(c[1] for c in keep); y0 = min(c[2] for c in keep)
    x1 = max(c[3] for c in keep); y1 = max(c[4] for c in keep)
    boxes = [[x0 / tw, y0 / th, x1 / tw, y1 / th, total_sup]]
    return boxes, {"support_px": int(total_sup), "n_plumes": 1,
                   "n_fragments": len(keep), "visible": True}


def contrast_check(comp: np.ndarray, ctrl: np.ndarray, alpha: np.ndarray,
                   target_long: int) -> float:
    """Mean |ΔL| (0..255) over support at student resolution, comp vs control."""
    H, W = comp.shape[:2]
    s = target_long / max(H, W)
    tw, th = max(1, int(round(W * s))), max(1, int(round(H * s)))
    c1 = cv2.resize(comp, (tw, th), interpolation=cv2.INTER_AREA)
    c0 = cv2.resize(ctrl, (tw, th), interpolation=cv2.INTER_AREA)
    a_s = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_AREA) > SUPPORT_T
    if a_s.sum() == 0:
        return 0.0
    l1 = c1 @ np.array([0.299, 0.587, 0.114], np.float32)
    l0 = c0 @ np.array([0.299, 0.587, 0.114], np.float32)
    return float(np.abs(l1 - l0)[a_s].mean() * 255.0)


def compose_one(bg_path: str, asset_path: str, asset_mode: str,
                placement: Placement, degrade: Degrade, out_stem: Path,
                rng: random.Random, write_control: bool = True) -> dict:
    """Full pipeline for one (background, asset, params) triple.

    Writes <stem>.jpg, <stem>.txt (YOLO), <stem>_mask.png, and (optionally)
    <stem>_ctrl.jpg — the SAME background through the SAME degrade/JPEG.
    Returns the provenance record (caller appends to the manifest).
    """
    bg = np.asarray(Image.open(bg_path).convert("RGB")).astype(np.float32) / 255.0
    rgb, a = load_asset(asset_path, asset_mode)
    comp, full_a = place_asset(bg, rgb, a, placement, asset_mode,
                               seed=rng.randrange(1 << 30))
    comp = degrade_pair(comp, degrade, rng)
    ctrl = degrade_pair(bg, degrade, rng)
    comp = jpeg_roundtrip(comp, degrade.jpeg_q)
    ctrl = jpeg_roundtrip(ctrl, degrade.jpeg_q)

    boxes, stats = boxes_from_alpha(full_a, bg.shape[:2], degrade.target_long)
    stats["contrast"] = contrast_check(comp, ctrl, full_a, degrade.target_long)
    visible = stats["visible"] and stats["contrast"] >= MIN_CONTRAST

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((comp * 255).astype(np.uint8)).save(f"{out_stem}.jpg",
                                                        quality=92)
    Image.fromarray((np.clip(full_a, 0, 1) * 255).astype(np.uint8)).save(
        f"{out_stem}_mask.png")
    lines = []
    if visible:
        for x0, y0, x1, y1, _ in boxes:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            w, h = x1 - x0, y1 - y0
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    Path(f"{out_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    if write_control:
        Image.fromarray((ctrl * 255).astype(np.uint8)).save(
            f"{out_stem}_ctrl.jpg", quality=92)
        Path(f"{out_stem}_ctrl.txt").write_text("")

    return {"bg": bg_path, "asset": asset_path, "asset_mode": asset_mode,
            "placement": asdict(placement), "degrade": asdict(degrade),
            "boxes_norm": [b[:4] for b in boxes] if visible else [],
            "visible": bool(visible), **stats,
            "out": f"{out_stem}.jpg",
            "control": f"{out_stem}_ctrl.jpg" if write_control else None}


def sample_placement(rng: random.Random, faint: bool,
                     horizon: float | None = None,
                     kind: str = "smoke") -> Placement:
    """Horizon-aware: smoke bases on terrain at/below the horizon line;
    cloud/fog confounders live in the sky band or hug the horizon."""
    if horizon is None:
        y_frac, y_mode = rng.uniform(0.15, 0.75), "center"
    elif kind == "smoke":
        y_frac = min(0.97, horizon + rng.uniform(-0.01, 0.22))
        y_mode = "bottom"
    else:  # confounder: cloud high in sky or fog band near horizon
        if rng.random() < 0.5:
            y_frac, y_mode = rng.uniform(0.05, max(0.06, horizon * 0.7)), "center"
        else:
            y_frac, y_mode = horizon + rng.uniform(-0.04, 0.04), "center"
    return Placement(
        x_frac=rng.uniform(0.08, 0.92),
        y_frac=y_frac,
        scale=rng.uniform(0.02, 0.10) if faint else rng.uniform(0.06, 0.30),
        opacity=rng.uniform(0.25, 0.55) if faint else rng.uniform(0.5, 0.95),
        flip=rng.random() < 0.5,
        rot_deg=rng.uniform(-8, 8),
        tint_strength=rng.uniform(0.15, 0.45),
        occlude_frac=rng.uniform(0.0, 0.35),
        y_mode=y_mode,
    )


def sample_degrade(rng: random.Random, target_long: int = 1280) -> Degrade:
    return Degrade(blur_sigma=rng.uniform(0.0, 1.2),
                   noise_sigma=rng.uniform(0.0, 0.012),
                   jpeg_q=rng.randint(62, 95),
                   target_long=target_long)
