"""V5b thin-smoke segmentation — CPU inference (image -> smoke probability map).

Usable by the demo's /segment slot:
    from scripts.v5b.seg_infer import SmokeSegmenter
    seg = SmokeSegmenter("models/v5b/seg/segV5b_a/seg01/lora/best.pt")      # CPU by default
    prob = seg.predict(pil_image)            # float32 HxW in [0,1], ORIGINAL image size
    seg.predict_png(pil_image, "out.png")    # uint8 prob map PNG (+ optional overlay)
CLI:
    python scripts/v5b/seg_infer.py --weights <best.pt> --image in.jpg --out prob.png [--overlay ov.png]
Backbone weights are never downloaded: DINOv3-L comes from a local HF snapshot
(SMOKE_DINO_BACKBONE_DIR, or models/v3/hf/dinov3-vitl16 / models/v2/dinov3-vitl16);
the checkpoint holds only the trained head/neck (+LoRA) tensors.
Measured: ~3 s/image at 1280x736 on this 192-core host (32 threads), ~1.1 s at 960x544.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.v5b.seg_head import build_model, load_saved_state_dict  # noqa: E402
from src.v5b.seg_data import preprocess_image  # noqa: E402

BACKBONE_CANDIDATES = [ROOT / "models/v3/hf/dinov3-vitl16", ROOT / "models/v2/dinov3-vitl16"]


def resolve_backbone_dir(explicit: str | None) -> str:
    for c in [explicit, os.environ.get("SMOKE_DINO_BACKBONE_DIR")] + BACKBONE_CANDIDATES:
        if c and Path(c).exists() and (Path(c) / "config.json").exists():
            return str(c)
    raise FileNotFoundError("no local dinov3-vitl16 snapshot; set SMOKE_DINO_BACKBONE_DIR")


class SmokeSegmenter:
    def __init__(self, weights: str, device: str = "cpu", backbone_dir: str | None = None,
                 width: int | None = None, height: int | None = None, threads: int = 32):
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        a = ck["args"]
        self.kind = ck.get("kind", a["kind"])
        self.width = width or a.get("width", 1280)
        self.height = height or a.get("height", 736)
        if device == "cpu":
            torch.set_num_threads(min(threads, os.cpu_count() or 1))
        if self.kind == "r50":
            model = build_model("r50", r50_weights=None)
            # frozen ImageNet trunk is not in the checkpoint -> load from the local file
            r50w = ROOT / "models/v5b/seg/pretrained/resnet50-11ad3fa6.pth"
            model = build_model("r50", r50_weights=str(r50w) if r50w.exists() else None)
            self.backbone_dir = None
        else:
            self.backbone_dir = resolve_backbone_dir(backbone_dir)
            model = build_model(self.kind, self.backbone_dir, None, a.get("lora", 16), a.get("lora_alpha"))
        load_saved_state_dict(model, ck["model"])
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.meta = {"kind": self.kind, "weights": str(weights), "backbone_dir": self.backbone_dir,
                     "input_wh": [self.width, self.height], "best_step": ck.get("best_step"),
                     "val_metrics": ck.get("val_metrics"), "param_counts": ck.get("param_counts")}

    @torch.no_grad()
    def predict(self, img: Image.Image) -> np.ndarray:
        x, m = preprocess_image(img, self.width, self.height)
        logits, _ = self.model(x.to(self.device))
        prob = torch.sigmoid(logits[0, 0, : m["nh"], : m["nw"]]).cpu()
        prob = torch.nn.functional.interpolate(prob[None, None], size=(m["orig_h"], m["orig_w"]),
                                               mode="bilinear", align_corners=False)[0, 0]
        return prob.numpy().astype(np.float32)

    def predict_png(self, img: Image.Image, out_png: str, overlay_png: str | None = None,
                    thr: float = 0.5) -> dict:
        t = time.time()
        prob = self.predict(img)
        Image.fromarray((prob * 255).round().astype(np.uint8)).save(out_png)
        if overlay_png:
            overlay(img, prob, thr).save(overlay_png)
        return {"out": out_png, "overlay": overlay_png, "max_prob": float(prob.max()),
                "smoke_frac@thr": float((prob >= thr).mean()), "seconds": round(time.time() - t, 2)}


def overlay(img: Image.Image, prob: np.ndarray, thr: float = 0.5, alpha: float = 0.55) -> Image.Image:
    base = np.asarray(img.convert("RGB")).astype(np.float32)
    heat = np.zeros_like(base)
    heat[..., 0] = 255 * prob          # red = probability
    heat[..., 1] = 255 * np.clip(prob - 0.5, 0, 0.5) * 2  # yellowish where confident
    w = (alpha * prob)[..., None]
    out = base * (1 - w) + heat * w
    edge = (prob >= thr).astype(np.uint8)
    # thin contour at the threshold
    e = np.zeros_like(edge)
    e[1:, :] |= edge[1:, :] ^ edge[:-1, :]
    e[:, 1:] |= edge[:, 1:] ^ edge[:, :-1]
    out[e.astype(bool)] = (0, 255, 255)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    seg = SmokeSegmenter(a.weights, a.device, width=a.width, height=a.height)
    r = seg.predict_png(Image.open(a.image), a.out, a.overlay, a.thr)
    print(r)


if __name__ == "__main__":
    main()
