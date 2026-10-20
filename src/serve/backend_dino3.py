"""DINOv3 detector backend (frozen/LoRA ViT + SFP neck + Faster R-CNN head),
satisfying the src/serve/backends.py contract.

Model: src/v3/det_modeling.py (iteration-3 Track B/C stack). Iteration 4 adds
hand-rolled LoRA on the ViT attention q/k/v/o projections (`inject_lora`,
`LoRALinear`); the LoRA runs (v4B4_lora_L on dinov3-vitl16, v4B6_lora_Hplus
on dinov3-vith16plus) are the checkpoints this backend must serve.

Checkpoint format (train_det.py): {"model": trainable_state_dict, "opt", "step",
"best_map", "args"}. The trainable state holds neck + head + LoRA A/B only;
the frozen backbone weights come from the local HF snapshot. A LoRA
checkpoint therefore only loads if the SAME injection as training is applied
BEFORE load_state_dict (otherwise "unexpected keys ... lora_A/lora_B"). This
mirrors cluster/v3/dino_infer.py `--lora R [--lora-alpha A] [--lora-merge]`
exactly: ViTBackbone(frozen=True, lora_r=R, lora_alpha=A) -> build_detector
-> load_trainable_state_dict(model, ck["model"]).

Auto-detection (so an operator only sets SMOKE_WEIGHTS_PATH):
  * LoRA rank  : from the `.lora_A` tensors in the checkpoint (shape [r, d]).
  * LoRA alpha : checkpoint args (None -> 2*r, same default as training).
  * geometry   : checkpoint args width/height (1280x736 for every v3/v4 run).
  * backbone   : checkpoint args backbone_dir basename under models/v3/hf/,
                 else by the neck input dim (1024 -> dinov3-vitl16,
                 1280 -> dinov3-vith16plus, 4096 -> dinov3-vit7b16).
                 Gated weights are NEVER downloaded; a missing snapshot is a
                 hard error naming the expected directory.
Every auto-detected value can be overridden by constructor kwargs / SMOKE_*
settings (see src/serve/config.py); a mismatching override is a hard error,
not a silent fallback.

PREPROCESSING (parity with training/eval, src/v3/det_data.py + dino_infer.py):
PIL RGB -> bilinear resize by r = min(W/iw, H/ih) -> pasted top-left on a
(W, H) 114-gray canvas -> /255 -> ImageNet mean/std. The Faster R-CNN
transform is a no-op at this fixed size (build_detector). Boxes come back in
canvas pixels and are divided by r to reach original-image pixels; they are
NOT clipped to the original image (a box may extend into the pad region), so
they match the cluster preds jsonl bit-for-bit in convention.

Raw scores are Faster R-CNN softmax confidences, NOT calibrated probabilities.
Adapted (LoRA/full) backbones saturate near 1.0 (reports/v4/dino_adapt_results.md):
the val-selected operating thresholds for B4/B6 are >= 0.97, so the YOLO
service defaults (0.30 / 0.15) are meaningless for this backend and must be
overridden via SMOKE_SINGLE_FRAME_ALERT_CONF / SMOKE_SEQ_FRAME_CONF.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.serve.detector import Detection
from src.v3.det_modeling import (IMAGENET_MEAN, IMAGENET_STD, LoRALinear, ViTBackbone,
                                 build_detector, load_trainable_state_dict)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HF_ROOT = _REPO_ROOT / "models" / "v3" / "hf"
# neck input dim (== backbone hidden_size) -> local HF snapshot directory name
_EMBED_TO_SNAPSHOT = {1024: "dinov3-vitl16", 1280: "dinov3-vith16plus", 4096: "dinov3-vit7b16"}
# SimpleFeaturePyramid stage '2' (scale 1.0) starts with Conv2d(in_dim, 256, 1)
_NECK_PROBE_KEY = "backbone.neck.stages.2.0.weight"


def letterbox_params(iw: int, ih: int, tw: int, th: int):
    """Identical to src/v3/det_data.py::letterbox_params (kept local so the
    serving path does not import the training dataset module)."""
    r = min(tw / iw, th / ih)
    nw, nh = round(iw * r), round(ih * r)
    return r, nw, nh


def inspect_checkpoint(ck) -> dict:
    """Read what the backend needs from a train_det.py checkpoint (or a bare
    state dict): trainable state, recorded args, LoRA rank (0 = none),
    neck input dim, step / best_map."""
    if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
        sd, meta = ck["model"], ck
    else:
        sd, meta = ck, {}
    args = meta.get("args") or {}
    if not isinstance(args, dict):  # argparse.Namespace pickled
        args = vars(args)
    lora_a = [k for k in sd if k.endswith(".lora_A")]
    ranks = {int(sd[k].shape[0]) for k in lora_a}
    if len(ranks) > 1:
        raise ValueError(f"checkpoint mixes LoRA ranks {sorted(ranks)}")
    embed_dim = int(sd[_NECK_PROBE_KEY].shape[1]) if _NECK_PROBE_KEY in sd else None
    return {"state": sd, "args": args,
            "lora_r": next(iter(ranks)) if ranks else 0,
            "n_lora_tensors": 2 * len(lora_a),
            "embed_dim": embed_dim,
            "step": meta.get("step"), "best_map": meta.get("best_map")}


def resolve_backbone_dir(explicit: str | None, ck_args: dict, embed_dim: int | None) -> Path:
    """Locate the local HF snapshot. Never downloads (gated weights)."""
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    else:
        rec = ck_args.get("backbone_dir")
        if rec:
            cands.append(Path(rec))                 # as recorded (cluster path)
            cands.append(_HF_ROOT / Path(rec).name)  # same snapshot name, local
        if embed_dim in _EMBED_TO_SNAPSHOT:
            cands.append(_HF_ROOT / _EMBED_TO_SNAPSHOT[embed_dim])
    for c in cands:
        if (c / "config.json").is_file():
            has_w = any(c.glob("*.safetensors")) or (c / "pytorch_model.bin").is_file()
            if not has_w:
                raise FileNotFoundError(
                    f"backbone snapshot {c} has config.json but no weights; gated DINOv3 "
                    "weights are not downloaded by the service -- stage them "
                    "(scripts/v3/stage_dinov3.sh) or set SMOKE_DINO_BACKBONE_DIR")
            return c
    raise FileNotFoundError(
        "no DINOv3 backbone snapshot found (tried: "
        + ", ".join(str(c) for c in cands)
        + "). Set SMOKE_DINO_BACKBONE_DIR to a local HF snapshot; weights are never downloaded.")


@torch.no_grad()
def merge_lora_(model: nn.Module) -> int:
    """Fold W += (alpha/r) * B @ A into each LoRALinear's base nn.Linear and
    drop the LoRA branch (deployment form: identical function, plain-backbone
    latency). Mirrors dino_infer.py --lora-merge; the merged Linear is created
    on the base weight's device so this also works after model.to(cuda)."""
    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, LoRALinear):
            continue
        base = mod.base
        w = base.weight.float() + (mod.lora_B.float() @ mod.lora_A.float()) * mod.scaling
        lin = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
                        dtype=base.weight.dtype, device=base.weight.device)
        lin.weight.copy_(w.to(base.weight.dtype))
        if base.bias is not None:
            lin.bias.copy_(base.bias)
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[-1], lin)
        n += 1
    return n


class Dino3Backend:
    """Single-frame DINOv3 + Faster R-CNN detector wrapper.

    Args:
        weights_path: train_det.py checkpoint (best.pt / last.pt), frozen,
            LoRA or full-unfreeze. LoRA is auto-detected from its keys.
        device: "cpu" or "cuda[:i]".
        imgsz: service-wide setting; used ONLY as the letterbox width when the
            checkpoint records no geometry (all v3/v4 runs record 1280x736).
        conf_floor: box_score_thresh AND post-filter; scores below are dropped.
        backbone_dir: local HF snapshot override (default: auto, see module doc).
        lora_r: None = auto-detect (0 when the checkpoint has no LoRA keys);
            an explicit value must equal the detected rank.
        lora_alpha: None = checkpoint's training value (None -> 2*r).
        lora_merge: fold LoRA into the base linears after loading.
        width/height: letterbox geometry override (multiples of the patch size).
        class_names: {id: name} override; default {0: "smoke"}.
    """

    def __init__(self, weights_path: str, device: str = "cpu", imgsz: int = 1280,
                 conf_floor: float = 0.05, backbone_dir: str | None = None,
                 lora_r: int | None = None, lora_alpha: float | None = None,
                 lora_merge: bool = False, width: int | None = None,
                 height: int | None = None, class_names: dict | None = None, **kw):
        self.weights_path = str(weights_path)
        self.device = torch.device(device)
        self.conf_floor = float(conf_floor)
        # contract parity with SmokeDetector.filter_by_conf; never emits tiles
        self.tile_conf_offset = float(kw.get("tile_conf_offset", 0.0))
        self.class_names = ({int(k): str(v) for k, v in class_names.items()}
                            if class_names else {0: "smoke"})

        ck = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        info = inspect_checkpoint(ck)
        ck_args = info["args"]

        # ---- LoRA rank / alpha: auto from checkpoint, overrides must agree
        detected_r = info["lora_r"]
        if lora_r is None:
            r = detected_r
        else:
            r = int(lora_r)
            if r != detected_r:
                have = f"rank {detected_r}" if detected_r else "no LoRA keys"
                raise ValueError(f"lora_r={r} requested but checkpoint has {have} "
                                 f"({self.weights_path})")
        alpha = lora_alpha if lora_alpha is not None else ck_args.get("lora_alpha")
        if r == 0:
            alpha = None

        # ---- geometry: checkpoint-recorded letterbox size wins
        self.width = int(width or ck_args.get("width") or imgsz)
        self.height = int(height or ck_args.get("height") or 736)
        if not width and ck_args.get("width") and int(ck_args["width"]) != int(imgsz):
            print(f"[dino3] note: imgsz={imgsz} ignored; checkpoint geometry "
                  f"{self.width}x{self.height} used", flush=True)

        # ---- backbone snapshot (local only)
        self.backbone_dir = resolve_backbone_dir(backbone_dir, ck_args, info["embed_dim"])
        bb = ViTBackbone(str(self.backbone_dir), frozen=True, lora_r=r, lora_alpha=alpha)
        if info["embed_dim"] is not None and bb.embed_dim != info["embed_dim"]:
            raise ValueError(f"checkpoint neck expects {info['embed_dim']}-d features but "
                             f"backbone {self.backbone_dir} has hidden_size {bb.embed_dim}")
        ps = bb.patch_size
        if self.width % ps or self.height % ps:
            raise ValueError(f"letterbox {self.width}x{self.height} not a multiple of patch {ps}")

        model = build_detector(bb, image_size=(self.width, self.height),
                               box_score_thresh=self.conf_floor)
        # strict on head/neck/LoRA keys; only frozen backbone keys may be missing
        load_trainable_state_dict(model, info["state"])
        model.to(self.device).eval()
        self.n_lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
        self.n_merged = merge_lora_(model) if (lora_merge and r > 0) else 0
        self._model = model

        self.lora_r = r
        self.lora_alpha = bb.lora_alpha
        self.lora_modules = len(bb.lora_modules)
        self.patch_size = ps
        self.ckpt_step = info["step"]
        self.ckpt_best_map = info["best_map"]
        self.n_ckpt_tensors = len(info["state"])
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    # ---------------------------------------------------------------- contract
    def filter_by_conf(self, dets: list[Detection], conf: float,
                       tile_conf_offset: float | None = None) -> list[Detection]:
        """Same semantics as SmokeDetector.filter_by_conf (no tiles here)."""
        off = self.tile_conf_offset if tile_conf_offset is None else tile_conf_offset
        return [d for d in dets
                if d.score >= (conf + off if d.source == "tile" else conf)]

    def info(self) -> dict:
        return {"backbone_dir": str(self.backbone_dir), "letterbox": [self.width, self.height],
                "lora_r": self.lora_r, "lora_alpha": self.lora_alpha,
                "lora_modules": self.lora_modules, "lora_params": self.n_lora_params,
                "lora_merged_linears": self.n_merged, "ckpt_step": self.ckpt_step,
                "ckpt_best_map": self.ckpt_best_map, "ckpt_tensors": self.n_ckpt_tensors,
                "conf_floor": self.conf_floor, "device": str(self.device)}

    def _preprocess(self, img: np.ndarray):
        """HWC uint8 BGR -> (normalized CHW float tensor [3,H,W] on CPU, r)."""
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        pil = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))  # BGR -> RGB
        iw, ih = pil.size
        r, nw, nh = letterbox_params(iw, ih, self.width, self.height)
        pil = pil.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.width, self.height), (114, 114, 114))
        canvas.paste(pil, (0, 0))
        x = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255.0
        x = (x - self._mean) / self._std
        return x, r

    @torch.inference_mode()
    def detect(self, img: np.ndarray) -> tuple[list[Detection], dict]:
        """img: HWC uint8 BGR (cv2 order), same as SmokeDetector.detect."""
        t0 = time.perf_counter()
        x, r = self._preprocess(img)
        x = x.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_pre = time.perf_counter()

        out = self._model([x])[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_inf = time.perf_counter()

        boxes = out["boxes"].float().cpu().numpy() / r   # canvas px -> original px
        scores = out["scores"].float().cpu().numpy()
        labels = out["labels"].cpu().numpy()
        dets = []
        for b, s, lab in zip(boxes, scores, labels):
            if float(s) < self.conf_floor:
                continue
            cid = int(lab) - 1  # torchvision label 0 = background
            dets.append(Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                                  float(s), cid, self.class_names.get(cid, str(cid)), "full"))
        t1 = time.perf_counter()
        timing = {"latency_ms": round((t1 - t0) * 1000, 1),
                  "preprocess_ms": round((t_pre - t0) * 1000, 1),
                  "infer_ms": round((t_inf - t_pre) * 1000, 1),
                  "postprocess_ms": round((t1 - t_inf) * 1000, 1),
                  # contract parity with the yolo backend's timing keys
                  "full_pass_ms": round((t_inf - t0) * 1000, 1),
                  "tile_pass_ms": 0.0, "n_tiles": 0}
        return dets, timing
