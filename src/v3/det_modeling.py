"""Track B/C detection model: frozen ViT backbone (DINOv3 / stand-in DINOv2)
+ ViTDet-style SimpleFeaturePyramid neck + torchvision Faster R-CNN head.

Design notes (see reports/v3/trackBC_prep.md):
- Backbone loaded via HF transformers (DINOv3ViTModel once gated access lands,
  Dinov2Model as the ungated stand-in). DINOv3 uses RoPE -> arbitrary input
  sizes with no pos-embed interpolation; input H,W must be multiples of the
  patch size (16 for dinov3-*, 14 for dinov2-large).
- Neck/head are identical across B1/B2/B3/C1 (only the input projection dim
  differs with the backbone's embed dim - forced difference, documented).
- The 7B precompute route bypasses the backbone at train time:
  `forward_with_features()` drives the same RPN/RoI internals from cached
  feature maps, so head weights/losses are identical to the end-to-end path.

License: our code. Deps: transformers (Apache-2.0), torchvision (BSD-3).
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)  # dinov3/data/transforms.py (pinned repo)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Exact hyperparameters from facebookresearch/dinov3 @ 6876159a
# (dinov3/hub/backbones.py), used ONLY to build random-init wiring-test models
# until gated weights land. All have patch 16, 4 register/storage tokens, RoPE.
DINOV3_SPECS = {
    "dinov3_vitl16": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                          intermediate_size=4096, use_gated_mlp=False, num_register_tokens=4,
                          patch_size=16, query_bias=True, key_bias=False, value_bias=True),
    "dinov3_vith16plus": dict(hidden_size=1280, num_hidden_layers=32, num_attention_heads=20,
                              # swiglu ffn_ratio 6.0 -> intermediate 1280*6 (gated)
                              intermediate_size=7680, use_gated_mlp=True, num_register_tokens=4,
                              patch_size=16, query_bias=True, key_bias=False, value_bias=True),
    "dinov3_vit7b16": dict(hidden_size=4096, num_hidden_layers=40, num_attention_heads=32,
                           # swiglu64 ffn_ratio 3 -> 4096*3=12288, rounded to /64 -> 12288
                           intermediate_size=12288, use_gated_mlp=True, num_register_tokens=4,
                           patch_size=16, query_bias=False, key_bias=False, value_bias=False),
    # not a real model: cheap wiring tests of the dinov3 code path (CPU/pod)
    "dinov3_test_tiny": dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                             intermediate_size=128, use_gated_mlp=False, num_register_tokens=4,
                             patch_size=16, query_bias=True, key_bias=False, value_bias=True),
}


class LoRALinear(nn.Module):
    """Hand-rolled LoRA wrapper (no peft: offline pods, smoke-dino3:v1 image).

    Replaces an nn.Linear: y = base(x) + (x @ A^T @ B^T) * (alpha / r).
    Base weights are frozen; only lora_A/lora_B train. A is kaiming-init,
    B zero-init -> the wrapped layer is exactly the base layer at step 0.
    state_dict keys: <name>.base.weight[/bias], <name>.lora_A, <name>.lora_B.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        assert r > 0
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r
        dt = base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

    def extra_repr(self):
        return f"r={self.r}, alpha={self.alpha}"


# Attention projection leaf names per HF layout. Matching additionally
# requires ".attention" in the qualified module path so e.g. a pooler
# "dense" could never be hit.
#   DINOv3ViTModel (transformers 5.17): model.layer.N.attention.{q,k,v,o}_proj
#   Dinov2Model: encoder.layer.N.attention.attention.{query,key,value}
#                + encoder.layer.N.attention.output.dense (the o-projection)
LORA_TARGET_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj",
                      "query", "key", "value", "dense")


def inject_lora(hf_model: nn.Module, r: int, alpha: float) -> list[str]:
    """Replace ViT attention q/k/v/o nn.Linear modules with LoRALinear.
    Returns qualified names of the replaced modules. Idempotence guard: an
    already-wrapped module (LoRALinear.base) is skipped because the leaf
    name check runs on the ORIGINAL module list snapshot."""
    replaced = []
    for name, module in list(hf_model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in LORA_TARGET_LEAVES or ".attention" not in name:
            continue
        if name.endswith(".base"):  # already wrapped
            continue
        parent = hf_model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else hf_model
        setattr(parent, leaf, LoRALinear(module, r, alpha))
        replaced.append(name)
    if not replaced:
        raise RuntimeError("inject_lora found no attention projections — "
                           "unknown backbone layout")
    return replaced


class ViTBackbone(nn.Module):
    """HF ViT encoder -> single stride-`patch_size` 2D feature map.

    model_dir: local HF snapshot dir (pods are offline). model_type is read
    from its config.json ("dinov3_vit" or "dinov2").
    frozen: no grads, module kept in eval() mode permanently.
    unfreeze_top: number of final transformer blocks to leave trainable
    (B3 route); implies frozen prefix + gradient checkpointing.
    unfreeze_all: entire ViT trainable (incl. patch/pos embeddings, cls and
    register tokens).
    lora_r/lora_alpha: inject hand-rolled LoRA on attention q/k/v/o linears;
    all base backbone weights frozen, only LoRA A/B train
    (alpha defaults to 2*r).
    grad_ckpt: enable HF gradient checkpointing (memory for compute) — only
    meaningful when some backbone part trains. unfreeze_top keeps its
    original always-on behavior regardless of this flag.
    unfreeze_top / unfreeze_all / lora_r are mutually exclusive.
    """

    def __init__(self, model_dir: str, frozen: bool = True, unfreeze_top: int = 0,
                 dtype: torch.dtype | None = None, random_init_spec: str | None = None,
                 unfreeze_all: bool = False, lora_r: int = 0,
                 lora_alpha: float | None = None, grad_ckpt: bool = False):
        super().__init__()
        if random_init_spec is not None:
            # gated-weights wiring path: build from exact spec, random weights
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
            spec = dict(DINOV3_SPECS[random_init_spec])
            cfg = DINOv3ViTConfig(**spec)
            self.hf = DINOv3ViTModel(cfg)
            self.model_type = "dinov3_vit"
        else:
            from transformers import AutoConfig, AutoModel
            cfg = AutoConfig.from_pretrained(model_dir)
            self.model_type = cfg.model_type
            kw = {}
            if dtype is not None:
                kw["dtype"] = dtype
            self.hf = AutoModel.from_pretrained(model_dir, **kw)
        self.config = self.hf.config
        self.embed_dim = self.config.hidden_size
        self.patch_size = self.config.patch_size
        # tokens preceding the patch grid: CLS + register tokens (dinov3: 4)
        self.num_prefix = 1 + int(getattr(self.config, "num_register_tokens", 0))
        if sum([unfreeze_top > 0, unfreeze_all, lora_r > 0]) > 1:
            raise ValueError("unfreeze_top / unfreeze_all / lora_r are mutually exclusive")
        self.frozen = (frozen and unfreeze_top == 0 and not unfreeze_all
                       and lora_r == 0)
        self.unfreeze_top = unfreeze_top
        self.unfreeze_all = unfreeze_all
        self.lora_r = lora_r
        self.lora_alpha = (float(lora_alpha) if lora_alpha is not None
                           else 2.0 * lora_r) if lora_r > 0 else None
        self.lora_modules: list[str] = []
        if self.frozen:
            self.hf.requires_grad_(False)
            self.hf.eval()
        elif lora_r > 0:
            self.hf.requires_grad_(False)
            self.lora_modules = inject_lora(self.hf, lora_r, self.lora_alpha)
        elif unfreeze_all:
            self.hf.requires_grad_(True)
            # mask_token is only consumed when bool_masked_pos is passed
            # (never in this stack) -> it would get no grad and crash DDP as
            # an unused parameter; keep it frozen. Everything else trains,
            # incl. patch/pos embeddings, cls + register tokens.
            for n, p in self.hf.named_parameters():
                if n.endswith("mask_token"):
                    p.requires_grad_(False)
        elif unfreeze_top > 0:
            self.hf.requires_grad_(False)
            blocks = self._blocks()
            for b in blocks[-unfreeze_top:]:
                b.requires_grad_(True)
            # final norm feeds the head -> keep trainable alongside top blocks
            for name in ("layernorm", "norm"):
                m = getattr(self.hf, name, None)
                if m is not None:
                    m.requires_grad_(True)
            self.hf.gradient_checkpointing_enable()
        if grad_ckpt and not self.frozen and unfreeze_top == 0 \
                and hasattr(self.hf, "gradient_checkpointing_enable"):
            self.hf.gradient_checkpointing_enable()

    def _blocks(self):
        # dinov2: hf.encoder.layer; dinov3 (transformers 5.17): hf.model.layer
        for holder in (getattr(self.hf, "encoder", None),
                       getattr(self.hf, "model", None), self.hf):
            if holder is None:
                continue
            for attr in ("layer", "layers", "blocks"):
                if hasattr(holder, attr):
                    return list(getattr(holder, attr))
        raise AttributeError("cannot locate transformer blocks")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.hf.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        ps = self.patch_size
        assert h % ps == 0 and w % ps == 0, f"input {h}x{w} not multiple of patch {ps}"
        if self.frozen and not torch.is_grad_enabled():
            out = self.hf(pixel_values=x)
        elif self.frozen:
            with torch.no_grad():
                out = self.hf(pixel_values=x)
        else:
            out = self.hf(pixel_values=x)
        tok = out.last_hidden_state[:, self.num_prefix:, :]  # [B, N, C]
        gh, gw = h // ps, w // ps
        assert tok.shape[1] == gh * gw, (
            f"token count {tok.shape[1]} != grid {gh}x{gw}; prefix={self.num_prefix}")
        fmap = tok.transpose(1, 2).reshape(b, self.embed_dim, gh, gw)
        return fmap.float()


class LayerNorm2d(nn.Module):
    """channels-first LayerNorm (ViTDet uses LN in the pyramid convs)."""

    def __init__(self, ch, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias = nn.Parameter(torch.zeros(ch))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class SimpleFeaturePyramid(nn.Module):
    """ViTDet-style pyramid from ONE stride-ps map -> strides ~{ps/4, ps/2, ps, 2ps}
    (+pool level). Names '0'..'3' + 'pool' to match torchvision FPN conventions.
    """

    SCALES = (4.0, 2.0, 1.0, 0.5)

    def __init__(self, in_dim: int, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        stages = []
        for s in self.SCALES:
            if s == 4.0:
                layers = [nn.ConvTranspose2d(in_dim, in_dim // 2, 2, 2),
                          LayerNorm2d(in_dim // 2), nn.GELU(),
                          nn.ConvTranspose2d(in_dim // 2, in_dim // 4, 2, 2)]
                dim = in_dim // 4
            elif s == 2.0:
                layers = [nn.ConvTranspose2d(in_dim, in_dim // 2, 2, 2)]
                dim = in_dim // 2
            elif s == 1.0:
                layers = []
                dim = in_dim
            else:
                layers = [nn.MaxPool2d(2, 2)]
                dim = in_dim
            layers += [nn.Conv2d(dim, out_channels, 1, bias=False),
                       LayerNorm2d(out_channels),
                       nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                       LayerNorm2d(out_channels)]
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)

    def forward(self, fmap: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        out = OrderedDict()
        for i, stage in enumerate(self.stages):
            out[str(i)] = stage(fmap)
        out["pool"] = F.max_pool2d(out["3"], 1, 2, 0)
        return out


class BackboneWithNeck(nn.Module):
    """Composition consumed by torchvision FasterRCNN (needs .out_channels)."""

    def __init__(self, backbone: ViTBackbone, neck: SimpleFeaturePyramid):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.out_channels = neck.out_channels

    def forward(self, x):
        return self.neck(self.backbone(x))


def build_detector(backbone: ViTBackbone | None, embed_dim: int | None = None,
                   image_size=(1280, 736), box_score_thresh=0.05,
                   detections_per_img=100):
    """FasterRCNN with anchors shifted down for small smoke.

    image_size = (W, H); must be multiples of the backbone patch size.
    If backbone is None (cached-feature route), a neck-only model is built
    and forward_with_features() must be used.
    """
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    dim = backbone.embed_dim if backbone is not None else embed_dim
    neck = SimpleFeaturePyramid(dim)
    if backbone is not None:
        bb = BackboneWithNeck(backbone, neck)
        patch = backbone.patch_size
    else:
        bb = BackboneWithNeck(nn.Identity(), neck)  # backbone bypassed
        bb.backbone = None
        patch = 16
    w, h = image_size
    anchor_gen = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    model = FasterRCNN(
        bb, num_classes=2,  # background + smoke
        rpn_anchor_generator=anchor_gen,
        min_size=min(h, w), max_size=max(h, w),
        image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0],  # dataset normalizes
        box_score_thresh=box_score_thresh,
        box_detections_per_img=detections_per_img,
    )
    # Dataset letterboxes to exactly (h, w); make the internal transform a
    # no-op: fixed_size skips the min/max resize, size_divisible=patch keeps
    # the padded canvas on the patch grid (both dims are already multiples).
    model.transform.fixed_size = (h, w)  # GeneralizedRCNNTransform: (h, w)
    model.transform.size_divisible = patch
    return model


@torch.no_grad()
def _imagelist_for(shapes, device):
    from torchvision.models.detection.image_list import ImageList
    h = max(s[0] for s in shapes)
    w = max(s[1] for s in shapes)
    dummy = torch.zeros((len(shapes), 3, h, w), device=device)
    return ImageList(dummy, [tuple(s) for s in shapes])


def forward_with_features(model, fmaps: torch.Tensor, image_shapes, targets=None):
    """Drive the SAME RPN/RoI internals from a precomputed backbone feature map
    (C1 cached-feature route). fmaps: [B, C, gh, gw] float. image_shapes:
    list of (h, w) the features were computed at. Returns (losses, detections).
    """
    device = fmaps.device
    images = _imagelist_for(image_shapes, device)
    if targets is not None:
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    features = model.backbone.neck(fmaps)
    proposals, proposal_losses = model.rpn(images, features, targets)
    detections, detector_losses = model.roi_heads(
        features, proposals, images.image_sizes, targets)
    losses = {}
    losses.update(proposal_losses)
    losses.update(detector_losses)
    return losses, detections


def param_group_counts(model) -> dict:
    """Trainable/total parameter counts per group for a build_detector()
    model: backbone (ViT), neck (SimpleFeaturePyramid), head (RPN + RoI).
    Iteration reports need the exact integers — do not round."""
    counts = {g: {"trainable": 0, "total": 0} for g in ("backbone", "neck", "head")}
    for n, p in model.named_parameters():
        if n.startswith("backbone.backbone."):
            g = "backbone"
        elif n.startswith("backbone.neck."):
            g = "neck"
        else:
            g = "head"
        counts[g]["total"] += p.numel()
        if p.requires_grad:
            counts[g]["trainable"] += p.numel()
    counts["all"] = {"trainable": sum(c["trainable"] for c in counts.values()),
                     "total": sum(c["total"] for c in counts.values())}
    return counts


def trainable_state_dict(model) -> dict:
    """Checkpoint only what can change: everything except frozen backbone
    params. Keeps B1/B2 checkpoints ~60MB instead of 1.2-3.4GB."""
    frozen = set()
    bb = getattr(model.backbone, "backbone", None)
    if isinstance(bb, ViTBackbone):
        for name, p in bb.named_parameters():
            if not p.requires_grad:
                frozen.add("backbone.backbone." + name)
    return {k: v for k, v in model.state_dict().items() if k not in frozen}


def load_trainable_state_dict(model, state: dict):
    missing, unexpected = model.load_state_dict(state, strict=False)
    # every missing key must belong to the frozen backbone (came from_pretrained)
    bad = [k for k in missing if not k.startswith("backbone.backbone.")]
    if bad or unexpected:
        raise RuntimeError(f"ckpt mismatch: missing={bad[:5]} unexpected={list(unexpected)[:5]}")
