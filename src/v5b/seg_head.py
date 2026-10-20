"""V5b thin-smoke segmentation head on the iteration-3/4 DINOv3 detection stack.

Architecture (DinoSeg):
  ViTBackbone (DINOv3-L/16, frozen or LoRA r16)  -> stride-16 map [B,1024,H/16,W/16]
  SimpleFeaturePyramid (src/v3)                   -> '0'..'3' = strides 4/8/16/32, 256 ch
  FPNLiteSegHead (this file, UPerNet-lite)        -> 1-channel smoke logit at stride 4
  bilinear upsample to input resolution.
Baseline (ResNet50Seg): frozen ImageNet ResNet-50 C2..C5 + the same U-Net/FPN-style
decoder, trained on identical data (the "simpler segmentation baseline").
Losses: BCE + soft Dice on frames WITH a mask only; letterbox padding (valid==0)
and frames without a mask are excluded from the loss.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # repo layout
    from src.v3.det_modeling import (ViTBackbone, SimpleFeaturePyramid,
                                     IMAGENET_MEAN, IMAGENET_STD)
except ImportError:  # flat pod tarball (/tmp/job)
    from det_modeling import (ViTBackbone, SimpleFeaturePyramid,  # type: ignore
                              IMAGENET_MEAN, IMAGENET_STD)


def conv_gn_act(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.GroupNorm(32 if cout % 32 == 0 else 8, cout), nn.GELU())


class PPM(nn.Module):
    """UPerNet pyramid pooling on the coarsest level."""

    def __init__(self, cin, cout, bins=(1, 2, 3, 6)):
        super().__init__()
        self.bins = bins
        self.branches = nn.ModuleList([conv_gn_act(cin, cout, 1) for _ in bins])
        self.fuse = conv_gn_act(cin + cout * len(bins), cout, 3)

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [x]
        for b, br in zip(self.bins, self.branches):
            y = br(F.adaptive_avg_pool2d(x, b))
            outs.append(F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(outs, 1))


class FPNLiteSegHead(nn.Module):
    """UPerNet-lite: PPM on stride-32, top-down FPN with lateral 1x1s, fuse all
    levels at stride 4, predict 1 logit. in_channels: per level (strides 4,8,16,32)."""

    def __init__(self, in_channels=(256, 256, 256, 256), ch=128, dropout=0.1):
        super().__init__()
        self.lat = nn.ModuleList([conv_gn_act(c, ch, 1) for c in in_channels[:-1]])
        self.ppm = PPM(in_channels[-1], ch)
        self.smooth = nn.ModuleList([conv_gn_act(ch, ch, 3) for _ in in_channels[:-1]])
        self.fuse = nn.Sequential(conv_gn_act(ch * len(in_channels), ch, 3),
                                  conv_gn_act(ch, ch, 3), nn.Dropout2d(dropout))
        self.cls = nn.Conv2d(ch, 1, 1)
        nn.init.constant_(self.cls.bias, -2.0)  # smoke is the minority class

    def forward(self, feats):
        c2, c3, c4, c5 = feats
        p5 = self.ppm(c5)
        laterals = [l(c) for l, c in zip(self.lat, (c2, c3, c4))]
        p = p5
        outs = [p5]
        for i in (2, 1, 0):
            p = laterals[i] + F.interpolate(p, size=laterals[i].shape[-2:],
                                            mode="bilinear", align_corners=False)
            p = self.smooth[i](p)
            outs.append(p)
        tgt = outs[-1].shape[-2:]
        cat = torch.cat([F.interpolate(o, size=tgt, mode="bilinear", align_corners=False)
                         if o.shape[-2:] != tgt else o for o in outs], 1)
        return self.cls(self.fuse(cat))  # [B,1,H/4,W/4]


class DinoSeg(nn.Module):
    def __init__(self, backbone: ViTBackbone, neck_ch=256, head_ch=128):
        super().__init__()
        self.backbone = backbone
        self.neck = SimpleFeaturePyramid(backbone.embed_dim, neck_ch)
        self.head = FPNLiteSegHead((neck_ch,) * 4, head_ch)
        self.kind = "dino"

    def forward(self, x):
        fm = self.backbone(x)
        pyr = self.neck(fm)
        low = self.head([pyr["0"], pyr["1"], pyr["2"], pyr["3"]])
        return F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False), low

    def is_saved_key(self, k: str) -> bool:
        # everything except the frozen HF base weights (LoRA A/B are saved)
        return (not k.startswith("backbone.hf.")) or (".lora_" in k)


class ResNet50Seg(nn.Module):
    """Frozen ImageNet ResNet-50 (torchvision IMAGENET1K_V2 weights loaded from a
    local file; pods are offline) + identical FPN-lite/U-Net-style decoder."""

    def __init__(self, weights_path: str | None, head_ch=128, frozen=True):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet50(weights=None)
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu")
            missing, unexpected = r.load_state_dict(sd, strict=False)
            assert not [m for m in missing if not m.startswith("fc.")], missing
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.frozen = frozen
        if frozen:
            for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
                m.requires_grad_(False)
        self.head = FPNLiteSegHead((256, 512, 1024, 2048), head_ch)
        self.kind = "r50"

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:  # keep BN statistics fixed
            for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
                m.eval()
        return self

    def forward(self, x):
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            c1 = self.stem(x)
            c2 = self.layer1(c1)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
        low = self.head([c2.float(), c3.float(), c4.float(), c5.float()])
        return F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False), low

    def is_saved_key(self, k: str) -> bool:
        return k.startswith("head.")


def build_model(kind: str, backbone_dir: str | None = None, random_spec: str | None = None,
                lora_r: int = 0, lora_alpha: float | None = None, grad_ckpt: bool = False,
                r50_weights: str | None = None) -> nn.Module:
    if kind == "r50":
        return ResNet50Seg(r50_weights)
    if kind == "dino_frozen":
        bb = ViTBackbone(backbone_dir or "", frozen=True, random_init_spec=random_spec)
    elif kind == "dino_lora":
        bb = ViTBackbone(backbone_dir or "", frozen=False, lora_r=lora_r or 16,
                         lora_alpha=lora_alpha, grad_ckpt=grad_ckpt, random_init_spec=random_spec)
    else:
        raise ValueError(kind)
    return DinoSeg(bb)


def seg_loss(logits, target, valid, has_mask, bce_w=1.0, dice_w=1.0):
    """logits/target/valid: [B,1,H,W]; has_mask: [B] bool. Loss only over valid
    pixels of frames that have a mask (incl. all-zero masks of verified negatives)."""
    hm = has_mask.view(-1, 1, 1, 1).to(logits.dtype)
    w = valid.to(logits.dtype) * hm
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    bce = (bce * w).sum() / w.sum().clamp_min(1.0)
    p = torch.sigmoid(logits.float()) * w
    t = target.float() * w
    inter = (p * t).sum((1, 2, 3))
    denom = p.sum((1, 2, 3)) + t.sum((1, 2, 3))
    dice = 1.0 - (2 * inter + 1.0) / (denom + 1.0)
    n = has_mask.to(dice.dtype).sum().clamp_min(1.0)
    dice = (dice * has_mask.to(dice.dtype)).sum() / n
    return {"bce": bce, "dice": dice, "loss": bce_w * bce + dice_w * dice}


def saved_state_dict(model) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if model.is_saved_key(k)}


def load_saved_state_dict(model, state: dict):
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [m for m in missing if model.is_saved_key(m)]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={bad[:5]} unexpected={unexpected[:5]}")


def param_counts(model) -> dict:
    out = {}
    for name in ("backbone", "neck", "head"):
        m = getattr(model, name, None)
        if m is None and name == "backbone" and model.kind == "r50":
            ps = [p for n, p in model.named_parameters() if not n.startswith("head.")]
        elif m is None:
            continue
        else:
            ps = list(m.parameters())
        out[name] = {"trainable": sum(p.numel() for p in ps if p.requires_grad),
                     "total": sum(p.numel() for p in ps)}
    ps = list(model.parameters())
    out["all"] = {"trainable": sum(p.numel() for p in ps if p.requires_grad),
                  "total": sum(p.numel() for p in ps)}
    return out
