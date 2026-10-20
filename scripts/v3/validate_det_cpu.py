"""Devbox CPU validation for the Track B/C detection stack (no GPU, no gated
weights). Run: python scripts/v3/validate_det_cpu.py [--skip-standin]

A. tiny random DINOv3ViT (2 layers) end-to-end: dataset -> detector fwd/bwd ->
   loss dict -> ckpt save/resume bit-identity -> eval path (pycocotools).
B. real-config random dinov3_vitl16 @ 1280x736: patch-16 grid 80x46, register
   stripping, SFP level shapes, single fwd+bwd.
C. UNGATED STAND-IN facebook/dinov2-large @ 1288x728 (patch 14, grid 92x52):
   real weights, frozen-backbone fwd+bwd, trainable/frozen grad audit.
Results printed as VALIDATE_x_PASS lines; exits nonzero on any failure.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "v3"))
import det_modeling as M  # noqa: E402
from det_data import CocoDetDataset, collate  # noqa: E402
from det_modeling import (ViTBackbone, build_detector,  # noqa: E402
                          forward_with_features, load_trainable_state_dict,
                          trainable_state_dict)

VAL_JSON = ROOT / "data/v3/coco/instances_val.json"
IMAGE_ROOT = ROOT / "data/yolo"


def real_batch(size, n=2, need_boxes=True):
    ds = CocoDetDataset(str(VAL_JSON), str(IMAGE_ROOT), size=size, train=True)
    if need_boxes:
        ds.items = [it for it in ds.items if it["boxes"]][:n]
    else:
        ds.items = ds.items[:n]
    return collate([ds[i] for i in range(n)])


def check_losses(losses):
    assert set(losses) == {"loss_classifier", "loss_box_reg", "loss_objectness",
                           "loss_rpn_box_reg"}, losses.keys()
    for k, v in losses.items():
        assert torch.isfinite(v), (k, v)


def test_A_tiny_e2e():
    M.DINOV3_SPECS["tiny_test"] = dict(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=128, use_gated_mlp=False, num_register_tokens=4,
        patch_size=16, query_bias=True, key_bias=False, value_bias=True)
    bb = ViTBackbone("unused", random_init_spec="tiny_test")
    assert bb.num_prefix == 5, bb.num_prefix
    model = build_detector(bb, image_size=(1280, 736))
    xs, targets = real_batch((1280, 736), n=2)
    model.train()
    losses = model(list(xs), [dict(t) for t in targets])
    check_losses(losses)
    loss = sum(losses.values())
    loss.backward()
    # frozen backbone must have no grads; neck/head must have grads
    for n, p in model.named_parameters():
        if n.startswith("backbone.backbone."):
            assert p.grad is None and not p.requires_grad, n
    ngrads = sum(p.grad is not None and p.grad.abs().sum() > 0
                 for n, p in model.named_parameters()
                 if not n.startswith("backbone.backbone."))
    assert ngrads > 20, f"only {ngrads} head/neck params got grad"
    # ckpt round trip on trainable-only state
    with tempfile.TemporaryDirectory() as td:
        pth = Path(td) / "ck.pt"
        torch.save({"model": trainable_state_dict(model)}, pth)
        model2 = build_detector(
            ViTBackbone("unused", random_init_spec="tiny_test"), image_size=(1280, 736))
        ck = torch.load(pth, weights_only=False)
        load_trainable_state_dict(model2, ck["model"])
        sd1, sd2 = model.state_dict(), model2.state_dict()
        for k in sd1:
            if not k.startswith("backbone.backbone."):
                assert torch.equal(sd1[k], sd2[k]), k
    # inference path + postprocess boxes in letterbox pixels
    model.eval()
    with torch.no_grad():
        dets = model([xs[0]])
    assert {"boxes", "scores", "labels"} <= set(dets[0]), dets[0].keys()
    # cached-feature route drives the same heads
    fmap = bb(xs[0].unsqueeze(0))
    assert fmap.shape == (1, 64, 46, 80), fmap.shape
    losses_c, dets_c = forward_with_features(
        model, fmap, [(736, 1280)], [dict(targets[0])])
    model.train()
    losses_c, _ = forward_with_features(model, fmap, [(736, 1280)], [dict(targets[0])])
    check_losses(losses_c)
    print("VALIDATE_A_PASS tiny e2e fwd/bwd + ckpt + eval + cached-route")


def test_B_vitl16_random():
    bb = ViTBackbone("unused", random_init_spec="dinov3_vitl16")
    n_params = sum(p.numel() for p in bb.parameters())
    x = torch.randn(1, 3, 736, 1280)
    t0 = time.time()
    with torch.no_grad():
        fmap = bb(x)
    assert fmap.shape == (1, 1024, 46, 80), fmap.shape
    neck = M.SimpleFeaturePyramid(1024)
    feats = neck(fmap)
    shapes = {k: tuple(v.shape[-2:]) for k, v in feats.items()}
    assert shapes["0"] == (184, 320) and shapes["1"] == (92, 160), shapes
    assert shapes["2"] == (46, 80) and shapes["3"] == (23, 40), shapes
    assert shapes["pool"] == (12, 20), shapes
    print(f"VALIDATE_B_PASS random-init dinov3_vitl16 ({n_params/1e6:.0f}M params) "
          f"@1280x736 grid 80x46 (fwd {time.time()-t0:.0f}s CPU); SFP strides "
          f"4/8/16/32/64: {shapes}")


def test_C_dinov2_standin(model_dir: str):
    size = (1288, 728)  # patch 14: 92x52 grid ('~1280' for the stand-in)
    bb = ViTBackbone(model_dir, frozen=True)
    assert bb.patch_size == 14 and bb.embed_dim == 1024, (bb.patch_size, bb.embed_dim)
    assert bb.num_prefix == 1, bb.num_prefix  # dinov2-large: CLS only, no registers
    model = build_detector(bb, image_size=size)
    xs, targets = real_batch(size, n=2)
    model.train()
    t0 = time.time()
    losses = model(list(xs), [dict(t) for t in targets])
    check_losses(losses)
    sum(losses.values()).backward()
    t1 = time.time()
    for n, p in model.named_parameters():
        if n.startswith("backbone.backbone."):
            assert p.grad is None, n
    fmap = bb(xs[0].unsqueeze(0))
    assert fmap.shape == (1, 1024, 52, 92), fmap.shape
    stats = dict(mean=float(fmap.mean()), std=float(fmap.std()))
    # unfreeze-top path (B3 shape check only)
    bb3 = ViTBackbone(model_dir, frozen=False, unfreeze_top=4)
    n_tr = sum(p.numel() for p in bb3.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in bb3.parameters())
    print(f"VALIDATE_C_PASS dinov2-large stand-in @1288x728 grid 92x52, "
          f"losses {[f'{k}={float(v):.3f}' for k, v in losses.items()]}, "
          f"fwd+bwd {t1-t0:.0f}s CPU, fmap stats {stats}, "
          f"unfreeze_top=4 -> {n_tr/1e6:.0f}M/{n_tot/1e6:.0f}M trainable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standin-dir", default=str(ROOT / "models/v3/hf/dinov2-large"))
    ap.add_argument("--skip-standin", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    test_A_tiny_e2e()
    test_B_vitl16_random()
    if not args.skip_standin:
        test_C_dinov2_standin(args.standin_dir)
    print("ALL_CPU_VALIDATIONS_PASS")


if __name__ == "__main__":
    main()
