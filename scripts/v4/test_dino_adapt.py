"""Iteration-4 visual-adaptation tests for the DINOv3 detection stack (CPU).

Verifies, per the v4 spec:
  (a) LoRA injection touches ONLY attention q/k/v/o projections, on both HF
      layouts (DINOv3ViTModel: model.layer.N.attention.{q,k,v,o}_proj;
      Dinov2Model: encoder.layer.N.attention.attention.{query,key,value} +
      attention.output.dense), with exact param counts.
  (b) one forward+backward step per mode (frozen / lora r16 / unfreeze-top 4 /
      unfreeze-all) on a tiny input (320x192 for patch-16; 322x196 for the
      patch-14 dinov2-large stand-in) with 2 synthetic boxes; gradients land
      exactly where each mode says they should. Plus --grad-ckpt variants.
  (c) checkpoint save/load round-trip preserves LoRA weights bit-exact via the
      SAME trainable_state_dict/load_trainable_state_dict path train_det.py
      uses, and a full state_dict() strict round-trip.
  (d) exact trainable-vs-total per-group counts for the REAL local snapshots
      (dinov3-vitl16, dinov2-large) in all four modes — numbers feed
      cluster/v4/DINO_ADAPT_NOTES.md.

Run: source .venv/bin/activate && python scripts/v4/test_dino_adapt.py
Exit code 0 iff every check passed.
"""
from __future__ import annotations

import gc
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v3"))
from det_modeling import (LoRALinear, ViTBackbone, build_detector,  # noqa: E402
                          load_trainable_state_dict, param_group_counts,
                          trainable_state_dict)

VITL_DIR = ROOT / "models/v3/hf/dinov3-vitl16"
D2L_DIR = ROOT / "models/v3/hf/dinov2-large"
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
          flush=True)


def make_tiny_snapshot(layout: str, dst: str) -> str:
    """6-block random-init tiny model saved as an HF snapshot dir, so the
    tests exercise the same AutoModel.from_pretrained path as real runs.
    6 blocks so unfreeze-top 4 is a strict subset."""
    torch.manual_seed(0)
    if layout == "dinov3":
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        cfg = DINOv3ViTConfig(hidden_size=64, num_hidden_layers=6,
                              num_attention_heads=2, intermediate_size=128,
                              use_gated_mlp=False, num_register_tokens=4,
                              patch_size=16, query_bias=True, key_bias=False,
                              value_bias=True)
        m = DINOv3ViTModel(cfg)
    else:
        from transformers import Dinov2Config, Dinov2Model
        cfg = Dinov2Config(hidden_size=64, num_hidden_layers=6,
                           num_attention_heads=2, patch_size=14)
        m = Dinov2Model(cfg)
    m.save_pretrained(dst)
    return dst


def build(model_dir, mode, grad_ckpt=False, r=16, size=(320, 192)):
    kw = dict(frozen=True, grad_ckpt=grad_ckpt)
    if mode == "lora":
        kw.update(lora_r=r)
    elif mode == "top4":
        kw.update(frozen=False, unfreeze_top=4)
    elif mode == "all":
        kw.update(unfreeze_all=True)
    bb = ViTBackbone(str(model_dir), **kw)
    return build_detector(bb, image_size=size), bb


def fwd_bwd(model, w, h, seed=0):
    torch.manual_seed(seed)
    model.train()
    x = torch.randn(3, h, w)
    t = {"boxes": torch.tensor([[40.0, 30.0, 120.0, 100.0],
                                [200.0, 80.0, 300.0, 170.0]]),
         "labels": torch.ones(2, dtype=torch.int64)}
    losses = model([x], [t])
    loss = sum(losses.values())
    loss.backward()
    return float(loss.detach())


def grad_audit(model, mode, bb, r=16):
    """Every trainable param must have received a grad; every frozen param
    must not. Trainable backbone set must match the mode's contract."""
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    leaked = [n for n, p in model.named_parameters()
              if not p.requires_grad and p.grad is not None]
    bb_train = {n: p for n, p in bb.named_parameters() if p.requires_grad}
    ok, why = True, ""
    if mode == "frozen":
        ok = not bb_train
        why = f"{len(bb_train)} backbone tensors trainable (want 0)"
    elif mode == "lora":
        names_ok = all(n.endswith(("lora_A", "lora_B")) for n in bb_train)
        n_mods = len(bb.lora_modules)
        expect = sum(r * (m.base.in_features + m.base.out_features)
                     for m in bb.hf.modules() if isinstance(m, LoRALinear))
        got = sum(p.numel() for p in bb_train.values())
        # lora_B is zero-init so dL/dA == 0 at step0, but the grad TENSOR must
        # exist; lora_B grads must be nonzero (real signal flows).
        b_norm = sum(float(p.grad.abs().sum()) for n, p in bb_train.items()
                     if n.endswith("lora_B") and p.grad is not None)
        ok = (names_ok and len(bb_train) == 2 * n_mods and got == expect
              and b_norm > 0)
        why = (f"names_ok={names_ok} tensors={len(bb_train)}/{2*n_mods} "
               f"numel={got}/{expect} sum|grad_B|={b_norm:.3e}")
    elif mode == "top4":
        blocks = bb._blocks()
        expect = sum(p.numel() for b in blocks[-4:] for p in b.parameters())
        for attr in ("layernorm", "norm"):
            m = getattr(bb.hf, attr, None)
            if m is not None:
                expect += sum(p.numel() for p in m.parameters())
        got = sum(p.numel() for p in bb_train.values())
        ok = got == expect
        why = f"trainable bb numel {got} vs blocks[-4:]+finalnorm {expect}"
    elif mode == "all":
        # everything trains EXCEPT mask_token (unused in fwd -> kept frozen
        # so DDP never sees an unused parameter)
        bad = [n for n, p in bb.named_parameters()
               if p.requires_grad != (not n.endswith("mask_token"))]
        ok = not bad
        why = f"wrong requires_grad on {bad[:3]}" if bad else "mask_token frozen, rest trains"
    return ok and not missing and not leaked, \
        why + (f" | grad-missing={missing[:3]}" if missing else "") \
            + (f" | grad-on-frozen={leaked[:3]}" if leaked else "")


def lora_structure_test(tag, model_dir, n_layers, dim):
    """(a) injection changes only intended modules + exact counts."""
    bb = ViTBackbone(str(model_dir), lora_r=16)
    n_expect = n_layers * 4
    check(f"{tag}: lora replaced module count == {n_expect}",
          len(bb.lora_modules) == n_expect, f"got {len(bb.lora_modules)}")
    wrapped = {n for n, m in bb.hf.named_modules() if isinstance(m, LoRALinear)}
    check(f"{tag}: replaced names are attention q/k/v/o only",
          wrapped == set(bb.lora_modules)
          and all(".attention" in n for n in wrapped),
          f"e.g. {sorted(wrapped)[:2]}")
    # no MLP / embedding / norm linears were touched
    untouched = [n for n, m in bb.hf.named_modules() if isinstance(m, nn.Linear)
                 and not n.endswith(".base")]
    check(f"{tag}: mlp/other linears untouched",
          all(("mlp" in n) or ("fc" in n) for n in untouched),
          f"{len(untouched)} plain Linears left, e.g. {untouched[:2]}")
    train = {n: p for n, p in bb.named_parameters() if p.requires_grad}
    expect_numel = n_layers * 4 * 16 * (dim + dim)
    check(f"{tag}: trainable == lora_A/B only, numel == {expect_numel}",
          all(n.endswith(("lora_A", "lora_B")) for n in train)
          and sum(p.numel() for p in train.values()) == expect_numel,
          f"got {sum(p.numel() for p in train.values())} in {len(train)} tensors")
    check(f"{tag}: default alpha == 2*r", bb.lora_alpha == 32.0)
    # zero-init B -> wrapped layer output identical to base at step 0
    some = next(m for m in bb.hf.modules() if isinstance(m, LoRALinear))
    xin = torch.randn(2, some.base.in_features)
    check(f"{tag}: LoRA output == base output at init (B zero-init)",
          torch.equal(some(xin), some.base(xin)))
    del bb
    gc.collect()


def mode_matrix_test(tag, model_dir, w, h):
    """(b) fwd+bwd in each mode with grad audit."""
    for mode in ("frozen", "lora", "top4", "all"):
        t0 = time.time()
        model, bb = build(model_dir, mode, size=(w, h))
        loss = fwd_bwd(model, w, h)
        ok, why = grad_audit(model, mode, bb)
        check(f"{tag}: fwd+bwd [{mode}] loss finite + grads correct",
              ok and loss == loss and loss < 1e4,
              f"loss={loss:.4f} {why} {time.time()-t0:.1f}s")
        del model, bb
        gc.collect()


def grad_ckpt_test(tag, model_dir, w, h):
    for mode in ("lora", "all"):
        model, bb = build(model_dir, mode, grad_ckpt=True, size=(w, h))
        on = bool(getattr(bb.hf, "is_gradient_checkpointing", False))
        loss = fwd_bwd(model, w, h)
        ok, why = grad_audit(model, mode, bb)
        check(f"{tag}: --grad-ckpt [{mode}] enabled + grads survive ckpt",
              on and ok and loss == loss, f"ckpt_on={on} loss={loss:.4f} {why}")
        del model, bb
        gc.collect()


def roundtrip_test(tag, model_dir, w, h, tmp):
    """(c) trainable_state_dict save -> torch.load -> fresh model -> bit-exact
    LoRA weights; plus full state_dict strict round-trip."""
    torch.manual_seed(123)
    m1, bb1 = build(model_dir, "lora", size=(w, h))
    fwd_bwd(m1, w, h)  # populate grads
    with torch.no_grad():  # perturb so lora_B is nonzero before saving
        for p in m1.parameters():
            if p.requires_grad and p.grad is not None:
                p.add_(p.grad, alpha=-0.05)
    ckpt_path = Path(tmp) / f"rt_{tag}.pt"
    torch.save({"model": trainable_state_dict(m1), "step": 1}, ckpt_path)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_lora_keys = sum(1 for k in ck["model"] if "lora_" in k)
    torch.manual_seed(999)  # fresh model gets DIFFERENT random lora_A init
    m2, bb2 = build(model_dir, "lora", size=(w, h))
    pre_equal = all(torch.equal(a, b) for (_, a), (_, b) in zip(
        ((n, p) for n, p in m1.named_parameters() if "lora_" in n),
        ((n, p) for n, p in m2.named_parameters() if "lora_" in n)))
    load_trainable_state_dict(m2, ck["model"])
    d1 = {n: p for n, p in m1.named_parameters()}
    lora_names = [n for n in d1 if "lora_" in n]
    bit_exact = all(torch.equal(d1[n], dict(m2.named_parameters())[n])
                    for n in lora_names)
    check(f"{tag}: ckpt stores LoRA params", n_lora_keys == len(lora_names) > 0,
          f"{n_lora_keys} lora keys in ckpt")
    check(f"{tag}: fresh model differs before load (test is meaningful)",
          not pre_equal)
    check(f"{tag}: torch.load round-trip LoRA weights bit-exact", bit_exact,
          f"{len(lora_names)} tensors compared")
    # identical eval behavior end to end
    m1.eval(); m2.eval()
    torch.manual_seed(7)
    x = torch.randn(3, h, w)
    with torch.no_grad():
        f1 = bb1(x.unsqueeze(0))
        f2 = bb2(x.unsqueeze(0))
    check(f"{tag}: backbone features identical after load",
          torch.equal(f1, f2))
    # full state_dict (base weights + lora) strict round-trip
    torch.manual_seed(31)
    m3, _ = build(model_dir, "lora", size=(w, h))
    m3.load_state_dict(m1.state_dict(), strict=True)
    check(f"{tag}: full state_dict strict load (base+LoRA key layout stable)",
          all(torch.equal(a, b) for (n, a), (_, b) in
              zip(m1.state_dict().items(), m3.state_dict().items())))
    del m1, m2, m3, bb1, bb2
    gc.collect()


def real_counts(tag, model_dir, w, h):
    """(d) exact per-group counts on a real snapshot for the report."""
    print(f"\n--- {tag}: per-mode param counts (real weights, {w}x{h} build) ---")
    rows = {}
    mask_numel = 0
    for mode in ("frozen", "lora", "top4", "all"):
        model, bb = build(model_dir, mode, size=(w, h))
        pc = param_group_counts(model)
        rows[mode] = pc
        if mode == "all":
            mask_numel = sum(p.numel() for n, p in bb.named_parameters()
                             if n.endswith("mask_token"))
        print(f"  mode={mode:7s} " + " | ".join(
            f"{g}: {pc[g]['trainable']:,}/{pc[g]['total']:,}"
            for g in ("backbone", "neck", "head", "all")), flush=True)
        del model, bb
        gc.collect()
    same_total = len({rows[m]["all"]["total"] for m in rows
                      if m != "lora"}) == 1  # lora adds A/B params to total
    check(f"{tag}: totals consistent across modes (lora adds only A/B)",
          same_total and rows["lora"]["all"]["total"]
          - rows["frozen"]["all"]["total"]
          == rows["lora"]["backbone"]["trainable"])
    check(f"{tag}: frozen mode trains no backbone params",
          rows["frozen"]["backbone"]["trainable"] == 0)
    check(f"{tag}: unfreeze-all trains every backbone param except mask_token",
          rows["all"]["backbone"]["trainable"]
          == rows["all"]["backbone"]["total"] - mask_numel,
          f"mask_token numel={mask_numel}")
    return rows


def main():
    torch.set_num_threads(min(32, torch.get_num_threads() or 32))
    with tempfile.TemporaryDirectory() as tmp:
        d3_tiny = make_tiny_snapshot("dinov3", tmp + "/d3tiny")
        d2_tiny = make_tiny_snapshot("dinov2", tmp + "/d2tiny")

        print("\n=== (a) LoRA injection structure ===")
        lora_structure_test("d3tiny(DINOv3ViTModel layout)", d3_tiny, 6, 64)
        lora_structure_test("d2tiny(Dinov2Model layout)", d2_tiny, 6, 64)

        print("\n=== (b) fwd+bwd matrix, tiny models ===")
        mode_matrix_test("d3tiny@320x192", d3_tiny, 320, 192)
        mode_matrix_test("d2tiny@322x196", d2_tiny, 322, 196)  # patch 14 grid

        print("\n=== (b) --grad-ckpt ===")
        grad_ckpt_test("d3tiny@320x192", d3_tiny, 320, 192)
        grad_ckpt_test("d2tiny@322x196", d2_tiny, 322, 196)

        print("\n=== (c) checkpoint round-trip (train_det save path) ===")
        roundtrip_test("d3tiny", d3_tiny, 320, 192, tmp)
        roundtrip_test("d2tiny", d2_tiny, 322, 196, tmp)

        print("\n=== real snapshots ===")
        if D2L_DIR.exists():
            print("\n=== (b) fwd+bwd matrix, dinov2-large stand-in (real weights) ===")
            mode_matrix_test("dinov2-large@322x196", D2L_DIR, 322, 196)
            real_counts("dinov2-large", D2L_DIR, 322, 196)
        else:
            print("dinov2-large snapshot absent — skipped")
        if VITL_DIR.exists():
            rows = real_counts("dinov3-vitl16", VITL_DIR, 320, 192)
            check("dinov3-vitl16: LoRA r16 trainable backbone == 24*4*16*2048",
                  rows["lora"]["backbone"]["trainable"] == 24 * 4 * 16 * 2048,
                  f"got {rows['lora']['backbone']['trainable']}")
        else:
            print("dinov3-vitl16 snapshot absent — skipped")

    n_fail = sum(1 for _, ok in CHECKS if not ok)
    print(f"\n{'='*60}\n{len(CHECKS)} checks, {n_fail} failed -> "
          f"{'ALL PASS' if n_fail == 0 else 'FAILURES'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
