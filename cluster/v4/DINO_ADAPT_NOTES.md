# v4 — DINOv3 backbone visual adaptation (LoRA / top-N / full unfreeze)

Extends the iteration-3 stack in `src/v3/{det_modeling,train_det}.py`
**backward-compatibly**: with none of the new flags, behavior is bit-identical
to v3 (frozen backbone, `--unfreeze-top` unchanged incl. its 0.1x
`--backbone-lr-scale` default and always-on gradient checkpointing).
Verified by `scripts/v4/test_dino_adapt.py` (45/45 checks, CPU, real local
snapshots) plus 2-step trainer smoke runs for every mode.

## New train_det.py flags

| flag | default | meaning |
|---|---|---|
| `--lora R` | 0 (off) | hand-rolled LoRA (no peft — pods are offline, image may lack it) on ViT attention q/k/v/o linears. Base backbone weights frozen; only `lora_A` (kaiming) / `lora_B` (zero-init → identity at step 0) train. Scaling `alpha/R`. |
| `--lora-alpha A` | `2*R` | LoRA scaling numerator. |
| `--unfreeze-all` | off | entire ViT trainable incl. patch/pos embeddings, cls + register tokens. Exception: `mask_token` stays frozen — it never enters our forward pass (`bool_masked_pos` is never passed) and would crash DDP as an unused parameter. |
| `--backbone-lr LR` | None | absolute LR for the backbone param group (LoRA params or unfrozen blocks). When unset, the v3 behavior `--lr * --backbone-lr-scale` (scale default 0.1) applies — kept so existing B3 commands reproduce exactly. Head/neck always use `--lr`. |
| `--grad-ckpt` | off | HF `gradient_checkpointing_enable()` on the backbone, for the `--unfreeze-all` / `--lora` memory case. `--unfreeze-top` keeps its v3 always-on checkpointing regardless. |

`--lora` / `--unfreeze-top` / `--unfreeze-all` are mutually exclusive (hard
error), and all require a real backbone (rejected with `--feat-root`).

Required startup output (iteration report numbers) — rank0 now always prints:

```
param_counts[backbone]: trainable=... total=...
param_counts[neck]:     trainable=... total=...
param_counts[head]:     trainable=... total=...
param_counts[all]:      trainable=... total=...
lr groups: head <lr> (<n> params) | backbone <lr> (<n> params)   # if any bb trainable
```

## Module layout facts (transformers 5.17, discovered by instantiation)

- `DINOv3ViTModel`: blocks at `hf.model.layer.N`, attention linears
  `attention.{q_proj,k_proj,v_proj,o_proj}` (**k_proj has no bias** in ViT-L),
  final norm at `hf.norm`.
- `Dinov2Model` (stand-in): blocks at `hf.encoder.layer.N`, attention linears
  `attention.attention.{query,key,value}` and the o-projection is
  `attention.output.dense`; final norm at `hf.layernorm`.
- Injection matches leaf names within paths containing `.attention` only —
  MLP (`mlp.up_proj/down_proj`, `mlp.fc1/fc2`) and embeddings untouched
  (asserted by the test).
- Checkpoint keys under LoRA become `...q_proj.base.weight` + `...q_proj.lora_A/B`.
  `trainable_state_dict()` (unchanged) stores every LoRA param and skips
  frozen base weights; `load_trainable_state_dict()` round-trips bit-exact
  (test-verified), and a full `state_dict()` strict load also works.

## Exact trainable params — dinov3-vitl16 (24 layers, d=1024), real snapshot

Neck (SimpleFeaturePyramid) = 7,805,184 and head (RPN+RoI) = 14,499,865 are
always fully trainable and identical across modes.

| mode | backbone trainable / total | ALL trainable / total |
|---|---|---|
| frozen head-only (v3 B1) | 0 / 303,129,600 | 22,305,049 / 325,434,649 |
| LoRA r16 (a=32) | 3,145,728 / 306,275,328 | 25,450,777 / 328,580,377 |
| `--unfreeze-top 4` (v3 B3) | 50,391,040 / 303,129,600 | 72,696,089 / 325,434,649 |
| `--unfreeze-all` | 303,128,576 / 303,129,600 | 325,433,625 / 325,434,649 |

LoRA r16 = 24 layers x 4 projections x 16 x (1024+1024) = 3,145,728 exactly
(1.04% of backbone). unfreeze-all leaves exactly 1,024 params frozen (the
unused `mask_token`). dinov2-large stand-in counts (patch 14): same shape,
backbone total 304,368,640; LoRA r16 identical 3,145,728; top-4 50,395,136.

## Memory expectations — L40S (~44 GB usable), ViT-L @ 1280x736

Grid 80x46 = 3,680 patch tokens (+1 cls +4 registers = 3,685). bf16 autocast,
AdamW. v3 reference batch sizes (cluster/v3/launch): **B1 frozen bs 4/GPU**,
**B3 top-4 bs 2/GPU** (ckpt always on), 8x GPU per node. Estimates:

- **frozen (B1)**: backbone fwd under no_grad, no activation storage → bs 4
  as in v3. Unchanged.
- **LoRA r16**: optimizer/grad state is negligible (+3.1M params ≈ 50 MB),
  but activations are stored through ALL 24 blocks (grads must reach every
  lora_A/B), roughly full-backward activation cost ≈ 1.5–2 GB/image bf16.
  Start with **bs 2 (no ckpt)** — B3-comparable; **bs 4 needs `--grad-ckpt`**.
- **top-4 (B3)**: unchanged v3 path, bs 2, ckpt always on.
- **full `--unfreeze-all`**: fp32 params+grads+2 AdamW moments for 325M ≈
  5.2 GB static, plus full activations → **bs 2 + `--grad-ckpt`** (ckpt drops
  activations to ~1 stored tensor/block; compute +~30%). bs 4 likely tight;
  try only after observing headroom.

## Command-line deltas vs v3 launch scripts

`cluster/v3/dino_entry.sh` appends `${EXTRA_ARGS}` verbatim to the
`train_det.py` invocation and `orchestrate_dino.py --extra-args` feeds it, so
**no cluster/v3 changes are needed** — v4 arms are the v3 commands plus
`--extra-args` (code tarball is re-uploaded per segment, so the edited
`src/v3` files ship automatically):

```bash
# B4-lora: B1 command + LoRA r16 (bs 2 without ckpt; use 4 + --grad-ckpt to match B1 throughput)
python cluster/v3/orchestrate_dino.py \
  --run B4_dino3vitl16_lora16_1280 \
  --model-dir /scratch/scratch/tianrgua-sandbox/models/dinov3-vitl16 \
  --width 1280 --height 736 --epochs 12 --bs 2 --lr 1e-4 --warmup 500 \
  --eval-max 1500 --save-steps 400 --workers 6 \
  --extra-args "--lora 16 --backbone-lr 1e-4"

# B5-full: full unfreeze, low backbone LR, checkpointing on
python cluster/v3/orchestrate_dino.py \
  --run B5_dino3vitl16_full_1280 \
  --model-dir /scratch/scratch/tianrgua-sandbox/models/dinov3-vitl16 \
  --width 1280 --height 736 --epochs 12 --bs 2 --lr 1e-4 --warmup 500 \
  --eval-max 1500 --save-steps 400 --workers 6 \
  --extra-args "--unfreeze-all --grad-ckpt --backbone-lr 1e-5"

# B3 stays byte-identical (no --extra-args): --unfreeze-top 4, bs 2,
# backbone LR = 1e-4 * 0.1 via the untouched --backbone-lr-scale default.
# Optionally sweep it now via --extra-args "--backbone-lr 5e-5".
```

Notes: `--backbone-lr` given explicitly in the examples because the unset
default falls back to `0.1 * --lr` (v3 rule) — for LoRA the usual practice is
backbone LR == head LR or higher, hence `1e-4`. Resume across segments works
for all modes (opt state has the same 2 param groups; verified locally).

## Verification

`python scripts/v4/test_dino_adapt.py` (CPU, repo venv) — 45/45 PASS:
LoRA-injection structure on both layouts, fwd+bwd + exact-grad audit in all 4
modes on tiny 6-block snapshots (320x192 / 322x196 patch-14) AND on real
dinov2-large weights, `--grad-ckpt` grad survival, bit-exact ckpt round-trips,
real ViT-L per-mode counts above. Plus `train_det.py` 2-step CPU smokes:
lora / resume-from-lora-ckpt / unfreeze-all+grad-ckpt / default-frozen /
mutual-exclusion error.
