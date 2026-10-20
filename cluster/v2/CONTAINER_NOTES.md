# smoke-detect-vlm — VLM training containers (Qwen3.5, green g6e / 8x L40S)

Tags: **v1** = base SFT stack (below). **v2** = v1 + bitsandbytes 0.50.2
(QLoRA for 27B+; Dockerfile.vlm2). **v3** = v2 + trl 1.13.0
(GRPO; Dockerfile.vlm3; digest sha256:e2cfd6535706b5b5172105beaea81ab53
770af53ef137c75bc9223e3a553e4cd, pushed 2026-09-12). trl 1.13.0 deps all
satisfied by the pinned stack (transformers>=4.56.2, datasets>=4.7.0,
accelerate>=1.4.0, peft>=0.13.0); GRPOTrainer has native VLM support and
its VLM path requires transformers>=5.0 (we ship 5.17.0).

# smoke-detect-vlm:v1 — VLM training container (Qwen3.5 SFT, green g6e / 8x L40S)

Built + verified 2026-09-11 on devbox (CPU-only). Dockerfile: `cluster/v2/Dockerfile.vlm`.

## Image

- URI: `000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-detect-vlm:v1`
- Digest: `sha256:fa2198be618a25d68e788310f3dc7ffba81109bf0d92c552d9105bb782483c25`
  (pushed 2026-09-11T03:08 PDT; local image id `ddb27e26dde1`, 9.24 GB local /
  4.7 GB compressed in ECR)
- Base: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
  @ `sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385`
  (same verified base as iteration-1 `cluster/Dockerfile`; Python 3.11.13 —
  NOT 3.12 — torch 2.8.0+cu128, torchvision 0.23.0+cu128, cxx11abi TRUE, no nvcc).
- ECR repo `smoke-detect-vlm` created in us-east-1 with the exact repo policy of
  the existing `smoke-detect` repo (Sid `LeviathanPull`: BatchGetImage /
  GetDownloadUrlForLayer / BatchCheckLayerAvailability for
  `arn:aws:iam::000000000006:root` and `arn:aws:iam::000000000005:root`);
  policy applied + read back verified.
- `ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` baked (pods have no internet).
- apt: `curl ca-certificates git` (base ships NO curl — presigned uploads
  silently no-op without it; iteration-1 lesson).
- Model weights NOT baked (19–250 GB) — staged separately to /scratch.

## Pinned python stack (all verified in-image)

| package        | pin        | note |
|----------------|------------|------|
| transformers   | 5.17.0     | see deviation below — 4.57.6 does NOT support qwen3_5 |
| accelerate     | 1.15.0     | |
| peft           | 0.20.0     | |
| deepspeed      | 0.19.6     | |
| datasets       | 5.0.1      | |
| pillow         | 12.3.0     | |
| pandas         | 3.0.5      | |
| pyarrow        | 25.0.1     | |
| uvloop         | 0.22.1     | |
| flash-attn     | 2.8.3.post1| prebuilt wheel, see below |
| torch          | 2.8.0+cu128| from base image |
| torchvision    | 0.23.0+cu128| from base image |

## DEVIATION: transformers 4.57.6 → 5.17.0

The requested pin transformers==4.57.6 is **broken for Qwen3.5**: the v4.57.6
source tree has no `models/qwen3_5` (only qwen3, qwen3_moe, qwen3_next,
qwen3_vl, qwen3_vl_moe). Verified against GitHub tags: `qwen3_5` (and
`qwen3_5_moe`, needed later for 35B-A3B / 122B-A10B) first appear in
**v5.2.0**. The Qwen/Qwen3.5-9B model card says "The latest `transformers`
is required for Qwen3.5" → pinned latest stable **5.17.0** (requires
torch>=2.5, python>=3.10 — both satisfied). STATUS.md's earlier
"needs transformers>=4.57" note was wrong (likely confused with Qwen3-VL).

Transformers 5.x API fallout found + FIXED in `src/train/train_vlm.py`
(fetched per-job via presigned GET → no image rebuild needed):
- `TrainingArguments(warmup_ratio=0.03)` → removed in 5.x; replaced with
  `warmup_steps=0.03` (float in [0,1) = ratio of total steps, same semantics).
- `TrainingArguments(save_safetensors=True)` → removed (safetensors is the
  only format now); dropped.
- `from_pretrained(torch_dtype=...)` still ACCEPTED in 5.17.0 (functionally
  verified: loads bf16) — left as-is.
- Full kwarg surface of train_vlm.py's TrainingArguments/LoraConfig
  re-validated in-image after the fixes: OK.

## qwen-vl-utils: NOT installed (deliberate)

Package exists on PyPI (0.0.14) but the Qwen3.5-9B model card (rev c2022362)
never mentions or uses it (usage is via processor/chat-template directly, or
serving engines), and nothing in src/ or cluster/ imports `qwen_vl_utils`.
Processor path verified end-to-end without it (see below).

## Local verification (CPU devbox, offline env, no weights) — ALL PASS

Config/processor/tokenizer files (NOT weights) of `Qwen/Qwen3.5-9B` at pinned
revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (= current `.sha` from the
HF API) mounted read-only at /model:

1. imports + versions: transformers/accelerate/peft/deepspeed/datasets/etc.
   all import; versions match pins; offline env vars active in image.
2. `AutoConfig.from_pretrained` → model_type `qwen3_5`,
   architectures `["Qwen3_5ForConditionalGeneration"]`, 32 text layers.
3. `AutoProcessor.from_pretrained` → `Qwen3VLProcessor`
   (Qwen2Tokenizer + Qwen2VLImageProcessor); exercised end-to-end:
   chat template + real 448x448 PIL image → input_ids [1,217],
   pixel_values [784,1536], incl. image_grid_thw / mm_token_type_ids.
4. `AutoModelForImageTextToText.from_config` on meta device →
   `Qwen3_5ForConditionalGeneration`, 9,409,813,744 params (matches "9B").
5. flash_attn 2.8.3.post1 imports cleanly (incl. `flash_attn_func`).
6. cv2 NOT present, and nothing in this stack pulls opencv (GUI-cv2/libxcb
   issue moot; if a later layer ever needs cv2, use opencv-python-headless).
7. curl 7.81.0 + git 2.34.1 present.

## flash-attn decision: INCLUDED (prebuilt wheel)

- NOT built from source. Installed Dao-AILab GitHub release wheel
  `flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl`
  — exact match for the base image (cp311, torch 2.8, cu12, cxx11abi TRUE
  verified via `torch._C._GLIBCXX_USE_CXX11_ABI`). Import verified.
- L40S = sm_89 (Ada), supported by FA2 kernels.
- NOTE: the task brief said cp312; the base image is actually Python 3.11.
- train_vlm.py currently passes `attn_implementation="sdpa"` — that keeps
  working; `flash_attention_2` is now available as an option for the 8
  full-attention layers.

## Known limitations / follow-ups (not blockers)

- **Hybrid linear attention**: 24/32 text layers are `linear_attention`
  (gated-deltanet style). `fla` / `causal-conv1d` fast kernels are NOT
  installed; transformers falls back to the torch-native path (no warnings at
  instantiation; expect slower linear-attn layers). Revisit only if measured
  throughput is poor — prebuilt-wheel availability for cu12/torch2.8/cp311
  must be re-checked before adding.
- **LoRA coverage**: linear-attention layers use leaf names `in_proj_qkv`,
  `in_proj_a`, `in_proj_b`, `in_proj_z`, `out_proj` — train_vlm.py's current
  target list (q/k/v/o_proj + MLP) covers 128/249 text linears: all 32 MLPs
  but attention only in the 8 full-attention layers. If more adaptation
  capacity is wanted, add `in_proj_qkv`/`out_proj` (+optionally
  `in_proj_a/b/z`) to the target leaf set. Deliberately NOT changed here
  (training-behavior decision, flagged for the training owner).
- **No nvcc in image** (runtime base): deepspeed ops cannot JIT-compile on
  pods. Current zero2.json/zero3.json define no deepspeed optimizer → HF
  Trainer uses torch AdamW (no compiled ops needed). ZeRO-offload/CPUAdam
  (122B plan) would require a -devel base or prebuilt deepspeed ops — flag
  for the QLoRA/offload prep.
- Vision tower linears are `qkv`/`proj`/`linear_fc1`/`linear_fc2` —
  `--train-vision` would currently match nothing (vision stays frozen either
  way by default); also flagged for the training owner.
