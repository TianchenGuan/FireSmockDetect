# GRPO feasibility spike — Qwen3.5-9B + SFT LoRA (iteration-2 §6)

Date: 2026-09-12. Feasibility GATE only — the 500-step experiment was NOT
launched (per brief).

## VERDICT: GO

Ready for a 500-step small experiment, with the prerequisites listed at the
bottom (segment/resume support in the GRPO entry, and two hyperparameter
adjustments). All four gate criteria passed on the FIRST cluster attempt.

| Criterion | Result | Evidence |
|---|---|---|
| 20 optimizer steps without crash | PASS | job `tianrgua-grpoe2ea1-wyyj` SUCCEEDED, `GRPO_TRAIN_DONE steps=20 elapsed=571s`, train rc=0, 0 tracebacks |
| Reward non-constant across groups | PASS | 7 distinct reward levels over 160 completions (hist: -1.5:11, -1:32, 0:5, 0.5:10, 1:70, 1.5:31, 2:1); per-step group reward_std 0–1.34; grad_norm > 0 on 14/20 steps |
| No collapse to always-negative in 20 steps | PASS | pred-negative fraction 0.50 (steps 0–9) → 0.53 (steps 10–19); steps 15/17 were all-positive; parse_ok improved 0.86 → 0.96 |
| Throughput measured | PASS | steady-state **10.7 s/step** (train_runtime 213.5 s / 20 steps, 0.094 steps/s); one-time startup (model+adapter load, first generation) ≈ 354 s; whole job 10 min wall on 1 node |

## Setup that ran

- **Stack**: `smoke-detect-vlm:v3` = v2 + `trl==1.13.0`
  (digest `sha256:e2cfd6535706b5b5172105beaea81ab53770af53ef137c75bc9223e3a553e4cd`,
  us-east-1 ECR). trl 1.13.0 verified compatible with the pinned stack on
  devbox: requires transformers>=4.56.2 (have 5.17.0), datasets>=4.7.0
  (5.0.1), accelerate>=1.4.0 (1.15.0), peft>=0.13.0 (0.20.0).
- **trl multimodal support confirmed from source** (wheel inspected +
  CPU-executed): GRPOTrainer signature columns `["prompt","image","images"]`,
  `prepare_multimodal_messages` injects the PIL image into string-content
  conversational prompts, Qwen-style `image_grid_thw`/`pixel_values`
  splitting in the forward pass; the VLM path explicitly requires
  transformers>=5.0 (we run 5.17.0). No `max_prompt_length` truncation on
  the VLM path.
- **Model**: base `/scratch/scratch/tianrgua-sandbox/models/qwen3.5-9b`
  bf16 + SFT LoRA adapter (S3 `v2/runs/trkA_9b_lora/seg05/final_adapter.tgz`)
  loaded `is_trainable=True` — GRPO trains the SFT adapter in place
  (29.1M trainable params, the SFT LoRA r=16 over 128 targets).
  `beta=0` → no reference model in memory.
- **Config**: 20 steps, group size G=4, gen batch 8 (2 groups/step),
  max_completion_length=128, temperature 1.0, lr 5e-6 constant, bf16,
  grad checkpointing (non-reentrant + `enable_input_require_grads`,
  the iteration-2 OOM lesson), pure DDP torchrun 8×L40S, loss_type=dapo
  (trl default), scale_rewards=group.
- **Data**: `data/v2/sft/grpo_probe_32.jsonl` — seed-0 sample from the SFT
  train set: 16 positives (8 tiny) + 16 negatives, 5 cropped rows,
  sources dfire:14 / pyro-sdis:18. GT boxes re-derived from the SFT answer
  JSON (crop-frame consistent, normalized 0–1).
- **Reward** (`src/train/grpo_reward.py`, uses the eval-time
  `parse_answer` from `src/eval/eval_vlm.py`): +1 correct
  `smoke_present` / −1 wrong or unparseable; +0.5 per GT box matched
  (greedy 1-1, IoU≥0.4); −0.5 per unmatched predicted box; clip [−2,2].
  No verbosity bonus (verbose-but-correct == compact-correct, never more).
  18/18 unit tests pass on devbox (`src/train/test_grpo_reward.py`),
  covering the four required adversarial policies + clipping + IoU
  boundary + trl batch calling convention.

## Measured numbers (job attempt 1)

- 20/20 optimizer steps, 160 completions total, per-step wall (dt between
  log lines): 10.4–11.1 s steady after step 1 (363.9 s incl. startup).
- Per-step mean reward ranged −1.25 … +1.31 (no trend expected at 20
  steps; first-half mean 0.556, second-half 0.381 — flat/noisy, fine for
  a mechanics gate).
- frac_reward_zero_std = 1.0 on 6/20 steps (all-identical rewards within
  both groups → zero advantage → grad_norm 0 those steps). 14/20 steps
  carried gradient signal.
- completions/mean_length 65–119 tokens; clipped_ratio up to 0.62 on some
  steps (128-token cap truncates chatty rollouts; sampled at T=1.0 the
  SFT model often re-grows CoT-style prefixes before the JSON).
- parse_ok: 86% (steps 0–9) → 96% (steps 10–19).
- Artifacts: `models/v2/runs/grpo_e2e/attempt01/{status.json,
  trainer_log.jsonl, reward_log.jsonl, console_tail.txt, final_adapter.tgz}`
  + pod-side `/scratch/scratch/tianrgua-sandbox/smoke/runs_v2/grpo_e2e/attempt_1789205569/`.
- GPU cost of the gate: 1 node × 8 L40S × ~10 min.

## Pre-flight verification done before spending cluster attempts

1. Reward unit tests on devbox (18 asserts, no trl needed).
2. Full GRPO wiring test ON CPU inside the vlm image with a tiny (32.4M)
   random-weight qwen3_5 built from the real 9B config
   (`cluster/v2/make_tiny_qwen35.py`): dataset build → chat-template +
   image injection → generation → reward invocation → GRPO loss/step →
   logs + adapter save all exercised; random model gave constant −1
   reward ⇒ zero advantage ⇒ loss 0, grad 0 (correct GRPO behavior).

## Exact repro

```bash
# reward unit tests (devbox)
.venv/bin/python src/train/test_grpo_reward.py

# probe dataset (deterministic, seed 0)
.venv/bin/python scripts/build_grpo_probe.py

# image (already pushed): cluster/v2/Dockerfile.vlm3
docker build -t 000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-detect-vlm:v3 \
  -f cluster/v2/Dockerfile.vlm3 cluster/v2/ && docker push ...:v3

# CPU wiring test (optional, ~1 min inside image)
# see cluster/v2/make_tiny_qwen35.py docstring + report section above

# the cluster probe (what produced the numbers above)
.venv/bin/python cluster/v2/orchestrate_grpo.py --run grpo_e2e --attempt 1 \
  --model-dir /scratch/scratch/tianrgua-sandbox/models/qwen3.5-9b
```

## Prerequisites / recommendations for the 500-step experiment (NOT launched)

1. **Segmenting/resume**: `grpo_entry.sh` is one-shot. 500 steps ≈
   500 × 10.7 s ≈ 90 min > the ~50 min presigned-URL budget → port the
   vlm_entry segment-resume pattern (GRPOTrainer checkpoints like any HF
   Trainer; save_steps + `resume_from_checkpoint`).
2. **Group size**: 30% of probe steps had zero advantage with G=4 on
   gen-batch 8. Use G=8 (gen batch 8 → 1 group/step) or grow
   steps_per_generation, so fewer steps are wasted.
3. **Completion cap**: clipped_ratio up to 0.62 at 128 tokens; use 192
   (matches eval) or add a small compactness term to the reward if chatty
   rollouts persist.
4. Expected cost at current throughput: 500 steps ≈ 1.5–1.7 h on one
   g6e node (plus ~6 min startup per segment).
