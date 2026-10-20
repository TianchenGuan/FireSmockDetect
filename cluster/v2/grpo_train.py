"""GRPO e2e feasibility probe for Qwen3.5-9B + SFT LoRA adapter
(iteration-2 brief section 6). Runs inside smoke-detect-vlm:v3 (adds
trl==1.13.0) on one green g6e node, pure DDP via torchrun.

Design (validated against trl 1.13.0 source):
- dataset rows: {"prompt": [{"role":"user","content": <str>}],
  "image": PIL, "gt_boxes": [[x1..]0-1], "row_id"} — string content;
  GRPOTrainer inserts the image block via prepare_multimodal_messages.
- model: base bf16 + PeftModel.from_pretrained(is_trainable=True) — GRPO
  trains the existing SFT adapter in place (beta=0 -> no ref model).
- reward: grpo_reward.reward_fn (verifiable JSON/box reward, clip [-2,2]);
  per-completion records appended to $GRPO_REWARD_LOG.rank<r>.jsonl.

Example (pod):
  torchrun --nproc-per-node 8 grpo_train.py --model-path /scratch/.../qwen3.5-9b \
    --adapter /tmp/job/adapter/final_adapter --data /tmp/job/probe.jsonl \
    --image-root-map "/local/home/tianrgua/cvproject/data/yolo=/scratch/scratch/tianrgua-sandbox/smoke_data_v2" \
    --output-dir /scratch/.../runs_v2/grpo_e2e --max-steps 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent))  # grpo_reward.py
from grpo_reward import reward_fn  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter", required=True,
                    help="dir containing the SFT LoRA adapter")
    ap.add_argument("--data", required=True, help="probe jsonl")
    ap.add_argument("--image-root-map", default="", help="src=dst rewrite")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--per-device-bs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--max-pixels", type=int, default=1048576)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu-test", action="store_true",
                    help="devbox wiring test: fp32, use_cpu, no grad ckpt")
    return ap.parse_args()


def load_probe(jsonl_path, root_map, max_pixels):
    """Materialize the probe as an HF dataset with pre-cropped/resized
    images (normalized boxes are resize-invariant; crop-frame boxes are
    already crop-relative in the probe file)."""
    from datasets import Dataset, Features, Image, Value
    rows = [json.loads(l) for l in open(jsonl_path)]
    out = []
    for r in rows:
        p = r["image_path"]
        if root_map:
            src, dst = root_map.split("=", 1)
            if p.startswith(src):
                p = dst + p[len(src):]
        img = PILImage.open(p).convert("RGB")
        if r.get("crop_box"):
            img = img.crop(tuple(r["crop_box"]))
        w, h = img.size
        if w * h > max_pixels:
            s = (max_pixels / (w * h)) ** 0.5
            img = img.resize((max(28, int(w * s)), max(28, int(h * s))),
                             PILImage.BILINEAR)
        out.append({"prompt": [{"role": "user", "content": r["instruction"]}],
                    "image": img, "gt_boxes": r["gt_boxes"],
                    "row_id": r["id"]})
    feats = Features({"prompt": [{"role": Value("string"),
                                  "content": Value("string")}],
                      "image": Image(),
                      "gt_boxes": [[Value("float64")]],
                      "row_id": Value("string")})
    return Dataset.from_list(out, features=feats)


class StepLogger:
    """TrainerCallback: one jsonl line per log event + wall time per step."""

    def __init__(self, path):
        self.path = path
        self.t_last = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        now = time.time()
        rec = {"step": state.global_step, "t": round(now, 2),
               "dt_s": round(now - self.t_last, 2), **(logs or {})}
        self.t_last = now
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def main():
    args = parse_args()
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              TrainerCallback)
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer

    rank = int(os.environ.get("RANK", "0"))
    t0 = time.time()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    # per-rank reward log (reward_fn runs on every rank over its local rows)
    base_log = os.environ.get("GRPO_REWARD_LOG",
                              str(outdir / "reward_log.jsonl"))
    os.environ["GRPO_REWARD_LOG"] = f"{base_log}.rank{rank}"

    try:
        processor = AutoProcessor.from_pretrained(
            args.model_path, max_pixels=args.max_pixels,
            padding_side="left", truncation_side="left")
    except TypeError:
        processor = AutoProcessor.from_pretrained(
            args.model_path, padding_side="left", truncation_side="left")
    if hasattr(getattr(processor, "image_processor", None), "max_pixels"):
        processor.image_processor.max_pixels = args.max_pixels

    dtype = torch.float32 if args.cpu_test else torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=dtype,
        attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    # SFT lesson: gradient checkpointing is inert under PEFT without this
    model.enable_input_require_grads()
    if rank == 0:
        trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"PARAMS trainable={trn}", flush=True)

    ds = load_probe(args.data, args.image_root_map, args.max_pixels)
    if rank == 0:
        n_pos = sum(1 for r in ds if r["gt_boxes"])
        print(f"DATASET n={len(ds)} pos={n_pos}", flush=True)

    cfg = GRPOConfig(
        output_dir=str(outdir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        temperature=args.temperature,
        beta=args.beta,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        bf16=not args.cpu_test,
        use_cpu=args.cpu_test,
        gradient_checkpointing=not args.cpu_test,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    class _CB(TrainerCallback):
        def __init__(self, path):
            self.inner = StepLogger(path)

        def on_log(self, args, state, control, logs=None, **kw):
            if state.is_world_process_zero:
                self.inner.on_log(args, state, control, logs=logs, **kw)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=ds,
        processing_class=processor,
        callbacks=[_CB(str(outdir / "trainer_log.jsonl"))],
    )
    trainer.train()
    trainer.save_model(str(outdir / "final_adapter"))
    if rank == 0:
        elapsed = time.time() - t0
        print(f"GRPO_TRAIN_DONE steps={args.max_steps} "
              f"elapsed={elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
