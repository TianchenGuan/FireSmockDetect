"""LoRA SFT for Qwen3.5 (multimodal) on smoke-detection JSON outputs.

Runs inside the smoke-detect-vlm container on 8x L40S. Pods are offline:
model from a staged /scratch dir, images from a staged root, data JSONL
passed alongside. Resumable via HF Trainer checkpoints.

Answer format (fixed): {"smoke_present": bool, "boxes": [[x1,y1,x2,y2],...]}
with coords in integer thousandths of the model-input image (after crop).

Example:
  deepspeed --num_gpus 8 train_vlm.py --model-path /scratch/.../qwen3.5-9b \
    --data train.jsonl --val val_mini.jsonl --image-root-map \
    "/local/home/tianrgua/cvproject/data/yolo=/scratch/scratch/tianrgua-sandbox/smoke_data_v2" \
    --output-dir /scratch/.../runs_v2/trkA_9b_lora --max-steps 3000 \
    --deepspeed zero2.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--image-root-map", default="", help="src=dst path prefix rewrite")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per-device-bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--train-vision", action="store_true",
                    help="also apply LoRA to vision tower (default: vision frozen)")
    ap.add_argument("--max-pixels", type=int, default=1048576, help="image token budget cap")
    ap.add_argument("--min-pixels", type=int, default=200704)
    ap.add_argument("--deepspeed", default=None)
    ap.add_argument("--quant", default="none", choices=["none", "4bit"],
                    help="4bit = QLoRA (NF4 base, bf16 LoRA) — required for 27B+ on L40S where zero3 all-gather over PCIe is ~40x slow")
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--local_rank", type=int, default=-1)  # deepspeed launcher
    return ap.parse_args()


class SmokeSftDataset(Dataset):
    def __init__(self, jsonl_path, processor, root_map, max_pixels, min_pixels):
        self.rows = [json.loads(l) for l in open(jsonl_path)]
        self.processor = processor
        self.root_map = root_map
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

    def __len__(self):
        return len(self.rows)

    def _path(self, p):
        if self.root_map:
            src, dst = self.root_map.split("=", 1)
            if p.startswith(src):
                p = dst + p[len(src):]
        return p

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self._path(r["image_path"])).convert("RGB")
        if r.get("crop_box"):
            img = img.crop(tuple(r["crop_box"]))
        # enforce the pixel budget HERE: the per-image max_pixels hint in the
        # message dict is a qwen-vl-utils convention the bare processor
        # ignores -> a 12MP dfire image otherwise becomes ~16k vision tokens
        # and a single >11GiB allocation. Boxes are normalized, so resizing
        # does not touch labels.
        w, h = img.size
        if w * h > self.max_pixels:
            s = (self.max_pixels / (w * h)) ** 0.5
            img = img.resize((max(28, int(w * s)), max(28, int(h * s))),
                             Image.BILINEAR)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img,
                 "max_pixels": self.max_pixels, "min_pixels": self.min_pixels},
                {"type": "text", "text": r["instruction"]},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": r["answer"]}]},
        ]
        return messages


def make_collator(processor):
    tok = processor.tokenizer

    def collate(batch):
        texts, image_lists = [], []
        prompt_texts = []
        for messages in batch:
            texts.append(processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False))
            prompt_texts.append(processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True))
            imgs = [c["image"] for m in messages for c in m["content"]
                    if isinstance(c, dict) and c.get("type") == "image"]
            image_lists.append(imgs)
        enc = processor(text=texts, images=image_lists, padding=True,
                        return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        # mask everything before the assistant answer (per sample)
        for bi, ptxt in enumerate(prompt_texts):
            plen = len(tok(ptxt, add_special_tokens=False)["input_ids"])
            # image placeholder expansion means token counts differ between
            # raw text and processed ids; recompute via processed prompt enc
        penc = processor(text=prompt_texts, images=image_lists, padding=True,
                         return_tensors="pt")
        for bi in range(labels.size(0)):
            plen = int(penc["attention_mask"][bi].sum())
            # tokens are left-aligned (right padding): mask first plen tokens
            labels[bi, :plen] = -100
        enc["labels"] = labels
        return enc

    return collate


def main():
    args = parse_args()
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model

    t0 = time.time()
    targs = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_steps=0.03,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed,
        save_steps=args.save_steps, save_total_limit=2,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        per_device_eval_batch_size=args.per_device_bs,
        dataloader_num_workers=args.workers,
        remove_unused_columns=False,
        seed=args.seed, report_to=[],
    )
    try:
        processor = AutoProcessor.from_pretrained(
            args.model_path, max_pixels=args.max_pixels,
            min_pixels=args.min_pixels)
    except TypeError:
        processor = AutoProcessor.from_pretrained(args.model_path)
    for attr, val in (("max_pixels", args.max_pixels),
                      ("min_pixels", args.min_pixels)):
        if hasattr(getattr(processor, "image_processor", None), attr):
            setattr(processor.image_processor, attr, val)
    quant_cfg = None
    if args.quant == "4bit":
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", quantization_config=quant_cfg)
    if args.quant == "4bit":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    # LoRA on language linear layers; vision tower frozen unless requested
    target_modules = []
    vision_prefixes = ("visual", "vision_tower", "vision_model")
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            is_vision = name.startswith(vision_prefixes) or ".visual" in name
            if is_vision and not args.train_vision:
                continue
            # MoE: adapting every expert FFN multiplies LoRA params ~64x and
            # blows optimizer memory; adapt attention + shared layers only.
            if ".experts." in name or ".expert." in name:
                continue
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"):
                target_modules.append(name)
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout,
                      target_modules=target_modules, bias="none",
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    # required for gradient checkpointing to actually engage with PEFT
    # (frozen embeddings otherwise stop grad flow and GC saves nothing -> OOM)
    model.enable_input_require_grads()
    if int(os.environ.get("RANK", "0")) == 0:
        tot = sum(p.numel() for p in model.parameters())
        trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        vis_frozen = not args.train_vision
        info = {"total_params": tot, "trainable_params": trn,
                "trainable_pct": round(100 * trn / tot, 3),
                "n_lora_targets": len(target_modules),
                "vision_frozen": vis_frozen, "model_path": args.model_path}
        print("PARAMS " + json.dumps(info), flush=True)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "params.json").write_text(json.dumps(info, indent=2))

    train_ds = SmokeSftDataset(args.data, processor, args.image_root_map,
                               args.max_pixels, args.min_pixels)
    val_ds = (SmokeSftDataset(args.val, processor, args.image_root_map,
                              args.max_pixels, args.min_pixels)
              if args.val else None)

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=make_collator(processor))
    last = None
    ckpts = sorted(Path(args.output_dir).glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        last = str(ckpts[-1])
        print(f"RESUMING from {last}", flush=True)
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(os.path.join(args.output_dir, "final_adapter"))
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"VLM_TRAIN_DONE elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
