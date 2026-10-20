"""Devbox-only helper: build a TINY random-weight qwen3_5 model + LoRA
adapter from the real Qwen3.5-9B config/processor files, to wire-test
grpo_train.py on CPU inside the vlm image before spending cluster attempts.

Usage (inside container):
  python make_tiny_qwen35.py --src /model9b --out /tmp/tiny_model \
      --adapter-out /tmp/tiny_adapter
"""
import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="real 9B model dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter-out", required=True)
    args = ap.parse_args()

    from transformers import (AutoConfig, AutoModelForImageTextToText,
                              AutoProcessor)
    from peft import LoraConfig, get_peft_model

    cfg = AutoConfig.from_pretrained(args.src)
    tc, vc = cfg.text_config, cfg.vision_config
    tc.hidden_size = 64
    tc.num_hidden_layers = 4
    tc.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    tc.head_dim = 32
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    tc.intermediate_size = 128
    tc.linear_key_head_dim = 16
    tc.linear_num_key_heads = 2
    tc.linear_num_value_heads = 4
    tc.linear_value_head_dim = 16
    # rotary dim = head_dim * partial_rotary_factor = 8 -> half = 4
    tc.rope_parameters["mrope_section"] = [2, 1, 1]
    vc.depth = 4
    vc.hidden_size = 64
    vc.num_heads = 4
    vc.intermediate_size = 128
    vc.out_hidden_size = tc.hidden_size

    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(cfg)
    model = model.to(torch.float32)
    n = sum(p.numel() for p in model.parameters())
    print(f"tiny model params: {n/1e6:.1f}M")
    model.save_pretrained(args.out)
    AutoProcessor.from_pretrained(args.src).save_pretrained(args.out)

    # same target-selection rule as src/train/train_vlm.py
    targets = []
    vision_prefixes = ("visual", "vision_tower", "vision_model")
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            if name.startswith(vision_prefixes) or ".visual" in name:
                continue
            if ".experts." in name or ".expert." in name:
                continue
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"):
                targets.append(name)
    print(f"lora targets: {len(targets)}")
    lcfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                      target_modules=targets, bias="none",
                      task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, lcfg)
    peft_model.save_pretrained(args.adapter_out)
    print(json.dumps({"out": args.out, "adapter": args.adapter_out}))


if __name__ == "__main__":
    main()
