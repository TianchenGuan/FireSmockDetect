"""VLM smoke-detection evaluation: generate JSON predictions on a val/test
image list, parse boxes, match vs GT (same IoU=0.4 protocol as
eval_detection.py). Runs inside the vlm container on a GPU node.

Outputs per-image raw predictions (jsonl) + a summary json. Supports
base model (no adapter) and base+LoRA adapter.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image

PROMPT = ("Detect all wildfire smoke in this image. Respond with compact JSON "
          'exactly like {"smoke_present": true, "boxes": [[x1,y1,x2,y2]]} '
          "with coordinates as integers 0-1000 relative to image width and "
          'height, or {"smoke_present": false, "boxes": []} if there is none.')


def parse_answer(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        # fallback for chatty/truncated outputs: accept explicit coordinate
        # arrays mentioned in prose; require 4-int arrays to avoid junk.
        arrs = re.findall(r"\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]", text)
        neg = re.search(r'"?smoke_present"?\s*[:=]?\s*false|\bno (visible )?smoke\b', text, re.I)
        if arrs:
            boxes = []
            for a in arrs:
                x1, y1, x2, y2 = [min(max(float(v), 0), 1000) / 1000 for v in a]
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
            if boxes:
                return {"smoke_present": True, "boxes": boxes, "parse_ok": True,
                        "fallback": True}
        if neg:
            return {"smoke_present": False, "boxes": [], "parse_ok": True,
                    "fallback": True}
        return None
    try:
        d = json.loads(m.group(0))
        boxes = []
        for b in d.get("boxes", []):
            if isinstance(b, list) and len(b) == 4:
                x1, y1, x2, y2 = [min(max(float(v), 0), 1000) / 1000 for v in b]
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
        return {"smoke_present": bool(d.get("smoke_present")), "boxes": boxes,
                "parse_ok": True}
    except Exception:
        return None


def load_gt(img_path, root_map):
    p = img_path
    if root_map:
        src, dst = root_map.split("=", 1)
        if p.startswith(src):
            p = dst + p[len(src):]
    lp = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
    boxes = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            t = line.split()
            if len(t) >= 5:
                cx, cy, w, h = map(float, t[1:5])
                boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2, w*h])
    return p, boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--list", required=True, help="txt file of image paths")
    ap.add_argument("--root-map", default="")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=1048576)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n for multi-GPU sharding")
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--quant", default=None, choices=[None, "4bit"],
                    help="nf4 quantized load (models too big for one L40S)")
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(args.model_path)
    qkw = {}
    if args.quant == "4bit":
        from transformers import BitsAndBytesConfig
        qkw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": args.device_index}, **qkw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    imgs = [l.strip() for l in open(args.list) if l.strip()]
    si, sn = map(int, args.shard.split("/"))
    imgs = imgs[si::sn]
    if args.max_images:
        imgs = imgs[:args.max_images]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_pos = tp_img = fp_img = 0
    box_tp = box_fp = box_fn = 0
    tiny_tp = tiny_n = 0
    lat = []
    with out.open("w") as fh:
        for i, ip in enumerate(imgs):
            path, gt = load_gt(ip, args.root_map)
            img = Image.open(path).convert("RGB")
            w, h = img.size
            if w * h > args.max_pixels:
                s = (args.max_pixels / (w * h)) ** 0.5
                img = img.resize((int(w*s), int(h*s)), Image.BILINEAR)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT}]}]
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
            enc = processor(text=[text], images=[[img]], return_tensors="pt"
                            ).to(model.device)
            t1 = time.time()
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
            lat.append(time.time() - t1)
            ans = processor.tokenizer.decode(
                gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            parsed = parse_answer(ans) or {"smoke_present": None, "boxes": [],
                                           "parse_ok": False}
            rec = {"image": ip, "gt_boxes": gt, "pred": parsed,
                   "raw": ans[:900], "latency_s": round(lat[-1], 3)}
            fh.write(json.dumps(rec) + "\n")
            # image-level + box-level accounting
            has_gt = len(gt) > 0
            pred_pos = bool(parsed["smoke_present"]) or bool(parsed["boxes"])
            n_pos += has_gt
            if has_gt and pred_pos:
                tp_img += 1
            if not has_gt and pred_pos:
                fp_img += 1
            matched = set()
            for pb in parsed["boxes"]:
                best_j, best = -1, 0.0
                for j, g in enumerate(gt):
                    if j in matched:
                        continue
                    v = iou(pb, g[:4])
                    if v > best:
                        best_j, best = j, v
                if best >= 0.4:
                    matched.add(best_j)
                    box_tp += 1
                else:
                    box_fp += 1
            box_fn += len(gt) - len(matched)
            for j, g in enumerate(gt):
                if g[4] < 0.001:
                    tiny_n += 1
                    if j in matched:
                        tiny_tp += 1
            if (i+1) % 50 == 0:
                print(f"{i+1}/{len(imgs)} img_recall={tp_img/max(n_pos,1):.3f}",
                      flush=True)

    n_neg = len(imgs) - n_pos
    summary = {
        "n_images": len(imgs), "n_pos": n_pos, "n_neg": n_neg,
        "img_recall": round(tp_img/max(n_pos, 1), 4),
        "img_fp_rate_neg": round(fp_img/max(n_neg, 1), 4),
        "box_precision": round(box_tp/max(box_tp+box_fp, 1), 4),
        "box_recall": round(box_tp/max(box_tp+box_fn, 1), 4),
        "tiny_recall": round(tiny_tp/max(tiny_n, 1), 4), "tiny_n": tiny_n,
        "parse_fail": sum(1 for _ in open(out) if '"parse_ok": false' in _),
        "mean_latency_s": round(sum(lat)/max(len(lat), 1), 3),
        "adapter": args.adapter or "none", "model": args.model_path,
        "elapsed_s": round(time.time()-t0, 1),
    }
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY " + json.dumps(summary), flush=True)
    print("VLM_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
