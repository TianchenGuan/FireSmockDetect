"""Build data/v2/sft/grpo_probe_32.jsonl: 32 prompts for the GRPO e2e
feasibility probe (iteration-2 brief section 6).

Deterministic (seed 0) sample from data/v2/sft/train.jsonl:
  16 positives (8 tiny + 8 non-tiny) + 16 negatives.
gt_boxes are re-derived from the SFT answer JSON (integer thousandths of
the model-input frame -> normalized 0-1), so they are already in the same
frame as the (optionally cropped) image, matching what the model sees.

Run: .venv/bin/python scripts/build_grpo_probe.py
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/v2/sft/train.jsonl"
OUT = ROOT / "data/v2/sft/grpo_probe_32.jsonl"

rows = [json.loads(l) for l in open(SRC)]
pos_tiny = [r for r in rows if r["n_boxes"] > 0 and r["tiny"]]
pos_rest = [r for r in rows if r["n_boxes"] > 0 and not r["tiny"]]
neg = [r for r in rows if r["n_boxes"] == 0]

rng = random.Random(0)
sample = (rng.sample(pos_tiny, 8) + rng.sample(pos_rest, 8)
          + rng.sample(neg, 16))

out = []
for r in sample:
    ans = json.loads(r["answer"])
    gt = [[c / 1000.0 for c in b] for b in ans["boxes"]]
    assert len(gt) == r["n_boxes"]
    for b in gt:
        assert 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1, (r["id"], b)
    out.append({"id": r["id"], "image_path": r["image_path"],
                "crop_box": r.get("crop_box"),
                "instruction": r["instruction"],
                "gt_boxes": gt, "n_boxes": r["n_boxes"],
                "tiny": r["tiny"], "source": r["source"]})

rng.shuffle(out)
with open(OUT, "w") as fh:
    for r in out:
        fh.write(json.dumps(r) + "\n")
n_pos = sum(1 for r in out if r["n_boxes"] > 0)
n_tiny = sum(1 for r in out if r["tiny"])
n_crop = sum(1 for r in out if r["crop_box"])
print(f"wrote {len(out)} rows -> {OUT}")
print(f"  positives={n_pos} (tiny={n_tiny}) negatives={len(out)-n_pos} "
      f"cropped={n_crop}")
srcs = {}
for r in out:
    srcs[r["source"]] = srcs.get(r["source"], 0) + 1
print(f"  sources={srcs}")
