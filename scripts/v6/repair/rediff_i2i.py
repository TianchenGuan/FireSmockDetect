"""Recompute i2i construction-diff with a STRUCTURAL metric (v2) and re-derive
decisions. The v1 metric diffed full-resolution pixels; GLM re-renders every
pixel with film-grain texture, so even faithful reconstructions exceeded the
threshold and 98-100% of i2i items were flagged global_restyle. v2 downsamples
to ~128px (grain averages out) before diffing, so only structural/color
changes count. VLM stages are NOT re-run; only i2i_diff and decision change.

Usage:
  .venv/bin/python scripts/v6/repair/rediff_i2i.py \
      --review data/v6/repair/review_pilot_c.jsonl \
      --queue data/v6/repair/q_pilot_c.jsonl \
      --src data/v6/gen/glm_repair_pilot_c \
      --out data/v6/repair/review_pilot_c_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/v6/repair"))
from blind_review import decide, union_box  # noqa: E402


def i2i_diff_v2(gen: Image.Image, bg_path: str):
    try:
        bg = Image.open(bg_path).convert("RGB").resize(gen.size, Image.LANCZOS)
    except Exception as e:
        return {"diff_error": str(e)[:120]}
    w, h = gen.size
    dw, dh = max(32, w // 8), max(32, h // 8)
    a = np.asarray(gen.convert("RGB").resize((dw, dh), Image.BOX), np.float32) / 255
    b = np.asarray(bg.resize((dw, dh), Image.BOX), np.float32) / 255
    d = np.abs(a - b).mean(-1)
    sup = d > 0.08
    ys, xs = np.nonzero(sup)
    box = None
    if len(ys) > 3:
        box = [round(float(xs.min() / dw), 3), round(float(ys.min() / dh), 3),
               round(float(xs.max() / dw), 3), round(float(ys.max() / dh), 3)]
    color_shift = float(np.abs(a.mean((0, 1)) - b.mean((0, 1))).sum())
    return {"metric": "v2_structural_ds8",
            "change_frac": round(float(sup.mean()), 4),
            "mean_change": round(float(d.mean()), 4),
            "p99_change": round(float(np.percentile(d, 99)), 4),
            "diff_box": box,
            "color_shift": round(color_shift, 4),
            "global_restyle": bool((d > 0.06).mean() > 0.45
                                   or color_shift > 0.12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    q = {json.loads(l)["hash"]: json.loads(l)
         for l in (ROOT / args.queue).open()}
    src = ROOT / args.src
    n_re = n_ch = 0
    with (ROOT / args.out).open("w") as f:
        for line in (ROOT / args.review).open():
            r = json.loads(line)
            if not r.get("error") and r.get("route") == "i2i" and \
                    q.get(r["hash"], {}).get("bg_src_path"):
                gen = Image.open(src / f"{r['hash']}.jpg")
                diff = i2i_diff_v2(gen, q[r["hash"]]["bg_src_path"])
                old = r.get("decision")
                r["i2i_diff"] = diff
                r["weak_box_union"] = union_box(
                    (r.get("stage1_blind") or {}).get("boxes")) or diff.get("diff_box")
                r["decision"] = decide(r.get("kind"), r.get("stage1_blind"),
                                       r.get("stage2_crop"), diff)
                n_re += 1
                n_ch += old != r["decision"]
            f.write(json.dumps(r) + "\n")
    print(f"recomputed {n_re} i2i records, {n_ch} decisions changed -> {args.out}")


if __name__ == "__main__":
    main()
