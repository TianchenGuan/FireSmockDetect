"""Markdown tables for SEGMENTATION_V5B.md from fetched artifacts.
  python scripts/v5b/seg_report_tables.py --run segV5b_a --seg 1"""
import argparse, json, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--seg", type=int, default=1)
ap.add_argument("--cfgs", default="lora,frozen,r50"); a = ap.parse_args()
rd = ROOT / "models/v5b/seg" / a.run / f"seg{a.seg:02d}"
cfgs = a.cfgs.split(",")
NAMES = {"lora": "DINOv3-L + LoRA r16 + SFP + FPN-lite head", "frozen": "DINOv3-L frozen + SFP + FPN-lite head",
         "r50": "ResNet-50 (ImageNet, frozen) + FPN-lite/U-Net-style head (baseline)"}
S = {}
for c in cfgs:
    f = rd / c / "val_metrics_best.json"
    if f.exists():
        S[c] = json.loads(f.read_text())
def g(c, k, fmt="{:.3f}"):
    v = S.get(c, {}).get("summary", {}).get(k)
    return fmt.format(v) if isinstance(v, (int, float)) else "-"
print("### Headline (heinola VAL, 382 frames: 329 SAM-mask positives + 7 manual-mask positives + 46 author negatives; thr 0.5; metrics on valid letterboxed pixels at stride 4)\n")
print("| config | best ep/step | smoke IoU (all) | Dice | precision | recall | mIoU pos frames | IoU vs SAM masks (329) | IoU vs manual masks (7) | mIoU manual (7) | neg FP pixel frac | neg frame FA (any comp) | neg frame FA (comp>=1%) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in cfgs:
    if c not in S: continue
    print(f"| {NAMES[c]} | {S[c].get('best_step')} | {g(c,'iou')} | {g(c,'dice')} | {g(c,'precision')} | {g(c,'recall')} | {g(c,'miou_pos')} | {g(c,'iou_sam_model_assisted')} | {g(c,'iou_manual_human')} | {g(c,'miou_manual_human')} | {g(c,'neg_fp_pixel_frac','{:.4f}')} | {g(c,'neg_frame_false_alarm')} | {g(c,'neg_frame_fa_1pct')} |")
print("\n### Mask -> box recall vs the authors' boxes (336 boxed heinola frames; A: any component bbox IoU>=0.4; B: union bbox IoU>=0.4; C: >=30% of predicted pixels inside the GT box and >=10% of the box covered)\n")
print("| config | A comp IoU>=0.4 | B union IoU>=0.4 | C hit | n boxes | GT-mask reference A / B / C |")
print("|---|---|---|---|---|---|")
for c in cfgs:
    if c not in S: continue
    print(f"| {NAMES[c]} | {g(c,'box_recall_A_iou04')} | {g(c,'box_recall_B_union_iou04')} | {g(c,'box_recall_C_hit')} | {g(c,'box_recall_n_boxes','{}')} | {g(c,'box_recall_gtmask_ref_A_iou04')} / {g(c,'box_recall_gtmask_ref_B_union_iou04')} / {g(c,'box_recall_gtmask_ref_C_hit')} |")
print("\n### Parameters / training\n")
print("| config | backbone trainable / total | neck | head | all trainable | steps | s/it (last) | epochs evaluated |")
print("|---|---|---|---|---|---|---|---|")
for c in cfgs:
    pf = rd / c / "param_counts.json"
    if not pf.exists(): continue
    p = json.loads(pf.read_text())
    ev = steps = 0; spi = "-"
    mf = rd / c / "metrics.jsonl"
    if mf.exists():
        recs = [json.loads(l) for l in mf.read_text().splitlines()]
        ev = sum("eval" in r for r in recs); tr = [r for r in recs if "loss" in r]
        if tr: steps = tr[-1]["step"]; spi = f"{tr[-1]['s_per_it']:.2f}"
    bb = p.get("backbone", {"trainable": 0, "total": 0})
    print(f"| {NAMES[c]} | {bb['trainable']:,} / {bb['total']:,} | {p.get('neck',{}).get('total','-') if isinstance(p.get('neck',{}).get('total','-'),str) else format(p['neck']['total'],',')} | {p['head']['total']:,} | {p['all']['trainable']:,} | {steps} | {spi} | {ev} |")
print("\n### Per-epoch heinola VAL IoU\n")
print("| config | " + " | ".join(f"ep{i}" for i in range(1, 11)) + " |")
print("|---|" + "---|" * 10)
for c in cfgs:
    mf = rd / c / "metrics.jsonl"
    if not mf.exists(): continue
    ev = {r["epoch"]: r["eval"]["iou"] for r in map(json.loads, mf.read_text().splitlines()) if "eval" in r}
    print(f"| {c} | " + " | ".join(f"{ev[i]:.3f}" if i in ev else "-" for i in range(1, 11)) + " |")
print("\n### Per-flight IoU on heinola (main configs; positives only, mean per-image IoU)\n")
pf = {}
for c in cfgs:
    f = rd / c / "val_per_image.json"
    if f.exists():
        pf[c] = json.loads(f.read_text())
fl = sorted({r["flight"] for v in pf.values() for r in v})
print("| flight | n pos | n neg | " + " | ".join(f"{c} mIoU" for c in pf) + " | " + " | ".join(f"{c} neg FA" for c in pf) + " |")
print("|---|---|---|" + "---|" * (2 * len(pf)))
for f_ in fl:
    cells = []; negs = []
    for c, v in pf.items():
        pos = [r["iou"] for r in v if r["flight"] == f_ and not r["is_negative"]]
        neg = [r["n_comp"] > 0 for r in v if r["flight"] == f_ and r["is_negative"]]
        cells.append(f"{np.mean(pos):.3f}" if pos else "-"); negs.append(f"{np.mean(neg):.2f}" if neg else "-")
        npos, nneg = len(pos), len(neg)
    print(f"| {f_} | {npos} | {nneg} | " + " | ".join(cells) + " | " + " | ".join(negs) + " |")
