"""Threshold sweep on heinola VAL from the dumped stride-4 probability maps
(models/v5b/seg/<run>/seg<NN>/<cfg>/val_probs/*.png) vs the GT masks letterboxed
to the same grid. Reports IoU / Dice / P / R on positives+negatives and the
negative-frame false-alarm rate (any component >= 0.05% / >= 1% of valid pixels).
  python scripts/v5b/seg_thr_sweep.py --run segV5b_a --seg 1 --cfg lora"""
import argparse, json, tarfile
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))
from src.v5b.train_seg import comp_boxes  # noqa: E402
DATA = ROOT / "data/v5b/stage/seg_v5b"

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--seg", type=int, default=1)
ap.add_argument("--cfg", default="lora"); ap.add_argument("--thrs", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
ap.add_argument("--out", default=None); a = ap.parse_args()
d = ROOT / "models/v5b/seg" / a.run / f"seg{a.seg:02d}" / a.cfg
if not (d / "val_probs").exists():
    with tarfile.open(d / "val_probs.tar") as tf: tf.extractall(d)
rows = [r for r in json.loads((DATA / "manifest.json").read_text()) if r["role"] == "val"]
thrs = [float(t) for t in a.thrs.split(",")]
W, H = 1280 // 4, 736 // 4
acc = {t: dict(tp=0, fp=0, fn=0, neg_fa=0, neg_fa1=0, neg_fp=0, neg_valid=0, pos_iou=[]) for t in thrs}
n_neg = 0
for r in rows:
    pr = np.asarray(Image.open(d / "val_probs" / f"{r['id']}.png")).astype(np.float32) / 255.0
    s = min(1280 / r["orig_w"], 736 / r["orig_h"]); nw, nh = round(r["orig_w"] * s / 4), round(r["orig_h"] * s / 4)
    valid = np.zeros((H, W), bool); valid[:nh, :nw] = True
    gt = np.zeros((H, W), bool)
    if r.get("mask"):
        m = Image.open(DATA / r["mask"]).convert("L").resize((nw, nh), Image.NEAREST)
        gt[:nh, :nw] = np.asarray(m) > 127
    nvalid = int(valid.sum()); min_area = max(8, int(0.0005 * nvalid))
    if r["is_negative"]: n_neg += 1
    for t in thrs:
        p = (pr >= t) & valid
        tp = int((p & gt).sum()); fp = int((p & ~gt).sum()); fn = int((~p & gt).sum())
        A = acc[t]; A["tp"] += tp; A["fp"] += fp; A["fn"] += fn
        if r["is_negative"]:
            comps = comp_boxes(torch.from_numpy(p), min_area)
            A["neg_fa"] += int(len(comps) > 0); A["neg_fa1"] += int(bool(comps) and max(c[4] for c in comps) >= 0.01 * nvalid)
            A["neg_fp"] += fp; A["neg_valid"] += nvalid
        else:
            A["pos_iou"].append(tp / max(tp + fp + fn, 1))
out = []
print(f"| thr | IoU | Dice | precision | recall | mIoU pos | neg frame FA (any) | neg frame FA (>=1%) | neg FP pixel frac |")
print("|---|---|---|---|---|---|---|---|---|")
for t in thrs:
    A = acc[t]; iou = A["tp"] / max(A["tp"] + A["fp"] + A["fn"], 1); dice = 2 * A["tp"] / max(2 * A["tp"] + A["fp"] + A["fn"], 1)
    P = A["tp"] / max(A["tp"] + A["fp"], 1); R = A["tp"] / max(A["tp"] + A["fn"], 1)
    rec = dict(thr=t, iou=iou, dice=dice, precision=P, recall=R, miou_pos=float(np.mean(A["pos_iou"])),
               neg_fa=A["neg_fa"] / max(n_neg, 1), neg_fa1=A["neg_fa1"] / max(n_neg, 1), neg_fp_frac=A["neg_fp"] / max(A["neg_valid"], 1))
    out.append(rec)
    print(f"| {t:.1f} | {iou:.3f} | {dice:.3f} | {P:.3f} | {R:.3f} | {rec['miou_pos']:.3f} | {rec['neg_fa']:.3f} | {rec['neg_fa1']:.3f} | {rec['neg_fp_frac']:.4f} |")
(Path(a.out) if a.out else d / "thr_sweep.json").write_text(json.dumps(out, indent=1))
