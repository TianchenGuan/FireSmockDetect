"""DIAGNOSTIC (not review): score 200 random FireSpot labelled images (pos + explicit neg) with the
v2 expE YOLO11m checkpoint on CPU at imgsz 1280 and report agreement with the author labels.
Output: data/v5b/manifests/_cache/firespot_sanity_preds.parquet + firespot_sanity_summary.json"""
import os, json, random, numpy as np, pandas as pd, torch
from ultralytics import YOLO
ROOT = "/local/home/tianrgua/cvproject"
IMG = f"{ROOT}/data/v5b/raw/firespot/dataset/images"; LAB = f"{ROOT}/data/v5b/raw/firespot/dataset/labels"
CKPT = f"{ROOT}/models/trained/expE_mix_y11m_1280_v2/seg05/best.pt"
torch.set_num_threads(48)
random.seed(20261002)
names = sorted(f[:-4] for f in os.listdir(LAB) if f.endswith(".txt"))
sample = random.sample(names, 200)
model = YOLO(CKPT)
rows = []
for i, n in enumerate(sample):
    r = model.predict(f"{IMG}/{n}.jpg", imgsz=1280, conf=0.001, device="cpu", verbose=False)[0]
    W, H = r.orig_shape[1], r.orig_shape[0]
    dets = [(float(c), *[float(v) for v in b]) for c, b in zip(r.boxes.conf.tolist(), r.boxes.xyxyn.tolist())]
    gts = []
    for ln in open(f"{LAB}/{n}.txt"):
        p = ln.split()
        if len(p) == 5:
            xc, yc, w, h = map(float, p[1:]); gts.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    rows.append({"stem": n, "n_gt": len(gts), "gts": json.dumps(gts), "dets": json.dumps(dets), "W": W, "H": H})
    if i % 25 == 0: print(i, flush=True)
df = pd.DataFrame(rows); df.to_parquet(f"{ROOT}/data/v5b/manifests/_cache/firespot_sanity_preds.parquet", index=False)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0
summ = {}
for conf_t in [0.1, 0.2, 0.3]:
    gt_total = gt_hit = 0; img_pos = img_pos_hit = 0; neg = neg_fp = 0; pos_imgs_any = 0
    for _, r in df.iterrows():
        gts = json.loads(r.gts); dets = [d for d in json.loads(r.dets) if d[0] >= conf_t]
        if r.n_gt > 0:
            img_pos += 1; hit_any = False
            for g in gts:
                gt_total += 1
                if any(iou(g, d[1:]) >= 0.4 for d in dets): gt_hit += 1; hit_any = True
            img_pos_hit += hit_any; pos_imgs_any += bool(dets)
        else:
            neg += 1; neg_fp += bool(dets)
    summ[f"conf_{conf_t}"] = {"n_pos_images": img_pos, "n_author_boxes": gt_total, "box_recall_iou0.4": round(gt_hit / max(1, gt_total), 4),
                              "pos_image_recall_any_box_iou0.4": round(img_pos_hit / max(1, img_pos), 4),
                              "pos_images_with_any_det": pos_imgs_any,
                              "n_explicit_neg_images": neg, "neg_images_with_any_det": neg_fp, "neg_fp_rate": round(neg_fp / max(1, neg), 4)}
summ["checkpoint"] = CKPT; summ["imgsz"] = 1280; summ["device"] = "cpu"; summ["n_images"] = len(df); summ["seed"] = 20261002
summ["note"] = "model-assisted DIAGNOSTIC only; not human review; headline = box recall at conf 0.2 and explicit-neg FP at conf 0.3"
json.dump(summ, open(f"{ROOT}/data/v5b/manifests/_cache/firespot_sanity_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
