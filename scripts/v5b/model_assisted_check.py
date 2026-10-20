#!/usr/bin/env python
"""Model-assisted sanity DIAGNOSTIC (not human review) for a v5b YOLO-labelled source.
Scores N random images that carry author smoke boxes plus ALL explicit-negative and
fire-only images with a frozen detector on CPU (imgsz 1280), then reports
  * recall of author smoke boxes at conf>=0.2 with IoU>=0.4 (box-level and image-level)
  * explicit-negative images with any detection >= 0.3
Usage: python scripts/v5b/model_assisted_check.py --root <extracted> --source sheffield_uav \
         --smoke-class 1 --out-prefix data/v5b/manifests/_cache/sheffield_uav_modelcheck [--n 200]
"""
import argparse, glob, json, os, random, time
import numpy as np, pandas as pd

def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def read_labels(lf, W, H, smoke_cls):
    sm, fi = [], []
    for l in open(lf).read().splitlines():
        p = l.split()
        if len(p) < 5: continue
        k = int(float(p[0])); x, y, w, h = map(float, p[1:5])
        box = [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]
        (sm if k == smoke_cls else fi).append(box)
    return sm, fi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--source", required=True)
    ap.add_argument("--smoke-class", type=int, default=1); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--weights", default="models/trained/expE_mix_y11m_1280_v2/seg05/best.pt")
    ap.add_argument("--out-prefix", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from ultralytics import YOLO
    from PIL import Image
    imgs = sorted(glob.glob(f"{a.root}/*/images/*.jpg"))
    groups = {"smoke": [], "explicit_negative": [], "fire_only": []}
    meta = {}
    for im in imgs:
        lf = im.replace("/images/", "/labels/")[:-4] + ".txt"
        if not os.path.exists(lf): continue
        with Image.open(im) as I: W, H = I.size
        sm, fi = read_labels(lf, W, H, a.smoke_class)
        meta[im] = (sm, fi)
        groups["smoke" if sm else ("fire_only" if fi else "explicit_negative")].append(im)
    random.seed(a.seed)
    sample = random.sample(groups["smoke"], min(a.n, len(groups["smoke"]))) + groups["explicit_negative"] + groups["fire_only"]
    model = YOLO(a.weights); names = model.names
    print("model classes:", names, "| scoring", len(sample), "images on CPU", flush=True)
    rows = []; t0 = time.time()
    for i in range(0, len(sample), 8):
        res = model.predict(sample[i:i + 8], imgsz=1280, conf=0.05, device="cpu", verbose=False)
        for im, r in zip(sample[i:i + 8], res):
            b = r.boxes
            rows.append(dict(image=im, group=("smoke" if meta[im][0] else ("fire_only" if meta[im][1] else "explicit_negative")),
                             n_author_smoke=len(meta[im][0]), n_author_fire=len(meta[im][1]),
                             pred=json.dumps([[float(v) for v in xyxy] + [float(c), int(k)] for xyxy, c, k in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist())])))
        if (i // 8) % 5 == 0: print(f"  {i+len(res)}/{len(sample)} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_parquet(a.out_prefix + "_preds.parquet", index=False)
    # metrics
    smoke_rows = df[df.group == "smoke"]; tp = fn = 0; img_hit = 0
    for r in smoke_rows.itertuples():
        preds = [p for p in json.loads(r.pred) if p[4] >= 0.2]
        hits = 0
        for g in meta[r.image][0]:
            if any(iou(g, p[:4]) >= 0.4 for p in preds): hits += 1
        tp += hits; fn += len(meta[r.image][0]) - hits
        img_hit += int(hits > 0)
    neg = df[df.group == "explicit_negative"]; neg_fire = {}
    for thr in (0.3, 0.5):
        neg_fire[str(thr)] = int(sum(any(p[4] >= thr for p in json.loads(r.pred)) for r in neg.itertuples()))
    fo = df[df.group == "fire_only"]
    fo_fire = int(sum(any(p[4] >= 0.3 for p in json.loads(r.pred)) for r in fo.itertuples()))
    summ = {"source": a.source, "weights": a.weights, "imgsz": 1280, "device": "cpu", "model_classes": names,
            "n_smoke_images_scored": int(len(smoke_rows)), "author_smoke_boxes_in_sample": int(tp + fn),
            "box_recall_conf0.2_iou0.4": round(tp / max(1, tp + fn), 4),
            "image_level_recall_conf0.2_iou0.4": round(img_hit / max(1, len(smoke_rows)), 4),
            "n_explicit_negative_images": int(len(neg)), "explicit_negative_with_det_ge0.3": neg_fire["0.3"],
            "explicit_negative_with_det_ge0.5": neg_fire["0.5"], "n_fire_only_images": int(len(fo)),
            "fire_only_with_det_ge0.3": fo_fire, "seed": a.seed,
            "note": "DIAGNOSTIC agreement between author labels and a frozen v4-era detector; NOT human review."}
    json.dump(summ, open(a.out_prefix + "_summary.json", "w"), indent=2); print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    main()
