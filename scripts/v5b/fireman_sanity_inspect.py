"""DIAGNOSTIC (not review): score 200 random labelled RGB FireMan images with expE_mix_y11m_1280_v2 on CPU.
Reports: recall of author smoke boxes at IoU>=0.4 (pred conf>=0.2); fraction of explicit-negative images with any detection >=0.3."""
import os, json, random, sys, numpy as np, pandas as pd, torch
from ultralytics import YOLO
ROOT = '/local/home/tianrgua/cvproject'; RAW = f'{ROOT}/data/v5b/raw/fireman'
W = f'{ROOT}/models/trained/expE_mix_y11m_1280_v2/seg05/best.pt'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200; seed = 0
df = pd.read_parquet(f'{ROOT}/data/v5b/manifests/fireman_manifest.parquet')
lab = df[(df.modality == 'rgb') & (df.proposed_split != 'quarantine') & (df.label_type.isin(['box', 'image'])) & (df.label_conflict == '') & (df.decode_err == '')]
pos = lab[lab.label_type == 'box']; neg = lab[lab.is_explicit_negative]
random.seed(seed); n_neg = min(len(neg), N // 2); n_pos = N - n_neg
samp = pd.concat([pos.sample(n_pos, random_state=seed), neg.sample(n_neg, random_state=seed)]) if n_neg else pos.sample(N, random_state=seed)
torch.set_num_threads(min(64, os.cpu_count())); model = YOLO(W)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
rec = []; tp = 0; nb = 0; neg_any = 0; neg_n = 0; pos_any = 0
for r in samp.itertuples():
    res = model.predict(f'{RAW}/{r.rel_path}', imgsz=1280, conf=0.2, device='cpu', verbose=False)[0]
    preds = [(b.tolist(), float(c)) for b, c in zip(res.boxes.xyxy.cpu(), res.boxes.conf.cpu())]
    gts = json.loads(r.smoke_boxes)
    hit = [max([iou(g, p[0]) for p in preds], default=0) >= 0.4 for g in gts]
    tp += sum(hit); nb += len(gts)
    if r.is_explicit_negative: neg_n += 1; neg_any += int(any(c >= 0.3 for _, c in preds))
    else: pos_any += int(any(c >= 0.3 for _, c in preds))
    rec.append({'rel_path': r.rel_path, 'video_id': r.video_id, 'parent_unit': r.parent_unit, 'label_type': r.label_type, 'is_explicit_negative': bool(r.is_explicit_negative), 'n_gt': len(gts), 'n_gt_hit_iou04': int(sum(hit)),
                'n_pred_conf02': len(preds), 'max_conf': max([c for _, c in preds], default=0.0), 'gt_boxes': gts, 'preds': [[round(v, 1) for v in p[0]] + [round(p[1], 3)] for p in preds]})
out = {'model': W, 'imgsz': 1280, 'device': 'cpu', 'n_images': len(samp), 'n_pos_box_images': int(len(samp) - neg_n), 'n_explicit_neg_images': neg_n,
       'author_box_recall_iou04_conf02': tp / max(nb, 1), 'n_author_boxes': nb, 'pos_images_any_det_conf03': pos_any / max(len(samp) - neg_n, 1),
       'explicit_neg_images_any_det_conf03': neg_any / max(neg_n, 1), 'note': 'DIAGNOSTIC agreement with a v2 model; NOT human review. Author boxes are large-region boxes; low IoU recall may reflect box-style mismatch.'}
os.makedirs(f'{ROOT}/data/v5b/manifests/_inspect', exist_ok=True)
json.dump({'summary': out, 'per_image': rec}, open(f'{ROOT}/data/v5b/manifests/_inspect/fireman_sanity_inspect.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
