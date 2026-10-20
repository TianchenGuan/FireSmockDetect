#!/usr/bin/env python
"""Model-assisted DIAGNOSTIC (not review) for the WSDataset material we hold.

Scores (a) 200 random WSDataset positives (MultiNatSmoke copy; labels = v4
mask->box conversion in data/v4/yolo/B_G_v1/labels) and (b) all explicit
negatives from the author's GitHub demo (VMOKED dataset/demo) with
models/trained/expE_mix_y11m_1280_v2/seg05/best.pt on CPU, imgsz 1280.
Reports: recall of label boxes at IoU>=0.4 with conf>=0.2; fraction of explicit
negatives with any detection >=0.3.  expE_mix never trained on WSDataset
(data/yolo/lists/local/expB_train.txt has 0 WSDataset rows).
"""
import json, os, random, sys, glob, time
import numpy as np
from ultralytics import YOLO
ROOT='/local/home/tianrgua/cvproject'
WEIGHTS=f'{ROOT}/models/trained/expE_mix_y11m_1280_v2/seg05/best.pt'
LBL=f'{ROOT}/data/v4/yolo/B_G_v1/labels'
IMG=f'{ROOT}/data/v4/yolo/B_G_v1/images'
OUT=f'{ROOT}/data/v5b/manifests/wsdataset_sanity_check.json'
N=int(sys.argv[1]) if len(sys.argv)>1 else 200

def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1])); inter=ix*iy
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

random.seed(20261002)
stems=sorted(s[:-4] for s in os.listdir(LBL) if '__WSDataset__' in s)
sample=random.sample(stems, N)
model=YOLO(WEIGHTS)
t0=time.time(); rows=[]; tp=0; ngt=0; img_hit=0
for i,s in enumerate(sample):
    ip=f'{IMG}/{s}.jpg'
    gts=[]
    for ln in open(f'{LBL}/{s}.txt'):
        p=ln.split()
        if len(p)==5:
            cx,cy,w,h=map(float,p[1:]); gts.append((cx-w/2,cy-h/2,cx+w/2,cy+h/2))
    r=model.predict(ip, imgsz=1280, conf=0.2, device='cpu', verbose=False)[0]
    H,W=r.orig_shape
    dets=[(b[0]/W,b[1]/H,b[2]/W,b[3]/H) for b in r.boxes.xyxy.cpu().numpy().tolist()]
    confs=r.boxes.conf.cpu().numpy().tolist()
    matched=[any(iou(g,d)>=0.4 for d in dets) for g in gts]
    tp+=sum(matched); ngt+=len(gts); img_hit+=int(len(dets)>0)
    rows.append({'stem':s,'n_gt':len(gts),'n_det':len(dets),'gt_matched':sum(matched),'max_conf':max(confs) if confs else 0.0})
    if (i+1)%50==0: print(f'{i+1}/{N} box-recall so far {tp}/{ngt} ({time.time()-t0:.0f}s)', flush=True)
pos={'n_images':N,'n_gt_boxes':ngt,'gt_boxes_recalled_iou04_conf02':tp,'box_recall':tp/max(ngt,1),'images_with_any_det_conf02':img_hit,'image_recall':img_hit/N}
# explicit negatives from GitHub demo (unique: train copy only; test is byte-identical)
negs=sorted(glob.glob(f'{ROOT}/data/v5b/raw/wsdataset_github_demo/dataset/demo/classification/train/nonsmoke/*/*.jpg'))+sorted(glob.glob(f'{ROOT}/data/v5b/raw/wsdataset_github_demo/dataset/demo/detection/train/images/*.jpg'))
nrows=[]; nfp=0
for ip in negs:
    r=model.predict(ip, imgsz=1280, conf=0.3, device='cpu', verbose=False)[0]
    c=r.boxes.conf.cpu().numpy().tolist(); nfp+=int(len(c)>0)
    nrows.append({'path':os.path.relpath(ip,ROOT),'n_det_conf03':len(c),'max_conf':max(c) if c else 0.0})
neg={'n_images':len(negs),'images_with_any_det_conf03':nfp,'fp_image_rate':nfp/max(len(negs),1)}
# also demo smoke sequence (image-level positive, no boxes)
smk=sorted(glob.glob(f'{ROOT}/data/v5b/raw/wsdataset_github_demo/dataset/demo/classification/train/smoke/*/*.jpg')); sh=0
for ip in smk:
    r=model.predict(ip, imgsz=1280, conf=0.2, device='cpu', verbose=False)[0]; sh+=int(len(r.boxes)>0)
json.dump({'weights':os.path.relpath(WEIGHTS,ROOT),'imgsz':1280,'device':'cpu','seed':20261002,'note':'DIAGNOSTIC only; model-assisted, not human review; labels for (a) are v4 mask->box conversions of MultiNatSmoke model-assisted masks, NOT author boxes',
           'wsd_mns_positives_200':pos,'github_demo_explicit_negatives':neg,'github_demo_smoke_sequence_imagelevel':{'n_images':len(smk),'images_with_any_det_conf02':sh,'rate':sh/max(len(smk),1)},
           'per_image_pos':rows,'per_image_neg':nrows,'elapsed_s':time.time()-t0}, open(OUT,'w'), indent=1)
print('DONE', json.dumps(pos), json.dumps(neg), 'demo smoke seq', sh, '/', len(smk))
