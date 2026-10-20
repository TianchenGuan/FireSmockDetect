#!/usr/bin/env python
"""Per-frame 'whitish plume' proxy for MmodalFire interference/blank frames (CPU).
NOT a label. Used only to interpolate between visually inspected frames and to flag transitions.
proxy = fraction of ROI pixels that are brighter than the per-session per-pixel 10th-percentile
background by > DV and have low saturation (< SMAX). People in dark/red clothing are excluded by
the saturation/brightness test; lighting changes are handled by splitting Mist4 at 16:35:00.
Output: data/v5/mmodalfire/plume_proxy.csv (frame, session, idx, proxy_frac, roi_mean_dv)
"""
import glob, os, cv2, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
FR="data/v4/raw/mmodalfire/frames_1fps"
ROI = {"Mist1":(500,100,1450,950),"Mist2":(500,100,1450,950),"Mist3":(500,60,1450,910),"Mist4":(500,60,1450,910),
       "DryIce1":(500,100,1450,950),"DryIce2":(500,100,1450,950),"DryIce3":(500,100,1450,950),"DryIce4":(500,100,1450,950),
       "BlankControl1":(500,100,1450,950),"BlankControl2":(500,100,1450,950)}
DS=4  # downsample factor for speed
DV=10; SMAX=70

def load(p, roi):
    x0,y0,x1,y1=roi; im=cv2.imread(p)[y0:y1,x0:x1]
    return cv2.resize(im,((x1-x0)//DS,(y1-y0)//DS),interpolation=cv2.INTER_AREA)

def run_session(sess):
    fr=sorted(glob.glob(f"{FR}/{sess}/*.jpg")); roi=ROI[sess]
    ims=[load(p,roi) for p in fr]
    # segments for background (lighting change in Mist4 at 16:35:00)
    segs=[(0,len(fr))]
    if sess=="Mist4":
        cut=[i for i,p in enumerate(fr) if os.path.basename(p)[-10:-4]>="163500"][0]; segs=[(0,cut),(cut,len(fr))]
    rows=[]
    for a,b in segs:
        stack=np.stack([cv2.cvtColor(im,cv2.COLOR_BGR2HSV) for im in ims[a:b]])  # N,h,w,3
        bgV=np.percentile(stack[...,2],10,axis=0)
        for k in range(a,b):
            hsv=stack[k-a]; V=hsv[...,2].astype(np.float32); Sat=hsv[...,1]
            dv=V-bgV; mask=(dv>DV)&(Sat<SMAX)
            rows.append(dict(frame=fr[k],session=sess,idx=k,proxy_frac=round(float(mask.mean()),4),roi_mean_dv=round(float(dv.mean()),2)))
    return rows

if __name__=="__main__":
    with ProcessPoolExecutor(10) as ex: res=list(ex.map(run_session, list(ROI)))
    df=pd.DataFrame([r for rs in res for r in rs]); os.makedirs("data/v5/mmodalfire",exist_ok=True)
    df.to_csv("data/v5/mmodalfire/plume_proxy.csv",index=False)
    print(df.groupby("session").proxy_frac.describe().round(4).to_string())
