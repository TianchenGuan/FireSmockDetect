#!/usr/bin/env python
"""Extract 1-fps frames from the raw MmodalFire DVR MP4s for a wall-clock window (CPU).
Needed because XLSX session intervals are minute-granular and some fires (e.g. Polyurethane1)
are already burning at the first in-session frame. OSD start of each DVR file was read visually
from frame 0 (data/v5/mmodalfire/_review_sheets/osd_frame0.jpg); fps from container metadata.
Writes <out>/<Session>_pre_YYYYMMDD_HHMMSS.jpg and an OSD strip sheet for manual clock verification.
"""
import argparse, os, cv2, datetime as dt, numpy as np
V="data/v4/raw/mmodalfire/extracted/原始数据/Video"
OSD_START = {  # read visually from frame 0
 "Downtest_ch01_00000000014000100": "2022-08-08 09:51:44",
 "Downtest_ch01_00000000007000100": "2022-08-03 09:58:26",
 "Downtest_ch01_00000000019000200": "2022-08-10 09:50:09",
 "Downtest_ch01_00000000010000000": "2022-08-03 15:28:28",
 "Downtest_ch01_00000000009000000": "2022-08-03 13:38:32",
 "Downtest_ch01_00000000011000100": "2022-08-04 10:54:03",
}
ap=argparse.ArgumentParser()
ap.add_argument("--stem",required=True); ap.add_argument("--session",required=True)
ap.add_argument("--start",required=True,help="YYYY-MM-DD HH:MM:SS"); ap.add_argument("--seconds",type=int,required=True)
ap.add_argument("--out",required=True)
a=ap.parse_args()
cap=cv2.VideoCapture(f"{V}/{a.stem}.mp4"); fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
t0=dt.datetime.fromisoformat(OSD_START[a.stem]); ts=dt.datetime.fromisoformat(a.start)
off=(ts-t0).total_seconds(); assert off>=0, "window before file start"
os.makedirs(a.out,exist_ok=True); strips=[]
idx0=int(round(off*fps)); cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)
cur=idx0; want=[int(round((off+k)*fps)) for k in range(a.seconds)]; k=0
while k<len(want):
    ok,fr=cap.read()
    if not ok: print("EOF at",cur); break
    if cur==want[k]:
        t=ts+dt.timedelta(seconds=k); name=f"{a.session}_pre_{t:%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(os.path.join(a.out,name), fr, [cv2.IMWRITE_JPEG_QUALITY,90])
        if k%10==0:
            s=fr[0:60,0:800].copy(); cv2.putText(s,f"pred {t:%H:%M:%S} k={k}",(560,45),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2); strips.append(s)
        k+=1
    cur+=1
cv2.imwrite(os.path.join(a.out,f"_osd_check_{a.session}.jpg"), np.vstack(strips))
print(a.session, "wrote", k, "frames from", a.stem, "idx0", idx0, "of", n, "fps", round(fps,3))
