"""Extract 1-fps RGB frames from the BAMBI flight_98 'point1' segment (V5 Day-1).
Frame k (k=0..) = video frame round(k*fps); written as bambi_flight98_tKKKK.jpg
(the `_tNNNN` 1-fps naming that scripts/v5/review_common.parse_ts_from_name reads).
Absolute local time per sampled frame is taken from the SRT (DJI per-frame block).
Deterministic; CPU only."""
import csv, re, sys, cv2
from pathlib import Path
ROOT = Path("/local/home/tianrgua/cvproject")
D = ROOT / "data/v5/frames/bambi_flight98"
mp4 = D / "DJI_20230426173033_0001_V_point1.MP4"; srt = D / "DJI_20230426173033_0001_V_point1.SRT"
out = D / "frames"; out.mkdir(exist_ok=True)
# SRT: block index -> local datetime string
ts = {}
cur = None
for line in open(srt, errors="ignore"):
    m = re.match(r"FrameCnt: (\d+)", line.strip().replace('<font size="28">', ''))
    if m: cur = int(m.group(1)); continue
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)", line.strip())
    if m and cur is not None: ts[cur] = m.group(1); cur = None
cap = cv2.VideoCapture(str(mp4)); fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
rows = []; k = 0
while True:
    fi = int(round(k * fps))
    if fi >= n: break
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, im = cap.read()
    if not ok: break
    name = f"bambi_flight98_t{k:04d}.jpg"; p = out / name
    cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    rows.append(dict(idx=k, rel_s=k, video_frame=fi, abs_local=ts.get(fi + 1, ""), path=str(p), w=im.shape[1], h=im.shape[0]))
    k += 1
with open(D / "frames_1fps.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"fps={fps:.3f} total_frames={n} extracted={len(rows)} -> {out}")
