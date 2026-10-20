#!/usr/bin/env python3
"""Extract 1-fps RGB frames (JPEG q90, native 4K) from a BAMBI raw flight zip and parse the DJI SRT
per-frame telemetry (timestamp, lat/lon, altitude, gimbal pitch). CPU only, cv2 sequential grab().

Usage: bambi_extract_frames.py --zip <flight_N.zip> --flight N --out <flight_dir> [--keep-zip]
Outputs in <flight_dir>: frames_1fps/*.jpg, frames_1fps/index.csv, meta/ (SRT, Airdata CSV, calib JSON),
_zip_listing.json, _extract_log.json. Deletes the zip afterwards unless --keep-zip (md5 is logged beforehand).
Frame name: f<flight>_s<seg>_<YYYY-MM-DD>T<HH-MM-SS>Z_v<videoframe>.jpg (ISO = UTC, parsed by review_common)."""
import argparse, csv, datetime as dt, hashlib, json, os, re, sys, time, zipfile, zoneinfo
from pathlib import Path
import cv2

TZ = zoneinfo.ZoneInfo("Europe/Vienna")
UTC = dt.timezone.utc
RE_TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[.,](\d{1,3}))?")
RE_KV = re.compile(r"\[?\s*([a-z_]+)\s*:\s*(-?\d+(?:\.\d+)?)")
RE_FRAMECNT = re.compile(r"FrameCnt\s*:\s*(\d+)")
RE_DJI_START = re.compile(r"DJI_(\d{14})")


def md5sum(p, buf=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def parse_srt(text):
    """Return dict FrameCnt -> telemetry dict (ts_local, lat, lon, rel_alt, abs_alt, gb_pitch, ...)."""
    out = {}
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        body = " ".join(lines[2:])
        m = RE_FRAMECNT.search(body)
        if not m:
            continue
        fc = int(m.group(1))
        rec = {}
        t = RE_TS.search(body)
        if t:
            frac = (t.group(2) or "0").ljust(3, "0")[:3]
            rec["ts_local"] = f"{t.group(1)}.{frac}"
        for k, v in RE_KV.findall(body):
            if k in ("latitude", "longitude", "rel_alt", "abs_alt", "gb_pitch", "gb_yaw", "gb_roll", "focal_len", "iso", "fnum", "ev", "dzoom_ratio"):
                rec[k] = float(v)
        out[fc] = rec
    return out


def local_to_utc(s):
    d = dt.datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    if len(s) > 20:
        d = d.replace(microsecond=int(s[20:23].ljust(3, "0")) * 1000)
    return d.replace(tzinfo=TZ).astimezone(UTC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--flight", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--fps-out", type=float, default=1.0)
    ap.add_argument("--jpeg-q", type=int, default=90)
    a = ap.parse_args()
    out = Path(a.out); fr_dir = out / "frames_1fps"; meta_dir = out / "meta"; tmp_dir = out / "_tmp_video"
    for d in (fr_dir, meta_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    z = zipfile.ZipFile(a.zip)
    infos = z.infolist()
    listing = [dict(name=i.filename, bytes=i.file_size) for i in infos]
    json.dump(listing, open(out / "_zip_listing.json", "w"), indent=0)
    zip_md5 = md5sum(a.zip)
    # small metadata files kept
    for i in infos:
        ext = os.path.splitext(i.filename)[1].lower()
        if ext in (".srt", ".csv", ".json", ".txt") and i.file_size < 200e6:
            z.extract(i, meta_dir)
    videos = sorted([i for i in infos if i.filename.lower().endswith((".mp4", ".mov"))], key=lambda i: i.filename)
    rows = []; vlog = []; seg = 0
    for vi in videos:
        name = os.path.basename(vi.filename); stem = os.path.splitext(name)[0]
        up = stem.upper()
        # thermal streams are 640x512 and named *_T*; decide by decode size as well
        tmp = Path(z.extract(vi, tmp_dir))
        cap = cv2.VideoCapture(str(tmp))
        if not cap.isOpened():
            vlog.append(dict(video=name, status="cannot_open")); tmp.unlink(); continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 29.97; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        is_thermal = (w <= 1024) or re.search(r"_T(_|\.|$)", up) is not None
        if is_thermal:
            vlog.append(dict(video=name, status="skipped_thermal", w=w, h=h, frames=n, fps=round(fps, 3))); cap.release(); tmp.unlink(); continue
        seg += 1
        # SRT lookup (same stem)
        srt = None
        for cand in meta_dir.rglob("*"):
            if cand.suffix.lower() == ".srt" and cand.stem == stem:
                srt = parse_srt(open(cand, encoding="utf-8", errors="replace").read()); break
        m = RE_DJI_START.search(stem)
        seg_start_local = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=TZ) if m else None
        step = fps / a.fps_out; next_pick = 0.0; vf = 0; n_written = 0
        while True:
            if not cap.grab():
                break
            if vf >= next_pick - 1e-6:
                ok, img = cap.retrieve()
                if ok:
                    tel = (srt or {}).get(vf + 1) or {}
                    ts_src = "srt"
                    if tel.get("ts_local"):
                        ts_utc = local_to_utc(tel["ts_local"])
                    elif seg_start_local is not None:
                        ts_utc = (seg_start_local + dt.timedelta(seconds=vf / fps)).astimezone(UTC); ts_src = "filename_start+frame/fps"
                    else:
                        ts_utc = None; ts_src = "none"
                    iso = ts_utc.strftime("%Y-%m-%dT%H-%M-%S") + "Z" if ts_utc else f"t{int(vf/fps):06d}"
                    fn = f"f{a.flight}_s{seg:02d}_{iso}_v{vf:06d}.jpg"
                    cv2.imwrite(str(fr_dir / fn), img, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_q])
                    rows.append(dict(flight_id=f"flight_{a.flight}", segment=seg, video_file=name, video_frame=vf, fps=round(fps, 4),
                                     rel_s_in_video=round(vf / fps, 3), ts_utc=ts_utc.isoformat() if ts_utc else "", ts_local=tel.get("ts_local", ""),
                                     ts_source=ts_src, lat=tel.get("latitude", ""), lon=tel.get("longitude", ""), rel_alt_m=tel.get("rel_alt", ""),
                                     abs_alt_m=tel.get("abs_alt", ""), gimbal_pitch_deg=tel.get("gb_pitch", ""), focal_len=tel.get("focal_len", ""),
                                     width=w, height=h, frame_file=fn))
                    n_written += 1
                next_pick += step
            vf += 1
        cap.release(); tmp.unlink()
        vlog.append(dict(video=name, status="ok", w=w, h=h, frames_reported=n, frames_read=vf, fps=round(fps, 3), dur_s=round(vf / fps, 1),
                         frames_written=n_written, srt_found=srt is not None, srt_entries=len(srt) if srt else 0))
        print(f"  {name}: {w}x{h} {vf} fr @ {fps:.2f} -> {n_written} jpg (srt={'y' if srt else 'n'})", flush=True)
    with open(fr_dir / "index.csv", "w", newline="") as fh:
        if rows:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)
    tmp_dir.rmdir()
    log = dict(flight_id=f"flight_{a.flight}", zip=os.path.basename(a.zip), zip_bytes=os.path.getsize(a.zip), zip_md5=zip_md5, n_zip_entries=len(infos),
               videos=vlog, n_frames_written=len(rows), total_rgb_video_s=round(sum(v.get("dur_s", 0) for v in vlog if v["status"] == "ok"), 1),
               fps_out=a.fps_out, jpeg_q=a.jpeg_q, extracted_utc=dt.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), elapsed_s=round(time.time() - t0, 1),
               zip_deleted=not a.keep_zip)
    json.dump(log, open(out / "_extract_log.json", "w"), indent=1)
    if not a.keep_zip:
        os.remove(a.zip)
    print(f"flight_{a.flight}: {len(rows)} frames, {log['total_rgb_video_s']} s RGB video, {log['elapsed_s']} s elapsed, zip_deleted={not a.keep_zip}")


if __name__ == "__main__":
    main()
