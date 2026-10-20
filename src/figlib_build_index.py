#!/usr/bin/env python3
"""Build figlib_events_index.csv from crawled event index pages + tarball HEADs."""
import csv, html, os, re, sys
from urllib.parse import unquote

IDX_DIR = "/tmp/figlib_idx"
EVENTS_FILE = "/tmp/figlib_all_events.txt"
OUT_CSV = "/local/home/tianrgua/cvproject/data/manifests/figlib_events_index.csv"
BASE = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data"

FRAME_RE = re.compile(r"href=([^ >]+\.jpg)")
NIMG_RE = re.compile(r"MP4 summary for (\d+) images")
OFFSET_RE = re.compile(r"^(\d+)_([+-]\d+)\.jpg$")

def split_event(ev):
    """Handle both 20160604_FIRE_rm-n-mobo-c and 20220831-Border32fire-pi-s-mobo-m
    and 20231123.134319-SageFire-sm-e-mobo-c styles."""
    toks = re.split(r"[_-]", ev)
    date = toks[0][:8]
    # camera anchor: 'mobo' or token starting with 'iqeye'
    anchor = None
    for i, t in enumerate(toks):
        if t == "mobo" or t.startswith("iqeye") or t.startswith("axis"):
            anchor = i
    if anchor is None:
        return date, "_".join(toks[1:]), ""
    # direction (n/s/e/w/ne/ws...) or tcsN before anchor -> include site token too
    if anchor >= 2 and re.match(r"^([nsew]{1,2}|tcs\d+)$", toks[anchor - 1]):
        start = anchor - 2
    else:
        start = anchor - 1
    camera = "-".join(toks[start:])
    fire = "_".join(toks[1:start])
    return date, fire, camera

def parse_event(ev):
    row = {"event_id": ev}
    date, fire, camera = split_event(ev)
    row["date"] = date
    row["fire_name"] = fire
    row["camera_id"] = camera
    row["url_dir"] = f"{BASE}/{ev}/"
    row["url_tgz"] = f"{BASE}/Tar/{ev}.tgz"

    idx_path = os.path.join(IDX_DIR, ev + ".html")
    n_images = ""
    n_frames_listed = n_with_offset = 0
    min_off = max_off = ""
    listed_ok = False
    if os.path.exists(idx_path) and os.path.getsize(idx_path) > 0:
        t = open(idx_path, encoding="utf-8", errors="replace").read()
        m = NIMG_RE.search(t)
        if m:
            n_images = int(m.group(1))
        offs = []
        for href in FRAME_RE.findall(t):
            name = html.unescape(unquote(href))
            n_frames_listed += 1
            mo = OFFSET_RE.match(name)
            if mo:
                n_with_offset += 1
                offs.append(int(mo.group(2)))
        if offs:
            min_off, max_off = min(offs), max(offs)
        listed_ok = n_frames_listed > 0
    row["n_images"] = n_images if n_images != "" else (n_frames_listed if listed_ok else "")
    row["n_frames_listed"] = n_frames_listed
    row["n_frames_with_offset"] = n_with_offset
    row["min_offset_s"] = min_off
    row["max_offset_s"] = max_off
    row["listing_ok"] = listed_ok

    head_path = os.path.join(IDX_DIR, ev + ".head")
    size = ""
    if os.path.exists(head_path):
        h = open(head_path, encoding="utf-8", errors="replace").read()
        sm = re.search(r"(?i)^content-length:\s*(\d+)", h, re.M)
        ok = re.search(r"HTTP/[\d.]+\s+200", h)
        if sm and ok:
            size = int(sm.group(1))
    row["tgz_size_bytes"] = size
    return row

def main():
    events = [l.strip() for l in open(EVENTS_FILE) if l.strip()]
    rows = [parse_event(ev) for ev in sorted(events)]
    cols = ["event_id", "date", "fire_name", "camera_id", "n_images",
            "n_frames_listed", "n_frames_with_offset", "min_offset_s", "max_offset_s",
            "tgz_size_bytes", "url_dir", "url_tgz", "listing_ok"]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n_sz = sum(1 for r in rows if r["tgz_size_bytes"] != "")
    tot = sum(r["tgz_size_bytes"] for r in rows if r["tgz_size_bytes"] != "")
    n_off = sum(1 for r in rows if r["n_frames_with_offset"] > 0)
    print(f"events={len(rows)} with_listing={sum(1 for r in rows if r['listing_ok'])} "
          f"with_tgz_size={n_sz} total_tgz_bytes={tot} ({tot/1e9:.2f} GB) events_with_offset_frames={n_off}")

if __name__ == "__main__":
    main()
