#!/usr/bin/env python3
"""Validate downloaded FIgLib LOCKBOX v3 events (iteration 3) and build manifest.

Identical validation to iterations 1 and 2 (src/figlib_lockbox_manifest.py):
frame counts vs remote listing, monotonic offsets, monotonic epochs,
epoch-delta == offset-delta, pre- and post-smoke frames present,
exact-or-near +00000 frame.

File-level validation ONLY. No image content is inspected or rendered.
"""
import os, re, sys
from datetime import datetime, timezone
import pandas as pd

ROOT = "/local/home/tianrgua/cvproject"
RAW = f"{ROOT}/data/v3/figlib_lockbox"
IDX = f"{ROOT}/data/manifests/figlib_events_index.csv"
OUT_PQ = f"{ROOT}/data/v3/manifests/figlib_lockbox_v3_manifest.parquet"
OUT_CSV = f"{ROOT}/data/v3/manifests/figlib_lockbox_v3_manifest.csv.gz"
REPORT = f"{ROOT}/logs/figlib_lockbox_v3_validation_report.csv"

FRAME_RE = re.compile(r"^(\d+)_([+-]\d+)\.jpg$")
NEAR_ZERO_S = 60  # "near +00000": nearest frame to offset 0 within this many seconds

LICENSE = ("HPWREN FIgLib: publicly available; use requires a credit reference to "
           "https://hpwren.ucsd.edu/ in derivative work; data provided as-is, no guarantees. "
           "Source: http://hpwren.ucsd.edu/FIgLib/HPWREN-FIgLib/ (retrieved 2026-09-13)")

def main():
    idx = pd.read_csv(IDX).set_index("event_id")
    events = sorted(d for d in os.listdir(RAW)
                    if os.path.isdir(os.path.join(RAW, d)) and not d.startswith("_"))
    rows, report = [], []
    for ev in events:
        evdir = os.path.join(RAW, ev)
        frames, skipped = [], []
        for fn in sorted(os.listdir(evdir)):
            m = FRAME_RE.match(fn)
            if m:
                if os.path.getsize(os.path.join(evdir, fn)) == 0:
                    skipped.append(fn)  # empty file: treat as invalid jpg
                else:
                    frames.append((int(m.group(1)), int(m.group(2)), fn))
            elif fn.lower().endswith(".jpg"):
                skipped.append(fn)
        frames.sort(key=lambda x: x[0])  # sort by epoch timestamp
        offs = [f[1] for f in frames]
        epochs = [f[0] for f in frames]
        mono = all(b > a for a, b in zip(offs, offs[1:]))
        mono_epoch = all(b > a for a, b in zip(epochs, epochs[1:]))
        deltas_ok = all((e2 - e1) == (o2 - o1)
                        for (e1, o1, _), (e2, o2, _) in zip(frames, frames[1:]))
        has_neg = any(o < 0 for o in offs)
        has_pos = any(o > 0 for o in offs)
        has_zero = any(o == 0 for o in offs)
        nearest = min((abs(o) for o in offs), default=None)
        zero_ok = has_zero or (nearest is not None and nearest <= NEAR_ZERO_S)
        cam = idx.loc[ev, "camera_id"] if ev in idx.index else ""
        n_listed = idx.loc[ev, "n_frames_listed"] if ev in idx.index else None
        valid = (len(frames) > 0 and len(skipped) == 0
                 and (n_listed is None or len(frames) == n_listed)
                 and mono and mono_epoch and deltas_ok
                 and has_neg and has_pos and zero_ok)
        report.append(dict(event_id=ev, n_frames=len(frames), n_listed_remote=n_listed,
                           n_nonoffset_jpg=len(skipped), min_off=min(offs) if offs else None,
                           max_off=max(offs) if offs else None,
                           monotonic_offsets=mono, monotonic_epochs=mono_epoch,
                           epoch_offset_deltas_match=deltas_ok,
                           has_pre_smoke=has_neg, has_post_smoke=has_pos,
                           has_zero_frame=has_zero, nearest_abs_offset_s=nearest,
                           valid=valid))
        if not valid:
            continue
        for epoch, off, fn in frames:
            rows.append(dict(
                event_id=ev,
                camera_id=cam,
                frame_path=os.path.relpath(os.path.join(evdir, fn), ROOT),
                time_offset_s=off,
                datetime_utc=datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(),
                unix_time=epoch,
                source="figlib",
                license_terms=LICENSE,
            ))
    man = pd.DataFrame(rows)
    man.to_parquet(OUT_PQ, index=False)
    man.to_csv(OUT_CSV, index=False, compression="gzip")
    rep = pd.DataFrame(report)
    rep.to_csv(REPORT, index=False)
    pd.set_option("display.width", 300)
    print(rep.to_string(index=False))
    print(f"\nTOTAL events={len(rep)} valid={int(rep.valid.sum())} frames={len(man)}")
    print("all monotonic:", rep.monotonic_offsets.all() and rep.monotonic_epochs.all())
    print("all deltas match:", rep.epoch_offset_deltas_match.all())
    print("all pre+post:", (rep.has_pre_smoke & rep.has_post_smoke).all())
    print("events with exact zero frame:", int(rep.has_zero_frame.sum()))
    print("count vs remote listing mismatches:",
          int((rep.n_frames != rep.n_listed_remote).sum()))

if __name__ == "__main__":
    main()
