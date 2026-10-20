#!/usr/bin/env python3
"""Select FIgLib LOCKBOX v3 candidates (iteration 3 final-evaluation set).

Follows the iteration-2 lockbox recipe (see data/v2/manifests/LOCKBOX_FROZEN.md):
deterministic seeded selection from data/manifests/figlib_events_index.csv of
~25 fresh events, disjoint from ALL previously used FIgLib events
(iteration-1 39-event set + iteration-2 25-event lockbox = 64 events) by
event_id, camera_id, AND site+view (camera_id with the -mobo[-c|-m] / -iqeye
hardware suffix stripped; 'tdllns' normalized to 'tdlln-s'), spread across
years, with distinct cameras / views / dates inside the new set.

Selection is file-metadata only; no image content is read.
"""
import re
import numpy as np
import pandas as pd

ROOT = "/local/home/tianrgua/cvproject"
IDX = f"{ROOT}/data/manifests/figlib_events_index.csv"
USED_MANIFESTS = [
    f"{ROOT}/data/manifests/figlib_subset_manifest.parquet",     # iter-1, 39 events
    f"{ROOT}/data/v2/manifests/figlib_lockbox_manifest.parquet", # iter-2 lockbox, 25 events
]
OUT = f"{ROOT}/data/v3/manifests/lockbox_v3_candidates.csv"

SEED = 20260913
N_TARGET = 25
PER_YEAR_MIN = 2
BACKUPS_PER_YEAR = 2


def view_key(cam: str) -> str:
    """site+view: strip camera-hardware suffix; normalize tdllns naming."""
    v = re.sub(r"-(mobo(-[cm])?|iqeye)$", "", str(cam))
    return "tdlln-s" if v == "tdllns" else v


def main():
    idx = pd.read_csv(IDX)
    idx["view"] = idx.camera_id.map(view_key)
    idx["year"] = idx.date.astype(str).str[:4].astype(int)

    used_ev, used_cam = set(), set()
    for p in USED_MANIFESTS:
        m = pd.read_parquet(p)
        used_ev |= set(m.event_id.unique())
        used_cam |= set(m.camera_id.unique())
    used_view = {view_key(c) for c in used_cam}
    print(f"used: {len(used_ev)} events, {len(used_cam)} cameras, {len(used_view)} site+views")

    ok = idx[idx.listing_ok
             & (idx.n_frames_listed == idx.n_frames_with_offset)
             & (idx.min_offset_s < 0) & (idx.max_offset_s > 0)].copy()
    pool = ok[~ok.event_id.isin(used_ev)
              & ~ok.camera_id.isin(used_cam)
              & ~ok.view.isin(used_view)].copy()
    print(f"quality-pass {len(ok)}/{len(idx)}; disjoint pool {len(pool)} events, "
          f"{pool.camera_id.nunique()} cams, {pool.view.nunique()} views, "
          f"years {sorted(pool.year.unique())}")

    # deterministic per-year order
    rng = np.random.default_rng(SEED)
    pool = pool.sort_values("event_id").reset_index(drop=True)
    pool["rand"] = rng.random(len(pool))
    pool = pool.sort_values(["year", "rand"]).reset_index(drop=True)

    sel_cam, sel_view, sel_date = set(), set(), set()
    primary, backup = [], []

    def try_take(row, dest):
        # NB: row["view"] not row.view — .view is a pandas Series method.
        if (row["camera_id"] in sel_cam or row["view"] in sel_view
                or row["date"] in sel_date):
            return False
        sel_cam.add(row["camera_id"]); sel_view.add(row["view"])
        sel_date.add(row["date"])
        dest.append(row["event_id"])
        return True

    years = sorted(pool.year.unique())
    by_year = {y: pool[pool.year == y] for y in years}
    cursor = {y: 0 for y in years}
    # scarce years pick first in pass 1: some views are contested across years
    # (e.g. tp-w needed by both 2021 and 2022; smer-tcs3 by 2016 and 2022),
    # and flexible years can route around them.
    scarcity = sorted(years, key=lambda y: (by_year[y].view.nunique(),
                                            by_year[y].date.nunique(),
                                            len(by_year[y]), y))

    def next_valid(y, dest):
        g = by_year[y]
        while cursor[y] < len(g):
            row = g.iloc[cursor[y]]
            cursor[y] += 1
            if try_take(row, dest):
                return True
        return False

    # pass 1: PER_YEAR_MIN per year, scarcest years first
    for _ in range(PER_YEAR_MIN):
        for y in scarcity:
            if len(primary) < N_TARGET:
                next_valid(y, primary)
    # pass 2: round-robin extras up to N_TARGET
    progressed = True
    while len(primary) < N_TARGET and progressed:
        progressed = False
        for y in years:
            if len(primary) >= N_TARGET:
                break
            if next_valid(y, primary):
                progressed = True
    # backups: same constraints (also disjoint from primaries)
    for _ in range(BACKUPS_PER_YEAR):
        for y in years:
            next_valid(y, backup)

    out = pool[pool.event_id.isin(primary + backup)].copy()
    out["cd"] = list(zip(out.camera_id, out.date.astype(str)))
    out["role"] = np.where(out.event_id.isin(primary), "primary", "backup")
    out = out.drop(columns=["rand"]).sort_values(["role", "year", "event_id"],
                                                 ascending=[False, True, True])
    out.to_csv(OUT, index=False)

    p = out[out.role == "primary"]
    print(f"\nprimary: {len(p)} events, {p.camera_id.nunique()} cams, "
          f"{p.view.nunique()} views, {p.date.nunique()} dates")
    print(p.groupby("year").size().to_string())
    print(f"backup: {len(out) - len(p)} events")
    print(f"total tgz bytes (primary): {p.tgz_size_bytes.sum()/1e9:.2f} GB")
    print(f"wrote {OUT}")
    print("\nPRIMARY EVENTS:")
    for e in sorted(p.event_id):
        print(" ", e)


if __name__ == "__main__":
    main()
