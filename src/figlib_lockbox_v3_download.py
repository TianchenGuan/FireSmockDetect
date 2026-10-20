#!/usr/bin/env python3
"""Download + extract FIgLib LOCKBOX v3 primary events (file-level only).

Mirrors the iteration-2 lockbox download: fetch each event tarball from the
HPWREN CDN, verify byte-exact size against the events index
(tgz_size_bytes), keep tarballs in _tars/, extract frames to
data/v3/figlib_lockbox/<event_id>/. No image content is inspected.
"""
import os
import subprocess
import sys
import tarfile
import pandas as pd

ROOT = "/local/home/tianrgua/cvproject"
CAND = f"{ROOT}/data/v3/manifests/lockbox_v3_candidates.csv"
RAW = f"{ROOT}/data/v3/figlib_lockbox"
TARS = f"{RAW}/_tars"
LOG = f"{ROOT}/logs/figlib_lockbox_v3_download.log"


def main():
    os.makedirs(TARS, exist_ok=True)
    cand = pd.read_csv(CAND)
    prim = cand[cand.role == "primary"]
    failures = []
    with open(LOG, "a") as log:
        for _, r in prim.iterrows():
            ev, url, want = r["event_id"], r["url_tgz"], int(r["tgz_size_bytes"])
            tgz = os.path.join(TARS, f"{ev}.tgz")
            evdir = os.path.join(RAW, ev)
            if os.path.isdir(evdir) and os.path.exists(tgz) \
                    and os.path.getsize(tgz) == want:
                print(f"SKIP (done) {ev}"); log.write(f"SKIP {ev}\n"); continue
            ok = False
            for attempt in range(3):
                rc = subprocess.run(
                    ["curl", "-fsSL", "--retry", "3", "-o", tgz, url],
                    capture_output=True, text=True)
                if rc.returncode == 0 and os.path.getsize(tgz) == want:
                    ok = True
                    break
                print(f"  retry {attempt+1} {ev}: rc={rc.returncode} "
                      f"size={os.path.getsize(tgz) if os.path.exists(tgz) else 'NA'}"
                      f" want={want}", flush=True)
            if not ok:
                failures.append(ev)
                print(f"FAIL {ev}"); log.write(f"FAIL {ev}\n"); continue
            # extract. Older tarballs contain <event_id>/<epoch>_<offset>.jpg;
            # 2024+ tarballs nest it as
            # Data/HPWREN-FIgLib/HPWREN-FIgLib-Data/<event_id>/... — normalize.
            with tarfile.open(tgz, "r:gz") as tf:
                tf.extractall(RAW, filter="data")
            nested = os.path.join(RAW, "Data", "HPWREN-FIgLib",
                                  "HPWREN-FIgLib-Data", ev)
            if os.path.isdir(nested) and not os.path.isdir(evdir):
                os.rename(nested, evdir)
            if os.path.isdir(os.path.join(RAW, "Data")):
                # remove the (now empty) nesting dirs if nothing is left
                for d in (os.path.join(RAW, "Data", "HPWREN-FIgLib",
                                       "HPWREN-FIgLib-Data"),
                          os.path.join(RAW, "Data", "HPWREN-FIgLib"),
                          os.path.join(RAW, "Data")):
                    try:
                        os.rmdir(d)
                    except OSError:
                        pass
            if not os.path.isdir(evdir):
                failures.append(ev)
                print(f"FAIL (layout) {ev}"); log.write(f"FAIL layout {ev}\n")
                continue
            print(f"OK {ev}", flush=True); log.write(f"OK {ev}\n")
    print(f"\ndone: {len(prim) - len(failures)}/{len(prim)} OK")
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
