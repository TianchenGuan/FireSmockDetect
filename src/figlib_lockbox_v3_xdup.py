#!/usr/bin/env python3
"""Cross-duplicate spot-check: LOCKBOX v3 vs BOTH previously used FIgLib sets.

Same protocol/thresholds as the iteration-2 audit
(data/v2/manifests/xdup_lockbox_vs_iter1_figlib.json): 200 seeded-random v3
lockbox frames hashed (sha256 + phash) against ALL frames of the iteration-1
39-event set AND the iteration-2 25-event lockbox. Flags: exact sha256,
phash hamming==0, near hamming<=6. Pairs with hamming<=10 are listed for
transparency only (shared-landscape similarity across different cameras is
NOT a duplicate). Hash computation only; no frame is rendered or inspected.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image
import imagehash

ROOT = "/local/home/tianrgua/cvproject"
NEW_PQ = f"{ROOT}/data/v3/manifests/figlib_lockbox_v3_manifest.parquet"
OLD_SETS = {
    "iter1_figlib": f"{ROOT}/data/manifests/figlib_subset_manifest.parquet",
    "iter2_lockbox": f"{ROOT}/data/v2/manifests/figlib_lockbox_manifest.parquet",
}
OUT = f"{ROOT}/data/v3/manifests/xdup_lockbox_v3_vs_prev_figlib.json"
SEED = 20260913
N_SAMPLE = 200
NEAR_H = 6
REPORT_H = 10


def phash_bits(path):
    with Image.open(path) as im:
        return np.array(imagehash.phash(im).hash, dtype=bool).ravel()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    new = pd.read_parquet(NEW_PQ)
    rng = np.random.default_rng(SEED)
    samp = new.iloc[np.sort(rng.choice(len(new), size=N_SAMPLE, replace=False))]
    print(f"sampled {len(samp)} of {len(new)} v3 frames (seed {SEED})")

    new_hash = np.stack([phash_bits(f"{ROOT}/{p}") for p in samp.frame_path])
    new_sha = {sha256(f"{ROOT}/{p}") for p in samp.frame_path}
    new_cam = samp.camera_id.to_numpy()

    result = dict(n_sample=int(len(samp)), seed=SEED, sets={})
    for name, pq in OLD_SETS.items():
        old = pd.read_parquet(pq)
        old_hash = np.stack([phash_bits(f"{ROOT}/{p}") for p in old.frame_path])
        old_sha = {sha256(f"{ROOT}/{p}") for p in old.frame_path}
        old_cam = old.camera_id.to_numpy()
        # hamming: (200, n_old)
        d = (new_hash[:, None, :] != old_hash[None, :, :]).sum(axis=2)
        pairs = {}
        for i, j in zip(*np.where(d <= REPORT_H)):
            key = (new_cam[i], old_cam[j])
            pairs[key] = pairs.get(key, 0) + 1
        result["sets"][name] = dict(
            n_old=int(len(old)),
            sha256_exact=int(len(new_sha & old_sha)),
            phash_exact_h0=int((d == 0).sum()),
            phash_near_h_le6=int((d <= NEAR_H).sum()),
            min_hamming=int(d.min()),
            h_le10_pairs=[dict(lb_cam=a, old_cam=b, n=n)
                          for (a, b), n in sorted(pairs.items())],
        )
        r = result["sets"][name]
        print(f"{name}: n_old={r['n_old']} sha_exact={r['sha256_exact']} "
              f"h0={r['phash_exact_h0']} h<=6={r['phash_near_h_le6']} "
              f"min_h={r['min_hamming']} pairs_h<=10={len(r['h_le10_pairs'])}")
    result["note"] = ("thresholds match iteration-1/2 xdup audits (exact + near "
                      "hamming<=6); h<=10 pairs listed for transparency only — "
                      "shared-landscape similarity across different cameras is "
                      "not a duplicate")
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
