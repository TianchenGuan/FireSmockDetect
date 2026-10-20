#!/usr/bin/env python
"""Compute phash (imagehash.phash, 64-bit, v4 convention) for every row of the v4 hash cache that lacks one.
Writes data/v5b/manifests/_cache/phash_v5b.parquet [path, sha256, phash]. Never modifies the v4 cache."""
import pandas as pd, os, sys
from multiprocessing import Pool
from PIL import Image
import imagehash
ROOT='/local/home/tianrgua/cvproject'
def ph(p):
    try:
        with Image.open(p) as im: return p, str(imagehash.phash(im))
    except Exception as e: return p, ''
if __name__=='__main__':
    c=pd.read_parquet(f'{ROOT}/data/v4/manifests/_cache/hash_cache.parquet')
    todo=c[(c.phash=='')&(c.sha256!='')]
    print('to hash', len(todo), flush=True)
    with Pool(96) as pool: res=pool.map(ph, todo.path.tolist(), chunksize=64)
    d=dict(res); out=pd.DataFrame({'path':todo.path.values,'sha256':todo.sha256.values,'phash':[d.get(p,'') for p in todo.path]})
    out.to_parquet(f'{ROOT}/data/v5b/manifests/_cache/phash_v5b.parquet', index=False); print('done', len(out), 'empty', (out.phash=='').sum())
