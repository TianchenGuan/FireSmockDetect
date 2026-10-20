"""Parallel ranged download of a Zenodo file with md5 verification (resume-safe per chunk)."""
import os, sys, hashlib, urllib.request, time, json
from concurrent.futures import ThreadPoolExecutor
url, out, md5_expect = sys.argv[1], sys.argv[2], sys.argv[3]
nthreads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
req = urllib.request.Request(url, method='HEAD'); r = urllib.request.urlopen(req)
size = int(r.headers['Content-Length']); final = r.geturl()
chunk = 256 * 1024 * 1024
ranges = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]
done_file = out + '.chunks.json'
done = set(json.load(open(done_file))) if os.path.exists(done_file) else set()
if not os.path.exists(out):
    with open(out, 'wb') as f: f.truncate(size)
print(f'size {size/1e9:.2f} GB, {len(ranges)} chunks, {len(done)} already done', flush=True)
t0 = time.time(); got = [0]
def fetch(i):
    s, e = ranges[i]
    if i in done: return i
    for attempt in range(6):
        try:
            rq = urllib.request.Request(final, headers={'Range': f'bytes={s}-{e}'})
            with urllib.request.urlopen(rq, timeout=120) as resp, open(out, 'r+b') as f:
                f.seek(s); n = 0
                while True:
                    b = resp.read(8 * 1024 * 1024)
                    if not b: break
                    f.write(b); n += len(b)
            if n != e - s + 1: raise IOError(f'short chunk {n}')
            got[0] += n
            return i
        except Exception as ex:
            print(f'chunk {i} attempt {attempt} failed: {ex}', flush=True); time.sleep(5 * (attempt + 1))
    raise SystemExit(f'chunk {i} failed')
with ThreadPoolExecutor(nthreads) as ex:
    for k, i in enumerate(ex.map(fetch, range(len(ranges)))):
        done.add(i); json.dump(sorted(done), open(done_file, 'w'))
        if k % 8 == 0:
            el = time.time() - t0; print(f'{k+1}/{len(ranges)} chunks, {got[0]/1e9:.2f} GB new, {got[0]/1e6/max(el,1):.1f} MB/s', flush=True)
print('download complete; verifying md5', flush=True)
h = hashlib.md5()
with open(out, 'rb') as f:
    for b in iter(lambda: f.read(64 * 1024 * 1024), b''): h.update(b)
print('md5', h.hexdigest(), 'expected', md5_expect, 'OK' if h.hexdigest() == md5_expect else 'MISMATCH', flush=True)
