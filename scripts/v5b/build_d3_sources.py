#!/usr/bin/env python
"""V5b arm D3: assemble the UNLABELLED smoke-positive, train-eligible material to be pseudo-boxed by the teacher pool.
  (1) Boreal Subset-B TRAIN-burn clip frames (evo + karkkila; sequences_v4.parquet source=boreal split=train; 1 fps;
      video-level label: 104 clips smoke / 1 karkkila clip no_smoke [scored for the FP stat, never trained]); frames are
      3840/4096x2160 -> RESIZED copies, max side 1920 (JPEG q95): teachers and the student both run at 1280.
  (2) FIRESENSE smoke_pos frames (= sequences_v4 firesense split=train, 13 videos; staged near-field smoke; video-level).
  (3) D-Fire official-train FIRE-ONLY images (label file has only class-1 lines; 944; never in E; smoke unknown).
Writes data/v5b/yolo/D3_src/{images,sources.csv}, data/v5b/teacher/lists/{train_d3_sources.txt,TARGETS_D3.json} and the
teacher staging tar data/v5b/stage/D3_src.tar (dereferenced). Nothing here is a label."""
import csv, hashlib, json, os, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path('/local/home/tianrgua/cvproject')
SRC = ROOT / 'data/v5b/yolo/D3_src'
LISTS = ROOT / 'data/v5b/teacher/lists'
STAGE = ROOT / 'data/v5b/stage'
SEQ = ROOT / 'data/v4/sequences/sequences_v4.parquet'
FIRESENSE = ROOT / 'data/v4/raw/firesense/frames/smoke_pos'
DFIRE = ROOT / 'data/raw/dfire/extracted/train'
MAX_SIDE = 1920
POD_ROOT = '/scratch/scratch/tianrgua-sandbox/smoke/v5bdata/D3_src'


def resize_job(job):
    src, dst = job
    from PIL import Image
    im = Image.open(src); w, h = im.size
    s = MAX_SIDE / max(w, h)
    if s < 1:
        im = im.convert('RGB').resize((round(w * s), round(h * s)), Image.LANCZOS)
    im.save(dst, 'JPEG', quality=95)
    return (w, h, im.size[0], im.size[1])


def main():
    if SRC.exists():
        shutil.rmtree(SRC)
    (SRC / 'images').mkdir(parents=True)
    rows = []
    df = pd.read_parquet(SEQ)
    bt = df[(df.source == 'boreal') & (df.split == 'train')].sort_values(['seq_id', 'frame_idx'])
    assert set(bt.parent_unit) <= {'burn_evo', 'burn_karkkila'}, set(bt.parent_unit)
    jobs = []
    for r in bt.itertuples():
        name = f'borealclip__{Path(r.abs_path).name}'
        jobs.append((r.abs_path, str(SRC / 'images' / name)))
        rows.append(dict(name=name, local_src=r.abs_path, source='borealclip', unit=r.seq_id, parent_unit=r.parent_unit, frame_idx=int(r.frame_idx),
                         video_label=r.seq_label, video_level_support=str(r.seq_label == 'smoke').lower(), train_eligible=str(r.seq_label == 'smoke').lower(),
                         resized='true'))
    print(f'resizing {len(jobs)} boreal clip frames ...', flush=True)
    with ProcessPoolExecutor(max_workers=32) as ex:
        dims = list(ex.map(resize_job, jobs, chunksize=8))
    for row, (w, h, nw, nh) in zip(rows, dims):
        row.update(orig_w=w, orig_h=h, w=nw, h=nh)
    ft = df[(df.source == 'firesense') & (df.split == 'train')].sort_values(['seq_id', 'frame_idx'])
    assert all('/smoke_pos/' in p for p in ft.abs_path) and len(ft) == sum(len(os.listdir(FIRESENSE / v)) for v in os.listdir(FIRESENSE))
    from PIL import Image
    for r in ft.itertuples():
        name = f'firesense__{Path(r.abs_path).name}'
        os.symlink(r.abs_path, SRC / 'images' / name)
        w, h = Image.open(r.abs_path).size
        rows.append(dict(name=name, local_src=r.abs_path, source='firesense', unit=r.seq_id, parent_unit='firesense_smoke_pos', frame_idx=int(r.frame_idx),
                         video_label=r.seq_label, video_level_support='true', train_eligible='true', resized='false', orig_w=w, orig_h=h, w=w, h=h))
    E = {Path(l).stem for l in open(ROOT / 'data/yolo/lists/local/expB_train.txt') if 'dfire' in l}
    T = {Path(l.strip()).stem for l in open(ROOT / 'data/yolo/lists/local/dfire_test_raw.txt')}
    n_fire_only = 0
    for f in sorted(os.listdir(DFIRE / 'labels')):
        cs = {l.split()[0] for l in open(DFIRE / 'labels' / f) if l.strip()}
        if cs != {'1'}:
            continue
        stem = f[:-4]; assert stem not in E and stem not in T, stem
        img = DFIRE / 'images' / f'{stem}.jpg'; assert img.exists(), img
        name = f'dfirefire__{stem}.jpg'
        os.symlink(img, SRC / 'images' / name)
        w, h = Image.open(img).size
        rows.append(dict(name=name, local_src=str(img), source='dfirefire', unit=stem, parent_unit='dfire_official_train_fire_only', frame_idx=0,
                         video_label='fire_only_smoke_unknown', video_level_support='false', train_eligible='true', resized='false', orig_w=w, orig_h=h, w=w, h=h))
        n_fire_only += 1
    assert n_fire_only == 944, n_fire_only
    with open(SRC / 'sources.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    paths = [str(SRC / 'images' / r['name']) for r in rows]
    assert len(set(paths)) == len(paths) and all(os.path.exists(p) for p in paths)
    LISTS.mkdir(parents=True, exist_ok=True)
    (LISTS / 'train_d3_sources.txt').write_text('\n'.join(paths) + '\n')
    sha = hashlib.sha256((LISTS / 'train_d3_sources.txt').read_bytes()).hexdigest()
    # leak check: basenames of originals vs every eval list / boreal val+test units / firesense val
    bn = {Path(r['local_src']).name for r in rows}
    leak = {}
    for lf in sorted((ROOT / 'data/yolo/lists/local').glob('*.txt')):
        if 'train' in lf.name: continue
        ls = [l.strip() for l in open(lf) if l.strip()]
        leak[str(lf.relative_to(ROOT))] = dict(n=len(ls), basename_overlap=len({Path(x).name for x in ls} & bn))
    other = df[~((df.source == 'boreal') & (df.split == 'train')) & ~((df.source == 'firesense') & (df.split == 'train'))]
    leak['sequences_v4 non-train-boreal/firesense rows (heinola val, firesense smoke_neg, ausmoke...)'] = dict(n=len(other), basename_overlap=len({Path(p).name for p in other.abs_path} & bn))
    leak['boreal parent_units_used'] = sorted(set(bt.parent_unit))
    counts = {s: sum(1 for r in rows if r['source'] == s) for s in ('borealclip', 'firesense', 'dfirefire')}
    counts['borealclip_no_smoke_clip_frames_scored_not_trained'] = sum(1 for r in rows if r['source'] == 'borealclip' and r['video_label'] == 'no_smoke')
    counts['borealclip_clips'] = {'smoke': int(bt[bt.seq_label == 'smoke'].seq_id.nunique()), 'no_smoke': int(bt[bt.seq_label == 'no_smoke'].seq_id.nunique())}
    counts['firesense_videos'] = int(ft.seq_id.nunique()); counts['total'] = len(rows)
    print('tar ...', flush=True)
    tar = STAGE / 'D3_src.tar'
    subprocess.run(['tar', '-chf', str(tar), '-C', str(ROOT / 'data/v5b/yolo'), 'D3_src'], check=True)
    h = hashlib.sha256()
    with open(tar, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    members = subprocess.run(['tar', '-tf', str(tar)], check=True, capture_output=True, text=True).stdout
    (STAGE / 'D3_src.tar.members.txt').write_text(members)
    man = dict(purpose='V5b arm D3 teacher-pool TARGETS: UNLABELLED smoke-positive train-eligible material (video-level or fire-only); model-assisted boxes will be produced on these; NOT labels',
               created_utc=datetime.now(timezone.utc).strftime('%FT%TZ'), list='data/v5b/teacher/lists/train_d3_sources.txt', sha256=sha, counts=counts,
               sources=dict(borealclip=f'sequences_v4.parquet source=boreal split=train (evo+karkkila Subset-B clips, 1 fps), RESIZED copies max side {MAX_SIDE} JPEG q95',
                            firesense='sequences_v4.parquet source=firesense split=train = frames/smoke_pos (13 videos; staged near-field smoke; video-level)',
                            dfirefire='data/raw/dfire/extracted/train labels with only class 1 (fire) -> 944 images never in E; fire present, smoke unknown'),
               sources_csv=str(SRC / 'sources.csv'), rootmap_local_to_pod=f'{SRC}={POD_ROOT}',
               tar=dict(path=str(tar), bytes=tar.stat().st_size, sha256=h.hexdigest(), n_members=len(members.split()) - 1, kraken_data_tar_arg='v5b_d3src=data/v5b/stage/D3_src.tar',
                        pod_extract_root='/scratch/scratch/tianrgua-sandbox/smoke/v5bdata'),
               leak_check=leak)
    (LISTS / 'TARGETS_D3.json').write_text(json.dumps(man, indent=1))
    print(json.dumps(counts), 'tar MB', round(tar.stat().st_size / 1e6), 'sha', h.hexdigest()[:16])
    bad = {k: v for k, v in leak.items() if isinstance(v, dict) and v.get('basename_overlap')}
    print('LEAKS:', bad if bad else 'none')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
