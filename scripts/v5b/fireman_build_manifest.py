"""Build data/v5b/manifests/fireman_manifest.parquet from the extracted FireMan-UAV-RGBT Multiclass package.
Usage: python scripts/v5b/fireman_build_manifest.py [--units data/v5b/manifests/fireman_video_units.json]
Requires: data/v5b/manifests/_cache/ref_hashes.parquet (scripts/v5b/build_ref_hashes.py)."""
import os, sys, json, hashlib, argparse, collections
import numpy as np, pandas as pd
from multiprocessing import Pool
from PIL import Image
import imagehash
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fireman_parse_cvat import parse_cvat
ROOT = '/local/home/tianrgua/cvproject'
RAW = f'{ROOT}/data/v5b/raw/fireman/Multiclass/Unimodal'
OUT = f'{ROOT}/data/v5b/manifests'
LICENCE = 'CC BY 4.0'

def hash_one(p):
    try:
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
        with Image.open(p) as im:
            w, h2 = im.size; ph = str(imagehash.phash(im.convert('RGB')))
        return (p, h.hexdigest(), ph, w, h2, '')
    except Exception as e:
        return (p, '', '', 0, 0, str(e)[:80])

def hex2u64(s):
    return np.array([int(x, 16) if x else 0 for x in s], dtype=np.uint64)

def popcount64(a):
    a = a.astype(np.uint64)
    return np.unpackbits(a.view(np.uint8).reshape(-1, 8), axis=1).sum(1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--units', default=f'{OUT}/fireman_video_units.json'); ap.add_argument('--procs', type=int, default=128)
    a = ap.parse_args()
    units = json.load(open(a.units)) if os.path.exists(a.units) else {}
    if not units: print('WARNING: no units file; parent_unit will be video-level and split=unassigned', flush=True)
    rows = []; video_stats = []
    for modality in ('rgb', 'thermal'):
        for split in ('train', 'val', 'test'):
            d = f'{RAW}/{modality}/{split}'
            if not os.path.isdir(d): continue
            for vid in sorted(os.listdir(d)):
                vd = f'{d}/{vid}'; xmlp = f'{vd}/annotations.xml'
                imgs = sorted(f for f in os.listdir(f'{vd}/images') if f.lower().endswith('.jpg'))
                info, frames = parse_cvat(xmlp) if os.path.exists(xmlp) else ({'task_name': None}, [])
                fmap = {f['name']: f for f in frames}
                stems = {os.path.splitext(f)[0] for f in imgs}
                xml_only = sorted(set(fmap) - stems); img_only = sorted(stems - set(fmap))
                key = f'{modality}/{split}/{vid}'; u = units.get(key, {})
                video_stats.append({'video': key, 'task_name': info.get('task_name'), 'n_images': len(imgs), 'n_xml_frames': len(frames),
                                    'n_xml_without_image': len(xml_only), 'n_image_without_xml': len(img_only), **{k: u.get(k) for k in ('burn', 'date', 'location', 'paired_video', 'evidence')}})
                for i, f in enumerate(imgs):
                    stem = os.path.splitext(f)[0]; fr = fmap.get(stem)
                    rel = f'Multiclass/Unimodal/{key}/images/{f}'
                    r = {'source': 'fireman', 'rel_path': rel, 'modality': modality, 'official_split': split, 'video_id': key, 'frame_idx': i,
                         'parent_unit': u.get('burn', f'fireman_video:{key}'), 'burn_date': u.get('date'), 'burn_location': u.get('location'), 'paired_video': u.get('paired_video')}
                    if fr is None:
                        r.update(label_type='none', has_smoke_label=False, has_fire_label=False, is_explicit_negative=False, n_smoke_boxes=0, n_fire_boxes=0, n_other_boxes=0,
                                 annotation_origin='unlabeled', author_box_source='none', smoke_presence='ABSENT', fire_presence='ABSENT', smoke_intensity=None,
                                 camera_perspective=None, smoke_interaction=None, weather_condition=None, sky_brightness=None, label_conflict='no_xml_entry', smoke_boxes=json.dumps([]))
                    else:
                        sb = fr['boxes'].get('smoke_region', []); fb = fr['boxes'].get('fire_region', [])
                        other = sum(len(v) for k, v in fr['boxes'].items() if k not in ('smoke_region', 'fire_region'))
                        st = fr['tags'].get('smoke') or fr['tags'].get('smoke_tag') or {}; ft = fr['tags'].get('fire') or fr['tags'].get('fire_tag') or {}
                        sp = st.get('smoke_presence', 'ABSENT'); fp = ft.get('fire_presence', 'ABSENT')
                        srcs = {b[4] for b in sb}; abs_ = 'none' if not sb else ('mixed' if len(srcs) > 1 else next(iter(srcs)))
                        conflict = ''
                        if sp == 'no' and sb: conflict = 'presence_no_but_boxes'
                        elif sp == 'yes' and not sb: conflict = 'presence_yes_no_boxes'
                        elif sp in ('ABSENT', 'unknown') and sb: conflict = 'boxes_without_presence_tag'
                        if fr['tag_conflict']: conflict = (conflict + ';' if conflict else '') + 'tag_conflict:' + ','.join(fr['tag_conflict'])
                        has_smoke = (sp == 'yes') or bool(sb)
                        label_type = 'box' if sb else ('image' if sp in ('yes', 'no') else 'none')
                        expl_neg = (sp == 'no') and not sb
                        origin = 'author' if (sp in ('yes', 'no') or sb or fb or fp in ('yes', 'no')) else 'unlabeled'
                        cam = fr['tags'].get('camera', {}); wx = fr['tags'].get('weather', {})
                        r.update(label_type=label_type, has_smoke_label=has_smoke, has_fire_label=(fp == 'yes') or bool(fb), is_explicit_negative=expl_neg,
                                 n_smoke_boxes=len(sb), n_fire_boxes=len(fb), n_other_boxes=other, annotation_origin=origin, author_box_source=abs_,
                                 smoke_presence=sp, fire_presence=fp, smoke_intensity=st.get('smoke_intensity'), camera_perspective=cam.get('camera_perspective'),
                                 smoke_interaction=cam.get('smoke_interaction'), weather_condition=wx.get('condition'), sky_brightness=wx.get('sky_brightness'),
                                 label_conflict=conflict, smoke_boxes=json.dumps([[round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)] for b in sb]),
                                 xml_width=fr['width'], xml_height=fr['height'])
                    rows.append(r)
    df = pd.DataFrame(rows); print('rows', len(df), flush=True)
    pd.DataFrame(video_stats).to_csv(f'{OUT}/fireman_video_stats.csv', index=False)
    # hashes
    paths = [f'{ROOT}/data/v5b/raw/fireman/{p}' for p in df.rel_path]
    cache_p = f'{OUT}/_cache/fireman_hashes.parquet'; hm = {}
    if os.path.exists(cache_p):
        c = pd.read_parquet(cache_p); hm = {r.path: (r.sha256, r.phash, int(r.width), int(r.height), r.decode_err) for r in c.itertuples()}
    todo = [p for p in paths if p not in hm]; print('hashing', len(todo), 'cached', len(hm), flush=True)
    if todo:
        with Pool(a.procs) as pool: res = pool.map(hash_one, todo, chunksize=32)
        for r in res: hm[r[0]] = tuple(r[1:])
        pd.DataFrame([{'path': k, 'sha256': v[0], 'phash': v[1], 'width': v[2], 'height': v[3], 'decode_err': v[4]} for k, v in hm.items()]).to_parquet(cache_p, index=False)
    df['sha256'] = [hm[p][0] for p in paths]; df['phash'] = [hm[p][1] for p in paths]; df['width'] = [hm[p][2] for p in paths]; df['height'] = [hm[p][3] for p in paths]; df['decode_err'] = [hm[p][4] for p in paths]
    print('decode errors', (df.decode_err != '').sum(), flush=True)
    # dedup vs reference
    ref = pd.read_parquet(f'{OUT}/_cache/ref_hashes.parquet'); ref = ref[ref.phash.astype(str).str.len() == 16].reset_index(drop=True)
    sha2ref = dict(zip(ref.sha256, ref.index))
    rq = hex2u64(df.phash.tolist()); rr = hex2u64(ref.phash.tolist())
    best_i = np.full(len(df), -1); best_d = np.full(len(df), 99)
    for s in range(0, len(df), 512):
        q = rq[s:s + 512]
        dist = popcount64((q[:, None] ^ rr[None, :]).ravel()).reshape(len(q), -1)
        bi = dist.argmin(1); best_i[s:s + 512] = bi; best_d[s:s + 512] = dist[np.arange(len(q)), bi]
    dup_class = []; dup_role = []; dup_ref = []; dup_dist = []
    for k, (sha, ph) in enumerate(zip(df.sha256, df.phash)):
        if sha in sha2ref:
            j = sha2ref[sha]; kind = 'exact'
        elif ph and best_d[k] <= 6:
            j = best_i[k]; kind = 'near'
        else:
            dup_class.append('none'); dup_role.append(''); dup_ref.append(''); dup_dist.append(int(best_d[k]) if ph else -1); continue
        src = ref.ref_source.iat[j]
        dup_class.append('dup_multinat' if src == 'multinatsmoke' else f'dup_{kind}_{src}'); dup_role.append(ref.ref_role.iat[j]); dup_ref.append(ref.path.iat[j].replace(ROOT + '/', '')); dup_dist.append(0 if kind == 'exact' else int(best_d[k]))
    df['dup_class'] = dup_class; df['dup_ref_role'] = dup_role; df['dup_ref_path'] = dup_ref; df['phash_min_dist_ref'] = dup_dist
    df['dup_note'] = np.where((df.dup_class == 'dup_near_dfire') & (df.phash_min_dist_ref == 6), 'phash_d6_visually_checked_false_positive(dfire WEB02564/WEB04678 unrelated scenes)', '')
    # within-dataset exact duplicates
    df['dup_within_fireman'] = df.duplicated('sha256', keep=False) & (df.sha256 != '')
    # split: unit-grouped, inherited roles
    unit_split = {k: v.get('proposed_split') for k, v in units.items() if v.get('proposed_split')}
    burn_split = {}
    for k, v in units.items():
        if v.get('burn') and v.get('proposed_split'): burn_split[v['burn']] = v['proposed_split']
    ps = []
    for k, r in df.iterrows():
        s = burn_split.get(r.parent_unit) or unit_split.get(r.video_id) or 'unassigned'
        if r.dup_class != 'none':
            role = r.dup_ref_role
            if role == 'quarantine': s = 'quarantine'
            elif role == 'val' and s != 'quarantine': s = 'val'
        ps.append(s)
    df['proposed_split'] = ps
    df['licence'] = LICENCE; df['permission_status'] = 'confirmed'
    cols = ['source', 'rel_path', 'sha256', 'phash', 'parent_unit', 'label_type', 'has_smoke_label', 'has_fire_label', 'is_explicit_negative', 'n_smoke_boxes', 'annotation_origin', 'licence', 'permission_status', 'dup_class', 'proposed_split']
    rest = [c for c in df.columns if c not in cols]
    df = df[cols + rest]
    df.to_parquet(f'{OUT}/fireman_manifest.parquet', index=False)
    print(df.groupby(['modality', 'proposed_split']).size().to_string())
    print(df[df.modality == 'rgb'].groupby(['label_type', 'is_explicit_negative']).size().to_string())
    print('dup_class:', df.dup_class.value_counts().to_dict())
    print('conflicts:', df.label_conflict.value_counts().to_dict())
if __name__ == '__main__':
    main()
