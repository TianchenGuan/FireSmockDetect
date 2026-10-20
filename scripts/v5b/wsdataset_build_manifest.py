#!/usr/bin/env python
"""V5b data agent — WSDataset (gloryvu).  Builds data/v5b/manifests/wsdataset_manifest.parquet.

Material:
  (A) MultiNatSmoke copies of the 11,539 WSDataset positives (data/v4/raw/multinatsmoke/...),
      640x640 re-encodes with MODEL-ASSISTED masks (not author boxes) -> dup_multinat, NOT new data.
  (B) Author GitHub demo (GloryVu/VMOKED @731b65b, dataset/demo): 214 unique images, author labels
      (folder = class; detection labels empty = explicit negatives); test/ is a byte-identical copy of train/.
Original Kaggle archive (18.4 GB, author boxes + 12,481 nonsmoke frames): BLOCKED (download needs Kaggle login).
Unit recovery for (A): original video ids parsed from the unauthenticated Kaggle file listing
(wildfire_smoke_dataset/.../detection/<split>/images/<smoke_NN>_<frame>.jpg) matched to MNS filenames.
Dedup: exact sha256 + phash (hamming<=6, v4 convention) vs data/v4/manifests/_cache/hash_cache.parquet
(pyro-sdis, D-Fire, FIgLib subset, FIgLib lockbox, Boreal, firesense, all MultiNatSmoke usable rows).
"""
import argparse, hashlib, json, os, re, sys, collections, random
import numpy as np, pandas as pd
from PIL import Image
import imagehash

ROOT = '/local/home/tianrgua/cvproject'
MNS_RAW = f'{ROOT}/data/v4/raw/multinatsmoke'
GH_RAW = f'{ROOT}/data/v5b/raw/wsdataset_github_demo'
OUT_DIR = f'{ROOT}/data/v5b/manifests'
YOLO_DIR = f'{ROOT}/data/v5b/yolo/wsdataset'
V4_LBL = f'{ROOT}/data/v4/yolo/B_G_v1/labels'
PHASH_MAX = 6
LIC_MNS = ('MIT per MultiNatSmoke README table (second-hand); author Kaggle page metadata licenseName="MIT" '
           '(api/v1/datasets/view, 2026-10-02); MIT text itself not inspected (archive download blocked); '
           'MultiNatSmoke aggregate redistribution carries no licence')
LIC_GH = 'none stated (GitHub GloryVu/VMOKED @731b65b: no LICENSE file, API license=null)'
BOX_CONV = ('mask->box via scripts/v4/build_corpus.py:mask_to_lines (thresh>=128, 8-conn components, '
            'min_area=max(9,2e-5*W*H), >8 comps -> morphological close, keep <=50 largest)')

def hex2u64(h):
    return np.array([int(x, 16) if isinstance(x, str) and len(x) == 16 else 0 for x in h], dtype=np.uint64)

POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
def popcount64(a):
    v = a.view(np.uint8).reshape(a.shape + (8,))
    return POP[v].sum(-1)

def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def load_roles():
    """abs path -> role tag for every hashed image we hold."""
    roles = {}
    for f, tag in [('data/yolo/lists/local/sdis_val.txt', 'eval:pyro_sdis_val'), ('data/yolo/lists/local/sdis_test.txt', 'eval:pyro_sdis_test')]:
        for ln in open(f'{ROOT}/{f}'):
            roles[ln.strip()] = tag
    bm = pd.read_parquet(f'{ROOT}/data/v4/manifests/boreal_manifest.parquet')
    for rp, u in zip(bm.rel_path, bm.parent_unit):
        roles[f'{ROOT}/data/v4/raw/boreal/{rp}'] = {'burn_heinola': 'eval:boreal_heinola_val', 'burn_ruokolahti': 'eval:boreal_ruokolahti_test'}.get(u, f'train:boreal_{u}')
    mm = pd.read_parquet(f'{ROOT}/data/v4/manifests/multinatsmoke_manifest.parquet')
    fh = json.load(open(f'{ROOT}/data/v4/manifests/forest_holdout_v4.json'))
    aus_role = {}
    for split in ('val', 'test'):
        for g in fh['assignment'][split]:
            for u in fh['groups'][g]:
                aus_role[u] = f'eval:ausmoke_{split}'
    for rp, sub, oc, pu in zip(mm.rel_path, mm.subset, mm.origin_class, mm.parent_unit):
        p = f'{MNS_RAW}/{rp}'
        if sub == 'WSDataset':
            roles[p] = 'self:wsdataset'
        elif oc.startswith('quarantine') or sub in ('FIgLib', 'SmokeSeg', 'aiformankind', 'firecam'):
            roles[p] = f'quarantine:mns_{sub}'
        elif sub == 'AuSmoke':
            roles[p] = aus_role.get(pu, 'train:mns_AuSmoke')
        else:
            roles[p] = f'train:mns_{sub}'
    return roles

def role_of(path, roles):
    if path in roles:
        return roles[path]
    if '/data/raw/figlib/' in path: return 'eval:figlib_subset'
    if '/figlib_lockbox/' in path: return 'eval:figlib_lockbox'
    if '/data/yolo/pyro_sdis/' in path: return 'train:pyro_sdis'
    if '/data/raw/dfire/' in path: return 'train:dfire'
    if '/firesense/' in path: return 'train:firesense'
    if '/multinatsmoke/' in path: return 'train:mns_other'
    return 'other'

def parse_kaggle(path):
    """Kaggle listing -> (summary, name->(unit,split) maps).  Layout (verified from the unauthenticated
    api/v1/datasets/list listing): wildfire_smoke_dataset/wildfire_smoke_dataset/
        classification/{train,test}/{smoke,nonsmoke}/<unit>/<frame>.jpg      (unit = video segment)
        {train,test}/{images,labels}/<unit>_<frame>.{jpg,txt}                 (detection, YOLO txt)
    MNS Train copies carry the classification <frame> name (unit folder lost); MNS Test copies carry the
    detection <unit>_<frame> name."""
    if not path or not os.path.exists(path):
        return None, {}, {}
    rows = [json.loads(l) for l in open(path)]
    P = 'wildfire_smoke_dataset/wildfire_smoke_dataset/'
    cls = collections.defaultdict(lambda: {'frames': 0, 'bytes': 0}); det = collections.defaultdict(lambda: {'images': 0, 'labels': 0, 'bytes': 0})
    frame_map, exact_map, other = {}, {}, []
    det_noprefix = {}
    known_units = set()
    for r in rows:
        m = re.match(re.escape(P) + r'classification/(train|test)/(smoke|nonsmoke)/([^/]+)/', r['name'])
        if m: known_units.add(m.group(3))
    for r in rows:
        n = r['name']; b = r.get('totalBytes') or 0
        m = re.match(re.escape(P) + r'classification/(train|test)/(smoke|nonsmoke)/([^/]+)/([^/]+)$', n)
        if m:
            split, cl, unit, fn = m.groups(); cls[(split, cl, unit)]['frames'] += 1; cls[(split, cl, unit)]['bytes'] += b
            stem = os.path.splitext(fn)[0]
            frame_map.setdefault(stem, set()).add((unit, split, cl)); exact_map.setdefault(f'{unit}_{stem}', set()).add((unit, split, cl))
            continue
        m = re.match(re.escape(P) + r'(train|test)/(images|labels)/([^/]+)$', n)
        if m:
            split, kind, fn = m.groups(); stem = os.path.splitext(fn)[0]
            um = re.match(r'(smoke_\d+(?:_\d+)?|non_?smoke_\d+)_(.+)$', stem)
            if um and um.group(1) not in known_units:   # e.g. 'smoke_04210_camera_16...' = unit-less frame name
                um = None
            unit = um.group(1) if um else 'noprefix'
            det[(split, unit)][kind] += 1; det[(split, unit)]['bytes'] += b
            if kind == 'images' and um:
                cl = 'nonsmoke' if 'non' in unit else 'smoke'
                exact_map.setdefault(stem, set()).add((unit, split, cl)); frame_map.setdefault(um.group(2), set()).add((unit, split, cl))
            elif kind == 'images':
                det_noprefix.setdefault(stem, []).append(split)
            continue
        other.append((n, b))
    def agg(d, keyf):
        out = collections.defaultdict(lambda: {'units': 0, 'frames': 0, 'bytes': 0})
        for k, v in d.items():
            o = out[keyf(k)]; o['units'] += 1; o['frames'] += v.get('frames', v.get('images', 0)); o['bytes'] += v['bytes']
        return dict(out)
    summ = {'n_entries': len(rows), 'total_bytes': sum((r.get('totalBytes') or 0) for r in rows),
            'classification_by_split_class': agg(cls, lambda k: f'{k[0]}/{k[1]}'),
            'detection_by_split_class': agg(det, lambda k: f'{k[0]}/{"nonsmoke" if "non" in k[1] else "smoke"}'),
            'detection_label_files': {f'{s}': sum(v['labels'] for (ss, u), v in det.items() if ss == s) for s in ('train', 'test')},
            'classification_units': {f'{s}/{c}/{u}': v['frames'] for (s, c, u), v in sorted(cls.items())},
            'detection_units': {f'{s}/{u}': v for (s, u), v in sorted(det.items())},
            'other_files': other[:20], 'detection_noprefix_images': len(det_noprefix), 'known_units': len(known_units)}
    # resolve to single (unit, split) when unambiguous on unit
    def squash(mp):
        out = {}
        for k, v in mp.items():
            units = sorted(set(u for u, _, _ in v))
            if len(units) == 1:
                splits = sorted(set(s for _, s, _ in v)); out[k] = (units[0], '+'.join(splits))
            else:
                out[k] = ('ambiguous:' + '|'.join(units), '')
        return out
    return summ, squash(frame_map), squash(exact_map)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--kaggle-list', default='/tmp/kaggle_files.jsonl'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(f'{YOLO_DIR}/images', exist_ok=True); os.makedirs(f'{YOLO_DIR}/labels', exist_ok=True)
    cache = pd.read_parquet(f'{ROOT}/data/v4/manifests/_cache/hash_cache.parquet')
    fill = pd.read_parquet(f'{ROOT}/data/v5b/manifests/_cache/phash_v5b.parquet')   # scripts/v5b/phash_fill.py
    fmap = dict(zip(fill.path, fill.phash))
    cache['phash'] = [ph if ph else fmap.get(p, '') for p, ph in zip(cache.path, cache.phash)]
    # data/v4/raw rows carry phash but no sha256 in the v4 cache -> take sha256 from the v4 source manifests
    smap = {}
    for src, raw in [('multinatsmoke', MNS_RAW), ('boreal', f'{ROOT}/data/v4/raw/boreal'), ('firesense', f'{ROOT}/data/v4/raw/firesense')]:
        d = pd.read_parquet(f'{ROOT}/data/v4/manifests/{src}_manifest.parquet')
        smap.update({f'{raw}/{rp}': sh for rp, sh in zip(d.rel_path, d.sha256)})
    cache['sha256'] = [sh if sh else smap.get(p, '') for p, sh in zip(cache.path, cache.sha256)]
    print('cache sha still empty', int((cache.sha256 == '').sum()), 'phash empty', int((cache.phash == '').sum()))
    cache = cache[(cache.phash != '')].reset_index(drop=True)
    roles = load_roles()
    cache['role'] = [role_of(p, roles) for p in cache.path]
    print('cache rows', len(cache), cache.role.str.split(':').str[0].value_counts().to_dict())
    sha2paths = collections.defaultdict(list)
    for p, s in zip(cache.path, cache.sha256): sha2paths[s].append(p)
    cph = hex2u64(cache.phash.values); cpaths = cache.path.values; croles = cache.role.values

    # ---------- (A) MultiNatSmoke copies ----------
    mm = pd.read_parquet(f'{ROOT}/data/v4/manifests/multinatsmoke_manifest.parquet')
    w = mm[mm.subset == 'WSDataset'].copy().reset_index(drop=True)
    w['abs'] = MNS_RAW + '/' + w.rel_path
    c_by_path = dict(zip(cache.path, cache.phash))
    w['phash'] = [c_by_path.get(p, '') for p in w['abs']]
    assert (w.phash != '').all(), 'missing phash in cache'
    # verify cache freshness (size+mtime) on a sample and sha on 30 random files
    rs = random.Random(a.seed)
    chk = rs.sample(range(len(w)), 30)
    bad = sum(sha_file(w['abs'][i]) != w.sha256[i] for i in chk); print('sha spot-check mismatches (30):', bad)
    # unit recovery from Kaggle listing
    kg_summary, suffix_map, exact_map = parse_kaggle(a.kaggle_list)
    orig_unit, orig_split, method = [], [], []
    for rp in w.rel_path:
        stem = os.path.splitext(os.path.basename(rp))[0]
        if stem in exact_map and not exact_map[stem][0].startswith('ambiguous'):
            u, s = exact_map[stem]; orig_unit.append(u); orig_split.append(s); method.append('kaggle_listing_unit_frame_name')
        elif stem in suffix_map and not suffix_map[stem][0].startswith('ambiguous'):
            u, s = suffix_map[stem]; orig_unit.append(u); orig_split.append(s); method.append('kaggle_listing_frame_name')
        elif stem in suffix_map or stem in exact_map:
            u, s = (exact_map.get(stem) or suffix_map.get(stem)); orig_unit.append(u); orig_split.append(''); method.append('kaggle_listing_ambiguous')
        else:
            orig_unit.append(''); orig_split.append(''); method.append('unmatched')
    w['orig_unit'] = orig_unit; w['orig_split_kaggle'] = orig_split; w['unit_recovery_method'] = method
    w['parent_unit_final'] = [ ('wsd_orig_' + u) if (u and not u.startswith('ambiguous')) else ('mns_' + pu) for u, pu in zip(w.orig_unit, w.parent_unit)]
    # dedup
    wph = hex2u64(w.phash.values)
    dup_detail = [[] for _ in range(len(w))]
    wsd_self_near = 0; self_nb = np.zeros(len(w), dtype=bool)
    for i0 in range(0, len(w), 256):
        blk = wph[i0:i0 + 256]
        d = popcount64(blk[:, None] ^ cph[None, :])
        for bi, gi in zip(*np.nonzero(d <= PHASH_MAX)):
            i = i0 + bi; p = cpaths[gi]
            if p == w['abs'][i]: continue
            r = croles[gi]
            if r == 'self:wsdataset': wsd_self_near += 1; self_nb[i] = True; continue
            kind = 'exact' if cache.sha256.values[gi] == w.sha256[i] else f'near{int(d[bi, gi])}'
            dup_detail[i].append(f'{kind}:{r}:{os.path.relpath(p, ROOT)}')
    for i, s in enumerate(w.sha256):   # exact sha hits not caught (phash differs) — belt and braces
        for p in sha2paths.get(s, []):
            if p != w['abs'][i] and roles.get(p, role_of(p, roles)) != 'self:wsdataset' and not any(p.endswith(x.split(':', 2)[-1]) for x in dup_detail[i]):
                dup_detail[i].append(f'exact:{role_of(p, roles)}:{os.path.relpath(p, ROOT)}')
    def classify(dd):
        tags = ['dup_multinat']
        for x in dd:
            kind, role = x.split(':')[0], x.split(':')[1]
            if role == 'eval': tags.append(f'{"exact" if kind=="exact" else "near"}dup_eval:{x.split(":")[2]}')
            elif role == 'quarantine': tags.append(f'{"exact" if kind=="exact" else "near"}dup_figlib_family:{x.split(":")[2]}')
            elif role == 'train': tags.append(f'{"exact" if kind=="exact" else "near"}dup_train:{x.split(":")[2]}')
        return ';'.join(sorted(set(tags), key=tags.index))
    w['dup_class'] = [classify(dd) for dd in dup_detail]
    w['dup_detail'] = [' | '.join(dd[:5]) + (f' | ... (+{len(dd)-5} more)' if len(dd) > 5 else '') for dd in dup_detail]
    def split_of(dc):
        if 'figlib' in dc or 'ruokolahti' in dc or 'pyro_sdis_test' in dc or 'ausmoke_test' in dc: return 'quarantine'
        if 'heinola' in dc or 'pyro_sdis_val' in dc or 'ausmoke_val' in dc: return 'val'
        return 'train'   # inherits v4 role: all 11,539 were in G (B_G_v1, C_EGT_v1)
    w['proposed_split'] = [split_of(dc) for dc in w.dup_class]
    # unit-level 70/15/15 suggestion over recovered original units (for the day the Kaggle archive arrives)
    units = sorted(set(u for u in w.parent_unit_final)); rr = random.Random(a.seed); rr.shuffle(units)
    n = len(units); sug = {}
    for k, u in enumerate(units): sug[u] = 'train' if k < round(0.7 * n) else ('val' if k < round(0.85 * n) else 'holdout_if_author_boxes_obtained')
    w['unit_split_suggestion_70_15_15'] = [sug[u] for u in w.parent_unit_final]
    # YOLO: copy v4 derived labels, symlink images
    n_boxes, lbl_rel = [], []
    used = set()
    for rp in w.rel_path:
        v4name = 'multinatsmoke__' + rp.replace('/', '__').rsplit('.', 1)[0] + '.txt'
        src = f'{V4_LBL}/{v4name}'; assert os.path.exists(src), src
        lines = [l for l in open(src).read().splitlines() if l.strip()]
        base = os.path.splitext(os.path.basename(rp))[0]
        name = base if base not in used else rp.split('/')[1] + '__' + base
        used.add(name)
        with open(f'{YOLO_DIR}/labels/{name}.txt', 'w') as f: f.write('\n'.join(lines) + ('\n' if lines else ''))
        dst = f'{YOLO_DIR}/images/{name}.jpg'
        if not os.path.lexists(dst): os.symlink(f'{MNS_RAW}/{rp}', dst)
        n_boxes.append(len(lines)); lbl_rel.append(os.path.relpath(f'{YOLO_DIR}/labels/{name}.txt', ROOT))
    A = pd.DataFrame({
        'source': 'wsdataset', 'material': 'multinatsmoke_copy', 'rel_path': 'data/v4/raw/multinatsmoke/' + w.rel_path, 'sha256': w.sha256, 'phash': w.phash,
        'parent_unit': w.parent_unit_final, 'label_type': 'mask', 'has_smoke_label': True, 'has_fire_label': False, 'is_explicit_negative': False,
        'n_smoke_boxes': n_boxes, 'annotation_origin': 'model_assisted', 'licence': LIC_MNS, 'permission_status': 'pending',
        'dup_class': w.dup_class, 'proposed_split': w.proposed_split, 'new_data': False, 'v4_role': 'train (G: B_G_v1 x1, C_EGT_v1 x2)',
        'mns_parent_unit': w.parent_unit, 'mns_split_official': w.split_official, 'orig_unit_kaggle': w.orig_unit, 'orig_split_kaggle': w.orig_split_kaggle,
        'unit_recovery_method': w.unit_recovery_method, 'unit_split_suggestion_70_15_15': w.unit_split_suggestion_70_15_15,
        'mask_rel_path': 'data/v4/raw/multinatsmoke/' + w.mask_rel_path, 'mask_n_components': w.n_smoke_boxes, 'mask_fg_frac': w.mask_fg_frac,
        'derived_box_label': lbl_rel, 'box_conversion': BOX_CONV, 'samescene_units': '', 'min_phash_hamming_vs_held': -1, 'width': w.width, 'height': w.height, 'dup_detail': w.dup_detail, 'also_at': ''})
    a_unit_of = dict(zip(w['abs'], w.parent_unit_final))
    print('A rows', len(A), 'wsd self near-dup pairs (adjacent frames, not a dup_class):', wsd_self_near)

    # ---------- (B) GitHub demo ----------
    B = []
    demo = f'{GH_RAW}/dataset/demo'
    for sub in ('classification/train/nonsmoke', 'classification/train/smoke', 'detection/train/images'):
        for dp, _, fs in os.walk(f'{demo}/{sub}'):
            for fn in sorted(fs):
                if not fn.endswith('.jpg'): continue
                p = f'{dp}/{fn}'; rel = os.path.relpath(p, ROOT); twin = rel.replace('/train/', '/test/')
                sha = sha_file(p); im = Image.open(p); ph = str(imagehash.phash(im)); W, H = im.size
                twin_same = os.path.exists(f'{ROOT}/{twin}') and sha_file(f'{ROOT}/{twin}') == sha
                if 'classification' in sub:
                    unit = 'wsd_gh_' + os.path.basename(dp); is_neg = 'nonsmoke' in sub; lt = 'image'; nb = 0; lbl = ''
                else:
                    stem = os.path.splitext(fn)[0]; um = re.match(r'(smoke_\d+(?:_\d+)?|non_smoke_\d+)_', stem); unit = 'wsd_gh_' + (um.group(1) if um else 'det')
                    lp = f'{demo}/detection/train/labels/{stem}.txt'; lines = [l for l in open(lp).read().splitlines() if l.strip()]
                    nb = len(lines); is_neg = nb == 0; lt = 'box'
                    name = 'gh__' + stem
                    with open(f'{YOLO_DIR}/labels/{name}.txt', 'w') as f: f.write('\n'.join(lines) + ('\n' if lines else ''))
                    dst = f'{YOLO_DIR}/images/{name}.jpg'
                    if not os.path.lexists(dst): os.symlink(p, dst)
                    lbl = os.path.relpath(f'{YOLO_DIR}/labels/{name}.txt', ROOT)
                # dedup vs cache (incl. WSD MNS copies) by sha and phash
                dd = []
                for q in sha2paths.get(sha, []): dd.append(f'exact:{role_of(q, roles)}:{os.path.relpath(q, ROOT)}')
                d = popcount64(hex2u64([ph])[:, None] ^ cph[None, :])[0]
                for gi in np.nonzero(d <= PHASH_MAX)[0]:
                    if cache.sha256.values[gi] != sha: dd.append(f'near{int(d[gi])}:{croles[gi]}:{os.path.relpath(cpaths[gi], ROOT)}')
                tags = []
                for x in dd:
                    kind, role, rest = x.split(':', 2)
                    tags.append(('exact' if kind == 'exact' else 'near') + ('dup_eval:' if role == 'eval' else 'dup_multinat:' if role == 'self' else 'dup_figlib_family:' if role == 'quarantine' else 'dup_train:') + rest)
                tags = sorted(set(t.split(':')[0] + ':' + t.split(':')[1].split('/')[0] for t in tags)) if tags else []
                tags = [t.replace('neardup_multinat', 'samescene_neardup_multinat') for t in tags]   # phash<=6 on a fixed camera = same scene, not same frame
                dc = ';'.join(tags) + f' (n={len(dd)})' if tags else 'none'
                scene_units = sorted(set(a_unit_of.get(f"{ROOT}/{x.split(':self:wsdataset:', 1)[1]}", '?') for x in dd if ':self:wsdataset:' in x))
                min_ham = min([0 if x.startswith('exact') else int(re.match(r'near(\d+)', x).group(1)) for x in dd] or [99])
                if any(x.startswith('near0:self:wsdataset:') or x.startswith('exact:self:wsdataset:') for x in dd):
                    dc = 'dup_multinat_resized;' + dc      # phash identical to an MNS WSDataset frame -> same frame at another resolution
                is_new = not dc.startswith('dup_multinat_resized')
                if 'figlib' in dc or 'ruokolahti' in dc or 'pyro_sdis_test' in dc: ps = 'quarantine'
                elif any(u.startswith('wsd_orig_') or u.startswith('mns_') for u in scene_units): ps = 'train'   # same camera/scene as v4-train WSDataset positives -> inherits train
                else: ps = 'val'
                B.append({'source': 'wsdataset', 'material': 'github_demo_author', 'rel_path': rel, 'sha256': sha, 'phash': ph, 'parent_unit': unit, 'label_type': lt,
                          'has_smoke_label': (not is_neg), 'has_fire_label': False, 'is_explicit_negative': is_neg, 'n_smoke_boxes': nb, 'annotation_origin': 'author',
                          'licence': LIC_GH, 'permission_status': 'pending', 'dup_class': dc, 'proposed_split': ps, 'new_data': is_new, 'v4_role': 'none (never held before 2026-10-02)' if is_new else 'train via MNS copy of the same frames (G)',
                          'mns_parent_unit': '', 'mns_split_official': '', 'orig_unit_kaggle': unit.replace('wsd_gh_', ''), 'orig_split_kaggle': 'train(+byte-identical test copy)',
                          'unit_recovery_method': 'github_folder_name', 'unit_split_suggestion_70_15_15': 'val (3 tiny units; author test split is a byte copy of train -> no author holdout)',
                          'mask_rel_path': '', 'mask_n_components': 0, 'mask_fg_frac': 0.0, 'derived_box_label': lbl, 'box_conversion': 'none (author YOLO txt copied verbatim)' if lt == 'box' else 'n/a (image-level)',
                          'samescene_units': ';'.join(scene_units), 'min_phash_hamming_vs_held': min_ham, 'width': W, 'height': H, 'dup_detail': ' | '.join(dd[:5]) + (f' | ... (+{len(dd)-5} more)' if len(dd) > 5 else ''), 'also_at': twin if twin_same else ''})
    B = pd.DataFrame(B); print('B rows', len(B), B.groupby(['parent_unit', 'label_type', 'is_explicit_negative']).size().to_dict())
    M = pd.concat([A, B], ignore_index=True)
    for c in ['has_smoke_label', 'has_fire_label', 'is_explicit_negative', 'new_data']: M[c] = M[c].astype(bool)
    M.to_parquet(f'{OUT_DIR}/wsdataset_manifest.parquet', index=False)
    U = M.groupby('parent_unit').agg(material=('material', 'first'), n_images=('rel_path', 'size'), label_type=('label_type', 'first'),
                                     n_boxes=('n_smoke_boxes', 'sum'), explicit_neg=('is_explicit_negative', 'sum'),
                                     mns_camera_group=('mns_parent_unit', lambda x: '|'.join(sorted(set(v for v in x if v)))),
                                     mns_split=('mns_split_official', lambda x: '|'.join(sorted(set(v for v in x if v)))),
                                     kaggle_split=('orig_split_kaggle', lambda x: '|'.join(sorted(set(v for v in x if v)))),
                                     proposed_split=('proposed_split', 'first'), suggestion_70_15_15=('unit_split_suggestion_70_15_15', 'first'),
                                     samescene_units=('samescene_units', 'first')).reset_index().sort_values('n_images', ascending=False)
    U.to_csv(f'{OUT_DIR}/wsdataset_units.csv', index=False)
    summ = {'rows': len(M), 'by_material': M.material.value_counts().to_dict(), 'by_label_type': M.label_type.value_counts().to_dict(),
            'by_origin': M.annotation_origin.value_counts().to_dict(), 'explicit_negatives': int(M.is_explicit_negative.sum()),
            'new_data_images': int(M.new_data.sum()), 'new_data_bytes': int(sum(os.path.getsize(f'{ROOT}/{r}') for r in M.rel_path[M.new_data])),
            'units_A': int(A.parent_unit.nunique()), 'units_A_recovered_from_kaggle': int(A.parent_unit.str.startswith('wsd_orig_').sum()),
            'unit_recovery_method': A.unit_recovery_method.value_counts().to_dict(), 'units_B': int(B.parent_unit.nunique()),
            'A_dup_class': A.dup_class.str.split(';').str[1:].str.join(';').replace('', 'dup_multinat only').value_counts().to_dict(),
            'A_proposed_split': A.proposed_split.value_counts().to_dict(), 'B_dup_class': B.dup_class.value_counts().head(10).to_dict(),
            'A_eval_or_figlib_hits': int(A.dup_class.str.contains('eval|figlib').sum()), 'wsd_self_neardup_pairs': int(wsd_self_near), 'wsd_frames_with_self_neardup': int(self_nb.sum()),
            'A_boxes_total': int(A.n_smoke_boxes.sum()), 'A_boxes_per_image': A.n_smoke_boxes.describe()[['min', '50%', 'mean', 'max']].to_dict(),
            'A_units_by_final_unit': A.parent_unit.value_counts().to_dict(), 'A_orig_split_kaggle': A.orig_split_kaggle.value_counts().to_dict(),
            'kaggle_original_structure': kg_summary}
    json.dump(summ, open(f'{OUT_DIR}/wsdataset_build_summary.json', 'w'), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k not in ('A_units_by_final_unit', 'kaggle_original_structure')}, indent=0, default=str)[:3000])

if __name__ == '__main__':
    main()
