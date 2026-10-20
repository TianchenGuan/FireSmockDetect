"""Build reference sha256/phash table of all existing project material with inherited roles.
Output: data/v5b/manifests/_cache/ref_hashes.parquet [path, sha256, phash, ref_source, ref_role, ref_unit]
Roles: train | val | quarantine (held-out/FIgLib/final-test) | dup_multinat (pending-permission source)."""
import os, json, re, sys
import pandas as pd
from multiprocessing import Pool
from PIL import Image
import imagehash
ROOT='/local/home/tianrgua/cvproject'
cache=pd.read_parquet(f'{ROOT}/data/v4/manifests/_cache/hash_cache.parquet')
cache['phash']=cache['phash'].fillna('').astype(str)
def ph(p):
    try:
        with Image.open(p) as im: return (p, str(imagehash.phash(im.convert('RGB'))))
    except Exception as e: return (p, '')
missing=cache.loc[cache.phash=='','path'].tolist()
print('missing phash:',len(missing), flush=True)
if missing:
    with Pool(96) as pool:
        res=dict(pool.map(ph, missing, chunksize=64))
    cache['phash']=[res.get(p,h) if h=='' else h for p,h in zip(cache.path,cache.phash)]
print('phash empty after compute:',(cache.phash=='').sum(), flush=True)
# roles
sdis=pd.read_parquet(f'{ROOT}/data/manifests/pyro_sdis_manifest.parquet')[['image_name','site']]
site2role={}
sp=json.load(open(f'{ROOT}/data/manifests/splits_frozen.json'))['site_split']
for r,sites in sp.items():
    for s in sites: site2role[s]={'train':'train','val':'val','test':'quarantine'}[r]
name2site=dict(zip(sdis.image_name, sdis.site))
evt=json.load(open(f'{ROOT}/data/v4/eval/EVALTEST_MANIFEST.json'))
evv=json.load(open(f'{ROOT}/data/v4/eval/EVALVAL_MANIFEST.json'))
def _collect(d):
    out=set()
    if isinstance(d,dict):
        for v in d.values(): out|=_collect(v)
    elif isinstance(d,list):
        for v in d: out|=_collect(v)
    elif isinstance(d,str) and ('/' in d): out.add(os.path.basename(d))
    return out
forest_test_names=_collect(evt.get('forest_test',{})); forest_val_names=_collect(evv.get('forest_val',{}))
print('forest_test names',len(forest_test_names),'forest_val names',len(forest_val_names))
src=[];role=[];unit=[]
for p in cache.path:
    b=os.path.basename(p); low=p.lower()
    if '/figlib' in low: src.append('figlib'); role.append('quarantine'); unit.append(p.split('/')[-2])
    elif '/v4/raw/boreal/' in low:
        m=re.search(r'(evo|heinola|karkkila|ruokolahti)', b.lower()) or re.search(r'(evo|heinola|karkkila|ruokolahti)', low)
        site=m.group(1) if m else 'unknown'
        src.append('boreal'); unit.append('burn_'+site)
        role.append({'heinola':'val','ruokolahti':'quarantine','evo':'train','karkkila':'train'}.get(site,'quarantine'))
    elif '/pyro_sdis/' in low:
        s=name2site.get(b, name2site.get(os.path.splitext(b)[0],'unknown')); src.append('pyro_sdis'); unit.append(s); role.append(site2role.get(s,'quarantine'))
    elif '/dfire/' in low: src.append('dfire'); role.append('train'); unit.append('dfire')
    elif '/multinatsmoke/' in low:
        src.append('multinatsmoke'); unit.append(p.split('/multinatsmoke/')[1].split('/')[0])
        role.append('quarantine' if b in forest_test_names else ('val' if b in forest_val_names else 'dup_multinat'))
    elif '/firesense/' in low: src.append('firesense'); role.append('train'); unit.append('firesense')
    else: src.append('other'); role.append('quarantine'); unit.append('other')
cache['ref_source']=src; cache['ref_role']=role; cache['ref_unit']=unit
out=cache[['path','sha256','phash','ref_source','ref_role','ref_unit']]
os.makedirs(f'{ROOT}/data/v5b/manifests/_cache',exist_ok=True)
out.to_parquet(f'{ROOT}/data/v5b/manifests/_cache/ref_hashes.parquet',index=False)
print(out.groupby(['ref_source','ref_role']).size().to_string())
