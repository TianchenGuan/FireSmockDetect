"""Scene-embedding evidence for burn identification: DINOv2 CLS embeddings of sampled frames per FireMan RGB video and per Boreal site.
Outputs data/v5b/manifests/_inspect/fireman_scene_embed.json (per-video nearest videos / nearest Boreal sites) and a cluster listing."""
import os, json, numpy as np, torch
from PIL import Image
ROOT='/local/home/tianrgua/cvproject'; RAW=f'{ROOT}/data/v5b/raw/fireman/Multiclass/Unimodal/rgb'; BOR=f'{ROOT}/data/v4/raw/boreal/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-A'
OUT=f'{ROOT}/data/v5b/manifests/_inspect'
torch.set_num_threads(96)
model=torch.hub.load('facebookresearch/dinov2','dinov2_vitb14'); model.eval()
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
def embed(paths, size=308):
    out=[]
    for i in range(0,len(paths),16):
        ims=[]
        for p in paths[i:i+16]:
            im=Image.open(p).convert('RGB').resize((size,size)); ims.append(torch.from_numpy(np.asarray(im)).permute(2,0,1).float()/255)
        x=(torch.stack(ims)-mean)/std
        with torch.no_grad(): f=model(x)
        out.append(torch.nn.functional.normalize(f,dim=1))
    return torch.cat(out).numpy()
groups={}
for split in ('train','val','test'):
    for v in sorted(os.listdir(f'{RAW}/{split}')):
        d=f'{RAW}/{split}/{v}/images'; fs=sorted(os.listdir(d)); idx=np.linspace(0,len(fs)-1,24).astype(int)
        groups[f'fireman:{split}/{v}']=[f'{d}/{fs[i]}' for i in idx]
for site in ('Evo','Heinola','Karkkila','Ruokolahti'):
    d=f'{BOR}/{site}-Images'; fs=sorted(os.listdir(d)); idx=np.linspace(0,len(fs)-1,60).astype(int)
    groups[f'boreal:{site}']=[f'{d}/{fs[i]}' for i in idx]
names=list(groups); allp=[p for n in names for p in groups[n]]; lab=np.array([n for n in names for _ in groups[n]])
E=embed(allp); print('embedded',E.shape, flush=True)
np.save(f'{OUT}/fireman_scene_embed.npy',E); json.dump({'paths':allp,'labels':lab.tolist()},open(f'{OUT}/fireman_scene_embed_index.json','w'))
S=E@E.T
res={}
for n in names:
    m=lab==n; sims={}
    for n2 in names:
        if n2==n: continue
        m2=lab==n2; block=S[np.ix_(m,m2)]
        sims[n2]=(float(block.max(1).mean()), float(block.mean()))  # mean over frames of best-match similarity; mean sim
    top=sorted(sims.items(), key=lambda kv:-kv[1][0])[:6]
    res[n]={'top_matches':[(k,round(v[0],3),round(v[1],3)) for k,v in top]}
json.dump(res,open(f'{OUT}/fireman_scene_embed.json','w'),indent=1)
for n in names:
    print(n.ljust(28), ' | '.join(f'{k.replace("fireman:","")}:{v1:.2f}' for k,v1,v2 in res[n]['top_matches']))
# simple agglomerative clustering of fireman videos on mean-of-best-match similarity
fn=[n for n in names if n.startswith('fireman')]
M=np.zeros((len(fn),len(fn)))
for i,a in enumerate(fn):
    for j,b in enumerate(fn):
        if i!=j:
            block=S[np.ix_(lab==a,lab==b)]; M[i,j]=block.max(1).mean()
M=(M+M.T)/2
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
D=1-M; np.fill_diagonal(D,0); Z=linkage(squareform(D,checks=False),'average')
for t in (0.25,0.3,0.35):
    cl=fcluster(Z,t,'distance'); print(f'--- clusters at dist<{t}:')
    for c in sorted(set(cl)): print('  ',c, [fn[i].replace('fireman:','') for i in range(len(fn)) if cl[i]==c])
