"""Print markdown summary tables from fireman_manifest.parquet for the CARD."""
import pandas as pd, json, sys
ROOT='/local/home/tianrgua/cvproject'
df=pd.read_parquet(f'{ROOT}/data/v5b/manifests/fireman_manifest.parquet')
def md(t):
    if isinstance(t,pd.Series): t=t.to_frame()
    t=t.reset_index(); cols=[str(c) for c in t.columns]
    print('| '+' | '.join(cols)+' |'); print('|'+'---|'*len(cols))
    for row in t.itertuples(index=False): print('| '+' | '.join(str(v) for v in row)+' |')
    print()
rgb=df[df.modality=='rgb']; th=df[df.modality=='thermal']
print('## rows by modality/split'); md(df.groupby(['modality','official_split']).size().unstack(fill_value=0))
print('## RGB smoke label type x origin'); md(rgb.groupby(['label_type','annotation_origin']).size().unstack(fill_value=0))
print('## RGB smoke_presence x has boxes'); md(pd.crosstab(rgb.smoke_presence, rgb.n_smoke_boxes>0))
print('## RGB fire_presence x smoke_presence'); md(pd.crosstab(rgb.fire_presence, rgb.smoke_presence))
print('## RGB author_box_source'); md(rgb.author_box_source.value_counts().to_frame())
print('## RGB smoke_intensity'); md(rgb.smoke_intensity.value_counts(dropna=False).to_frame())
print('## RGB camera_perspective / smoke_interaction'); md(pd.crosstab(rgb.camera_perspective.fillna('NA'), rgb.smoke_interaction.fillna('NA')))
print('## RGB resolutions'); md(rgb.groupby(['width','height']).size().to_frame('n'))
print('## per video (RGB)')
g=rgb.groupby(['video_id','parent_unit','burn_date','proposed_split'],dropna=False).agg(n=('rel_path','size'),box_pos=('n_smoke_boxes',lambda s:(s>0).sum()),expl_neg=('is_explicit_negative','sum'),img_only_pos=('label_type',lambda s:0),fire=('has_fire_label','sum'),conflicts=('label_conflict',lambda s:(s!='').sum()),boxes=('n_smoke_boxes','sum'),w=('width','first'),h=('height','first'))
g['img_only_pos']=rgb[(rgb.label_type=='image')&(~rgb.is_explicit_negative)].groupby('video_id').size().reindex(g.index.get_level_values(0)).fillna(0).astype(int).values
md(g)
print('## per unit (RGB)'); md(rgb.groupby(['parent_unit','proposed_split']).agg(n_frames=('rel_path','size'),n_videos=('video_id','nunique'),box_pos=('n_smoke_boxes',lambda s:(s>0).sum()),expl_neg=('is_explicit_negative','sum'),boxes=('n_smoke_boxes','sum')))
print('## dup_class'); md(df.groupby(['modality','dup_class']).size().unstack(fill_value=0))
print('## label conflicts'); md(rgb.label_conflict.value_counts().to_frame())
print('## thermal'); md(th.groupby(['video_id','parent_unit']).agg(n=('rel_path','size'),fire_boxes=('n_fire_boxes','sum'),smoke_presence=('smoke_presence',lambda s:','.join(sorted(set(s)))),w=('width','first'),h=('height','first')))
