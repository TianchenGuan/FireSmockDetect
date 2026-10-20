"""Parse FireMan-UAV-RGBT CVAT-for-images XML into per-frame records.
Record: name, width, height, boxes{label:[(xtl,ytl,xbr,ybr,source)]}, tags{label:{attr:val}}, tag_sources, n_tags{label:count}"""
import xml.etree.ElementTree as ET
from collections import defaultdict
def parse_cvat(path):
    root=ET.parse(path).getroot(); meta=root.find('meta'); task=meta.find('task') if meta is not None else None
    info={'task_name': task.findtext('name') if task is not None else None,
          'task_size': int(task.findtext('size')) if task is not None and task.findtext('size') else None,
          'subset': task.findtext('subset') if task is not None else None,
          'created': task.findtext('created') if task is not None else None,
          'labels': [l.findtext('name') for l in meta.iter('label')] if meta is not None else []}
    frames=[]
    for img in root.iter('image'):
        rec={'id':int(img.get('id')),'name':img.get('name'),'width':int(img.get('width')),'height':int(img.get('height')),
             'boxes':defaultdict(list),'tags':{},'n_tags':defaultdict(int),'tag_conflict':[]}
        for b in img.findall('box'):
            rec['boxes'][b.get('label')].append((float(b.get('xtl')),float(b.get('ytl')),float(b.get('xbr')),float(b.get('ybr')),b.get('source','?')))
        for t in img.findall('tag'):
            lab=t.get('label'); attrs={a.get('name'):(a.text or '').strip() for a in t.findall('attribute')}; attrs['_source']=t.get('source','?')
            rec['n_tags'][lab]+=1
            if lab in rec['tags'] and {k:v for k,v in rec['tags'][lab].items() if k!='_source'}!={k:v for k,v in attrs.items() if k!='_source'}:
                rec['tag_conflict'].append(lab)
            rec['tags'].setdefault(lab, attrs)
        for p in img.findall('polygon'): rec['boxes']['POLY_'+p.get('label')].append(p.get('points'))
        frames.append(rec)
    return info, frames
if __name__=='__main__':
    import sys, collections
    for path in sys.argv[1:]:
        info, frames=parse_cvat(path); print('==',path.split('/')[-1], info['task_name'], 'size',info['task_size'], 'frames',len(frames), 'labels',info['labels'])
        c=collections.Counter(); boxsrc=collections.Counter(); tagvals=collections.Counter(); conflicts=collections.Counter()
        for f in frames:
            for lab,bs in f['boxes'].items():
                c['box:'+lab]+=len(bs); c['img_with_box:'+lab]+=1
                for b in bs: boxsrc[(lab,b[4] if isinstance(b,tuple) else 'poly')]+=1
            for lab,at in f['tags'].items():
                for k,v in at.items():
                    if k!='_source': tagvals[(lab,k,v)]+=1
            for lab in f['tag_conflict']: conflicts[lab]+=1
            smoke_tag=f['tags'].get('smoke') or f['tags'].get('smoke_tag') or {}
            sp=smoke_tag.get('smoke_presence','ABSENT'); nb=len(f['boxes'].get('smoke_region',[]))
            c[f'smoke_presence={sp} & n_smoke_boxes{">0" if nb>0 else "=0"}']+=1
            ft=f['tags'].get('fire') or f['tags'].get('fire_tag') or {}; c[f"fire_presence={ft.get('fire_presence','ABSENT')}"]+=1
        for k,v in sorted(c.items()): print('  ',v,k)
        print('   box sources:',dict(boxsrc)); print('   tag conflicts:',dict(conflicts))
        print('   tag attr values:',{f'{a}.{b}={c_}':n for (a,b,c_),n in sorted(tagvals.items())})
