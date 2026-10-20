"""Export FireMan RGB labels to YOLO txt (class 0 = smoke) under data/v5b/yolo/fireman/{images,labels}/<split>/ with images symlinked.
- box rows -> txt with boxes; explicit negatives (smoke_presence=no, no boxes) -> empty txt; image-level-only positives -> NO txt (listed in image_level_only.txt).
- quarantine rows and thermal rows are NOT exported. Rows with label_conflict are NOT exported (listed in conflicts.txt)."""
import os, json, pandas as pd
ROOT = '/local/home/tianrgua/cvproject'; RAW = f'{ROOT}/data/v5b/raw/fireman'; Y = f'{ROOT}/data/v5b/yolo/fireman'
df = pd.read_parquet(f'{ROOT}/data/v5b/manifests/fireman_manifest.parquet')
df = df[(df.modality == 'rgb') & (df.proposed_split.isin(['train', 'val', 'holdout'])) & (df.decode_err == '')]
stats = {}; img_level = []; conflicts = []
for split, g in df.groupby('proposed_split'):
    os.makedirs(f'{Y}/images/{split}', exist_ok=True); os.makedirs(f'{Y}/labels/{split}', exist_ok=True)
    c = {'box_files': 0, 'empty_neg_files': 0, 'image_level_only_no_file': 0, 'unlabeled_no_file': 0, 'conflict_skipped': 0, 'boxes': 0}
    for r in g.itertuples():
        stem = r.video_id.replace('/', '_') + '_' + os.path.splitext(os.path.basename(r.rel_path))[0]
        if r.label_conflict and r.label_conflict != '': conflicts.append(f'{split}\t{r.rel_path}\t{r.label_conflict}'); c['conflict_skipped'] += 1; continue
        if r.label_type == 'none': c['unlabeled_no_file'] += 1; continue
        if r.label_type == 'image' and not r.is_explicit_negative: img_level.append(f'{split}\t{r.rel_path}'); c['image_level_only_no_file'] += 1; continue
        src = f'{RAW}/{r.rel_path}'; dst = f'{Y}/images/{split}/{stem}.jpg'
        if not os.path.lexists(dst): os.symlink(src, dst)
        lines = []
        if r.label_type == 'box':
            W, H = (r.width or r.xml_width), (r.height or r.xml_height)
            for x1, y1, x2, y2 in json.loads(r.smoke_boxes):
                x1, x2 = max(0, min(W, x1)), max(0, min(W, x2)); y1, y2 = max(0, min(H, y1)), max(0, min(H, y2))
                if x2 - x1 < 1 or y2 - y1 < 1: continue
                lines.append(f'0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}')
            c['box_files'] += 1; c['boxes'] += len(lines)
        else: c['empty_neg_files'] += 1
        open(f'{Y}/labels/{split}/{stem}.txt', 'w').write('\n'.join(lines) + ('\n' if lines else ''))
    stats[split] = c
open(f'{Y}/image_level_only.txt', 'w').write('\n'.join(img_level) + '\n'); open(f'{Y}/conflicts.txt', 'w').write('\n'.join(conflicts) + '\n')
open(f'{Y}/README.md', 'w').write('# FireMan-UAV-RGBT YOLO export (class 0 = smoke)\nSource: data/v5b/manifests/fireman_manifest.parquet. Boxes = author smoke_region (CVAT, source semi-auto/manual). Empty txt = author image-level smoke_presence=no. Image-level positives without boxes have no txt (image_level_only.txt). Quarantine/thermal/conflict rows not exported.\n' + json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
