#!/usr/bin/env python
"""V5b training-corpus builder: arms B (source-balanced), C (missing-label-aware ignore
masks), D (teacher-assisted pseudo-boxes).  Conventions follow scripts/v4/build_corpus.py
(list-file corpora, per-component multiplicities, seed-0 shuffle, dereferenced stage tar
containing only NEW files, frozen CORPUS_<corpus>.json manifest).

Nothing here is human review.  Teacher outputs are MODEL-ASSISTED labels (never author
labels); masked copies are an ignore-region approximation for a detector without ignore
support (labels unchanged, pixels in the uncertain region replaced by the image mean).

Usage:  python scripts/v5b/build_corpus_v5b.py analyze           # teacher analysis cache
        python scripts/v5b/build_corpus_v5b.py build B|C|D [--no-tar]
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, pickle, random, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path('/local/home/tianrgua/cvproject')
LOCAL_E_ROOT, POD_E_ROOT = str(ROOT / 'data/yolo'), '/mnt/local/smoke_data'
LOCAL_T_ROOT = str(ROOT / 'data/v4/yolo/E_ET_v1')
POD_T_ROOT = '/scratch/scratch/tianrgua-sandbox/smoke/v4data/E_ET_v1'
POD_V5B_PREFIX = '/scratch/scratch/tianrgua-sandbox/smoke/v5bdata'
POD_SDIS_VAL = '/mnt/local/smoke_data/lists/pod/sdis_val.txt'
POD_SDIS_TEST = '/mnt/local/smoke_data/lists/pod/sdis_test.txt'
LOCAL_SDIS_VAL = ROOT / 'data/yolo/lists/local/sdis_val.txt'
LOCAL_SDIS_TEST = ROOT / 'data/yolo/lists/local/sdis_test.txt'
E_LIST = ROOT / 'data/yolo/lists/local/expB_train.txt'
T_LIST = ROOT / 'data/v5b/teacher/lists/train_T_boreal.txt'
FIREMAN = ROOT / 'data/v5b/yolo/fireman'
SHEFFIELD = ROOT / 'data/v5b/yolo/sheffield_uav'
TEACHER_DIR = ROOT / 'models/v5b/teacher'
RECIPE = ROOT / 'models/v5b/fusion/accurate_recipe.json'
RFDETR_VAL_PARQUET = ROOT / 'reports/v3/eval_rfdetrL_e48_val.raw.parquet'
OUT_YOLO, OUT_STAGE, OUT_MAN = ROOT / 'data/v5b/yolo', ROOT / 'data/v5b/stage', ROOT / 'data/v5b/manifests'
D_PSEUDO = OUT_YOLO / 'D_pseudo'
EXAMPLES_DIR = ROOT / 'reports/v5b/ignore_examples'
ANALYSIS_PKL = OUT_STAGE / 'teacher_analysis_v5b.pkl'

ARM_A_LINES, ARM_A_EPOCHS = 38454, 60
ARM_A_EXPOSURES = ARM_A_LINES * ARM_A_EPOCHS
SEED, FIREMAN_CAP, MIN_EPOCHS = 0, 600, 20
MULT = {'E': 1, 'boreal': 2, 'fireman': 1, 'sheffield': 2}
TEACHERS = ['dino_b4_lora_L', 'dino_b6_lora_Hplus', 'rfdetr_L']
RECIPE_KEY = {'dino_b4_lora_L': 'dino_B4', 'dino_b6_lora_Hplus': 'dino_B6'}
AGREE_IOU = 0.5          # teacher-teacher box agreement (C and D)
NMS_IOU = 0.5            # within-teacher duplicate suppression (RF-DETR ships no NMS)
C_GT_IOU_MAX = 0.1       # region must have IoU < 0.1 with every GT box
C_GT_COVER_MAX = 0.3     # extra guard: region may hide < 30% of any GT box's area
C_REGION_AREA_MAX = 0.5  # extra guard: skip regions whose enclosing box > 50% of image
D_GT_IOU_MAX = 0.3       # pseudo box only if no GT box within IoU >= 0.3
D_DEDUP_IOU = 0.3
# ---- D2 ("compute-assisted"): calibrated 3-teacher WBF (scripts/v5b/fusion.py) over ALL B sources incl. FireMan/Sheffield
D2_PSEUDO = OUT_YOLO / 'D2_pseudo'
ANALYSIS_D2_PKL = OUT_STAGE / 'teacher_analysis_v5b_d2.pkl'
D2_RECIPE = ROOT / 'models/v5b/fusion/teacher_ensemble_d2.json'
B_CORPUS_DIR = OUT_YOLO / 'B_v1'
POD_B_ROOT = f'{POD_V5B_PREFIX}/B_v1'
B_TAR_MEMBERS = OUT_STAGE / 'B_v1.tar.members.txt'
D2_FUSED_FP = 0.10        # fused rule: fused calibrated score >= recipe thresholds_oof['fp0.10']
D2_SINGLE_CAL = 0.98      # single-teacher rule: calibrated score >= 0.98 (= 2% neg-image FP for that teacher on sdis VAL negatives)
D2_DISAGREE_CAL = 0.90    # reporting only: some teacher >= 10%-FP conf away from GT but no pseudo box -> "disagreement image", kept unchanged
D2_GT_COVER_MAX = 0.5     # extra guard (not in D): drop a candidate that is >=50% inside a GT box or that covers >=50% of a GT box (FireMan GT = large region boxes)
D2_INCLUDE_AGREE_RULE = True   # third rule = arm D's rule (>=2 teachers agree IoU>=0.5, each >= its 20%-FP conf) so that D2 pseudo-supervision is a superset of D's
# ---- D3 (teacher -> student on UNLABELLED smoke-positive material): corpus B + pseudo-boxed D3 sources x1 (scripts/v5b/build_d3_sources.py)
D3_SRC = OUT_YOLO / 'D3_src'
D3_PSEUDO = OUT_YOLO / 'D3_pseudo'
ANALYSIS_D3_PKL = OUT_STAGE / 'teacher_analysis_v5b_d3.pkl'
POD_D3_SRC_ROOT = f'{POD_V5B_PREFIX}/D3_src'
D3_TAR_MEMBERS = OUT_STAGE / 'D3_src.tar.members.txt'
D3_EVALS = {'dino_b4_lora_L': 'b4L_d3', 'dino_b6_lora_Hplus': 'b6H_d3', 'rfdetr_L': 'rfdetrL_d3'}


def utcnow():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def log(msg):
    print(f'[{utcnow()}] {msg}', flush=True)


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def read_lines(p):
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def label_path(img):
    return img.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'


def parse_yolo(txt_path):
    """-> (raw_lines, boxes xyxy normalised)"""
    raw, boxes = [], []
    p = Path(txt_path)
    if not p.exists():
        return raw, boxes
    for l in p.read_text().splitlines():
        s = l.split()
        if len(s) < 5:
            continue
        raw.append(l.strip())
        cx, cy, w, h = map(float, s[1:5])
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return raw, boxes


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def inter_over_b(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return (ix * iy) / ab if ab > 0 else 0.0


def nms(boxes, thr):
    keep = []
    for b in sorted(boxes, key=lambda x: -x[4]):
        if all(iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def agree_components(per_teacher, iou_thr):
    nodes = [(t, b) for t, bs in per_teacher.items() for b in bs]
    n = len(nodes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if nodes[i][0] != nodes[j][0] and iou(nodes[i][1], nodes[j][1]) >= iou_thr:
                parent[find(i)] = find(j)
    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(nodes[i])
    return [c for c in comps.values() if len({t for t, _ in c}) >= 2]


# ----------------------------------------------------------------------------- thresholds
def teacher_thresholds():
    """calibrated raw-score thresholds at 10% / 20% neg-image FP on pyro-sdis VAL negatives.
    B4/B6: inverse of the accurate_recipe calibration knots (c = 1 - negFP).  RF-DETR: no
    sdis-val matched_fp json exists in reports/v3 -> 90th/80th percentile of per-image max
    score on sdis-val negatives (reports/v3/eval_rfdetrL_e48_val.raw.parquet)."""
    import pandas as pd
    rec = json.loads(RECIPE.read_text())
    out = {}
    for t, k in RECIPE_KEY.items():
        kr = np.array(rec['models'][k]['calibration']['knots_raw'])
        kc = np.array(rec['models'][k]['calibration']['knots_cal'])
        assert np.all(np.diff(kc) >= 0)

        def inv(c):
            i = min(int(np.searchsorted(kc, c, side='left')), len(kr) - 1)
            return float(kr[i])
        out[t] = {'thr10': inv(0.90), 'thr20': inv(0.80),
                  'method': f'inverse of accurate_recipe.json models.{k}.calibration knots (n_neg={rec["models"][k]["calibration"]["meta"]["n_neg"]})'}
    df = pd.read_parquet(RFDETR_VAL_PARQUET)
    neg = df[df['gt'].apply(len) == 0]
    mx = neg['preds'].apply(lambda p: max([b[4] for b in p]) if len(p) else 0.0).values
    out['rfdetr_L'] = {'thr10': float(np.quantile(mx, 0.9)), 'thr20': float(np.quantile(mx, 0.8)),
                       'method': f'90th/80th percentile of per-image max score on {len(neg)} sdis-val negatives ({RFDETR_VAL_PARQUET.relative_to(ROOT)}); no sdis-val matched_fp json in reports/v3'}
    return out


# ----------------------------------------------------------------------------- entries
def src_of(local_img):
    if '/pyro_sdis/' in local_img:
        return 'sdis'
    if '/dfire_smoke/' in local_img:
        return 'dfire'
    if '/E_ET_v1/' in local_img:
        return 'boreal'
    raise ValueError(local_img)


def base_entries():
    """E + Boreal + FireMan(capped) + Sheffield(all) with local/pod image paths."""
    ents = []
    for l in read_lines(E_LIST):
        ents.append(dict(comp='E', src=src_of(l), local_img=l, local_lbl=label_path(l),
                         pod_img=l.replace(LOCAL_E_ROOT, POD_E_ROOT), origin='author', unit=None))
    for l in read_lines(T_LIST):
        ents.append(dict(comp='boreal', src='boreal', local_img=l, local_lbl=label_path(l),
                         pod_img=l.replace(LOCAL_T_ROOT, POD_T_ROOT),
                         origin='author_verified_empty' if 'Empty-Images' in l else 'author', unit=None))
    # FireMan train split only, per-video cap with uniform stride
    vids = defaultdict(list)
    for f in sorted(os.listdir(FIREMAN / 'images/train')):
        vid, fr = f.rsplit('_frame_', 1)
        vids[vid].append((int(fr.split('.')[0]), f))
    fireman_sel = {}
    for vid, frames in sorted(vids.items()):
        frames.sort()
        if len(frames) > FIREMAN_CAP:
            idx = np.round(np.linspace(0, len(frames) - 1, FIREMAN_CAP)).astype(int)
            pick = [frames[i] for i in sorted(set(idx.tolist()))]
        else:
            pick = frames
        fireman_sel[vid] = {'n_frames': len(frames), 'n_selected': len(pick)}
        for _, f in pick:
            li = str(FIREMAN / 'images/train' / f)
            ents.append(dict(comp='fireman', src='fireman', local_img=li, local_lbl=label_path(li),
                             pod_img=None, origin='author', unit=vid, name=f'fireman__{f}'))
    # Sheffield: ONE burn session -> all of it as TRAIN (train+val+holdout exports), never val/holdout
    for sp in ('train', 'val', 'holdout'):
        for f in sorted(os.listdir(SHEFFIELD / 'images' / sp)):
            li = str(SHEFFIELD / 'images' / sp / f)
            ents.append(dict(comp='sheffield', src='sheffield', local_img=li, local_lbl=label_path(li),
                             pod_img=None, origin='author', unit=f'sheffield_export_{sp}', name=f'sheffield__{sp}__{f}'))
    return ents, fireman_sel


# ----------------------------------------------------------------------------- teacher analysis
def load_teachers():
    T = {}
    for t in TEACHERS:
        d = {}
        with open(TEACHER_DIR / f'{t}__train.jsonl') as f:
            for line in f:
                r = json.loads(line)
                d[r['image']] = (r.get('preds') or [], r.get('w'), r.get('h'), r.get('error'))
        T[t] = d
    return T


def teacher_key(e):
    """key in the teacher jsonl files: local E/T image path, or the B_v1 symlink path for FireMan/Sheffield (as listed for inference)."""
    return str(B_CORPUS_DIR / 'images' / e['name']) if e['comp'] in ('fireman', 'sheffield') else e['local_img']


def load_teachers_all():
    T = {}
    for t in TEACHERS:
        d = {}
        for suffix in ('__train.jsonl', '__train_new.jsonl'):
            p = TEACHER_DIR / f'{t}{suffix}'
            if not p.exists():
                raise FileNotFoundError(p)
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    d[r['image']] = (r.get('preds') or [], r.get('w'), r.get('h'), r.get('error'))
        T[t] = d
    return T


def build_d2_recipe():
    """3-teacher calibrated-WBF recipe (models/v5b/fusion/teacher_ensemble_d2.json): B4/B6 calibration knots copied from
    accurate_recipe.json; RF-DETR knots fitted with ScoreCalibrator.fit on the per-image max score of the 1,158 sdis-val
    negatives (reports/v3/eval_rfdetrL_e48_val.raw.parquet -- the same values whose 90th/80th pct gave D's thr10/thr20);
    WBF params (conf_type/iou_thr/skip_box_thr) and thresholds_oof copied from the recipe, weights 1.0 each."""
    sys.path.insert(0, str(ROOT))
    from scripts.v5b.fusion import ScoreCalibrator
    import pandas as pd
    rec = json.loads(RECIPE.read_text())
    models = {}
    for t, k in RECIPE_KEY.items():
        m = rec['models'][k]
        models[t] = dict(family=m['family'], checkpoint=m['checkpoint'], calibration=m['calibration'],
                         calibration_source=f'accurate_recipe.json models.{k}.calibration (copied; n_neg={m["calibration"]["meta"]["n_neg"]})')
    df = pd.read_parquet(RFDETR_VAL_PARQUET)
    neg = df[df['gt'].apply(len) == 0]
    mx = neg['preds'].apply(lambda p: max([b[4] for b in p]) if len(p) else 0.0).values
    cal = ScoreCalibrator.fit(mx, meta={'fit_fold': 'all_val', 'source': str(RFDETR_VAL_PARQUET.relative_to(ROOT))})
    models['rfdetr_L'] = dict(family='RF-DETR-L @1280 (v3 e48, expB list)', checkpoint='models/v3/rfdetr/checkpoint_best_ema.pth', calibration=cal.to_dict(),
                              calibration_source=f'ScoreCalibrator.fit on per-image max score of {len(neg)} sdis-val negatives ({RFDETR_VAL_PARQUET.relative_to(ROOT)}); '
                                                 'arm D used the 90th/80th percentile of the same values as thr10/thr20',
                              raw_score_at_calibrated={str(c): round(cal.inverse(c), 6) for c in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99)})
    fusion = {k: v for k, v in rec['fusion'].items() if k != 'weights'}
    fusion['weights'] = {t: 1.0 for t in TEACHERS}
    d2 = dict(recipe='teacher_ensemble_d2', created_utc=utcnow(),
              purpose='V5b arm D2 pseudo-labelling only: calibrated WBF of the 3 teacher detectors (NOT the serving ACCURATE recipe; same calibrated scale + WBF params)',
              score_scale=rec['score_scale'], models=models, fusion=fusion, thresholds_oof=rec['thresholds_oof'],
              thresholds_note=f"thresholds_oof copied from accurate_recipe.json (fit on pooled sdis-val OOF for its 4-model ensemble); D2 fused rule uses "
                              f"fp{D2_FUSED_FP:.2f} = {rec['thresholds_oof'][f'fp{D2_FUSED_FP:.2f}']}; single-teacher rule uses calibrated >= {D2_SINGLE_CAL} "
                              f"(2% neg-image FP per teacher). NOTE conf_type={fusion['conf_type']}: the fused score is the max member calibrated score.",
              image_level=rec.get('image_level', {}))
    D2_RECIPE.write_text(json.dumps(d2, indent=1))
    return d2


def cmd_analyze_d2(args):
    sys.path.insert(0, str(ROOT))
    from scripts.v5b.fusion import FusionRecipe, fuse, iou_matrix
    d2 = build_d2_recipe()
    rec = FusionRecipe(d2)
    thr_fused = rec.threshold_for_fp(D2_FUSED_FP)
    cals = rec.calibrators
    thr = {t: {'raw_at_cal0.98_single_rule': cals[t].inverse(D2_SINGLE_CAL), 'raw_at_cal0.90': cals[t].inverse(0.90), 'raw_at_cal0.80': cals[t].inverse(0.80),
               'method': d2['models'][t]['calibration_source']} for t in TEACHERS}
    thr['fused'] = {'calibrated_threshold': thr_fused, 'method': f"accurate_recipe.json thresholds_oof['fp{D2_FUSED_FP:.2f}'] applied to the 3-teacher WBF ({rec.params})"}
    thr['agree'] = {'enabled': D2_INCLUDE_AGREE_RULE, 'calibrated_threshold_per_teacher': 0.80, 'agree_iou': AGREE_IOU,
                    'method': "arm D rule re-applied on the calibrated scale: >=2 of 3 teachers agree (IoU>=0.5), each box calibrated >= 0.80 (20% neg-image FP); box = raw-score-weighted mean"}
    log(f'D2 thresholds: {json.dumps({t: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items() if k != "method"} for t, d in thr.items()})}')
    ents, _ = base_entries()
    T = load_teachers_all()
    res, stats = {}, Counter()
    for e in ents:
        key = teacher_key(e)
        raw, gt = parse_yolo(e['local_lbl'])
        explicit_neg = e['origin'] == 'author_verified_empty' or (e['comp'] in ('fireman', 'sheffield') and not gt)
        per = {}
        for t in TEACHERS:
            row = T[t].get(key)
            if row is None or row[3]:
                stats[f'missing_or_error__{t}'] += 1; per[t] = []
                continue
            per[t] = nms([list(p) for p in row[0]], NMS_IOU)
        calb = {}
        for t in TEACHERS:
            if per[t]:
                a = np.array(per[t], dtype=float); calb[t] = np.column_stack([a[:, :4], cals[t](a[:, 4])])
            else:
                calb[t] = np.zeros((0, 5))
        fused = np.array(fuse({t: per[t] for t in TEACHERS}, rec), dtype=float) if any(per.values()) else np.zeros((0, 5))
        single_fire = sorted(t for t in TEACHERS if len(calb[t]) and calb[t][:, 4].max() >= D2_SINGLE_CAL)
        r = dict(src=e['src'], comp=e['comp'], n_gt=len(gt), explicit_neg=explicit_neg, pseudo=[], skipped=Counter(), single_fire_2pct=single_fire,
                 fused_fire=bool(len(fused) and fused[:, 4].max() >= thr_fused))
        stats['images'] += 1
        if explicit_neg:
            # author says "no smoke": kept unchanged; teacher activity here is recorded (not acted on)
            stats['explicit_neg_images'] += 1
            if r['fused_fire']:
                stats['explicit_neg_images_fused_fire'] += 1; stats[f"explicit_neg_images_fused_fire__{e['src']}"] += 1
            if single_fire:
                stats['explicit_neg_images_single_fire_2pct'] += 1; stats[f"explicit_neg_images_single_fire_2pct__{e['src']}"] += 1
            res[key] = r
            continue
        cands = []
        for fb in fused:
            if fb[4] < thr_fused:
                continue
            members, sc_raw, sc_cal = [], {}, {}
            for t in TEACHERS:
                if not len(calb[t]):
                    continue
                ious = iou_matrix(calb[t][:, :4], fb[None, :4])[:, 0]
                j = int(np.argmax(ious))
                if ious[j] > rec.params['iou_thr']:
                    members.append(t); sc_raw[t] = float(per[t][j][4]); sc_cal[t] = float(calb[t][j, 4])
            cands.append(dict(box=[float(min(max(v, 0.0), 1.0)) for v in fb[:4]], rule='fused', fused_score=float(fb[4]), teachers=members,
                              scores=sc_raw, cal=sc_cal, order=float(fb[4])))
        for t in TEACHERS:
            cb = calb[t]
            for i in np.where(cb[:, 4] >= D2_SINGLE_CAL)[0]:
                cands.append(dict(box=[float(v) for v in cb[i, :4]], rule='single', fused_score=None, teachers=[t],
                                  scores={t: float(per[t][i][4])}, cal={t: float(cb[i, 4])}, order=float(cb[i, 4]) - 1.0))
        if D2_INCLUDE_AGREE_RULE:
            per20 = {t: [list(per[t][i]) + [float(calb[t][i, 4])] for i in np.where(calb[t][:, 4] >= 0.80)[0]] for t in TEACHERS if len(calb[t])}
            per20 = {t: v for t, v in per20.items() if v}
            for comp in agree_components(per20, AGREE_IOU):
                arr = np.array([b[:5] for _, b in comp], dtype=float)
                w = arr[:, 4]
                box = (arr[:, :4] * w[:, None]).sum(0) / w.sum()
                tset = sorted({t for t, _ in comp})
                cands.append(dict(box=[float(min(max(v, 0.0), 1.0)) for v in box], rule='agree', fused_score=None, teachers=tset,
                                  scores={t: float(max(b[4] for tt, b in comp if tt == t)) for t in tset},
                                  cal={t: float(max(b[5] for tt, b in comp if tt == t)) for t in tset},
                                  order=float(np.mean([max(b[5] for tt, b in comp if tt == t) for t in tset])) - 2.0))
        cands.sort(key=lambda c: -c['order'])          # fused (score desc) > single (cal desc) > agree (mean cal desc)
        for c in cands:
            if any(iou(c['box'], g) >= D_GT_IOU_MAX for g in gt):
                r['skipped'][f"{c['rule']}_near_gt"] += 1
                continue
            if any(inter_over_b(g, c['box']) >= D2_GT_COVER_MAX or inter_over_b(c['box'], g) >= D2_GT_COVER_MAX for g in gt):
                r['skipped'][f"{c['rule']}_inside_or_covers_gt"] += 1
                continue
            if any(iou(c['box'], p['box']) >= D_DEDUP_IOU for p in r['pseudo']):
                r['skipped'][f"{c['rule']}_dedup"] += 1
                continue
            c.pop('order', None)
            r['pseudo'].append(c)
        if not r['pseudo']:
            fires = [t for t in TEACHERS if any(cb[4] >= D2_DISAGREE_CAL and not any(iou(cb[:4], g) >= D_GT_IOU_MAX for g in gt) for cb in calb[t])]
            if fires:
                stats['disagreement_images_fire10pct_noGT_unchanged'] += 1
                stats[f"disagreement_images_fire10pct_noGT_unchanged__{e['src']}"] += 1
        res[key] = r
        stats['D2_images_pseudo'] += bool(r['pseudo']); stats['D2_boxes'] += len(r['pseudo'])
        for p in r['pseudo']:
            stats[f"D2_boxes_rule_{p['rule']}"] += 1
            stats[f"D2_boxes_rule_{p['rule']}__{e['src']}"] += 1
        for k, v in r['skipped'].items():
            stats[k] += v
    OUT_STAGE.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_D2_PKL, 'wb') as f:
        pickle.dump({'thresholds': thr, 'results': res, 'stats': dict(stats), 'created_utc': utcnow(), 'recipe': str(D2_RECIPE),
                     'params': dict(NMS_IOU=NMS_IOU, AGREE_IOU=AGREE_IOU, D_GT_IOU_MAX=D_GT_IOU_MAX, D_DEDUP_IOU=D_DEDUP_IOU, D2_FUSED_FP=D2_FUSED_FP,
                                    D2_SINGLE_CAL=D2_SINGLE_CAL, D2_DISAGREE_CAL=D2_DISAGREE_CAL, D2_GT_COVER_MAX=D2_GT_COVER_MAX,
                                    D2_INCLUDE_AGREE_RULE=D2_INCLUDE_AGREE_RULE, wbf=rec.params, weights=rec.weights)}, f)
    log(f'D2 analysis -> {ANALYSIS_D2_PKL}: {json.dumps(dict(sorted(stats.items())))}')
    return 0


def d3_candidates(per, calb, fused, rec, thr_fused):
    """D2 rules without any GT filter: fused>=fp0.10 | single>=0.98 | agree(>=2 teachers cal>=0.80, IoU>=0.5); fused > single > agree; dedup IoU>=D_DEDUP_IOU."""
    from scripts.v5b.fusion import iou_matrix
    cands = []
    for fb in fused:
        if fb[4] < thr_fused:
            continue
        members, sc_raw, sc_cal = [], {}, {}
        for t in TEACHERS:
            if not len(calb[t]):
                continue
            ious = iou_matrix(calb[t][:, :4], fb[None, :4])[:, 0]; j = int(np.argmax(ious))
            if ious[j] > rec.params['iou_thr']:
                members.append(t); sc_raw[t] = float(per[t][j][4]); sc_cal[t] = float(calb[t][j, 4])
        cands.append(dict(box=[float(min(max(v, 0.0), 1.0)) for v in fb[:4]], rule='fused', fused_score=float(fb[4]), teachers=members, scores=sc_raw, cal=sc_cal, order=float(fb[4])))
    for t in TEACHERS:
        cb = calb[t]
        for i in np.where(cb[:, 4] >= D2_SINGLE_CAL)[0]:
            cands.append(dict(box=[float(v) for v in cb[i, :4]], rule='single', fused_score=None, teachers=[t], scores={t: float(per[t][i][4])}, cal={t: float(cb[i, 4])}, order=float(cb[i, 4]) - 1.0))
    per20 = {t: [list(per[t][i]) + [float(calb[t][i, 4])] for i in np.where(calb[t][:, 4] >= 0.80)[0]] for t in TEACHERS if len(calb[t])}
    per20 = {t: v for t, v in per20.items() if v}
    for comp in agree_components(per20, AGREE_IOU):
        arr = np.array([b[:5] for _, b in comp], dtype=float); w = arr[:, 4]
        box = (arr[:, :4] * w[:, None]).sum(0) / w.sum(); tset = sorted({t for t, _ in comp})
        cands.append(dict(box=[float(min(max(v, 0.0), 1.0)) for v in box], rule='agree', fused_score=None, teachers=tset,
                          scores={t: float(max(b[4] for tt, b in comp if tt == t)) for t in tset}, cal={t: float(max(b[5] for tt, b in comp if tt == t)) for t in tset},
                          order=float(np.mean([max(b[5] for tt, b in comp if tt == t) for t in tset])) - 2.0))
    cands.sort(key=lambda c: -c['order'])
    kept, skipped = [], Counter()
    for c in cands:
        if any(iou(c['box'], p['box']) >= D_DEDUP_IOU for p in kept):
            skipped[f"{c['rule']}_dedup"] += 1; continue
        c.pop('order', None); kept.append(c)
    return kept, skipped


def cmd_analyze_d3(args):
    """merge raw D3 teacher shards -> <teacher>__train_d3.jsonl (+ TEACHER_POOL.md section), then pseudo-box every D3 source image."""
    import gzip
    sys.path.insert(0, str(ROOT))
    from scripts.v5b.fusion import FusionRecipe, fuse
    srcs = list(csv.DictReader(open(D3_SRC / 'sources.csv')))
    targets = [str(D3_SRC / 'images' / r['name']) for r in srcs]; tset = set(targets)
    by_name = {str(D3_SRC / 'images' / r['name']): r for r in srcs}
    tman = json.loads((ROOT / 'data/v5b/teacher/lists/TARGETS_D3.json').read_text())
    T, pool = {}, {}
    for t, ev in D3_EVALS.items():
        rows = {}
        for f in sorted((TEACHER_DIR / 'raw' / ev).glob('*_preds.jsonl.gz')):
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line); rows[r['image']] = r
        extra = [p for p in rows if p not in tset]; assert not extra, (t, extra[:2])
        missing = [p for p in targets if p not in rows]; errs = [p for p in targets if p in rows and 'error' in rows[p]]
        job = json.loads((TEACHER_DIR / 'raw' / ev / 'job.json').read_text())
        summ = [json.loads(l) for l in (TEACHER_DIR / 'raw' / ev / 'summary.jsonl').read_text().splitlines() if l.strip()]
        with (TEACHER_DIR / f'{t}__train_d3.jsonl').open('w') as fh:
            for p in targets:
                if p not in rows: continue
                r = rows[p]; rec = {'image': p, 'teacher': t, 'label_type': 'model-assisted'}
                if 'error' in r: rec['error'] = r['error']
                else: rec.update(w=r.get('w'), h=r.get('h'), preds=r.get('preds', []))
                fh.write(json.dumps(rec) + '\n')
        (TEACHER_DIR / f'missing__{t}_d3.txt').write_text('\n'.join(missing) + ('\n' if missing else ''))
        T[t] = {p: (rows[p].get('preds') or [], rows[p].get('w'), rows[p].get('h'), rows[p].get('error')) for p in targets if p in rows}
        ms = {p: max((b[4] for b in rows[p]['preds']), default=0.0) for p in targets if p in rows and 'error' not in rows[p]}
        pool[t] = dict(job=job.get('job'), final_state=job.get('final_state'), wall_s=job.get('wall_s'), ckpt_sha256=summ[0]['ckpt_sha256'] if summ else None, expect_sha=job.get('expect_sha'),
                       rows=len(rows), targets=len(targets), unreadable=len(errs), missing=len(missing),
                       firing_by_source={s: {str(th): sum(1 for p, m in ms.items() if by_name[p]['source'] == s and m >= th) for th in (0.5, 0.9)} for s in ('borealclip', 'firesense', 'dfirefire')},
                       n_by_source={s: sum(1 for p in ms if by_name[p]['source'] == s) for s in ('borealclip', 'firesense', 'dfirefire')})
    md = [f'\n## D3 extension -- UNLABELLED smoke-positive material (model-assisted; generated {utcnow()})\n',
          f'Targets `data/v5b/teacher/lists/train_d3_sources.txt` (sha {tman["sha256"][:16]}; counts {json.dumps(tman["counts"])}); images staged from `data/v5b/stage/D3_src.tar` '
          f'via DATA_TARS to `{POD_D3_SRC_ROOT}` (Boreal clip frames resized to max side 1920). No author boxes exist here; video-level labels only (D-Fire fire-only: smoke unknown).\n',
          '| teacher | ckpt sha256 | job (wall) | rows / targets | unreadable | missing | borealclip firing @0.5 / @0.9 | firesense @0.5 / @0.9 | dfirefire @0.5 / @0.9 |', '|---|---|---|---|---|---|---|---|---|']
    for t, p in pool.items():
        fb = p['firing_by_source']; nb = p['n_by_source']
        md.append(f"| {t} | {(p['ckpt_sha256'] or '?')[:16]} | {p['job']} {p['final_state']} {round((p['wall_s'] or 0) / 60)} min | {p['rows']:,} / {p['targets']:,} | {p['unreadable']} | {p['missing']} | "
                  + ' | '.join(f"{fb[s]['0.5']:,} / {fb[s]['0.9']:,} (of {nb[s]:,})" for s in ('borealclip', 'firesense', 'dfirefire')) + ' |')
    with (TEACHER_DIR / 'TEACHER_POOL.md').open('a') as fh:
        fh.write('\n'.join(md) + '\n')
    sp = TEACHER_DIR / 'teacher_pool_stats.json'; allst = json.loads(sp.read_text()) if sp.exists() else {}
    allst['d3_sources'] = dict(generated_utc=utcnow(), targets=tman['counts'], teachers=pool); sp.write_text(json.dumps(allst, indent=1))
    # ---- pseudo boxes (D2 rules, no GT)
    d2 = json.loads(D2_RECIPE.read_text()) if D2_RECIPE.exists() else build_d2_recipe()
    rec = FusionRecipe(d2); thr_fused = rec.threshold_for_fp(D2_FUSED_FP); cals = rec.calibrators
    thr = {t: {'raw_at_cal0.98_single_rule': cals[t].inverse(D2_SINGLE_CAL), 'raw_at_cal0.80_agree_rule': cals[t].inverse(0.80), 'method': d2['models'][t]['calibration_source']} for t in TEACHERS}
    thr['fused'] = {'calibrated_threshold': thr_fused}; thr['agree'] = {'calibrated_threshold_per_teacher': 0.80, 'agree_iou': AGREE_IOU}
    res, stats = {}, Counter()
    for p in targets:
        meta = by_name[p]; src = meta['source']
        per = {}
        for t in TEACHERS:
            row = T[t].get(p)
            if row is None or row[3]:
                stats[f'missing_or_error__{t}'] += 1; per[t] = []; continue
            per[t] = nms([list(b) for b in row[0]], NMS_IOU)
        calb = {}
        for t in TEACHERS:
            if per[t]:
                a = np.array(per[t], dtype=float); calb[t] = np.column_stack([a[:, :4], cals[t](a[:, 4])])
            else:
                calb[t] = np.zeros((0, 5))
        fused = np.array(fuse({t: per[t] for t in TEACHERS}, rec), dtype=float) if any(per.values()) else np.zeros((0, 5))
        kept, skipped = d3_candidates(per, calb, fused, rec, thr_fused)
        max_cal = max([calb[t][:, 4].max() for t in TEACHERS if len(calb[t])], default=0.0)
        res[p] = dict(src=src, unit=meta['unit'], video_label=meta['video_label'], video_level_support=meta['video_level_support'] == 'true',
                      train_eligible=meta['train_eligible'] == 'true', pseudo=kept, skipped=skipped, max_cal=float(max_cal),
                      any_teacher_box=bool(any(per.values())))
        stats[f'{src}__images'] += 1
        stats[f'{src}__images_with_pseudo'] += bool(kept); stats[f'{src}__boxes'] += len(kept)
        for c in kept: stats[f"{src}__boxes_rule_{c['rule']}"] += 1
        if not meta['train_eligible'] == 'true':
            stats[f'{src}__not_train_eligible_images'] += 1; stats[f'{src}__not_train_eligible_images_with_pseudo'] += bool(kept)
        if src == 'dfirefire' and not kept:
            stats['dfirefire__excluded_no_pseudo_box'] += 1
    OUT_STAGE.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_D3_PKL, 'wb') as f:
        pickle.dump({'thresholds': thr, 'results': res, 'stats': dict(stats), 'created_utc': utcnow(), 'recipe': str(D2_RECIPE), 'pool': pool,
                     'params': dict(NMS_IOU=NMS_IOU, AGREE_IOU=AGREE_IOU, D_DEDUP_IOU=D_DEDUP_IOU, D2_FUSED_FP=D2_FUSED_FP, D2_SINGLE_CAL=D2_SINGLE_CAL,
                                    gt_filter='none (no author boxes exist)', keep_rule='image kept only if >=1 pseudo box; no_smoke clip frames and D-Fire fire-only images without a box are EXCLUDED',
                                    wbf=rec.params, weights=rec.weights)}, f)
    log(f'D3 analysis -> {ANALYSIS_D3_PKL}: {json.dumps(dict(sorted(stats.items())))}')
    return 0


def write_d2_tar(tar_path, cdir):
    """D2 stage tar = labels/lists/yamls + E/Boreal pseudo images DEREFERENCED (copies) + FireMan/Sheffield pseudo images as
    RELATIVE symlinks (../../B_v1/images/<name>, resolved on the pod against the arm-B stage) -> small tar.
    Returns (member names, {symlink member: normalised link target relative to the extract root})."""
    import tarfile
    names, links = set(), {}
    with tarfile.open(tar_path, 'w') as tf:
        for dp, dns, fns in os.walk(cdir):
            dns.sort(); fns.sort()
            arc_dir = os.path.relpath(dp, OUT_YOLO)
            tf.add(dp, arcname=arc_dir, recursive=False)
            for fn in fns:
                p = os.path.join(dp, fn); arc = f'{arc_dir}/{fn}'
                if os.path.islink(p) and not os.path.isabs(os.readlink(p)):
                    tf.add(p, arcname=arc, recursive=False)                      # keep the relative symlink
                    links[arc] = os.path.normpath(os.path.join(arc_dir, os.readlink(p)))
                else:
                    tf.add(os.path.realpath(p), arcname=arc, recursive=False)    # dereference
                names.add(arc)
    return names, links


def cmd_analyze(args):
    thr = teacher_thresholds()
    log(f'thresholds: {json.dumps({t: {k: round(v, 5) for k, v in d.items() if k != "method"} for t, d in thr.items()})}')
    ents, _ = base_entries()
    targets = [e for e in ents if e['comp'] in ('E', 'boreal')]
    T = load_teachers()
    res = {}
    stats = Counter()
    for e in targets:
        img = e['local_img']
        raw, gt = parse_yolo(e['local_lbl'])
        verified_empty = e['origin'] == 'author_verified_empty'
        per10, per20, fire20 = {}, {}, []
        for t in TEACHERS:
            preds = T[t][img][0]
            p10 = nms([p for p in preds if p[4] >= thr[t]['thr10']], NMS_IOU)
            p20 = nms([p for p in preds if p[4] >= thr[t]['thr20']], NMS_IOU)
            if p10:
                per10[t] = p10
            if p20:
                per20[t] = p20
                fire20.append(t)
        r = dict(src=e['src'], n_gt=len(gt), n_fire20=len(fire20), regions=[], pseudo=[], skipped=Counter())
        # --- C: ignore regions at 10%-FP, >=2 teachers agree, away from GT
        if verified_empty:
            if per10 and agree_components(per10, AGREE_IOU):
                r['skipped']['C_verified_empty_boreal'] += 1
        else:
            for comp in agree_components(per10, AGREE_IOU):
                rects = [b[:4] for _, b in comp]
                R = [min(x[0] for x in rects), min(x[1] for x in rects), max(x[2] for x in rects), max(x[3] for x in rects)]
                if (R[2] - R[0]) * (R[3] - R[1]) > C_REGION_AREA_MAX:
                    r['skipped']['C_region_too_large'] += 1
                    continue
                if any(iou(R, g) >= C_GT_IOU_MAX for g in gt):
                    r['skipped']['C_overlaps_gt_iou'] += 1
                    continue
                if any(inter_over_b(R, g) >= C_GT_COVER_MAX for g in gt):
                    r['skipped']['C_would_cover_gt'] += 1
                    continue
                r['regions'].append(dict(enclosing=R, rects=rects, teachers=sorted({t for t, _ in comp}),
                                         scores={t: max(b[4] for tt, b in comp if tt == t) for t in {t for t, _ in comp}}))
        # --- D: pseudo boxes at 20%-FP, >=2 teachers agree IoU>=0.5, no GT within IoU>=0.3
        if verified_empty:
            if per20 and agree_components(per20, AGREE_IOU):
                r['skipped']['D_verified_empty_boreal'] += 1
        else:
            cands = []
            for comp in agree_components(per20, AGREE_IOU):
                arr = np.array([b for _, b in comp])
                w = arr[:, 4]
                box = (arr[:, :4] * w[:, None]).sum(0) / w.sum()
                box = [float(min(max(v, 0.0), 1.0)) for v in box]
                tset = sorted({t for t, _ in comp})
                sc = {t: float(max(b[4] for tt, b in comp if tt == t)) for t in tset}
                cands.append(dict(box=box, teachers=tset, scores=sc, score=float(np.mean(list(sc.values())))))
            cands.sort(key=lambda c: -c['score'])
            for c in cands:
                if any(iou(c['box'], g) >= D_GT_IOU_MAX for g in gt):
                    r['skipped']['D_near_gt'] += 1
                    continue
                if any(iou(c['box'], p['box']) >= D_DEDUP_IOU for p in r['pseudo']):
                    r['skipped']['D_dedup'] += 1
                    continue
                r['pseudo'].append(c)
        res[img] = r
        stats['images'] += 1
        stats['C_images_masked'] += bool(r['regions']); stats['C_regions'] += len(r['regions'])
        stats['D_images_pseudo'] += bool(r['pseudo']); stats['D_boxes'] += len(r['pseudo'])
        stats[f'D_disagree_exactly1_fire20'] += (len(fire20) == 1)
        stats[f'D_disagree_exactly1_fire20_noGT'] += (len(fire20) == 1 and not gt)
        for k, v in r['skipped'].items():
            stats[k] += v
    OUT_STAGE.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_PKL, 'wb') as f:
        pickle.dump({'thresholds': thr, 'results': res, 'stats': dict(stats), 'created_utc': utcnow(),
                     'params': dict(AGREE_IOU=AGREE_IOU, NMS_IOU=NMS_IOU, C_GT_IOU_MAX=C_GT_IOU_MAX, C_GT_COVER_MAX=C_GT_COVER_MAX,
                                    C_REGION_AREA_MAX=C_REGION_AREA_MAX, D_GT_IOU_MAX=D_GT_IOU_MAX, D_DEDUP_IOU=D_DEDUP_IOU)}, f)
    log(f'analysis -> {ANALYSIS_PKL}: {json.dumps(dict(stats))}')
    return 0


# ----------------------------------------------------------------------------- masking worker
def _mask_worker(job):
    src, dst, rects = job
    import cv2
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    if im is None:
        return (src, 'unreadable', 0)
    H, W = im.shape[:2]
    mean = im.reshape(-1, 3).mean(0).round().astype(np.uint8)
    n = 0
    for x1, y1, x2, y2 in rects:
        X1, Y1 = max(0, int(round(x1 * W))), max(0, int(round(y1 * H)))
        X2, Y2 = min(W, int(round(x2 * W))), min(H, int(round(y2 * H)))
        if X2 > X1 and Y2 > Y1:
            im[Y1:Y2, X1:X2] = mean
            n += 1
    ok = cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return (src, 'ok' if ok else 'write_failed', n)


def write_examples(picks, analysis):
    import cv2
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for old in EXAMPLES_DIR.glob('*'):
        old.unlink()
    index = []
    for i, (orig, masked, r) in enumerate(picks):
        a = cv2.imread(orig); b = cv2.imread(masked)
        H, W = a.shape[:2]
        _, gt = parse_yolo(label_path(orig))
        for g in gt:
            cv2.rectangle(a, (int(g[0] * W), int(g[1] * H)), (int(g[2] * W), int(g[3] * H)), (0, 255, 0), max(2, W // 640))
        for reg in r['regions']:
            for x1, y1, x2, y2 in reg['rects']:
                cv2.rectangle(a, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), (0, 0, 255), max(2, W // 640))
        s = 960 / W
        a = cv2.resize(a, (960, int(H * s))); b = cv2.resize(b, (960, int(H * s)))
        panel = np.concatenate([a, b], axis=1)
        cv2.putText(panel, 'BEFORE: GT green, ignore-region red', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(panel, 'AFTER: region filled with image mean (labels unchanged)', (970, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out = EXAMPLES_DIR / f'{i:02d}_{r["src"]}_{Path(orig).stem[:60]}.jpg'
        cv2.imwrite(str(out), panel, [cv2.IMWRITE_JPEG_QUALITY, 85])
        index.append(dict(example=str(out), original=orig, masked_copy=masked, n_gt=r['n_gt'],
                          regions=[dict(enclosing=reg['enclosing'], teachers=reg['teachers'], scores=reg['scores']) for reg in r['regions']]))
    (EXAMPLES_DIR / 'index.json').write_text(json.dumps(dict(note='agent-selected examples of the C ignore-region approximation; NOT human review', examples=index), indent=1))


# ----------------------------------------------------------------------------- build
def cmd_build(args):
    corpus = f'{args.arm}_v1'
    cdir = OUT_YOLO / corpus
    pod_root = f'{POD_V5B_PREFIX}/{corpus}'
    if cdir.exists():
        shutil.rmtree(cdir)
    for d in ('images', 'labels', 'lists/local', 'lists/pod', 'yamls'):
        (cdir / d).mkdir(parents=True)
    ents, fireman_sel = base_entries()
    analysis = pickle.load(open({'D2': ANALYSIS_D2_PKL, 'D3': ANALYSIS_D3_PKL}.get(args.arm, ANALYSIS_PKL), 'rb')) if args.arm in ('C', 'D', 'D2', 'D3') else None
    mult = dict(MULT)
    if args.arm == 'D3':
        mult['d3'] = 1

    # --- new-data files (FireMan capped train + Sheffield all): symlinks (tar dereferences) + label copies
    n_new_boxes = Counter(); n_new_empty = Counter()
    for e in ents:
        if e['comp'] in ('fireman', 'sheffield') and args.arm in ('D2', 'D3'):
            # NOT re-staged: unchanged new-source images point at the B_v1 stage (local symlink dir / pod root staged by arm B)
            b_img = B_CORPUS_DIR / 'images' / e['name']; b_lbl = B_CORPUS_DIR / 'labels' / (Path(e['name']).stem + '.txt')
            raw, boxes = parse_yolo(e['local_lbl'])
            assert b_img.exists() and b_lbl.exists() and read_lines(b_lbl) == raw, (b_img, 'B_v1 file missing or label differs from source')
            e['corpus_img'] = str(b_img); e['pod_img'] = f"{POD_B_ROOT}/images/{e['name']}"
            n_new_boxes[e['comp']] += len(boxes); n_new_empty[e['comp']] += (len(boxes) == 0)
        elif e['comp'] in ('fireman', 'sheffield'):
            os.symlink(e['local_img'], cdir / 'images' / e['name'])
            raw, boxes = parse_yolo(e['local_lbl'])
            assert all(l.split()[0] == '0' for l in raw), e['local_lbl']
            (cdir / 'labels' / (Path(e['name']).stem + '.txt')).write_text('\n'.join(raw) + ('\n' if raw else ''))
            e['corpus_img'] = str(cdir / 'images' / e['name'])
            e['pod_img'] = f"{pod_root}/images/{e['name']}"
            n_new_boxes[e['comp']] += len(boxes); n_new_empty[e['comp']] += (len(boxes) == 0)

    masked_info, pseudo_info = None, None
    # --- C: masked copies
    if args.arm == 'C':
        res = analysis['results']
        jobs = []
        for e in ents:
            r = res.get(e['local_img'])
            if r and r['regions']:
                name = f"masked__{e['src']}__{Path(e['local_img']).name}"
                dst = cdir / 'images' / name
                jobs.append((e['local_img'], str(dst), [rc for reg in r['regions'] for rc in reg['rects']]))
                raw, _ = parse_yolo(e['local_lbl'])
                (cdir / 'labels' / (Path(name).stem + '.txt')).write_text('\n'.join(raw) + ('\n' if raw else ''))
                e['corpus_img'] = str(dst); e['pod_img'] = f'{pod_root}/images/{name}'; e['origin'] = 'author_labels+masked_pixels'
        log(f'C: writing {len(jobs)} masked copies ...')
        status = Counter(); n_rects = 0
        with ProcessPoolExecutor(max_workers=48) as ex:
            for src, st, n in ex.map(_mask_worker, jobs, chunksize=16):
                status[st] += 1; n_rects += n
        assert status.get('ok', 0) == len(jobs), status
        by_src = Counter(); reg_by_src = Counter(); teacher_combo = Counter()
        for img, r in res.items():
            if r['regions']:
                by_src[r['src']] += 1; reg_by_src[r['src']] += len(r['regions'])
                for reg in r['regions']:
                    teacher_combo['+'.join(reg['teachers'])] += 1
        masked_info = dict(images_masked=len(jobs), regions_masked=sum(len(res[j[0]]['regions']) for j in jobs),
                           rects_filled=n_rects, images_by_source=dict(by_src), regions_by_source=dict(reg_by_src),
                           regions_by_teacher_combo=dict(teacher_combo), skipped=analysis['stats'],
                           dfire_fire_only_note='D-Fire fire-only images (944 official-train) are NOT in E (excluded at v1 export) -> none in this corpus, 0 masked',
                           fireman_unknown_conflict_note='FireMan train RGB has smoke_presence yes/no only (no unknown); 9 CVAT-conflict frames were quarantined at export (data/v5b/yolo/fireman/conflicts.txt) and are not in this corpus; FireMan/Sheffield have no teacher predictions -> no masks on them')
        # examples: up to 6 sdis / 4 dfire / 2 boreal
        rnd = random.Random(SEED)
        bysrc = defaultdict(list)
        for src, dst, _ in jobs:
            bysrc[res[src]['src']].append((src, dst, res[src]))
        picks = []
        for s, k in (('sdis', 6), ('dfire', 4), ('boreal', 2)):
            rnd.shuffle(bysrc[s]); picks += bysrc[s][:k]
        for s in ('sdis', 'dfire', 'boreal'):
            for x in bysrc[s][12:]:
                if len(picks) >= 12: break
                if x not in picks: picks.append(x)
        write_examples(picks[:12], analysis)
        masked_info['examples_dir'] = str(EXAMPLES_DIR)

    # --- D: pseudo labels (GT lines first, then pseudo), image symlink copies, sidecar CSV
    if args.arm == 'D':
        res = analysis['results']
        if D_PSEUDO.exists():
            shutil.rmtree(D_PSEUDO)
        (D_PSEUDO / 'labels').mkdir(parents=True)
        rows = []; by_src_img = Counter(); by_src_box = Counter(); on_gt_empty = Counter(); combo = Counter()
        for e in ents:
            r = res.get(e['local_img'])
            if r and r['pseudo']:
                name = f"pseudo__{e['src']}__{Path(e['local_img']).name}"
                raw, _ = parse_yolo(e['local_lbl'])
                lines = list(raw)
                for p in r['pseudo']:
                    x1, y1, x2, y2 = p['box']
                    lines.append(f'0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}')
                    rows.append(dict(image=e['local_img'], pod_image=f'{pod_root}/images/{name}', source=e['src'],
                                     x1n=f'{x1:.6f}', y1n=f'{y1:.6f}', x2n=f'{x2:.6f}', y2n=f'{y2:.6f}',
                                     teachers=';'.join(p['teachers']), scores=';'.join(f"{p['scores'][t]:.4f}" for t in p['teachers']),
                                     n_teachers=len(p['teachers']), n_gt_boxes=r['n_gt'], label_type='model-assisted'))
                    combo['+'.join(p['teachers'])] += 1
                (D_PSEUDO / 'labels' / (Path(name).stem + '.txt')).write_text('\n'.join(lines) + '\n')
                os.symlink(D_PSEUDO / 'labels' / (Path(name).stem + '.txt'), cdir / 'labels' / (Path(name).stem + '.txt'))
                os.symlink(e['local_img'], cdir / 'images' / name)
                e['corpus_img'] = str(cdir / 'images' / name); e['pod_img'] = f'{pod_root}/images/{name}'
                e['origin'] = 'author+model-assisted'; e['mult_override'] = 1
                by_src_img[e['src']] += 1; by_src_box[e['src']] += len(r['pseudo']); on_gt_empty[e['src']] += (r['n_gt'] == 0)
        with open(D_PSEUDO / 'pseudo_boxes.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        (D_PSEUDO / 'README.md').write_text(
            '# D_pseudo -- MODEL-ASSISTED pseudo boxes for arm D (never author labels, never human-reviewed)\n'
            f'Generated {utcnow()} by scripts/v5b/build_corpus_v5b.py from models/v5b/teacher/*__train.jsonl.\n'
            f'Rule: >=2 of 3 teachers agree (IoU>={AGREE_IOU}), each >= its calibrated 20%-neg-FP conf '
            f'({json.dumps({t: round(d["thr20"], 4) for t, d in analysis["thresholds"].items()})}), no GT box within IoU>={D_GT_IOU_MAX}; '
            'Boreal human-verified-empty frames excluded. Label files = GT lines first, then pseudo lines. Sidecar: pseudo_boxes.csv.\n')
        pseudo_info = dict(images_with_pseudo=sum(by_src_img.values()), pseudo_boxes=len(rows), images_by_source=dict(by_src_img),
                           boxes_by_source=dict(by_src_box), images_gt_empty_by_source=dict(on_gt_empty), boxes_by_teacher_combo=dict(combo),
                           disagreement_images_exactly_one_teacher_fires_20pct=analysis['stats'].get('D_disagree_exactly1_fire20', 0),
                           disagreement_images_exactly_one_teacher_fires_20pct_and_no_gt=analysis['stats'].get('D_disagree_exactly1_fire20_noGT', 0),
                           skipped=analysis['stats'], labels_dir=str(D_PSEUDO / 'labels'), sidecar_csv=str(D_PSEUDO / 'pseudo_boxes.csv'),
                           weighting_approximation='images that received pseudo boxes appear x1 in the list (Boreal normally x2); GT-only images keep B multiplicities; explicit negatives and teacher-disagreement images kept unchanged')

    # --- D2: ensemble pseudo labels on ALL sources (GT lines first, then pseudo), sidecar CSV, images as symlinks (see write_d2_tar)
    if args.arm == 'D2':
        res = analysis['results']; st = analysis['stats']; thr = analysis['thresholds']
        if D2_PSEUDO.exists():
            shutil.rmtree(D2_PSEUDO)
        (D2_PSEUDO / 'labels').mkdir(parents=True)
        rows = []; by_src_img = Counter(); by_src_box = Counter(); on_gt_empty = Counter(); combo = Counter(); by_rule = Counter()
        by_rule_src = defaultdict(Counter); fused_nt = Counter(); single_t = Counter(); by_comp_img = Counter()
        for e in ents:
            r = res.get(teacher_key(e))
            if not (r and r['pseudo']):
                continue
            name = f"pseudo__{e['name']}" if e['comp'] in ('fireman', 'sheffield') else f"pseudo__{e['src']}__{Path(e['local_img']).name}"
            raw, _ = parse_yolo(e['local_lbl'])
            lines = list(raw)
            for p in r['pseudo']:
                x1, y1, x2, y2 = p['box']
                lines.append(f'0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}')
                rows.append(dict(image=e['local_img'], pod_image=f'{pod_root}/images/{name}', source=e['src'],
                                 x1n=f'{x1:.6f}', y1n=f'{y1:.6f}', x2n=f'{x2:.6f}', y2n=f'{y2:.6f}', rule=p['rule'],
                                 teachers=';'.join(p['teachers']), scores_raw=';'.join(f"{p['scores'][t]:.4f}" for t in p['teachers']),
                                 scores_cal=';'.join(f"{p['cal'][t]:.4f}" for t in p['teachers']),
                                 fused_score='' if p['fused_score'] is None else f"{p['fused_score']:.4f}",
                                 n_teachers=len(p['teachers']), n_gt_boxes=r['n_gt'], label_type='model-assisted'))
                combo['+'.join(p['teachers'])] += 1; by_rule[p['rule']] += 1; by_rule_src[e['src']][p['rule']] += 1
                if p['rule'] == 'fused':
                    fused_nt[str(len(p['teachers']))] += 1
                elif p['rule'] == 'single':
                    single_t[p['teachers'][0]] += 1
            lbl = D2_PSEUDO / 'labels' / (Path(name).stem + '.txt')
            lbl.write_text('\n'.join(lines) + '\n')
            os.symlink(lbl, cdir / 'labels' / lbl.name)
            if e['comp'] in ('fireman', 'sheffield'):
                os.symlink(f"../../B_v1/images/{e['name']}", cdir / 'images' / name)     # relative: resolves locally AND on the pod
            else:
                os.symlink(e['local_img'], cdir / 'images' / name)                      # absolute local: dereferenced into the tar
            assert os.path.exists(cdir / 'images' / name), name
            e['corpus_img'] = str(cdir / 'images' / name); e['pod_img'] = f'{pod_root}/images/{name}'
            e['origin'] = 'author+model-assisted'; e['mult_override'] = 1
            by_src_img[e['src']] += 1; by_src_box[e['src']] += len(r['pseudo']); on_gt_empty[e['src']] += (r['n_gt'] == 0); by_comp_img[e['comp']] += 1
        with open(D2_PSEUDO / 'pseudo_boxes.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        rule_txt = (f"(a) FUSED: calibrated WBF of the 3 teachers (recipe {D2_RECIPE.relative_to(ROOT)}: conf_type={analysis['params']['wbf']['conf_type']}, "
                    f"iou_thr={analysis['params']['wbf']['iou_thr']}), fused calibrated score >= {thr['fused']['calibrated_threshold']} "
                    f"(accurate_recipe thresholds_oof fp{D2_FUSED_FP:.2f}); (b) SINGLE: any one teacher's box with calibrated score >= {D2_SINGLE_CAL} "
                    f"(= 2% neg-image FP on sdis VAL negatives; raw thresholds {json.dumps({t: round(thr[t]['raw_at_cal0.98_single_rule'], 4) for t in TEACHERS})}); "
                    + (f"; (c) AGREE (arm D rule, so D2 is a superset of D): >=2 of 3 teachers agree IoU>={AGREE_IOU}, each calibrated >= 0.80 (20% neg-FP), raw-score-weighted mean box" if analysis['params'].get('D2_INCLUDE_AGREE_RULE') else '')
                    + f"; candidates fused > single > agree, dedup IoU>={D_DEDUP_IOU}, dropped if any GT box IoU>={D_GT_IOU_MAX} or if >={int(100 * D2_GT_COVER_MAX)}% of the candidate lies inside a GT box / the candidate covers >={int(100 * D2_GT_COVER_MAX)}% of a GT box (guard vs FireMan large region GT; not in D). Explicit negatives "
                    f"(FireMan/Sheffield empty author labels, Boreal verified-empty) and all images without a qualifying box are UNCHANGED.")
        (D2_PSEUDO / 'README.md').write_text(
            '# D2_pseudo -- MODEL-ASSISTED pseudo boxes for arm D2 (never author labels, never human-reviewed)\n'
            f'Generated {utcnow()} by scripts/v5b/build_corpus_v5b.py from models/v5b/teacher/*__train.jsonl + *__train_new.jsonl.\n'
            f'Rule: {rule_txt}\nLabel files = GT lines first, then pseudo lines. Sidecar: pseudo_boxes.csv (rule, teachers, raw+calibrated scores).\n')
        pseudo_info = dict(rule=rule_txt, images_with_pseudo=sum(by_src_img.values()), pseudo_boxes=len(rows), images_by_source=dict(by_src_img),
                           images_by_component=dict(by_comp_img), boxes_by_source=dict(by_src_box), boxes_by_rule=dict(by_rule),
                           boxes_by_rule_and_source={s: dict(c) for s, c in by_rule_src.items()}, fused_boxes_by_n_teachers=dict(fused_nt),
                           single_boxes_by_teacher=dict(single_t), boxes_by_teacher_combo=dict(combo), images_gt_empty_by_source=dict(on_gt_empty),
                           explicit_negatives_unchanged=dict(images=st.get('explicit_neg_images', 0),
                                                             teacher_fused_fire=st.get('explicit_neg_images_fused_fire', 0),
                                                             teacher_single_fire_2pct=st.get('explicit_neg_images_single_fire_2pct', 0),
                                                             by_source={k.split('__', 1)[1]: v for k, v in st.items() if k.startswith('explicit_neg_images_single_fire_2pct__')}),
                           disagreement_images_unchanged=st.get('disagreement_images_fire10pct_noGT_unchanged', 0),
                           disagreement_images_unchanged_by_source={k.split('__', 1)[1]: v for k, v in st.items() if k.startswith('disagreement_images_fire10pct_noGT_unchanged__')},
                           missing_teacher_rows={k.split('__', 1)[1]: v for k, v in st.items() if k.startswith('missing_or_error__')},
                           skipped={k: v for k, v in st.items() if k.endswith('_near_gt') or k.endswith('_dedup') or k.endswith('_gt')}, analysis_stats=st,
                           labels_dir=str(D2_PSEUDO / 'labels'), sidecar_csv=str(D2_PSEUDO / 'pseudo_boxes.csv'), recipe=str(D2_RECIPE),
                           weighting_approximation='images that received pseudo boxes appear x1 in the list (Boreal/Sheffield normally x2); GT-only images keep B multiplicities; '
                                                   'explicit negatives and teacher-disagreement images kept unchanged (same files as B)')

    # --- D3: append pseudo-boxed UNLABELLED images x1 (labels = pseudo lines only; images = relative symlinks into ../../D3_src/images)
    d3_info = None
    if args.arm == 'D3':
        res = analysis['results']; st = analysis['stats']; thr = analysis['thresholds']
        if D3_PSEUDO.exists():
            shutil.rmtree(D3_PSEUDO)
        (D3_PSEUDO / 'labels').mkdir(parents=True)
        srcs = {str(D3_SRC / 'images' / r['name']): r for r in csv.DictReader(open(D3_SRC / 'sources.csv'))}
        rows = []; areas = defaultdict(list); cal_sc = defaultdict(list); by_rule_src = defaultdict(Counter); clip = defaultdict(lambda: Counter())
        n_img = Counter(); n_box = Counter(); excluded = Counter(); combo = Counter()
        for p, r in res.items():
            meta = srcs[p]; src = r['src']
            if r['pseudo']:
                clip[(src, r['unit'])]['frames_with_boxes'] += 1
            clip[(src, r['unit'])]['frames'] += 1; clip[(src, r['unit'])]['video_label'] = r['video_label']
            if not r['train_eligible']:
                excluded[f'{src}__not_train_eligible'] += 1; continue
            if not r['pseudo']:
                excluded[f'{src}__no_pseudo_box'] += 1; continue
            name = meta['name']
            lines = []
            for c in r['pseudo']:
                x1, y1, x2, y2 = c['box']
                lines.append(f'0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}')
                area = (x2 - x1) * (y2 - y1); areas[src].append(area)
                cal_sc[src].append(max(c['cal'].values()))
                rows.append(dict(image=p, local_src=meta['local_src'], pod_image=f'{pod_root}/images/{name}', source=src, unit=r['unit'], video_label=r['video_label'],
                                 video_level_support=str(r['video_level_support']).lower(), x1n=f'{x1:.6f}', y1n=f'{y1:.6f}', x2n=f'{x2:.6f}', y2n=f'{y2:.6f}', area_n=f'{area:.6f}',
                                 rule=c['rule'], teachers=';'.join(c['teachers']), scores_raw=';'.join(f"{c['scores'][t]:.4f}" for t in c['teachers']),
                                 scores_cal=';'.join(f"{c['cal'][t]:.4f}" for t in c['teachers']), fused_score='' if c['fused_score'] is None else f"{c['fused_score']:.4f}",
                                 n_teachers=len(c['teachers']), label_type='model-assisted'))
                by_rule_src[src][c['rule']] += 1; combo['+'.join(c['teachers'])] += 1
            lbl = D3_PSEUDO / 'labels' / (Path(name).stem + '.txt'); lbl.write_text('\n'.join(lines) + '\n')
            os.symlink(lbl, cdir / 'labels' / lbl.name)
            os.symlink(f'../../D3_src/images/{name}', cdir / 'images' / name)
            assert os.path.exists(cdir / 'images' / name), name
            ents.append(dict(comp='d3', src=src, local_img=str(cdir / 'images' / name), local_lbl=str(lbl), pod_img=f'{pod_root}/images/{name}', origin='model-assisted',
                             unit=r['unit'], name=name, corpus_img=str(cdir / 'images' / name), mult_override=1))
            n_img[src] += 1; n_box[src] += len(r['pseudo'])
        with open(D3_PSEUDO / 'pseudo_boxes.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        clip_stats = {}
        for (src, unit), c in clip.items():
            clip_stats.setdefault(src, {})[unit] = dict(frames=c['frames'], frames_with_boxes=c['frames_with_boxes'], video_label=c['video_label'])
        clip_summary = {}
        for src, d in clip_stats.items():
            by_label = defaultdict(lambda: Counter())
            for u, v in d.items():
                by_label[v['video_label']]['clips'] += 1; by_label[v['video_label']]['frames'] += v['frames']; by_label[v['video_label']]['frames_with_boxes'] += v['frames_with_boxes']
                by_label[v['video_label']]['clips_with_any_box'] += bool(v['frames_with_boxes'])
            clip_summary[src] = {k: dict(v) for k, v in by_label.items()}
        rule_txt = (f"D2 rules without GT filter (no author boxes exist): (a) FUSED calibrated WBF >= {thr['fused']['calibrated_threshold']} (accurate_recipe fp0.10); "
                    f"(b) SINGLE teacher calibrated >= {D2_SINGLE_CAL} (2% neg-FP; raw {json.dumps({t: round(thr[t]['raw_at_cal0.98_single_rule'], 4) for t in TEACHERS})}); "
                    f"(c) AGREE >=2 teachers IoU>={AGREE_IOU} each calibrated >= 0.80; fused > single > agree, dedup IoU>={D_DEDUP_IOU}. Image kept ONLY if >=1 pseudo box "
                    "(no teacher-silence negatives from unlabelled footage); the karkkila no_smoke clip is scored for the FP statistic but never trained; D-Fire fire-only images without a box are excluded.")
        (D3_PSEUDO / 'README.md').write_text('# D3_pseudo -- MODEL-ASSISTED boxes on UNLABELLED smoke-positive material (never author labels, never human-reviewed)\n'
                                             f'Generated {utcnow()} by scripts/v5b/build_corpus_v5b.py from models/v5b/teacher/*__train_d3.jsonl (sources: data/v5b/yolo/D3_src/sources.csv).\n'
                                             f'Rule: {rule_txt}\nLabel files = pseudo lines only. Sidecar: pseudo_boxes.csv (label_type, video_level_support, rule, teachers, scores, area).\n')
        d3_info = dict(rule=rule_txt, images_scored_by_source={s: st.get(f'{s}__images', 0) for s in ('borealclip', 'firesense', 'dfirefire')},
                       images_with_pseudo_by_source=dict(n_img), boxes_by_source=dict(n_box), boxes_by_rule_and_source={s: dict(c) for s, c in by_rule_src.items()},
                       boxes_by_teacher_combo=dict(combo), excluded=dict(excluded),
                       median_box_area_n_by_source={s: float(np.median(v)) for s, v in areas.items()},
                       box_area_n_quantiles_by_source={s: [float(x) for x in np.quantile(v, [0.1, 0.5, 0.9])] for s, v in areas.items()},
                       boxes_per_kept_image_by_source={s: round(n_box[s] / n_img[s], 3) for s in n_img},
                       median_max_calibrated_score_by_source={s: float(np.median(v)) for s, v in cal_sc.items()},
                       clip_level=clip_summary, clip_level_detail=clip_stats,
                       video_level_support={'true': sum(1 for r in rows if r['video_level_support'] == 'true'), 'false': sum(1 for r in rows if r['video_level_support'] == 'false')},
                       teacher_thresholds=thr, pool=analysis['pool'], labels_dir=str(D3_PSEUDO / 'labels'), sidecar_csv=str(D3_PSEUDO / 'pseudo_boxes.csv'), recipe=analysis['recipe'],
                       sources_manifest=str(ROOT / 'data/v5b/teacher/lists/TARGETS_D3.json'),
                       weighting='D3 images x1; all B components keep B multiplicities and B files (E baked, Boreal E_ET_v1, FireMan/Sheffield at the B_v1 pod root)')

    # --- lists with multiplicities
    for e in ents:
        e.setdefault('corpus_img', e['local_img'])
        e['mult'] = e.get('mult_override', mult[e['comp']])
    pairs = []
    for e in ents:
        pairs += [(e['corpus_img'], e['pod_img'])] * e['mult']
    random.Random(SEED).shuffle(pairs)
    lists = {}
    for rel, idx in (('lists/local/train.txt', 0), ('lists/pod/train.txt', 1)):
        p = cdir / rel
        p.write_text('\n'.join(x[idx] for x in pairs) + '\n')
        lists[rel] = {'n': len(pairs), 'sha256': sha256_file(p)}
    for comp in mult:
        sub = [e for e in ents if e['comp'] == comp]
        for rel, key in ((f'lists/local/train_{comp}.txt', 'corpus_img'), (f'lists/pod/train_{comp}.txt', 'pod_img')):
            p = cdir / rel
            p.write_text('\n'.join(e[key] for e in sub) + '\n')
            lists[rel] = {'n': len(sub), 'sha256': sha256_file(p)}
    miss = [l for l in {x[0] for x in pairs} if not os.path.exists(l)]
    assert not miss, miss[:3]
    # --- yamls
    local_yaml = (f'# {corpus} (iteration 5b, arm {args.arm}, local) -- generated {utcnow()}\npath: {cdir}\n'
                  f"train: {cdir / 'lists/local/train.txt'}\nval: {LOCAL_SDIS_VAL}\ntest: {LOCAL_SDIS_TEST}\nnc: 1\nnames: [smoke]\n")
    pod_yaml = (f'# {corpus} (iteration 5b, arm {args.arm}, pod) -- generated {utcnow()}\npath: {pod_root}\n'
                f'train: {pod_root}/lists/pod/train.txt\nval: {POD_SDIS_VAL}\ntest: {POD_SDIS_TEST}\nnc: 1\nnames: [smoke]\n')
    (cdir / 'yamls/local.yaml').write_text(local_yaml); (cdir / 'yamls/pod.yaml').write_text(pod_yaml)
    for kind, text in (('local', local_yaml), ('pod', pod_yaml)):
        d = OUT_YOLO / 'yamls' / kind; d.mkdir(parents=True, exist_ok=True)
        (d / f'{corpus}.yaml').write_text(text)

    # --- budget
    n_lines = len(pairs)
    epochs = max(MIN_EPOCHS, int(round(ARM_A_EXPOSURES / n_lines)))
    budget = dict(arm_A=dict(list='expB.yaml / expB_train.txt', n_lines=ARM_A_LINES, epochs=ARM_A_EPOCHS, exposures=ARM_A_EXPOSURES),
                  n_lines=n_lines, epochs=epochs, exposures=n_lines * epochs,
                  delta_vs_A_pct=round(100 * (n_lines * epochs / ARM_A_EXPOSURES - 1), 3))

    # --- stage tar (dereferenced; only NEW files: fireman/sheffield images+labels, masked copies, pseudo labels, lists, yamls)
    tar_info = None
    if not args.no_tar:
        OUT_STAGE.mkdir(parents=True, exist_ok=True)
        tar_path = OUT_STAGE / f'{corpus}.tar'
        log(f'writing {tar_path} ' + ('(tarfile: labels/lists/yamls + E/Boreal pseudo images dereferenced, new-source pseudo images as relative symlinks)' if args.arm in ('D2', 'D3') else '(tar -ch, dereferenced)') + ' ...')
        links, b_names = {}, set()
        if args.arm in ('D2', 'D3'):
            names, links = write_d2_tar(tar_path, cdir)
            if not B_TAR_MEMBERS.exists():
                log('listing B_v1.tar members (once) ...')
                B_TAR_MEMBERS.write_text(subprocess.run(['tar', '-tf', str(OUT_STAGE / 'B_v1.tar')], check=True, capture_output=True, text=True).stdout)
            b_names = set(read_lines(B_TAR_MEMBERS))
            if args.arm == 'D3':
                b_names |= set(read_lines(D3_TAR_MEMBERS))   # D3_src.tar members (staged by the D3 teacher job and re-passed to the training job)
            bad_links = {k: v for k, v in links.items() if v not in b_names}
            assert not bad_links, list(bad_links.items())[:3]
            log(f'{args.arm} tar: {len(names)} members, {len(links)} relative symlinks (all resolve to B_v1.tar / D3_src.tar members)')
        else:
            subprocess.run(['tar', '-chf', str(tar_path), '-C', str(OUT_YOLO), corpus], check=True)
            names = set(subprocess.run(['tar', '-tf', str(tar_path)], check=True, capture_output=True, text=True).stdout.split('\n'))
        tar_info = {'path': str(tar_path), 'bytes': tar_path.stat().st_size, 'sha256': sha256_file(tar_path), 'n_members': len(names) - 1,
                    'kraken_data_tar_arg': f'v5b_{args.arm.lower()}={tar_path.relative_to(ROOT)}', 'pod_extract_root': POD_V5B_PREFIX}
        # verify 20 random pod list lines resolve pod-side
        rnd = random.Random(SEED + 1)
        pod_lines = [x[1] for x in pairs]
        t_pod_set = set(read_lines(ROOT / 'data/v4/yolo/E_ET_v1/lists/pod/train_T.txt'))
        checks = []
        for pl in rnd.sample(pod_lines, 20):
            if pl.startswith(POD_E_ROOT + '/'):
                ok = os.path.exists(pl.replace(POD_E_ROOT, LOCAL_E_ROOT)); how = 'baked /mnt/local/smoke_data (local mirror exists)'
            elif pl.startswith(POD_T_ROOT + '/'):
                ok = pl in t_pod_set and os.path.exists(pl.replace(POD_T_ROOT, LOCAL_T_ROOT)); how = 'staged v4data/E_ET_v1 (in E_ET_v1 pod list + local exists)'
            elif pl.startswith(pod_root + '/'):
                rel = pl[len(POD_V5B_PREFIX) + 1:]
                lbl = rel.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
                ok = rel in names and lbl in names; how = 'tar member (image + label)'
                if rel in links:
                    ok = ok and links[rel] in b_names; how = f'tar member label + relative symlink -> {links[rel]} (B_v1.tar member, staged by arm B)'
            elif args.arm in ('D2', 'D3') and pl.startswith(POD_B_ROOT + '/'):
                rel = pl[len(POD_V5B_PREFIX) + 1:]
                lbl = rel.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
                ok = rel in b_names and lbl in b_names; how = 'B_v1.tar member (image + label), staged on /scratch by arm B job'
            else:
                ok, how = False, 'unknown root'
            checks.append(dict(line=pl, ok=bool(ok), how=how))
        assert all(c['ok'] for c in checks), [c for c in checks if not c['ok']]
        tar_info['pod_resolution_check'] = dict(n=20, all_ok=True, checks=checks)
        if args.arm == 'D3':
            tman = json.loads((ROOT / 'data/v5b/teacher/lists/TARGETS_D3.json').read_text())
            tar_info['n_relative_symlinks_into_D3_src'] = len(links)
            tar_info['requires_staged'] = dict(B_v1=dict(pod_root=POD_B_ROOT, **{k: json.loads((OUT_MAN / 'CORPUS_B_v1.json').read_text())['tar'][k] for k in ('sha256', 'bytes', 'n_members')}),
                                               D3_src=dict(pod_root=POD_D3_SRC_ROOT, **{k: tman['tar'][k] for k in ('path', 'sha256', 'bytes', 'n_members', 'kraken_data_tar_arg')},
                                                           note='staged by the D3 teacher jobs (DATA_TARS v5b_d3src); pass it again to the training job: --data-tar v5b_d3src=... (skipped via .done marker if present)'))
            tar_info['kraken_data_tar_args'] = ['v5b_d3src=data/v5b/stage/D3_src.tar', 'v5b_d3=data/v5b/stage/D3_v1.tar']
        if args.arm == 'D2':
            tar_info['n_relative_symlinks_into_B_v1'] = len(links)
            tar_info['requires_staged'] = dict(B_v1=dict(pod_root=POD_B_ROOT, tar=str(OUT_STAGE / 'B_v1.tar'),
                                                         **{k: json.loads((OUT_MAN / 'CORPUS_B_v1.json').read_text())['tar'][k] for k in ('sha256', 'bytes', 'n_members')},
                                                         staged_by='arm B job (DATA_TARS v5b_b -> V4_DATA_ROOT, .v5b_b.done marker)'))
        log('20/20 random pod lines resolve')

    # --- frozen manifest
    per_comp = Counter(e['comp'] for e in ents); per_src = Counter(e['src'] for e in ents); per_origin = Counter(e['origin'] for e in ents)
    eff = Counter()
    for e in ents:
        eff[e['comp']] += e['mult']
    n_gt_boxes = Counter()
    for e in ents:
        if e['comp'] != 'd3':
            n_gt_boxes[e['src']] += len(parse_yolo(e['local_lbl'])[1])
    man = dict(
        arm=args.arm, version='v1', corpus=corpus, created_utc=utcnow(), seed=SEED, iteration='5b',
        composition=dict(E='data/yolo/lists/local/expB_train.txt (pyro-sdis train sites + D-Fire; pod /mnt/local/smoke_data)',
                         boreal='data/v5b/teacher/lists/train_T_boreal.txt = E_ET_v1 T list minus AuSmoke (karkkila+evo train burns + 37 verified-empty; pod v4data/E_ET_v1)',
                         fireman=f'FireMan-UAV-RGBT RGB TRAIN split only (burn A), per-video cap {FIREMAN_CAP} frames uniform stride, class 0 smoke, empty txt = author explicit negative',
                         sheffield='Sheffield UAV all exported frames (train+val+holdout exports; ONE burn session -> TRAIN only, never val/holdout); 3 fire-only + 6 flagged empties already excluded at export'),
        excluded=['FireSpot (licence pending)', 'WSDataset (blocked)', 'AuSmoke/MNS (permission pending; 10,904 T lines)', 'FIgLib (eval-only)',
                  'Boreal heinola (aerial VAL) / ruokolahti (historical test)', 'FireMan val/holdout/quarantine + 9 CVAT conflicts', 'Sheffield 3 fire-only + 6 smoke-visible empties',
                  'D-Fire fire-only (944; never in E)'],
        counts=dict(per_component=dict(per_comp), per_source=dict(per_src), per_label_origin=dict(per_origin),
                    n_train_lines=n_lines, n_gt_boxes_by_source=dict(n_gt_boxes),
                    fireman_videos=fireman_sel, fireman_explicit_negatives=n_new_empty.get('fireman', 0), sheffield_explicit_negatives=n_new_empty.get('sheffield', 0),
                    new_source_effective_lines=eff['boreal'] + eff['fireman'] + eff['sheffield'],
                    new_source_effective_over_E_pct=round(100 * (eff['boreal'] + eff['fireman'] + eff['sheffield']) / eff['E'], 1)),
        multiplicities={c: dict(n_raw=per_comp[c], multiplicity=mult[c], n_effective=eff[c]) for c in mult},
        multiplicity_note=('D: images that received pseudo boxes are x1 (override), see pseudo.weighting_approximation' if args.arm in ('D', 'D2') else
                           'D3: corpus B unchanged (B multiplicities) + pseudo-boxed unlabelled images x1' if args.arm == 'D3' else None),
        budget=budget, lists=lists, yamls=dict(local=str(cdir / 'yamls/local.yaml'), pod=str(cdir / 'yamls/pod.yaml'),
                                              pod_copy=str(OUT_YOLO / 'yamls/pod' / f'{corpus}.yaml')),
        val=dict(local=str(LOCAL_SDIS_VAL), pod=POD_SDIS_VAL, note='ultralytics model selection on sdis val, as arm A / v4'),
        pod_root=pod_root, tar=tar_info,
        teacher_thresholds=analysis['thresholds'] if analysis else None, analysis_params=analysis['params'] if analysis else None,
        stage_reuse=(dict(note='D2 re-uses the arm-B stage: unchanged FireMan/Sheffield images+labels are listed at the B_v1 pod root; D2_v1.tar holds only '
                               'pseudo-labelled copies/symlinks + labels/lists/yamls', b_v1_pod_root=POD_B_ROOT, teacher_new_jsonl=[str(TEACHER_DIR / f'{t}__train_new.jsonl') for t in TEACHERS],
                          d2_recipe=str(D2_RECIPE)) if args.arm == 'D2' else
                     dict(note='D3 re-uses the arm-B stage for unchanged FireMan/Sheffield (B_v1 pod root) and the D3_src stage (teacher job) for the pseudo-boxed images; '
                               'D3_v1.tar holds labels/lists/yamls + relative symlinks', b_v1_pod_root=POD_B_ROOT, d3_src_pod_root=POD_D3_SRC_ROOT,
                          teacher_d3_jsonl=[str(TEACHER_DIR / f'{t}__train_d3.jsonl') for t in TEACHERS], d2_recipe=str(D2_RECIPE)) if args.arm == 'D3' else None),
        masked=masked_info, pseudo=pseudo_info, d3=d3_info,
        label_type_note='author = dataset annotations (not our human review); masked = author labels unchanged + pixels of teacher-agreed unlabelled regions replaced by image mean; model-assisted = teacher pseudo boxes',
        builder='scripts/v5b/build_corpus_v5b.py')
    OUT_MAN.mkdir(parents=True, exist_ok=True)
    mp = OUT_MAN / f'CORPUS_{corpus}.json'
    mp.write_text(json.dumps(man, indent=1, default=str))
    log(f'FROZEN manifest {mp}; n_lines={n_lines} epochs={epochs} exposures={n_lines * epochs} ({budget["delta_vs_A_pct"]:+.2f}% vs A)')
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('analyze'); sub.add_parser('analyze-d2'); sub.add_parser('analyze-d3')
    b = sub.add_parser('build'); b.add_argument('arm', choices=['B', 'C', 'D', 'D2', 'D3']); b.add_argument('--no-tar', action='store_true')
    args = ap.parse_args()
    return {'analyze': cmd_analyze, 'analyze-d2': cmd_analyze_d2, 'analyze-d3': cmd_analyze_d3}.get(args.cmd, cmd_build)(args)


if __name__ == '__main__':
    raise SystemExit(main())
