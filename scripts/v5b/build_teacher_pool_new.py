#!/usr/bin/env python
"""V5b arm D2: merge raw teacher shards for the NEW B-corpus sources (FireMan burn-A project-TRAIN capped 600/video +
Sheffield single-session exports; list data/v5b/teacher/lists/train_new_sources.txt) into
models/v5b/teacher/<teacher>__train_new.jsonl (same schema as __train.jsonl; "image" = LOCAL B_v1 symlink path exactly
as listed for inference) + missing__<teacher>_new.txt; appends a section to TEACHER_POOL.md and a "new_sources" block to
teacher_pool_stats.json.  Every row is a MODEL-ASSISTED label (teacher prediction, conf floor 0.02): not author labels,
not human-reviewed; the statistics are automated inspection, not human review."""
import collections, gzip, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path('/local/home/tianrgua/cvproject')
RAW, OUT, LISTS = ROOT / 'models/v5b/teacher/raw', ROOT / 'models/v5b/teacher', ROOT / 'data/v5b/teacher/lists'
TEACHERS = {'dino_b4_lora_L': 'b4L_new', 'dino_b6_lora_Hplus': 'b6H_new', 'rfdetr_L': 'rfdetrL_new'}
THRS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]


def source(p):
    return 'fireman' if '/fireman__' in p else 'sheffield'


def author_boxes(p):
    L = Path(p.replace('/images/', '/labels/')).with_suffix('.txt')
    if not L.exists():
        return None
    out = []
    for r in L.read_text().splitlines():
        f = r.split()
        if len(f) >= 5:
            cx, cy, w, h = map(float, f[1:5]); out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return out


def iou_mat(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a)[:, :4], np.asarray(b)[:, :4]
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def main():
    now = datetime.now(timezone.utc).strftime('%FT%TZ')
    targets = [l.strip() for l in (LISTS / 'train_new_sources.txt').read_text().splitlines() if l.strip()]
    tset = set(targets); tman = json.loads((LISTS / 'TARGETS_NEW.json').read_text())
    src_of = {p: source(p) for p in targets}
    auth = {p: author_boxes(p) for p in targets}
    assert all(v is not None for v in auth.values()), 'label file missing for a target'
    n_by_src = collections.Counter(src_of.values())
    apos = {s: sum(1 for p in targets if src_of[p] == s and auth[p]) for s in n_by_src}
    aneg = {s: sum(1 for p in targets if src_of[p] == s and not auth[p]) for s in n_by_src}
    stats = {'generated_utc': now, 'label_type': 'model-assisted (teacher predictions); NOT human-reviewed; NOT author labels',
             'targets': tman['counts'], 'list_sha256': tman['sha256'], 'rootmap': tman['rootmap_local_to_pod'],
             'author_reference': {'positive_images': apos, 'explicit_negative_images': aneg,
                                  'author_boxes': {s: sum(len(auth[p]) for p in targets if src_of[p] == s) for s in n_by_src}},
             'teachers': {}}
    md = [f'\n## NEW-source extension for arm D2 -- generated {now}\n',
          '**Model-assisted labels, same caveats as above.** Targets: the exact B-corpus TRAIN lists for the two new sources '
          f'(`data/v5b/teacher/lists/train_new_sources.txt`, sha256 {tman["sha256"][:16]}; `TARGETS_NEW.json` leak check vs sdis val/test, '
          f'D-Fire test, FireMan project val/holdout: 0 overlaps): FireMan burn-A project-TRAIN capped 600/video = {n_by_src["fireman"]:,} '
          f'(author positives {apos["fireman"]:,} / explicit negatives {aneg["fireman"]:,}), Sheffield single session = {n_by_src["sheffield"]:,} '
          f'(positives {apos["sheffield"]:,} / explicit negatives {aneg["sheffield"]:,}). Images read from the arm-B stage on /scratch '
          f'(`{tman["rootmap_local_to_pod"]}`). Outputs: `models/v5b/teacher/<teacher>__train_new.jsonl` (image = local B_v1 path as listed).\n',
          '| teacher | ckpt sha256 (pod-verified) | kraken job (wall) | rows / targets | unreadable | missing | fireman firing @0.5 / @0.9 (pos imgs) | '
          'fireman explicit-neg firing @0.5 / @0.9 | sheffield firing @0.5 / @0.9 (pos imgs) | sheffield explicit-neg firing @0.5 / @0.9 | author boxes matched IoU>=0.4 @0.5 (fireman / sheffield) |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    boxes_at, maxs = {}, {}
    for t, ev in TEACHERS.items():
        d = RAW / ev
        rows, files = {}, []
        for f in sorted(d.glob('*_preds.jsonl.gz')):
            n = 0
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line); rows[r['image']] = r; n += 1
            files.append((str(f.relative_to(ROOT)), n))
        extra = [p for p in rows if p not in tset]
        assert not extra, f'{t}: {len(extra)} rows for NON-target images, e.g. {extra[:2]} -- refusing'
        missing = [p for p in targets if p not in rows]
        errs = [p for p in targets if p in rows and 'error' in rows[p]]
        job = json.loads((d / 'job.json').read_text())
        summ = [json.loads(l) for l in (d / 'summary.jsonl').read_text().splitlines() if l.strip()]
        sha = summ[0]['ckpt_sha256'] if summ else None
        with (OUT / f'{t}__train_new.jsonl').open('w') as fh:
            for p in targets:
                if p not in rows:
                    continue
                r = rows[p]
                rec = {'image': p, 'teacher': t, 'label_type': 'model-assisted'}
                if 'error' in r:
                    rec['error'] = r['error']
                else:
                    rec.update(w=r.get('w'), h=r.get('h'), preds=r.get('preds', []))
                fh.write(json.dumps(rec) + '\n')
        (OUT / f'missing__{t}_new.txt').write_text('\n'.join(missing) + ('\n' if missing else ''))
        ok = [p for p in targets if p in rows and 'error' not in rows[p]]
        ms = {p: max((b[4] for b in rows[p]['preds']), default=0.0) for p in ok}
        maxs[t] = ms; boxes_at[t] = {p: rows[p]['preds'] for p in ok}
        nbox = np.array([len(rows[p]['preds']) for p in ok]) if ok else np.zeros(0)
        allsc = np.array([b[4] for p in ok for b in rows[p]['preds']])
        fire = {str(th): {s: {'pos': sum(1 for p in ok if src_of[p] == s and auth[p] and ms[p] >= th),
                              'explicit_neg': sum(1 for p in ok if src_of[p] == s and not auth[p] and ms[p] >= th)} for s in n_by_src} for th in THRS}
        matched = {}
        for s in n_by_src:
            na = nb = 0
            for p in ok:
                if src_of[p] != s or not auth[p]:
                    continue
                tb = [b for b in rows[p]['preds'] if b[4] >= 0.5]
                m = iou_mat(auth[p], tb)
                na += len(auth[p]); nb += int((m.max(axis=1) >= 0.4).sum()) if m.size else 0
            matched[s] = {'author_boxes': na, 'matched_by_teacher_at_0.5': nb, 'frac': round(nb / na, 4) if na else None}
        st = {'eval_name': ev, 'job': job.get('job'), 'final_state': job.get('final_state'), 'wall_s': job.get('wall_s'), 'ckpt_pod': job.get('ckpt'),
              'ckpt_sha256': sha, 'expect_sha': job.get('expect_sha'), 'raw_files': files, 'rows': len(rows), 'targets': len(targets),
              'unreadable': len(errs), 'missing': len(missing), 'coverage_by_source': {s: sum(1 for p in ok if src_of[p] == s) for s in n_by_src},
              'boxes': {'total_ge_floor': int(nbox.sum()), 'per_image_mean': float(nbox.mean()) if len(nbox) else None,
                        'score_quantiles_p10_p50_p90_p99': [float(np.quantile(allsc, q)) for q in (0.1, 0.5, 0.9, 0.99)] if len(allsc) else None},
              'max_score_hist': {f'[{a},{b})': int(sum(1 for p in ok if a <= ms[p] < b)) for a, b in zip([0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99], [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0001])},
              'images_no_box': int(sum(1 for p in ok if ms[p] == 0.0)), 'images_firing_at_thr': fire, 'author_boxes_matched': matched}
        stats['teachers'][t] = st
        f5, f9 = fire['0.5'], fire['0.9']
        md.append(f'| {t} | {sha[:16] if sha else "?"} (expect {job.get("expect_sha")}) | {job.get("job")} {job.get("final_state")} {round((job.get("wall_s") or 0) / 60)} min | '
                  f'{len(rows):,} / {len(targets):,} | {len(errs)} | {len(missing)} | '
                  f'{f5["fireman"]["pos"]:,} / {f9["fireman"]["pos"]:,} (of {apos["fireman"]:,}) | {f5["fireman"]["explicit_neg"]:,} / {f9["fireman"]["explicit_neg"]:,} (of {aneg["fireman"]:,}) | '
                  f'{f5["sheffield"]["pos"]:,} / {f9["sheffield"]["pos"]:,} (of {apos["sheffield"]:,}) | {f5["sheffield"]["explicit_neg"]:,} / {f9["sheffield"]["explicit_neg"]:,} (of {aneg["sheffield"]:,}) | '
                  f'{matched["fireman"]["frac"]} / {matched["sheffield"]["frac"]} |')
    # pairwise image-level agreement at 0.5 on images scored by both
    agree = {}
    tl = list(TEACHERS)
    for i in range(len(tl)):
        for j in range(i + 1, len(tl)):
            a, b = tl[i], tl[j]
            common = [p for p in targets if p in maxs[a] and p in maxs[b]]
            for th in (0.2, 0.5, 0.9):
                fa = np.array([maxs[a][p] >= th for p in common]); fb = np.array([maxs[b][p] >= th for p in common])
                po = float(np.mean(fa == fb)); pe = fa.mean() * fb.mean() + (1 - fa.mean()) * (1 - fb.mean())
                agree[f'{a}|{b}@{th}'] = {'n': len(common), 'both': int((fa & fb).sum()), f'only_{a}': int((fa & ~fb).sum()), f'only_{b}': int((~fa & fb).sum()),
                                          'neither': int((~fa & ~fb).sum()), 'kappa': round(float((po - pe) / (1 - pe)) if pe < 1 else 1.0, 4)}
    stats['agreement_image_level'] = agree
    md.append('\nImage-level agreement (max score >= thr on both; n = images scored by both): ' +
              '; '.join(f'{k}: both {v["both"]:,}, kappa {v["kappa"]}' for k, v in agree.items() if k.endswith('@0.5')) + '.')
    md.append('\nUsed by: `scripts/v5b/build_corpus_v5b.py analyze-d2` / `build D2` (calibrated 3-teacher WBF, models/v5b/fusion/teacher_ensemble_d2.json).\n')
    sp = OUT / 'teacher_pool_stats.json'
    allst = json.loads(sp.read_text()) if sp.exists() else {}
    allst['new_sources'] = stats
    sp.write_text(json.dumps(allst, indent=1))
    with (OUT / 'TEACHER_POOL.md').open('a') as fh:
        fh.write('\n'.join(md))
    print(json.dumps({t: {k: st[k] for k in ('job', 'final_state', 'rows', 'unreadable', 'missing', 'coverage_by_source', 'ckpt_sha256')} for t, st in stats['teachers'].items()}, indent=None))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
