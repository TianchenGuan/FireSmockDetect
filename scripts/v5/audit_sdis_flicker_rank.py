#!/usr/bin/env python
"""V5 Day-1 audit helper (B): rank pyro-sdis TRAIN sequences (data/v3/sequences/
sequences_train.json; train sites only) by label flicker = number of has-box/no-box
transitions between consecutive frames. Picks top 20 (ties: higher flip rate,
then seq_id). VAL/TEST sites are never read."""
import json, pandas as pd
ROOT = '/local/home/tianrgua/cvproject'
S = json.load(open(f'{ROOT}/data/v3/sequences/sequences_train.json'))
rows = []
for s in S:
    assert s['split'] == 'train'
    lab = [bool(f['smoke']) for f in s['frames']]
    flips = sum(a != b for a, b in zip(lab, lab[1:]))
    # pos->neg->pos "islands": negatives sandwiched between positives
    patt = ''.join('P' if x else 'n' for x in lab)
    neg_islands = sum(1 for i in range(1, len(lab) - 1) if not lab[i] and (lab[i-1] or lab[i+1]))
    rows.append(dict(seq_id=s['seq_id'], camera=s['camera'], site=s['site'], n_frames=s['n_frames'], duration_s=s['duration_s'],
                     label_pattern=s['label_pattern'], n_flips=flips, flip_rate=flips / max(1, len(lab) - 1),
                     n_pos=sum(lab), n_neg=len(lab) - sum(lab), pattern=patt))
df = pd.DataFrame(rows)
print('train sequences', len(df), 'with >=1 flip', (df.n_flips > 0).sum(), 'flicker-pattern', (df.label_pattern == 'flicker').sum())
top = df.sort_values(['n_flips', 'flip_rate', 'seq_id'], ascending=[False, False, True]).head(20)
top.to_csv(f'{ROOT}/data/v5/audit/sdis_flicker_top20.csv', index=False)
df.to_csv(f'{ROOT}/data/v5/audit/sdis_train_flicker_all.csv', index=False)
pd.set_option('display.width', 250)
print(top[['seq_id', 'site', 'n_frames', 'duration_s', 'n_flips', 'flip_rate', 'n_pos', 'n_neg']].to_string())
print(top.site.value_counts())
