#!/usr/bin/env python
"""V5 Day-1: data inventory BY PURPOSE (PROTOCOL_V5.md section 2).

CPU only. Reads existing manifests/artifacts, never modifies them. Writes
  data/v5/manifests/DATA_INVENTORY_V5.csv
  data/v5/manifests/DATA_INVENTORY_V5.md
Every number is computed from (or quoted with citation to) an on-disk artifact.
Lockbox / test IMAGES are never opened: only manifests and lists are read.
"""
import json, os, sys, datetime as dt
import numpy as np, pandas as pd

ROOT = '/local/home/tianrgua/cvproject'
os.chdir(ROOT)
OUT_CSV = 'data/v5/manifests/DATA_INVENTORY_V5.csv'
OUT_MD = 'data/v5/manifests/DATA_INVENTORY_V5.md'

ENUM_NEG = {'human-annotated', 'video-level label', 'author-verified empty', 'none'}
ENUM_ONSET = {'yes', 'partial', 'no'}
ENUM_SCEN = {'fixed-camera', 'aerial', 'forest-ground', 'indoor', 'web'}
ENUM_PURP = {'train', 'calibration/val', 'final-test', 'historical', 'quarantine',
             'permission-pending', 'aux-train-candidate'}
ENUM_PERM = {'confirmed', 'pending', 'none', 'blocked'}

rows = []
def row(source, unit_group, n_units, n_img, hours, pos, vneg, negev, onset, scen, purp,
        licence, perm, notes, artifacts, valid_h=0.0, neg_h=0.0, onset_events=0):
    assert negev in ENUM_NEG, negev
    assert onset in ENUM_ONSET, onset
    assert scen in ENUM_SCEN, scen
    assert purp in ENUM_PURP, purp
    assert perm in ENUM_PERM, perm
    rows.append(dict(source=source, unit_group=unit_group, n_units=n_units,
                     n_images_or_frames=n_img, hours_if_video=hours, positives=pos,
                     verified_negatives=vneg, negative_evidence_type=negev,
                     onset_truth=onset, scenario=scen, purpose_v5=purp, licence=licence,
                     permission_status=perm, notes=notes, artifacts=artifacts,
                     valid_hours_num=round(float(valid_h), 4),
                     neg_hours_num=round(float(neg_h), 4),
                     onset_events_num=int(onset_events)))

# ----------------------------------------------------------------------------
# 1. pyro-sdis (fixed-camera; Apache-2.0)
# ----------------------------------------------------------------------------
P_SDIS = 'data/manifests/pyro_sdis_manifest.parquet'
P_SPLITS = 'data/manifests/splits_frozen.json'
p = pd.read_parquet(P_SDIS)
splits = json.load(open(P_SPLITS))
site2split = {s: k for k, v in splits['site_split'].items() for s in v}
p['split'] = p.site.map(site2split)
p['pos'] = p.n_boxes > 0
p['datetime'] = pd.to_datetime(p.datetime, errors='coerce')

def cadence_stats(g):
    """valid observed time per PROTOCOL section 3: gaps > 3x median cadence excluded,
    cameras never concatenated. Returns (valid_all_h, valid_negneg_h, median_cadence_s, neg_runs)."""
    va = vn = 0.0; meds = []; runs = 0
    for cam, gg in g.groupby('camera_id'):
        gg = gg.dropna(subset=['datetime']).sort_values('datetime')
        t = gg.datetime.values.astype('datetime64[s]').astype(np.int64); pos = gg.pos.values
        if len(t) < 2: continue
        d = np.diff(t); med = float(np.median(d)); meds.append(med); ok = d <= 3 * med
        va += d[ok].sum(); vn += d[ok & (~pos[:-1]) & (~pos[1:])].sum()
        i = 0; n = len(t)
        while i < n:
            if not pos[i]:
                j = i
                while j + 1 < n and (not pos[j + 1]) and ok[j]: j += 1
                if j > i: runs += 1
                i = j + 1
            else: i += 1
    return va / 3600, vn / 3600, (float(np.median(meds)) if meds else float('nan')), runs

# v3 sequence onset patterns per site (pyro-sdis, 29 s cadence, flicker 29-36%)
V3_TR = 'data/v3/sequences/sequences_train.json'; V3_VA = 'data/v3/sequences/sequences_val.json'
v3 = json.load(open(V3_TR)) + json.load(open(V3_VA))
v3df = pd.DataFrame([{k: s[k] for k in ('seq_id', 'site', 'split', 'n_frames', 'duration_s', 'label_pattern')} for s in v3])
def v3_site(site):
    g = v3df[v3df.site == site]
    app = g[g.label_pattern.isin(['appears', 'appears_disappears'])]
    alln = g[g.label_pattern == 'all_neg']
    flick = g[g.label_pattern == 'flicker']
    return len(g), len(app), len(alln), alln.duration_s.sum() / 3600, len(flick)

purpose_map = {'train': 'train', 'val': 'calibration/val', 'test': 'final-test'}
sdis_site_stats = {}
for (split, site), g in p.groupby(['split', 'site']):
    va, vn, med, runs = cadence_stats(g)
    n = len(g); pos = int(g.pos.sum()); neg = n - pos; cams = g.camera_id.nunique()
    nseq, napp, nalln, alln_h, nflick = v3_site(site)
    sdis_site_stats[site] = dict(split=split, n=n, pos=pos, neg=neg, cams=cams, va=va, vn=vn, med=med, runs=runs,
                                 nseq=nseq, napp=napp, nalln=nalln, alln_h=alln_h, nflick=nflick)
    purp = purpose_map[split]
    extra = ''
    if split == 'val':
        extra = ' PROTOCOL s2: CALIBRATION + model selection; embedded negatives -> fixed-camera episodes/h.'
    if split == 'test':
        extra = ' PROTOCOL s2: FINAL TEST, used once per iteration (sdis_test list sha 88ea60e2..., images never opened here).'
    row('pyro-sdis', f'site:{site} ({split})', cams, n,
        f'{va:.2f} h valid all-frames; {vn:.2f} h valid neg-neg intervals; median cadence {med:.0f} s sampled',
        pos, neg, 'human-annotated', 'partial', 'fixed-camera', purp, 'Apache-2.0', 'confirmed',
        f'partner {g.partner.iloc[0]}; {cams} cameras; negatives = frames with no annotated box (NOT re-verified no-smoke; '
        f'v3 sequence label flicker 29% train / 36% val). {neg} neg frames in {runs} all-negative runs (>=2 frames, valid gaps). '
        f'v3 sequences: {nseq} total, {napp} appears-type (background->smoke within 29 s cadence, box-presence only, not human-reviewed intervals), '
        f'{nalln} all-negative seqs ({alln_h:.2f} h), {nflick} flicker.' + extra,
        f'{P_SDIS}; {P_SPLITS}; {V3_TR}; {V3_VA}', va, vn, napp)

# ----------------------------------------------------------------------------
# 2. D-Fire (web imagery; CC0)
# ----------------------------------------------------------------------------
P_DF = 'data/manifests/dfire_manifest.parquet'
d = pd.read_parquet(P_DF)
d['has_smoke'] = d.class_ids.apply(lambda c: 1 in list(c)); d['has_fire'] = d.class_ids.apply(lambda c: 0 in list(c))
dtr = d[d.split_official == 'train']; dte = d[d.split_official == 'test']
sup = splits['dfire_train_supplement']
row('dfire', 'train_supplement (in E = expB_train)', 'ungrouped (web images, no site/event structure)', sup['kept'],
    'n/a (stills)', sup['kept'] - sup['negatives'], sup['negatives'], 'human-annotated', 'no', 'web', 'train',
    'CC0-1.0', 'confirmed',
    f"splits_frozen.json dfire_train_supplement: kept {sup['kept']}, negatives {sup['negatives']}, excluded_fire_only {sup['excluded_fire_only']}; "
    f"official train split has {len(dtr)} images ({int((dtr.n_boxes==0).sum())} empty-label). Negatives = empty label files (author-annotated no fire/smoke box). "
    f"E = sdis_train 22,177 + D-Fire 16,277 = expB_train 38,454 (sha 51ac91f7...). PROTOCOL s2: TRAIN unchanged.",
    f'{P_DF}; {P_SPLITS}')
row('dfire', 'official_test (dfire_test_raw.txt)', 'ungrouped', len(dte), 'n/a (stills)', int((dte.n_boxes > 0).sum()),
    int((dte.n_boxes == 0).sum()), 'human-annotated', 'no', 'web', 'historical', 'CC0-1.0', 'confirmed',
    'Not assigned any v5 role in PROTOCOL s2; web imagery without camera/site structure; listed in the v4 leak-check reference lists only (4,306 lines).',
    f'{P_DF}; reports/v4/LOCKBOX_V4_FROZEN.md')

# ----------------------------------------------------------------------------
# 3. FIgLib / HPWREN (fixed-camera; onset truth = first visible plume, 60 s cadence)
# ----------------------------------------------------------------------------
P_F39 = 'data/manifests/figlib_subset_manifest.parquet'; P_L2 = 'data/v2/manifests/figlib_lockbox_manifest.parquet'
P_L3 = 'data/v3/manifests/figlib_lockbox_v3_manifest.parquet'; P_IDX = 'data/manifests/figlib_events_index.csv'
HP = 'CC BY-NC-ND 4.0 (HPWREN/ALERTCalifornia terms, verified 2026-10-02; PROTOCOL_V5 Amendment A1) — eval-only, no derivative release'
def figlib_row(path, name, purpose_note):
    f = pd.read_parquet(path)
    ev = f.event_id.nunique(); pre = int((f.time_offset_s < 0).sum()); post = int((f.time_offset_s >= 0).sum())
    cad = f.sort_values(['event_id', 'time_offset_s']).groupby('event_id').time_offset_s.diff().median()
    pre_h = (-f[f.time_offset_s < 0].groupby('event_id').time_offset_s.min()).sum() / 3600
    row('figlib', name, ev, len(f),
        f'{pre_h:.1f} h pre-onset (offset<0) + post-onset frames; {cad:.0f} s sampled; per-event clips, never concatenated',
        post, pre, 'human-annotated', 'yes', 'fixed-camera', 'historical', HP, 'confirmed',
        f'{ev} events / {f.camera_id.nunique()} cameras; offset 0 = HPWREN first-visible-plume label (author-labelled onset; frames offset<0 labelled pre-visible). '
        + purpose_note, path, pre_h, pre_h, ev)
    return ev
e39 = figlib_row(P_F39, 'figlib39_replay (v1-v4 early-detection set)',
                 'PROTOCOL s2: HISTORICAL comparison only; banned from training/tuning/selection. Used as the >=30/39 fixed-camera calibration CONSTRAINT (s3), not as a fresh test. 65/89 opened events are also inside public MultiNatSmoke (eval leak for externally pretrained models).')
e2 = figlib_row(P_L2, 'lockbox_v2 (opened once at v2 close)', 'PROTOCOL s2: HISTORICAL. Frozen 2026-09-11 (data/v2/manifests/LOCKBOX_FROZEN.md).')
e3 = figlib_row(P_L3, 'lockbox_v3 (opened once at v3 close)', 'PROTOCOL s2: HISTORICAL. Frozen 2026-09-13 (data/v3/manifests/LOCKBOX_V3_FROZEN.md).')
idx = pd.read_csv(P_IDX)
opened = set(pd.read_parquet(P_F39).event_id) | set(pd.read_parquet(P_L2).event_id) | set(pd.read_parquet(P_L3).event_id)
rem = idx[~idx.event_id.isin(opened)]
row('figlib', 'unopened_remainder (index only; NOT on disk)', len(rem), 0,
    f'~{(rem.n_frames_with_offset.sum()*60/3600):.0f} h TOTAL listed (pre+post onset) at 60 s cadence, roughly half pre-onset (~40 min per event per index min_offset); nothing downloaded',
    0, 0, 'none', 'yes', 'fixed-camera', 'final-test', HP, 'confirmed',
    f'{len(rem)} events never opened by any iteration ({int(rem.listing_ok.sum())} with listing_ok; {rem.camera_id.nunique()} cameras; {int(rem.n_frames_listed.sum())} frames listed; years '
    f'{rem.date.astype(str).str[:4].min()}-{rem.date.astype(str).str[:4].max()}). CANDIDATE pool for fixed-camera onset FINAL TEST: author (HPWREN) first-visible-plume labels exist, '
    'but PROTOCOL s3/s6 require human-reviewed [last_clear_no_smoke, first_clear_smoke] intervals labelled WITHOUT seeing candidate-model outputs, and s2 requires assignment '
    'by whole event at ingest with a dated protocol amendment. Count here = index rows, not verified downloads. Different camera network from pyro-sdis (HPWREN, 60 s cadence).',
    f'{P_IDX}; {P_F39}; {P_L2}; {P_L3}', 0, 0, len(rem))

# ----------------------------------------------------------------------------
# 4. Boreal (aerial UAV; CC BY 4.0)
# ----------------------------------------------------------------------------
P_BOR = 'data/v4/manifests/boreal_manifest.parquet'; P_AER = 'data/v4/manifests/aerial_split_v4.json'
P_SEQ = 'data/v4/sequences/sequences_v4.parquet'; P_EVV = 'data/v4/eval/EVALVAL_MANIFEST.json'; P_EVT = 'data/v4/eval/EVALTEST_MANIFEST.json'
b = pd.read_parquet(P_BOR)
b['subset'] = b.rel_path.str.extract(r'(Subset-[ABC])')[0]
b['is_empty_img'] = b.rel_path.str.contains('Empty-Images')
aer = json.load(open(P_AER))['sources']['boreal']
burn_purpose = {u: 'train' for u in aer['train']}; burn_purpose.update({u: 'calibration/val' for u in aer['val']}); burn_purpose.update({u: 'historical' for u in aer['test']})
burn_date = {'burn_evo': '2022-05-24', 'burn_heinola': '2022-06-28', 'burn_karkkila': '2022-07-06', 'burn_ruokolahti': '2022-08-15'}
seq = pd.read_parquet(P_SEQ)
bs = seq[seq.source == 'boreal']
evv = json.load(open(P_EVV)); evt = json.load(open(P_EVT))
CCBY = 'CC-BY-4.0 (Fairdata/Etsin 1dce1023-493a-4d63-a906-f2a44f831898)'
for burn in ['burn_evo', 'burn_karkkila', 'burn_heinola', 'burn_ruokolahti']:
    a = b[(b.parent_unit == burn) & (b.subset == 'Subset-A')]
    pos = int(((a.label_type == 'box') & a.has_smoke_label).sum()); neg = int(a.is_empty_img.sum()); unl = int((a.label_type == 'none').sum())
    flights = a.camera_or_flight_id.nunique(); purp = burn_purpose[burn]
    note = (f'Prescribed burn {burn_date[burn]}, DJI Phantom 4, {flights} DJI recordings; frames = every 48th video frame (~1 s apart, intra-day time not recoverable -> no valid-hours figure). '
            f'Negatives = Subset-A Empty-Images with explicit zero-box label files (author-reviewed; 8 of 256 viewed by us). {unl} frames with visible smoke but empty/missing label excluded (never negatives). ')
    if burn == 'burn_heinola':
        note += (f"PROTOCOL s2: AERIAL CALIBRATION/VAL, NEVER training; raw video of this burn inherits VAL. EVALVAL boreal_val = {evv['boreal_val']['images']} imgs / {evv['boreal_val']['positives']} pos / {evv['boreal_val']['negatives']} neg "
                 f"(coco sha {evv['boreal_val']['coco_sha256'][:8]}...). Aerial constraint: frame recall >= 0.90 here. ONE burn = anecdotal (aerial_split tiny_n_warning).")
    elif burn == 'burn_ruokolahti':
        note += (f"PROTOCOL s2: AERIAL FINAL TEST opened ONCE at v4 close -> now HISTORICAL, not reusable as a fresh final set. LOCKBOX_V4: {evt['boreal_test']['images']} imgs / {evt['boreal_test']['positives']} pos / {evt['boreal_test']['negatives']} neg, 1 event. "
                 "Public MultiNatSmoke re-ships 2,215 ruokolahti frames (1,748 in its official Train) -> externally pretrained checkpoints are contaminated here.")
    else:
        note += 'PROTOCOL s2: TRAIN (T component of E_ET_v1 / C_EGT_v1: boreal 2,062 lines = 2,025 pos + 37 neg across evo+karkkila; verified_empty_negatives=true).'
    row('boreal', f'{burn} Subset-A frames', 1, len(a), 'n/a (sampled stills; flights not contiguous)', pos, neg,
        'author-verified empty' if neg else 'none', 'no', 'aerial', purp, CCBY, 'confirmed', note,
        f'{P_BOR}; {P_AER}; {P_EVV}; {P_EVT}; data/v4/manifests/boreal_CARD.md')
# Subset-B clips (video-level labels)
def clip_row(burns, label, purp, name, note):
    sb = b[(b.subset == 'Subset-B') & (b.parent_unit.isin(burns)) & (b.has_smoke_label == (label == 'smoke'))]
    nclips = len(sb)
    fr = bs[(bs.parent_unit.isin(burns)) & (bs.seq_label == ('smoke' if label == 'smoke' else 'no_smoke'))]
    nfr = len(fr); hrs = nfr / 3600.0
    if nfr == 0: hours = f'{nclips} x ~30 s clips (~{nclips*30/3600:.2f} h nominal); not frame-extracted'
    else: hours = f'{hrs:.3f} h @1 fps extracted ({nfr} frames, {fr.seq_id.nunique()} clips); clips are independent, never concatenated'
    isneg = label != 'smoke'
    row('boreal', name, nclips, nfr, hours, 0 if isneg else nfr, nfr if isneg else 0,
        'video-level label' if isneg else 'none', 'no', 'aerial', purp, CCBY, 'confirmed', note,
        f'{P_BOR}; {P_SEQ}; data/v4/sequences/ONSET_AUDIT.md', hrs if isneg else 0, hrs if isneg else 0)
clip_row(['burn_evo', 'burn_karkkila'], 'smoke', 'train', 'Subset-B smoke clips evo+karkkila (train burns)',
         'Per-30 s-clip boolean smoke label only (authors: double-checked, not validated for training). Frame onset UNKNOWN and not invented; 30 s clips cannot carry a full onset but may contain a transition -> under limited human audit (Day-1 onset-audit agent). Weak video-level positives; NOT in any YOLO corpus.')
clip_row(['burn_karkkila'], 'no_smoke', 'train', 'Subset-B non-smoke clip karkkila_44 (train burn)',
         'The single train-burn non-smoke clip; PROTOCOL s2 = train negative. Also sits in replay_val_v4.jsonl pool flagged as TRAIN-site material.')
clip_row(['burn_heinola'], 'smoke', 'calibration/val', 'Subset-B smoke clips heinola (val burn)',
         'Video-level positives of the VAL burn; clip-level detection replay only; no frame truth. Never training.')
clip_row(['burn_heinola'], 'no_smoke', 'calibration/val', 'Subset-B non-smoke clips heinola (val burn)',
         'THE ONLY aerial no-smoke video on disk: 14 clips x ~31 s. Headline uncontaminated aerial negative pool in EARLY_WARNING_V4 (0.122 h). ONE burn/day/camera -> 1 independent unit. PROTOCOL: never training.')
clip_row(['burn_ruokolahti'], 'smoke', 'historical', 'Subset-B smoke clips ruokolahti (test burn)',
         'All 58 clips labelled smoke; never frame-extracted or read (test burn). Burn is HISTORICAL after v4 final test.')
sc = b[b.subset == 'Subset-C']
row('boreal', 'Subset-C (downsampled re-release of Subset-A + SAM masks)', 4, len(sc), 'n/a', int(sc.has_smoke_label.sum()), 0, 'none', 'no', 'aerial',
    'quarantine', CCBY, 'confirmed',
    f'{len(sc)} rows: 1,463 are downsampled duplicates of Subset-A frames (excluded from corpora/eval as subsetC_duplicate_of_subsetA: val 382, test 453) + 9 unlabeled extra frames with visible smoke (label_type none). '
    'Masks are SAM model-assisted (1,417) + 40 manual (test). Quarantined to prevent double counting / A-vs-C split leakage; masks may be reused for the SAME frames only.',
    f'{P_BOR}; {P_EVV}; {P_EVT}; data/v4/manifests/boreal_CARD.md')

# ----------------------------------------------------------------------------
# 5. FIRESENSE (CC BY 4.0; video-level labels; 2005-2011 footage)
# ----------------------------------------------------------------------------
P_FS = 'data/v4/manifests/firesense_manifest.parquet'; P_FX = 'data/v4/raw/firesense/frame_extraction_summary.json'
fs = pd.read_parquet(P_FS); fx = json.load(open(P_FX))
def fs_group(prefix):
    cat = {'smoke_videos': 'smoke', 'fire_videos': 'flame'}[prefix.split('/')[0]] + '_' + prefix.split('/')[1]
    vids = fs[fs.rel_path.str.endswith('.avi') & fs.rel_path.str.startswith(prefix + '/')]
    frames = fs[fs.rel_path.str.startswith(f'frames/{cat}/')]
    secs = sum(v['decoded_frames'] / v['fps'] for k, v in fx.items() if k.startswith(cat + '/') and v['fps'])
    assert len(vids) > 0 and len(frames) > 0, (prefix, cat)
    return len(vids), len(frames), secs / 3600
nv, nf, h = fs_group('smoke_videos/neg')
row('firesense', 'smoke_neg (9 videos)', nv, nf, f'{h:.3f} h decoded video ({nf} frames @~1 fps); 9 independent short clips, never concatenated', 0, nf,
    'video-level label', 'no', 'fixed-camera', 'historical', 'CC-BY-4.0 (Zenodo 10.5281/zenodo.836749)', 'confirmed',
    'PROTOCOL s2: fixed-camera-LIKE negative REPLAY only. Confounders (blossom tree, icy lot, office, bushes, reeds, highway x2, clouds behind trees, flags); 146 sampled frames eye-audited in v4. '
    'CONTAMINATED for v4-trained arms B_G/C_EGT/D_adapt (these 292 frames are background negatives in those corpora) -> optimistic FP for them; clean for expB/expE/A/E_ET. testneg03 decodes only 6 s. Not forest, not UAV.',
    f'{P_FS}; {P_FX}; data/v4/manifests/firesense_CARD.md; data/v4/sequences/REPORT.md', h, h)
nv, nf, h = fs_group('smoke_videos/pos')
row('firesense', 'smoke_pos (13 videos)', nv, nf, f'{h:.3f} h decoded ({nf} frames @~1 fps)', nf, 0, 'none', 'no', 'web', 'quarantine',
    'CC-BY-4.0 (Zenodo 10.5281/zenodo.836749)', 'confirmed',
    'Video-level positive only: smoke not visible in every frame, no boxes; mostly staged near-field smoke pots/grenades, one NIST structure burn, burned-in overlays/detector contours (shortcut risk). '
    'Quarantined from frame-level training/eval (no frame truth, domain mismatch); not an onset source (no label of first visibility).',
    f'{P_FS}; {P_FX}; data/v4/manifests/firesense_CARD.md')
nv, nf, h = fs_group('fire_videos/neg')
row('firesense', 'flame_neg (16 videos)', nv, nf, f'{h:.3f} h decoded ({nf} frames @~1 fps)', 0, 0, 'video-level label', 'no', 'web', 'aux-train-candidate',
    'CC-BY-4.0 (Zenodo 10.5281/zenodo.836749)', 'confirmed',
    'Flame-negative label is NOT a smoke-negative label (exhaust/dust in traffic clips not checked frame-by-frame) -> verified_negatives=0 until human smoke-free review. Urban/indoor confounders (night traffic glare, trains, offices). Not forest/UAV.',
    f'{P_FS}; {P_FX}; data/v4/manifests/firesense_CARD.md', 0, 0)
nv, nf, h = fs_group('fire_videos/pos')
row('firesense', 'flame_pos (11 videos)', nv, nf, f'{h:.3f} h decoded ({nf} frames @~1 fps)', 0, 0, 'none', 'no', 'web', 'quarantine',
    'CC-BY-4.0 (Zenodo 10.5281/zenodo.836749)', 'confirmed',
    'Flame-positive videos with UNLABELLED smoke (bus fire, crown fire, trash fire); watermarks (UltimateChase.com). No smoke truth -> neither positive nor negative for smoke.',
    f'{P_FS}; {P_FX}; data/v4/manifests/firesense_CARD.md')

# ----------------------------------------------------------------------------
# 6. MmodalFire (indoor lab; CC BY 4.0; smoke never annotated)
# ----------------------------------------------------------------------------
P_MM = 'data/v4/manifests/mmodalfire_manifest.parquet'
mm = pd.read_parquet(P_MM); mf = mm[mm.item_kind == 'frame']
MMLIC = 'CC-BY-4.0 (figshare 10.6084/m9.figshare.28804448.v1)'
def mm_row(sel, name, purp, note, vneg_ok=False):
    g = mf[sel]; n = len(g); sess = g.parent_unit.nunique(); h = n / 3600
    row('mmodalfire', name, sess, n, f'{h:.3f} h @1 fps inside XLSX session intervals ({n} frames); sessions independent, never concatenated',
        0, 0, 'video-level label' if vneg_ok else 'none', 'no', 'indoor', purp, MMLIC, 'confirmed', note,
        f'{P_MM}; data/v4/manifests/mmodalfire_CARD.md', h if vneg_ok else 0, 0)
mm_row(mf.session_label == 'fire', 'fire sessions (cotton/n-heptane/polyurethane/wood)', 'aux-train-candidate',
       'Session-level fire label only; per-frame smoke visibility NOT annotated (early frames may show none; late frames whiteout). PROTOCOL s2: AUXILIARY TRAIN candidate after human visual labelling; never a forest/UAV eval. 1 room / 1 camera / 1 week = 1 site. Chinese OSD + watermark on every frame (crop at train time).')
mm_row(mf.material == 'water-mist', 'interference: water-mist (humidifier plume)', 'aux-train-candidate',
       'Labelled non-fire; VISIBLE white vertical plume = best smoke look-alike hard negative in the deposit. Smoke-free by experiment design (no combustion), but verified_negatives kept 0 until the human review pass labels frames (PROTOCOL: detector/sensor silence is not evidence). Indoor only.', True)
mm_row(mf.material == 'dry-ice', 'interference: dry-ice (no visible fog)', 'aux-train-candidate',
       'Labelled non-fire; visually near-identical to blank room (sensor-only interference). Easy negatives; same human-review requirement; indoor only.', True)
mm_row(mf.session_label == 'blank', 'blank control sessions', 'aux-train-candidate',
       'Empty room, labelled blank. Human review still required before counting as verified negatives; indoor only.', True)
occ = mm[(mm.item_kind == 'video') & mm.rel_path.str.contains('Video/')]
row('mmodalfire', 'occlusion sessions (2, video only) + raw DVR files', 2, 0, '33 h raw DVR total (1,982 min); ~26 h OUTSIDE session intervals not extracted', 0, 0, 'none', 'no', 'indoor', 'quarantine',
    MMLIC, 'confirmed',
    f'{int((mm.item_kind=="video").sum())} raw DVR MP4s. 2 occlusion sessions = lens covered (no usable video). Footage outside XLSX intervals is unlabelled idle room / setup / smoke clearing -> residual smoke possible -> NOT negatives.',
    f'{P_MM}; data/v4/manifests/mmodalfire_CARD.md')

# ----------------------------------------------------------------------------
# 7. MultiNatSmoke / AuSmoke (PERMISSION PENDING) + quarantined subsets
# ----------------------------------------------------------------------------
P_MNS = 'data/v4/manifests/multinatsmoke_manifest.parquet'; P_FH = 'data/v4/manifests/forest_holdout_v4.json'
P_DD = 'data/v4/manifests/dedup_v4_meta.json'
mns = pd.read_parquet(P_MNS); fh = json.load(open(P_FH)); dd = json.load(open(P_DD))
au = mns[mns.subset == 'AuSmoke']
unit_n = au.groupby('parent_unit').size().to_dict()
grp_role = {g: r for r, gs in fh['assignment'].items() for g in gs}
AULIC = 'unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING'
def au_scenario(gname):
    if gname.startswith('g_dji'): return 'aerial'
    if gname == 'g_axis_fixedcam': return 'fixed-camera'
    return 'forest-ground'
for gname, units in fh['groups'].items():
    n = sum(unit_n['aus_' + u.replace('aus_', '')] if ('aus_' + u.replace('aus_', '')) in unit_n else unit_n.get(u, 0) for u in units)
    role = grp_role[gname]; scen = au_scenario(gname)
    extra = {'train': 'v4 role TRAIN (T component; 10,904 lines in E_ET/C_EGT after 2 near-dup bans). ',
             'val': 'v4 role VAL (forest_val 1,471 frames, 0 negatives; thresholds had to come from boreal_val). ',
             'test': 'v4 role TEST (forest_test; opened ONCE at v4 close -> historical for fresh-test purposes). '}[role]
    special = ''
    if gname == 'g_20240209_smokemachine': special = 'SYNTHETIC-SOURCE smoke machine in a real scene: train-only, never eval. '
    if gname == 'g_dji_20230922': special = 'ONE DJI flight (3 videos) incl. staged TEST_RUN_SMOKE unit (90 fr) -> report separately. '
    if gname == 'g_axis_fixedcam': special = 'Axis Q6215LE PTZ tower (Mount Stromlo): 2 burns from ONE fixed camera; OSD banners burned in. '
    if gname == 'g_unknown00': special = 'aus_000xx units of unknown provenance kept as one conservative group. '
    row('multinatsmoke/AuSmoke', f'{gname} [{role}] units={len(units)}', len(units), n, 'unknown (fps not recorded; filename offsets at 1 s or 60 s/frame; no per-hour rate emitted)',
        n, 0, 'none', 'no', scen, 'permission-pending', AULIC, 'pending',
        f'ACT Australia (ANU BRCoE). ALL frames positive (mask_fg_px>0); 0 negatives; 0/26 sequences observe an onset (segments trimmed at/after first visible smoke). Masks model-assisted (SAM/pseudo-label), generous on faint plumes. '
        + extra + special + 'PROTOCOL s2: no NEW training use until licence confirmed in writing; existing v4 use disclosed, not extended. Author contact in multinatsmoke_CARD.md.',
        f'{P_MNS}; {P_FH}; data/v4/sequences/ONSET_AUDIT.md; data/v4/manifests/multinatsmoke_CARD.md')
for sub, lic, scen, note in [
    ('WSDataset', 'MIT per MultiNatSmoke README (kaggle gloryvu/wildfire-smoke); aggregate unlicensed -> PERMISSION PENDING per PROTOCOL s2', 'fixed-camera',
     'Vietnam (Lam Dong) VMOKED camera footage, 8 derivable groups (~4 cameras); positives-only; no timestamps for 2,484 frames. In G (B_G_v1 / C_EGT_v1 / D_pre) in v4; v4 verdict: G adds nothing on top of T.'),
    ('FireSpot', 'CC-BY-4.0 per MultiNatSmoke README (Biometrix-4/FireSpot-CNX); aggregate unlicensed -> PERMISSION PENDING per PROTOCOL s2', 'fixed-camera',
     'Thailand (Chiang Mai) towers, 29 site groups; positives-only here (original 1,179 non-smoke images NOT included in MNS). In G in v4.'),
]:
    g = mns[(mns.subset == sub) & (mns.origin_class == 'usable')]
    row('multinatsmoke/' + sub, f'{sub} usable (G component)', g.parent_unit.nunique(), len(g), 'n/a (stills; fps/time not recorded)', int(g.has_smoke_label.sum()), 0, 'none', 'no', scen,
        'permission-pending', lic, 'pending', note + ' Original-source licence NOT independently verified at source; PROTOCOL s2 lists all non-AuSmoke usable (G) as permission PENDING.',
        f'{P_MNS}; data/v4/manifests/CORPUS_B_G_v1.json; data/v4/manifests/multinatsmoke_CARD.md')
g = mns[(mns.subset == 'FESB-MLID') & (mns.origin_class == 'usable')]
row('multinatsmoke/FESB-MLID', 'FESB-MLID usable (G component)', g.parent_unit.nunique(), len(g), 'n/a (stills)', int(g.has_smoke_label.sum()), 0, 'none', 'no', 'fixed-camera',
    'permission-pending', 'unspecified (wildfire.fesb.hr, no licence stated) -> PERMISSION PENDING', 'pending',
    'Croatia FESB stills/short sequences; 46 further FESB images are byte-identical to local D-Fire (quarantined as dup). Mask origin unknown. In G in v4.',
    f'{P_MNS}; data/v4/manifests/multinatsmoke_CARD.md')
qf = mns[mns.origin_class == 'quarantine_figlib']
row('multinatsmoke/FIgLib-family', 'FIgLib + SmokeSeg + aiformankind + firecam + SMOKE5K_real (quarantine_figlib)', int(qf.parent_unit.nunique()), len(qf), 'n/a', int(qf.has_smoke_label.sum()), 0, 'none', 'no', 'fixed-camera',
    'quarantine', 'FIgLib HPWREN terms / CC-BY-NC-SA-4.0 (aiformankind, firecam) / unspecified (SmokeSeg, SMOKE5K)', 'blocked',
    f"EVAL LEAK: 65 of our 89 opened FIgLib eval events appear (1,929 byte-identical frames); dedup_v4_meta lists {len(dd['evalhit_units'])} evalhit_units with exact-dup eval frames. Subset-level quarantine (crops/re-encodes evade sha). "
    f"Subsets: FIgLib {int((mns.subset=='FIgLib').sum())}, SmokeSeg {int((mns.subset=='SmokeSeg').sum())}, aiformankind {int((mns.subset=='aiformankind').sum())}, firecam {int((mns.subset=='firecam').sum())}, SMOKE5K_real {int(((mns.subset=='SMOKE5K_real')&(mns.origin_class=='quarantine_figlib')).sum())}. Also non-commercial licences on two subsets.",
    f'{P_MNS}; {P_DD}; data/v4/manifests/multinatsmoke_CARD.md')
qd = mns[mns.origin_class == 'dup_dfire']
row('multinatsmoke/D-Fire-dup', 'D-Fire + FESB(46) + SMOKE5K(26) byte-identical to local D-Fire (dup_dfire)', 'ungrouped', len(qd), 'n/a', len(qd), 0, 'none', 'no', 'web', 'quarantine',
    'CC0-1.0 (D-Fire) / unspecified', 'blocked',
    'Already in E via local D-Fire; quarantined to avoid double counting. MNS adds SAM-from-box masks for these images (possible extra supervision on the SAME training images only).',
    f'{P_MNS}; {P_DD}')
qb = mns[mns.origin_class == 'dup_boreal']
row('multinatsmoke/Boreal-dup', 'Boreal re-shipped inside MultiNatSmoke (dup_boreal, 30 DJI videos)', int(qb.parent_unit.nunique()), len(qb), 'n/a', len(qb), 0, 'none', 'no', 'aerial', 'quarantine',
    'CC-BY-4.0 (Boreal)', 'blocked',
    'Byte-identical Boreal copies incl. heinola (VAL) and ruokolahti (TEST; 2,215 rows, 1,748 in MNS official Train). Must stay excluded from every corpus (LOCKBOX_V4 rule 5); any checkpoint pretrained on public MultiNatSmoke is CONTAMINATED for boreal val/test.',
    f'{P_MNS}; reports/v4/LOCKBOX_V4_FROZEN.md')
qu = mns[mns.origin_class == 'quarantine_unknown']
row('multinatsmoke/Forest_Fire', 'Forest_Fire (masks only, images not shipped)', len(qu), 0, 'n/a', 0, 0, 'none', 'no', 'web', 'quarantine',
    'unspecified (Kaggle re-upload kutaykutlu/forest-fire of USTC video frames)', 'none',
    f'{len(qu)} mask-only rows (image_present=False); images need a separate Kaggle download; no licence; provenance unknown.',
    f'{P_MNS}; data/v4/manifests/multinatsmoke_CARD.md')

# ----------------------------------------------------------------------------
# 8. FASDD_UAV (zip on disk, not manifested) — reports/fasdd_scout.md
# ----------------------------------------------------------------------------
row('fasdd_uav', 'FASDD_UAV.zip (downloaded 2026-09-10, not extracted/manifested)', 'unknown (no per-video grouping shipped)', 25097,
    'n/a (stills/video frames; heavy near-duplicate redundancy)', 7821 + 5080, 0, 'none', 'no', 'web', 'quarantine',
    'CC-BY-SA-4.0 (ScienceDB 10.57760/sciencedb.j00104.00103; share-alike)', 'confirmed',
    'Filename classes: neitherFireNorSmoke 11,986 / bothFireAndSmoke 7,821 / smoke 5,080 / fire 210 (scout counts). 60-image visual sample: ~30 of ~33 negatives are HPWREN fixed watchtower frames (FIgLib family -> eval-contamination risk, and NOT UAV); positives look like genuine low-altitude drone frames of grass burns. '
    'Quarantined pending dedup + per-image provenance filter; author class labels not re-verified by us; share-alike licence viral for derived annotations. Zip 14,888,184,890 B md5 c5ea9651 verified.',
    'reports/fasdd_scout.md; data/raw/fasdd_uav/FASDD_UAV.zip')

# ----------------------------------------------------------------------------
# 9. DFS (blocked), v2 synth pilot (historical), v5 new footage (none yet)
# ----------------------------------------------------------------------------
row('dfs', 'DFS-FIRE-SMOKE-Dataset (0 bytes acquired)', 0, 0, 'n/a', 0, 0, 'none', 'no', 'web', 'quarantine',
    'none found (GitHub license None; citation requested)', 'blocked',
    'OneDrive dead; Baidu Pan share alive but needs Chinese-phone account + netdisk client; no author mirror. README claims 9,462 images with fire/smoke/other boxes (unverified). Needs a human download + legal review + D-Fire near-dup check if ever acquired.',
    'data/v4/manifests/dfs_CARD.md')
gs = json.load(open('data/v2/synth/generation_stats.json'))
n_png = sum(s['n_png'] for s in gs['segments'].values())
row('v2_synth_pilot', 'Qwen-Image-Edit composites on 150 TRAIN-split sdis negative backgrounds', 150, n_png, 'n/a (stills)', 0, 0, 'none', 'no', 'fixed-camera', 'historical',
    'derived from pyro-sdis (Apache-2.0); generator Apache-2.0', 'confirmed',
    f'{n_png} generated PNGs (smoke / control / confounder jobs; label_status=unverified-synthetic; screening showed requests largely NOT honoured); 72 quarantined (data/v2/synth/quarantine.txt). Never trained on (FINAL_REPORT_V2). Not a data source for v5.',
    'data/v2/synth/generation_stats.json; data/v2/synth/PILOT.md; data/v2/synth/quarantine.txt')
row('v5_new_acquisitions', '(none ingested as of this inventory)', 0, 0, '0 h', 0, 0, 'none', 'no', 'aerial', 'calibration/val',
    'n/a', 'none',
    'PLACEHOLDER ROW: scenario/purpose values here are placeholders (enum requires a value) and will be set per unit at ingest. PROTOCOL s2: any newly acquired footage is assigned by WHOLE site/flight/date at ingest and recorded under data/v5/manifests BEFORE any model sees it. Nothing has arrived yet (acquisition agent + FGI / AuSmoke email drafts pending user review).',
    'data/v5/manifests/ (empty before this file)')

# ----------------------------------------------------------------------------
# write CSV
# ----------------------------------------------------------------------------
df = pd.DataFrame(rows)
cols = ['source', 'unit_group', 'n_units', 'n_images_or_frames', 'hours_if_video', 'positives', 'verified_negatives',
        'negative_evidence_type', 'onset_truth', 'scenario', 'purpose_v5', 'licence', 'permission_status', 'notes',
        'artifacts', 'valid_hours_num', 'neg_hours_num', 'onset_events_num']
df = df[cols]
df.to_csv(OUT_CSV, index=False)

# ----------------------------------------------------------------------------
# MD
# ----------------------------------------------------------------------------
num = df.copy()
for c in ['n_images_or_frames', 'positives', 'verified_negatives']:
    num[c] = pd.to_numeric(num[c], errors='coerce').fillna(0).astype(int)
def tbl(frame, cols, headers=None):
    headers = headers or cols
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(cols)]
    for _, r in frame.iterrows():
        out.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(out)


def pivot_md(pv):
    pv = pv.reset_index()
    cols = list(pv.columns)
    return tbl(pv, cols, [str(c) for c in cols])

purp_order = ['train', 'calibration/val', 'final-test', 'historical', 'aux-train-candidate', 'permission-pending', 'quarantine']
by_p = num.groupby('purpose_v5').agg(rows=('source', 'size'), images_or_frames=('n_images_or_frames', 'sum'), positives=('positives', 'sum'),
                                     verified_negatives=('verified_negatives', 'sum'), verified_neg_video_h=('neg_hours_num', 'sum')).reindex(purp_order).fillna(0)
by_p['verified_neg_video_h'] = by_p.verified_neg_video_h.round(3)
by_p = by_p.reset_index()
for c in ['rows', 'images_or_frames', 'positives', 'verified_negatives']: by_p[c] = by_p[c].astype(int)

scen_order = ['fixed-camera', 'aerial', 'forest-ground', 'indoor', 'web']
by_s = num.groupby('scenario').agg(rows=('source', 'size'), images_or_frames=('n_images_or_frames', 'sum'), positives=('positives', 'sum'),
                                   verified_negatives=('verified_negatives', 'sum'), verified_neg_video_h=('neg_hours_num', 'sum')).reindex(scen_order).fillna(0).reset_index()
by_s['verified_neg_video_h'] = by_s.verified_neg_video_h.round(3)
for c in ['rows', 'images_or_frames', 'positives', 'verified_negatives']: by_s[c] = by_s[c].astype(int)

# scenario x purpose verified negatives (frames and hours)
sp = num.pivot_table(index='scenario', columns='purpose_v5', values='verified_negatives', aggfunc='sum', fill_value=0).reindex(index=scen_order, columns=purp_order).fillna(0).astype(int)
sph = num.pivot_table(index='scenario', columns='purpose_v5', values='neg_hours_num', aggfunc='sum', fill_value=0).reindex(index=scen_order, columns=purp_order).fillna(0).round(3)

S = sdis_site_stats
val_sites = [s for s in S if S[s]['split'] == 'val']; test_sites = [s for s in S if S[s]['split'] == 'test']; train_sites = [s for s in S if S[s]['split'] == 'train']
val_vn = sum(S[s]['vn'] for s in val_sites); val_va = sum(S[s]['va'] for s in val_sites); val_neg = sum(S[s]['neg'] for s in val_sites); val_cams = sum(S[s]['cams'] for s in val_sites)
val_alln = sum(S[s]['nalln'] for s in val_sites); val_alln_h = sum(S[s]['alln_h'] for s in val_sites); val_app = sum(S[s]['napp'] for s in val_sites)
test_vn = sum(S[s]['vn'] for s in test_sites); test_neg = sum(S[s]['neg'] for s in test_sites); test_cams = sum(S[s]['cams'] for s in test_sites)
train_vn = sum(S[s]['vn'] for s in train_sites); train_neg = sum(S[s]['neg'] for s in train_sites); train_app = sum(S[s]['napp'] for s in train_sites)
hein_clip_h = num[(num.source == 'boreal') & num.unit_group.str.startswith('Subset-B non-smoke clips heinola')].neg_hours_num.sum()
hein_clip_n = int(num[(num.source == 'boreal') & num.unit_group.str.startswith('Subset-B non-smoke clips heinola')].n_units.iloc[0])
kark_clip_h = num[(num.source == 'boreal') & num.unit_group.str.startswith('Subset-B non-smoke clip karkkila')].neg_hours_num.sum()
fs_neg_h = num[(num.source == 'firesense') & num.unit_group.str.startswith('smoke_neg')].neg_hours_num.sum()
mm_cand_h = num[(num.source == 'mmodalfire') & (num.purpose_v5 == 'aux-train-candidate') & (num.negative_evidence_type == 'video-level label')].valid_hours_num.sum()
au_frames = int(num[num.source == 'multinatsmoke/AuSmoke'].n_images_or_frames.sum()); au_units = int(num[num.source == 'multinatsmoke/AuSmoke'].n_units.sum())
pp = num[num.purpose_v5 == 'permission-pending']
qq = num[num.purpose_v5 == 'quarantine']
now = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

md = f"""# DATA INVENTORY BY PURPOSE — V5 (Day 1)

Generated {now} by `scripts/v5/make_data_inventory_v5.py` (CPU only; reads manifests/lists, opens no test image).
Companion CSV: `data/v5/manifests/DATA_INVENTORY_V5.csv` ({len(df)} rows; one row per source x unit-group / site / burn / video group;
extra machine columns `artifacts`, `valid_hours_num`, `neg_hours_num`, `onset_events_num`). Purpose assignments follow
`reports/v5/PROTOCOL_V5.md` section 2 verbatim; where the protocol is silent the row says so in `notes`.
Every number below is computed from the artifact named in the row's `artifacts` column or cited inline.

Conventions. "Valid observed time" follows PROTOCOL section 3: per camera/clip, gaps > 3x the median cadence are excluded and
independent clips are never concatenated. "verified_negatives" counts frames (or clip-frames) whose no-smoke status rests on a
human label or an author-verified empty annotation; detector silence counts for nothing. For pyro-sdis, a negative = a human-annotated
image with no smoke box (annotation absence, not a re-verified "no smoke" review; v3 sequence flicker is 29% train / 36% val).
"onset_truth": yes = human first-visible-smoke label per event; partial = background->smoke seen in box presence only (not a
reviewed interval); no = none. Reviewed onset INTERVALS per PROTOCOL section 3 ([last_clear_no_smoke, first_clear_smoke]) exist for
**0** events on disk today.

## 1. Totals per purpose (PROTOCOL section 2)

{tbl(by_p, ['purpose_v5', 'rows', 'images_or_frames', 'positives', 'verified_negatives', 'verified_neg_video_h'],
     ['purpose_v5', 'rows', 'images / frames', 'positives', 'verified negatives (frames)', 'verified-negative VIDEO hours (valid)'])}

Notes on the totals: `images / frames` sums stills and extracted video frames and double-counts nothing on disk (Subset-C duplicates
and MultiNatSmoke re-ships are in the quarantine row only). `final-test` includes the FIgLib unopened remainder as a CANDIDATE pool with
0 frames on disk (see section 3). The `permission-pending` positives were used in v4 (disclosed) and may not be extended.

## 2. Totals per scenario (PROTOCOL section 1: fixed-camera and aerial are calibrated and reported separately)

{tbl(by_s, ['scenario', 'rows', 'images_or_frames', 'positives', 'verified_negatives', 'verified_neg_video_h'],
     ['scenario', 'rows', 'images / frames', 'positives', 'verified negatives (frames)', 'verified-negative VIDEO hours (valid)'])}

### 2a. Verified negatives — scenario x purpose (frames)

{pivot_md(sp)}

### 2b. Verified-negative VIDEO hours (valid observed time) — scenario x purpose

{pivot_md(sph)}

Reading guide for the hours: pyro-sdis hours are neg-neg intervals at ~29-30 s sampled cadence (`pyro_sdis_manifest.parquet` datetimes, gap rule 3x median);
FIgLib hours are pre-onset time at 60 s cadence (historical only); Boreal/FIRESENSE hours are 1 fps extracted clip time. Still-image
negatives (Boreal Subset-A 217/37/2, D-Fire 7,833) carry no hours because their frames are not contiguous.

### 2c. Verified no-smoke footage per scenario, by source (what the calibration/acceptance can actually use)

| scenario | source / unit group | purpose | units | verified negatives | valid hours; cadence | evidence | artifact |
|---|---|---|---|---|---|---|---|
| fixed-camera | pyro-sdis VAL sites cabanelle + marguerite | calibration/val | {val_cams} cameras / 2 sites | {val_neg} frames | {val_vn:.2f} h neg-neg ({val_va:.2f} h all-frames); ~30 s sampled | human-annotated (box absence) | pyro_sdis_manifest.parquet + splits_frozen.json |
| fixed-camera | pyro-sdis VAL all-negative v3 sequences | calibration/val | {val_alln} seqs | 219 frames | {val_alln_h:.2f} h recomputed (EARLY_WARNING_V4 quotes 1.521 h); 29 s cadence | human-annotated | data/v3/sequences/sequences_val.json; reports/v4/EARLY_WARNING_V4.md |
| fixed-camera | pyro-sdis TEST sites ferion + croix-augas | final-test | {test_cams} cameras / 2 sites | {test_neg} frames | {test_vn:.2f} h neg-neg; ~30 s | human-annotated | same |
| fixed-camera | pyro-sdis TRAIN sites (4) | train | 21 cameras | {train_neg} frames | {train_vn:.2f} h neg-neg; ~30 s | human-annotated | same |
| fixed-camera | FIgLib-39 + lockbox v2 + v3 pre-onset frames | historical | 89 events | {int(num[(num.source=='figlib')&(num.purpose_v5=='historical')].verified_negatives.sum())} frames | {num[(num.source=='figlib')&(num.purpose_v5=='historical')].neg_hours_num.sum():.1f} h pre-onset; 60 s | human-annotated (HPWREN onset label) | figlib manifests |
| fixed-camera-like | FIRESENSE smoke_neg | historical (replay only) | 9 videos | 292 frames | {fs_neg_h:.3f} h; 1 fps | video-level label + eye audit | firesense_manifest.parquet; frame_extraction_summary.json |
| aerial | Boreal heinola Subset-A Empty-Images | calibration/val | 1 burn (9 flights) | 217 frames | n/a (sampled stills) | author-verified empty | boreal_manifest.parquet; EVALVAL_MANIFEST.json |
| aerial | Boreal heinola Subset-B non-smoke clips | calibration/val | 1 burn, {hein_clip_n} clips | 439 frames | **{hein_clip_h:.3f} h**; 1 fps | video-level label | sequences_v4.parquet |
| aerial | Boreal karkkila Empty-Images + clip 44 | train | 1 burn | 37 + 32 frames | {kark_clip_h:.3f} h; 1 fps | author-verified empty / video-level | boreal_manifest.parquet; sequences_v4.parquet |
| aerial | Boreal ruokolahti Empty-Images | historical | 1 burn | 2 frames | n/a | author-verified empty | EVALTEST_MANIFEST.json |
| aerial | AuSmoke DJI groups (g_dji_20230922 test, g_dji_0792 val) | permission-pending | 2 flights | 0 | unknown fps | none (positives-only) | multinatsmoke_manifest.parquet |
| forest-ground | AuSmoke ground groups | permission-pending | 13 groups | 0 | unknown fps | none (positives-only) | multinatsmoke_manifest.parquet |
| indoor | MmodalFire water-mist / dry-ice / blank | aux-train-candidate | 10 sessions | 0 until human review ({mm_cand_h:.2f} h candidate) | 1 fps | video-level label (non-fire) | mmodalfire_manifest.parquet |

## 3. Onset truth available (PROTOCOL section 3 early-warning metric; section 6 acceptance = >=10 independent events with REVIEWED intervals)

| scenario | source | events | truth type | status for v5 | artifact |
|---|---|---|---|---|---|
| fixed-camera | FIgLib-39 replay | {e39} | author first-visible-plume (60 s cadence) | HISTORICAL: models v1-v4 already run; usable only as the >=30/39 calibration constraint, never as fresh test | figlib_subset_manifest.parquet |
| fixed-camera | FIgLib lockbox v2 / v3 | {e2} / {e3} | author first-visible-plume | HISTORICAL (opened once each) | lockbox manifests |
| fixed-camera | FIgLib unopened remainder | {len(rem)} (index) | author first-visible-plume | CANDIDATE final-test pool; 0 on disk; needs download, human interval review blind to model outputs, and a dated protocol amendment before assignment | figlib_events_index.csv |
| fixed-camera | pyro-sdis VAL appears-type v3 sequences | {val_app} seqs | box presence flips at 29 s cadence, 36% flicker | PARTIAL: candidates for human interval review (Day-1 flicker-audit agent); not reviewed intervals | sequences_val.json |
| fixed-camera | pyro-sdis TRAIN appears-type v3 sequences | {train_app} seqs | same, 29% flicker | PARTIAL: training-side only (never eval) | sequences_train.json |
| aerial | Boreal Subset-B smoke clips (train burns 104, val burn 111) | 0 observed | video-level only; 30 s clips | NO onset truth; transitions inside clips under limited audit; cannot give first-visible time of the burn | ONSET_AUDIT.md |
| aerial / forest-ground | AuSmoke 26 sequences | 0 of 26 | n/a (all frames positive, trimmed at/after first visible smoke) | NONE | ONSET_AUDIT.md |
| indoor | MmodalFire fire sessions | 26 sessions | session start time (XLSX, +/-30 s); smoke visibility unlabelled | not a forest/UAV onset source; sensor CSVs are audit aids only | mmodalfire_CARD.md |

**Reviewed onset intervals today: 0 fixed-camera, 0 aerial (acceptance needs >=10 independent events, final-test onsets labelled without seeing candidate outputs).**

## 4. PERMISSION PENDING (PROTOCOL section 2; no new training use until written confirmation; v4 use disclosed, not extended)

{tbl(pp, ['source', 'unit_group', 'n_units', 'n_images_or_frames', 'scenario', 'licence'], ['source', 'unit group', 'units', 'frames', 'scenario', 'licence status'])}

Summary: AuSmoke {au_units} units / {au_frames} frames (16 burn-grouped units: 9 train / 3 val / 4 test groups) — no licence anywhere (HF/GitHub);
MultiNatSmoke non-AuSmoke "usable" (G): WSDataset {int(num[num.source=='multinatsmoke/WSDataset'].n_images_or_frames.sum())} (MIT per README), FireSpot {int(num[num.source=='multinatsmoke/FireSpot'].n_images_or_frames.sum())} (CC BY 4.0 per README),
FESB-MLID {int(num[num.source=='multinatsmoke/FESB-MLID'].n_images_or_frames.sum())} (unspecified). The aggregate MultiNatSmoke release carries no licence; original-source licences are quoted from its README and were NOT
verified at source. These sets contribute 0 negatives and 0 onsets regardless of permission. Email draft to the AuSmoke/MultiNatSmoke authors is a separate Day-1 deliverable (user review before sending).

## 5. QUARANTINED (excluded from every model-facing use) and why

{tbl(qq, ['source', 'unit_group', 'n_images_or_frames', 'notes'], ['source', 'unit group', 'frames', 'reason (abridged in notes)'])}

Reasons in one line each: FIgLib-family inside MultiNatSmoke = eval leak (65/89 opened eval events, {len(dd['evalhit_units'])} exact-hit units; two subsets also CC-BY-NC-SA);
D-Fire/FESB/SMOKE5K dup = byte-identical to local D-Fire already in E; Boreal dup = byte-identical copies of VAL/TEST burns; Forest_Fire = no images, no licence;
Boreal Subset-C = downsampled duplicates of Subset-A (+9 unlabeled smoke frames); FIRESENSE smoke_pos/flame_pos = no frame truth, unlabeled smoke, overlays;
MmodalFire occlusion/out-of-session = lens covered / unlabeled residual smoke; FASDD_UAV = ~90% of sampled negatives are HPWREN tower frames (FIgLib-family
contamination risk, not UAV), unverified provenance, share-alike licence; DFS = unobtainable + unlicensed.

## 6. Acceptance gaps vs PROTOCOL section 6 ("data ready")

| criterion | fixed-camera | aerial / UAV | forest-ground (new region) |
|---|---|---|---|
| negatives: >=2 h valid, human-verified | pyro-sdis VAL {val_vn:.2f} h neg-neg ({val_neg} frames, ~30 s sampled) — MET on hours, but same source domain as training (not "matched footage ADDED") and labels are box-absence with 36% sequence flicker -> human spot-check recommended via the Day-1 review tool | **{hein_clip_h:.3f} h of 2 h** (14 heinola clips) + 217 sampled stills; gap = {2-hein_clip_h:.2f} h | **0 h of 2 h** (AuSmoke positives-only; permission pending anyway) |
| negatives: >=3 independent units | {val_cams} cameras but only **2 sites** (cabanelle, marguerite); independence judged at site level = NOT MET, at camera level = met | **1 burn of 3** (heinola; karkkila clip is train-side) | **0 of 3** |
| negatives: licence recorded | Apache-2.0 (yes) | CC BY 4.0 (yes) | n/a |
| onsets: >=10 independent events with reviewed intervals | **0 of 10 reviewed**; candidates: {len(rem)} unopened FIgLib events (author onset labels, not on disk), {val_app} sdis VAL appears-type sequences (flicker) | **0 of 10**; no pre-onset aerial footage exists on disk (Boreal clips are 30 s, AuSmoke trimmed) | **0 of 10** |
| final-test onsets labelled blind to model outputs | FIgLib-39 / lockboxes already seen by models -> cannot serve; remainder could, after freeze | none | none |
| training negatives from the NEW domains (Q1 premise) | FIRESENSE 292 frames (staged scenes, already in v4 corpora) | karkkila 37 frames + 1 clip (32 frames) = **69 frames** | **0** |

Bottom line: fixed-camera calibration (experiment A) can run now on pyro-sdis VAL embedded negatives (reported as "~30 s sampled, {val_vn:.2f} valid h")
with the FIgLib-39 >=30/39 constraint as a historical check. Aerial calibration can only report point estimates on {hein_clip_h:.3f} h / 1 burn (CIs will be
very wide, as in EARLY_WARNING_V4). Neither onset acceptance nor UAV negative acceptance is met; both depend on acquisition (FGI raw flights incl.
pre-ignition and no-smoke legs; any authorised no-smoke forest aerial footage) or on freezing a FIgLib remainder package for fixed-camera onsets.

## 7. Other facts the inventory relies on (artifact-cited)

- Frozen E corpus: expB_train 38,454 = sdis_train 22,177 + D-Fire supplement 16,277 (`splits_frozen.json`; sha 51ac91f7... identical in E_ET_v1 and C_EGT_v1 `train_E.txt`).
- v4 corpora: B_G_v1 14,901 (G: 14,609 MNS usable positives + 292 FIRESENSE negatives); E_ET_v1 51,420 (E + T 12,966 = boreal 2,062 + AuSmoke 10,904); C_EGT_v1 94,188 (G x2, T x2); D_adapt_v1 53,608 (`CORPUS_*_v1.json`). v4 added 329 negatives vs 27,538 positives (`DATA_CARD_V4.md` section 2).
- Dedup pass (`dedup_v4_meta.json`): 102,930 rows, ban 40,961 (all multinatsmoke: origin_class 40,959 + neardup 2); 16,493 eval frames hashed; firesense/mmodalfire/boreal 0 bans.
- Sealed v4 test (`EVALTEST_MANIFEST.json`, `LOCKBOX_V4_FROZEN.md`): boreal_test 1,764 / 1,762 pos / 2 neg (1 burn); forest_test 2,871 positives-only / 4 events; sdis_test 4,124 (917 neg) list sha 88ea60e2...; leak check PASS (0 path / 0 sha overlap vs 84,680 train/val images). Opened once -> boreal_test and forest_test are historical; sdis_test stays FINAL TEST per PROTOCOL section 2.
- Replay pool (`data/v4/sequences/REPORT.md`): 139 val sequences; negative pool 24 seqs / 763 frames (boreal 15 clips / 471 frames + FIRESENSE 9 / 292) ~0.21 h — the entire new-domain no-smoke footage on disk.
- MultiNatSmoke composition (`multinatsmoke_manifest.parquet`): 70,816 rows; usable 29,857 / quarantine_figlib 23,863 / dup_dfire 10,597 / dup_boreal 6,110 / quarantine_unknown 389; only 4 empty masks in the whole release (positives-only).
- Boreal (`boreal_manifest.parquet`): 6,714 rows = 4,954 Subset-A + 1,472 Subset-C + 288 Subset-B clips; Empty-Images 256 = heinola 217 / karkkila 37 / ruokolahti 2; Subset-B labels 273 smoke / 15 non-smoke (heinola 14, karkkila 1).

## 8. Limitations of this inventory

- Hours for pyro-sdis are derived from image timestamps (median cadence ~29-30 s); they are "sampled" time, not continuous video, and the neg-neg rule
  treats a 30 s interval between two unannotated frames as no-smoke time — the flicker rate says ~1/3 of sequences have inconsistent labels.
- Boreal Subset-A frames are 1 s apart within a flight but the manifest has only the burn date, so no valid-hours figure is given for stills.
- AuSmoke fps was never recorded (filenames suggest 1 s or 60 s per frame); no hours are stated.
- FIgLib remainder counts come from the HPWREN index (`figlib_events_index.csv`), not from downloaded files.
- FASDD_UAV class counts and provenance fractions come from the 2026-09-10 scout (60-image sample), not from a manifest.
- Licence strings are as recorded in the v4 cards/manifests; none were re-verified at source today. AuSmoke/MultiNatSmoke permission is PENDING, not "internal research OK".
- Discrepancy noted, not resolved: `dedup_v4_meta.json` has {len(dd['evalhit_units'])} `evalhit_units` keys (45 figlib + 8 smoke5k) while `reports/v4/DATA_CARD_V4.md` section 1.4 quotes 52; this file uses the direct artifact count.
- Other Day-1 agents (negative review, onset audit, MmodalFire review) may raise `verified_negatives`/`onset_truth` for specific rows; this file reflects artifacts present at generation time and should be regenerated (same script) when they land.
"""
open(OUT_MD, 'w').write(md)
print('rows', len(df)); print(by_p.to_string()); print(by_s.to_string())
print('wrote', OUT_CSV, OUT_MD)
