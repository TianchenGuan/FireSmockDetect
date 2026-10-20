# DATA CARD — iteration 4 (v4) data: acquired, built, split, sealed

Written 2026-09-14 from on-disk artifacts only; no model was run and no manifest, corpus or
other report was modified. Every number cites the file it was read from (paths relative to
`/local/home/tianrgua/cvproject/`). Timestamps are UTC (`created_utc` where the artifact has
one, otherwise file mtime). Sealed test package (`reports/v4/LOCKBOX_V4_FROZEN.md`,
`data/v4/eval/EVALTEST_MANIFEST.json`) did not exist when this card was written — see §6.

Scope: five candidate sources (DFS, FIRESENSE, MmodalFire, Boreal Forest Fire,
AusSmoke/MultiNatSmoke), one central dedup/contamination pass, two grouped holdout splits,
four training corpora + one strict source-only val, one temporal sequence set, and one frozen
val eval package. The existing frozen corpus E (pyro-sdis train + D-Fire, 38,454 lines) is
reused unchanged.

## 1. Sources

### 1.1 Summary table

| source | provenance | license (verbatim status) | downloaded | label semantics found | annotation origin | unique images | independent units | used in v4 corpora |
|---|---|---|---|---|---|---|---|---|
| DFS | github.com/siyuanwu/DFS-FIRE-SMOKE-Dataset; Wu et al. MTAP 2022 | **none found** (GitHub `license: None`; citation requested) | **0 bytes** — BLOCKED | (unverified, README) VOC boxes, classes fire/smoke/other | (unverified) labelme by authors | 0 (README claims 9,462) | 0 | no (`missing_sources: ["dfs"]` in B_G/C_EGT) |
| FIRESENSE | Zenodo 10.5281/zenodo.836749 (ITI-CERTH / Bilkent, FP7 FIRESENSE) | CC-BY-4.0 | 2 zips, 622,676,845 + 197,589,846 B, md5 verified | video-level pos/neg per task (flame, smoke) ONLY; no boxes/masks/frame labels | original-human (whole-video curation) | 1,752 frames @1 fps (49 videos) | 43 parent_units (49 videos, 6 same-site pairs merged) | smoke_neg 292 frames as negatives in G |
| MmodalFire | figshare 10.6084/m9.figshare.28804448.v1; Zhang et al. Sci. Data 2026 | CC BY 4.0 | MmodalFire.rar 4,929,210,010 B, md5 verified | session-level fire / non-fire via OSD clock -> XLSX intervals; smoke NEVER annotated | original-human (authors' experiment log) | 23,571 frames @1 fps (28 DVR videos, 33 h) | 38 sessions (36 with frames); 1 room, 1 camera, 1 week | none (all excluded: fire_only / no_usable_label) |
| Boreal Forest Fire | Fairdata/Etsin 1dce1023-493a-4d63-a906-f2a44f831898; Pesonen et al. Sci. Data 2025 (FGI) | CC BY 4.0 | 14,592 files, 125.1 GB, every file sha256-verified vs Metax | Subset A: YOLO boxes (single class smoke, "large box" style); Subset B: per-30 s-clip boolean; Subset C: boxes + SAM masks (+40 manual masks) | original-human (boxes, video labels, manual masks); model-assisted (1,417 SAM masks) | 4,963 (4,954 A + 9 unlabeled C extras; 1,463 C files are downsampled dups of A) | 4 burns (31 DJI recordings) | T: 2,025 pos + 37 neg (train burns); val/test held out |
| AusSmoke / MultiNatSmoke | HF hongjinzhao0615/MultiNatSmoke rev 5b7a4e0f; Li et al. WACV 2026 (ANU) | **unspecified** for aggregate and AuSmoke (no LICENSE on HF/GitHub; citation requested); per-subset: WSDataset MIT, FireSpot CC BY 4.0, D-Fire CC0, Boreal CC BY 4.0, AI-for-Mankind/Firecam CC BY-NC-SA 4.0, FIgLib HPWREN terms, SmokeSeg/Smoke5K/FESB/Forest_Fire unspecified | MultiNatSmokeDataset.zip 43,677,465,490 B (163,324 files, 42 GB) | one binary smoke mask per image; no fire class; only 4 empty masks in 70,816 -> positives-only | model-assisted 62,963 / original-human 7,602 (SmokeSeg, Smoke5K) / unknown 251 (FESB) | 70,427 image files, 70,001 unique sha256 | AuSmoke 29 videos; WSDataset 8 groups; FireSpot 29 site groups; FESB 67; FIgLib 391 events | G: 14,609 (WSDataset+FireSpot+FESB-usable); T: 10,904 AuSmoke train units |

Sources: `data/v4/manifests/{dfs,firesense,mmodalfire,boreal,multinatsmoke}_CARD.md`; manifest
parquets inspected with pandas (row/column counts below).

### 1.2 Per-source detail

**DFS** (`dfs_CARD.md`, 08:51Z). OneDrive link dead (`accessDenied`, dead since >=2025-06 per
issues #7/#8). Baidu Pan share alive, code `pnxx` accepted anonymously, folder enumerated
(`DFS.rar` 742.2 MB + `fireLabel.rar` 369.4 MB, bundled with unrelated Smoke100k), but anonymous
download never yields a link; needs a Chinese-phone Baidu account + netdisk client. Policy:
Baidu-only = blocker; no third-party mirrors used. No manifest parquet produced. Label
semantics NOT audited (no files). Contamination vs D-Fire plausible but uncheckable.

**FIRESENSE** (`firesense_CARD.md`, `firesense_manifest.parquet` 1,801 rows = 49 video + 1,752
frame rows; sha256 all unique). Per-class frames: flame_pos 339 / flame_neg 591 / smoke_pos 530 /
smoke_neg 292. Manifest: `label_type=videolabel` on all 1,801 rows; `has_smoke_label=True` on 13
rows (smoke_pos videos only), `has_fire_label=True` on 11 (flame_pos videos); all frame rows
False/False (no frame truth). Independent units 43 (merged pairs: fencedhill, lakeside,
courtyard, overpass_20100728, nightstreet, trainstation_20100531; fencedhill spans a smoke_pos
AND a flame_pos video). Confounders: clouds behind trees, reeds/foliage motion, flags, highway
traffic day+night, headlight glare, red text/objects, glitter bokeh, indoor people, snow.
Shortcut risks: burned-in "Live (FPS)" text, cyan detector contours (testpos05), watermarks
(UltimateChase.com, NIST), timestamp bars. Defect: testneg03.809 (Indeo IV50) decodes only
182/748 frames. Contamination: 0 filename overlap vs dfire/figlib/pyro-sdis; central pass
banned 0 rows (`dedup_v4_meta.json`).

**MmodalFire** (`mmodalfire_CARD.md`, `mmodalfire_manifest.parquet` 23,599 rows = 23,571 frames
+ 28 videos; 64 parent_units = 36 sessions + 28 DVR stems; 1 camera `ch01_room1`). Deposit is
ONE lab (paper describes two). `has_smoke_label=False` on all 23,599 rows; `has_fire_label=True`
on 19,690 (19,671 fire-session frames + 19 video rows); `label_type=none` on 7 session-free DVR
files. Frames by session_label: fire 19,671 / interference 3,660 (dry-ice 1,921 + water-mist
1,741) / blank 240. Confounders: humidifier plume (labeled non-fire), dry ice, people in frame,
pink/blue/purple lighting, dense-smoke whiteout, Chinese OSD + watermark on every frame. Two
occlusion sessions = lens covered (video-only). Contamination: none plausible (indoor lab,
2022-08, published 2026-03); 0 rows banned.

**Boreal** (`boreal_CARD.md`, `boreal_manifest.parquet` 6,714 rows = 4,954 A + 1,472 C + 288 B).
label_type: box 4,995 / mask 1,417 / videolabel 288 / none 14; origin: original-human 5,323 /
model-assisted 1,377 / unknown 14; `has_fire_label=False` everywhere (flames visible but never
annotated). Manifest rows per burn (`aerial_split_v4.json` frames_per_unit): evo 1,276 /
heinola 1,630 / karkkila 1,526 / ruokolahti 2,282. Empty-Images negatives 256 (Heinola 217 /
Karkkila 37 / Ruokolahti 2), reviewed smoke-free. Subset B clips: Evo 31 (all smoke), Heinola
125 (111/14), Karkkila 74 (73/1), Ruokolahti 58 (all smoke). 5 A frames + 9 C extras have
visible smoke but NO label (`label_type=none`; never negatives). Boxes are single "large box"
around the plume incl. background (4,670 images with exactly 1 box). Confounders: clouds/hazy
sky behind plumes, lake sun-glint, clearcuts, gravel roads. 94.8% of labeled A frames are
positive. Contamination: 0 filename overlap; MNS re-ships 6,110 Boreal frames -> banned as
`dup_boreal`.

**AusSmoke / MultiNatSmoke** (`multinatsmoke_CARD.md`, `multinatsmoke_manifest.parquet` 70,816
rows; 389 mask-only rows `image_present=False`). `origin_class`: usable 29,857 /
quarantine_figlib 23,863 / dup_dfire 10,597 / dup_boreal 6,110 / quarantine_unknown 389.
Subsets: AuSmoke 15,248 / FIgLib 12,435 / WSDataset 11,539 / D-Fire 10,525 / SmokeSeg 6,144 /
Boreal 6,110 / FireSpot 2,865 / aiformankind 2,191 / firecam 1,661 / SMOKE5K_real 1,458 /
Forest_Fire 389 / FESB-MLID 251. Official Train/Test split is frame-level within the same 29
AuSmoke videos; 306 rows (147 sha) byte-identical across Train and Test. Smoke-size median
fg fraction: AuSmoke 0.0030, FIgLib 0.0018 (small/faint) vs Boreal 0.17. `n_smoke_boxes` in this
manifest = connected-component count of the mask, NOT boxes (sum 1,189,044 — speckly masks,
90th pct 34 components). Confounders inside AuSmoke: smoke-machine unit
(`20240209_SVF_Arbo_SmokeMachine`, 127 frames), staged drone `TEST_RUN_SMOKE`, Axis PTZ OSD
banners, urban Canberra vistas. NO negatives anywhere in the dataset.

### 1.3 Blockers

| blocker | source | artifact |
|---|---|---|
| Dead OneDrive; Baidu needs account+client; no mirror | DFS | `dfs_CARD.md` |
| Zero smoke annotations (fire/non-fire session labels only) | MmodalFire | `mmodalfire_CARD.md`; manifest `has_smoke_label` all False |
| Unlicensed aggregate; AuSmoke license unspecified | MultiNatSmoke | `multinatsmoke_CARD.md` §License |
| Forest_Fire images not shipped (Kaggle), FESB no license | MultiNatSmoke | `multinatsmoke_CARD.md` |
| No negatives in MultiNatSmoke; 15 negative clips + 256 negative frames in Boreal only | MNS, Boreal | cards |

### 1.4 Contamination findings (central pass, `dedup_v4_meta.json` 12:05:25Z, `dedup_v4.parquet`)

- Pass over 102,930 rows: keep 61,969 / ban 40,961. All bans are multinatsmoke: `origin_class`
  40,959 (quarantine_figlib 23,863 + dup_dfire 10,597 + dup_boreal 6,110 + quarantine_unknown
  389) + `neardup_v4` 2. firesense/mmodalfire/boreal: 0 bans. `group_banned_rows` 0.
- Exact sha256 checked for ALL rows against 16,493 eval frames (FIgLib subset 3,056 + lockbox
  1,978 + pyro-sdis val/test) and 43,704 hashed train-material files; phash restricted to the
  36,259 train-eligible keep rows.
- **FIgLib eval leak**: `multinatsmoke_CARD.md` — MNS FIgLib+SmokeSeg cover 404 FIgLib events;
  **65 of our 89 opened eval events** appear, 2,626 frames of which 1,929 are byte-identical to
  local eval frames. `dedup_v4_meta.json` lists 52 `evalhit_units` with an exact-dup eval frame
  (FIgLib, firecam, SMOKE5K_real units; hits in both figlib_subset and figlib_lockbox). The whole
  FIgLib-family (FIgLib, SmokeSeg, aiformankind, firecam, SMOKE5K_real) is quarantined at
  subset level, not just per matching sha, because crops/re-encodes evade sha matching.
- **D-Fire duplicates**: 10,525/10,525 MNS D-Fire images byte-identical to local D-Fire raw
  (+46 FESB-MLID, +26 SMOKE5K_real) -> `dup_dfire`, 10,597 banned (D-Fire is already in E).
- **Near-dups (phash)**: 2 rows banned — AuSmoke
  `17240209_IanTanner_Cleland_Prescribed_Burn_MVI_3592_000975-+000974.jpg` and `_000977-+000976.jpg`
  at Hamming distance 6 to boreal train frame `Evo-Images/evoDJI_0001_frame55.jpg`
  (`dedup_v4.parquet` reason `neardup_v4:...:h=6`). An Australian ground-camera frame matching a
  Finnish drone frame is almost certainly a phash collision on low-texture content, but both
  rows were banned conservatively; effect on T is -2 frames.
- Internal: 426 exact-duplicate files inside MNS (70,427 files / 70,001 sha); Boreal in MNS has
  two filename families of the same videos (1,416 stems in both, different bytes).
- Iteration-4 negatives contamination: firesense smoke_neg frames are BOTH the only continuous
  negative replay pool (`replay_val_v4.jsonl`) AND background negatives inside B_G_v1 (292
  lines), C_EGT_v1 (584 = x2) and D_adapt_v1 (39) -> false-alert numbers on that pool are
  optimistic for arms B/C/D_adapt (`data/v4/sequences/REPORT.md`, `EARLY_WARNING_V4.md` §3C).

## 2. Label policy (what became a positive, a negative, or was excluded)

Rule source: `scripts/v4/build_corpus.py::train_eligibility` and `mask_to_lines`; counts from
`CORPUS_*_v1.json` `excluded` / `counts.per_label_origin`. Principle: **missing labels are not
negatives**; only an author-verified empty annotation or a video-level negative may be a
negative.

| rows | disposition | reason / rule | count (artifact) |
|---|---|---|---|
| boreal Subset-A/C rows with `label_type=box`, `has_smoke_label=True`, train burns | POSITIVE `box-yolo` | YOLO txt copied | 2,025 (E_ET, C_EGT `per_label_origin.box-yolo`) |
| AuSmoke train-unit mask rows | POSITIVE `mask-derived` | mask>=128 -> 8-conn components; min_area max(9, 2e-5*W*H), sides>=2 px; closing if >8 comps; <=50 boxes largest-first | 10,904 (E_ET `per_source.multinatsmoke`); 2 more banned as near-dups |
| WSDataset + FireSpot + FESB-usable mask rows | POSITIVE `mask-derived` (G) | same converter | 14,609 (B_G `per_source.multinatsmoke`) |
| firesense `frames/smoke_neg/*` | NEGATIVE `background_neg` | video-level negative => every frame negative (9 videos verified by eye) | 292 (B_G, E_ET absent, C_EGT 292 x2) |
| boreal Empty-Images in train burns (Karkkila 37) | NEGATIVE `background_neg` ONLY when `--verified-empty-negatives` | annotation file exists with zero boxes = author-verified empty | 37 (E_ET/C_EGT `verified_empty_negatives: true`; B_G false, boreal not in G) |
| firesense smoke_pos / flame_pos / flame_neg frames | EXCLUDED `no_usable_label` | video-positive frames carry no frame truth; flame_neg not smoke-verified frame-by-frame | 1,460 (part of 5,360 in B_G) |
| mmodalfire fire-session frames | EXCLUDED `fire_only` | has_fire_label without smoke label | 19,671 (B_G, C_EGT) |
| mmodalfire interference + blank frames | EXCLUDED `no_usable_label` | smoke never annotated; not treated as negatives | 3,900 (3,660 + 240; part of 5,360) |
| boreal `label_type=none` rows in train burns | EXCLUDED `no_usable_label` | visible smoke, missing label | 7 (E_ET) |
| boreal Subset-C rows in train burns | EXCLUDED `convert:mask_not_found` | converter did not resolve a mask; Subset C = downsampled re-releases of Subset A, so each frame enters once via its A copy (converter outcome, not an explicit dedup rule) | 628 (= evo 314 + karkkila 319 - 5 unlabeled extras) |
| video rows (.avi/.mp4) | EXCLUDED `non_image_file` | not images | 77 (B_G: 49 firesense + 28 mmodalfire); 105 (E_ET: evo 31 + karkkila 74 clips); 182 (C_EGT) |
| boreal heinola + ruokolahti rows | EXCLUDED `aerial_holdout` | val/test burns | 3,912 (1,630 + 2,282) |
| AuSmoke val/test units | EXCLUDED `forest_holdout` | held-out groups | 4,342 (1,471 + 2,871) |
| MNS quarantine/dup rows | EXCLUDED `dedup:origin_class` | §1.4 | 40,959 |
| MNS near-dups | EXCLUDED `dedup:neardup_v4` | §1.4 | 2 |
| DFS | absent | blocked | 0 |

Consequence: v4 contributes 329 negatives in total (292 firesense + 37 boreal) against 27,538
new positives; every new-domain camera (AuSmoke, WSDataset, FireSpot) enters with ZERO
negatives. See §5 for why this mattered.

## 3. Corpus versions

Arm/component naming: E = existing frozen corpus; G = generic/non-target smoke (WSDataset,
FireSpot, FESB-usable, firesense smoke_neg); T = target domains (Boreal train burns, AuSmoke
train units). D_pre = stage-1 pretrain on G, D_adapt = stage-2 on E+T with G replay.

| corpus | created_utc | lines | composition (multiplicity) | per source | boxes (v4 part) | `lists/local/train.txt` sha256 | tar bytes / sha256 |
|---|---|---|---|---|---|---|---|
| **B_G_v1** | 2026-09-14T10:40:38Z | 14,901 | G x1 | firesense 292 (neg) + multinatsmoke 14,609 | 16,794 | `d3e28471a77695f099cc38bdecaff1459e8771dc46b9a42d39e518f6b4e85f60` | 3,373,783,040 / `65adf8ca…400b694` |
| B_G_v1 gval addendum | 2026-09-14T10:56:35Z | train90 14,591 + gval 310 | parent_unit md5-bucket split, 105 units, target 0.10 | — | — | train90 `794de869…40641`, gval `a452ca82…8225` | — |
| **E_ET_v1** | 2026-09-14T12:06:26Z | 51,420 | E 38,454 x1 + T 12,966 x1 | existing 38,454; boreal 2,062 (2,025 pos + 37 neg); multinatsmoke 10,904 | 16,993 | `113ef7e1209ce73b3bf07b82627b9bd567fa8059a463b10d2de539f6da7728cb` | 13,949,634,560 / `88094055…549c91` |
| **C_EGT_v1** | 2026-09-14T12:07:18Z | 94,188 | E 38,454 x1 + G 14,901 **x2** (29,802) + T 12,966 **x2** (25,932) | existing 38,454; firesense 292; multinatsmoke 25,513; boreal 2,062 (raw, before x2) | 33,787 | `e3d322a40bcacc76760cbf162096ac50eca0b043697c151e1680955596bfbb90` | 17,332,858,880 / `f64825d2…68641` |
| **D_adapt_v1** | 2026-09-14T14:09Z (mtime) | 53,608 | E_ET 51,420 + G replay 2,188 (0.15 x train90 14,591) | as E_ET + replay | — | `a8f716281c0d5d922abaf2dc7ef1b226862b0c4303b0a2abe5174358961ad87c` | D_adapt_v1.tar 5,539,840 (lists only) |

Sources: `data/v4/manifests/CORPUS_B_G_v1.json`, `CORPUS_B_G_v1_gval_addendum.json`,
`CORPUS_E_ET_v1.json`, `CORPUS_C_EGT_v1.json`, `CORPUS_D_adapt_v1.json`; tar mtimes
`data/v4/stage/`.

Notes:
- **E is the frozen expB train list, unchanged: 38,454 lines** (pyro-sdis train sites
  brison/serre-de-barre/courmettes/valbonne 22,177 + D-Fire 16,277; `data/manifests/splits_frozen.json`,
  verified in `reports/v4/RECONCILIATION_V4.md`). `train_E.txt` sha256
  `51ac91f791de57211d0f31c45365c047230d30ff5f8a4869c71b3891a05f358d` is identical in E_ET_v1 and
  C_EGT_v1. `train_G.txt` differs between B_G_v1 and C_EGT_v1 only because lines carry
  corpus-specific absolute image paths (both n=14,901).
- All builds seed 0; `box_count_mismatches.n = 0` in every corpus; val yaml for training points
  at the existing pyro-sdis val list (`sdis_val.txt`).
- B_G_v1 was built against an earlier `dedup_v4.parquet` (sha `82ad1914…deb36`); E_ET/C_EGT
  reference the current one (sha `98b03ab6…40a0`, regenerated 12:05Z). The delta is the 2
  AuSmoke near-dups, which are not in G.
- D_adapt_v1: stage 1 = `expV4Dpre_g_s0` (61 ep on B_G_v1 train90, ~0.89M exposures), stage 2 =
  26 ep on the 53,608-line list. Replay fraction 0.15 of train90.
- COCO conversion of E_ET_v1 (`data/v4/coco/CONVERSION_STATS_eet.json`, 14:51Z): 51,420 images,
  48,480 annotations; E: 31,487 boxes, 11,257 empty-label negatives, 7 boxes <1 px dropped; T:
  16,993 boxes, 37 negatives, 0 dropped. `instances_train_eet.json` sha `d1467a0e…a697`.
- `target_subsources` recorded as AuSmoke, Boreal, Forest_Fire in every corpus json;
  Forest_Fire contributed 0 rows (quarantine_unknown, images not shipped).

## 4. Splits

### 4.1 Aerial (boreal, burn-level) — `data/v4/manifests/aerial_split_v4.json` / `.txt` (12:05:28Z)

Policy: split by ENTIRE parent_unit, greedy on unit count, seed 0, ratio 0.6/0.2/0.2; >=1 unit
forced into val and test.

| split | burns | manifest rows | images used |
|---|---|---|---|
| train | burn_evo (2022-05-24), burn_karkkila (2022-07-06) | 1,276 + 1,526 = 2,802 | 2,062 in T (2,025 pos + 37 neg) |
| val | burn_heinola (2022-06-28) | 1,630 | 1,123 (906 pos + 217 neg) in boreal_val |
| test | burn_ruokolahti (2022-08-15) | 2,282 | sealed (§6) |

Independent events: train 2 / **val 1 / test 1**. The artifact itself carries
`tiny_n_warning`: "only 4 independent parent_units; 60/20/20 by unit count is statistically
weak ... Treat aerial val/test numbers as anecdotal." Each burn is one day, one weather, one
camera; frames 1-2 s apart.

### 4.2 Forest holdout (AuSmoke, manual burn-group split) — `data/v4/manifests/forest_holdout_v4.json` (11:53Z)

29 AuSmoke units were grouped by hand into 16 burn/date/camera groups because several units are
the SAME burn seen from different cameras. A naive per-unit hash split would have put the two
cameras of one burn on opposite sides for: FB769 Bonshaw (3 units: S1020005, 202403251112,
202403261044_Panasonic), FB867 Black Mountain (2), FB746 Goorooyaroo (2), FB902 Yarralumla (2),
2024-10-21 (2), 2024-04-12 Sony/Panasonic (2), FB389 (2), and the DJI 2023-09-22 flight (3
videos). The three `aus_000xx` units of unknown provenance were kept as ONE conservative group
(train); the two Axis Q6215LE fixed-camera units share a static viewpoint -> one group.

| split | groups | units | frames | independent events |
|---|---|---|---|---|
| train | 9: g_unknown00, g_20240412, g_20240209_cleland, g_20240209_smokemachine, g_20240404_fb389, g_20231010_demo, g_20240306_fb746, g_20240308_fb902, g_20240328_fb867 | 16 | 10,906 (10,904 after near-dup ban) | <=9 (aus_000xx unknown) |
| val | 3: g_axis_fixedcam (2 burns, one fixed camera), g_20240903, g_dji_0792 | 4 | 1,471 | 3 (report axis pair as "same-camera") |
| test | 4: g_dji_20230922 (3 videos = ONE aerial flight incl. staged `TEST_RUN_SMOKE`), g_20241021, g_20240326_fb769, g_20240517_fb884 | 9 | 2,871 | **4** (3 ground burns + 1 aerial flight) |

`g_20240209_smokemachine` (synthetic-source smoke in a real scene) is train-only, never eval.
The DJI flight's TEST_RUN video is to be reported separately (`PREREGISTRATION_V4.md`).

### 4.3 Small-n warnings (applies to every number derived from these splits)

boreal val = 1 burn, boreal test = 1 burn; forest val ~3 events; forest test 4 events, one of
which is an aerial flight containing a staged test run. Denominators for negatives: boreal val
217 frames (Heinola Empty-Images), 14 heinola non-smoke clips (+1 karkkila clip from a TRAIN
burn), firesense 9 negative videos; forest val/test have 0 negatives (positives-only source).

## 5. Sequences and onset — `data/v4/sequences/{REPORT.md,ONSET_AUDIT.md}` (13:36Z), `sequences_v4.parquet`

| source | split | sequences | frames | frame truth | onset observed |
|---|---|---|---|---|---|
| ausmoke | train | 22 | 10,905 | box (mask-derived) | 0 / 22 |
| ausmoke | val | 4 | 1,471 | box | 0 / 4 |
| boreal | train | 105 | 3,370 (@1 fps) | videolabel (104 smoke / 1 no-smoke) | not invented |
| boreal | val | 125 | 3,871 | videolabel (111 / 14) | not invented |
| firesense | train | 13 | 530 | videolabel (13 smoke_pos) | not invented |
| firesense | val | 9 | 292 | videolabel (9 smoke_neg) | n/a |
| **total** | | **278** | **20,439** | | |

- **0 of 26 AuSmoke sequences observe a background->smoke onset.** Every shipped frame has a
  non-empty mask; filenames `<unit>_<fidx>-<+offset>` have offset >= 0 everywhere (offset 0 in
  14/20 kept units), i.e. segments were trimmed at/after first visible smoke. Boreal and
  FIRESENSE positives are video-level only and no frame onsets were fabricated. **v4 adds no
  observed-onset supervision**; onset signal still comes only from v3 pyro-sdis "appears"
  sequences and FIgLib replay.
- AuSmoke label stability: presence flicker 0% (structural — positives-only, NOT comparable to
  v3 pyro-sdis 29%/36%); mask-area >2x frame-to-frame jumps 6.9% of 12,350 adjacent pairs, 25/26
  sequences (96.2%) — the honest instability metric.
- Val replay package `replay_val_v4.jsonl`: 139 sequences; negative-footage pool 24 sequences /
  763 frames (~0.21 h): boreal non-smoke 15 clips / 471 frames + firesense smoke_neg 9 / 292.
- **Data-quality lesson from `reports/v4/EARLY_WARNING_V4.md` (domain shortcut).** Temporal
  heads trained with the v4 extension (10,905 AuSmoke frames, ALL positive, from a new camera
  domain with zero negatives from that domain; 98% of added frames positive) fire on **43-47%
  of boreal non-smoke frames and 83-98% of firesense negative frames** at thresholds where
  v3-only heads fire on 0% (40-90 false episodes/h vs 0). A capped ablation (5 sequences /
  2,571 frames) shows dose-response: mean head score on v4 replay frames 0.13-0.21 (v3-only) ->
  0.48-0.64 (capped) -> 0.73-0.89 (full). "Looks like a non-pyro-sdis camera" became the smoke
  cue; pyro-sdis val selection could not see it. Conclusion for data acquisition: **all-positive
  new-domain frames without matched negatives from the same domain are harmful**, and the
  tiny v4 replay negatives (7 min boreal, 5 min firesense) were the only thing that caught it.

## 6. Frozen evaluation packages

### 6.1 Val — `data/v4/eval/EVALVAL_MANIFEST.json` (15:36Z)

| set | images | positives | negatives | boxes | COCO json sha256 |
|---|---|---|---|---|---|
| boreal_val (burn_heinola) | 1,123 | 906 | 217 | 917 | `3140ebceaec791ab4b96ba76f5dcce6a63eb0382f6a5083905cb2c077b83265c` |
| forest_val (AuSmoke 4 units) | 1,471 | 1,471 | 0 | 1,975 | `c5ccc4929622fa7d381b2125679edfd7e419beb3aa352a5f4472f6f5945f3bd9` |

Excluded from boreal_val: 382 Subset-C duplicates of Subset-A frames, 125 Subset-B mp4 clips.
Lists: `data/v4/eval/{boreal_val,forest_val}/lists/local/val.txt` (1,123 / 1,471 lines); yamls
in `data/v4/eval/yamls/`. Package tar `data/v4/stage/v4_evalval.tar` 3,207,987,200 B, sha256
`25e0b55507052bfd1d1794ff93e857eb51028c4a37128cc958cdd704e73f36ab`. forest_val has no negatives,
so forest thresholds must be set on boreal_val negatives (`PREREGISTRATION_V4.md`).

### 6.2 Sealed test package

Not present at write time (`reports/v4/LOCKBOX_V4_FROZEN.md`, `data/v4/eval/EVALTEST_MANIFEST.json`
absent; being built by another agent). **Sealed test package: see LOCKBOX_V4_FROZEN.md.** Per
`reports/v4/PREREGISTRATION_V4.md` it will contain: pyro-sdis TEST (ferion + croix-augas, frozen
since iteration 1), boreal TEST = burn_ruokolahti (2,282 manifest rows, ONE burn), AuSmoke forest
TEST = 9 units / 4 events / 2,871 frames (positives-only; TEST_RUN video reported separately),
and FIgLib-39 replay labelled historical. Evaluated once at iteration close.

## 7. Known limitations and v5 acquisition priorities

Limitations of the v4 data as built:
1. Negatives are nearly absent from the new domains: 329 v4 negatives total (292 firesense
   staged-scene, 37 boreal aerial) vs 27,538 v4 positives; AuSmoke/WSDataset/FireSpot have 0.
   This produced the §5 domain shortcut and makes matched-FP evaluation impossible on forest.
2. No observed onsets (0/26) — v4 cannot improve time-to-detect supervision.
3. Independent-event counts are tiny: boreal 4 burns (1 val, 1 test), forest test 4 events;
   MmodalFire is one room/one camera; FIRESENSE 43 units of 2005-2011 low-res footage.
4. Labels are loose: boreal "large box" style, AuSmoke masks generously oversized on faint
   plumes, SAM/pseudo-label origin for 62,963 MNS masks, speckly masks (converter closes and
   caps at 50 boxes). MmodalFire has no smoke labels at all and contributed nothing.
5. Licensing: AuSmoke and the MultiNatSmoke aggregate are unlicensed (citation requested only);
   FESB/Forest_Fire unspecified; DFS unlicensed and unobtainable.
6. Shortcut artifacts remain in-frame: FIRESENSE overlays/watermarks, MmodalFire OSD, Axis PTZ
   banners in AuSmoke; boreal and MNS-Boreal share content (banned in MNS, but keep in mind).
7. The firesense smoke_neg pool is simultaneously a training negative (B/C/D_adapt) and the
   replay false-alert pool -> optimistic for those arms.
8. The eval-side FIgLib leak inside MultiNatSmoke means any external model trained on
   MultiNatSmoke is contaminated for our FIgLib evals (as pyronear's baseline already was).

What a v5 acquisition should prioritize (in order):
1. **Negative footage from the new domains** — smoke-free aerial/drone flights over forest,
   Australian bush fixed-camera footage without smoke, matched in camera, season and
   viewpoint to the positives. This is the single highest-value item (EARLY_WARNING_V4 §0.6).
2. **Real onset sequences** — untrimmed recordings that start before first visible smoke,
   with per-frame truth (or at least a human-marked onset frame), for both aerial and ground
   cameras; the boreal Subset-B positive clips are a candidate for manual onset annotation.
3. **DFS via a human Baidu download** of `DFS.rar` (742.2 MB) + `fireLabel.rar` (369.4 MB) from
   the live share into `data/v4/raw/dfs/`, followed by license/legal review and a D-Fire
   near-dup check before any use; its `other` confounder class is the reason it was shortlisted.
4. **AuSmoke license confirmation** with the authors (contact in `multinatsmoke_CARD.md`)
   before any use beyond the internal prototype; same for FESB-MLID.
5. More independent burns/events per domain (boreal-style UAV data from other operators) so
   val/test can hold >1 event each; tighter instance boxes or mask re-annotation for the
   faint-plume subsets used as the primary objective.
