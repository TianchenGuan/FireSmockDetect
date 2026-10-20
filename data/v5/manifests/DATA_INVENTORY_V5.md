# DATA INVENTORY BY PURPOSE — V5 (Day 1)

Generated 2026-10-02 02:47 UTC by `scripts/v5/make_data_inventory_v5.py` (CPU only; reads manifests/lists, opens no test image).
Companion CSV: `data/v5/manifests/DATA_INVENTORY_V5.csv` (60 rows; one row per source x unit-group / site / burn / video group;
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

| purpose_v5 | rows | images / frames | positives | verified negatives (frames) | verified-negative VIDEO hours (valid) |
|---|---|---|---|---|---|
| train | 9 | 43888 | 32560 | 11326 | 12.95 |
| calibration/val | 6 | 12329 | 10515 | 1814 | 4.494 |
| final-test | 3 | 4124 | 3207 | 917 | 3.758 |
| historical | 8 | 13828 | 7628 | 5747 | 58.838 |
| aux-train-candidate | 5 | 24162 | 0 | 0 | 0.0 |
| permission-pending | 19 | 29857 | 29857 | 0 | 0.0 |
| quarantine | 10 | 68008 | 55414 | 0 | 0.0 |

Notes on the totals: `images / frames` sums stills and extracted video frames and double-counts nothing on disk (Subset-C duplicates
and MultiNatSmoke re-ships are in the quarantine row only). `final-test` includes the FIgLib unopened remainder as a CANDIDATE pool with
0 frames on disk (see section 3). The `permission-pending` positives were used in v4 (disclosed) and may not be extended.

## 2. Totals per scenario (PROTOCOL section 1: fixed-camera and aerial are calibrated and reported separately)

| scenario | rows | images / frames | positives | verified negatives (frames) | verified-negative VIDEO hours (valid) |
|---|---|---|---|---|---|
| fixed-camera | 19 | 80711 | 71018 | 9239 | 79.909 |
| aerial | 14 | 20230 | 19443 | 727 | 0.131 |
| forest-ground | 13 | 13947 | 13947 | 0 | 0.0 |
| indoor | 5 | 23571 | 0 | 0 | 0.0 |
| web | 9 | 57737 | 34773 | 9838 | 0.0 |

### 2a. Verified negatives — scenario x purpose (frames)

| scenario | train | calibration/val | final-test | historical | aux-train-candidate | permission-pending | quarantine |
|---|---|---|---|---|---|---|---|
| fixed-camera | 3424 | 1158 | 917 | 3740 | 0 | 0 | 0 |
| aerial | 69 | 656 | 0 | 2 | 0 | 0 | 0 |
| forest-ground | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| indoor | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| web | 7833 | 0 | 0 | 2005 | 0 | 0 | 0 |

### 2b. Verified-negative VIDEO hours (valid observed time) — scenario x purpose

| scenario | train | calibration/val | final-test | historical | aux-train-candidate | permission-pending | quarantine |
|---|---|---|---|---|---|---|---|
| fixed-camera | 12.941 | 4.372 | 3.758 | 58.838 | 0.0 | 0.0 | 0.0 |
| aerial | 0.009 | 0.122 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forest-ground | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| indoor | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| web | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Reading guide for the hours: pyro-sdis hours are neg-neg intervals at ~29-30 s sampled cadence (`pyro_sdis_manifest.parquet` datetimes, gap rule 3x median);
FIgLib hours are pre-onset time at 60 s cadence (historical only); Boreal/FIRESENSE hours are 1 fps extracted clip time. Still-image
negatives (Boreal Subset-A 217/37/2, D-Fire 7,833) carry no hours because their frames are not contiguous.

### 2c. Verified no-smoke footage per scenario, by source (what the calibration/acceptance can actually use)

| scenario | source / unit group | purpose | units | verified negatives | valid hours; cadence | evidence | artifact |
|---|---|---|---|---|---|---|---|
| fixed-camera | pyro-sdis VAL sites cabanelle + marguerite | calibration/val | 8 cameras / 2 sites | 1158 frames | 4.37 h neg-neg (43.79 h all-frames); ~30 s sampled | human-annotated (box absence) | pyro_sdis_manifest.parquet + splits_frozen.json |
| fixed-camera | pyro-sdis VAL all-negative v3 sequences | calibration/val | 12 seqs | 219 frames | 1.47 h recomputed (EARLY_WARNING_V4 quotes 1.521 h); 29 s cadence | human-annotated | data/v3/sequences/sequences_val.json; reports/v4/EARLY_WARNING_V4.md |
| fixed-camera | pyro-sdis TEST sites ferion + croix-augas | final-test | 11 cameras / 2 sites | 917 frames | 3.76 h neg-neg; ~30 s | human-annotated | same |
| fixed-camera | pyro-sdis TRAIN sites (4) | train | 21 cameras | 3424 frames | 12.94 h neg-neg; ~30 s | human-annotated | same |
| fixed-camera | FIgLib-39 + lockbox v2 + v3 pre-onset frames | historical | 89 events | 3448 frames | 58.8 h pre-onset; 60 s | human-annotated (HPWREN onset label) | figlib manifests |
| fixed-camera-like | FIRESENSE smoke_neg | historical (replay only) | 9 videos | 292 frames | 0.079 h; 1 fps | video-level label + eye audit | firesense_manifest.parquet; frame_extraction_summary.json |
| aerial | Boreal heinola Subset-A Empty-Images | calibration/val | 1 burn (9 flights) | 217 frames | n/a (sampled stills) | author-verified empty | boreal_manifest.parquet; EVALVAL_MANIFEST.json |
| aerial | Boreal heinola Subset-B non-smoke clips | calibration/val | 1 burn, 14 clips | 439 frames | **0.122 h**; 1 fps | video-level label | sequences_v4.parquet |
| aerial | Boreal karkkila Empty-Images + clip 44 | train | 1 burn | 37 + 32 frames | 0.009 h; 1 fps | author-verified empty / video-level | boreal_manifest.parquet; sequences_v4.parquet |
| aerial | Boreal ruokolahti Empty-Images | historical | 1 burn | 2 frames | n/a | author-verified empty | EVALTEST_MANIFEST.json |
| aerial | AuSmoke DJI groups (g_dji_20230922 test, g_dji_0792 val) | permission-pending | 2 flights | 0 | unknown fps | none (positives-only) | multinatsmoke_manifest.parquet |
| forest-ground | AuSmoke ground groups | permission-pending | 13 groups | 0 | unknown fps | none (positives-only) | multinatsmoke_manifest.parquet |
| indoor | MmodalFire water-mist / dry-ice / blank | aux-train-candidate | 10 sessions | 0 until human review (1.08 h candidate) | 1 fps | video-level label (non-fire) | mmodalfire_manifest.parquet |

## 3. Onset truth available (PROTOCOL section 3 early-warning metric; section 6 acceptance = >=10 independent events with REVIEWED intervals)

| scenario | source | events | truth type | status for v5 | artifact |
|---|---|---|---|---|---|
| fixed-camera | FIgLib-39 replay | 39 | author first-visible-plume (60 s cadence) | HISTORICAL: models v1-v4 already run; usable only as the >=30/39 calibration constraint, never as fresh test | figlib_subset_manifest.parquet |
| fixed-camera | FIgLib lockbox v2 / v3 | 25 / 25 | author first-visible-plume | HISTORICAL (opened once each) | lockbox manifests |
| fixed-camera | FIgLib unopened remainder | 436 (index) | author first-visible-plume | CANDIDATE final-test pool; 0 on disk; needs download, human interval review blind to model outputs, and a dated protocol amendment before assignment | figlib_events_index.csv |
| fixed-camera | pyro-sdis VAL appears-type v3 sequences | 36 seqs | box presence flips at 29 s cadence, 36% flicker | PARTIAL: candidates for human interval review (Day-1 flicker-audit agent); not reviewed intervals | sequences_val.json |
| fixed-camera | pyro-sdis TRAIN appears-type v3 sequences | 136 seqs | same, 29% flicker | PARTIAL: training-side only (never eval) | sequences_train.json |
| aerial | Boreal Subset-B smoke clips (train burns 104, val burn 111) | 0 observed | video-level only; 30 s clips | NO onset truth; transitions inside clips under limited audit; cannot give first-visible time of the burn | ONSET_AUDIT.md |
| aerial / forest-ground | AuSmoke 26 sequences | 0 of 26 | n/a (all frames positive, trimmed at/after first visible smoke) | NONE | ONSET_AUDIT.md |
| indoor | MmodalFire fire sessions | 26 sessions | session start time (XLSX, +/-30 s); smoke visibility unlabelled | not a forest/UAV onset source; sensor CSVs are audit aids only | mmodalfire_CARD.md |

**Reviewed onset intervals today: 0 fixed-camera, 0 aerial (acceptance needs >=10 independent events, final-test onsets labelled without seeing candidate outputs).**

## 4. PERMISSION PENDING (PROTOCOL section 2; no new training use until written confirmation; v4 use disclosed, not extended)

| source | unit group | units | frames | scenario | licence status |
|---|---|---|---|---|---|
| multinatsmoke/AuSmoke | g_unknown00 [train] units=3 | 3 | 2074 | forest-ground | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_20240412 [train] units=2 | 2 | 953 | forest-ground | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_20240209_cleland [train] units=1 | 1 | 1089 | forest-ground | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_20240209_smokemachine [train] units=1 | 1 | 127 | forest-ground | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_20240404_fb389 [train] units=2 | 2 | 1161 | forest-ground | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_20231010_demo [train] units=1 | 1 | 487 | forest-ground | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_20240306_fb746 [train] units=2 | 2 | 2738 | forest-ground | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_20240517_fb884 [test] units=1 | 1 | 404 | forest-ground | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_20241021 [test] units=2 | 2 | 946 | forest-ground | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_20240308_fb902 [train] units=2 | 2 | 1007 | forest-ground | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_20240328_fb867 [train] units=2 | 2 | 1270 | forest-ground | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_20240903 [val] units=1 | 1 | 428 | forest-ground | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_20240326_fb769 [test] units=3 | 3 | 1263 | forest-ground | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_axis_fixedcam [val] units=2 | 2 | 848 | fixed-camera | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_dji_20230922 [test] units=3 | 3 | 258 | aerial | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/AuSmoke | g_dji_0792 [val] units=1 | 1 | 195 | aerial | unspecified (no LICENSE on HF/GitHub; citation requested) -> PERMISSION PENDING |
| multinatsmoke/WSDataset | WSDataset usable (G component) | 8 | 11539 | fixed-camera | MIT per MultiNatSmoke README (kaggle gloryvu/wildfire-smoke); aggregate unlicensed -> PERMISSION PENDING per PROTOCOL s2 |
| multinatsmoke/FireSpot | FireSpot usable (G component) | 29 | 2865 | fixed-camera | CC-BY-4.0 per MultiNatSmoke README (Biometrix-4/FireSpot-CNX); aggregate unlicensed -> PERMISSION PENDING per PROTOCOL s2 |
| multinatsmoke/FESB-MLID | FESB-MLID usable (G component) | 59 | 205 | fixed-camera | unspecified (wildfire.fesb.hr, no licence stated) -> PERMISSION PENDING |

Summary: AuSmoke 29 units / 15248 frames (16 burn-grouped units: 9 train / 3 val / 4 test groups) — no licence anywhere (HF/GitHub);
MultiNatSmoke non-AuSmoke "usable" (G): WSDataset 11539 (MIT per README), FireSpot 2865 (CC BY 4.0 per README),
FESB-MLID 205 (unspecified). The aggregate MultiNatSmoke release carries no licence; original-source licences are quoted from its README and were NOT
verified at source. These sets contribute 0 negatives and 0 onsets regardless of permission. Email draft to the AuSmoke/MultiNatSmoke authors is a separate Day-1 deliverable (user review before sending).

## 5. QUARANTINED (excluded from every model-facing use) and why

| source | unit group | frames | reason (abridged in notes) |
|---|---|---|---|
| boreal | Subset-C (downsampled re-release of Subset-A + SAM masks) | 1472 | 1472 rows: 1,463 are downsampled duplicates of Subset-A frames (excluded from corpora/eval as subsetC_duplicate_of_subsetA: val 382, test 453) + 9 unlabeled extra frames with visible smoke (label_type none). Masks are SAM model-assisted (1,417) + 40 manual (test). Quarantined to prevent double counting / A-vs-C split leakage; masks may be reused for the SAME frames only. |
| firesense | smoke_pos (13 videos) | 530 | Video-level positive only: smoke not visible in every frame, no boxes; mostly staged near-field smoke pots/grenades, one NIST structure burn, burned-in overlays/detector contours (shortcut risk). Quarantined from frame-level training/eval (no frame truth, domain mismatch); not an onset source (no label of first visibility). |
| firesense | flame_pos (11 videos) | 339 | Flame-positive videos with UNLABELLED smoke (bus fire, crown fire, trash fire); watermarks (UltimateChase.com). No smoke truth -> neither positive nor negative for smoke. |
| mmodalfire | occlusion sessions (2, video only) + raw DVR files | 0 | 28 raw DVR MP4s. 2 occlusion sessions = lens covered (no usable video). Footage outside XLSX intervals is unlabelled idle room / setup / smoke clearing -> residual smoke possible -> NOT negatives. |
| multinatsmoke/FIgLib-family | FIgLib + SmokeSeg + aiformankind + firecam + SMOKE5K_real (quarantine_figlib) | 23863 | EVAL LEAK: 65 of our 89 opened FIgLib eval events appear (1,929 byte-identical frames); dedup_v4_meta lists 53 evalhit_units with exact-dup eval frames. Subset-level quarantine (crops/re-encodes evade sha). Subsets: FIgLib 12435, SmokeSeg 6144, aiformankind 2191, firecam 1661, SMOKE5K_real 1432. Also non-commercial licences on two subsets. |
| multinatsmoke/D-Fire-dup | D-Fire + FESB(46) + SMOKE5K(26) byte-identical to local D-Fire (dup_dfire) | 10597 | Already in E via local D-Fire; quarantined to avoid double counting. MNS adds SAM-from-box masks for these images (possible extra supervision on the SAME training images only). |
| multinatsmoke/Boreal-dup | Boreal re-shipped inside MultiNatSmoke (dup_boreal, 30 DJI videos) | 6110 | Byte-identical Boreal copies incl. heinola (VAL) and ruokolahti (TEST; 2,215 rows, 1,748 in MNS official Train). Must stay excluded from every corpus (LOCKBOX_V4 rule 5); any checkpoint pretrained on public MultiNatSmoke is CONTAMINATED for boreal val/test. |
| multinatsmoke/Forest_Fire | Forest_Fire (masks only, images not shipped) | 0 | 389 mask-only rows (image_present=False); images need a separate Kaggle download; no licence; provenance unknown. |
| fasdd_uav | FASDD_UAV.zip (downloaded 2026-09-10, not extracted/manifested) | 25097 | Filename classes: neitherFireNorSmoke 11,986 / bothFireAndSmoke 7,821 / smoke 5,080 / fire 210 (scout counts). 60-image visual sample: ~30 of ~33 negatives are HPWREN fixed watchtower frames (FIgLib family -> eval-contamination risk, and NOT UAV); positives look like genuine low-altitude drone frames of grass burns. Quarantined pending dedup + per-image provenance filter; author class labels not re-verified by us; share-alike licence viral for derived annotations. Zip 14,888,184,890 B md5 c5ea9651 verified. |
| dfs | DFS-FIRE-SMOKE-Dataset (0 bytes acquired) | 0 | OneDrive dead; Baidu Pan share alive but needs Chinese-phone account + netdisk client; no author mirror. README claims 9,462 images with fire/smoke/other boxes (unverified). Needs a human download + legal review + D-Fire near-dup check if ever acquired. |

Reasons in one line each: FIgLib-family inside MultiNatSmoke = eval leak (65/89 opened eval events, 53 exact-hit units; two subsets also CC-BY-NC-SA);
D-Fire/FESB/SMOKE5K dup = byte-identical to local D-Fire already in E; Boreal dup = byte-identical copies of VAL/TEST burns; Forest_Fire = no images, no licence;
Boreal Subset-C = downsampled duplicates of Subset-A (+9 unlabeled smoke frames); FIRESENSE smoke_pos/flame_pos = no frame truth, unlabeled smoke, overlays;
MmodalFire occlusion/out-of-session = lens covered / unlabeled residual smoke; FASDD_UAV = ~90% of sampled negatives are HPWREN tower frames (FIgLib-family
contamination risk, not UAV), unverified provenance, share-alike licence; DFS = unobtainable + unlicensed.

## 6. Acceptance gaps vs PROTOCOL section 6 ("data ready")

| criterion | fixed-camera | aerial / UAV | forest-ground (new region) |
|---|---|---|---|
| negatives: >=2 h valid, human-verified | pyro-sdis VAL 4.37 h neg-neg (1158 frames, ~30 s sampled) — MET on hours, but same source domain as training (not "matched footage ADDED") and labels are box-absence with 36% sequence flicker -> human spot-check recommended via the Day-1 review tool | **0.122 h of 2 h** (14 heinola clips) + 217 sampled stills; gap = 1.88 h | **0 h of 2 h** (AuSmoke positives-only; permission pending anyway) |
| negatives: >=3 independent units | 8 cameras but only **2 sites** (cabanelle, marguerite); independence judged at site level = NOT MET, at camera level = met | **1 burn of 3** (heinola; karkkila clip is train-side) | **0 of 3** |
| negatives: licence recorded | Apache-2.0 (yes) | CC BY 4.0 (yes) | n/a |
| onsets: >=10 independent events with reviewed intervals | **0 of 10 reviewed**; candidates: 436 unopened FIgLib events (author onset labels, not on disk), 36 sdis VAL appears-type sequences (flicker) | **0 of 10**; no pre-onset aerial footage exists on disk (Boreal clips are 30 s, AuSmoke trimmed) | **0 of 10** |
| final-test onsets labelled blind to model outputs | FIgLib-39 / lockboxes already seen by models -> cannot serve; remainder could, after freeze | none | none |
| training negatives from the NEW domains (Q1 premise) | FIRESENSE 292 frames (staged scenes, already in v4 corpora) | karkkila 37 frames + 1 clip (32 frames) = **69 frames** | **0** |

Bottom line: fixed-camera calibration (experiment A) can run now on pyro-sdis VAL embedded negatives (reported as "~30 s sampled, 4.37 valid h")
with the FIgLib-39 >=30/39 constraint as a historical check. Aerial calibration can only report point estimates on 0.122 h / 1 burn (CIs will be
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
- Discrepancy noted, not resolved: `dedup_v4_meta.json` has 53 `evalhit_units` keys (45 figlib + 8 smoke5k) while `reports/v4/DATA_CARD_V4.md` section 1.4 quotes 52; this file uses the direct artifact count.
- Other Day-1 agents (negative review, onset audit, MmodalFire review) may raise `verified_negatives`/`onset_truth` for specific rows; this file reflects artifacts present at generation time and should be regenerated (same script) when they land.


---
## V5 acquisitions (appended by scripts/v5/inventory_v5_extras.py)
| source | units | frames | candidate hours | purpose | licence | verified |
|---|---|---|---|---|---|---|
| BAMBI raw UAV (Austria) | 13 flights / 11 sites | 4140 | 1.03 h reviewable @1 fps | aerial calibration/val candidate | CC-BY-4.0 | 0 h (pending human review) |
| MONIMET MC124 phenocam (Finland) | 1 camera-year / 183 camera-days | 6375 | 30-min sampled (frame-FP only) | fixed-camera calibration/val candidate | CC BY 4.0 | 0 (pending; contains ≥2 real-smoke frames) |

Acceptance (PROTOCOL §6) after V5 Day-1: aerial candidate footage 1.15 h of the 2 h bar, 0 h verified; fixed-camera continuous footage still 0 h (MONIMET is sampled); onset events human-reviewed 0 of 10.
FIgLib licence = CC BY-NC-ND 4.0 (Amendment A1) is now in the generator.
