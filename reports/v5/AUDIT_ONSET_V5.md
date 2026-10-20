# AUDIT_ONSET_V5 — limited spot-check of onset material (V5 Day-1)

**Status: model-free, human-style spot check performed by an AI agent (contact sheets / same-region crop comparison, zoom crops where noted). No detector output was used for any verdict. Every verdict below REQUIRES human confirmation before it is used as onset truth, as a label fix, or as a negative.** Scope and protocol: reports/v5/PROTOCOL_V5.md §2 (data use) and §3 (onset = interval [last_clear_no_smoke, first_clear_smoke]).

Artifacts: `data/v5/audit/boreal_trainclips_audit.csv`, `data/v5/audit/sdis_flicker_audit.csv`, `data/v5/audit/onset_candidates_v5.csv`, contact sheets under `data/v5/audit/contact_sheets/{boreal,sdis}/`, helper scripts `scripts/v5/audit_boreal_sample.py`, `scripts/v5/audit_sdis_flicker_rank.py`, `scripts/v5/audit_sdis_flicker_sheets.py`. Compute: CPU only; no cluster reservation.

## A. Boreal Subset-B TRAIN-burn clips (evo + karkkila only)

Population: 105 train clips (31 evo + 74 karkkila; 104 videolabel smoke, 1 non-smoke = karkkila_44) from `data/v4/sequences/sequences_v4.parquet`, 1-fps frames in `data/v4/sequences/boreal_frames/`. heinola (VAL) and ruokolahti (TEST) were not read. Sample: 30 clips, `random.Random(0).sample(sorted_ids, 30)` -> 6 evo + 24 karkkila (list in `data/v5/audit/boreal_sample_seed0.json`). Each clip viewed as a full 1-fps contact sheet (28-34 frames); a phase-correlation global-shift cue was overlaid only to flag camera turns, not to decide anything.

| verdict | clips |
|---|---|
| already_smoke_at_start | **30** |
| camera_turn_then_smoke (primary) | 0 |
| true_first_visible | **0** |
| no_smoke | 0 |
| uncertain | 0 |

Secondary observations (all inside already-smoke clips):
- **Turn-induced re-entry: 1** (boreal_evo_14: plume leaves the frame during a pan at t17-t20 — forest + lake, no visible smoke in frame — and re-enters at t21 only because the camera turns back). These 4 frames are view-dependent smoke-free frames, not evidence of a no-smoke scene, and must not be labelled as an onset.
- **Faint early-stage smoke at t00: 2** (boreal_karkkila_2, boreal_karkkila_54: translucent plume plus small flames already visible at t00, growing slowly over 30 s; confirmed on zoom crops). Candidates for faint-smoke supervision only; the first-visible moment is before the clip starts.
- Camera moves with smoke present before and after the move: 2 (evo_26, evo_6).

Consequence: **in the 30-clip sample, 0 clips contain an observable no-smoke -> first-visible-smoke transition.** The 30-s Subset-B clips start after ignition is visible. This is consistent with the v4 note that frame onsets cannot be derived from Subset-B, and it is now a looked-at result rather than an inference from clip length. Only the raw flight recordings (FGI request) can provide Boreal onsets; if obtained, heinola material inherits VAL (PROTOCOL §2).

## B. pyro-sdis flicker sequences (TRAIN sites only)

Source: `data/v3/sequences/sequences_train.json` (915 sequences; 496 have >=1 has-box/no-box flip; 265 carry label_pattern=flicker). Ranked by number of flips; top 20 selected (15 courmettes, 3 valbonne, 2 brison; 10-23 flips each; `data/v5/audit/sdis_flicker_top20.csv`). For every bounded run of unlabeled frames ("island", 140 islands / 493 unlabeled frames in total) the same image region (>=3x the neighbouring labelled box, >=200 px) was tiled as labelled-before / unlabeled / labelled-after and compared by eye. Rows where the labelled boxes jump across the frame degenerate to whole-frame crops and were marked unjudgeable.

| sequence verdict | n |
|---|---|
| label_omission (smoke visible while unlabeled in essentially all islands) | **11** |
| mixed_omission_ambiguous | 8 |
| ambiguous (likely label noise on haze, no clear plume) | 1 |
| genuine disappearance as the dominant cause | 0 |

Island-level tally (agent judgement, approximate for faint rows): omission **96**, genuine thinning/dispersal **1**, ambiguous 40, unjudgeable (whole-frame) 3.

Findings:
1. **Flicker in pyro-sdis train labels is overwhelmingly annotation omission, not smoke disappearing.** In 11/20 sequences a plainly visible plume sits in every unlabeled frame (e.g. courmettes-160 2024-04-23: plume grows over f053-f058 while unlabeled). Genuine fading was seen in one island (courmettes-212 2024-01-22 f059-f067).
2. **Label-derived "appears" patterns are not onset evidence.** Six of the 20 sequences begin with 1-6 unlabeled frames. Checked frame-by-frame: in 4/6 the plume is already clearly visible in frame 0 (valbonne-349 04-04, courmettes-160 04-03, brison-110, courmettes-160 01-16 [fog confound]); 2/6 are ambiguous faint smudges (valbonne-349 02-04, courmettes-275 09-17). **0/6 verified true first-visible.** The 1-sec / 29-sec cadence frames before the first box are therefore NOT a reliable "last clear no-smoke" frame.
3. Label-noise patterns worth a human pass: horizon/valley haze bands boxed as smoke (brison-290 f025, brison-110 f125/f128), sporadic tiny boxes in a hazy valley with no visible plume (courmettes-212 2024-01-12, 157 frames), a cumulus cloud near labelled areas (courmettes-160 2024-04-03 09:50).
4. Non-independence: courmettes-275 2024-04-08 19:22 and courmettes-212 2024-04-08 19:26 are the same fire from two cameras at the same site; any onset/event counting must group them.
5. Observed cadences inside these sequences: mostly 29-31 s, with 1-10 s bursts and gaps up to 1400 s; episode/valid-time rules in PROTOCOL §3 apply.

## C. Candidate onset list (`data/v5/audit/onset_candidates_v5.csv`)

Counts from the audited material (TRAIN only):
- **TRUE first-visible onset candidates verified by viewing: 0** (Boreal 0/30; sdis 0/6 label-starts checked).
- Turn-induced smoke entry: 1 clip segment (boreal_evo_14 t17-t21) — flagged, not an onset.
- Already-smoke-at-start: 30/30 Boreal clips (2 of them faint early-stage: karkkila_2, karkkila_54 -> faint-smoke supervision candidates).
- sdis sequences flagged for label repair (omission-dominated flicker): 20 (verdict column).
- Unverified label-only pool for human onset review: 76 `appears` + 60 `appears_disappears` train sequences (from v3 audit.json), listed with onset_status=unverified_label_only. Given 0/6 checked label-starts were real onsets, the prior that any of these is a true first-visible event is low; each needs the same frame-0 check (strip builder: `scripts/v5/audit_sdis_flicker_sheets.py` pattern, `lead_*.jpg`).

PROTOCOL §6 acceptance (>=10 independent human-reviewed onset events) is **not reachable from existing train material** on this evidence; the onset/temporal arm stays DEFERRED until new footage with pre-ignition coverage arrives (FGI raw flights; fixed-camera recordings with verified pre-smoke frames). No onset truth has been created by this audit.

## Limitations
- One AI reviewer, one pass, contact-sheet resolution (Boreal thumbnails 512x270 of 4096x2160; sdis crops 300-440 px of 1280x720). Faint-smoke calls (karkkila_2/54, valbonne-349, courmettes-275) are the least certain. Thumbnail-scale review can miss very small or very thin smoke, so "no visible smoke" statements (evo_14 t17-t20) are also provisional.
- Boreal: 30 of 105 clips (28.6%) sampled; the 1 non-smoke train clip (karkkila_44) was not in the seed-0 sample.
- sdis: 20 of 496 flipping train sequences; selection favours high-flip sequences, so the omission rate here need not generalise to low-flip sequences. Island counts per sequence are agent estimates recorded in the CSV, not a second annotator's.
- VAL/TEST sites, heinola, ruokolahti, AuSmoke, MultiNatSmoke and all lockboxes were not opened. AuSmoke/MultiNatSmoke remain PERMISSION PENDING and were not used.
