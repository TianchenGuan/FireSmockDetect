# v4 sequences — data-prep report (iteration 4, DATA PREP track)

Deliverables in this directory: `sequences_v4.parquet`, `ONSET_AUDIT.md`, `replay_val_v4.jsonl`, `boreal_frames/`, `audit_v4.json`. Built by `scripts/v4/build_sequences.py all` (CPU-only, deterministic; v3 data and v4 manifests untouched).

## Counts by source and split

| source | split | sequences | frames | wall-clock |
|---|---|---|---|---|
| ausmoke | train | 22 | 10905 | unknown (order units; original fps not recorded) |
| ausmoke | val | 4 | 1471 | unknown (order units; original fps not recorded) |
| boreal | train | 105 | 3370 | 0.94 h @1fps |
| boreal | val | 125 | 3871 | 1.08 h @1fps |
| firesense | train | 13 | 530 | 0.15 h @1fps |
| firesense | val | 9 | 292 | 0.08 h @1fps |
| **total** |  | **278** | **20439** |  |

Splits: AuSmoke per `forest_holdout_v4.json` (test units never loaded); boreal per `aerial_split_v4.json` burn-site split (train=evo+karkkila, val=heinola, ruokolahti=test never read); FIRESENSE has no unit split — policy: smoke_neg->val (replay negative footage), smoke_pos->train (weak video-level positives).

## Onset (see ONSET_AUDIT.md)

- AuSmoke: **0 of 26 sequences observe a background->smoke onset** — every shipped frame is annotated positive; filename offsets (>=0, 0 = first annotated frame) show segments were trimmed at/after first visible smoke.
- Boreal + FIRESENSE positives: video-level labels only; frame onsets deliberately NOT invented.
- Consequence: **v4 adds NO observed-onset training signal**; onset supervision still comes only from v3 pyro-sdis sequences (train seqs with 'appears' pattern) and FIgLib-style replay eval.

## Label consistency (AuSmoke)

- Annotation-presence flicker: 0% (structurally — positive-only corpus), vs v3 pyro-sdis sequence flicker 29% train / 36% val. NOT comparable as label quality.
- Derived-box-presence flips (mask->box min-area filter): 0.00% of adjacent pairs, 0.0% of sequences.
- Mask-area discontinuities >2x: 6.9% of adjacent pairs, 96.2% of sequences — the honest instability metric for this source.

## VAL alert-replay package (replay_val_v4.jsonl)

- 139 sequences; negative-footage pool: 24 sequences / 763 frames (~0.21 h @1fps).
- CONTAMINATION FLAGS carried per line: firesense smoke_neg frames are background negatives inside v4 YOLO train corpora (CORPUS_B_G_v1 et al.) -> optimistic false-alert numbers for models trained on them; the single karkkila non-smoke clip is from a TRAIN burn site.
- AuSmoke val sequences contribute NO pre-onset negative segments (none exist).

## Trust assessment

**Usable for supervised temporal training (frame-level):** AuSmoke train sequences (box labels from masks) — but positive-only: they teach smoke persistence/appearance, NOT the background->smoke transition; negatives must come from v3 pyro-sdis negatives / firesense / boreal no-smoke clips. aus_20240209_SVF_Arbo_SmokeMachine is synthetic-source smoke (train-only per holdout notes, never eval).
**Weak/sequence-level only:** boreal clips and firesense videos (videolabel) — usable for sequence-level supervision or alert-replay eval, NOT for frame-supervised training.
**Export-for-review:** AuSmoke frames with derived-box flips or >2x area jumps (listed in audit_v4.json per_seq_consistency); boreal positive clips if frame-level use is ever wanted (would need manual onset annotation).
**Val caveats:** AuSmoke val = 4 units but only ~3 independent events (axis fixed-cam pair shares one camera); boreal val = 1 burn (heinola) — treat aerial val numbers as anecdotal (tiny-N warning in aerial_split_v4.json).

