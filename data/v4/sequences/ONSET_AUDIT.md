# ONSET + LABEL-CONSISTENCY AUDIT — v4 sequences (iteration 4)

Generated deterministically by `scripts/v4/build_sequences.py audit`. Frame-level truth exists ONLY for AuSmoke (per-frame masks). Boreal and FIRESENSE are video-level labels: no frame onsets are invented for them.

## AuSmoke (multinatsmoke, mask-labelled)

| split | sequences | observed onset (bg->smoke) | positive from start (onset NOT observed) | negative only | ambiguous |
|---|---|---|---|---|---|
| train | 22 | 0 | 22 | 0 | 0 |
| val | 4 | 0 | 4 | 0 | 0 |
| **total** | **26** | **0** | **26** | **0** | **0** |

**Sequences genuinely spanning a background->smoke transition: 0 of 26.**

Why: the AuSmoke corpus ships ONLY annotated smoke frames — all 12376 kept frames have a non-empty mask (mask_fg_px > 0). The filename convention `<unit>_<fidx>-<+offset>` is FIgLib-like: offset >= 0 everywhere and offset==0 exists in 14/20 kept units, i.e. the shipped segments start at (or after) first visible smoke. Pre-onset background frames were not shipped, so NO AuSmoke sequence observes an onset. Candidate 'first annotated frame' is always frame 0 -> flagged `onset_observed=false` for every sequence.

## Boreal (30-s clips, video-level labels only)

| split | clips | videolabel positive (frame onset UNKNOWN) | non-smoke | observed frame onset |
|---|---|---|---|---|
| train | 105 | 104 | 1 | 0 (no frame labels — not invented) |
| val | 125 | 111 | 14 | 0 (no frame labels — not invented) |

## FIRESENSE (video-level labels only)

| split | videos | videolabel positive (frame onset UNKNOWN) | non-smoke | observed frame onset |
|---|---|---|---|---|
| train | 13 | 13 | 0 | 0 (no frame labels — not invented) |
| val | 9 | 0 | 9 | 0 (no frame labels — not invented) |

## AuSmoke label consistency

- Adjacent frame pairs: 12350
- Annotation-presence flips (mask nonempty <-> empty): 0 (0.0% of pairs) — 0 by construction: the corpus contains no annotated-negative frames.
- Derived-box-presence flips (converter min-area filter): 0 (0.00% of pairs); sequences affected: 0 (0.0%).
- Mask-area discontinuities (>2x frame-to-frame area change): 853 pairs (6.9%); sequences affected: 25 (96.2%).

**v3 pyro-sdis comparison**: v3 sequences with label flicker: train 265/915 (29.0%), val 97/266 (36.5%). AuSmoke's 0% presence flicker is NOT better labelling — it is a positive-only corpus, so presence flicker is structurally impossible. The comparable instability signal here is mask-AREA discontinuity, which affects 96.2% of sequences.

