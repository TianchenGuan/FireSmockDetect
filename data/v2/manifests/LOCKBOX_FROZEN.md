# FIgLib LOCKBOX — FROZEN (iteration 2 final evaluation set)

**Frozen at:** 2026-09-11T09:03:23Z
**Manifest:** `data/v2/manifests/figlib_lockbox_manifest.parquet`
**sha256 (parquet):** `1c7778451516f877b3f0c782135e410ffc4f8bebe4448493a3485dd8e8c9ebab`
**sha256 (csv.gz):**  `98e917bc201bbab81c071c6333202f5eb9b639f957b2c09c3944d6e66f7eba53`

## THE RULE

**NO model — detector, VLM, or otherwise — may be evaluated on, run over, or
tuned against any frame in this lockbox until FINAL REPORT time, and then
exactly ONCE.** No contact sheets, no visual inspection, no per-frame scoring,
no threshold selection, no error analysis loops. Any run against this set
before the single final evaluation invalidates it as a held-out test set.
Only file-level validation (names, sizes, offsets, hashes) has touched it.

## Contents

- 25 events, 1,979 frames, 25 distinct cameras, 25 distinct dates,
  years 2016–2026 (>=2 events per year, 11 years).
- Frames: `data/v2/figlib_lockbox/<event_id>/<unix_epoch>_<signed_offset>.jpg`,
  offset 0 = first visible plume. Original tarballs kept in
  `data/v2/figlib_lockbox/_tars/` (all byte-exact vs index `tgz_size_bytes`).
- Source: https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/ (retrieved 2026-09-11).
  License: HPWREN FIgLib, publicly available, credit reference to
  https://hpwren.ucsd.edu/ required in derivative work.

## Selection & validation provenance

- Selection: deterministic (seed 20260911) from
  `data/manifests/figlib_events_index.csv`; recipe + backups in
  `data/v2/manifests/lockbox_candidates.csv`.
- Validation (identical to iteration 1, `src/figlib_lockbox_manifest.py`):
  25/25 events pass — frame count == remote listing, monotonic offsets and
  epochs, epoch-delta == offset-delta, pre- and post-smoke frames present,
  zero frame exact (22 events) or near within <=4 s (3 events: Jimfire +4 s,
  Clubfire +2 s, JunctionFire +1 s). Report:
  `logs/figlib_lockbox_validation_report.csv`.

## Disjointness vs the 39-event iteration-1 FIgLib set

(`data/manifests/figlib_subset_manifest.parquet` — already inspected in
iteration 1, no longer pristine)

- Event-id overlap: **0**
- (camera_id, date) overlap: **0**
- camera_id overlap: **0** (all 25 lockbox cameras are new)
- Site+direction view overlap (e.g. lp-e color vs lp-e mono): **0**
  (three such collisions in the initial pick were swapped out on 2026-09-11
  before freezing)
- Honest disclosure: 14 physical HPWREN towers appear in both sets with
  DIFFERENT view directions (bh, bl, bm, lp, ml, mlo, om, pi, rm, sm, smer,
  sp, stgo, ws). Different headings = different scenes; retained.
- phash spot-check (200 seeded-random lockbox frames vs all 3,056 iter-1
  frames, thresholds as iteration-1 xdup audit): 0 exact sha256, 0 phash
  hamming=0, 0 near (hamming<=6); minimum observed hamming = 8, and all
  hamming<=10 pairs are across different cameras/views (generic landscape/sky
  similarity, not duplicates). Details:
  `data/v2/manifests/xdup_lockbox_vs_iter1_figlib.json`.

## Integrity re-check

```
sha256sum data/v2/manifests/figlib_lockbox_manifest.parquet
# must print 1c7778451516f877b3f0c782135e410ffc4f8bebe4448493a3485dd8e8c9ebab
```
