# FIgLib LOCKBOX v3 — FROZEN (iteration 3 final evaluation set)

**Frozen at:** 2026-09-13T08:56:48Z
**Manifest:** `data/v3/manifests/figlib_lockbox_v3_manifest.parquet`
**sha256 (parquet):** `66514b03aed0ed3aefd86a0ed1cab7979734e9161e89e555cc4715ca4ac07608`
**sha256 (csv.gz):**  `e65e82a78b3b19469855c6f66f709a8dc6cdb629a85faba68857cedf4a5a8471`

## THE RULE

**NO model — detector, VLM, or otherwise — may be evaluated on, run over, or
tuned against any frame in this lockbox until FINAL REPORT time, and then
exactly ONCE.** No contact sheets, no visual inspection, no per-frame scoring,
no threshold selection, no error analysis loops. Any run against this set
before the single final evaluation invalidates it as a held-out test set.
Only file-level validation (names, sizes, offsets, hashes) has touched it.

## Contents

- 25 events, 1,978 frames, 25 distinct cameras, 25 distinct site+view
  headings, 25 distinct dates, years 2016–2026 (11 years, >=2 events per year
  for 10 of 11 years; 2022 has exactly 1 — see shortfall note below).
- Frames: `data/v3/figlib_lockbox/<event_id>/<unix_epoch>_<signed_offset>.jpg`,
  offset 0 = first visible plume. Original tarballs kept in
  `data/v3/figlib_lockbox/_tars/` (all 25 byte-size-exact vs index
  `tgz_size_bytes`). 2024+ tarballs arrive nested under
  `Data/HPWREN-FIgLib/HPWREN-FIgLib-Data/`; layout normalized at extraction.
- Source: https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/ (retrieved
  2026-09-13). License: HPWREN FIgLib, publicly available, credit reference
  to https://hpwren.ucsd.edu/ required in derivative work.

## Selection & validation provenance

- Selection: deterministic (seed 20260913), `src/figlib_lockbox_v3_select.py`,
  from `data/manifests/figlib_events_index.csv` (525 events); recipe +
  10 backups in `data/v3/manifests/lockbox_v3_candidates.csv`.
  Pool after quality filter (listing_ok, listed==offset-named counts,
  pre+post offsets) and disjointness vs ALL 64 previously used events
  (iter-1 39 + iter-2 lockbox 25): 127 events, 46 usable site+views.
- Validation (identical to iterations 1–2, `src/figlib_lockbox_v3_manifest.py`):
  25/25 events pass — frame count == remote listing (0 mismatches),
  monotonic offsets and epochs, epoch-delta == offset-delta, pre- and
  post-smoke frames present, zero frame exact (24 events) or near within
  1 s (1 event: 20260629_JunctionFire_hp-s-mobo-c +1 s). Report:
  `logs/figlib_lockbox_v3_validation_report.csv`.

## Disjointness vs ALL 64 previously used FIgLib events

(iter-1 `data/manifests/figlib_subset_manifest.parquet`, 39 events, opened in
iteration 1; iter-2 `data/v2/manifests/figlib_lockbox_manifest.parquet`,
25 events, opened at iteration-2 final report — both no longer pristine)

- Event-id overlap: **0**
- camera_id overlap: **0** (all 25 v3 cameras are new)
- (camera_id, date) overlap: **0**
- Site+view (heading) overlap, hardware suffix stripped (mobo-c/mobo-m/iqeye
  treated as the same view, e.g. lp-e color vs lp-e mono): **0**
- Honest disclosures:
  - 14 physical HPWREN towers appear in both v3 and a used set with
    DIFFERENT view directions (bh, bl, hp, lo, marconi, mg, ml, rm, smer,
    sp, stgo, tp, vo, wilson). Different headings = different scenes;
    retained (same policy as iteration 2).
  - 8 v3 events share a DATE with a used event — likely the same physical
    fire seen from a different tower/scene (20160604, 20160722, 20170711,
    20190924, 20220210, 20231027, 20251102, 20260629). Iteration-2 criteria
    (event-id / camera-id / cam+date / site+view) permit this; disclosed
    because the plume itself is shared even though the viewpoint and
    background are not.
- phash spot-check (200 seeded-random v3 frames, seed 20260913, vs ALL
  3,056 iter-1 frames AND ALL 1,979 iter-2 lockbox frames; thresholds as
  iteration-1/2 xdup audits):
  - vs iter-2 lockbox: 0 exact sha256, 0 h=0, 0 near (h<=6), min hamming 8.
  - vs iter-1: 0 exact sha256, 0 h=0; 61 pairs at exactly h==6, ALL from a
    single near-black night frame (Bonitafire tp-w -1921 s, grayscale mean
    5.0) sitting h==6 from 61 similarly dark frames spanning the ENTIRE
    iter-1 EmeraldFire marconi-w event (pre and post smoke, different
    camera/site/date/year) — a degenerate low-texture hash collision on
    dark sky, NOT duplication (one frame cannot duplicate an 80-min
    sequence). Excluding that frame: 0 near (h<=6), min hamming 8.
  - All h<=10 pairs are across different cameras/views (generic
    landscape/sky similarity). Details + follow-up analysis:
    `data/v3/manifests/xdup_lockbox_v3_vs_prev_figlib.json`.

## Year-coverage shortfall (disclosed)

2022 contributes only 1 event (not 2). This is structural, not a sampling
choice: after disjointness vs the 64 used events, the 2016 pool's two usable
views are {smer-tcs3, mw-e}, and the second slots of BOTH 2021 and 2022
require view tp-w — under within-set camera/view/date disjointness at most
5 events total can come from {2016, 2021, 2022}. The selection achieves that
maximum (2016: 2, 2021: 2, 2022: 1).

## Integrity re-check

```
sha256sum data/v3/manifests/figlib_lockbox_v3_manifest.parquet
# must print 66514b03aed0ed3aefd86a0ed1cab7979734e9161e89e555cc4715ca4ac07608
```
