# Tile-pathway threshold tuning — VAL ONLY (iteration 2, Track E)

Date: 2026-09-11. Script: `src/eval/tile_tuning.py`. Raw numbers:
`reports/v2/tile_tuning_val.json` (+ per-image preds in
`tile_tuning_val.raw.parquet`, so re-sweeps need no re-inference).

## Question
Iteration-1 tiling (full pass + 1024-px tiles, class-aware NMS merge) was
rejected because tile detections at the shared confidence threshold exploded
false alarms (FIgLib pre-smoke alert rate 0.10 -> 0.45). Hypothesis: tile
detections need their OWN, higher threshold. `SmokeDetector` now takes
`tile_conf_offset` (default 0.0 = unchanged behavior) applied in the new
post-filter helper `filter_by_conf`: tile-sourced detections must clear
`base_conf + offset`, full-image detections just `base_conf`.

## Setup
- Split: frozen VAL only (`data/yolo/lists/local/sdis_val.txt`, 7,335 imgs).
  **Sampled 2,000 images** with `np.random.default_rng(0)` (sorted index
  choice) to keep CPU time sane; 1,712 positive / 288 negative images,
  1,817 GT boxes (609 tiny <0.1% area, 1,118 small, 83 medium, 7 large).
  Test split and FIgLib untouched.
- Model `models/expB_best.pt`, imgsz 1280, tile 1024 (at 1280x720 that is
  2 tiles + 1 full pass), conf_floor 0.02.
- Matching logic identical to `src/eval/eval_detection.py`: greedy
  score-descending, IoU >= 0.4, size buckets, negative-image FP rate.
- Tiled pathway measured exactly as deployed: merge (class-aware NMS at
  floor) THEN per-source filter. The no-tiling baseline uses the **raw
  full-pass** detections — not the `source=="full"` subset of the merged
  list, because NMS can suppress a full-image det under a tile det.

## Results (val sample, tiny = GT area_frac < 0.001)

### No tiling (baseline)
| conf | precision | recall | tiny recall | neg-image FP |
|------|-----------|--------|-------------|--------------|
| 0.05 | 0.528 | 0.849 | 0.795 | 0.458 |
| 0.10 | 0.629 | 0.806 | 0.760 | 0.420 |
| 0.15 | 0.690 | 0.767 | 0.709 | 0.375 |
| 0.20 | 0.740 | 0.738 | 0.670 | 0.354 |
| 0.25 | 0.779 | 0.708 | 0.642 | 0.319 |
| **0.30** | **0.801** | **0.675** | **0.594** | **0.299** |
| 0.40 | 0.848 | 0.598 | 0.493 | 0.233 |
| 0.50 | 0.889 | 0.507 | 0.376 | 0.167 |
| 0.60 | 0.918 | 0.401 | 0.271 | 0.118 |
| 0.70 | 0.939 | 0.236 | 0.113 | 0.063 |

(bold = iteration-1 deployed single-frame operating point)

### Tiled, by tile_conf_offset
| conf | offset | precision | recall | tiny recall | neg-image FP |
|------|--------|-----------|--------|-------------|--------------|
| 0.05 | 0.0 | 0.524 | 0.886 | 0.856 | 0.549 |
| 0.10 | 0.0 | 0.621 | 0.848 | 0.813 | 0.504 |
| 0.15 | 0.0 | 0.673 | 0.816 | 0.772 | 0.438 |
| 0.20 | 0.0 | 0.706 | 0.789 | 0.742 | 0.406 |
| 0.25 | 0.0 | 0.742 | 0.761 | 0.709 | 0.365 |
| 0.30 | 0.0 | 0.765 | 0.739 | 0.685 | 0.361 |
| 0.40 | 0.0 | 0.816 | 0.673 | 0.609 | 0.278 |
| 0.50 | 0.0 | 0.854 | 0.585 | 0.484 | 0.201 |
| 0.05 | 0.1 | 0.611 | 0.845 | 0.800 | 0.479 |
| 0.10 | 0.1 | 0.669 | 0.812 | 0.770 | 0.441 |
| 0.15 | 0.1 | 0.713 | 0.785 | 0.736 | 0.396 |
| 0.20 | 0.1 | 0.744 | 0.759 | 0.708 | 0.379 |
| **0.25** | **0.1** | **0.773** | **0.733** | **0.680** | **0.319** |
| 0.30 | 0.1 | 0.793 | 0.701 | 0.634 | 0.309 |
| 0.40 | 0.1 | 0.837 | 0.623 | 0.532 | 0.236 |
| 0.50 | 0.1 | 0.885 | 0.515 | 0.412 | 0.160 |
| 0.05 | 0.2 | 0.641 | 0.816 | 0.764 | 0.455 |
| 0.10 | 0.2 | 0.704 | 0.785 | 0.736 | 0.420 |
| 0.15 | 0.2 | 0.743 | 0.760 | 0.706 | 0.358 |
| 0.20 | 0.2 | 0.770 | 0.723 | 0.659 | 0.337 |
| **0.25** | **0.2** | **0.796** | **0.685** | **0.613** | **0.295** |
| 0.30 | 0.2 | 0.812 | 0.652 | 0.558 | 0.267 |
| 0.40 | 0.2 | 0.864 | 0.554 | 0.460 | 0.194 |
| 0.50 | 0.2 | 0.907 | 0.436 | 0.301 | 0.125 |
| 0.05 | 0.3 | 0.665 | 0.794 | 0.737 | 0.424 |
| 0.10 | 0.3 | 0.726 | 0.751 | 0.688 | 0.379 |
| 0.15 | 0.3 | 0.763 | 0.713 | 0.642 | 0.340 |
| 0.20 | 0.3 | 0.787 | 0.675 | 0.585 | 0.299 |
| 0.25 | 0.3 | 0.815 | 0.630 | 0.539 | 0.257 |
| 0.30 | 0.3 | 0.837 | 0.586 | 0.488 | 0.226 |

(conf 0.6/0.7 rows omitted here — strictly dominated; full grid in the JSON.)

## Reading
1. **offset 0.0 (shared threshold — the iteration-1 pathway) confirms the
   rejection**: at every base conf it buys tiny recall by inflating
   neg-image FP (at conf 0.30: FP 0.299 -> 0.361). Never FP-neutral.
2. **offset 0.1–0.2 makes tiling FP-neutral and mildly positive.**
   - At the deployed FP level (<= 0.299): **(base 0.25, offset 0.2)** gives
     tiny recall 0.613 vs 0.594 (+1.8pp), overall recall 0.685 vs 0.675,
     neg-image FP 0.295 vs 0.299, precision 0.796 vs 0.801.
   - Exactly FP-matched at 0.319: **(base 0.25, offset 0.1)** vs no-tiling
     conf 0.25: tiny 0.680 vs 0.642 (+3.8pp), recall 0.733 vs 0.708,
     precision 0.773 vs 0.779.
3. **offset 0.3 over-suppresses**: at FP 0.299 tiny recall (0.585) drops
   below the no-tiling baseline (0.594).

## Recommendation
**(base_conf 0.25, tile_conf_offset 0.2)** — satisfies the acceptance bar:
beats no-tiling tiny recall (+1.8pp) at slightly LOWER neg-image FP than the
deployed no-tiling point (0.295 vs 0.299). If a marginally higher alert rate
is acceptable, (0.25, 0.1) at FP 0.319 gives the bigger tiny-recall win
(+3.8pp over the FP-matched baseline).

Honest caveats:
- Gains are real but **modest**: +1.8pp tiny = ~11 of 609 tiny boxes;
  the FP edge is ~1 negative image of 288. Not transformative.
- CPU latency ~3x per frame (~190ms full-only -> ~560ms with 2 tiles).
- Neg-image FP on sdis val is a **proxy** for the FIgLib pre-smoke alert
  rate that drove the iteration-1 rejection. Before shipping tiling, a
  causal FIgLib replay at (0.25, 0.2) must confirm the pre-smoke alert rate
  stays at the no-tiling level — as an EVALUATION, not tuning, step.
- Numbers are on a seeded 2,000-image val sample (seed 0); the frozen val
  split has 7,335 images. Sweep can be extended from the cached parquet.
