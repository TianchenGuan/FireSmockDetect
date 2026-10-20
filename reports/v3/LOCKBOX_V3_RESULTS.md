# FIgLib LOCKBOX v3 — one-time evaluation results (iteration 3 final)

Date: 2026-09-14 ~06:50 UTC. Set: `data/v3/manifests/figlib_lockbox_v3_manifest.parquet`
(25 events / 1,978 frames / 25 new cameras, frozen 2026-09-13, pristine until
this run — freeze record `data/v3/manifests/LOCKBOX_V3_FROZEN.md`; parquet
sha256 re-verified `66514b03...c07608` immediately before evaluation).

**This evaluation ran exactly once.** Systems and thresholds were
pre-registered before any lockbox frame was scored; no threshold was adjusted
and no system was re-run afterward. All three systems are contamination-free
on FIgLib (trained only on pyro-sdis/D-Fire).

Pre-registered systems:
1. **expE** — YOLO11m full-image 1280, deployed thresholds 0.30/0.15
   (3-of-5) — the deployment control.
2. **rfdetrL_e48** — RF-DETR-Large 1280, its val-selected thresholds
   0.5349/0.382 (from `rfdetr_eval.md` §4).
3. **B1** — frozen dinov3-vitl16 + SFP + Faster R-CNN (Track B), its
   val-selected thresholds 0.5624/0.3454 — the FIgLib-39 event-coverage
   leader (`trackBC_results.md` §2c).

Mechanics: identical to the FIgLib-39 replays — per-frame max-score
scoring with each system's own inference code (expE: live `replay_eval`
CPU; rfdetrL: `score_frames_cpu.py`, 6x8-thread shards; B1:
`score_frames_cpu_dino.py`, 16x8-thread shards; 0 frames without score for
all systems), then offline `AlertEngine` replay (seq window 5, 3 hits,
conf floor 0.05, fresh state per event, chronological).

## Results (25 events)

| system | detected | missed | median delay | mean delay | R@1 min | R@3 min | R@5 min | pre-visible alerts |
|---|---|---|---|---|---|---|---|---|
| expE y11m @0.30/0.15 | 21/25 | 4 | **120 s** | 434 s | **9/25** | 11/25 | 12/25 | 5 |
| rfdetrL @0.5349/0.382 | 20/25 | 5 | 179 s | 540 s | 7/25 | 11/25 | 11/25 | 4 |
| B1 dinov3-L @0.5624/0.3454 | **24/25** | **1** | 477 s | 490 s | 8/25 | **12/25** | 12/25 | 4 |

R@k = events whose first alert came within k minutes of the first-visible
frame (pre-visible alerts count as delay 0, per the established convention,
and are also listed separately). Median delays are over each system's OWN
detected set and are NOT directly comparable across systems — B1's extra
3-4 detections are hard events found late (477-1,560 s), which drags its
median up; the paired rows below are the like-for-like read.

### Paired comparisons (shared-detected events; delta = first minus second)

| pair | shared | only-first | only-second | median delta | mean delta | first earlier / later / equal |
|---|---|---|---|---|---|---|
| expE vs rfdetrL | 19 | 2 | 1 | 0 s | +35 s | 6 / 7 / 6 |
| expE vs B1 | 20 | 1 | 4 | 0 s | -6 s | 5 / 9 / 6 |
| rfdetrL vs B1 | 20 | 0 | 4 | 0 s | +123 s | 6 / 9 / 5 |

On events that both systems detect, delays are statistically
indistinguishable (median delta 0 s in every pairing). The systems differ in
*coverage*, not speed.

### Missed events

- expE (4): 20170722_FIRE_mg-n-iqeye, 20180726_FIRE_so-w-mobo-c,
  20200930_DeLuzFire_rm-w-mobo-c, 20240909_BridgeFire_marconi-n-mobo-c
- rfdetrL (5): the same 20170722/20200930/20240909 + 20171016_FIRE_sdsc-e,
  20180602_Alison_sp-w
- B1 (1): 20171016_FIRE_sdsc-e-mobo-c

B1 detects every event expE misses (20170722 at +720 s, 20180726 at +660 s,
20200930 at +1,080 s, 20240909 at +60 s) and 3 of rfdetrL's 5 misses; its
single miss (20171016 sdsc-e) is detected by expE (pre-visible) but also
missed by rfdetrL.

### Pre-visible alerts (reported separately, not counted as errors)

- expE (5): 20171016 sdsc-e, 20180602 sp-w, 20210209 hp-e,
  20231026 prescribed cp-w, 20241006 Saddleback stgo-w
- rfdetrL (4): 20201205 sclm-e, 20220210 Emerald signal-s,
  20231026 prescribed cp-w, 20241006 Saddleback stgo-w
- B1 (4): 20160722 mw-e, 20180813 mg-w, 20191005 vo-n, 20210209 hp-e

## Reading

1. **The FIgLib-39 pattern replicates on pristine data**: B1 (frozen
   DINOv3-L) finds substantially more events (24/25 vs 21/25 vs 20/25) at
   equal delay on shared events. Its coverage advantage is real, not an
   artifact of the (opened) iter-1 set. With n=25, 24 vs 21 is at the edge
   of binomial noise (±2-3 events), but the direction matches FIgLib-39
   (34/39 vs 30/39) — two independent sets now agree.
2. **Early detection (the deployment metric) is a tie**: R@1/3/5 min are
   within one event across all three systems. B1's extra coverage arrives
   minutes-to-tens-of-minutes late — valuable for "never miss a fire",
   not for "alert in the first minute". This matches the frame-level
   result that B1 is *below* expE at matched FP (val pAUC 0.487 vs 0.520):
   its detections are lower-confidence but more broadly distributed
   across events.
3. **rfdetrL's FIgLib-39 advantage does NOT replicate** (33/39 > expE 30/39
   there; 20/25 < 21/25 here) — consistent with its ±2-3-event noise band
   and a reminder that single-set event-count differences of that size
   should not drive decisions.
4. Delay medians over different detected sets mislead (B1 477 s vs expE
   120 s reads terribly until paired: median delta 0 s). Any future use of
   these numbers must quote the paired rows.

Caveats: event-level alerting only (no boxes); delay measured from
first-visible frame, not ignition; fixed-camera HPWREN imagery (no drone
claim); pre-visible alerts may be true faint smoke or false positives —
uninspected (the lockbox rule forbade visual inspection before this run,
and per-frame review after the fact is error-analysis this report
deliberately stops short of). HPWREN FIgLib credit:
https://hpwren.ucsd.edu/.

## Files

- `reports/v3/replay_lockboxv3_{expE,rfdetrL,B1}.json` (full per-event,
  per-frame records)
- `models/v3/evals/lockbox_v3/{rfdetrL,B1}_preds.jsonl` (frame scores)
- Frame list: absolute paths of the manifest's 1,978 `frame_path` rows

**Lockbox v3 status: OPENED (this evaluation, once) — 2026-09-14. It is now
historical: banned from any future training, mining, tuning, or selection.**
