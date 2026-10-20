# EARLY WARNING v4 — temporal models on the extended sequence set

Iteration 4, EARLY-WARNING track. Validation only: pyro-sdis test sites, FIgLib,
and every lockbox were never read. All thresholds and checkpoints selected on
VALIDATION. GPU work: 3 single-node green g6e jobs (8x L40S), 0.72 node-h wall
(0.97 node-h reserved), all reserved/released through `cluster/v4/registry.py`.

Full numeric source: `reports/v4/trackd_v4_eval.json` (every table below is
rendered from it by `scripts/v4/trackd_report_tables.py`; nothing retyped).
Per-example predictions: `models/v4/trackd/preds/{preds_sdis_val.npz,
preds_val_replay_v4.npz}`.

## 0. Verdicts

1. **The learned causal temporal head still beats the deployed 3-of-5
   persistence rule at every matched FPR on pyro-sdis val** — 30/30 cells
   (6 heads x 5 FPR targets), in both seeds of all three training recipes
   (v3-only, v4-extended, v4-capped). Same conclusion as v3.
2. **The v4 data extension did not deliver what it was meant to.** On
   pyro-sdis val it moves the temporal head by an amount that is within or
   just above seed noise (seed-mean PR-AUC 0.940 -> 0.942; R@FPR5% 0.248 ->
   0.269). On the new out-of-domain footage it is harmful: heads trained with
   the full AuSmoke extension fire on **43-47% of frames of boreal NON-smoke
   clips** and **83-98% of firesense NEGATIVE frames** at thresholds where
   the v3-only heads fire on 0%. The mechanism is a **domain shortcut**: the
   extension adds 10,905 frames that are ALL positive and come from a new
   camera domain with zero negatives from that domain, so "looks like a
   non-pyro-sdis camera" becomes a smoke cue. A capped ablation (5 AuSmoke
   sequences / 2,571 frames instead of 22 / 10,905) shows a clean
   dose-response: mean head score on v4 replay frames 0.13-0.21 (v3-only)
   -> 0.48-0.64 (capped) -> 0.73-0.89 (full).
3. **Most of the learned gain is spatial readout, not history** (again). The
   t1-only control (identical module, window = current frame) matches or
   beats the temporal head at every FPR target on seed-mean recall, and is
   the best system at the first frame of a sequence.
4. **T=1 works but is weak for the temporal head.** All heads emit a finite
   score with no buffer requirement. At the FPR-5% threshold the temporal
   heads recall 1-4% of positive FIRST frames vs 7.6% for raw expE and 11.9%
   for the t1-only readout; persistence is 0% by construction.
5. **YOLO + low-frequency DINO sweep: beats YOLO alone at matched FP in every
   cell (60/60), but the margin at the <=20% compute cap is negligible** —
   N=70 adds +0.4 to +0.8 pt episode detection (2-4 of 469 episodes), 91% of
   val sequences never receive a DINO pass, and the matched-FP comparison
   rounds to integer FP frames so the combination carries <=0.2 uncharged
   false-alert frames. At N=20 (+70% compute) the gain is +0.6 to +2.1 pt.
   Not a case for deploying DINO sweeps on this evidence.
6. **Recommendation:** do NOT ship the v4-extended heads. Ship candidates
   remain the v3-only heads (t1-only preferred). The single most valuable
   data item for this track is **negative footage from the new camera
   domains** (aerial, Australian bush), not more positives — the extension
   failed for lack of matched negatives, not lack of smoke.

## 1. What went in (frozen manifest)

`data/v4/sequences/trackd/MANIFEST_trackd_v4.json` (inputs hashed:
sequences_v4.parquet sha256 a428e6d9..., replay_val_v4.jsonl c394d4e0...).
Converter: `scripts/v4/build_trackd_v4.py`.

| set | source | seqs | frames | positive frames | frame truth | onset observed |
|---|---|---|---|---|---|---|
| v3 train (unchanged) | pyro-sdis train sites | 915 | 22,170 | 18,748 | yes (boxes) | yes (76 appears + 60 appears_disappears + flicker) |
| **v4 train_ext** | AuSmoke train | 22 | 10,905 | 10,905 | yes (mask-derived boxes) | **0 / 22** |
| **v4 train_ext** | boreal train no-smoke clip (karkkila_44) | 1 | 32 | 0 | yes (video-level NEGATIVE implies every frame negative) | n/a |
| v3 val (selection + section A) | pyro-sdis val sites | 266 | 7,335 | 6,177 | yes | yes |
| **v4 val replay** | AuSmoke val | 4 (~3 independent events) | 1,471 | 1,471 | yes | 0 / 4 |
| **v4 val replay** | boreal val positive (heinola = ONE burn) | 111 | 3,432 | video-level only | **no** — clip-level detection only | no frame labels |
| **v4 val replay** | boreal non-smoke (14 heinola + 1 karkkila) | 15 | 471 | 0 | yes | n/a |
| **v4 val replay** | firesense smoke_neg | 9 | 292 | 0 | yes | n/a — **contaminated** (see 3C) |

Deliberately EXCLUDED from training, with reasons recorded in the manifest:
boreal (104 seqs / 3,338 frames) and firesense (13 / 530) video-level
POSITIVES (no frame truth; per-frame positives would be fabricated), and the
firesense smoke_neg frames (they are the only continuous negative replay pool
and are already background negatives inside v4 YOLO corpora B_G_v1 / C_EGT_v1 /
D_adapt_v1).

Merged training sets actually seen by the heads (from pod console):
- **v3-only**: 915 seqs / 22,170 frames (18,748 pos / 3,422 neg)
- **v4-extended**: 938 seqs / 33,107 frames (29,653 pos / 3,454 neg) — the
  extension is 33% of frames and 98% of the added frames are positive
- **v4-capped** (`AUSMOKE_KEEP=5`): 921 seqs / 24,773 frames (21,319 pos / 3,454 neg)

Features: the UNCHANGED v3 extractor `src/v3/trackd_features.py` (frozen expE
YOLO11m `models/trained/expE_mix_y11m_1280_v2/seg05/best.pt`, imgsz 1280,
hooks on layers 16/19/22 -> 194ch x 12x20 fp16 + per-frame max-conf). 16,571
unique v4 frames were shipped at NATIVE resolution, byte-identical
(`reports/v4/resize_equiv_check.json`: 7/7 checked frames give feature diff
0.0). A pre-resize path was built and rejected because JPEG q95 re-encode alone
moved per-frame max-conf by up to 0.050.

Timestamps: boreal/firesense are 1 fps. AuSmoke `ts_or_order` is a filename
offset with unrecorded fps; the dt feature treats it as ~1 fps (documented
approximation) and **no per-hour rate is emitted for AuSmoke**. Cadence shift
disclosed: a T<=5 window spans ~5 s on v4 footage vs ~2 min on pyro-sdis
(median 29 s).

## 2. Systems compared and exact parameter counts

| id | system | trained here? | learned params | notes |
|---|---|---|---|---|
| a | `s1_expE` — single-frame detector baseline | no | ~20.1 M (YOLO11m) | per-frame max det conf; shared detector pass for everything below |
| a' | `s1_v4a_ctrlE` — best v4 YOLO arm on sdis val (expV4A_ctrl) | no (scores only) | ~20.1 M | `models/v4/evals/v4evalr2/v4a__sdisval.jsonl`; arms C/E/Dada unfinished at eval time |
| a'' | `s1_v4B1_dinoL` — v4 frozen DINOv3-L + FRCNN head | no (scores only) | 325.4 M total / 22.3 M trainable | `models/v4/evals/v4B1_frozen_L/sdis_val_preds.jsonl` |
| b | `t1_*` — learned single-frame readout | **yes**, 2 seeds x 3 recipes | **281,153** | identical module to (d) run with a 1-frame window; ConvGRU cell (223,104 of the params) exercised for one step |
| c | `s2_persist_3of5` — deployed persistence rule | no | 0 learned (K=3, N=5, thr 0.15, single 0.30) | scored as the 3rd-largest expE conf in the causal last-5 window; 0 for <3 frames of history |
| d | `tmp_*` — learned causal temporal head (ConvGRU, T<=5) | **yes**, 2 seeds x 3 recipes | **281,153** | encoder 49,664 + ConvGRU 223,104 + MLP 8,385; valid at T=1 by masking; dt channel; stride/drop augmentation as v3 |

Recipes: `v3only` (v3 checkpoints, unchanged), `v4` (full extension), `v4cap`
(extension capped to 5 AuSmoke seqs). 12 epochs, bs 256, lr 1e-3 cosine,
pos_weight = n_neg/n_pos, best-on-val-AP selection. Val AP of selected
checkpoints: v4 temporal 0.9388 / 0.9457, v4 t1-only 0.9473 / 0.9501, v4cap
temporal 0.9426 / 0.9430, v4cap t1-only 0.9469 / 0.9426, v3 temporal 0.9404 /
0.9391, v3 t1-only 0.9431 / 0.9399.

## 3. Results

### 3A. pyro-sdis val, frame level, matched FPR (headline)

7,335 frames (6,177 pos / 1,158 neg; 560 "clean" negs with no positive in the
past-5 window). Thresholds matched on the all_neg pool. Seed-mean recall:

| system | PR-AUC | R@FPR0.5% | R@FPR1% | R@FPR2% | R@FPR5% | R@FPR10% | beats persistence | beats single-frame |
|---|---|---|---|---|---|---|---|---|
| s1 expE (single frame) | 0.9351 | 0.035 | 0.062 | 0.100 | 0.205 | 0.340 | — | — |
| s1 v4a_ctrlE | 0.9364 | 0.041 | 0.058 | 0.110 | 0.219 | 0.337 | — | — |
| s1 v4B1 DINO-L | 0.9327 | 0.054 | 0.070 | 0.108 | 0.197 | 0.332 | — | — |
| s2 persistence 3-of-5 | 0.9169 | 0.012 | 0.051 | 0.083 | 0.163 | 0.275 | — | — |
| temporal, v3-only (2 seeds) | 0.9398 | 0.036 | 0.077 | 0.166 | 0.248 | 0.392 | 10/10 | 9/10 |
| temporal, v4-extended | 0.9423 | 0.035 | 0.080 | 0.136 | 0.269 | 0.406 | 10/10 | 9/10 |
| temporal, v4-capped | 0.9428 | 0.043 | 0.074 | 0.145 | 0.276 | 0.415 | 10/10 | 8/10 |
| t1-only readout, v3-only | 0.9415 | 0.045 | 0.059 | 0.120 | 0.252 | 0.413 | 10/10 | 9/10 |
| t1-only readout, v4-extended | 0.9487 | 0.037 | 0.079 | 0.164 | 0.334 | 0.481 | 10/10 | 8/10 |
| t1-only readout, v4-capped | 0.9447 | 0.029 | 0.078 | 0.136 | 0.278 | 0.438 | 10/10 | 8/10 |

"beats" = cells (seed x FPR target) where recall exceeds the baseline.
Per-seed numbers, the clean_neg pool, episode detection/latency, and the
deployed operating point (confirmed-only FPR 0.535 / recall 0.834;
+provisional 0.587 / 0.898 on these embedded negatives) are in the appendix.

Reading: the v4-extended heads look BETTER on this table (the t1-only
v4-extended row is the best learned system here). Sections 3B/3C show why that
is not the whole story.

### 3A'. First-frame (T=1) behaviour

266 val sequences, 185 begin on a positive frame. Recall on those first frames
at each system's FPR-5% threshold:

| system | first-frame recall | all-frame recall |
|---|---|---|
| s1 expE | 0.076 | 0.205 |
| s2 persistence | 0.000 (cannot confirm with <3 frames) | 0.163 |
| temporal v3-only s0 / s1 | 0.016 / 0.005 | 0.250 / 0.246 |
| temporal v4-extended s0 / s1 | 0.000 / 0.038 | 0.258 / 0.281 |
| temporal v4-capped s0 / s1 | 0.011 / 0.027 | 0.270 / 0.282 |
| t1-only v4-capped s0 | **0.119** | 0.290 |

The temporal head satisfies "T=1 must work" (finite output, no buffer), but
its first-frame score distribution sits below its steady-state one, so at a
single threshold it is LESS sensitive on frame 1 than the raw detector. A
per-history-length threshold or the t1-only readout would fix this.

### 3B. v4 val replay, per source

Thresholds carried over from 3A (pyro-sdis-matched FPR 1%); the v4 replay has
no frame-truth FPR pool of its own large enough to match on.

**AuSmoke val positives — 4 units, ~3 independent events, 1,471 frames, all
positive, no pre-onset segment, wall clock unknown.** Clips alerted / recall on
positive frames at the pyro-sdis-matched FPR-1% / 5% thresholds:

| system | @FPR1% | @FPR5% |
|---|---|---|
| s1 expE | 2/4, r=0.003 | 2/4, r=0.022 |
| s2 persistence | 1/4, r=0.001 | 2/4, r=0.017 |
| temporal v3-only (s0 / s1) | 1/4, 0.003 / 0/4, 0.000 | 3/4, 0.032 / 3/4, 0.020 |
| temporal v4-extended | 2/4, 0.033 / 1/4, 0.007 | 3/4, 0.114 / 3/4, 0.077 |
| temporal v4-capped | 3/4, 0.035 / 1/4, 0.031 | 3/4, 0.106 / 3/4, 0.114 |
| t1-only v4-extended | 0/4, 0.000 / 1/4, 0.018 | 4/4, 0.104 / 3/4, 0.143 |
| deployed rule (prov OR conf, 0.30 / 0.15) | 4/4, frame alert 0.682 | — |

AuSmoke smoke is low-confidence for expE-derived features: at pyro-sdis-tight
thresholds every system misses >85% of these positive frames; only the
deployed (loose) operating point covers them. No onset latency is reported —
none is observable (ONSET_AUDIT.md).

**Boreal val positives — 111 clips from ONE burn (heinola), video-level truth
only.** Clip-level alert rate at FPR-1% threshold: s1 expE 0.162, s2 0.153,
temporal v3-only 0.009 / 0.018, temporal v4-extended **0.991 / 0.991**,
temporal v4-capped 0.766 / 0.928, t1-only v4-extended 0.378 / 0.982,
deployed rule 0.982. Do NOT read the v4-extended 99% as recall: the same heads
alert on 43-47% of frames of the boreal NON-smoke clips (next table).

### 3C. False-alert episodes per hour on continuous negative footage

Episode = maximal run of alert-on frames; runs closer than the deployed 60 s
cooldown merge into one. **Provisional and confirmed alerts both count.**
Exact Poisson 95% CI in brackets. Thresholds = pyro-sdis-matched FPR 1%.

**Headline pools (uncontaminated): boreal heinola non-smoke clips (14 clips,
439 frames, 0.122 h, VAL burn site) and pyro-sdis val all-negative sequences
(12 seqs, 219 frames, 1.52 h, real 29 s cadence).**

| system | boreal non-smoke: frame alert | boreal non-smoke: ep/h [CI] | pyro-sdis all-neg: ep/h [CI] |
|---|---|---|---|
| s1 expE | 0.000 | 0.0 [0, 30.3] | 0.0 [0, 2.4] |
| s2 persistence | 0.000 | 0.0 [0, 30.3] | 0.0 [0, 2.4] |
| temporal v3-only s0 / s1 | 0.000 / 0.000 | 0.0 / 0.0 | 0.0 / 0.0 |
| t1-only v3-only s0 / s1 | 0.000 / 0.000 | 0.0 / 0.0 | 0.0 / 0.0 |
| temporal **v4-extended** s0 / s1 | **0.465 / 0.458** | **90.2 [45, 161] / 73.8 [34, 140]** | 0.0 / 0.0 |
| t1-only **v4-extended** s0 / s1 | **0.282 / 0.474** | **41.0 [13, 96] / 73.8 [34, 140]** | 0.0 / 0.0 |
| temporal v4-capped s0 / s1 | 0.175 / 0.296 | 49.2 [18, 107] / 57.4 [23, 118] | 0.0 / 0.0 |
| t1-only v4-capped s0 / s1 | 0.057 / 0.169 | 41.0 [13, 96] / 41.0 [13, 96] | 0.0 / 0.0 |
| deployed rule (0.30 / 0.15, prov OR conf) | 0.302 | 73.8 [34, 140] | **3.3 [1.1, 7.7]** |

- On pyro-sdis negatives (the domain everything was trained on) every
  matched system is at 0 episodes in 1.5 h (upper CI 2.4/h); the deployed
  loose operating point produces 3.3 episodes/h [1.1, 7.7] on the same
  footage, i.e. the deployed thresholds are far looser than FPR 1%.
- On boreal non-smoke footage the v4-extended heads produce 40-90 false
  episodes per hour — at 1 fps and 30 s clips that is "alerting on most
  clips". The v3-only heads produce none. The boreal pool is only 7 minutes
  of footage (wide CIs), but the frame-alert fractions (0.28-0.47 vs 0.000)
  are not a small-sample artefact.
- Including the karkkila (TRAIN burn site) clip changes nothing material
  (appendix).

**Contaminated pool — firesense smoke_neg (9 videos, 292 frames, 0.081 h).**
These frames are background negatives inside v4 YOLO CORPUS_B_G_v1 (292
lines), C_EGT_v1 (584) and D_adapt_v1 (39). Any false-alert number here is
OPTIMISTIC for YOLO arms B/C/Dada and anything derived from them. The heads
in this report read frozen expE (v3) features and expE never saw these frames,
so for THESE numbers the contamination is not active — the flag stays on the
table regardless. Frame alert at FPR 1%: v3-only heads 0.000, v4-extended
heads **0.83-0.98**, v4-capped 0.00-0.04, deployed rule 0.116 (37 ep/h).

### 3D. YOLO every frame + independent DINO pass every Nth frame

Measured batch-1 fp32 latency on one L40S: YOLO11m expE **11.24 ms**, frozen
DINOv3-L B1 **156.43 ms** (13.9x). N for <=20% added compute = **70**. Policy
simulated on pyro-sdis val sequences: alert_t = (yolo_t > thr_y) OR (t is a
sweep frame AND dino_t > thr_d), both thresholds matched to the same
per-frame FPR on the 1,158 val negatives, averaged over all N sweep phases.
Decisive column: YOLO alone re-thresholded to the SAME integer false-alert
frame count the combination actually spends. Two DINO sources: v3 B1
(`reports/v3/eval_B1_val.raw.parquet`) and the v4 frozen-L control
(`reports/v4/eval_v4B1_frozen_L_sdis_val.raw.parquet`).

| FPR target | YOLO alone: ep-det (FP frames) | N | added compute | combo ep-det (v3 B1) | YOLO @ same FP | delta (v3 B1) | delta (v4 B1) | added FP frames | median latency delta |
|---|---|---|---|---|---|---|---|---|---|
| 1% | 0.175 (11) | 70 | +19.9% | 0.183 | 0.175 | **+0.008** | +0.005 | +0.2 | 0 |
| 1% | | 20 | +69.6% | 0.198 | 0.181 | +0.017 | +0.009 | +0.6 | 0 |
| 2% | 0.281 (23) | 70 | +19.9% | 0.290 | 0.281 | **+0.008** | +0.007 | +0.2 | +1 frame (composition) |
| 2% | | 20 | +69.6% | 0.305 | 0.284 | +0.021 | +0.017 | +0.9 | +1 |
| 5% | 0.461 (57) | 70 | +19.9% | 0.465 | 0.461 | **+0.005** | +0.004 | +0.6 | +0.5 |
| 5% | | 20 | +69.6% | 0.476 | 0.467 | +0.009 | +0.006 | +2.1 | +0.5 |

Episode denominator: 469 positive episodes. 91.4% of val sequences (median
15.5 frames) never receive a DINO pass at N=70; 58.7% at N=20.

**Verdict, stated plainly:** the combination beats YOLO alone at matched FP in
every cell tested (2 sources x 5 targets x 6 values of N), so the v3 finding
that frozen DINO adds event COVERAGE survives. But at the <=20% compute cap
the gain is 2-4 episodes out of 469 (+0.4 to +0.8 pt), of the same order as
the <=0.2 fractional false-alert frame the integer-rounded matching does not
charge, and the v4 frozen-L control is slightly weaker than v3 B1. Added
latency: one 156 ms DINO pass per 70 frames; if run synchronously the swept
frame's decision is delayed ~145 ms, if asynchronously nothing is delayed.
The positive median-latency delta at 2% is a composition effect (the extra
episodes found are late ones); the OR rule can never fire later than YOLO
alone on any frame. **This does not justify deploying a DINO sweep on the
current evidence**; it would need either a much cheaper DINO pass (fp16 /
TensorRT — not measured) or footage where YOLO alone misses whole events,
which pyro-sdis val does not contain in quantity.

## 4. LIMITATIONS

1. **v4 added NO new observed-onset supervision.** AuSmoke observes 0 of 26
   background->smoke transitions (all clips start with smoke visible); boreal
   and firesense positives are video-level only and no frame labels were
   invented. Every onset / latency claim in this report — and every claim
   that the temporal head beats persistence — still rests exclusively on the
   pyro-sdis train/val sequences (v3), and, for event-level replay, on FIgLib
   which is eval-only. AuSmoke contributed persistence supervision only.
2. **The extension's positive-only composition produced a domain shortcut**
   (3B/3C). The heads that look best on pyro-sdis val are the ones that fail
   worst off-domain. Model selection on pyro-sdis val alone cannot see this;
   the v4 replay negatives were essential and are tiny (7 min boreal, 5 min
   firesense).
3. **Denominators are small and dependent.** AuSmoke val = 4 units but ~3
   independent events (axis fixed-camera pair shares a camera); boreal val =
   ONE burn (heinola); boreal non-smoke = 14 thirty-second clips; firesense
   negatives = 9 short videos. Episodes/hour on these pools extrapolate from
   minutes of footage — read the CIs, not the point estimates.
4. **Contamination flags:** firesense smoke_neg frames are in v4 YOLO train
   corpora B_G_v1 / C_EGT_v1 / D_adapt_v1 (optimistic for those arms); the
   karkkila non-smoke clip is from a TRAIN burn site; expE itself was trained
   on pyro-sdis train-split images, so train-time features are optimistically
   confident (v3 caveat, mitigated by val selection).
5. **Cadence and time semantics.** v4 footage is ~1 fps vs pyro-sdis median
   29 s; a 5-frame window means very different wall-clock context. AuSmoke
   fps was never recorded, so its dt feature is an assumption and no per-hour
   rate is reported for it.
6. **Thresholds on the v4 replay are transferred**, not matched locally —
   there is no frame-truth negative pool of usable size in the v4 val
   package. The deployed operating point (0.30 / 0.15) is reported as a
   discrete point for context only.
7. **YOLO arms C / E / Dada were still training** when this evaluation ran;
   the single-frame v4 baseline is arm A (existing-data control), which is
   also the best arm on sdis val among those scored (A / B / Dpre).
8. **DINO sweep** uses fp32 batch-1 latencies without fp16/TensorRT/compile
   on either model; the cost ratio (13.9x) could change materially with
   engineering that was not done. The matched-FP comparison rounds to integer
   FP frames (disclosed in 3D).
9. Nothing here touches test material. No production system was modified.

## 5. Artifacts and reproduction

- Manifest + converted sequences: `data/v4/sequences/trackd/{MANIFEST_trackd_v4.json, sequences_v4_train_ext.json, sequences_v4_val_replay.json, stage_plan.json}`
- Frames shipped: `data/v4/stage/trackd_v4.tar` (15.95 GB, sha256 0c7bab27...; S3 `v4/jobs/trackd/trackd_v4.tar`)
- Features (194ch 12x20 fp16, same space as v3): `data/v4/features/{train_ext,val_replay}/*.npz` (from `data/v4/features_tars/feats_v4.tgz`, 1.39 GB; S3 `v4/runs/trackd/feats_v4.tgz`)
- Heads: `models/v4/trackd/heads/unpacked/{v4_temporal_s0,v4_temporal_s1,v4_t1only_s0,v4_t1only_s1}/best.pt`, `models/v4/trackd/headsbal/unpacked/v4cap_*/best.pt`; index in `models/v4/trackd/RUNS.json`
- Predictions: `models/v4/trackd/preds/preds_sdis_val.npz` (7,335 frames x 16 systems + labels), `preds_val_replay_v4.npz` (5,666 frames x 14 systems)
- Eval JSON: `reports/v4/trackd_v4_eval.json`; resize check: `reports/v4/resize_equiv_check.json`
- Cluster jobs (green g6e, 1 node each, `smoke-detect:v6`): `tianrgua-extxv4s1-vgw5` (extract, 757 s), `tianrgua-traheadss1-phsy` (4 heads, 1007 s), `tianrgua-traheadsbals1-qqwh` (4 capped heads, 841 s). Registry: `res-v4trackdxt`, `res-v4trackdtr`, `res-v4trackdbal`, all released.
- Code: `scripts/v4/build_trackd_v4.py`, `scripts/v4/verify_resize_equiv.py`, `cluster/v4/{orchestrate_trackd_v4.py, trackd_v4_entry.sh}`, `src/v4/trackd_eval_v4.py`, `scripts/v4/trackd_report_tables.py`. `src/v3/trackd_features.py` and `src/v3/trackd_temporal.py` were reused unchanged.

Reproduce the evaluation (CPU, ~5 min):
```
python -m src.v4.trackd_eval_v4 --features-v3 data/v3/features --features-v4 data/v4/features \
  --seq-val-replay data/v4/sequences/trackd/sequences_v4_val_replay.json \
  --head v4_temporal_s0=models/v4/trackd/heads/unpacked/v4_temporal_s0/best.pt [... 11 more --head] \
  --ext-single v4a_ctrlE=models/v4/evals/v4evalr2/v4a__sdisval.jsonl \
  --ext-single v4B1_dinoL=models/v4/evals/v4B1_frozen_L/sdis_val_preds.jsonl \
  --dino-sweep b1_v3=reports/v3/eval_B1_val.raw.parquet \
  --dino-sweep b1_v4frozenL=reports/v4/eval_v4B1_frozen_L_sdis_val.raw.parquet \
  --device cpu --out reports/v4/trackd_v4_eval.json --pred-dir models/v4/trackd/preds
python scripts/v4/trackd_report_tables.py reports/v4/trackd_v4_eval.json
```

---

## Appendix — full tables (generated, `scripts/v4/trackd_report_tables.py`)

### A. pyro-sdis val, frame level, matched FPR on val negatives
Frames 7335 (pos 6177 / neg 1158; clean neg 560); tiny-GT pos frames 5881


**pool = all_neg**

| system | params | PR-AUC | ROC-AUC | R@FPR0.5% | R@FPR1% | R@FPR2% | R@FPR5% | R@FPR10% | tinyR@FPR5% |
|---|---|---|---|---|---|---|---|---|---|
| s1_expE | 0 (rule) | 0.9351 | 0.7685 | 0.0353 | 0.0622 | 0.1004 | 0.2053 | 0.3400 | 0.2095 |
| s1_v4a_ctrlE | 0 (rule) | 0.9364 | 0.7718 | 0.0414 | 0.0580 | 0.1101 | 0.2194 | 0.3371 | 0.2234 |
| s1_v4B1_dinoL | 0 (rule) | 0.9327 | 0.7535 | 0.0537 | 0.0699 | 0.1078 | 0.1967 | 0.3324 | 0.1988 |
| s2_persist_3of5 | 0 (rule) | 0.9169 | 0.7056 | 0.0123 | 0.0515 | 0.0826 | 0.1632 | 0.2752 | 0.1632 |
| t1_v3only_t1only_s0 | 281,153 | 0.9431 | 0.7798 | 0.0503 | 0.0639 | 0.1264 | 0.2589 | 0.4266 | 0.2573 |
| t1_v3only_t1only_s1 | 281,153 | 0.9399 | 0.7761 | 0.0401 | 0.0541 | 0.1130 | 0.2449 | 0.3987 | 0.2450 |
| t1_v4_t1only_s0 | 281,153 | 0.9473 | 0.7980 | 0.0448 | 0.0599 | 0.1305 | 0.3225 | 0.4646 | 0.3192 |
| t1_v4_t1only_s1 | 281,153 | 0.9501 | 0.8088 | 0.0283 | 0.0989 | 0.1965 | 0.3463 | 0.4967 | 0.3333 |
| t1_v4cap_t1only_s0 | 281,153 | 0.9469 | 0.7932 | 0.0380 | 0.1030 | 0.1463 | 0.2899 | 0.4412 | 0.2906 |
| t1_v4cap_t1only_s1 | 281,153 | 0.9426 | 0.7897 | 0.0204 | 0.0539 | 0.1260 | 0.2671 | 0.4345 | 0.2644 |
| tmp_v3only_temporal_s0 | 281,153 | 0.9404 | 0.7736 | 0.0423 | 0.0711 | 0.1512 | 0.2498 | 0.3983 | 0.2483 |
| tmp_v3only_temporal_s1 | 281,153 | 0.9391 | 0.7667 | 0.0299 | 0.0835 | 0.1813 | 0.2462 | 0.3861 | 0.2460 |
| tmp_v4_temporal_s0 | 281,153 | 0.9388 | 0.7662 | 0.0400 | 0.0737 | 0.1350 | 0.2576 | 0.3983 | 0.2472 |
| tmp_v4_temporal_s1 | 281,153 | 0.9457 | 0.7955 | 0.0303 | 0.0860 | 0.1376 | 0.2812 | 0.4133 | 0.2761 |
| tmp_v4cap_temporal_s0 | 281,153 | 0.9426 | 0.7810 | 0.0291 | 0.0565 | 0.1439 | 0.2699 | 0.4208 | 0.2683 |
| tmp_v4cap_temporal_s1 | 281,153 | 0.9430 | 0.7781 | 0.0562 | 0.0908 | 0.1462 | 0.2815 | 0.4102 | 0.2758 |

**pool = clean_neg**

| system | params | PR-AUC | ROC-AUC | R@FPR0.5% | R@FPR1% | R@FPR2% | R@FPR5% | R@FPR10% | tinyR@FPR5% |
|---|---|---|---|---|---|---|---|---|---|
| s1_expE | 0 (rule) | 0.9351 | 0.7685 | 0.1109 | 0.1412 | 0.2004 | 0.2895 | 0.4081 | 0.2935 |
| s1_v4a_ctrlE | 0 (rule) | 0.9364 | 0.7718 | 0.1101 | 0.1682 | 0.2213 | 0.2776 | 0.4096 | 0.2811 |
| s1_v4B1_dinoL | 0 (rule) | 0.9327 | 0.7535 | 0.0661 | 0.0759 | 0.1308 | 0.2075 | 0.3921 | 0.2100 |
| s2_persist_3of5 | 0 (rule) | 0.9169 | 0.7056 | 0.2056 | 0.2192 | 0.2368 | 0.3330 | 0.4277 | 0.3340 |
| t1_v3only_t1only_s0 | 281,153 | 0.9431 | 0.7798 | 0.0478 | 0.0563 | 0.2254 | 0.3864 | 0.4894 | 0.3884 |
| t1_v3only_t1only_s1 | 281,153 | 0.9399 | 0.7761 | 0.0652 | 0.1340 | 0.1829 | 0.3144 | 0.4442 | 0.3151 |
| t1_v4_t1only_s0 | 281,153 | 0.9473 | 0.7980 | 0.0293 | 0.0494 | 0.1305 | 0.3903 | 0.5179 | 0.3860 |
| t1_v4_t1only_s1 | 281,153 | 0.9501 | 0.8088 | 0.0283 | 0.1478 | 0.2270 | 0.3764 | 0.5132 | 0.3629 |
| t1_v4cap_t1only_s0 | 281,153 | 0.9469 | 0.7932 | 0.0986 | 0.1227 | 0.2513 | 0.3691 | 0.5019 | 0.3702 |
| t1_v4cap_t1only_s1 | 281,153 | 0.9426 | 0.7897 | 0.0932 | 0.1421 | 0.2650 | 0.4010 | 0.5080 | 0.4003 |
| tmp_v3only_temporal_s0 | 281,153 | 0.9404 | 0.7736 | 0.0476 | 0.1188 | 0.1690 | 0.2624 | 0.4457 | 0.2612 |
| tmp_v3only_temporal_s1 | 281,153 | 0.9391 | 0.7667 | 0.1153 | 0.1813 | 0.2257 | 0.3623 | 0.4565 | 0.3647 |
| tmp_v4_temporal_s0 | 281,153 | 0.9388 | 0.7662 | 0.0236 | 0.0439 | 0.1138 | 0.2815 | 0.4601 | 0.2712 |
| tmp_v4_temporal_s1 | 281,153 | 0.9457 | 0.7955 | 0.0172 | 0.1684 | 0.2067 | 0.3290 | 0.4829 | 0.3234 |
| tmp_v4cap_temporal_s0 | 281,153 | 0.9426 | 0.7810 | 0.0518 | 0.0625 | 0.2380 | 0.3361 | 0.4557 | 0.3340 |
| tmp_v4cap_temporal_s1 | 281,153 | 0.9430 | 0.7781 | 0.0562 | 0.1462 | 0.2367 | 0.4059 | 0.5103 | 0.4013 |

**episode detection + onset latency (thresholds matched on FRAME FPR, all_neg pool)**

| system | episodes | det@FPR1% | lat@FPR1% | det@FPR5% | lat@FPR5% |
|---|---|---|---|---|---|
| s1_expE | 469 | 0.1748 | 4.0 | 0.4606 | 1.5 |
| s1_v4a_ctrlE | 469 | 0.1898 | 4.0 | 0.4840 | 1.0 |
| s1_v4B1_dinoL | 469 | 0.1663 | 4.5 | 0.3348 | 2.0 |
| s2_persist_3of5 | 469 | 0.1173 | 6.0 | 0.2665 | 3.0 |
| t1_v3only_t1only_s0 | 469 | 0.1343 | 4.0 | 0.3795 | 2.0 |
| t1_v3only_t1only_s1 | 469 | 0.1429 | 4.0 | 0.4136 | 1.0 |
| t1_v4_t1only_s0 | 469 | 0.0981 | 6.5 | 0.4478 | 1.0 |
| t1_v4_t1only_s1 | 469 | 0.1855 | 5.0 | 0.4328 | 2.0 |
| t1_v4cap_t1only_s0 | 469 | 0.1620 | 3.5 | 0.4200 | 1.0 |
| t1_v4cap_t1only_s1 | 469 | 0.1322 | 4.0 | 0.4009 | 2.0 |
| tmp_v3only_temporal_s0 | 469 | 0.1322 | 6.0 | 0.3710 | 2.0 |
| tmp_v3only_temporal_s1 | 469 | 0.1514 | 5.0 | 0.3731 | 2.0 |
| tmp_v4_temporal_s0 | 469 | 0.1087 | 6.0 | 0.3454 | 2.0 |
| tmp_v4_temporal_s1 | 469 | 0.1620 | 5.0 | 0.4136 | 2.0 |
| tmp_v4cap_temporal_s0 | 469 | 0.1023 | 7.0 | 0.3667 | 2.0 |
| tmp_v4cap_temporal_s1 | 469 | 0.1322 | 6.0 | 0.3646 | 2.0 |

**deployed discrete operating point**

| point | FPR all_neg | FPR clean_neg | recall pos frames | recall tiny |
|---|---|---|---|---|
| confirmed_only(s2>=0.15) | 0.5345 | 0.2768 | 0.8342 | 0.8383 |
| confirmed_or_provisional(s2>=0.15|s1>=0.30) | 0.5872 | 0.3589 | 0.8982 | 0.9043 |

### B. v4 val replay, per source (thresholds from section A)


**ausmoke/positive** - 4 sequences / 1471 frames, wall clock UNKNOWN (original fps never recorded (order units) -> no per-hour rate is emitted for this source), truth_level=frame_boxes, frame_truth=True, units=4
  - CAVEAT: positive from frame 0 — NO pre-onset segment exists; contributes nothing to the negative-footage pool

| system | thr@FPR1% | clip alert | frame alert | episodes | ep/h | recall pos frames |
|---|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.5000 (2/4) | 0.0027 | 3 | - | 0.0027 |
| s2_persist_3of5 | 0.7874 | 0.2500 (1/4) | 0.0014 | 1 | - | 0.0014 |
| t1_v3only_t1only_s0 | 0.9367 | 0.7500 (3/4) | 0.0061 | 5 | - | 0.0061 |
| t1_v3only_t1only_s1 | 0.9601 | 0.0000 (0/4) | 0.0000 | 0 | - | 0.0 |
| t1_v4_t1only_s0 | 0.9951 | 0.0000 (0/4) | 0.0000 | 0 | - | 0.0 |
| t1_v4_t1only_s1 | 0.9637 | 0.2500 (1/4) | 0.0184 | 7 | - | 0.0184 |
| t1_v4cap_t1only_s0 | 0.9960 | 0.5000 (2/4) | 0.0163 | 9 | - | 0.0163 |
| t1_v4cap_t1only_s1 | 0.9347 | 0.5000 (2/4) | 0.0095 | 4 | - | 0.0095 |
| tmp_v3only_temporal_s0 | 0.9988 | 0.2500 (1/4) | 0.0027 | 1 | - | 0.0027 |
| tmp_v3only_temporal_s1 | 0.9556 | 0.0000 (0/4) | 0.0000 | 0 | - | 0.0 |
| tmp_v4_temporal_s0 | 0.9998 | 0.5000 (2/4) | 0.0333 | 3 | - | 0.0333 |
| tmp_v4_temporal_s1 | 0.9857 | 0.2500 (1/4) | 0.0068 | 4 | - | 0.0068 |
| tmp_v4cap_temporal_s0 | 0.9931 | 0.7500 (3/4) | 0.0347 | 8 | - | 0.0347 |
| tmp_v4cap_temporal_s1 | 0.9750 | 0.2500 (1/4) | 0.0306 | 5 | - | 0.0306 |
| DEPLOYED (prov OR conf) | - | 1.0000 | 0.6825 | 35 | - | - |

**boreal/negative_footage** - 15 sequences / 471 frames, 0.131 h wall clock, truth_level=videolabel, frame_truth=True, units=2
  - CAVEAT: clip from a TRAIN burn site (evo/karkkila) — background seen at train time; weaker negative evidence than val-site footage

| system | thr@FPR1% | clip alert | frame alert | episodes | ep/h |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.0000 (0/15) | 0.0000 | 0 | 0.00 [0.0, 28.2] |
| s2_persist_3of5 | 0.7874 | 0.0000 (0/15) | 0.0000 | 0 | 0.00 [0.0, 28.2] |
| t1_v3only_t1only_s0 | 0.9367 | 0.0000 (0/15) | 0.0000 | 0 | 0.00 [0.0, 28.2] |
| t1_v3only_t1only_s1 | 0.9601 | 0.0000 (0/15) | 0.0000 | 0 | 0.00 [0.0, 28.2] |
| t1_v4_t1only_s0 | 0.9951 | 0.3333 (5/15) | 0.2633 | 5 | 38.23 [12.41, 89.21] |
| t1_v4_t1only_s1 | 0.9637 | 0.6000 (9/15) | 0.4416 | 9 | 68.81 [31.46, 130.62] |
| t1_v4cap_t1only_s0 | 0.9960 | 0.3333 (5/15) | 0.0531 | 5 | 38.23 [12.41, 89.21] |
| t1_v4cap_t1only_s1 | 0.9347 | 0.3333 (5/15) | 0.1571 | 5 | 38.23 [12.41, 89.21] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.0000 (0/15) | 0.0000 | 0 | 0.00 [0.0, 28.2] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.0000 (0/15) | 0.0000 | 0 | 0.00 [0.0, 28.2] |
| tmp_v4_temporal_s0 | 0.9998 | 0.7333 (11/15) | 0.4331 | 11 | 84.10 [41.98, 150.47] |
| tmp_v4_temporal_s1 | 0.9857 | 0.6000 (9/15) | 0.4268 | 9 | 68.81 [31.46, 130.62] |
| tmp_v4cap_temporal_s0 | 0.9931 | 0.4000 (6/15) | 0.1635 | 6 | 45.87 [16.83, 99.84] |
| tmp_v4cap_temporal_s1 | 0.9750 | 0.4667 (7/15) | 0.2760 | 7 | 53.52 [21.52, 110.27] |
| DEPLOYED (prov OR conf) | - | 0.6000 | 0.3015 | 9 | 68.81 [31.46, 130.62] | - |

**boreal/positive** - 111 sequences / 3432 frames, 0.953 h wall clock, truth_level=videolabel, frame_truth=False, units=1

| system | thr@FPR1% | clip alert | frame alert | episodes | ep/h |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.1622 (18/111) | 0.0457 | 18 | 18.88 [11.19, 29.84] |
| s2_persist_3of5 | 0.7874 | 0.1532 (17/111) | 0.0600 | 17 | 17.83 [10.39, 28.55] |
| t1_v3only_t1only_s0 | 0.9367 | 0.0090 (1/111) | 0.0009 | 1 | 1.05 [0.03, 5.84] |
| t1_v3only_t1only_s1 | 0.9601 | 0.0000 (0/111) | 0.0000 | 0 | 0.00 [0.0, 3.87] |
| t1_v4_t1only_s0 | 0.9951 | 0.3784 (42/111) | 0.0664 | 42 | 44.06 [31.75, 59.55] |
| t1_v4_t1only_s1 | 0.9637 | 0.9820 (109/111) | 0.7946 | 109 | 114.34 [93.88, 137.93] |
| t1_v4cap_t1only_s0 | 0.9960 | 0.3514 (39/111) | 0.1072 | 39 | 40.91 [29.09, 55.93] |
| t1_v4cap_t1only_s1 | 0.9347 | 0.5495 (61/111) | 0.2506 | 61 | 63.99 [48.95, 82.2] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.0090 (1/111) | 0.0003 | 1 | 1.05 [0.03, 5.84] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.0180 (2/111) | 0.0012 | 2 | 2.10 [0.25, 7.58] |
| tmp_v4_temporal_s0 | 0.9998 | 0.9910 (110/111) | 0.8412 | 110 | 115.39 [94.84, 139.07] |
| tmp_v4_temporal_s1 | 0.9857 | 0.9910 (110/111) | 0.7637 | 110 | 115.39 [94.84, 139.07] |
| tmp_v4cap_temporal_s0 | 0.9931 | 0.7658 (85/111) | 0.3677 | 85 | 89.16 [71.22, 110.25] |
| tmp_v4cap_temporal_s1 | 0.9750 | 0.9279 (103/111) | 0.5828 | 103 | 108.05 [88.19, 131.04] |
| DEPLOYED (prov OR conf) | - | 0.9820 | 0.8083 | 109 | 114.34 [93.88, 137.93] | - |

**firesense/negative_footage** - 9 sequences / 292 frames, 0.081 h wall clock, truth_level=videolabel, frame_truth=True, units=9
  - CAVEAT: frames used as background negatives in v4 YOLO train corpora (e.g. CORPUS_B_G_v1: 292 firesense background_neg lines) — false-alert numbers are OPTIMISTIC for models trained on those corpora

| system | thr@FPR1% | clip alert | frame alert | episodes | ep/h |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.0000 (0/9) | 0.0000 | 0 | 0.00 [0.0, 45.49] |
| s2_persist_3of5 | 0.7874 | 0.0000 (0/9) | 0.0000 | 0 | 0.00 [0.0, 45.49] |
| t1_v3only_t1only_s0 | 0.9367 | 0.0000 (0/9) | 0.0000 | 0 | 0.00 [0.0, 45.49] |
| t1_v3only_t1only_s1 | 0.9601 | 0.0000 (0/9) | 0.0000 | 0 | 0.00 [0.0, 45.49] |
| t1_v4_t1only_s0 | 0.9951 | 0.7778 (7/9) | 0.8253 | 7 | 86.31 [34.7, 177.84] |
| t1_v4_t1only_s1 | 0.9637 | 1.0000 (9/9) | 0.9829 | 9 | 110.97 [50.74, 210.66] |
| t1_v4cap_t1only_s0 | 0.9960 | 0.0000 (0/9) | 0.0000 | 0 | 0.00 [0.0, 45.49] |
| t1_v4cap_t1only_s1 | 0.9347 | 0.0000 (0/9) | 0.0000 | 0 | 0.00 [0.0, 45.49] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.0000 (0/9) | 0.0000 | 0 | 0.00 [0.0, 45.49] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.0000 (0/9) | 0.0000 | 0 | 0.00 [0.0, 45.49] |
| tmp_v4_temporal_s0 | 0.9998 | 1.0000 (9/9) | 0.9247 | 9 | 110.97 [50.74, 210.66] |
| tmp_v4_temporal_s1 | 0.9857 | 0.8889 (8/9) | 0.8836 | 8 | 98.64 [42.59, 194.37] |
| tmp_v4cap_temporal_s0 | 0.9931 | 0.1111 (1/9) | 0.0445 | 1 | 12.33 [0.31, 68.7] |
| tmp_v4cap_temporal_s1 | 0.9750 | 0.0000 (0/9) | 0.0000 | 0 | 0.00 [0.0, 45.49] |
| DEPLOYED (prov OR conf) | - | 0.3333 | 0.1164 | 3 | 36.99 [7.63, 108.1] | - |

### C. False-alert EPISODES per hour on continuous negative footage
Episode = maximal run of alert-on frames; runs separated by < 60.0s (deployed cooldown) count as one. Provisional AND confirmed both count. CI = exact Poisson 95%.


**pyrosdis_val_all_neg_sequences** - 12 seqs / 219 frames / 1.521 h
  pyro-sdis val sequences whose every frame is negative; real 29 s-cadence wall clock; expE never saw the val sites

| system | thr@FPR1% | ep/h @FPR1% | CI95 | ep/h @FPR5% | CI95 |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.0 | [0.0, 2.42] | 0.0 | [0.0, 2.42] |
| s1_v4a_ctrlE | 0.8000 | 0.0 | [0.0, 2.42] | 0.0 | [0.0, 2.42] |
| s1_v4B1_dinoL | 0.9858 | 0.0 | [0.0, 2.42] | 0.0 | [0.0, 2.42] |
| s2_persist_3of5 | 0.7874 | 0.0 | [0.0, 2.42] | 0.0 | [0.0, 2.42] |
| t1_v3only_t1only_s0 | 0.9367 | 0.0 | [0.0, 2.42] | 0.66 | [0.02, 3.66] |
| t1_v3only_t1only_s1 | 0.9601 | 0.0 | [0.0, 2.42] | 0.0 | [0.0, 2.42] |
| t1_v4_t1only_s0 | 0.9951 | 0.0 | [0.0, 2.42] | 0.66 | [0.02, 3.66] |
| t1_v4_t1only_s1 | 0.9637 | 0.0 | [0.0, 2.42] | 0.66 | [0.02, 3.66] |
| t1_v4cap_t1only_s0 | 0.9960 | 0.0 | [0.0, 2.42] | 0.0 | [0.0, 2.42] |
| t1_v4cap_t1only_s1 | 0.9347 | 0.0 | [0.0, 2.42] | 0.0 | [0.0, 2.42] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.0 | [0.0, 2.42] | 1.31 | [0.16, 4.75] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.0 | [0.0, 2.42] | 0.0 | [0.0, 2.42] |
| tmp_v4_temporal_s0 | 0.9998 | 0.0 | [0.0, 2.42] | 0.0 | [0.0, 2.42] |
| tmp_v4_temporal_s1 | 0.9857 | 0.0 | [0.0, 2.42] | 0.66 | [0.02, 3.66] |
| tmp_v4cap_temporal_s0 | 0.9931 | 0.0 | [0.0, 2.42] | 0.0 | [0.0, 2.42] |
| tmp_v4cap_temporal_s1 | 0.9750 | 0.0 | [0.0, 2.42] | 0.0 | [0.0, 2.42] |
| DEPLOYED (prov OR conf) | 0.30 / 0.15 | 3.29 | [1.07, 7.67] | - | - |

**boreal_val_nonsmoke_heinola_only** - 14 seqs / 439 frames / 0.122 h
  boreal heinola (VAL burn site) non-smoke clips, 1 fps

| system | thr@FPR1% | ep/h @FPR1% | CI95 | ep/h @FPR5% | CI95 |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.0 | [0.0, 30.25] | 8.2 | [0.21, 45.69] |
| s2_persist_3of5 | 0.7874 | 0.0 | [0.0, 30.25] | 8.2 | [0.21, 45.69] |
| t1_v3only_t1only_s0 | 0.9367 | 0.0 | [0.0, 30.25] | 8.2 | [0.21, 45.69] |
| t1_v3only_t1only_s1 | 0.9601 | 0.0 | [0.0, 30.25] | 8.2 | [0.21, 45.69] |
| t1_v4_t1only_s0 | 0.9951 | 41.0 | [13.31, 95.69] | 82.0 | [39.32, 150.81] |
| t1_v4_t1only_s1 | 0.9637 | 73.8 | [33.75, 140.1] | 82.0 | [39.32, 150.81] |
| t1_v4cap_t1only_s0 | 0.9960 | 41.0 | [13.31, 95.69] | 57.4 | [23.08, 118.27] |
| t1_v4cap_t1only_s1 | 0.9347 | 41.0 | [13.31, 95.69] | 65.6 | [28.32, 129.27] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.0 | [0.0, 30.25] | 8.2 | [0.21, 45.69] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.0 | [0.0, 30.25] | 8.2 | [0.21, 45.69] |
| tmp_v4_temporal_s0 | 0.9998 | 90.21 | [45.03, 161.4] | 106.61 | [56.76, 182.3] |
| tmp_v4_temporal_s1 | 0.9857 | 73.8 | [33.75, 140.1] | 73.8 | [33.75, 140.1] |
| tmp_v4cap_temporal_s0 | 0.9931 | 49.2 | [18.06, 107.09] | 57.4 | [23.08, 118.27] |
| tmp_v4cap_temporal_s1 | 0.9750 | 57.4 | [23.08, 118.27] | 82.0 | [39.32, 150.81] |
| DEPLOYED (prov OR conf) | 0.30 / 0.15 | 73.8 | [33.75, 140.1] | - | - |

**boreal_nonsmoke_all_incl_train_site** - 15 seqs / 471 frames / 0.131 h
  adds the karkkila clip, whose background is a TRAIN burn site

| system | thr@FPR1% | ep/h @FPR1% | CI95 | ep/h @FPR5% | CI95 |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.0 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| s2_persist_3of5 | 0.7874 | 0.0 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| t1_v3only_t1only_s0 | 0.9367 | 0.0 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| t1_v3only_t1only_s1 | 0.9601 | 0.0 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| t1_v4_t1only_s0 | 0.9951 | 38.22 | [12.41, 89.18] | 76.43 | [36.65, 140.56] |
| t1_v4_t1only_s1 | 0.9637 | 68.79 | [31.46, 130.58] | 76.43 | [36.65, 140.56] |
| t1_v4cap_t1only_s0 | 0.9960 | 38.22 | [12.41, 89.18] | 53.5 | [21.51, 110.24] |
| t1_v4cap_t1only_s1 | 0.9347 | 38.22 | [12.41, 89.18] | 61.15 | [26.4, 120.48] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.0 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.0 | [0.0, 28.2] | 7.64 | [0.19, 42.59] |
| tmp_v4_temporal_s0 | 0.9998 | 84.08 | [41.97, 150.44] | 99.36 | [52.91, 169.91] |
| tmp_v4_temporal_s1 | 0.9857 | 68.79 | [31.46, 130.58] | 68.79 | [31.46, 130.58] |
| tmp_v4cap_temporal_s0 | 0.9931 | 45.86 | [16.83, 99.82] | 53.5 | [21.51, 110.24] |
| tmp_v4cap_temporal_s1 | 0.9750 | 53.5 | [21.51, 110.24] | 76.43 | [36.65, 140.56] |
| DEPLOYED (prov OR conf) | 0.30 / 0.15 | 68.79 | [31.46, 130.58] | - | - |

**firesense_negatives_CONTAMINATED** **[CONTAMINATED - see caveat]** - 9 seqs / 292 frames / 0.081 h
  these 292 frames are background negatives inside v4 YOLO CORPUS_B_G_v1 / C_EGT_v1 / D_adapt_v1 -> optimistic for any model trained on those corpora. The temporal heads here read frozen expE (v3) features and expE never saw them, but the numbers are still flagged.

| system | thr@FPR1% | ep/h @FPR1% | CI95 | ep/h @FPR5% | CI95 |
|---|---|---|---|---|---|
| s1_expE | 0.7997 | 0.0 | [0.0, 45.48] | 0.0 | [0.0, 45.48] |
| s2_persist_3of5 | 0.7874 | 0.0 | [0.0, 45.48] | 0.0 | [0.0, 45.48] |
| t1_v3only_t1only_s0 | 0.9367 | 0.0 | [0.0, 45.48] | 0.0 | [0.0, 45.48] |
| t1_v3only_t1only_s1 | 0.9601 | 0.0 | [0.0, 45.48] | 0.0 | [0.0, 45.48] |
| t1_v4_t1only_s0 | 0.9951 | 86.3 | [34.7, 177.81] | 98.63 | [42.58, 194.34] |
| t1_v4_t1only_s1 | 0.9637 | 110.96 | [50.74, 210.63] | 110.96 | [50.74, 210.63] |
| t1_v4cap_t1only_s0 | 0.9960 | 0.0 | [0.0, 45.48] | 0.0 | [0.0, 45.48] |
| t1_v4cap_t1only_s1 | 0.9347 | 0.0 | [0.0, 45.48] | 0.0 | [0.0, 45.48] |
| tmp_v3only_temporal_s0 | 0.9988 | 0.0 | [0.0, 45.48] | 0.0 | [0.0, 45.48] |
| tmp_v3only_temporal_s1 | 0.9556 | 0.0 | [0.0, 45.48] | 0.0 | [0.0, 45.48] |
| tmp_v4_temporal_s0 | 0.9998 | 110.96 | [50.74, 210.63] | 110.96 | [50.74, 210.63] |
| tmp_v4_temporal_s1 | 0.9857 | 98.63 | [42.58, 194.34] | 110.96 | [50.74, 210.63] |
| tmp_v4cap_temporal_s0 | 0.9931 | 12.33 | [0.31, 68.69] | 12.33 | [0.31, 68.69] |
| tmp_v4cap_temporal_s1 | 0.9750 | 0.0 | [0.0, 45.48] | 0.0 | [0.0, 45.48] |
| DEPLOYED (prov OR conf) | 0.30 / 0.15 | 36.99 | [7.63, 108.09] | - | - |

### D. YOLO every frame + independent DINO sweep every Nth frame

#### DINO source: b1_v3 (reports/v3/eval_B1_val.raw.parquet)
Measured batch-1 L40S fp32: YOLO11m expE 11.24 ms, frozen DINOv3-L B1 156.43 ms -> one DINO pass costs 13.92x one YOLO pass. N for <=20% added compute = **70**.

Val sequences: 266, median length 15.5 frames. Fraction that get ZERO DINO pass: {"N=5": 0.0263, "N=10": 0.3083, "N=20": 0.5865, "N=35": 0.7707, "N=70": 0.9135, "N=140": 0.985}


**matched_fpr_0.005** (thr_yolo 0.8143, thr_dino 0.9717). YOLO alone: episode detection 0.1322 of 469 episodes, median latency 5.5 frames, 5 FP frames of 1158 negatives.

| N | added compute | combo ep-det | added FP frames | YOLO alone at SAME FP | combo - yolo @matched FP | beats? | median lat delta |
|---|---|---|---|---|---|---|---|
| N=5 | 278.4% | 0.1744 | +1.0 | 0.1493 (FP 6) | +0.0252 | YES | -1.5 |
| N=10 | 139.2% | 0.1588 | +0.5 | 0.1493 (FP 6) | +0.0096 | YES | -0.5 |
| N=20 | 69.6% | 0.1478 | +0.2 | 0.1322 (FP 5) | +0.0156 | YES | 0.5 |
| N=35 | 39.8% | 0.1421 | +0.1 | 0.1322 (FP 5) | +0.0099 | YES | 0.5 |
| N=70 | 19.9% | 0.1375 | +0.1 | 0.1322 (FP 5) | +0.0053 | YES | 0.5 |
| N=140 | 9.9% | 0.1349 | +0.0 | 0.1322 (FP 5) | +0.0027 | YES | 0.5 |

**matched_fpr_0.01** (thr_yolo 0.7997, thr_dino 0.9633). YOLO alone: episode detection 0.1748 of 469 episodes, median latency 4.0 frames, 11 FP frames of 1158 negatives.

| N | added compute | combo ep-det | added FP frames | YOLO alone at SAME FP | combo - yolo @matched FP | beats? | median lat delta |
|---|---|---|---|---|---|---|---|
| N=5 | 278.4% | 0.2345 | +2.2 | 0.1919 (FP 13) | +0.0426 | YES | 0.0 |
| N=10 | 139.2% | 0.2139 | +1.1 | 0.1812 (FP 12) | +0.0326 | YES | 0.0 |
| N=20 | 69.6% | 0.1978 | +0.6 | 0.1812 (FP 12) | +0.0165 | YES | 0.0 |
| N=35 | 39.8% | 0.1894 | +0.3 | 0.1748 (FP 11) | +0.0146 | YES | 0.0 |
| N=70 | 19.9% | 0.1828 | +0.2 | 0.1748 (FP 11) | +0.0080 | YES | 0.0 |
| N=140 | 9.9% | 0.1788 | +0.1 | 0.1748 (FP 11) | +0.0040 | YES | 0.0 |

**matched_fpr_0.02** (thr_yolo 0.7749, thr_dino 0.9574). YOLO alone: episode detection 0.2814 of 469 episodes, median latency 2.0 frames, 23 FP frames of 1158 negatives.

| N | added compute | combo ep-det | added FP frames | YOLO alone at SAME FP | combo - yolo @matched FP | beats? | median lat delta |
|---|---|---|---|---|---|---|---|
| N=5 | 278.4% | 0.3407 | +3.4 | 0.2985 (FP 26) | +0.0422 | YES | 1.0 |
| N=10 | 139.2% | 0.3200 | +1.7 | 0.2836 (FP 25) | +0.0365 | YES | 1.0 |
| N=20 | 69.6% | 0.3045 | +0.9 | 0.2836 (FP 24) | +0.0209 | YES | 1.0 |
| N=35 | 39.8% | 0.2965 | +0.5 | 0.2814 (FP 23) | +0.0150 | YES | 1.0 |
| N=70 | 19.9% | 0.2898 | +0.2 | 0.2814 (FP 23) | +0.0083 | YES | 1.0 |
| N=140 | 9.9% | 0.2857 | +0.1 | 0.2814 (FP 23) | +0.0042 | YES | 0.0 |

**matched_fpr_0.05** (thr_yolo 0.7225, thr_dino 0.9382). YOLO alone: episode detection 0.4606 of 469 episodes, median latency 1.5 frames, 57 FP frames of 1158 negatives.

| N | added compute | combo ep-det | added FP frames | YOLO alone at SAME FP | combo - yolo @matched FP | beats? | median lat delta |
|---|---|---|---|---|---|---|---|
| N=5 | 278.4% | 0.5070 | +8.6 | 0.4797 (FP 66) | +0.0273 | YES | -0.5 |
| N=10 | 139.2% | 0.4900 | +4.3 | 0.4670 (FP 61) | +0.0230 | YES | 0.5 |
| N=20 | 69.6% | 0.4760 | +2.1 | 0.4670 (FP 59) | +0.0091 | YES | 0.5 |
| N=35 | 39.8% | 0.4696 | +1.2 | 0.4606 (FP 58) | +0.0090 | YES | 0.5 |
| N=70 | 19.9% | 0.4651 | +0.6 | 0.4606 (FP 58) | +0.0045 | YES | 0.5 |
| N=140 | 9.9% | 0.4628 | +0.3 | 0.4606 (FP 57) | +0.0023 | YES | 0.5 |

**matched_fpr_0.1** (thr_yolo 0.6554, thr_dino 0.9041). YOLO alone: episode detection 0.6141 of 469 episodes, median latency 1.0 frames, 115 FP frames of 1158 negatives.

| N | added compute | combo ep-det | added FP frames | YOLO alone at SAME FP | combo - yolo @matched FP | beats? | median lat delta |
|---|---|---|---|---|---|---|---|
| N=5 | 278.4% | 0.6507 | +13.8 | 0.6333 (FP 129) | +0.0175 | YES | 0.0 |
| N=10 | 139.2% | 0.6356 | +6.9 | 0.6247 (FP 122) | +0.0109 | YES | 0.0 |
| N=20 | 69.6% | 0.6257 | +3.5 | 0.6141 (FP 118) | +0.0116 | YES | 0.0 |
| N=35 | 39.8% | 0.6210 | +2.0 | 0.6141 (FP 117) | +0.0069 | YES | 0.0 |
| N=70 | 19.9% | 0.6175 | +1.0 | 0.6141 (FP 116) | +0.0035 | YES | 0.0 |
| N=140 | 9.9% | 0.6158 | +0.5 | 0.6141 (FP 115) | +0.0017 | YES | 0.0 |

#### DINO source: b1_v4frozenL (reports/v4/eval_v4B1_frozen_L_sdis_val.raw.parquet)
Measured batch-1 L40S fp32: YOLO11m expE 11.24 ms, frozen DINOv3-L B1 156.43 ms -> one DINO pass costs 13.92x one YOLO pass. N for <=20% added compute = **70**.

Val sequences: 266, median length 15.5 frames. Fraction that get ZERO DINO pass: {"N=5": 0.0263, "N=10": 0.3083, "N=20": 0.5865, "N=35": 0.7707, "N=70": 0.9135, "N=140": 0.985}


**matched_fpr_0.005** (thr_yolo 0.8143, thr_dino 0.9879). YOLO alone: episode detection 0.1322 of 469 episodes, median latency 5.5 frames, 5 FP frames of 1158 negatives.

| N | added compute | combo ep-det | added FP frames | YOLO alone at SAME FP | combo - yolo @matched FP | beats? | median lat delta |
|---|---|---|---|---|---|---|---|
| N=5 | 278.4% | 0.1684 | +1.0 | 0.1493 (FP 6) | +0.0192 | YES | 0.5 |
| N=10 | 139.2% | 0.1559 | +0.5 | 0.1493 (FP 6) | +0.0066 | YES | 0.5 |
| N=20 | 69.6% | 0.1447 | +0.2 | 0.1322 (FP 5) | +0.0125 | YES | 0.5 |
| N=35 | 39.8% | 0.1398 | +0.1 | 0.1322 (FP 5) | +0.0076 | YES | 0.5 |
| N=70 | 19.9% | 0.1362 | +0.1 | 0.1322 (FP 5) | +0.0041 | YES | 0.5 |
| N=140 | 9.9% | 0.1342 | +0.0 | 0.1322 (FP 5) | +0.0020 | YES | 0.5 |

**matched_fpr_0.01** (thr_yolo 0.7997, thr_dino 0.9858). YOLO alone: episode detection 0.1748 of 469 episodes, median latency 4.0 frames, 11 FP frames of 1158 negatives.

| N | added compute | combo ep-det | added FP frames | YOLO alone at SAME FP | combo - yolo @matched FP | beats? | median lat delta |
|---|---|---|---|---|---|---|---|
| N=5 | 278.4% | 0.2166 | +2.2 | 0.1919 (FP 13) | +0.0247 | YES | 0.0 |
| N=10 | 139.2% | 0.2030 | +1.1 | 0.1812 (FP 12) | +0.0217 | YES | 0.0 |
| N=20 | 69.6% | 0.1902 | +0.6 | 0.1812 (FP 12) | +0.0090 | YES | 0.0 |
| N=35 | 39.8% | 0.1848 | +0.3 | 0.1748 (FP 11) | +0.0099 | YES | 0.0 |
| N=70 | 19.9% | 0.1801 | +0.2 | 0.1748 (FP 11) | +0.0053 | YES | 0.0 |
| N=140 | 9.9% | 0.1775 | +0.1 | 0.1748 (FP 11) | +0.0027 | YES | 0.0 |

**matched_fpr_0.02** (thr_yolo 0.7749, thr_dino 0.9805). YOLO alone: episode detection 0.2814 of 469 episodes, median latency 2.0 frames, 23 FP frames of 1158 negatives.

| N | added compute | combo ep-det | added FP frames | YOLO alone at SAME FP | combo - yolo @matched FP | beats? | median lat delta |
|---|---|---|---|---|---|---|---|
| N=5 | 278.4% | 0.3258 | +3.4 | 0.2985 (FP 26) | +0.0273 | YES | 1.0 |
| N=10 | 139.2% | 0.3132 | +1.7 | 0.2836 (FP 25) | +0.0296 | YES | 1.0 |
| N=20 | 69.6% | 0.3004 | +0.9 | 0.2836 (FP 24) | +0.0168 | YES | 1.0 |
| N=35 | 39.8% | 0.2941 | +0.5 | 0.2814 (FP 23) | +0.0126 | YES | 1.0 |
| N=70 | 19.9% | 0.2884 | +0.2 | 0.2814 (FP 23) | +0.0070 | YES | 1.0 |
| N=140 | 9.9% | 0.2850 | +0.1 | 0.2814 (FP 23) | +0.0036 | YES | 0.0 |

**matched_fpr_0.05** (thr_yolo 0.7225, thr_dino 0.9674). YOLO alone: episode detection 0.4606 of 469 episodes, median latency 1.5 frames, 57 FP frames of 1158 negatives.

| N | added compute | combo ep-det | added FP frames | YOLO alone at SAME FP | combo - yolo @matched FP | beats? | median lat delta |
|---|---|---|---|---|---|---|---|
| N=5 | 278.4% | 0.4964 | +7.8 | 0.4797 (FP 65) | +0.0166 | YES | 0.5 |
| N=10 | 139.2% | 0.4825 | +3.9 | 0.4670 (FP 61) | +0.0156 | YES | 0.5 |
| N=20 | 69.6% | 0.4730 | +2.0 | 0.4670 (FP 59) | +0.0061 | YES | 0.5 |
| N=35 | 39.8% | 0.4684 | +1.1 | 0.4606 (FP 58) | +0.0079 | YES | 0.5 |
| N=70 | 19.9% | 0.4645 | +0.6 | 0.4606 (FP 58) | +0.0040 | YES | 0.5 |
| N=140 | 9.9% | 0.4625 | +0.3 | 0.4606 (FP 57) | +0.0020 | YES | 0.5 |

**matched_fpr_0.1** (thr_yolo 0.6554, thr_dino 0.9416). YOLO alone: episode detection 0.6141 of 469 episodes, median latency 1.0 frames, 115 FP frames of 1158 negatives.

| N | added compute | combo ep-det | added FP frames | YOLO alone at SAME FP | combo - yolo @matched FP | beats? | median lat delta |
|---|---|---|---|---|---|---|---|
| N=5 | 278.4% | 0.6448 | +13.2 | 0.6333 (FP 128) | +0.0115 | YES | 0.0 |
| N=10 | 139.2% | 0.6313 | +6.6 | 0.6247 (FP 122) | +0.0066 | YES | 0.0 |
| N=20 | 69.6% | 0.6232 | +3.3 | 0.6141 (FP 118) | +0.0092 | YES | 0.0 |
| N=35 | 39.8% | 0.6193 | +1.9 | 0.6141 (FP 117) | +0.0052 | YES | 0.0 |
| N=70 | 19.9% | 0.6167 | +0.9 | 0.6141 (FP 116) | +0.0026 | YES | 0.0 |
| N=140 | 9.9% | 0.6154 | +0.5 | 0.6141 (FP 115) | +0.0013 | YES | 0.0 |
