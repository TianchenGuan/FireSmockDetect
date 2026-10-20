# V5b presentation outline — 11 slides (+1 backup) — FINAL (test opened 2026-10-03 03:20-03:50Z)

_Arc: problem -> labelled-data repair -> model collaboration -> efficient deployment -> measured results -> **what did NOT work (the centrepiece, slide 10)** -> cost & decision. Every number on a slide is traceable to the path given under "show"; test numbers come from `reports/v5b/tables/T8a-e_*.md`, `reports/v5b/test/matched_fp_sdistest.json` and `reports/v5b/COMPARISON_V5B.csv`. The honest one-line answer to the research question is NO, and the deck says so on slide 9 before any softening._

---

## Slide 1 — The problem we are actually solving
- Early smoke detection at a usable false-alarm rate: recall at **matched negative-image FP** is the only cross-model metric we trust (confidences are not comparable across architectures).
- After v4: more "generic smoke" did not buy earlier/cleaner detection on fixed cameras; the hypothesis for V5b was that the missing ingredients were **author-labelled aerial/forest data with explicit negatives** and **models that help each other** (as teachers, as an ensemble).
- V5b in one 24 h window / 128-GPU ceiling: 4 data acquisitions, 6-arm student matrix (+ 3 seeds for A and B), ensemble + FAST/ACCURATE serving, segmentation overlay, B200 track (high-res DINO + control, Qwen smoke test).
- Discipline: pre-registered systems/metrics/rule (`reports/v5b/PREREGISTRATION_V5B.md`), sealed holdouts opened once (`LOCKBOX_V5B_FROZEN.md`, guards in `reports/v5b/test/`).
- **Show:** `reports/v5b/tables/T4a_*.md` header row (what "R / tiny / P at 1-20% FP" looks like) + one sdis VAL frame from `reports/v5b/demo/demo_fast_sdis.png`.
- **Speaker note:** state the two scenarios (fixed camera vs aerial) and that thresholds are set per scenario on VAL negatives; nothing on later slides was chosen on a test set.

## Slide 2 — Labelled-data repair: what we got, what was blocked
- Obtained + trainable: **FireMan-UAV-RGBT** (CC BY 4.0; 69,871 RGB frames, 64,867 with smoke boxes, 4,979 author negatives, 4 burns) and **Sheffield UAV** (CC BY 4.0; 1,336 frames, 5,283 smoke boxes, one 12-min session).
- Blocked: **FireSpot** (no licence text anywhere -> pending; its positives were already in v4 via MultiNatSmoke), **WSDataset** (Kaggle login; 13k author negatives sit behind one API token).
- Provenance is agent judgement: only FireMan burn C is GPS-confirmed; burn B quarantined as a possible Ruokolahti (= v4 test site) duplicate.
- Overlap vs all training lists: FireMan 0/0; Sheffield 0 external hits — but see slide 10 for the self-inflicted H3 contamination.
- **Show:** `reports/v5b/tables/T1_data_sources_permissions_and_use_in_v5b_training.md` (trim to 6 rows); contact sheet `data/v5b/manifests/_inspect/fireman_rgb_sheet_00.jpg`.
- **Speaker note:** "licence" column is verbatim from the source; "pending" means we did not train on it, full stop.

## Slide 3 — FireMan is NOT a faint-smoke benchmark (and why we still wanted it)
- On FireMan VAL (burn D, 1,706 frames / 231 author negatives) the v1-era baselines saturate: expE R 0.963 @1% FP, 0.993 @10%; expB 0.994 / 1.000 (`T4c_*.md`).
- Precision 0.36-0.52 is an annotation-convention artefact (one author region box vs several predicted sub-boxes), not misses.
- Boxes are interpolated/projected region boxes -> presence-grade supervision; faint-smoke evidence stays with sdis tiny-recall (2,277 tiny GT on VAL) and the frozen fixed-camera TEST.
- What it gave us: author-labelled aerial negatives for aerial thresholds, and an independent sealed burn (H1, recall only) — on which, as predicted, every system scored 0.975-1.000 at aerial confs (slide 9).
- **Show:** `reports/v5b/tables/T4c_*.md` next to `T4a_*.md` (same two models: 0.99 vs 0.19-0.33 at 5% FP).
- **Speaker note:** this re-framing was made on VAL before any arm finished; it is a property of the dataset, not of our models.

## Slide 4 — The training matrix and the "small by construction" finding
- A control (E, 60 ep, YOLO11m) vs B (E + Boreal x2 + FireMan + Sheffield x2 = 53,511 lines, 43 ep, exposures matched -0.27%).
- C = B + teacher ignore regions: **148 images / 300 rectangles**. D = B + teacher pseudo boxes: **495 boxes**. D2 (all sources): **408 boxes, FireMan 0**.
- Why: the three teachers (DINOv3 B4/B6 LoRA, RF-DETR-L) agree with the authors on labelled data (recall 0.999 at thr 0.2; teacher-teacher kappa 0.95) — **pseudo-labels can only add value on UNLABELLED positives**.
- D3 = B + 3,797 teacher boxes on 4,844 genuinely unlabelled smoke-positive frames (Boreal clips, FIRESENSE, D-Fire fire-only) = the real teacher->student test; caveat: whole-plume boxes (median 25-36% of frame).
- **Show:** `reports/v5b/tables/T2_*.md` (arm rows) and the 3-way agreement table in `models/v5b/teacher/TEACHER_POOL.md`.
- **Speaker note:** budget-matched by exposures, same recipe; A and B have 3 seeds (`reports/v5b/val/SEEDS_V5B.md`), C/D/D2/D3 one — the measured seed range is ±1.6 pt on sdis VAL R@10%.

## Slide 5 — Student results on VAL (and the selected arm)
- sdis VAL matched-FP R@5% / R@10% (tiny@10%): expB 0.204 / 0.329 (0.203); expE 0.190 / 0.309 (0.201); **A 0.202 / 0.302 (0.191); B 0.183 / 0.302 (0.174)**; C 0.183 / 0.276; D 0.170 / 0.287; D2 0.158 / 0.274; **D3 0.164 / 0.239 (tiny 0.113)** — no arm beats expB; the teacher-assisted arms are the worst on faint smoke.
- Aerial (Boreal heinola VAL) R@5%: A 0.800 -> **B 0.955**, C 0.946 (D/D2/D3 never reach 5% FP; R@2% 0.87-0.94); 3-seed paired B-A **+21.4 pt [+15.5, +31.0]**. FireMan VAL R@10%: everyone 0.993-1.000.
- Pre-registered rule (mean of sdis R@10% and FireMan-val R@10%): B 0.650 > D 0.643 > C 0.637 > D2 0.636 > D3 0.617 -> **arm B selected**; A's and B's median seeds (s3, s0) went to the test.
- Per-system thresholds frozen from VAL (`reports/v5b/tables/_derived/preregistered_confs_v5b.json`).
- **Show:** `reports/v5b/tables/T4a_*.md`, `T4b_*.md`, `T4c_*.md`; `reports/v5b/val/SELECTION_V5B.md`, `SEEDS_V5B.md`.
- **Speaker note:** on VAL "new data = fixed-camera tie, aerial +21 pt" was the honest reading; the test (slide 9) turned the tie into a loss.

## Slide 6 — Model collaboration: calibrated ensemble on sdis VAL (site-grouped OOF)
- Fit on one VAL site, evaluate on the other. Best single B6 macro 0.326; **A+B4+B6+B1 calibrated WBF 0.354 (+2.8 pt)**.
- Site-dependent: marguerite +3.9 pt (95% CI [+1.0, +8.3]); cabanelle +1.8 pt (CI [-0.9, +4.0]) and **-1.4 pt vs RF-DETR-L alone** there.
- Addition protocol (>= +1 pt): B6 kept (+1.16), B1 kept (+1.34), RF-DETR rejected (-0.39), expE rejected (+0.06). Learned fusion +0.46 pt -> not adopted.
- **Calibration does not transfer between sites** (5% threshold from one site -> 19-27% FP on the other) -> per-camera thresholds are mandatory.
- **Show:** `reports/v5b/tables/T5_*.md` (macro column + bootstrap block); `reports/v5b/ENSEMBLE_DEV_V5B.md` T8 (threshold transfer).
- **Speaker note:** the recipe never saw a test set; its holdout verdict (slide 9) is that the +2.8 pt did not transfer.

## Slide 7 — Efficient deployment: FAST / ACCURATE modes and what they cost
- FAST = unchanged production path (expB@1280, byte-identical to golden, 11.2 ms L40S class / 146-169 ms CPU). The V5b FAST **candidate** is S1_A (same backend, YOLO11m, 275 ms CPU batch-1) — recommended, not flipped (slide 11).
- ACCURATE = components in parallel + fusion: **352 ms/frame L40S (parallel), 677 ms serial**; CPU 5.8 s (2-model demo) to 12.5 s (4-model recipe) -> GPU or offline only; after the test it stays a **demo mode**, not a shipped mode.
- Per-model evidence, fused boxes, per-model alert state and model versions are all in the response and the demo UI; invalid inputs are rejected (422/413) without touching counters.
- Segmentation overlay (DINOv3-L frozen head, IoU 0.568 on heinola, 2.6-2.9 s CPU) is an explanation layer, not an alarm.
- **Show:** `reports/v5b/tables/T7_*.md` (ensemble rows + demo rows); screenshot `reports/v5b/demo/demo_accurate_sdis.png`.
- **Speaker note:** selective execution (cheap gate -> expensive models) was not measured; the 352 ms assumes one GPU per component.

## Slide 8 — B200 track: what we proved, what we ran, what it said
- Node works: 8x B200 183 GB, torch 2.8 cu128 sm_100, DDP + checkpoint save/resume/reload PASS, 0.62 node-h probe.
- Constraints that shape everything: **S3 egress explicitly denied**, cross-region image pull rejected, only `kraken jobs connect` (2.9 MB/s) in/out; pods are uid 1001; non-zero exit = silent restart.
- High-res DINOv3-L LoRA 1920x1088 vs a same-corpus, same-node 1280x736 control: subset COCO mAP identical (0.416 / 0.416) but **matched-FP on sdis VAL worse at every level** — 5% FP R 0.165 / tiny 0.096 vs 0.193 / 0.135; 10% 0.280 / 0.203 vs 0.307 / 0.232 — for 2.4x train time and 2.7x latency. **Negative.**
- Qwen3.5-27B bf16: 70.7 GB image pulled in 10.6 min, 200-step smoke test SUCCEEDED (7.69 s/step, 66.5 GB/GPU, 0.64 node-h) — plumbing proven, **no campaign, no quality claim**.
- **Show:** `reports/v5b/B200_PROBE.md` §0; `reports/v5b/B200_DINO_HIGHRES.md` §7 table; `reports/v5b/B200_QWEN27B_PLAN.md` §7.
- **Speaker note:** the resolution result is one run per resolution with the head's anchors left at 1280 tuning; "a fair high-res variant rescales anchors x1.5" is the follow-up, not a rescue.

## Slide 9 — FINAL TEST (opened once): the answer is NO
- **H2 pyro-sdis TEST** (4,124 imgs / 917 neg), matched-FP R / tiny at 1 / 5 / 10 / 20%: **A (existing recipe, YOLO11m) 0.108/0.035, 0.287/0.184, 0.388/0.283, 0.553/0.435 — best YOLO at 5/10/20%**; A seeds 0.373 / 0.383 @10% (3-run mean 0.381); deployed expB 0.354; expE 0.350; **B (VAL-selected) 0.309** (seeds 0.349 / 0.332); C 0.318; D 0.342; D2 0.348; D3 0.358; S5 ensemble 0.366 (@20% 0.552 ≈ A; @1% 0.072 < A); B6 LoRA-H+ 0.363.
- **Val->test reversal:** B tied A on VAL and was selected; on TEST B is the worst student and A leads by 8 pt @10% (seed range on VAL ±1.6 pt). Plainly: adding FireMan/Sheffield/Boreal labelled data **hurt** fixed-camera faint-smoke recall on the frozen test.
- **H1 FireMan burn C** (1,949 fr, 0 neg -> recall only) at sdis-val 10% confs: C 0.856, A 0.736, D3 0.725, B 0.718, expE 0.663, expB 0.590, D2 0.480, D 0.438, **A_s1 0.438 (!)**, A_s2 0.630 — 30 pt seed spread; at aerial confs everyone 0.975-1.000 (saturated). S5 0.981-0.983 at its own calibrated scale (not comparable).
- **H3 Sheffield** within-incident, contaminated for B/C/D/D2/D3 (rows excluded); clean rows A 0.023 / 0.299, expB 0.006 / 0.243 at sdis-10% / FireMan-10% confs; S5 (v4a_s2 + DINO, clean) 0.212. **H4** ruokolahti A 0.906, B 0.890, expB 0.886 @ sdis-10%; FIgLib-39: A 30/39 (6 pre-visible), B 29 (6), C 31 (7), D 29 (4), D2 31 (4), **D3 32 (14 pre-visible!)**, expE 30 (5) — event coverage unchanged.
- **Show:** `reports/v5b/tables/T8c_*.md`, `T8b_*.md`, `T8a_*.md` (roster with contamination flags), `reports/v5b/COMPARISON_V5B.csv`.
- **Speaker note:** read the deployment rule aloud before showing numbers; decision on slide 11. Say "second opening of the sdis TEST split" out loud.

## Slide 10 — What did NOT work (the centrepiece)
- **The core story did not hold:** existing labelled smoke data + complementary models + teacher-student did **not** improve faint-smoke recognition on the frozen fixed-camera test. Every V5b arm is below the plain control (−3.0 to −7.9 pt @10% FP, `T8c`); the only robust gain from new data is aerial recall (+21 pt boreal VAL @5% FP).
- **Distillation is a negative result:** teachers agree with the authors on labelled data (C 148 images, D 495 boxes, D2 0 FireMan boxes); whole-plume pseudo-boxes on 4,844 unlabelled frames (D3) do not help faint smoke (H2 0.358 vs A 0.388; VAL tiny@10% 0.113 = lowest) and D3 fires 14/39 pre-visible FIgLib alerts for +2 events.
- **Ensemble gains negligible/negative on test:** 0.366 vs best single 0.388 @10%, worse at <= 5% FP; the VAL +2.8 pt was site-dependent and calibration does not transfer.
- **B200 high-res is negative:** 1920x1088 worse than its 1280 control at every matched-FP level (−1.3 to −2.8 pt R, −1.8 to −3.9 pt tiny) for 2.4x the compute.
- **Selection noise beat us:** the VAL-selected arm (B) was the worst student on TEST; A's three seeds span 30 pt on H1 at fixed-camera confs; one A seed would fail the H1 "no loss" condition the shipped checkpoint passes.
- **Process failures:** arm-A control is a seed-0 replicate (launcher never passed `--seed`, also true of v4's "seeds"); H3 contaminated for the data arms (Sheffield corpus used all exported frames); seg-1 upload expiry (2.98 node-h, no training lost); credential expiry (~4.5 h wall-clock); the first two final-test batches died on presigned-URL lifetime (5.00 node-h, no results seen, guards intact); B200 font hang / rendezvous collision / `wait` hold.
- **Still blocked:** FireSpot (no licence), WSDataset (Kaggle), AuSmoke (pending); Qwen-27B campaign not run; segmentation false-alarms on every frame of one clear-cut flight (human-mask IoU 0.47).
- **Show:** `reports/v5b/FINAL_REPORT_V5B.md` §7-§8, §10-§11; `reports/v5b/RECONCILIATION_V5B.md` (R1-R3); `reports/v5b/tables/T9c_*.md` (FAILED rows); `reports/v5b/replay/replay_S4c_D3_figlib39.json`.
- **Speaker note:** each item names the artifact where it is recorded; none was removed from the frozen reports. This slide is the result of V5b — the deployment recommendation on slide 11 is a by-product (the control was simply better than the deployed model).

## Slide 11 — Cost, decision, next
- Compute (ledger, 82 jobs, snapshot 2026-10-03T03:54Z): **L40S 55.30 node-h = 442.4 GPU-h; B200 5.97 node-h = 47.7 GPU-h; 61.3 node-h total = 16% of the 384 node-h / 3,072 GPU-h cap**; failed/retry 8.18 L40S node-h (3.18 training, no loss; 5.00 the two dead final batches). Registry 58.2 / 6.13 node-h (incl. queue wait). v4 used 80.3 L40S node-h.
- **Decision 1 (FAST):** rule MET by the existing recipe at YOLO11m — A beats expB by +3.4 pt on H2 R@10% (all three seeds >= +1.9, mean +2.7) and does not lose on H1 (0.736 vs 0.590; 1.000 vs 0.999) -> **recommend switching the FAST default to `models/trained/expV5bA_ctrl_s3/seg05/best.pt`; production NOT flipped in this window (user decision)**; env block + test commands in `FINAL_REPORT_V5B.md` §8.
- **Decision 2 (ACCURATE):** NOT shipped — ensemble does not beat the best single model at matched FP (0.366 vs 0.388 @10%); optional demo mode only.
- **Decision 3:** data arms / distillation / high-res — negative; nothing deployed from them.
- Next: D3's mechanism on unlabelled *fixed-camera* positives; seeds for every compared arm; unblock WSDataset / FireSpot licence / FireMan burn mapping; per-camera calibration + selective execution; B200 anchors-rescaled high-res variant and the Qwen-27B bf16 arm (<= 4 node-h).
- **Show:** `reports/v5b/tables/T9a_*.md`, `T9b_*.md`; `reports/v5b/FINAL_REPORT_V5B.md` §8, §12.
- **Speaker note:** close on the three things that are true regardless of taste — thresholds must be per camera, pseudo-labelling needs unlabelled positives of the *target* scenario, and the simplest candidate won.

## Backup slide — Infrastructure facts people will ask about
- Leviathan pods: no logs, exit-code channel unreliable on obsidian-pu, presigned S3 on green only (59-min URL lifetime bounds every job segment — it killed the first two final-test batches), `kraken jobs connect` for B200; segment/resume design (`cluster/v5b/*`).
- One-shot guards: `reports/v5b/test/.EVALUATED_SYSTEMS` (per system), `.EVALUATED_ONCE` (batch stamps), `.DINO_EVALUATED_ONCE`.
- Reproduction one-liners: `.venv/bin/python scripts/v5b/make_report_tables_v5b.py --no-dump`; `.venv/bin/python scripts/v5b/make_comparison_csv_v5b.py`; `bash scripts/v5b/run_demo_v5b.sh`; `python scripts/v5b/ensemble_dev.py --workers 24`.
- **Show:** `reports/v5b/B200_PROBE.md` §5 and §8; `reports/v5b/tables/ALL_TABLES.md` index.
