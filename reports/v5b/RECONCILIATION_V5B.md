# V5b reconciliation notes (historical reports are NOT rewritten)

## R1 — v4 arm-A "3 seeds" were three seed-0 replicates (found 2026-10-02)
`cluster/v4/run_yolo_arm.sh` (used for arm A in iteration 4) never passed `--seed`;
`models/trained/expV4A_ctrl_s{0,1,2}/seg*/args.yaml` all record `seed: 0`. Arms
B/C/D/E used `run_v4_arm.sh` → `train_entry_v4.sh`, which does pass SEED, so
their seeds are genuine. Consequences: FINAL_REPORT_V4 T01–T05 "A ±std" is
DDP/non-determinism replicate spread, not seed spread (values unchanged;
interpretation narrowed); the median-seed rule for A picked among replicates.
The v5b control `expV5bA_ctrl_s3` was launched 07:35Z with the same launcher
and is therefore ALSO seed 0 — it is labelled "A replicate #4 (seed 0)" and
used only as the matched-budget control; any seeded repeats in v5b use
`run_v4_arm.sh --seed`. Launcher fixed: `run_yolo_arm.sh` now refuses to run
unless `ALLOW_SEED0_REPLICATE=1` is set.

## R2 — B6 LoRA-H+ shipped checkpoint ≠ run best
`v4B6_lora_Hplus/seg10/best.pt` (step 11,714, train mAP 0.4166) is what v4
evaluated/tested; the run's best (0.4217 @ step 8,428) survives at
`seg05/best.pt` (best-tracker reset on resume). Published v4 numbers are
consistent with the shipped file; the seg05 alternate is registered but
unevaluated (models/v5b/MODEL_REGISTRY.md).

## R3 — FireSpot positives were already in v4 corpora via MultiNatSmoke
2,862 of 2,865 MNS "FireSpot" rows match the original FireSpot-CNX images
(phash); all were in v4 `train_G.txt`, so v4 arms B/C/D saw them; 141 MNS
masks sit on author EXPLICIT NEGATIVES (spurious). FireSpot itself has no
licence → permission PENDING → not used in v5b training.
