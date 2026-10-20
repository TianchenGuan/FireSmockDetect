# Track B/C launch configs (DINOv3 detection, green g6e)

Preconditions (one-time, ~30 min total after gated access lands):
1. `bash scripts/v3/dinov3_gate_check.sh` exits 0 (all 3 models accessible).
2. `bash scripts/v3/stage_dinov3.sh` — downloads pinned snapshots, syncs to
   S3, stages byte-verified to `/scratch/scratch/tianrgua-sandbox/models/`.
3. Container `smoke-dino3:v1` already pushed
   (`000000000001.dkr.ecr.us-east-1.amazonaws.com/smoke-dino3:v1`,
   digest `sha256:aaf5b9d0...`), data already on `/scratch` (smoke_data_v2
   mirror, reused from v2). Nothing else to build.

Run order and what each gives us:
- `B1_vitl16.sh`       frozen dinov3-vitl16 + SFP + FRCNN head (the anchor run)
- `B2_vith16plus.sh`   frozen dinov3-vith16plus, same head/data/exposure
  (only forced difference: SFP input projection consumes 1280-dim instead of
  1024-dim features; trainable head param count differs by <2M)
- `B3_vitl16_unfreeze4.sh`  vitl16 with top-4 blocks unfrozen @ 0.1x LR
  (chosen over LoRA: full-rank adaptation, no new deps, native ckpting;
  grad-checkpointing keeps memory in budget — see trackBC_prep.md)
- `C1_7b_precompute.sh` then `C1_7b_head.sh`  frozen 7B via cached features
  (run precompute first; check status.json scratch_free_gb >= ~1600 before
  starting; see the 7B plan in reports/v3/trackBC_prep.md)
- `standin_overfit16.sh`  plumbing check with UNGATED dinov2-large — its
  numbers are NEVER DINOv3 results.

All runs: train list = frozen expB_train (COCO json from Track A), val =
sdis_val, 1280x736 letterbox (patch-16 grid 80x46), bf16 autocast, seed 0,
segmented ~45 min jobs with /scratch + S3-presigned resume. Monitor:
`models/v3/runs/<run>/seg*/{status.json,metrics.jsonl,console_tail.txt}`.
Detached launch (survives session restarts):
`setsid nohup bash cluster/v3/launch/B1_vitl16.sh > logs/v3_B1.log 2>&1 &`
