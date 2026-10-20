# V6 — Local image GENERATOR on the B200 node: infra feasibility plan

Written 2026-10-20 (read-only survey; no jobs submitted, no downloads, nothing deleted).
Candidates: **Qwen/Qwen-Image-2.1** and **zai-org/GLM-Image**, one node of
`obsidian-pu` B200 (8× B200 183 GB). Everything marked *measured* was measured today or is a
measured v5b number re-verified today; sizes come from the HF API (`?blobs=true`), classes from
running `python -c` inside the local `smoke-diffusion:v1` image and from the diffusers release
tags on GitHub. Primary sources: `reports/v5b/B200_PROBE.md`, `reports/v5b/B200_QWEN27B_PLAN.md`,
`cluster/v5b/{Dockerfile.vlm_b200,Dockerfile.dino3_b200,qwen_b200_entry.sh,orchestrate_b200_qwen.py}`,
`cluster/v2/Dockerfile.diffusion`.

## 0. Verdict

**GO — both generators fit a single B200 GPU fully resident in bf16 (33.1 / 35.8 GB vs 178 GiB
visible), so one node = 8 independent generation workers.** The whole v5b delivery pattern
(bake weights into an ap-south-1 ECR image → submit → retrieve via `jobs connect`) applies
unchanged, and the weight payload here (33–36 GB) is *smaller* than the 55.6 GB already proven
in v5b. Estimated ~3 h devbox-to-first-image (≈1.5 h of that is machine time).
Recommended first arm: **Qwen-Image-2.1** (smaller, 40-step single pipeline, same Qwen lineage
as the v2 synth stack; needs one pip pin bump to diffusers 0.41.0). GLM-Image runs on the
*existing* pinned stack (diffusers 0.40.0) but is a slower hybrid AR(9B)+DiT(7B) pipeline.

## 1. Queue status (measured 2026-10-20, `kraken queues list`)

| queue | Available | Used | Allocated | note |
|---|---|---|---|---|
| **obsidian-pu-hyperpod-p6-b200.48xlarge-ap-south-1-tas** | **1** | 0 | 1 | POC=SYSTEM, the queue we submit to; whole node = 8× B200 183,359 MiB; **idle right now** |
| green-hyperpod-g6e.48xlarge-us-east-1-tas | 2 | 37 | 39 | fallback, 8× L40S 48 GB |
| green-hyperpod-g6e.48xlarge-us-west-2-tas | 8 | 8 | 16 | fallback, 8× L40S 48 GB |

Other obsidian-pu B200 queues exist (forge/fai/pulsar/canopy) but are BINDLE/other-POC — not ours.
The B200 cap is still **1 node total**, shared with any other track we run.

## 2. The generators (HF API, no downloads; neither repo is gated)

| | Qwen/Qwen-Image-2.1 | zai-org/GLM-Image |
|---|---|---|
| weights (sum of safetensors, bf16) | **33.12 GB** | **35.77 GB** |
| breakdown | text_encoder (Qwen3-VL) 17.53, DiT transformer 14.23, VAE 1.35 | AR vision_language_encoder (9B GLM) 20.23, DiT transformer 13.85, T5/ByT5 text_encoder 0.87, VAE 0.81 |
| pipeline class (`model_index.json`) | `QwenImage21Pipeline`, `_diffusers_version 0.37.0.dev0` | `GlmImagePipeline`, `_diffusers_version 0.37.0.dev0` |
| needs (model card) | torch≥2.4, transformers≥5.17, diffusers w/ class, accelerate, pillow | transformers+diffusers w/ classes |
| class availability (verified) | diffusers **0.41.0** (latest PyPI release) has `QwenImage21Pipeline` + `QwenImage21Transformer2DModel` + `AutoencoderKLQwenImage21`; **NOT in 0.40.0** (checked in-image). `Qwen3VLForConditionalGeneration`/`Qwen3VLProcessor` present in our transformers 5.17.0 | **already complete in our existing stack**: `GlmImagePipeline`+`GlmImageTransformer2DModel` in diffusers 0.40.0, `GlmImageForConditionalGeneration`/`GlmImageProcessor`/`T5EncoderModel`/`ByT5Tokenizer` in transformers 5.17.0 (all checked inside `smoke-diffusion:v1`) |
| defaults | 2048×2048, 40 steps | 50 steps, `guidance 1.5`, h/w multiples of 32, `device_map="cuda"` |
| license | qwen-research (`license:other`, not gated) | MIT |

VRAM: both pipelines fully resident on ONE B200 (178.4 GiB visible) with >140 GB headroom →
run 8 per-GPU workers, no offload, no sharding. (L40S math in §7.)

## 3. Existing images / stack (verified today: `docker images` + `aws ecr describe-images` ap-south-1, read-only ada creds)

| image | where | stack | diffusers? |
|---|---|---|---|
| `smoke-diffusion:v1` 7.92 GB | local + **us-east-1 ECR only** | pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime (py 3.11.13, torch 2.8.0+cu128, **sm_100 verified** on this exact torch build in v5b), diffusers **0.40.0**, transformers 5.17.0, accelerate 1.15.0, safetensors 0.8.0, pillow 12.3.0, **curl** | yes — GLM-ready, Qwen-2.1 needs 0.41.0 |
| `smoke-detect-vlm-b200:v1` 70.7 GB | ap-south-1 ECR (53.8 GB compressed) + local | torch 2.8.0+cu128, transformers 5.17.0, peft/trl/accelerate | **no diffusers** |
| `smoke-dino3-b200:v1` 28.8 GB | ap-south-1 ECR (24.3 compressed) + local | smoke-detect:v4 + transformers 5.17.0 | **no diffusers** |
| `smoke-detect:v4` | ap-south-1 ECR | torch 2.8.0+cu128, ultralytics | no |

ap-south-1 repos that exist for us: `smoke-detect` (v1–v4), `smoke-dino3-b200` (v1),
`smoke-detect-vlm-b200` (v1). **No diffusion image in ap-south-1 yet** → new repo
`smoke-diffusion-b200` needed (same one-time pull-policy step as v5b).
Base of choice: `smoke-diffusion:v1` (already has curl + offline env + the right torch; one
`pip install diffusers==0.41.0` upgrade covers Qwen-2.1 and keeps GLM classes).

## 4. The proven v5b B200 delivery pattern (what to copy verbatim)

In/out channels (B200_PROBE §3/§5, all re-confirmed by the v5b Qwen smoke test):
- **No internet, no S3** from pods: presigned URLs → 403 explicit deny (aps1) / unroutable (use1);
  pod role has no bucket grant. `/scratch` FSx on prod-7 is mounted but **has no writable
  location for us → it is NOT an output channel and does NOT sync to S3** (v5b probe: sandbox
  path not creatable; outputs went 100% via `connect`).
- **In**: the container image (weights/data) + `kraken jobs connect -c` base64 (small scripts only).
- **Out**: `kraken -p obsidian-pu jobs connect -j <job> -c "sh -c '... | base64 -w0'"` at
  **2.9 MB/s measured** (sha-exact), + a status JSON polled the same way. Exit code is NOT
  surfaced; **entry must exit 0 always** (non-zero ⇒ auto-restart, `maxRestarts ≥ 1` enforced).
- Rules: image region MUST be ap-south-1 (cross-region rejected at submit); image validated at
  submit (repo policy must allow cluster account **000000000004**, policy JSON ready at
  `cluster/v5b/ecr_pull_policy_aps1_3accounts.json`); **job names must not contain underscores**;
  pods run uid 1001 (assets under `/opt`, world-readable, `COPY --chmod` not `RUN chmod -R`);
  `HOME=/tmp`, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, caches to /tmp, `unset PET_*`;
  ECR layer cap 52,000 MiB (both weight sets fit in ONE `COPY` layer each: 33.1 / 35.8 GB);
  end with a RETAIN loop waiting for `/tmp/ack` so the orchestrator can pull before exit.
- Measured rates to plan with: docker build 58 GB context **12.8 min** (~215 MB/s, hard-linked
  context); devbox→aps1 ECR push **31–36 MB/s** (70.7 GB in 33 min); node pull **~110 MB/s**
  (70.7 GB submit→RUNNING in 10 m 36 s; 28.8 GB in 3 m 23 s); in-pod cold read of baked
  safetensors **2.8 GB/s**; connect retrieval 2.9 MB/s.
- Reusable code: `cluster/v5b/orchestrate_b200_qwen.py` (+`orchestrate_b200_dino.py` for
  QUEUE/YAML_TMPL/connect/pull machinery) and `cluster/v5b/qwen_b200_entry.sh` as the entry
  template (ENTRY_B64 injection, status-file pattern, GPU sampler, RETAIN/ack).

## 5. Step list (Qwen-Image-2.1 first arm; times = measured-rate estimates)

| # | step | est. time |
|---|---|---|
| 1 | Devbox: `hf download Qwen/Qwen-Image-2.1` (33.1 GB, pin the revision; repo not gated) | 15–40 min (devbox internet) |
| 2 | Devbox: build dir (hard-link weights + a `gen_batch.py` worker + prompts jsonl); `Dockerfile.gen_b200`: `FROM smoke-diffusion:v1` → `pip install diffusers==0.41.0` → `COPY` weights to `/opt/models/qwen-image-2.1` (1 layer, <52 GB cap) → build-time assert (shard count/bytes, `from diffusers import QwenImage21Pipeline`) | build ~8–12 min (≈42 GB image) |
| 3 | Devbox: create ECR repo `smoke-diffusion-b200` (ap-south-1) + apply 3-account pull policy (000000000005, 000000000006, **000000000004**) | 5 min |
| 4 | Push :v1 to ap-south-1 (~42 GB, all layers new in that region) | ~20–25 min at 36 MB/s |
| 5 | Submit 1-node job (name w/o underscores, e.g. `tianrgua-v6gensmk`), entry modeled on `qwen_b200_entry.sh`: load pipeline per GPU (8 workers, `CUDA_VISIBLE_DEVICES=i`), smoke = 2 images/GPU, then the real prompt batch; exit 0 + RETAIN | submit→RUNNING ~6–8 min (42 GB pull); load ~1–2 min; first image ~1–2 min |
| 6 | Generate batch: 7B DiT, 40 steps — est. 5–15 s/img @1024², 15–45 s/img @2048² per GPU (UNMEASURED on B200; the smoke step measures it) → 8 GPUs ≈ 2–6 k img/h @1024² | depends on batch size |
| 7 | Retrieve via `connect` chunked base64 (**the only out-channel**; write JPEG q≈92, not PNG): 1,000×1024² JPEG ≈ 0.4 GB → ~4 min; 1,000×2048² ≈ 1.4 GB → ~11 min. tar+split into ~200 MB chunks with per-chunk sha256 (v5b pulled 40.5 MB sha-exact; 294 MB in 52 s) | ~4–15 min per 1k images |
| 8 | Devbox: verify checksums, stop job (ack file), release registry reservation | 10 min |

**Total to first generated image: ≈ 2.5–3 h** (≈1.5 h machine time + scripting/verification;
entry+orchestrator are ~80% reusable from v5b). A full 5–10 k image batch adds ~1–3 h node time.
GLM-Image second arm: same image if its 35.8 GB is baked too (image ≈78 GB — just above the
proven 70.7 GB, push ~37 min, pull ~12 min) or a separate :v2 tag; no pip change needed
(classes already in 0.40.0/5.17.0, and 0.41.0 keeps them).

## 6. Risks

1. **Single shared B200 node (quota 1)** — idle today, but any DINO/Qwen-SFT track blocks it.
   Generation is embarrassingly parallel per-GPU → can co-schedule nothing; plan short windows.
2. **Generation speed on B200 unmeasured** — sm_100 attention efficiency was the v5b wildcard
   (DINO hit ~12% MFU with HF SDPA). Mitigation: the smoke step measures s/img before the batch;
   worst case 2–3× slower than the §5 estimate, still fine for thousands of images.
3. **diffusers 0.41.0 vs torch 2.8.0+cu128 / transformers 5.17.0 compat** — one minor bump from
   the proven 0.40.0 pin; verified importable only via release notes/classes, not executed.
   Mitigation: build-time import assert + local 1-image CPU... (too slow) → assert import +
   config load at build; first GPU image in the smoke step.
4. **Retrieval volume** — connect is 2.9 MB/s; >10 GB of outputs ≈ >1 h. Keep JPEG, retrieve in
   chunks during generation (connect works while the job runs), don't bake PNG dumps.
5. **Restart semantics** — any non-zero exit re-runs the pod (and re-pays nothing on pull if
   cached, but re-generates). Entry must exit 0 always + `EXITED` marker (v5b pattern).
6. **Qwen-research license** (`license:other`): fine for internal synthetic-data experiments we
   already did with Qwen-Image-Edit in v2, but check before shipping generated data externally.
7. **ECR hygiene**: new repo needs the pull policy BEFORE submit (submit-time validation);
   account 000000000004 missing = instant `ACCESS_DENIED` (v5b saw exactly this).

## 7. Fallback: green L40S (g6e.48xlarge, 8× L40S 48 GB)

Availability now: **us-east-1 Avail 2** (39 allocated, busy), **us-west-2 Avail 8**.
This is the *easier* cluster: v2 ran Qwen-Image-Edit-2511 (57.7 GB pipeline!) here — green pods
fetch entry/code via **presigned GET**, weights staged once to the writable
`/scratch/scratch/tianrgua-sandbox/` FSx (us-east-1/prod-11), outputs uploaded via **presigned
PUT** (`cluster/v2/gen_synth_pod.py`) — no image baking, no connect bottleneck.
- **Qwen-Image-2.1 (33.1 GB bf16)**: does NOT fit resident with activation headroom at 2048² on
  48 GB; with `enable_model_cpu_offload()` peak ≈ largest component (17.5 GB encoder / 14.2 GB
  DiT) + activations ≈ 20–26 GB → **feasible bf16, same recipe as v2** (v2 peak was ~41 GB with
  a 40.9 GB transformer). At 1024² it may even run resident; measure.
- **GLM-Image (35.8 GB bf16)**: resident is tight (12 GB headroom incl. 1–4 K token AR stage);
  with cpu offload peak ≈ 20.2 GB AR encoder + activations → **feasible bf16**. No quantization
  needed for either; offload costs maybe 1.5–3× latency per image.
- Caveats: `smoke-diffusion:v1` exists in **us-east-1 ECR only** → us-west-2 nodes need a ~8 GB
  push to a us-west-2 repo (region rule applies everywhere) and the /scratch + presigned
  behaviour on usw2 is UNVERIFIED (v2 evidence is us-east-1); diffusers upgrade to 0.41.0 →
  new tag `smoke-diffusion:v2` either way for Qwen-2.1 (GLM runs on :v1 as-is).

## 8. Local disk / residue (report only — nothing deleted)

- Docker 25.0.14, root `/local/docker`, overlay2, same 15 TB volume as `/local`: **3.3 TB free
  (78% used)** → comfortably fits a 40–90 GB build context + image (**disk_ok**).
- `cluster/v5b/build_qwen/` still holds the v5b build context: **58 GB**
  (`weights_a`+`weights_b` = the 27B shards, `data`, `code`). Reclaimable later; left in place.
- Local docker already holds `smoke-detect-vlm-b200:v1` (70.7 GB) + `smoke-dino3-b200:v1`
  (28.8 GB) + the full v2/v3 tag zoo — another ~250 GB reclaimable via `docker image prune`
  if ever needed (not needed now).
