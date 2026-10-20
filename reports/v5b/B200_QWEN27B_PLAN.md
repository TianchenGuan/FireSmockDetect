# V5b — Qwen3.5-27B BF16 LoRA SFT on the B200 node: feasibility plan (no campaign run)

Written 2026-10-02 (V5b, B200 track) after the DINOv3 high-res run was launched on the same node.
Scope per brief: **plan + measured prerequisites only**; an optional 200-step smoke test is
recorded in §7 if it was run (it is a smoke test, not an SFT result). Everything marked
*measured* was measured in this session or the V5b probe (`reports/v5b/B200_PROBE.md`);
everything marked *estimate* is derived and should be treated as such.

## 0. Verdict
- **Feasible on the hardware** (B200 178 GiB/GPU holds 27B bf16 + LoRA without quantization or
  ZeRO-3) and **feasible on the staging side** with one accepted cost: the 55.6 GB base weights
  must be **baked into an ap-south-1 ECR image** (S3 egress from obsidian-pu pods is denied,
  `connect` push runs at 2.9 MB/s = 5.3 h). *Measured* devbox→ap-south-1 ECR push rate this
  session: **31 MB/s** (14.0 GB of new layers in 7.4 min) → a ~70 GB image pushes in ~40 min,
  not the 1.5–2.5 h feared in the probe.
- **Budget estimate for a follow-up window: ≤ 4 B200 node-hours** for one bf16, 1 MP, 1-epoch
  arm that matches the v2 QLoRA data/step budget, incl. the 2,000-image val eval; 200-step
  smoke test ≈ 0.6–0.9 node-h.
- **GO for a follow-up window; NO campaign was run in this V5b window.** The three preconditions were
  measured in §7 (200-step smoke test, 0.64 node-h): the 70.7 GB image pulls in 10.6 min and passes kraken
  validation, bf16 27B LoRA trains at **7.7 s/step at eff. batch 64 (2.2–2.6× the v2 L40S nf4 speed)** using
  66 GB/GPU, and adapters come back through `connect`. Remaining condition: the B200 node is the single
  shared node (cap 1) — the Qwen arm competes with the DINO track; and the 2,000-image eval must use batched
  generation to stay inside 4 node-h.

## 1. What has to be matched (the v2 Track-B QLoRA run) — *measured in v2*
| item | value | source |
|---|---|---|
| model | Qwen/Qwen3.5-27B @ `fc05daec18b0a78c049392ed2e771dde82bdf654`, 24 files, 55,586,167,982 B | `models/v2/PINS.md`, `cluster/v2/WEIGHTS_STAGING.md` |
| adaptation | nf4 double-quant base (bitsandbytes), bf16 LoRA r16 α32 dropout 0.05 on language-model linears (vision tower frozen; linear-attention `in_proj` untargeted), pure DDP (no DeepSpeed; ZeRO-3 measured 391 s/it on PCIe L40S) | `src/train/train_vlm.py`, STATUS 2026-09-11 |
| data | `data/v2/sft/train.jsonl` 59,999 examples = 38,454 full images (pyro-sdis 22,177 + D-Fire 16,277) + 19,294 positive crops + 2,251 negative crops; images are the pyro-sdis/D-Fire training images | `data/v2/sft/stats.json`, `DATASET.md` |
| schedule | 1 epoch = 938 steps at effective batch 64 (bs 1/GPU × grad-accum 8 × 8 GPUs), lr 1e-4, save every 200 | `cluster/v2/vlm_entry.sh`, `trkB_27b_qlora/seg10/status.json` (step 938) |
| visual input | **0.75 MP pixel budget** (image resized in the dataset so w·h ≤ budget; the 9B arm used 1 MP = 1,048,576 px). Qwen3.5 ViT: 16-px patches, 2×2 merge → ~32×32 px per LLM token → ~730 vision tokens/img at 0.75 MP, ~1,000 at 1 MP | STATUS 2026-09-12 (“27B/nf4/0.75MP”), `train_vlm.py --max-pixels` |
| throughput on 8×L40S | 17.2–20.2 s/it (eff. 64) → 3.2–3.7 examples/s/node; 100 steps per ~53-min segment | `prof_27b_qlora`, seg10 console |
| cost | 10 segments, Σ pod budgets 31,663 s = **8.80 node-h L40S** (+ pod start/fetch per segment ≈ 9.5 h wall) | seg01–10 `status.json` |
| result | tiny-R 0.302 / box-R 0.438 / P 0.754 / 18 % parse failures vs **9B bf16 1 MP SFT tiny-R 0.460** — confounded by quantization + resolution | `reports/FINAL_REPORT_V2.md` |
| eval | `src/eval/eval_vlm.py` on `data/v2/sft/val_eval_2000.txt` (2,000 val images), sharded over 8 GPUs, 192 new tokens (512 + fallback parser in the v2 re-eval) | `models/v2/evals/eval27b_sft_v2/` |

The B200 arm that answers the open question (“does 27B lose only because of nf4 + 0.75 MP?”)
is: **same data, same 938 steps at eff. 64, same lr/LoRA config, bf16 base, 1 MP** (= the 9B
arm's visual budget). An optional second arm at 0.75 MP isolates quantization alone.

## 2. Base weights — where they are and how they can reach ap-south-1 (*measured*)
| location | status |
|---|---|
| devbox `models/v2/qwen3.5-27b/` | present, 11 safetensors shards (2.15–5.37 GB each) + tokenizer/config, 52 GiB = 55.6 GB, sha-verified against HF at the pinned revision in v2 |
| S3 `s3://smoke-detect-data-000000000001-use1/weights/qwen3.5-27b/` | present (v2 staging) — **unreachable from obsidian-pu pods**: us-east-1 endpoints have no route, ap-south-1 presigned URLs get an explicit deny, the pod role has no grant (probe §3) |
| green `/scratch/scratch/tianrgua-sandbox/models/qwen3.5-27b/` | present on the green (prod-11) FSx — **not mounted on obsidian-pu prod-7** (different filesystem, probe §1) |
| obsidian-pu pod | reachable channels: the container image, or `kraken jobs connect` push at 2.9 MB/s (55.6 GB ≈ 5.3 h — rejected) |

Image bake constraints (*measured/looked up 2026-10-02*):
- ECR quotas (docs): **maximum layer size 52,000 MiB (= 54.5 GB), not adjustable**; no total
  image-size quota is documented; layer-part limits (10 MiB × 4,200) apply only to the raw API.
  55.59 GB in one `COPY` would exceed the layer cap → split into ≥ 2 layers (prepared:
  `cluster/v5b/build_qwen/weights_a` = shards 1–6, 31.9 GB; `weights_b` = shards 7–11 +
  tokenizer/config, 23.7 GB).
- Push rate devbox → ap-south-1 ECR: 14.0 GB new layers in 7.4 min = **31 MB/s** (this session,
  `smoke-dino3-b200:v1`). → weights 55.6 GB ≈ 30 min; a full ~71 GB image (vlm:v3 base 9.4 GB,
  whose pytorch base layers are not yet in ap-south-1, + pyro/D-Fire images 5.7 GB + weights)
  ≈ 40 min. Docker build (hard-linked context, two 25–32 GB layers) ≈ 10–15 min.
- **Measured this session (image actually built and pushed):** `smoke-detect-vlm-b200:v1`
  (`cluster/v5b/Dockerfile.vlm_b200`; FROM local `smoke-detect-vlm:v3` + pyro-sdis/D-Fire images + SFT jsonl
  + `train_vlm.py`/`eval_vlm.py` + the 55.6 GB weights in two layers of 32.0 and 23.6 GB) = **70.7 GB**;
  docker build 12 m 48 s (hard-linked context, 58 GB streamed at ~215 MB/s), `docker push` to ap-south-1
  **09:41:0x → 10:14:00Z = 33 min ≈ 36 MB/s** (all layers new in that region), build-time check: 11 shards /
  55,563,022,432 B present, 59,999 SFT rows, image paths resolve. ECR accepted the 32.0 GB layer (< 52,000 MiB).
  Repo `smoke-detect-vlm-b200` created with the 3-account pull policy. So the staging blocker of the probe is
  **solved on the devbox side**; what remains untested is the node-side pull of a 70.7 GB image.
- **Pull on the node — measured:** the 70.7 GB `smoke-detect-vlm-b200:v1` job went submit → RUNNING in
  **10 min 36 s** (13:46:40 → 13:57:16Z, all of it `PENDING:ADMITTING` = scheduling + layer pull; ≈ 110 MB/s
  effective), kraken's submit-time image validation accepted it, and the first cold read of all 11 safetensors
  shards from the image layer took ≈ 20 s (`load_check.json`). Risk 1 and 2 of §6 are therefore retired.
- Pull on the node (earlier datapoint): `smoke-dino3-b200:v1` (28.8 GB; 14.4 GB of it = v4 layers possibly cached on
  the nodepool) went PENDING→RUNNING in 3 min 23 s. A 70 GB image is *untested*; linear
  extrapolation 8–12 min, plus 55 GB of safetensors read from the local layer at model load.
- Region rule: the image must live in an **ap-south-1** repo (`smoke-detect-vlm-b200`, create
  with the 3-account pull policy `cluster/v5b/ecr_pull_policy_aps1_3accounts.json`).
- Cross-check of software: `smoke-detect-vlm:v3` (local docker copy) = Python 3.11.13,
  torch 2.8.0+cu128 (same build as the B200-verified v4 image → sm_100 kernels), transformers
  5.17.0, peft 0.20.0, trl 1.13.0, accelerate 1.15.0, deepspeed 0.19.6, bitsandbytes 0.50.2,
  flash-attn 2.8.3 (irrelevant: `train_vlm.py` uses `attn_implementation="sdpa"`; FA2 has no
  sm_100 build, so **do not switch to flash_attention_2 on B200**).
- Data: all SFT images are pyro-sdis/D-Fire training images (already in `smoke-detect:v4`;
  bake `data/yolo/{pyro_sdis,dfire_smoke}` 5.7 GB into the vlm image, root-map
  `/local/home/tianrgua/cvproject/data/yolo=/opt/smoke_data`); `train.jsonl` 35 MB.

## 3. Memory on a B200 (*estimate*, 178.4 GiB visible per GPU)
- bf16 weights 27B ≈ 54 GB; LoRA r16 on all LM linears ≈ 0.1–0.2 B params → params+grads+Adam
  (fp32 states) ≈ 1.5–2.5 GB; activations at ~1,150 tokens/sequence with gradient checkpointing
  (non-reentrant, `enable_input_require_grads`, v2 fix) ≈ 2–4 GB per sequence.
- → bs 1–4 per GPU fits with > 100 GiB headroom; **no nf4, no ZeRO** (every GPU holds the full
  bf16 copy, DDP on LoRA grads only, like the v2 QLoRA DDP but in bf16). Keep eff. batch 64 by
  `--per-device-bs 2 --grad-accum 4` (or 4 × 2) — identical optimizer trajectory up to batch
  composition.

## 4. Time on the B200 (*estimate*; one 8-GPU node)
- FLOPs per example ≈ 6 × 27e9 × ~1,250 tokens ≈ 2.0e14, +33 % for recompute ≈ 2.7e14. At a
  conservative 25–35 % of B200 dense bf16 peak (~2.2 PFLOP/s) → 0.35–0.5 s per example per
  GPU → 16–23 examples/s per node → **60k examples ≈ 45–65 min of pure training**. Reality check
  from this session: the DINOv3-L run reached only ~12 % MFU with HF SDPA on sm_100, so plan for
  **1.5–2.5 h for the epoch** (2–3× the ideal), vs 8.8 node-h on L40S/nf4.
- Fixed costs: image pull 10 min (untested), model load 55 GB ≈ 3–5 min, per-200-step checkpoints
  (adapter only, ~300 MB) negligible, eval of 2,000 val images sharded on 8 GPUs ≈ 15–25 min
  (v2: 17.8 s/img on L40S nf4 at bs 1; bf16 B200 should be several× faster), retrieval via
  `connect` (adapter ~300 MB ≈ 2 min at 2.9 MB/s; preds jsonl < 1 MB).
- **Total for the matched arm: ~2.5–3.5 node-h, cap 4.** Two arms (1 MP + 0.75 MP): 5–7 node-h.
- 200-step smoke test (12,800 examples): 10–25 min training + ~20 min fixed ≈ **0.6–0.9 node-h**.

## 5. Plan for a follow-up window (steps, owner = this track)
1. Devbox (≈ 1.5 h, no node time): `docker build -f cluster/v5b/Dockerfile.vlm_b200` FROM the
   local `smoke-detect-vlm:v3` → + `data/yolo/{pyro_sdis,dfire_smoke}`, `data/v2/sft/*`,
   `src/train/train_vlm.py`, `src/eval/eval_vlm.py`, weights in two layers; `chmod a+rX`
   via `COPY --chmod` (a `RUN chmod -R` duplicates the layer — cost us 7 GB in the DINO image);
   create repo `smoke-detect-vlm-b200` + 3-account policy; push (≈ 40 min). Verify in-container
   as uid 1001: `from_pretrained(..., dtype=bf16, device_map=None)` on CPU is too slow for 27B —
   verify shard presence/sha instead and run the 16-example overfit on the node.
2. Node (≤ 4 node-h, registry `--gpu-type B200 --track b200`): entry script modelled on
   `cluster/v5b/dino_b200_entry.sh` (exit 0 always, status via `connect`, `unset PET_*`,
   `torchrun --standalone` instead of the deepspeed launcher, `HOME=/tmp`, offline flags):
   `train_vlm.py --model-path /opt/models/qwen3.5-27b --data /opt/sft/train.jsonl --val
   /opt/sft/val_mini.jsonl --image-root-map /local/home/tianrgua/cvproject/data/yolo=/opt/smoke_data
   --epochs 1 --per-device-bs 2 --grad-accum 4 --lr 1e-4 --lora-r 16 --lora-alpha 32
   --max-pixels 1048576 --save-steps 200 --quant none` (no `--deepspeed`), then
   `eval_vlm.py --adapter <final> --list /opt/sft/val_eval_2000.txt --root-map ... --shard i/8`
   on 8 GPUs, then hold for `connect` pulls (adapter tgz, preds, metrics).
3. Devbox: score preds with the v2 protocol (`reports/v2` scripts), compare against the 9B bf16
   1 MP SFT and the 27B nf4 0.75 MP rows of `FINAL_REPORT_V2.md`; state that the base weights,
   data, steps and LoRA config are identical and only precision/resolution changed.

## 6. Risks / unknowns (ranked)
1. 70 GB image pull on the node (untested; if the nodepool evicts large images it is paid on
   every restart — and kraken restarts on any non-zero exit, so exit 0 always).
2. kraken submit-time image validation on a 70 GB image (untested).
3. HF SDPA efficiency on sm_100 (DINO run: ~12 % MFU) — the time estimate could double; still
   inside 4 node-h because the fixed costs dominate.
4. Single shared B200 node (cap 1): the Qwen arm cannot overlap the DINO track.
5. Devbox disk/network during the push (55–70 GB upload) — fine today (5.2 TB free), but it
   competes with `connect` pulls only marginally (different path).

## 7. Smoke test — RUN (200 steps; a plumbing/throughput measurement, NOT an SFT result)
Job `tianrgua-v5bqwensmk-2g5n` (registry `res-b200qwensmk`, image `smoke-detect-vlm-b200:v1`, entry
`cluster/v5b/qwen_b200_entry.sh`, orchestrator `cluster/v5b/orchestrate_b200_qwen.py`), submitted 13:46:40Z,
RUNNING 13:57:16Z, SUCCEEDED 14:35:55Z → **0.64 B200 node-hours** (pod exited 0 by itself after the pulls).

| measured item | value |
|---|---|
| image pull + scheduling (70.7 GB, cold) | 10 min 36 s submit→RUNNING; kraken validation OK |
| first read of the 11 shards (55.56 GB) from the image layer | 20 s (2.8 GB/s) |
| model | Qwen3.5-27B bf16, `attn_implementation=sdpa`, no quantization, no DeepSpeed, `torchrun --standalone` 8 GPUs (pure DDP) |
| LoRA | r16 α32 dropout 0.05 on 256 LM linears (q/k/v/o/gate/up/down; vision frozen) = **79,691,776 trainable (0.29 %)** |
| batch | bs 2/GPU × grad-accum 4 × 8 = eff. 64 (v2 value), 1 MP pixel budget, lr 1e-4 cosine |
| speed | **7.69 s/step** (train_runtime 1,537 s for 200 steps; 8.33 examples/s/node) vs v2 L40S nf4 0.75 MP: 17.2–20.2 s/step → **2.2–2.6× faster at a larger visual budget** |
| GPU memory | peak **66.5 GB used per GPU** (54 GB weights + ~12 GB LoRA/activations at bs 2) of 183 GB → bs 8/GPU would fit; GPU util 65–100 % |
| loss | 0.877 (step 5) → 0.523 (10) → 0.49 (15) … 0.426 (190) → **0.392 (200)**; v2 QLoRA nf4 was 0.385 @100 / 0.369 @200 — same regime |
| adapter | `models/v5b/b200/v5bB200_qwen27b_smoke/final_adapter.tgz` 294 MB, pulled via connect in 52 s |
| eval plumbing | `eval_vlm.py` with the adapter on 200 val images (8 shards × 25) = 471 s incl. 8 model loads; mean generation latency **14.6–16.3 s/img at 192 new tokens** (HF generate, bs 1) |
| eval quality | **not meaningful** (0.21 epoch): parse failures 72–88 % per shard (the known CoT-preamble habit of the 27B base, v2 note), img-recall 0.14–0.32 on 25-image shards — reported only to show the pipeline runs end to end |

What the smoke test changes in the plan:
- **GO for a follow-up window is confirmed**; all §6 risks 1–3 are measured away (pull 10.6 min, validation OK,
  7.7 s/step). Revised budget for the matched arm (938 steps, eff. 64, 1 MP, bf16): pull 11 min + training
  **2.0 h** + checkpoint/adapter pulls 5 min + eval of 2,000 images. The eval is now the dominant unknown:
  at 15 s/img, 2,000 images on 8 GPUs = **63 min** (192 tokens; the v2 re-eval needed 512 tokens + fallback
  parser → ~2.5 h) → **~3.3–4.5 node-h for one arm with a naive eval; ≤ 3.5 with batched generation
  (bs 8–16 per GPU fits easily in 66→110 GB) or `max_new_tokens 192` + a stop string.** Use bs 4 × ga 2 for
  the same eff. 64 to shave training time further.
- Everything is baked: a second arm (0.75 MP, isolating quantization) needs no new image.
- Not done here and not to be inferred from this: any quality claim about bf16-27B vs 9B/27B-nf4.

## 8. Accounting (Qwen arm)
Devbox: image build 12.8 min + push 32.9 min (70.7 GB; 36 MB/s). B200: **0.64 node-h** (smoke test). Artifacts:
`reports/v5b/b200_qwen/v5bB200_qwen27b_smoke/{orchestrator_summary.json,entry.log,load_check.json,params.json,
trainer_state.json,gpu_samples.csv,console_tail.txt,eval_shardlogs.txt}`, `models/v5b/b200/v5bB200_qwen27b_smoke/
{final_adapter.tgz,eval_preds.jsonl.gz}`, `cluster/generated/b200qwen_v5bqwensmk.yaml`, `logs/v5b/b200_qwen_v5bqwensmk.log`.
