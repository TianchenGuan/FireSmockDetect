# V5b — B200 (obsidian-pu, ap-south-1) end-to-end probe

Date 2026-10-02 (V5b phase 1). Bound: ~90 min / two meaningfully different attempts. Everything
below was measured in this session; nothing is carried over from v4 without re-verification.
Project `obsidian-pu`, queue `obsidian-pu-hyperpod-p6-b200.48xlarge-ap-south-1-tas` (1 node in quota,
whole node = 8×B200). Kraken job: `tianrgua-v5bb200p1-gfi3` (registry `res-b200probe1`, gpu-type B200).
Artifacts: `reports/v5b/b200_probe/v5bb200p1/` (pulled files), scripts in `cluster/v5b/`.

## 0. Headline
| item | result |
|---|---|
| node | 8× NVIDIA B200, 183,359 MiB each (178.4 GiB visible to torch), driver 580.167.08 (CUDA 13.0 driver), 192 vCPU, 2.0 TB RAM, /mnt/local NVMe 28 TB (25 TB free), /tmp tmpfs 1.9 TB |
| image stack | ap-south-1 `smoke-detect:v4`: torch 2.8.0+cu128, cuDNN 9.10.2, ultralytics 8.3.202, Python 3.11.13; `torch.cuda.get_arch_list()` includes **sm_100** (+sm_120); B200 reports capability (10,0); conv fwd+bwd + bf16 autocast matmul PASS |
| ECR pullable from obsidian-pu | ONLY after adding the cluster account **000000000004** to our repo policies (see §2). Kraken now validates images at submit time (`ACCESS_DENIED` before the fix). Cross-region (us-east-1) pull: see §2/§6 |
| S3 egress | **still blocked**: presigned PUT/GET (ap-south-1 regional endpoint) → HTTP 403 `AccessDenied ... with an explicit deny` (VPC-endpoint policy); us-east-1 endpoints resolve to public IPs but are unreachable (curl 000 / timeout); pod identity `ProjectRole-obsidian-pu-leviathan-prod-7-ap-south-1` → AccessDenied on our bucket even after a bucket-policy grant |
| retrieval channel that WORKS | `kraken -p obsidian-pu jobs connect -j <job> -c "<cmd>"` (kubectl-exec-like, non-interactive). Only the FIRST command of the `-c` string sees the pod filesystem; wrap in `sh -c '...'`. Pulled files with `tar czf - ... | base64 -w0` (§5) |
| toy DDP training | see §4 (attempt 1 stalled; attempt 2 fixed — numbers in §4) |

## 1. Access re-verification (facts)
- Kraken 1.0.1545.0. `kraken -p obsidian-pu queues list` → `obsidian-pu-hyperpod-p6-b200.48xlarge-ap-south-1-tas` Available 1 / Allocated 1 (in quota). Nodepool `obsidian_pu_hyperpod_p6-b200.48xlarge_ap-south-1_tas` 63 nodes, 53 used.
- obsidian-pu is **cluster `Leviathan-prod-7-ap-south-1`, AWS account 000000000004** (from `kraken jobs describe --json` of the stale iteration-1 job, confirmed by the kube context `kraken-000000000004-ScientistRole-obsidian-pu-Leviathan-prod-7-ap-south-1` and by STS inside the pod). This is NOT the iteration-1 obsidian cluster (prod-11, account 000000000005): the `/scratch` FSx is a different filesystem (12 TB, `/scratch/scratch/tianrgua-sandbox/` does not exist and is not creatable), so nothing from iteration 1 is on this node's FSx.
- Pods run as **uid 1001 (non-root)**; `/root` is unreadable → every asset baked under `/root/.config` (ultralytics font) is invisible. See §4 for the consequence.
- Stale iteration-1 job `tianrgua-sv4-dgnx` is STOPPED (not consuming).

## 2. ECR images (verified with `aws ecr describe-images`, ada creds in env only)
| registry | repo | tags present | repo policy pull principals (after this session) |
|---|---|---|---|
| ap-south-1 | smoke-detect | v1, v2, v3, v4 (+ an unrelated `nv6-hawthorn-v520`, 2026-09-20, not ours to use) | 000000000005, 000000000006, **000000000004 (added 07:47Z)** |
| ap-south-1 | smoke-dino3 | **repo does not exist** | — |
| ap-south-1 | smoke-detect-vlm | **repo does not exist** | — |
| us-east-1 | smoke-detect | v4, v5, v6, probe-cli | same three accounts (000000000004 added) |
| us-east-1 | smoke-dino3 | v1 | same |
| us-east-1 | smoke-detect-vlm | v1, v2, v3 (transformers 5.17.0 stack) | same |

- Submit #1 (before the policy change) failed at kraken's new pre-flight: `Job image validation failed ... smoke-detect:v4 (ACCESS_DENIED)`. After adding `arn:aws:iam::000000000004:root` to the `AllowLeviathanClusterPull`/`LeviathanPull` statements the same YAML validated and ran. Backups of the pre-change policies: `reports/v5b/b200_probe/policy_backup/*.before.json` (ECR ×4 + aps1 bucket). Revert = `aws ecr set-repository-policy --policy-text file://<before.json>`.
- Inside the pod DNS: `000000000001.dkr.ecr.ap-south-1.amazonaws.com → 10.0.13.228` (VPC endpoint) but `000000000001.dkr.ecr.us-east-1.amazonaws.com → 34.204.60.241` (public, unreachable). So a us-east-1 image (`smoke-detect:v6`, `smoke-dino3:v1`, `smoke-detect-vlm:v3`) cannot be pulled by this cluster unless it is also pushed to ap-south-1 ECR (devbox has local copies: `smoke-detect-vlm:v1/v2/v3`, `smoke-dino3:v1`; `docker tag` + push ≈ 5–10 GB each). Direct test of the cross-region pull: §6.

## 3. S3 egress from the pod (all three channels re-tested, all FAIL)
| channel | result (verbatim, truncated) |
|---|---|
| presigned PUT, `smoke-detect-data-...-aps1`, regional endpoint `s3.ap-south-1.amazonaws.com` (resolves to 3.5.208.177, public) | HTTP 403 `User: arn:aws:sts::000000000001:assumed-role/INTERNAL-ADMIN-ROLE/... is not authorized to perform: s3:PutObject ... with an explicit deny` |
| presigned GET, same bucket | HTTP 403, same explicit deny |
| presigned PUT/GET, `...-use1` via `s3.us-east-1.amazonaws.com` | curl `000` (TCP never completes; DNS resolves to a public IP) |
| pod identity `aws s3 cp` → aps1 (bucket policy grants 000000000005 + 000000000004 root) | `AccessDenied ... ProjectRole-obsidian-pu-leviathan-prod-7-ap-south-1 ... is not authorized to perform: s3:PutObject` |
| pod identity `aws s3 ls` aps1 | `AccessDenied ... s3:ListBucket` |
| pod identity → use1 | timeout (rc 124) |
| `aws sts get-caller-identity` (VPC endpoint 10.0.14.43) | OK: `arn:aws:sts::000000000004:assumed-role/ProjectRole-obsidian-pu-leviathan-prod-7-ap-south-1/...` |
| `/scratch` FSx (12 TB, 194 GB used) | mounted, but our sandbox path is not creatable (no writable location found for us) |

Conclusion: the v4 finding holds in the new cluster — S3 is explicitly denied for foreign principals at the VPC endpoint, and the project role has no grant to our bucket from the cluster side. Weight/data staging through S3 is impossible from these pods; the ONLY in-channels are the image itself and `kraken jobs connect` (push files in via base64 — slow, see §5); the only out-channels are the job exit code and `connect`.

## 4. Real forward+backward: toy DDP training on all 8 B200 (src/train/train_yolo.py shipped inline)
Toy set built in-pod from the image's baked data (`/opt/smoke_data`): first 2,048 lines of `lists/pod/expB_train.txt`
(pyro-sdis TRAIN sites; 1,420 with smoke boxes / 628 negatives) and first 128 lines of `sdis_val.txt` (105/23) as val.
No FIgLib, no test sites, no heinola. yolo11m COCO init, imgsz 640, batch 64 (= 8/GPU), workers 8, AMP, seed 0.

Three attempts, same node, same job:
| attempt | what | outcome |
|---|---|---|
| 1 — entry script as root-style (`cluster/v5b/b200_probe_entry.sh`, launched by the job) | stalled 20 min before spawning DDP, killed by its own `TRAIN_TIMEOUT` | **root cause**: pods run as uid 1001; `ultralytics check_det_dataset()->check_font()` finds no `Arial.ttf` under its (non-root) config dir and calls `downloads.is_url(github, check=True)` → `urlopen` with **no timeout**; the VPC accepts the TCP handshake to github.com (20.207.73.82:443, two ESTABLISHED sockets seen in `/proc/<pid>/net/tcp`) and blackholes it → infinite hang. After ~16 min the connection finally died and DDP started, 2 min before the timeout killed it. |
| 2 — same sequence relaunched in-pod via `connect` with `YOLO_CONFIG_DIR` + font staged | stalled again: (a) ultralytics resolves `USER_CONFIG_DIR = $YOLO_CONFIG_DIR/Ultralytics` (font was one level too high); (b) once fixed, `torch.distributed.run` honoured the operator-injected `PET_RDZV_BACKEND=c10d PET_RDZV_ENDPOINT=<pod>:23456 PET_RDZV_ID=<job>` and collided with attempt 1's late torchrun holding that rendezvous | diagnostic only |
| **3 — clean: font at `$YOLO_CONFIG_DIR/Ultralytics/Arial.ttf`, `YOLO_OFFLINE=true`, `PET_*` unset** (`cluster/v5b/b200_probe_attempt2.sh`, dir `/tmp/m2`) | **PASS, bits 56/56** — numbers below | |

Attempt-3 numbers (pod clock 08:14:56–08:16:30Z, 94 s wall for the whole sequence):
- run1 (epochs=3, killed by the monitor after the epoch-2 checkpoint landed): epoch 1 = 11.65 s, epoch 2 = 15.47 s (ultralytics `time` column, includes val on 128 images); steady-state **9.4–10.2 it/s at batch 64 → ≈600–650 img/s at 640 px** on 8×B200 for yolo11m (dataloader-bound: 32 iterations/epoch, 8 workers; not a GPU throughput ceiling). Train losses finite and decreasing (box 2.68→2.02, cls 10.13→2.13, dfl 1.72→1.39). `last.pt` with optimizer state = 121,027,178 B. Kill = SIGTERM to the process group (torchrun logged `SignalException ... signal: 15`), rc 143, 44 s wall; GPU memory back to 4 MiB on all 8.
- resume: checkpoint `epoch idx 1 | optimizer present True | train_args.epochs 3` → `Resuming training ... from epoch 3 to 3 total epochs` → epoch 3 = 12.63 s, `TRAIN_DONE`, rc 0, 40 s wall (incl. startup + final val + strip). results.csv rows 1–3 continuous.
- reload: final `last.pt` epoch −1, optimizer stripped (ultralytics final strip), `YOLO(last.pt).predict()` on a val image OK (300 raw boxes, 3.09 s first call incl. warm-up). Weights: `best.pt`, `epoch0.pt`, `last.pt` (sha256 `bc274474…2ed33`, 40,484,204 B).
- **Caveat**: val losses are `nan` and mAP 0 for all 3 epochs on this 128-image toy val set while train losses are finite. Not investigated (3 epochs from COCO init, tiny val); must be reproduced on an L40S node with the identical toy yaml before trusting B200 validation numerics in a real run.
- `check_amp` passed on B200 (AMP enabled); cuDNN 9.10.2; NCCL env from the operator: `NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO NCCL_ASYNC_ERROR_HANDLING=1`.

## 5. Log / checkpoint retrieval (what came back and how)
- S3 presigned PUT (ap-south-1 and us-east-1): **0 bytes retrieved** (403 / unreachable, §3). `/scratch` sandbox: not writable. Job exit code: works (see §6 for the observed code).
- **`kraken -p obsidian-pu jobs connect -j <job> -c "<cmd>"`**: works for obsidian-pu (kube context `kraken-000000000004-ScientistRole-obsidian-pu-Leviathan-prod-7-ap-south-1`). Rules learned: only the FIRST command of the `-c` string sees the pod filesystem → always `sh -c '<script>'`; never `pkill -f` a pattern that appears in your own `-c` text (killed my exec shell three times; use `^`-anchored patterns); don't `grep -v TTY` a base64 stream (it ate the blob) — filter with `^Updated kubeconfig|^Unable to use a TTY` only.
- Pulled with `tar czf - ... | base64 -w0` / `base64 -w0 file`: attempt-1 env files 4,518 B (tgz); attempts 2+3 logs 1,613,056 B (tgz); **`last.pt` 40,484,204 B in 14.0 s = 2.89 MB/s, sha256 identical to the in-pod value**. Extrapolation: a 1.2 GB DINOv3-L checkpoint ≈ 7 min, a 55 GB model ≈ 5.3 h per direction — usable for checkpoints/logs, not for staging large weights.
- Push INTO the pod worked the same way (scripts shipped as base64 in `-c`, 6.4 KB) — i.e. code can be injected at run time, data/weights only via the image.
- Archived: `reports/v5b/b200_probe/v5bb200p1/{attempt1_entry,attempt2_m,attempt3_m2}/`, `models/v5b/b200_probe/toy_yolo11m_640_3ep_b200_last.pt` (+ sha256 list).

## 6. Cross-region image pull, restart behaviour, exit-code channel
- **Cross-region pull is impossible by platform rule, not by permissions.** Job 2 (`cluster/v5b/b200_xregion_pull_probe.yaml`, us-east-1 `smoke-detect:probe-cli`, 332 MB) was rejected at submit: `Error: Requested image region (us-east-1) must match nodepool region (ap-south-1)`. Consumed 0 node-seconds (`reports/v5b/b200_probe/xregion/runner.log`). Consequence: every image for this node must be pushed to **ap-south-1** ECR (repos `smoke-dino3` and `smoke-detect-vlm` do not exist there yet; create + apply the 3-account pull policy + `docker tag/push` from the devbox copies — `smoke-detect-vlm:v3` 9.4 GB local, `smoke-dino3:v1` 7.9 GB local; `smoke-detect:v6` can be rebuilt FROM ap-south-1 `v4` in seconds since `Dockerfile.v6` only adds the entry script).
- **Non-zero exit ⇒ restart.** kraken enforces `maxRestarts ≥ 1`; with `restartPolicy: All` the container that exited `103` (100 + bitmask 3) was restarted ~7 s later and re-ran the whole probe on a node where the `/mnt/local` restart marker did not exist (`/scratch` sandbox is not writable, so no shared marker is possible). I stopped the job manually to save node-hours. For the exit-code channel to be usable here a probe must either exit 0 and carry results some other way, or find a shared writable location for a restart marker (none found), or kraken must accept a non-restarting policy (not tested — the only validation error seen was `max_restarts must be equal or greater than 1`).
- **Exit code visibility**: while the first container had already exited 103, `kraken jobs describe --json` showed `exitCode: []`, `restartCount: None`, `completionTime: ''`, i.e. the code was NOT surfaced (only the restart was). Treat the exit code as a last-resort channel; `connect` is the real one.
- Job 1 lifetime: RUNNING 07:50:20Z → stopped 08:27Z (≈0.62 node-h) + the queue wait. Registry B200 total for this probe: see §9.

## 7. GO / NO-GO
### (A) high-res DINOv3-L / H+ run on this node — **CONDITIONAL GO**
Verified: 8×B200 (178 GiB each), torch 2.8.0+cu128 with sm_100 kernels, cuDNN 9.10.2, fwd/bwd + bf16 + AMP + 8-GPU DDP + checkpoint save/resume/reload all work; checkpoints come back byte-exact via `connect` (1.2 GB ≈ 7 min at the measured 2.9 MB/s). Preconditions (all devbox-side, ~45–60 min):
1. Push `smoke-dino3:v1` (local 7.9 GB) to a new ap-south-1 repo `smoke-dino3` with the 3-account pull policy (000000000005, 000000000006, 000000000004) — cross-region pull is rejected by kraken.
2. Bake into that image everything the run needs: DINOv3 backbones (L `ea8dc286`, H+ `c807c9ee` — pinned HF revisions, devbox has internet), the COCO train json + images for the chosen corpus (only pyro-sdis + D-Fire are baked today), and put assets under `/opt`, not `/root` (pods run as uid 1001).
3. Job env: `HOME=/tmp`, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, and for ultralytics-style launchers `YOLO_OFFLINE=true` + a TTF at `$YOLO_CONFIG_DIR/Ultralytics/`; either `unset PET_*` before a self-spawned DDP launcher or launch with `torchrun` and let it use the operator rendezvous. Exit 0 always (non-zero ⇒ automatic restart; exit code is not surfaced); write status files and pull them with `connect`.
4. Open check before trusting B200 numbers: the toy run's `nan` val losses (§4) must be reproduced on L40S with the same yaml, or explained.
Budget: 1 node (cap), node-hours tracked as B200 in the registry; expect roughly ≥2× per-GPU L40S throughput for ViT-L at 1024–1280 px but no measurement exists yet.

### (B) Qwen3.5-27B BF16 LoRA SFT on this node — **NO-GO for the V5b window** (memory is fine; staging is the blocker)
- Memory: 27B in bf16 ≈ 54 GB weights per GPU + LoRA/activations fits in 178 GiB per GPU without quantization — no FSDP needed. That removes the v2 "27B loses under L40S quantization" confound.
- Image: `smoke-detect-vlm:v3` (transformers 5.17.0, trl 1.13.0) exists only in us-east-1 ECR; it is **not pullable in ap-south-1** (region rule). Devbox has the 9.4 GB image locally → push to a new ap-south-1 `smoke-detect-vlm` repo ≈ 15–25 min. Feasible.
- Base weights: Qwen3.5-27B (24 files, 55,586,167,982 B, S3 `smoke-detect-data-...-use1/weights/qwen3.5-27b/` and green `/scratch`) **cannot be staged through S3** — obsidian-pu pods get an explicit deny on presigned URLs (both regions) and the project role has no grant to our buckets (§3). Remaining channels: bake 55.6 GB into the image (≈1.5–2.5 h devbox→ECR ap-south-1 push + a long first pull; untested at this size) or `connect` push at 2.9 MB/s ≈ 5.3 h (impractical). SFT jsonl/images would need baking as well.
- Verdict: not within the 24 h V5b window unless the Qwen arm becomes a priority and the 55.6 GB image bake is accepted; retrieval of LoRA adapters (hundreds of MB) via `connect` would be fine.

## 8. Exact reproduction
```
# (all from repo root, .venv active, Midway + ada working; URLs/creds only ever in env or job YAML)
bash cluster/v5b/registry.sh reserve --job res-b200probe1 --nodes 1 --gpu-type B200 --track infra-probe --note "..."
# one-time (done 2026-10-02 ~07:47Z, backups in reports/v5b/b200_probe/policy_backup/): add arn:aws:iam::000000000004:root
#   to the pull statement of ECR repos ap-south-1/smoke-detect, us-east-1/{smoke-detect,smoke-dino3,smoke-detect-vlm}
#   and to the aps1 bucket policy (python/boto3 snippet in this session; revert with aws ecr set-repository-policy --policy-text file://<before.json>)
python cluster/v5b/submit_b200_probe.py --tag v5bb200p1 --retain-s 900 --pull-ckpt        # submits cluster/generated/b200probe_<tag>.yaml
python cluster/v5b/submit_b200_probe.py --tag v5bb200p1 --attach-job <jobname> --pull-ckpt  # re-attach poller
# in-pod diagnostics / fixes (first command of -c sees the pod FS; wrap scripts in sh -c):
kraken -p obsidian-pu jobs connect -j <jobname> -c "sh -c 'cat /tmp/probe_status; tail -n 20 /tmp/probe.log'"
B64=$(base64 -w0 cluster/v5b/b200_probe_attempt2.sh); kraken -p obsidian-pu jobs connect -j <jobname> \
  -c "sh -c 'mkdir -p /tmp/m && echo $B64 | base64 -d > /tmp/m/attempt2.sh && env -u PET_RDZV_BACKEND -u PET_RDZV_ENDPOINT -u PET_RDZV_ID -u PET_NNODES -u PET_NPROC_PER_NODE -u PET_MAX_RESTARTS nohup setsid bash /tmp/m/attempt2.sh > /tmp/m/nohup.out 2>&1 < /dev/null &'"
kraken -p obsidian-pu jobs connect -j <jobname> -c "sh -c 'tar czf - -C /tmp --exclude=*.pt m | base64 -w0'" 2>/dev/null \
  | grep -v "^Updated kubeconfig\|^Unable to use a TTY" | tr -d '\n' | base64 -d > pull.tgz
kraken -p obsidian-pu jobs connect -j <jobname> -c "base64 -w0 /tmp/m/run1/weights/last.pt" 2>/dev/null | grep -v "^Updated kubeconfig\|^Unable to use a TTY" | tr -d '\n' | base64 -d > last.pt
bash cluster/v5b/run_xregion_pull_probe.sh reports/v5b/b200_probe/xregion 480   # job 2 (rejected at submit: region rule)
kraken -p obsidian-pu jobs stop --job-name <jobname>; bash cluster/v5b/registry.sh release --job res-b200probe1 --outcome "..."
```
The repo copies of `b200_probe_entry.sh` / `b200_probe_attempt2.sh` already contain the font/offline fix; the entry still needs `PET_*` unset (or a torchrun launch) and an exit-0 policy before reuse.

## 9. Accounting and artifacts
- **B200 node-hours (separate from L40S)**: job `tianrgua-v5bb200p1-gfi3` RUNNING 07:50:20–08:27:30Z = **0.62 node-h = 4.96 GPU-h** (+ ~2 min PENDING/pull). Registry (reservation-based): 0.71 node-h over `res-b200probe1` (07:44–08:24Z, incl. the two rejected submits) + `res-b200probe2` (3 min, never scheduled). No L40S used by this agent. Both reservations released.
- Kraken: job 2 never created (submit rejected). Stale `tianrgua-sv4-dgnx` remains STOPPED.
- Files: `reports/v5b/B200_PROBE.md` (this), `reports/v5b/b200_probe/v5bb200p1/{attempt1_entry/,attempt2_m/,attempt3_m2/,describe_job1*.txt,orchestrator*.log,policy_backup/}`, `reports/v5b/b200_probe/xregion/`, `cluster/v5b/{b200_probe_entry.sh,b200_probe_attempt2.sh,submit_b200_probe.py,b200_xregion_pull_probe.yaml,run_xregion_pull_probe.sh}`, `cluster/generated/b200probe_v5bb200p1.yaml` (presigned URLs, expired ~08:45Z), `models/v5b/b200_probe/toy_yolo11m_640_3ep_b200_last.pt` (+ sha256 list). No annotations were produced or touched; no eval/test material used.
- Open items: `nan` val losses on the toy val set; kraken `restartPolicy` alternatives untested; `kraken jobs describe` does not expose the container exit code; `/scratch` on prod-7 has no writable location for us (ask the obsidian-pu owners for a sandbox if persistent storage is needed).
