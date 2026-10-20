# 5-minute demo script — FAST / ACCURATE smoke detection service (V5b)

_Everything below replays stored VALIDATION frames through the real service on a spare port (:8013). Production :8000 is never touched. Every request is tagged `source=historical_replay`; nothing here is a field observation. Source of truth for the setup: `reports/v5b/DEMO_V5B.md`; launcher: `scripts/v5b/run_demo_v5b.sh`._

_Post final test (2026-10-03, `reports/v5b/FINAL_REPORT_V5B.md` §7-§8): the FAST default shown is still the deployed expB; the V5b FAST candidate is S1_A = the existing recipe at YOLO11m (`models/trained/expV5bA_ctrl_s3/seg05/best.pt`, +3.4 pt H2 R@10% over expB, H1 not lost) — recommended, NOT flipped into production in this window; ACCURATE did not beat the best single model on the test (0.366 vs 0.388 @10% FP) and is shown here only as a demo of per-model evidence._

## 0. Before the audience arrives (T-10 min) — start and warm the server

```bash
cd /local/home/tianrgua/cvproject && source .venv/bin/activate
ss -ltn | grep -q ":8013 " && echo "port busy" || true          # must be free; never use 8000
# ACCURATE = yolo expV4A_ctrl_s2 (0.30/0.15) + dino3 v4B4_lora_L (0.93/0.82), fusion "or" (the demo set)
export SMOKE_ACCURATE_MODELS="yolo_A=yolo:models/trained/expV4A_ctrl_s2/seg05/best.pt,version=expV4A_ctrl_s2@seg05;dino_B4=dino3:models/v3/runs/v4B4_lora_L/seg05/best.pt,version=v4B4_lora_L@seg05-step15026,conf_single=0.93,conf_seq=0.82"
export SMOKE_DEMO_RUNS_DIR=data/v5b/demo/runs                     # the four saved replays live here already
nohup uvicorn src.serve.app:app --host 127.0.0.1 --port 8013 > /tmp/demo_8013.log 2>&1 &
# wait until both modes are loaded (DINOv3-L on CPU takes ~1-2 min)
until curl -sf http://127.0.0.1:8013/healthz | python -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d['status']=='ok' and d['modes']['accurate'].get('loaded') else 1)"; do sleep 3; done
# warm-up: one FAST and one ACCURATE request so the first live call is not the slow one
IMG=$(head -1 data/yolo/lists/local/sdis_val.txt)
curl -s -F image=@"$IMG" -F source=historical_replay -F mode=fast     http://127.0.0.1:8013/detect | python -c "import sys,json; print('fast warm', json.load(sys.stdin)['latency_ms'])"
curl -s -F image=@"$IMG" -F source=historical_replay -F mode=accurate http://127.0.0.1:8013/detect | python -c "import sys,json; print('accurate warm', json.load(sys.stdin)['latency_ms'])"
```
Open two browser tabs: **A** `http://127.0.0.1:8013/demo?run=acc_sdis&frame=10` and **B** `http://127.0.0.1:8013/demo?run=acc_heinola&frame=3` (saved runs load instantly, no inference). Keep a third tab on `http://127.0.0.1:8013/demo` for the live replay.

**Fallback if the server is not ready / dino fails to load:** start FAST-only (`unset SMOKE_ACCURATE_MODELS`) — the saved ACCURATE runs still render from `data/v5b/demo/runs/acc_*.json`; the screenshots in `reports/v5b/demo/*.png` are the last resort.

## 1. 0:00-0:40 — What you are looking at
- Tab A (saved run `acc_sdis`, frame 10 of pyro-sdis VAL sequence `val_force-06_cabanelle-125_20240403T101630`, 29 frames / 750 s, 14 frames carry a dataset smoke label).
- Point at the **server card** (from `/healthz`): default mode FAST = `expB-mix-y11s-1280@seg03` — the *deployed* default; the V5b FAST **candidate** S1_A (`expV5bA_ctrl_s3@seg05`, same yolo backend, H2 R@10% 0.388 vs expB 0.354, `reports/v5b/FINAL_REPORT_V5B.md` §8) is recommended but NOT flipped in — say so; ACCURATE components `expV4A_ctrl_s2@seg05` (yolo) + `v4B4_lora_L@seg05` (dino3, LoRA r16), fusion `or`, recipe status "loaded, not applied" with the reason (recipe needs the 4-model set).
- Say: "Every box you will see is attributed to a model; nothing is drawn that the service did not return."

## 2. 0:40-1:40 — Per-model evidence, fused boxes, alert state (sdis sequence)
- Frame view: **blue = yolo_A raw boxes, orange = dino_B4 raw boxes, white dashed = fused boxes with the supporting model names, thin grey = dataset author boxes** (toggle on).
- Models panel: backend, version, thresholds (0.30/0.15 vs 0.93/0.82), raw max, alert-equivalent max, each model's **own alert state**, per-model latency (yolo_A ~0.3 s, dino_B4 ~5.8 s CPU) and the parallel wall time.
- Timeline: click frame 2 — first `provisional` alert at +27 s in **both** modes (same frame as FAST, run `fast_sdis`); then frames 27-28: `confirmed` persists although max score is 0 — point out this is the existing 5-frame persistence semantics, not a V5b change.
- Say: "ACCURATE stays in alert on 2 extra frames because dino_B4 keeps scoring >= 0.82 where expB drops out (`DEMO_V5B.md` §5)."

## 3. 1:40-2:40 — Live replay (FAST is fast; ACCURATE is honest about CPU cost)
- Third tab: pick the same sdis sequence, mode **FAST**, Run -> 29 frames stream in ~5-10 s (146-169 ms/frame on this CPU). Watch the alert strip fill.
- Then run ACCURATE **with a limit**, from a shell (the UI would take ~3 min for 29 frames at 5.8 s/frame):
```bash
python scripts/v5b/demo_replay.py --server http://127.0.0.1:8013 --seq val_force-06_cabanelle-125_20240403T101630 \
  --mode accurate --limit 3 --run-id live_acc_sdis_3f --runs-dir data/v5b/demo/runs | tail -3
```
  then open `http://127.0.0.1:8013/demo?run=live_acc_sdis_3f&frame=2`.
- Say: "Measured, not claimed: FAST 11 ms on an L40S, ACCURATE 352 ms per frame in parallel on GPUs (`reports/v5b/tables/T7_*.md`); on this CPU box ACCURATE is 5.8 s per frame — it is a GPU or offline configuration."
- **Fallback if the live FAST replay stalls** (host under load): show `reports/v5b/demo/demo_ui_live_replay_fast.png` and skip the shell call.

## 4. 2:40-3:30 — Failure case: aerial no-smoke clip at fixed-camera thresholds
- Tab B (saved run `acc_heinola`, Boreal heinola_103, 30 frames @ 1 fps, clip label **no_smoke**, aerial VAL).
- Timeline: FAST (run `fast_heinola`) raises **1** false provisional alert (frame 11, +11 s); ACCURATE with `or` fusion raises **10** — yolo_A fires on frames 0-9 at raw 0.10-0.45 with the 0.30/0.15 thresholds, dino_B4 on frames 1-5.
- Say: "This is the union of two models' false alarms at the wrong thresholds. The aerial calibration (`reports/v5/CALIBRATION_AERIAL_V5.md`) puts B4's aerial operating point at 0.989 and expB's at 0.107 — the ensemble study (`reports/v5b/ENSEMBLE_DEV_V5B.md` T8) shows thresholds do not even transfer between two fixed cameras. Per-scenario, per-camera thresholds are a requirement, not a tuning detail."
- Click frame 3 to show the per-model evidence behind one false alert (which model, which raw score).

## 5. 3:30-4:15 — Invalid-input probes (the service refuses, counters do not move)
```bash
H=http://127.0.0.1:8013
curl -s -o /dev/null -w "unreadable bytes -> %{http_code}\n"  -F image=@/etc/hostname -F source=historical_replay $H/detect          # 422 not a decodable image
curl -s -o /dev/null -w "bad mode       -> %{http_code}\n"  -F image=@"$IMG" -F source=historical_replay -F mode=turbo $H/detect   # 422 unknown mode
curl -s -o /dev/null -w "bad source     -> %{http_code}\n"  -F image=@"$IMG" -F source=field_test $H/detect                         # 422 (checked before mode)
head -c 27000000 /dev/urandom > /tmp/big.bin; curl -s -o /dev/null -w "26 MB upload   -> %{http_code}\n" -F image=@/tmp/big.bin -F source=historical_replay $H/detect   # 413
curl -s $H/healthz | python -c "import sys,json; d=json.load(sys.stdin); print('requests_by_mode', d['modes']['requests_by_mode'])"
```
- Say: "Rejected requests never reach a model and never count as traffic (`scripts/v5b/test_modes.py` pass B)." The UI's probe panel shows the same four outcomes (`reports/v5b/demo/demo_invalid_input_probes.png`).

## 6. 4:15-4:50 — Optional: the segmentation overlay as an explanation layer (CPU, 3 s)
```bash
python scripts/v5b/seg_infer.py --weights models/v5b/seg/segV5b_a/seg02/frozen/best.pt \
  --image data/v4/eval/boreal_val/images/$(ls data/v4/eval/boreal_val/images | grep heinola_DJI_0033 | head -1) \
  --out /tmp/seg_prob.png --overlay /tmp/seg_overlay.png && echo "overlay at /tmp/seg_overlay.png"
```
- Open `/tmp/seg_overlay.png` (or the pre-rendered `reports/v5b/seg_examples/infer_demo_frozen_heinola_DJI_0033_frame48_overlay.png` if the CPU is busy).
- Say: "IoU 0.568 on heinola, 2.6-2.9 s per frame on CPU; but it false-alarms on every frame of one clear-cut flight (`reports/v5b/seg_examples/segV5b_a_lora_08_FAIL_false_alarm_heinola_DJI_0024_frame56.png`), so it is drawn as context under the detector's boxes, never as an alert. The `/segment` slot in this demo is still a stub (404) — nothing is faked."

## 7. 4:50-5:00 — Close
- "Three things this service now does that it did not before: it runs two models in parallel and shows you each one's evidence; it refuses to pretend a stored frame is a field observation; and it tells you when the calibrated recipe is *not* being applied. What it does not yet do is run ACCURATE in real time on a CPU, or set its own thresholds per camera — both are on the list."
- "And the two outcomes of the final test that frame this demo: the FAST model you saw is still the deployed expB (YOLO11s); the V5b candidate to replace it is the plain control A — the existing recipe retrained at YOLO11m — at +3.4 pt recall over expB at 10% FP on the frozen test, not flipped into production yet. The ACCURATE ensemble did *not* beat that single model at matched FP on the test (0.366 vs 0.388 at 10%), so what you saw is a demo mode for per-model evidence, not a shipped mode."
- Stop the server when done: `pkill -f "uvicorn src.serve.app:app --host 127.0.0.1 --port 8013"`; the saved runs stay in `data/v5b/demo/runs/`.

## Timing and fallbacks at a glance
| segment | planned | if slow / broken |
|---|---|---|
| start-up | T-10 min | FAST-only server + saved ACCURATE runs; screenshots `reports/v5b/demo/*.png` |
| live FAST replay (29 frames) | ~10 s | `demo_ui_live_replay_fast.png` |
| live ACCURATE (3 frames) | ~20 s | skip the shell call; saved `acc_sdis` run already open |
| invalid probes | 4 curls, <5 s | `demo_invalid_input_probes.png` |
| segmentation | ~3 s + load | pre-rendered overlay in `reports/v5b/seg_examples/` |
