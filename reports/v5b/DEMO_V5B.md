# V5b — FAST / ACCURATE service modes + upload/replay demo (2026-10-02)

Scope: additive serving modes in `src/serve/` and a browser demo. Production default behaviour is
unchanged: `SMOKE_MODE` file default = `fast` = the single yolo backend, `src/serve/config.py`
existing defaults untouched (expB@1280, 0.05/0.30/0.15), FAST `/detect` responses are
byte-identical to the pre-change contract (golden test below), `scripts/v5/test_shadow_service.py`
still passes unchanged. Production `:8000` was never touched (healthz uptime 27,805 s at the end
of the demo run). No cluster job was needed (CPU host only); nothing under `reports/v4|v5`,
`STATUS.md`, any test set, lockbox or FIgLib material was read for training or written.

## 1. What was built

| file | role |
|---|---|
| `src/serve/config.py` | **new optional fields only**: `mode` (`fast`), `accurate_models` (`""`), `accurate_fusion` (`or`), `accurate_recipe_path`, `accurate_fusion_iou`, `accurate_device`, `accurate_fused_fp_single/seq`, `demo_sdis_sequences`, `demo_boreal_clips`, `demo_runs_dir` |
| `src/serve/modes.py` | **new** — `parse_specs` (grammar `name=backend:weights[,key=value...]`), `AccurateEnsemble` (parallel `ThreadPoolExecutor` over the component models on the same full frame; per-model `AlertEngine` with the model's own thresholds; fusion `or` / `wbf` / `learned`; recipe loading; `/healthz` status), builtin `fuse_or` / `fuse_wbf`, the named `alert_equivalent` score map |
| `src/serve/app.py` | `/detect` form field `mode` (per-request override; 422 on unknown or unconfigured), ACCURATE branch `_detect_accurate` (additive keys `mode`, `fusion`, `models`, `model_errors`), `/healthz["modes"]` (both modes' model versions, fusion state, requests by mode) + `/healthz["demo"]`; demo router included. FAST branch and its response untouched |
| `src/serve/demo.py` | **new** — `GET /demo` (UI), `/demo/sequences` (allow-list: 7 pyro-sdis VAL sequences + Boreal `heinola_103`), `/demo/frame/{seq}/{idx}` (only allow-listed frames), `/demo/runs[/{id}]` (saved replays). Refuses pyro-sdis TEST sites and ruokolahti by construction |
| `src/serve/static/demo.html` | **new** — the demo UI (section 4). `static/index.html` (served at `/`) untouched |
| `scripts/v5b/test_modes.py` | mode tests (section 3) |
| `scripts/v5b/make_golden_fast.py` | captured `data/v5b/demo/golden_fast_responses.json` with the PRE-change code (4 images; `*_ms` stripped) |
| `scripts/v5b/demo_replay.py` | scripted historical replay of a `/demo/sequences` entry through `/detect` (`source=historical_replay` hard-wired via `src.serve.replay_client.build_form`), saves `data/v5b/demo/runs/<run_id>.json` that the UI renders (`/demo?run=<id>`) |
| `scripts/v5b/run_demo_v5b.sh` | start on :8013 -> replay sdis VAL seq + heinola clip in both modes -> headless-chrome screenshots -> stop |
| `scripts/v5b/demo_summary.py` | markdown tables from the saved runs (section 5) |

### ACCURATE response (additive to the FAST keys)
`detections` = fused boxes (`source=fused_<method>`, `models=[supporting models]`, `raw_scores`),
`max_smoke_score` = fused image score, `alert` = final alert (+ `by_model`, `decided_by`),
`model_version` = `accurate:<method>[<v1>+<v2>...]`, `latency_ms` = wall time of the parallel
section, `full_pass_ms` = slowest component, `mode="accurate"`, `fusion` = {method, impl, scale,
iou, recipe_path, recipe_applied, fallback_reason, normalization, final_alert_thresholds},
`models[name]` = {backend, model_version, weights, thresholds, detections (raw boxes + `score` +
`score_alert_equivalent` [+ `score_calibrated` when the recipe is applied]), n_smoke,
max_smoke_score, max_alert_equivalent, alert (that model's own state/reason/window), latency_ms,
timing}, `model_errors`. Alert history: FAST uses the existing engine; each ACCURATE component
has its own engine (own thresholds), so FAST responses do not change when ACCURATE requests
interleave (tested). The shadow recorder receives the fused verdict as `primary` in ACCURATE mode
(`primary.mode="accurate"`), FAST shadow records are unchanged.

### Score scales (named conversions, never a probability)
* `alert_equivalent` (builtin, default): per model, raw score mapped piecewise-linearly so that
  its own `(conf_seq, conf_single)` land on the FAST reference pair `(0.15, 0.30)`; identity for
  yolo models at 0.30/0.15; for dino B4 (0.93/0.82): 0.82 -> 0.15, 0.93 -> 0.30. `or` fusion =
  union + cross-model NMS on this scale; final alert = most severe per-model alert (true OR).
* `calibrated` (ensemble track): `scripts/v5b/fusion.py` + `models/v5b/fusion/accurate_recipe.json`
  (recipe `accurate_v5b`: yolo_A + dino_B4 + dino_B6 + dino_B1, WBF conf_type max, iou 0.3,
  neg-image-FP quantile scale). Applied ONLY when the configured component names equal the
  recipe model set (the recipe forbids silently emulating selective execution); then fused
  boxes come from `fuse()`, the image score from `image_score()`, and the final alert is a fused
  `AlertEngine` with `thresholds_oof[fp0.05]=0.999 / [fp0.10]=0.9918` (settings
  `accurate_fused_fp_single/seq`). Otherwise the recipe is loaded, displayed, and NOT applied,
  with the reason in `/healthz` and every response (`fusion.fallback_reason`). Verified end to end
  with all four recipe models (test E2 below).

## 2. How to run

```bash
cd /local/home/tianrgua/cvproject && source .venv/bin/activate
# production-identical FAST only (nothing set): uvicorn src.serve.app:app --port 8013
# demo configuration used here (FAST = file defaults expB@1280; ACCURATE = yolo v4a_s2 + dino3 B4 LoRA)
export SMOKE_ACCURATE_MODELS="yolo_A=yolo:models/trained/expV4A_ctrl_s2/seg05/best.pt,version=expV4A_ctrl_s2@seg05;dino_B4=dino3:models/v3/runs/v4B4_lora_L/seg05/best.pt,version=v4B4_lora_L@seg05-step15026,conf_single=0.93,conf_seq=0.82"
export SMOKE_DEMO_RUNS_DIR=data/v5b/demo/runs        # optional (default shown)
uvicorn src.serve.app:app --host 127.0.0.1 --port 8013   # then open http://127.0.0.1:8013/demo
# everything at once (server -> 4 replays -> screenshots -> stop):  bash scripts/v5b/run_demo_v5b.sh [--keep]
# scripted replay only:  python scripts/v5b/demo_replay.py --server http://127.0.0.1:8013 --seq boreal_heinola_103 --mode accurate --run-id acc_heinola
# per-request override:  curl -F image=@f.jpg -F source=historical_replay -F mode=accurate http://127.0.0.1:8013/detect
```
Env vars (all `SMOKE_*`, all optional): `MODE` (`fast`|`accurate`, default `fast`),
`ACCURATE_MODELS` (`;`-separated `name=backend:weights[,version=..,conf_single=..,conf_seq=..,conf_floor=..,imgsz=..,device=..,backbone_dir=..,lora_r=..,lora_alpha=..,lora_merge=..,width=..,height=..,tile_size=..]`; backend omitted -> yolo; thresholds omitted -> the FAST values),
`ACCURATE_FUSION` (`or`|`wbf`|`learned`; explicit value wins over the recipe),
`ACCURATE_RECIPE_PATH` (default `models/v5b/fusion/accurate_recipe.json`), `ACCURATE_FUSION_IOU`
(0.5; recipe `iou_thr` overrides unless set), `ACCURATE_DEVICE` (`""` = `device`),
`ACCURATE_FUSED_FP_SINGLE` / `_SEQ` (0.05 / 0.10 -> recipe `thresholds_oof` keys),
`DEMO_SDIS_SEQUENCES` (`""` = auto: the V5 shadow-demo sequence + the shortest 6-20-frame VAL
sequence per label pattern), `DEMO_BOREAL_CLIPS` (`heinola_103`; heinola only), `DEMO_RUNS_DIR`.
To get the calibrated recipe fusion, name the four components exactly `yolo_A`, `dino_B4`,
`dino_B6`, `dino_B1` with the recipe checkpoints (as test E2 does) — ~12 s/frame on this CPU.

## 3. Tests — `scripts/v5b/test_modes.py` (in-process TestClient)

| pass | checks | result |
|---|---|---|
| A file defaults | `/healthz.modes` (default fast, accurate disabled); 4 stored images (2 sdis cabanelle VAL + 2 heinola) -> FAST responses equal the pre-change golden (values AND key order, `*_ms` stripped); `mode=fast` explicit same keys; `mode=accurate` -> 422 "not configured"; `mode=turbo` -> 422; requests_by_mode {fast 5, accurate 0} | PASS |
| B accurate configured, default fast (+ shadow ON, recipe path isolated) | default requests still golden-identical with the ensemble loaded; `mode=accurate` x4: evidence for EVERY configured model (raw boxes, latency>0, version, own alert), fused boxes with supporting models, final alert = most severe per-model (or), `model_version` `accurate:or[...]`; the component sharing weights with FAST reproduces the FAST boxes exactly (evidence is real, not relabelled); alert_equivalent identity at 0.30/0.15 and threshold-crossing parity for dino (0.93->0.30, 0.82->0.15) in `--dino`; invalid inputs: unreadable -> 422 (both modes), oversized -> 413, bad source -> 422 (checked before mode), bad mode -> 422, empty mode -> default; 422/413 do not increment counters; shadow log = 1 record per 200 response (9), accurate records carry `primary.mode=accurate`, FAST records unchanged | PASS |
| C `SMOKE_MODE=accurate` | bare requests are ACCURATE; per-request `mode=fast` returns the golden FAST contract; counts {fast 4, accurate 1} | PASS |
| D fusion selection | simple recipe + explicit `wbf` -> builtin wbf on alert_equivalent with `fallback_reason`; explicit `or` wins over a recipe; `learned` without applicable recipe -> `or` + reason | PASS |
| E1 real recipe, 2 components | recipe loaded (`recipe.models` = 4 names), NOT applied, default stays `or`, reason names the expected model set | PASS |
| E2 `--recipe-full` (4 recipe components) | `method=wbf`, `recipe_applied`, `scale=calibrated`, thresholds from `thresholds_oof`, `impl=scripts/v5b/fusion.py`, per-model `score_calibrated`/`max_calibrated`; 4 images 200 | PASS (12.2-12.6 s/frame CPU: yolo_A 0.34-0.44 s, dino_B4 5.5-5.7 s, dino_B6 12.2-12.6 s, dino_B1 5.8-6.1 s) |
| `scripts/v5/test_shadow_service.py` (unchanged, run as subprocess) | response invariance + source tagging + JSONL | PASS |

Runs: default (`expB_alt`+`v4a_s2` components) PASS incl. shadow test; `--dino --skip-shadow-test`
(ACCURATE = v4a_s2 + dino3 B4, the demo set) PASS, ACCURATE latency 5.6-5.8 s/frame;
`--recipe-full --skip-shadow-test` PASS (logs: `data/v5b/demo/test_modes_dino.log`,
`data/v5b/demo/test_modes_recipe_full.log`). Golden: `data/v5b/demo/golden_fast_responses.json`
(sha256 of each input checked at test time).

## 4. Demo UI (`GET /demo`, screenshots `reports/v5b/demo/`)

* Mode selector (FAST / ACCURATE; ACCURATE disabled when the server has no components), upload
  (drag/drop -> `/detect` with `source=live_upload`; the UI says a stored frame uploaded this way
  is still not a field observation), historical sequence picker (replays frame by frame through
  `/detect` with `source=historical_replay`, real `capture_ts`, one `device_id` per sequence and
  mode, run tag in `metadata`; Stop button; progress), saved-run loader (`/demo/runs`), timeline
  JSON download.
* Frame view: per-model raw boxes in a fixed colour per model (dark-surface categorical palette
  validated with the dataviz palette checker: `#3987e5`, `#d95926`, `#199e70`), fused boxes
  white dashed with the supporting models, FAST boxes yellow; dataset author boxes (pyro-sdis
  labels) thin grey, toggle; legend chips.
* Timeline: per-frame max score (fused alert-equivalent in ACCURATE, raw in FAST; one axis),
  per-model alert-equivalent lines, reference threshold lines (0.15 watch / 0.30 provisional),
  marker colour and an alert strip by state (status palette + glyph, never colour alone), a
  dataset-label strip, hover tooltip, click-to-select frame, table view.
* Models panel for the selected frame: backend, version, thresholds, raw max, alert-equivalent
  max, own alert state, latency (per model + parallel wall + client wall), `score_semantics`.
* Server card from `/healthz`: default mode, FAST version, ACCURATE components + versions +
  LoRA info, fusion method / recipe status / fallback reason, requests by mode and by source,
  shadow state.
* Invalid-input probes with visible outcomes: unreadable bytes -> 422 "not a decodable image";
  26 MB upload -> 413 "upload exceeds 25 MB"; `source=field_test` -> 422; `mode=turbo` -> 422
  (verified in a real browser via the browser-use session: `demo_invalid_input_probes.png`;
  counters unchanged by the rejected requests).
* Mask overlay slot: checkbox stays disabled; the page probes `GET /segment` and reports
  "not available" (404) — nothing is drawn until a real `/segment` returns masks.
* URL params: `?run=<id>&frame=<idx>`, `?seq=<id>&mode=<m>&autorun=1` (used for the screenshots
  and for the live UI replay check `demo_ui_live_replay_fast.png`: 6-frame sdis VAL sequence
  replayed by the page itself).

Screenshots (headless Chrome for Testing 153, `~/.cache/ms-playwright/chromium-1243`):
`demo_landing.png`, `demo_fast_sdis.png`, `demo_accurate_sdis.png` (frame 10 of cabanelle-125),
`demo_fast_heinola.png` (frame 11, the FAST false alert), `demo_accurate_heinola.png` (frame 3),
`demo_invalid_input_probes.png`, `demo_ui_live_replay_fast.png` (the four run screenshots were re-taken on a
freshly started server after the scripted run, so their server card shows 0 requests). Timelines as JSON:
`data/v5b/demo/runs/{fast,acc}_{sdis,heinola}.json`; healthz snapshots
`data/v5b/demo/healthz_{start,end}.json`; server log `data/v5b/demo/server_8013.log`.

## 5. Replay results (historical replay, CPU host, fusion `or`; not field runs)

Sequences: pyro-sdis VAL `val_force-06_cabanelle-125_20240403T101630` (cabanelle = VAL site,
29 frames / 750 s, 14 frames carry a dataset smoke label, the V5 shadow-demo sequence) and
Boreal `heinola_103` (aerial VAL, 30 frames @ 1 fps, clip-level dataset label `no_smoke`).
Labels are dataset author labels; nothing here was human-reviewed in this project.

| run | mode | frames | none/watch/prov/conf | first provisional+ (frame, offset s) | server latency ms median / p90 / max | per-model latency ms median | per-model alert counts (none/watch/prov/conf) |
|---|---|---|---|---|---|---|---|
| fast_sdis | fast (expB) | 29 | 3/1/3/22 | 2, 27.0 | 145.9 / 151.0 / 293.8 | - | - |
| acc_sdis | accurate | 29 | 2/0/2/25 | 2, 27.0 | 5816.0 / 6887.8 / 7285.6 | yolo_A 319.1, dino_B4 5815.3 | yolo_A 9/1/3/16, dino_B4 2/0/2/25 |
| fast_heinola | fast (expB) | 30 | 29/0/1/0 | 11, 11.0 | 169.2 / 199.0 / 203.2 | - | - |
| acc_heinola | accurate | 30 | 19/1/2/8 | 0, 0.0 | 5764.4 / 5992.6 / 6372.0 | yolo_A 339.2, dino_B4 5763.6 | yolo_A 19/1/2/8, dino_B4 24/1/1/4 |

| seq | FAST frames in provisional/confirmed | ACCURATE frames | both | FAST only | ACCURATE only |
|---|---|---|---|---|---|
| cabanelle-125 (14 label-smoke frames of 29) | 25 | 27 | 25 | 0 | 2 |
| heinola_103 (clip label no_smoke) | 1 | 10 | 0 | 1 | 10 |

Readings (demo-scale, 2 sequences, no statistical claim):
* On the sdis VAL sequence both modes raise the first provisional alert at the same frame
  (27 s); ACCURATE (or) stays in alert on 2 more frames (dino_B4 keeps scoring >= 0.82 on
  frames where expB drops out). Note the `confirmed` state persists on frames with max score 0
  (frames 27-28): this is the existing persistence semantics (hits in the last 5 frames), not a
  V5b change.
* On the aerial no-smoke clip, `or` fusion is the union of false alerts: FAST 1 frame
  (frame 11), ACCURATE 10 frames (yolo_A at 0.30/0.15 fires on frames 0-9 with raw 0.10-0.45;
  dino_B4 at 0.93/0.82 fires on frames 1-5 with raw 0.42-0.97). All thresholds here are the
  FIXED-CAMERA val-selected points; the aerial calibration (reports/v5/CALIBRATION_AERIAL_V5.md)
  puts B4's aerial conf* at 0.989 and expB's at 0.107, so this clip must not be read as an
  ensemble-quality statement — it shows why the demo reports per-model evidence and why
  per-scenario thresholds (PROTOCOL_V5 §1) are needed before ACCURATE is deployed anywhere.
* Latency (CPU, 192-core host shared with other V5b agents): FAST 146-169 ms median;
  ACCURATE wall = slowest component = dino_B4 5.8 s median (yolo_A 0.32-0.34 s in parallel).
  BACKEND_DINO3_LORA.md measured B4 at 2.17 s on an idle host with torch's default threads;
  the difference is host load + two models sharing the CPU, not a serving change. GPU reference
  (v4, L40S batch 1): yolo 11.2 ms, B4 157.5 ms — ACCURATE is a GPU or offline configuration.

## 6. Limitations / honest notes
* CPU host only; ACCURATE is 5.8 s/frame here (12.3 s with the full 4-model recipe set) —
  historical replay only, no live or real-time run, no GPU serving measured in V5b.
* Historical replay only: every demo request is `source=historical_replay` (118/118 in the final
  run); a browser upload is tagged `live_upload` because the client declared it, which still
  does not make a stored frame a field observation.
* The recipe (`accurate_v5b`, 4 models, fixed-camera scenario) is NOT applied in the 2-model
  demo configuration asked for (yolo v4a_s2 + dino3 B4) — stated in `/healthz` and in every
  ACCURATE response; the demo fuses with builtin `or` on the alert-equivalent scale. The
  calibrated path is verified by test E2 only (4 images), not replayed in the UI.
* `alert_equivalent` is a threshold-anchored convenience scale for drawing/NMS, not a
  calibration; per-model alerts never depend on it. `calibrated` is the ensemble track's
  neg-image-FP quantile scale whose per-site transfer problem is documented in the recipe.
* ACCURATE requests are serialised (one request at a time per model object; the service already
  blocks the event loop in FAST), so throughput is 1/latency.
* `learned` fusion has no implementation yet (falls back to `or`, loudly).
* Mask overlay is a stub (no `/segment` on this server; nothing is faked).
* `/healthz` gained `modes` and `demo` keys (additive, as V5 added `shadow`); the FAST `/detect`
  payload gained nothing.
* The demo allow-list is VAL-only (sdis cabanelle/marguerite sequences, heinola clip); TEST
  sites, ruokolahti and FIgLib are refused by construction and were not read.
