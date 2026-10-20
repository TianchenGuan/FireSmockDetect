# V5 Day-1 — Shadow (bypass) recording for the smoke service: build + demo

Written 2026-10-02 (prep phase). Scope per PROTOCOL_V5.md §3 last bullet: every
shadow record carries `source` ∈ {live_upload, historical_replay}; replays are
never described as field runs. **Every record produced here is a historical
replay of stored frames. No live uploads exist.** (The 2 records tagged
`live_upload` are curl posts of stored frames made WITHOUT the form field, to
show the default tag — they are not field uploads either; see §5.)

Production `:8000` was not touched. It still runs the pre-V5 build
(healthz there has no `shadow` field); picking up the code below requires a
restart that only the owner should schedule.

## 1. What was built

| file | change |
|---|---|
| `src/serve/config.py` | **new optional settings, all default OFF** (`shadow_backend=""`): `shadow_backend`, `shadow_weights_path`, `shadow_model_version`, `shadow_imgsz`, `shadow_device` ("" = same as `device`), `shadow_conf_floor`, `shadow_conf_single`, `shadow_conf_seq`, `shadow_log_path` (JSONL, default `data/v5/shadow/shadow_log.jsonl`), `shadow_queue_max` (256), dino3-specific `shadow_dino_backbone_dir / shadow_lora_r / shadow_lora_alpha / shadow_lora_merge / shadow_det_width / shadow_det_height` (auto). `Settings()` with no env still gives yolo / `models/expB_best.pt` / 0.30 / 0.15 / 1280 (verified). |
| `src/serve/shadow.py` | **new** `ShadowRecorder`: loads the shadow backend via the same `backends.load_backend` registry (yolo, dino3 LoRA, …), owns a **separate `AlertEngine`** built from `settings.model_copy(update={single_frame_alert_conf: shadow_conf_single, seq_frame_conf: shadow_conf_seq})` (same persistence / cooldown / scene-change rules, own thresholds), runs in **one FIFO worker thread** (per-device history stays causal), appends one JSONL line per request, counts records by source / alert state, drops (and counts) when the bounded queue is full, never raises into the request path. `SOURCES = ("live_upload", "historical_replay")`. |
| `src/serve/app.py` | `/detect` gains optional form field `source` (default `live_upload`; anything outside `SOURCES` → 422). Per-source request counters. When a shadow is enabled the request hands the **same upload bytes** + the primary verdict to the worker **after** the primary result is computed and **without waiting** — the JSON response is built from the primary only and has no new keys. `/healthz` adds `sources_allowed`, `requests_by_source`, `shadow{enabled, backend, weights_path, model_version, device, conf_single, conf_seq, log_path, loaded, worker_alive, queue_depth, queue_max, records_written, records_by_source, alerts_by_state, dropped, errors, last_error, latency_ms_mean, latency_ms_max, backend_info}`. Shutdown hook drains/closes the log. |
| `src/serve/replay_client.py` | **always** sends `source=historical_replay` (module constant, not an option) and a `run_tag` (default `replay-<utc>-<label>`) inside `metadata`; every output record carries `source`, `run_tag`, `frame_path`. New `--sequence-json <sequences_*.json> --seq-id <id>` mode replays a pyro-sdis sequence with its real per-frame `ts` (the `--frames` FIgLib-offset parser mis-orders sdis names because they contain `-NN` tokens — documented, `--frames` left unchanged for FIgLib / 1-fps clips). Prints "HISTORICAL REPLAY … NOT a live field run". |
| `scripts/v5/test_shadow_service.py` | in-process test (FastAPI TestClient, app re-instantiated with/without shadow): response invariance, source tagging + 422, JSONL contents, replay-client contract. |
| `scripts/v5/run_shadow_demo.sh` | the demo driver below (spare port, stops its own server, proves :8000 untouched). |

### JSONL record schema (one line per `/detect`, written when the shadow pass finishes)
```
{ts, device_id, capture_ts, source, image_sha256, image_size, client_metadata,
 primary: {model_version, max_smoke, alert, alert_info{window_hits,window_size,history_reset,cooldown_active}, n_dets, latency_ms},
 shadow:  {backend, model_version, max_smoke, alert, alert_info{…}, n_dets, latency_ms, conf_single, conf_seq},
 queue_wait_ms}
```
`alert` is the AlertEngine state (`none|watch|provisional|confirmed`) — a detector
state, not a confirmed fire report; shadow silence is not evidence of no smoke.
`live_upload` means "the client did not declare a replay"; it is a channel tag,
not a provenance proof (a browser upload of a stored image is still tagged
`live_upload`).

## 2. Exact variables to enable the same shadow on :8000 later
```
cd /local/home/tianrgua/cvproject && source .venv/bin/activate
SMOKE_SHADOW_BACKEND=dino3 \
SMOKE_SHADOW_WEIGHTS_PATH=models/v3/runs/v4B4_lora_L/seg05/best.pt \
SMOKE_SHADOW_MODEL_VERSION=v4B4_lora_L@seg05-step15026 \
SMOKE_SHADOW_CONF_SINGLE=0.93 SMOKE_SHADOW_CONF_SEQ=0.82 \
SMOKE_SHADOW_LOG_PATH=data/v5/shadow/shadow_8000.jsonl \
SMOKE_SHADOW_DEVICE=cpu \
uvicorn src.serve.app:app --host 127.0.0.1 --port 8000
```
Primary stays at file defaults (expB@1280, 0.30/0.15) — no `SMOKE_BACKEND` /
`SMOKE_WEIGHTS_PATH` needed. Optional: `SMOKE_SHADOW_DEVICE=cuda:0`,
`SMOKE_SHADOW_QUEUE_MAX`, `SMOKE_SHADOW_LORA_MERGE=1`. The 0.93/0.82 pair is the
v4 val-selected operating point from reports/v4/BACKEND_DINO3_LORA.md (matched
expB's sdis-val neg-FP); the V5 fixed-camera calibration agent's output
supersedes it when available. **CPU caveat (§4):** on this host the dino3
shadow needs several seconds per frame; at a 29 s camera cadence it keeps up
(one device), but bursts queue (bounded, drops counted in `/healthz.shadow.dropped`).

## 3. Demo run `demo_20261002T0120Z` on :8012 (CPU, host load avg ≈100 from 7 other agents)
Artifacts: `data/v5/shadow/demo_20261002T0120Z_{shadow.jsonl, replay_sdis_val.jsonl,
replay_boreal_heinola_103.jsonl, healthz_start.json, healthz_end.json, server_8012.log, summary.json}`.
Server reached `status=ok, shadow.loaded=true` within ≈4 s (both models, warm cache);
`backend_info`: dinov3-vitl16, lora_r 16, letterbox 1280×736, ckpt_step 15026.

Inputs (all stored frames):
1. pyro-sdis **VAL** sequence `val_force-06_cabanelle-125_20240403T101630` (cabanelle = calibration/VAL site; 29 frames, 750 s span, median cadence 15 s, max gap 183 s; label pattern `flicker`: 14 frames labelled smoke / 15 not — flicker labels are known-noisy), replayed with real `ts` via `--sequence-json`, device `replay-sdis-cabanelle-125`.
2. Boreal **heinola_103** (aerial VAL, video-level label `no_smoke`, 30 frames @1 fps, 4096×2160), via `--frames --fps 1`, device `replay-boreal-heinola_103`. heinola material is VAL only and was not moved anywhere.
3. 2 curl posts with **no `source` field** (heinola_103_t0000 and sdis frame 10-16-42), device `demo-default-source-curl`.

### Counts by source (from `/healthz` at the end; identical in the JSONL)
| | requests_by_source | shadow.records_by_source |
|---|---|---|
| historical_replay | **59** | **59** |
| live_upload (default tag; curl posts of stored frames) | 2 | 2 |
| total | 61 | 61 (dropped 0, errors 0, queue_depth 0 at end) |

shadow `alerts_by_state` over 61 records: none 27, watch 2, provisional 3, confirmed 29.

### Per-clip outcome (frame states in order; N none, W watch, P provisional, C confirmed)
| clip | primary expB 0.30/0.15 | shadow dino3-B4 0.93/0.82 |
|---|---|---|
| sdis cabanelle-125 (14/29 frames labelled smoke) | `NWPCCCCCCCCCCCCCCCCCCNNPPCCCC` — 2 alert episodes; single-frame ≥0.30 vs frame labels: TP 10 FP 6 FN 4 TN 9 | `NNPPCCCCCCCCCCCCCCCCCCCCCCCCC` — 1 episode; ≥0.93: TP 10 FP 8 FN 4 TN 7 |
| heinola_103 (no_smoke clip) | `NNNNNNNNNNNPNNNNNNNNNNNNNNNNNN` — 1 false episode (t0011, 0.31) | `WPCCCCNNNNNNNNNNNNNNNNNNNNNNNN` — 1 false episode, stronger (t0001–t0003 at 0.97, confirmed t0002–t0005) |
| curl default-tag posts | none / watch (0.17) | watch (0.87) / none (0.64) |

Reading: on this one VAL flicker sequence both models alert on essentially the
whole sequence (the "FP" frames are mostly unlabelled frames between labelled
smoke frames, i.e. label flicker as much as detector error); on the one no-smoke
aerial clip **both** raise a false episode and B4 at its v4 thresholds is
worse, consistent with the V4 aerial findings. One clip each — anecdotal, no
rate is derived here (episodes/valid-hour is the calibration agents' job).
Note: pyro-sdis contains byte-identical consecutive frames (10-16-57 = 10-17-01,
10-28-30 = 10-28-34; `image_sha256` exposes this).

### Latency (CPU; host at load avg ≈100 — not a quiet-host measurement)
| | median | mean | max |
|---|---|---|---|
| primary expB@1280 (per response `latency_ms`) | 174 ms | 171 ms | 222 ms |
| shadow dino3-B4 1280×736 (`shadow.latency_ms`) | 9.06 s | 8.43 s | 10.25 s |

61 posts completed in **15 s wall** while the shadow worker needed **504 s** more to
drain (queue_wait up to 494 s): the primary path was not slowed by the shadow
(≈170 ms median either way, same as the test without a shadow). The v4 report
measured ≈2 s/frame for this backend on an idle CPU; today's 8–10 s is host
contention. For :8000 use `SMOKE_SHADOW_DEVICE=cuda:0` if a GPU is attached, or
accept lagged shadow records (they are time-stamped with the original
`capture_ts`, so ordering/analysis is unaffected).

### JSONL sample (record 3 of the sdis replay — first frame where both alert)
```json
{"ts":1790902030.732,"device_id":"replay-sdis-cabanelle-125","capture_ts":1712139417.0,
 "source":"historical_replay",
 "image_sha256":"efd4228a64e7f85119d111bb441de1b7c4302c8e62bf8ff22d99950bd67c2c84",
 "image_size":[1280,720],
 "client_metadata":"{\"run_tag\":\"demo_20261002T0120Z-sdis-val-cabanelle-125\",\"source\":\"historical_replay\",\"replay_client\":\"src.serve.replay_client\",\"frame\":\"force-06_cabanelle-125_2024-04-03T10-16-57.jpg\"}",
 "primary":{"model_version":"expB-mix-y11s-1280@seg03","max_smoke":0.6556,"alert":"provisional",
            "alert_info":{"window_hits":2,"window_size":3,"history_reset":null,"cooldown_active":false},
            "n_dets":1,"latency_ms":152.6},
 "shadow":{"backend":"dino3","model_version":"v4B4_lora_L@seg05-step15026","max_smoke":0.9486,"alert":"provisional",
           "alert_info":{"window_hits":1,"window_size":3,"history_reset":null,"cooldown_active":false},
           "n_dets":1,"latency_ms":5772.9,"conf_single":0.93,"conf_seq":0.82},
 "queue_wait_ms":11680.6}
```

Demo server stopped afterwards (`:8012` free); `:8000` verified still `ok`,
`expB-mix-y11s-1280@seg03`, uptime continuous (pre-V5 build).

## 4. Test — `scripts/v5/test_shadow_service.py` (PASS)
In-process, 4 stored frames (2 sdis cabanelle VAL, 2 heinola_103), app
instantiated twice (shadow OFF = file defaults; shadow ON = yolo expB as shadow
with 0.5/0.3 so the shadow path is exercised cheaply; `--shadow-dino` switches
to B4). Asserted: response key set+order identical and all values identical
after dropping `*_ms` timing keys (4/4; 0/4 byte-identical including timing,
as expected), response contains neither `shadow` nor `source`; default tag →
`live_upload`, explicit `historical_replay` accepted, `source=field_test` → 422
and not counted; `/healthz` `requests_by_source` = {live_upload 2,
historical_replay 2} in both passes and `shadow.records_by_source` equal in
the ON pass; JSONL has exactly 1 record per request with all required fields,
`source` = declared tag, `image_sha256` = SHA-256 of the uploaded bytes,
`primary.max_smoke/alert/model_version` = the HTTP response; `replay_client`
always emits `source=historical_replay`.

## 5. Statements and limitations
- **All 61 records are historical replays of stored frames; no live uploads exist.** The 2 `live_upload`-tagged records are deliberate default-tag demonstrations with stored frames (device `demo-default-source-curl`) and must not be read as field data.
- The tag records what the client declared. A client that omits the field is counted as `live_upload`; the web UI (`static/index.html`) omits it. If stricter provenance is wanted, make `source` required (one-line change, would break the current web UI and any legacy client).
- Thresholds 0.93/0.82 are v4 val-selected, not V5-calibrated; heinola_103 shows they are not an aerial operating point. V5 calibration results replace them before any :8000 enablement decision.
- Latency numbers are from a heavily shared CPU host; no GPU was used (no cluster reservation made).
- Demo-script quirk (fixed in the script, artifacts unchanged): `curl -F` cut the free-text `metadata` note of the 2 curl posts at a `;` — server-side echo only, no effect on records' required fields.
- `:8000` still serves the pre-V5 code; enabling the tag/shadow there requires an owner-scheduled restart. Nothing under reports/v4/, STATUS.md, pyro-sdis TEST, ruokolahti, AuSmoke test units or lockboxes was read or written by this task.
