# RECONCILIATION — Iteration 4 artifact audit (2026-09-14)

Scope: verify FINAL_REPORT.md, FINAL_REPORT_V2.md, FINAL_REPORT_V3.md,
reports/v3/LOCKBOX_V3_RESULTS.md and STATUS.md tail against on-disk artifacts.
Historical reports are NOT edited; all inconsistencies are recorded here only.

## 1. Verified claims (spot checks against artifacts)

Checkpoints (all exist, zip-CRC valid via `zipfile.testzip`):
- models/expB_best.pt 19,241,370 B; sha256 `cc56224c…ff0e` == byte-identical to
  models/trained/expB_mix_y11s_1280/seg03/best.pt (v1 claim "copy" verified).
- expA seg02 (19.2 MB), expE seg05 (40.6 MB), B1 seg02 (267.7 MB),
  B2 seg04 (301.7 MB), B3 seg04 (872.5 MB), C1 seg04 (1.143 GB),
  trackd heads best.pt (1,130,033 B ≈ 281k params) — all present and intact.
- models/v3/rfdetr/checkpoint_best_ema.pth 141,649,947 B; sha256
  `2cca1fe8…43d0` matches PROVENANCE.md exactly. (Note: the run dir
  models/v3/runs/rfdetrL_1280_e48/a01 holds only console/status; checkpoint
  location is documented in PROVENANCE.md — not a gap.)
- v2 adapters gzip -t OK: trkA_9b_lora/seg05, trkB_27b_qlora/seg10,
  grpo_9b_500/attempt02 final_adapter.tgz (46 MB / 295 MB / 112 MB).

Metrics reproduced EXACTLY from JSON artifacts:
- v1 test table @conf 0.2 == eval_{expA,expB,baseline_sdis}_test.json:
  expA 0.686/0.552/0.445/0.323; expB 0.684/0.646/0.541/0.353;
  baseline 0.558/0.880/0.872/0.547. n_images=4,124 as stated.
- v1 training mAPs == results.csv final epoch 60: expA 0.69834/0.46719,
  expB 0.73771/0.49564 (report: 0.698/0.467, 0.738/0.496). expE 0.74747
  matches STATUS "ultralytics' 0.747".
- FIgLib-39 replays == replay_*.json summaries: expA 30/39 med 180 s pre-vis
  10; expB 32/39 med 475 s pre-vis 5; tiled 37/39 & 36/39 med 0 s pre-vis 27;
  baseline 37/39 med 61 s; expE 30/39 med 240 s; rfdetrL 33/39 med 300 s;
  B1 34/39 med 240 s; B2 32/39; C1 30/39 med 360 s.
- v2 lockbox == reports/v2/LOCKBOX_replay_*.json: expE 20/25 med 299 s
  pre-vis 2; expB 20/25 med 240 s pre-vis 4; tiled 25/25 med 0 s pre-vis 20.
- Lockbox v3 == replay_lockboxv3_*.json: expE 21/25 med 120 s pre-vis 5;
  rfdetrL 20/25 med 179 s pre-vis 4; B1 24/25 med 477 s pre-vis 4.
  R@1/3/5 min recomputed from per-event records: 9/11/12, 7/11/11, 8/12/12 —
  all match. Frame scores exist (1,978 lines each in
  models/v3/evals/lockbox_v3/{B1,rfdetrL}_preds.jsonl = manifest frame count).
- Track B/C pAUC table reproduced exactly by the project's own
  scripts/v3/pauc.py on matched_fp_{val,test}_trackBC.json: val C1 0.531/0.430
  > expB 0.523/0.420 > expE 0.520/0.422 > B2 0.517/0.417 > rfdetrL 0.515/0.415
  > B1 0.487/0.378 > B1cached 0.477/0.358 > B3 0.474/0.357; test C1 collapse
  0.469/0.304 vs expE 0.526/0.435 — all match FINAL_REPORT_V3 §Tracks B/C.
- Track D == trackD_eval_val*.json: learned head R@FPR2% (all-neg) 0.1512 /
  0.1813 (seeds) vs persistence 0.0826 vs single-frame 0.1004; PR-AUC
  0.9404/0.9351/0.9169; t1-only mean 0.120 (0.1264, 0.113). All match.
- Latency artifact: rfdetr CPU-8t 1016 ms vs yolo11m 324 ms (STATUS) ==
  reports/v3/latency_*.json.

Split/lockbox integrity:
- data/manifests/splits_frozen.json: counts 22,177 / 7,335 / 4,124 / 38,454
  and sha256 of all four data/yolo/lists/local/*.txt match the frozen record
  (pod copies differ only by path prefix, as designed).
- Lockbox v2 parquet sha256 `1c777845…ebab` matches LOCKBOX_FROZEN.md.
- Lockbox v3 parquet sha256 `66514b03…c07608` matches LOCKBOX_V3_FROZEN.md
  and the value quoted in LOCKBOX_V3_RESULTS.md.

## 2. Discrepancies found (report text vs artifact)

D1. Serving default (v1/v2/v3 "deployed default = yolo + expB@1280").
    src/serve/config.py FILE defaults are weights_path
    `models/pyronear_yolov11s/best.pt` (the CONTAMINATED baseline),
    model_version `pyronear-yolov11s@a3db632`, imgsz 1024. expB is deployed
    ONLY via env override. The actually-running service (PID 3411031, up
    since Sep 10) has SMOKE_WEIGHTS_PATH=models/expB_best.pt,
    SMOKE_MODEL_VERSION=expB-mix-y11s-1280@seg03, SMOKE_IMGSZ=1280, and
    /healthz confirms expB@1280. So the deployment claim is TRUE for the
    running instance, but a bare `uvicorn src.serve.app:app` (no env) would
    serve the contaminated baseline at 1024. Reports never state this footgun.

D2. FINAL_REPORT_V3 Track A pAUCs are from an ad-hoc integration and do not
    match the canonical recomputation (scripts/v3/pauc.py on
    matched_fp_{val,test}.json):
    - val rfdetrL quoted 0.496 recall / 0.398 tiny; artifact 0.515 / 0.415.
    - test quoted rfdetrL/expE/expB 0.501/0.502/0.508 recall and
      0.382/0.413/0.394 tiny; artifact 0.505/0.526/0.521 and 0.385/0.435/0.405.
    pauc.py's docstring discloses "<=0.02" deltas, but expE test recall delta
    is 0.024, and the expE-vs-expB TEST ORDERING FLIPS (ad-hoc expB>expE;
    canonical expE>expB). Same report quotes expE test tiny pAUC as both
    0.413 (Track A table) and 0.435 (Track B/C paragraph). Headline verdict
    (RF-DETR does not beat YOLO at matched FP) survives either convention.

D3. FINAL_REPORT_V2 expE row cites reports/eval_expE_val.json ("*expE val
    numbers from reports/eval_expE_val.json") but the quoted values do not
    appear in that file: claimed box recall 0.746 / hard-neg FP 0.418 /
    tiny ~0.66 at conf 0.2; the file's conf-0.2 row is 0.7687 / 0.4335 /
    0.6834 (full 7,335-img val). Values were presumably computed on the
    2,000-image VLM-comparison subset, but no artifact containing
    0.746/0.418 was found. Citation is wrong or the subset artifact is
    missing.

D4. Pre-smoke frame alert rates are only approximately reproducible.
    Counting any non-"none" per-frame state in the replay JSONs gives
    v1: expA 0.116->0.508, expB 0.077->0.568 (report: 0.103->0.448/0.529);
    v2 lockbox: 0.052 full / 0.381 tiled (report: 0.044 / 0.376).
    Direction and magnitude corroborated; the exact rate definition used in
    the reports is not documented in any artifact.

D5. Minor: FINAL_REPORT_V3 quotes expE val recall pAUC as 0.521 (Track A)
    and 0.520 (Track B/C) for the same quantity; canonical script gives
    0.520 (matched_fp_val.json / _trackBC.json agree).

## 3. Artifacts missing or unverifiable (this pass)

- v2 VLM headline table (9B base/SFT/GRPO, 27B rows): only raw
  models/v2/evals/*/preds.jsonl exist; no summary JSON. Not independently
  recomputed (requires re-running the matching protocol). Partially
  corroborated: hard-neg FP 0.330 (SFT) and 0.413 (GRPO) appear in
  reports/v2/cascade_analysis*.txt.
- GPU-hour ledgers (v2 ~105 node-h total; v3 ~25.5 node-h) are labeled
  "approx" — they are ESTIMATES, not scheduler-derived accounting. Individual
  job wall-times in STATUS.md (e.g. rfdetr 5h34m, job
  tianrgua-rfdetrl1280e48a1-gs9y) are scheduler/job-derived; no aggregate
  scheduler accounting artifact exists in the repo.
- v1 live-serving spot check ("0.67 provisional, ~180 ms/img CPU") and v3 GPU
  batch-1 latencies (31.4 vs 11.2 ms) not re-measured; CPU latency artifacts
  verified instead.
- 27B/35B/122B blocked-run evidence (models/v2/runs/prof_*) present but only
  existence-checked.

Bottom line: every headline number that has a JSON artifact reproduces
exactly, lockbox/split integrity holds, and the two protocol-discipline
findings (C1 test collapse, B1 lockbox coverage) are artifact-backed. The
open items are the config.py default (D1), the Track A ad-hoc pAUC quotes
(D2), and the v2 expE citation (D3).
