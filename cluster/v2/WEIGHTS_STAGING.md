# v2 weight staging — Qwen3.5 family (2026-09-11)

All three models staged devbox -> S3 -> green g6e pod /scratch and verified
byte-exact at every hop. Tooling: `cluster/v2/stage_weights_job.py`
(manifest of relpath|size|presigned-GET per file -> pod curls with xargs -P8
-> per-file size verification -> status.json via presigned PUT; job exits 0
only if every size matches). Revisions pinned in `models/v2/PINS.md`.

## Staged models

| Model | HF repo @ revision | Files / bytes | S3 | /scratch (pod) | Job | Result |
|---|---|---|---|---|---|---|
| qwen3.5-9b | Qwen/Qwen3.5-9B @ c202236235762e1c871ad0ccb60c8ee5ba337b9a | 16 / 19,329,393,661 | s3://smoke-detect-data-000000000001-use1/weights/qwen3.5-9b/ | /scratch/scratch/tianrgua-sandbox/models/qwen3.5-9b/ | tianrgua-wsqwen359b-anwv | SUCCEEDED, ok=true, 0 mismatches |
| qwen3.5-27b | Qwen/Qwen3.5-27B @ fc05daec18b0a78c049392ed2e771dde82bdf654 | 24 / 55,586,167,982 | s3://smoke-detect-data-000000000001-use1/weights/qwen3.5-27b/ | /scratch/scratch/tianrgua-sandbox/models/qwen3.5-27b/ | tianrgua-wsqwen3527b-2799 | SUCCEEDED, ok=true, 0 mismatches |
| qwen3.5-35b-a3b | Qwen/Qwen3.5-35B-A3B @ 59d61f3ce65a6d9863b86d2e96597125219dc754 | 27 / 71,927,084,782 | s3://smoke-detect-data-000000000001-use1/weights/qwen3.5-35b-a3b/ | /scratch/scratch/tianrgua-sandbox/models/qwen3.5-35b-a3b/ | tianrgua-wsqwen3535ba3b-ni53 | SUCCEEDED, ok=true, 0 mismatches |

122B-A10B (dc4d3484, 250GB) NOT staged this round (out of scope).

## Verification chain (all three models)
1. Local vs HF: every file byte-identical to HfApi files_metadata sizes at
   the pinned revision (0 mismatches).
2. S3 vs local: `aws s3 sync` + pre-submit cross-check in
   stage_weights_job.py (raises on any size difference; `.cache/*` excluded).
3. /scratch vs S3: pod stat -c%s of every downloaded file vs manifest sizes;
   status.json uploaded via presigned PUT. Downloaded fresh from
   s3://.../v2/staging/<name>/status.json afterwards: ok=true, mismatches=0,
   sizes entries == file count, bytes_actual == bytes_expected == local sum.
   Copies in cluster/v2/generated/status_*.json.

Pod transfer times (RUNNING -> SUCCEEDED, 8-way curl, same-region):
9B ~2.3 min, 27B ~3 min, 35B ~3 min. Far below the ~55 min presigned-URL
lifetime; single job per model suffices up to ~150 GB (use --file-list to
split beyond that).

## Job history / gotchas
- tianrgua-wsqwen359b-h9e8 (first 9B attempt) SUCCEEDED but was a FALSE
  POSITIVE: manifest uploaded without trailing newline -> the pod-side
  `while read` loop silently dropped the LAST manifest line (vocab.json
  never downloaded, never verified, ok=true anyway). Fixed three ways in
  stage_weights_job.py: trailing newline on the manifest, `read ... ||
  [ -n "$P" ]` in both pod loops, and devbox-side re-verification that
  status.json covers exactly the listed file count/bytes. 9B re-staged
  (job -anwv) with all 16 files; -h9e8's partial copy simply overwritten.
- Presigned URLs die with the ada creds (~55 min). stage_weights_job.py
  mints fresh creds itself; run it right when you want the job submitted,
  and it stops+flags jobs still PENDING near expiry.
- Job command uses NO single quotes (kraken YAML constraint) and only
  presigned URLs in env vars (project-readable; time-limited, no static
  creds).

## Reproduce / re-stage
```
source .venv/bin/activate
python cluster/v2/stage_weights_job.py --name qwen3.5-9b        # poll + verify
python cluster/v2/stage_weights_job.py --name qwen3.5-27b --submit-only
```
Logs: logs/v2/wstage_*.log, logs/v2/dl_*.log. Generated YAMLs:
cluster/v2/generated/wstage_*.yaml.
