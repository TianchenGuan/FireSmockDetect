# Pinned model revisions (v2 weight staging)

Pinned 2026-09-11 via HfApi.model_info(...).sha. All Apache-2.0.

| Local dir (models/v2/) | HF repo | Revision (sha) | Size | Files |
|---|---|---|---|---|
| qwen3.5-9b | Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a | 19.3 GB | 16 |
| qwen3.5-27b | Qwen/Qwen3.5-27B | fc05daec18b0a78c049392ed2e771dde82bdf654 | 55.6 GB | 24 |
| qwen3.5-35b-a3b | Qwen/Qwen3.5-35B-A3B | 59d61f3ce65a6d9863b86d2e96597125219dc754 | 71.9 GB | 27 |

122B-A10B (dc4d3484) intentionally NOT staged this round.

S3 destination: s3://smoke-detect-data-000000000001-use1/weights/<local dir name>/
Pod destination: /scratch/scratch/tianrgua-sandbox/models/<local dir name>/
