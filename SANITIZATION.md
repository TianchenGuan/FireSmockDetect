# Sanitization notes for the public export

This repository is an export of a working research tree that ran on private
cloud infrastructure. Before publication, `scripts/export_github.sh`:

**Excluded entirely**
- datasets and all image/video files outside `reports/` figures
- model weights (`*.pt`, `*.safetensors`, `*.onnx`, `*.tflite`), eval
  prediction dumps (`*.parquet`, `*.jsonl.gz`), archives
- generated per-job configs (`cluster/generated/`) and scheduler job dumps
  (`describe.txt`/`submit.txt`) — they embed short-lived presigned URLs
- docker build contexts with baked weights (`cluster/*/build*`)
- live state: resource registry, spend ledgers, upload markers
- `reports/v6/private/` (cloud routing/pricing notes)

**Replaced with placeholders (in all text files)**
- 12-digit cloud account IDs → `000000000001…7` (one placeholder per
  distinct account; bucket/registry names embedding them are scrubbed by the
  same substitution)
- an internal IAM role name → `INTERNAL-ADMIN-ROLE`
- an internal credential-provider name → `INTERNAL-PROVIDER`
- developer-host names → `devbox`; internal tool paths stripped to bare names

**Verified clean**
- zero presigned URLs (`X-Amz-*`), AWS keys (`AKIA/ASIA…`), GitHub/HF/W&B
  tokens anywhere in the export (the only `X-Amz` matches are scrubbing
  regexes in report-generation code)

The cluster orchestration scripts are therefore **documentation of method**,
not runnable outside the original environment without supplying your own
storage buckets, container registry and scheduler.
