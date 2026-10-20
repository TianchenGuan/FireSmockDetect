#!/bin/bash
# Track B/C: the moment gated access lands, stage all three DINOv3 backbones
# devbox -> S3 -> green g6e /scratch, so training can start within ~1h.
#
#   bash scripts/v3/stage_dinov3.sh            # all three
#   bash scripts/v3/stage_dinov3.sh vitl16     # just one (vitl16|vith16plus|vit7b16)
#
# Reuses the byte-verified v2 staging pipeline (cluster/v2/stage_weights_job.py:
# manifest of relpath|size|presigned-GET -> pod 8-way curl -> per-file size
# verification -> status.json). Sizes: L ~1.7GB, H+ ~3.2GB, 7B ~27GB (fp32
# safetensors) — well under the ~150GB single-job staging limit.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

bash scripts/v3/dinov3_gate_check.sh || { echo "still gated — aborting"; exit 1; }

declare -A REPO=(
  [vitl16]=facebook/dinov3-vitl16-pretrain-lvd1689m
  [vith16plus]=facebook/dinov3-vith16plus-pretrain-lvd1689m
  [vit7b16]=facebook/dinov3-vit7b16-pretrain-lvd1689m
)
# revisions observed via API metadata on 2026-09-13 (metadata is visible even
# while files are gated); download hard-pins these
declare -A REV=(
  [vitl16]=ea8dc2863c51be0a264bab82070e3e8836b02d51
  [vith16plus]=c807c9eeea853df70aec4069e6f56b28ddc82acc
  [vit7b16]=b80367753773648a6793235ab9c65cdbb029506f
)
WANT=("${@:-vitl16 vith16plus vit7b16}")
[ $# -eq 0 ] && WANT=(vitl16 vith16plus vit7b16)

BUCKET=smoke-detect-data-000000000001-use1
ADA=ada

mkdir -p models/v3/hf models/v2
for name in "${WANT[@]}"; do
  repo="${REPO[$name]}"
  rev="${REV[$name]}"
  dst="models/v3/hf/dinov3-$name"
  echo "== downloading $repo @ $rev -> $dst"
  python - "$repo" "$rev" "$dst" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, rev, dst = sys.argv[1], sys.argv[2], sys.argv[3]
p = snapshot_download(repo, revision=rev, local_dir=dst)
open(dst + "/REVISION_PIN.txt", "w").write(rev + "\n")
print("downloaded to", p)
EOF
  # stage_weights_job.py conventions: S3 prefix weights/<name>/, local
  # cross-check dir models/v2/<name> -> symlink our v3 download there.
  ln -sfn "../v3/hf/dinov3-$name" "models/v2/dinov3-$name"
  echo "== syncing $dst -> s3://$BUCKET/weights/dinov3-$name/"
  CREDS=$($ADA credentials print --account 000000000001 --provider INTERNAL-PROVIDER \
      --role INTERNAL-ADMIN-ROLE --format json)
  export AWS_ACCESS_KEY_ID=$(echo "$CREDS" | python -c "import sys,json;print(json.load(sys.stdin)['AccessKeyId'])")
  export AWS_SECRET_ACCESS_KEY=$(echo "$CREDS" | python -c "import sys,json;print(json.load(sys.stdin)['SecretAccessKey'])")
  export AWS_SESSION_TOKEN=$(echo "$CREDS" | python -c "import sys,json;print(json.load(sys.stdin)['SessionToken'])")
  aws s3 sync "$dst" "s3://$BUCKET/weights/dinov3-$name/" \
    --exclude ".cache/*" --region us-east-1
  echo "== submitting /scratch staging job (S3 -> pod, byte-verified)"
  python cluster/v2/stage_weights_job.py --name "dinov3-$name"
done
echo "ALL STAGED to /scratch/scratch/tianrgua-sandbox/models/dinov3-<name>/."
echo "Next: bash cluster/v3/launch/B1_vitl16.sh (see cluster/v3/launch/README.md)"
