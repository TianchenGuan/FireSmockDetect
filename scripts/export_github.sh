#!/bin/bash
# Build a sanitized export tree of the project for the public GitHub repo.
# Includes code + reports + docs; excludes datasets, weights, build contexts,
# state/ledgers, generated job yamls (presigned URLs), private reports.
# Scrubs AWS account IDs, role names, internal provider/host names.
# Usage: bash scripts/export_github.sh [EXPORT_DIR]
set -euo pipefail
SRC=/local/home/tianrgua/cvproject
EXP=${1:-/local/home/tianrgua/cvproject_github}
mkdir -p "$EXP"

rsync -a --delete --prune-empty-dirs \
  --exclude '.git' \
  --include 'README.md' --include 'STATUS.md' --include 'SANITIZATION.md' \
  --include '/src/***' \
  --include '/scripts/***' \
  --include '/cluster/' --include '/cluster/**/' \
  --include '/cluster/**/*.py' --include '/cluster/**/*.sh' \
  --include '/cluster/**/Dockerfile*' --include '/cluster/**/*.md' \
  --include '/cluster/**/BUDGET*.json' \
  --include '/reports/' --include '/reports/**/' \
  --include '/reports/**/*.md' --include '/reports/**/*.csv' \
  --include '/reports/**/*.json' --include '/reports/**/*.png' \
  --include '/reports/**/*.jpg' --include '/reports/**/*.txt' \
  --include '/data/' --include '/data/**/' \
  --include '/data/**/*.md' \
  --include '/data/v6/manifests/*.json' --include '/data/v6/yamls/*.yaml' \
  --include '/data/v5b/manifests/*.json' --include '/data/v4/manifests/*.json' \
  --include '/models/' --include '/models/**/' \
  --include '/models/**/*.md' --include '/models/**/MODEL_REGISTRY*.json' \
  --exclude '*' \
  "$SRC/" "$EXP/"

# hard exclusions that the include rules above might have caught
rm -rf "$EXP/cluster/generated" "$EXP/cluster/v5b/build" "$EXP/cluster/v5b/build_qwen" \
       "$EXP/cluster/v6/build_glm_b200" "$EXP/reports/v6/private" \
       "$EXP"/cluster/*/__pycache__ "$EXP"/cluster/__pycache__ 2>/dev/null || true
find "$EXP" -name '.uploaded_*' -delete
find "$EXP" -name 'registry.json' -delete
find "$EXP" -name 'bedrock_ledger.json' -delete
find "$EXP" -name 'describe.txt' -delete   # kraken job dumps carry presigned env values
find "$EXP" -name 'submit.txt' -delete
find "$EXP" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
# size cap 8MB (weights/parquet never included, but guard anyway)
find "$EXP" -type f -size +8M -delete

# ---- scrub internal identifiers in all text files
mapfile -t FILES < <(grep -rIl . "$EXP" 2>/dev/null)
sed -i \
  -e 's/000000000001/000000000001/g' \
  -e 's/000000000002/000000000002/g' \
  -e 's/000000000003/000000000003/g' \
  -e 's/000000000004/000000000004/g' \
  -e 's/000000000005/000000000005/g' \
  -e 's/000000000006/000000000006/g' \
  -e 's/000000000007/000000000007/g' \
  -e 's/INTERNAL-ADMIN-ROLE/INTERNAL-ADMIN-ROLE/g' \
  -e 's/--provider[", ]*conduit/--provider INTERNAL-PROVIDER/g' \
  -e 's/"INTERNAL-PROVIDER"/"INTERNAL-PROVIDER"/g' \
  -e 's###g' \
  -e 's/dev-dsk-[A-Za-z0-9.-]*amazon\.com/devbox/g' \
  "${FILES[@]}"

# ---- secret / presigned scan (must be clean)
echo "--- secret scan ---"
BAD=0
for pat in 'X-Amz-' 'AKIA[0-9A-Z]\{16\}' 'ASIA[0-9A-Z]\{16\}' \
           'aws_secret_access_key *[=:] *["'"'"']\?[A-Za-z0-9/+]\{30\}' 'ghp_[A-Za-z0-9]\{20\}' \
           'hf_[A-Za-z0-9]\{30\}' 'WANDB_API_KEY=[A-Za-z0-9]'; do
  n=$( (grep -rIl "$pat" "$EXP" --exclude=make_report_tables_v5b.py \
      --exclude=export_github.sh 2>/dev/null || true) | wc -l)
  if [ "$n" -gt 0 ]; then
    echo "HIT ($n files): $pat"; grep -rIl "$pat" "$EXP" | head -3; BAD=1
  fi
done
[ $BAD -eq 0 ] && echo "clean: no presigned URLs / keys / tokens"
echo "--- residual account-id scan ---"
grep -rIoh '[0-9]\{12\}' "$EXP" --include='*.py' --include='*.sh' --include='*.md' 2>/dev/null \
  | sort | uniq -c | sort -rn | head -8 || true
echo "--- export stats ---"
du -sh --apparent-size "$EXP"; find "$EXP" -type f | wc -l
exit $BAD
