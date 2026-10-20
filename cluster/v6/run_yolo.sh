#!/bin/bash
# v6 training launch wrapper: registry reserve -> orchestrate_v6 -> release.
# Usage: run_yolo.sh <reg-job> <nodes> <track> "<note>" -- <orchestrate_v6 args...>
set -u
cd /local/home/tianrgua/cvproject
JOB=$1; NODES=$2; TRACK=$3; NOTE=$4; shift 4
[ "${1:-}" = "--" ] && shift
SMOKE_REGISTRY_DIR=cluster/v6 .venv/bin/python cluster/v4/registry.py reserve \
  --job "$JOB" --nodes "$NODES" --track "$TRACK" --note "$NOTE" || exit 3
SMOKE_REGISTRY_DIR=cluster/v6 .venv/bin/python cluster/v6/orchestrate_v6.py "$@"
rc=$?
out=FAILED; [ $rc -eq 0 ] && out=SUCCEEDED
SMOKE_REGISTRY_DIR=cluster/v6 .venv/bin/python cluster/v4/registry.py release \
  --job "$JOB" --outcome $out
exit $rc
