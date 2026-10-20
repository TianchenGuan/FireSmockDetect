#!/bin/bash
# One teacher-pool job end-to-end: registry reserve (L40S, track teacher; retry
# up to 3x 10 min apart on exit 3/4/5) -> submit+wait -> registry release.
# usage: scripts/v5b/teacher_job.sh <restag> "<note>" -- <submit_teacher_infer.py args>
cd /local/home/tianrgua/cvproject && source .venv/bin/activate
RES="res-$1"; shift; NOTE="$1"; shift; [ "$1" = "--" ] && shift
log(){ echo "[teacher_job $(date -u +%FT%TZ) $RES] $*"; }
for attempt in 1 2 3 4; do
  bash cluster/v5b/registry.sh reserve --job "$RES" --nodes 1 --gpu-type L40S --track teacher --note "$NOTE"; rc=$?
  [ $rc = 0 ] && break
  case $rc in 3|4|5) [ $attempt = 4 ] && { log "DEFERRED after 3 retries (registry rc=$rc)"; exit 3; }
                     log "registry denied rc=$rc; retry in 10 min"; sleep 600;;
             *) log "registry error rc=$rc"; exit $rc;; esac
done
log "reserved; submitting: $*"
python cluster/v5b/submit_teacher_infer.py "$@"; rc=$?
bash cluster/v5b/registry.sh release --job "$RES" --outcome "submit_rc=$rc"
log "done rc=$rc"; exit $rc
