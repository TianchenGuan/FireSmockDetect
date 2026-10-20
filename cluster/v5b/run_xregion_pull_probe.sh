#!/bin/bash
# V5b job 2 runner: submit cluster/v5b/b200_xregion_pull_probe.yaml to obsidian-pu, poll,
# read /tmp/probe_status + /tmp/sts.txt via `kraken jobs connect -c`, ack, record describe.
# Usage: bash cluster/v5b/run_xregion_pull_probe.sh <outdir> [pending_abort_s]
set -u
OUT=$1; ABORT=${2:-600}; mkdir -p "$OUT"
K=kraken; P=obsidian-pu
ts() { date -u +%H:%M:%SZ; }
SUB=$($K -p $P jobs create -i /local/home/tianrgua/cvproject/cluster/v5b/b200_xregion_pull_probe.yaml 2>&1); echo "$SUB" > $OUT/submit.txt
echo "[$(ts)] $(echo "$SUB" | sed -e 's#https://[^ ]*#<URL>#g' | head -3)"
JOB=$(echo "$SUB" | tr '",.:;' '     ' | tr ' ' '\n' | grep -m1 '^tianrgua-v5bb200xr')
[ -z "$JOB" ] && { echo "SUBMIT FAILED"; exit 2; }
echo "[$(ts)] job=$JOB"; T0=$(date +%s); LAST=""; RUN0=""
while :; do
  ST=$($K -p $P jobs list -r tianrgua 2>/dev/null | awk -v j="$JOB" '$1==j{print $2}')
  [ "$ST" != "$LAST" ] && { echo "[$(ts)] state $ST"; LAST=$ST; }
  case "${ST%%:*}" in
    SUCCEEDED|FAILED|STOPPED) break;;
    RUNNING)
      [ -z "$RUN0" ] && RUN0=$(date +%s)
      R=$(timeout 90 $K -p $P jobs connect -j $JOB -c "cat /tmp/probe_status; echo; cat /tmp/sts.txt" 2>/dev/null | grep -v kubeconfig)
      echo "[$(ts)] connect: $(echo "$R" | tr -d '\n' | cut -c1-300)"
      if echo "$R" | grep -q retain; then echo "$R" > $OUT/pod_status.txt; timeout 60 $K -p $P jobs connect -j $JOB -c "touch /tmp/ack" >/dev/null 2>&1; fi;;
    *) [ $(( $(date +%s)-T0 )) -gt $ABORT ] && { echo "[$(ts)] not RUNNING after ${ABORT}s -> stop (image pull likely failing)"; $K -p $P jobs describe --job-name $JOB > $OUT/describe_prestop.txt 2>&1; $K -p $P jobs stop --job-name $JOB; };;
  esac
  sleep 20
done
$K -p $P jobs describe --job-name $JOB > $OUT/describe.txt 2>&1
echo "[$(ts)] terminal $LAST; running_s=$([ -n "$RUN0" ] && echo $(( $(date +%s)-RUN0 )) || echo 0)"
grep -i "exit\|reason\|message\|image" $OUT/describe.txt | head -8
