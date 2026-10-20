#!/usr/bin/env bash
# Poll for md5-verified BAMBI zips and extract 1-fps frames (<=4 flights concurrently). Deletes each zip after extraction.
set -u
ROOT=/local/home/tianrgua/cvproject; RAW=$ROOT/data/v5/raw/bambi_raw
cd $ROOT
want=$(awk '{print $1}' $RAW/_download_list.txt)
declare -A done
while :; do
  running=$(jobs -rp | wc -l)
  all_done=1
  for fid in $want; do
    [ -n "${done[$fid]:-}" ] && continue
    all_done=0
    log=$RAW/$fid/_download_log.jsonl
    if [ -f "$log" ] && grep -q '"status":"MD5_OK"' "$log" && [ -f "$RAW/$fid/$fid.zip" ] && [ ! -f "$RAW/$fid/_extract_log.json" ]; then
      if [ "$running" -lt 4 ]; then
        n=${fid#flight_}
        echo "$(date -u +%FT%TZ) START extract $fid"
        ( python3 scripts/v5/bambi_extract_frames.py --zip $RAW/$fid/$fid.zip --flight $n --out $RAW/$fid > $RAW/$fid/_extract_stdout.log 2>&1; echo "$(date -u +%FT%TZ) END extract $fid rc=$?" ) &
        done[$fid]=1; running=$((running+1))
      fi
    elif [ -f "$log" ] && ! grep -q '"status":"MD5_OK"' "$log"; then
      echo "$(date -u +%FT%TZ) SKIP $fid (download status not MD5_OK): $(tail -1 $log)"; done[$fid]=1
    elif [ -f "$RAW/$fid/_extract_log.json" ]; then done[$fid]=1; fi
  done
  [ "$all_done" = 1 ] && break
  sleep 30
done
wait
echo "$(date -u +%FT%TZ) EXTRACT_ALL_DONE"
