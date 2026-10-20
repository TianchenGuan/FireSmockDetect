#!/usr/bin/env bash
# BAMBI raw flight downloader: resume (curl -C -), md5 verification, JSONL log per flight.
# Usage: bambi_download.sh <flight_id> <url> <md5> <bytes>   (driven by xargs -P from _download_list.txt)
set -u
ROOT=/local/home/tianrgua/cvproject/data/v5/raw/bambi_raw
UA="cvproject-v5-acquisition (research prototype; contact via repo owner)"
fid="$1"; url="$2"; md5="$3"; bytes="$4"
d="$ROOT/$fid"; mkdir -p "$d"; f="$d/$fid.zip"; log="$d/_download_log.jsonl"
for attempt in 1 2 3 4 5 6; do
  have=0; [ -f "$f" ] && have=$(stat -c %s "$f")
  if [ "$have" -ge "$bytes" ]; then break; fi
  echo "$(date -u +%FT%TZ) $fid attempt $attempt resume_from=$have"
  curl -sS -L -C - --retry 5 --retry-delay 15 --retry-all-errors -m 14400 -A "$UA" -o "$f" "$url" || sleep 20
done
sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
m=$(md5sum "$f" | cut -d' ' -f1)
st="MD5_OK"; [ "$m" = "$md5" ] || st="MD5_MISMATCH"; [ "$sz" -eq "$bytes" ] || st="${st}_SIZE_MISMATCH"
echo "{\"flight_id\":\"$fid\",\"file\":\"$fid.zip\",\"bytes\":$sz,\"expected_bytes\":$bytes,\"md5\":\"$m\",\"expected_md5\":\"$md5\",\"status\":\"$st\",\"url\":\"$url\",\"fetched_utc\":\"$(date -u +%FT%TZ)\"}" >> "$log"
echo "$(date -u +%FT%TZ) $fid DONE $st bytes=$sz"
