#!/usr/bin/env bash
# V5 Day-1 proactive acquisition: download ONLY explicitly-licensed, redistribution-permitted,
# <20 GB items. Logs size+sha256 per file. Run: nohup bash scripts/v5/acq_download_v5.sh > data/v5/raw/_acq_download.log 2>&1 &
set -u
ROOT=/local/home/tianrgua/cvproject/data/v5/raw
UA="cvproject-v5-acquisition (research prototype; contact via repo owner)"
dl(){ # dir url outname expected_md5(optional)
  local dir="$1" url="$2" out="$3" md5="${4:-}"
  mkdir -p "$ROOT/$dir"; local f="$ROOT/$dir/$out"
  if [ -s "$f" ]; then echo "SKIP exists $f"; else
    echo "GET $(date -u +%FT%TZ) $url -> $f"
    curl -sL --retry 3 --retry-delay 5 -m 7200 -A "$UA" -o "$f" "$url" || { echo "FAIL $url"; return 1; }
  fi
  local sz; sz=$(stat -c %s "$f"); local sha; sha=$(sha256sum "$f" | cut -d' ' -f1); local m=""
  if [ -n "$md5" ]; then m=$(md5sum "$f" | cut -d' ' -f1); [ "$m" = "$md5" ] && echo "MD5 OK $out" || echo "MD5 MISMATCH $out got $m want $md5"; fi
  echo "{\"dir\":\"$dir\",\"file\":\"$out\",\"bytes\":$sz,\"sha256\":\"$sha\",\"md5\":\"$m\",\"url\":\"$url\",\"fetched_utc\":\"$(date -u +%FT%TZ)\"}" >> "$ROOT/$dir/_download_log.jsonl"
}
# A. Zenodo 10.5281/zenodo.20506617 — Forest Smoke Video Dataset (OTE + NCSR Demokritos), CC BY 4.0
Z=https://zenodo.org/api/records/20506617/files
dl zenodo_forest_smoke_video $Z/6g-versus-smoke-1.mp4/content 6g-versus-smoke-1.mp4 84631c879b5e8001a0ab8ad4f4b6cedb
dl zenodo_forest_smoke_video $Z/6g-versus-smoke-2.mp4/content 6g-versus-smoke-2.mp4 fb308636cadf1147f2b80f940a9b7aae
dl zenodo_forest_smoke_video $Z/6g-versus-smoke-3.mp4/content 6g-versus-smoke-3.mp4 69e3a7425f097357c3f66ce401e1b0e2
dl zenodo_forest_smoke_video $Z/6g-versus-smoke-4.mp4/content 6g-versus-smoke-4.mp4 ddf990fd5406094a4aba49994fb38c68
dl zenodo_forest_smoke_video $Z/6g-versus-smoke-5.mp4/content 6g-versus-smoke-5.mp4 876daaf0dc852c65e1db82adb4c297aa
dl zenodo_forest_smoke_video $Z/6g-versus-smoke-6.mp4/content 6g-versus-smoke-6.mp4 8692604d3413de2096d62d4355ac6bfc
dl zenodo_forest_smoke_video $Z/6g-versus-smoke-7.mp4/content 6g-versus-smoke-7.mp4 e130ff09a684d1c0df9d2f80e25e58ca
curl -sL -A "$UA" -o "$ROOT/zenodo_forest_smoke_video/_zenodo_record.json" https://zenodo.org/api/records/20506617
# B. Wikimedia Commons copies of US-government (public domain) prescribed-burn videos (+1 BLM CC BY 2.0)
C=https://upload.wikimedia.org/wikipedia/commons
dl commons_usgov_burn_videos "$C/e/ef/NASA_FireSense_%28Fishlake_National_Forest%29_%28SVS14559_-_NASA_FireSense_DRONES_BROLL%29.webm" NASA_FireSense_SVS14559_DRONES_BROLL.webm
dl commons_usgov_burn_videos "$C/0/0a/NASA_FireSense_%28US_Forests_Services%E2%80%99s_FASMEE_-_Fishlake_National_Forest%2C_Utah%29_%28SVS14559_-_Prescribed_Burn_Area_2_Day2_BROLL%29.webm" NASA_FireSense_SVS14559_PrescribedBurn_Area2_Day2_BROLL.webm
dl commons_usgov_burn_videos "$C/8/8b/Prescribed_Burn_in_North_Derby_Gulch_Area%2C_CGNF_%2853229395208%29.webm" USFS_R1_Prescribed_Burn_North_Derby_Gulch_CGNF_53229395208.webm
dl commons_usgov_burn_videos "$C/3/32/Time-lapse-_Using_drip_torch_for_prescribed_burn_%2837896304436%29.webm" BLM_ORWA_Timelapse_drip_torch_prescribed_burn_37896304436.webm
dl commons_usgov_burn_videos "$C/5/5f/Loveless_Prescribed_Burn.webm" BLM_ORWA_Loveless_Prescribed_Burn_CCBY2.webm
cp /tmp/v5acq/commons_meta.json "$ROOT/commons_usgov_burn_videos/_commons_imageinfo.json" 2>/dev/null
# C. BAMBI raw (CC BY 4.0) — one flight as aerial no-fire forest sample: flight_98 (2023-04-26, Purkersdorf, Lower Austria, Mavic 3 Thermal)
dl bambi_raw_sample https://zenodo.org/api/records/19177615/files/flight_98.zip/content flight_98.zip 6b7a1feaa49e4e93c5e4f419474ff24f
curl -sL -A "$UA" -o "$ROOT/bambi_raw_sample/_zenodo_record_part47.json" https://zenodo.org/api/records/19177615
cp /tmp/v5acq/bambi_m98.json "$ROOT/bambi_raw_sample/98_metadata.json"; cp /tmp/v5acq/bambi_readme.md "$ROOT/bambi_raw_sample/_github_README.md"; cp /tmp/v5acq/bambi_versions.md "$ROOT/bambi_raw_sample/_github_dataset-versions.md"
# D. MONIMET phenocam MC124 (Lammi mixed stand, landscape view), year 2020 (CC BY 4.0)
dl monimet_mc124 https://zenodo.org/api/records/7500295/files/Monimet_MC124_2020_2020.zip/content Monimet_MC124_2020_2020.zip a686ef6cab4b820571e135fc1b721fbd
curl -sL -A "$UA" -o "$ROOT/monimet_mc124/_zenodo_record.json" https://zenodo.org/api/records/7500295
curl -sL -A "$UA" -o "$ROOT/monimet_mc124/_zenodo_record_datasheet_777952.json" https://zenodo.org/api/records/777952
echo "ALL DONE $(date -u +%FT%TZ)"
