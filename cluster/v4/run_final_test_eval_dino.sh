#!/bin/bash
# ================================================================================
# ITERATION-4 ONE-SHOT FINAL TEST EVAL -- DINOv3 track. RUN EXACTLY ONCE.
# Test predictions for the three pre-registered checkpoints
#   v4B6_lora_Hplus  (pre-registered pick)   v4B1_frozen_L (control)
#   v4B4_lora_L      (post-hoc efficiency candidate -- labelled post-hoc)
# on sealed sets sdis_test (4,124) / boreal_test (1,764) / forest_test (2,871),
# then offline sweep (src.eval.eval_detection --split test), matched-FP
# comparison, and recall at the PRE-REGISTERED val thresholds
# (reports/v4/test/dino_test_thresholds.json). GT is read locally only.
#   bash cluster/v4/run_final_test_eval_dino.sh --dry-run   # validate + print YAMLs, NO submission
#   bash cluster/v4/run_final_test_eval_dino.sh             # real run (refuses to run twice)
#   --skip-stage  assume v4_evaltest already on /scratch     --check-sha  sha256 the 6.75GB tar
# Route: cluster/v3/submit_dino_eval.py (fixed LoRA path, multi-root --root-map),
# one job per checkpoint, registry reserve/release per job. Never opens test images.
# ================================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
source .venv/bin/activate
DRY=0; SKIP_STAGE=0; CHECK_SHA=0
for a in "$@"; do case "$a" in
  --dry-run) DRY=1;; --skip-stage) SKIP_STAGE=1;; --check-sha) CHECK_SHA=1;;
  *) echo "unknown arg $a"; exit 2;; esac; done
GUARD=reports/v4/test/.DINO_EVALUATED_ONCE
OUTR=reports/v4/test
SCR=/scratch/scratch/tianrgua-sandbox
TAR=data/v4/stage/v4_evaltest.tar
TAR_SHA_PREFIX=f7cc4ad7
THR=$OUTR/dino_test_thresholds.json
RUNS=(v4B6_lora_Hplus v4B1_frozen_L v4B4_lora_L)
declare -A BB=( [v4B6_lora_Hplus]=$SCR/models/dinov3-vith16plus [v4B1_frozen_L]=$SCR/models/dinov3-vitl16 [v4B4_lora_L]=$SCR/models/dinov3-vitl16 )
declare -A EXTRA=( [v4B6_lora_Hplus]="--lora 16" [v4B1_frozen_L]="" [v4B4_lora_L]="--lora 16" )
declare -A LISTS=( [sdis_test]=data/yolo/lists/local/sdis_test.txt [boreal_test]=data/v4/eval/boreal_test/lists/local/test.txt [forest_test]=data/v4/eval/forest_test/lists/local/test.txt )
declare -A EXPECT=( [sdis_test]=4124 [boreal_test]=1764 [forest_test]=2871 )
declare -A YAMLS=( [sdis_test]=data/yolo/yamls/local/expB.yaml [boreal_test]=data/v4/eval/yamls/boreal_test.yaml [forest_test]=data/v4/eval/yamls/forest_test.yaml )
ROOTMAP="$REPO/data/yolo=$SCR/smoke_data_v2,$REPO/data/v4/eval=$SCR/smoke/v4data/v4_evaltest"
log(){ echo "[final-test-dino $(date -u +%FT%TZ)] $*"; }
fail(){ log "FATAL: $*"; exit 1; }

# ---- 0. refuse-to-run-twice --------------------------------------------------
if [ "$DRY" = 0 ] && [ -e "$GUARD" ]; then
  log "REFUSING: $GUARD exists -> the DINO test eval has already been run:"; cat "$GUARD"; exit 9
fi
mkdir -p "$OUTR"

# ---- 1. validate every local input (lists/labels/yamls) -- images never opened --
log "validating inputs (mode: $([ $DRY = 1 ] && echo DRY-RUN || echo REAL))"
for s in sdis_test boreal_test forest_test; do
  L=${LISTS[$s]}; [ -f "$L" ] || fail "list missing: $L"
  n=$(grep -c . "$L"); [ "$n" = "${EXPECT[$s]}" ] || fail "$s list has $n lines, expected ${EXPECT[$s]}"
  Y=${YAMLS[$s]}; [ -f "$Y" ] || fail "yaml missing: $Y"
  grep -q "^test:" "$Y" || fail "$Y has no test: key"
  if [ "$s" != sdis_test ]; then
    nl=$(ls data/v4/eval/$s/labels | wc -l); [ "$nl" = "$n" ] || fail "$s labels=$nl != list=$n"
    head -1 "$L" | grep -q "^$REPO/data/v4/eval/$s/images/" || fail "$s list paths not under data/v4/eval/$s/images"
    POD_ROOT="$SCR/smoke/v4data/v4_evaltest/$s/images"
  else
    head -1 "$L" | grep -q "^$REPO/data/yolo/" || fail "sdis_test list paths not under data/yolo"
    POD_ROOT="$SCR/smoke_data_v2/<rel>"
  fi
  log "  ok $s: $n images | local list $L | yaml $Y | pod images -> $POD_ROOT"
done
[ -f "$THR" ] || fail "thresholds json missing: $THR"
python - "$THR" <<'PY' || fail "thresholds json malformed"
import json,sys; d=json.load(open(sys.argv[1])); assert set(d["runs"])=={"v4B6_lora_Hplus","v4B1_frozen_L","v4B4_lora_L"}
for r,v in d["runs"].items(): assert "negfp_10pct" in v["sdis_val_source_only"] and "negfp_1pct" in v["boreal_val_target_adapted"]
print("  ok thresholds (sdis10%,boreal1%):", {r:(v["sdis_val_source_only"]["negfp_10pct"]["conf"], v["boreal_val_target_adapted"]["negfp_1pct"]["conf"]) for r,v in d["runs"].items()})
PY
for r in "${RUNS[@]}"; do
  ls models/v3/runs/$r/seg*/best.pt >/dev/null 2>&1 || fail "no local best.pt for $r (proxy for $SCR/smoke/runs_v3/$r/best.pt)"
  log "  ok ckpt $r -> $SCR/smoke/runs_v3/$r/best.pt (local copy $(ls models/v3/runs/$r/seg*/best.pt | tail -1)) infer-extra='${EXTRA[$r]}'"
done
for f in cluster/v3/submit_dino_eval.py cluster/v3/dino_eval_entry.sh cluster/v3/dino_infer.py cluster/v4/stage_evalval.py cluster/v4/registry.py scripts/v4/dino_test_at_thresholds.py scripts/v4/compare_matched_fp.py; do
  [ -f "$f" ] || fail "missing $f"; done
grep -q -- "--lora-merge" cluster/v3/dino_infer.py || fail "dino_infer.py lacks the v4 LoRA path"
grep -q -- "--dry-run" cluster/v3/submit_dino_eval.py || fail "submit_dino_eval.py lacks --dry-run"
[ -f "$TAR" ] || fail "sealed test tar missing: $TAR"
sz=$(stat -c %s "$TAR"); log "  ok tar $TAR ($sz bytes)"
[ "$sz" = 6747750400 ] || log "  WARNING: tar size $sz != 6747750400 recorded by the builder"
if [ "$CHECK_SHA" = 1 ]; then
  sha=$(sha256sum "$TAR" | cut -c1-64); [[ "$sha" == ${TAR_SHA_PREFIX}* ]] || fail "tar sha $sha does not start with $TAR_SHA_PREFIX"
  log "  ok tar sha256 $sha"
fi
tar -tf "$TAR" v4_evaltest/boreal_test/lists/pod/test.txt v4_evaltest/forest_test/lists/pod/test.txt >/dev/null 2>&1 \
  || fail "tar lacks v4_evaltest/{boreal_test,forest_test}/lists/pod/test.txt"
log "  ok tar layout (pod lists present; extracts to $SCR/smoke/v4data/v4_evaltest/)"
python cluster/v4/registry.py status | python -c "
import json,sys,datetime as dt; d=json.load(sys.stdin)
dl=dt.datetime.fromisoformat(d['deadline_utc'].replace('Z','+00:00')); now=dt.datetime.now(dt.timezone.utc)
print(f'  registry: active {d[\"active_nodes\"]}/{d[\"node_cap\"]} nodes, deadline {d[\"deadline_utc\"]} ({(dl-now).total_seconds()/3600:.1f} h left)')
sys.exit(0 if now<dl else 7)" || fail "past v4 GPU deadline"

# ---- 2. stage the sealed tar on /scratch (idempotent: pod .done marker) --------
STAGE_CMD=(python cluster/v4/stage_evalval.py --name v4_evaltest --tar "$TAR" --probe v4_evaltest/boreal_test/lists/pod/test.txt)
if [ "$SKIP_STAGE" = 1 ]; then
  log "staging skipped (--skip-stage): assuming $SCR/smoke/v4data/v4_evaltest is present (.v4_evaltest.done)"
elif [ "$DRY" = 1 ]; then
  log "[dry-run] would stage: ${STAGE_CMD[*]}  (s3 upload skipped if cluster/v4/.uploaded_v4_evaltest exists: $([ -f cluster/v4/.uploaded_v4_evaltest ] && echo yes || echo no); pod-side no-op if .v4_evaltest.done exists)"
else
  python cluster/v4/registry.py reserve --job res-stgevaltest-dino --nodes 1 --track dino --note "stage v4_evaltest (final test)" || fail "registry denied staging reservation"
  "${STAGE_CMD[@]}"; rc=$?
  python cluster/v4/registry.py release --job res-stgevaltest-dino --outcome "rc=$rc"
  [ $rc = 0 ] || fail "staging job failed rc=$rc"
  grep -q "boreal_test images: 1764" models/v4/stage/v4_evaltest/report.txt 2>/dev/null || fail "staging report does not show 1764 boreal_test images"
fi

# ---- 3. arm the once-guard, then one job per checkpoint (3 lists each) ----------
if [ "$DRY" = 0 ]; then
  { echo "DINO final test eval started $(date -u +%FT%TZ) by $(whoami) on $(hostname)"; echo "runs: ${RUNS[*]}"; echo "thresholds: $THR"; } > "$GUARD"
  log "once-guard armed: $GUARD"
fi
declare -A PIDS
for r in "${RUNS[@]}"; do
  CMD=(python cluster/v3/submit_dino_eval.py --eval-name "${r}_test"
       --backbone-dir "${BB[$r]}" --ckpt "$SCR/smoke/runs_v3/$r/best.pt"
       --splits sdis_test
       --extra-list "$REPO/${LISTS[boreal_test]}=boreal_test"
       --extra-list "$REPO/${LISTS[forest_test]}=forest_test"
       --rootmap-override "$ROOTMAP" --width 1280 --height 736 --conf-floor 0.02)
  [ -n "${EXTRA[$r]}" ] && CMD+=(--infer-extra "${EXTRA[$r]}")
  if [ "$DRY" = 1 ]; then
    log "[dry-run] job for $r:"; printf '    %s\n' "${CMD[@]}" --dry-run
    "${CMD[@]}" --dry-run || fail "dry-run submit failed for $r"; continue
  fi
  RES="res-ft$(echo "$r" | tr -d '_' | tr '[:upper:]' '[:lower:]')"
  python cluster/v4/registry.py reserve --job "$RES" --nodes 1 --track dino --note "FINAL TEST $r" || fail "registry denied $r"
  ( "${CMD[@]}" > "logs/v4/finaltest_$r.log" 2>&1; rc=$?
    python cluster/v4/registry.py release --job "$RES" --outcome "finaltest_rc=$rc"; exit $rc ) &
  PIDS[$r]=$!; log "submitted $r (pid ${PIDS[$r]}, log logs/v4/finaltest_$r.log)"; sleep 20
done
if [ "$DRY" = 1 ]; then
  log "[dry-run] would then: gunzip preds -> models/v4/evals/<run>_test/{sdis_test,boreal_test,forest_test}_preds.jsonl;"
  log "[dry-run]   sweep: python -m src.eval.eval_detection --data-yaml <yaml> --split test --preds-jsonl <preds> --out $OUTR/eval_<run>_<set>.json"
  log "[dry-run]   compare_matched_fp per set -> $OUTR/matched_fp_dino_<set>.json; score at $THR -> $OUTR/dino_test_at_preregistered_thresholds.json"
  log "DRY-RUN COMPLETE -- nothing submitted, no registry change, guard not armed, no test image opened"; exit 0
fi
FAILED=0
for r in "${RUNS[@]}"; do wait "${PIDS[$r]}" || { log "job for $r exited non-zero"; FAILED=1; }; done

# ---- 4. collect preds (hard-fail on wrong row count), sweep, compare, score ------
for r in "${RUNS[@]}"; do
  D=models/v4/evals/${r}_test; mkdir -p "$D"
  for s in sdis_test boreal_test forest_test; do
    SRC=models/v3/evals/${r}_test/${s}_preds.jsonl.gz
    [ -f "$SRC" ] && gunzip -cf "$SRC" > "$D/${s}_preds.jsonl"
    n=$(wc -l < "$D/${s}_preds.jsonl" 2>/dev/null || echo 0)
    if [ "$n" != "${EXPECT[$s]}" ]; then
      log "ERROR $r/$s: $n pred rows (expected ${EXPECT[$s]}) -- see models/v3/evals/${r}_test/shardlogs.txt"
      FAILED=1; [ "$n" = 0 ] && rm -f "$D/${s}_preds.jsonl"; continue
    fi
    python -m src.eval.eval_detection --data-yaml "${YAMLS[$s]}" --split test \
      --preds-jsonl "$D/${s}_preds.jsonl" --out "$OUTR/eval_${r}_${s}.json" > "logs/v4/finaltest_sweep_${r}_${s}.log" 2>&1 \
      && log "swept $r/$s -> $OUTR/eval_${r}_${s}.json" || { log "SWEEP FAILED $r/$s"; FAILED=1; }
  done
  cp -f models/v3/evals/${r}_test/shardlogs.txt "$D/" 2>/dev/null
done
for s in sdis_test boreal_test forest_test; do
  M=(); for r in "${RUNS[@]}"; do [ -f "$OUTR/eval_${r}_${s}.raw.parquet" ] && M+=(--model "$r=$OUTR/eval_${r}_${s}.raw.parquet"); done
  [ ${#M[@]} -gt 0 ] && python scripts/v4/compare_matched_fp.py "${M[@]}" --top-fine --fp-targets 0.01,0.02,0.05,0.10,0.20 \
      --out "$OUTR/matched_fp_dino_${s}.json" > "logs/v4/finaltest_mfp_${s}.log" 2>&1 && log "matched-FP $s -> $OUTR/matched_fp_dino_${s}.json"
done
python scripts/v4/dino_test_at_thresholds.py --thresholds "$THR" --evaldir "$OUTR" \
  --out "$OUTR/dino_test_at_preregistered_thresholds.json" && log "threshold scoring -> $OUTR/dino_test_at_preregistered_thresholds.json"
echo "completed $(date -u +%FT%TZ) failed=$FAILED" >> "$GUARD"
log "DONE (failed=$FAILED). Guard: $GUARD"
exit $FAILED
