#!/bin/bash
# ================================================================================
# V5B ONE-SHOT FINAL TEST EVAL -- DINO components of the ACCURATE ensemble (S5).
# RUN EXACTLY ONCE. Pre-registration: reports/v5b/PREREGISTRATION_V5B.md;
# seal: reports/v5b/LOCKBOX_V5B_FROZEN.md; recipe: models/v5b/fusion/accurate_recipe.json
#   dino_B4 = v4B4_lora_L  (seg05, --lora 16)   dino_B6 = v4B6_lora_Hplus (seg10, --lora 16)
#   dino_B1 = v4B1_frozen_L (seg03, no LoRA)
# Inference on TWO sealed lists only: H1 fireman_holdout_s10 (1,949) and H3 sheffield_holdout
# (327). sdis TEST / boreal_test / forest_test DINO preds already exist from the v4 close
# (reports/v4/test/eval_<run>_{sdis,boreal,forest}_test.raw.parquet) and are REUSED, never
# recomputed. One L40S node reserved once (res-v5bfinaldino); the three checkpoints run as
# three sequential kraken jobs inside that reservation. Images are never opened here; GT is
# read locally by src.eval.eval_detection from data/v5b/eval/<set>/labels.
#   bash cluster/v5b/run_final_test_eval_dino.sh --dry-run   # validate + print YAMLs, submits nothing
#   bash cluster/v5b/run_final_test_eval_dino.sh             # real run (refuses to run twice)
#   --check-sha   also sha256 the sealed eval_v5b.tar (466 MB) against the lockbox value
# Pod data: eval_v5b.tar was extracted by the v5b val jobs via cluster/v4/yolo_eval_entry.sh
# (DATA_TARS "eval_v5b@/scratch/.../smoke/v5bdata" -> marker <root>/.eval_v5b.done); the DINO
# entry (cluster/v3/dino_eval_entry.sh) does NOT stage, so this script checks the val-job
# evidence (1,706 fireman_val_s5 rows, 0 errors) and hard-fails post hoc on row counts.
# Afterwards: python scripts/v5b/fuse_test_preds.py --set {firemanholdout,sheffieldholdout,...}
# ================================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
source .venv/bin/activate
export SMOKE_REGISTRY_DIR=cluster/v5b
DRY=0; CHECK_SHA=0
for a in "$@"; do case "$a" in
  --dry-run) DRY=1;; --check-sha) CHECK_SHA=1;;
  *) echo "unknown arg $a"; exit 2;; esac; done
GUARD=reports/v5b/test/.DINO_EVALUATED_ONCE
OUTR=reports/v5b/test
LOGD=logs/v5b
SCR=/scratch/scratch/tianrgua-sandbox
POD_ROOT=$SCR/smoke/v5bdata                 # yolo_eval_entry.sh: name@root -> $root/.<name>.done
POD_EVAL=$POD_ROOT/eval_v5b
POD_MARKER=$POD_ROOT/.eval_v5b.done
RES=res-v5bfinaldino
RECIPE=models/v5b/fusion/accurate_recipe.json
TAR=data/v5b/stage/eval_v5b.tar; TAR_SIZE=466862080
TAR_SHA=a154656669bd00ae3bdfa9c15fc69f0ea16d20545524bc547d80ce45218ae7eb
VAL_EVIDENCE=models/v5b/evals/v5bval1/S1_A__firemanval.jsonl   # written by a job that read $POD_EVAL
RUNS=(v4B4_lora_L v4B6_lora_Hplus v4B1_frozen_L)
declare -A LCK=( [v4B4_lora_L]=models/v3/runs/v4B4_lora_L/seg05/best.pt
                 [v4B6_lora_Hplus]=models/v3/runs/v4B6_lora_Hplus/seg10/best.pt
                 [v4B1_frozen_L]=models/v3/runs/v4B1_frozen_L/seg03/best.pt )
declare -A RNAME=( [v4B4_lora_L]=dino_B4 [v4B6_lora_Hplus]=dino_B6 [v4B1_frozen_L]=dino_B1 )
declare -A BB=( [v4B4_lora_L]=$SCR/models/dinov3-vitl16 [v4B6_lora_Hplus]=$SCR/models/dinov3-vith16plus
                [v4B1_frozen_L]=$SCR/models/dinov3-vitl16 )
declare -A EXTRA=( [v4B4_lora_L]="--lora 16" [v4B6_lora_Hplus]="--lora 16" [v4B1_frozen_L]="" )
SETS=(firemanholdout sheffieldholdout)
declare -A SETDIR=( [firemanholdout]=fireman_holdout_s10 [sheffieldholdout]=sheffield_holdout )
declare -A EXPECT=( [firemanholdout]=1949 [sheffieldholdout]=327 )
# sha256 values copied from reports/v5b/LOCKBOX_V5B_FROZEN.md (seal; any change = re-freeze first)
declare -A LIST_SHA=( [firemanholdout]=e5113180dd96370907c44c81914a54e819e1cc4498828538dc1be70b584ec6bb
                      [sheffieldholdout]=b910f3103bd474c3b2812e33246b804cbd0a42634830a82be1e5a82dc13e5e09 )
declare -A PODLIST_SHA=( [firemanholdout]=38a4e780447c0a725cb12a848a5c0d454388fe26194b20e1b8189f5f0bb7f83b
                         [sheffieldholdout]=ebc35e54c97d5640c062347f9755216b2b644c49d130a12f06dc2735790b2624 )
declare -A YAML_SHA=( [firemanholdout]=3bdb66a01b0bd8ca35995bd705b7c9dd2a89cbfe89cfa55ae8705c14e613d092
                      [sheffieldholdout]=2abbaaf23fbff487afe299e43d1e34e7bad55eae923c5b0d46beb339704b0c47 )
declare -A SUMS_SHA=( [firemanholdout]=171f7c73dcbb90cece67400e3bdca78ba685be0653a652f11163e14fd2c6c71d
                      [sheffieldholdout]=937ed63f7f4e4456ecef1c89dd332e47721a273b55ccff1ffa34eb6b7a4bd15d )
ROOTMAP="$REPO/data/v5b/eval=$POD_EVAL"      # dino_infer.py --root-map localprefix=podprefix
log(){ echo "[final-test-dino-v5b $(date -u +%FT%TZ)] $*"; }
fail(){ log "FATAL: $*"; exit 1; }
sha(){ sha256sum "$1" | cut -c1-64; }

# ---- 0. refuse-to-run-twice ----------------------------------------------------
if [ "$DRY" = 0 ] && [ -e "$GUARD" ]; then
  log "REFUSING: $GUARD exists -> the v5b DINO holdout eval has already been run:"; cat "$GUARD"; exit 9
fi
mkdir -p "$OUTR" "$LOGD"
log "mode: $([ $DRY = 1 ] && echo DRY-RUN || echo REAL) | registry dir $SMOKE_REGISTRY_DIR | reservation $RES (1x L40S, track eval)"

# ---- 1. seal + input validation (lists/labels/yamls/hashes) -- images never opened --
for s in "${SETS[@]}"; do
  d=${SETDIR[$s]}; L=data/v5b/eval/$d/lists/local/$d.txt; P=data/v5b/eval/$d/lists/pod/$d.txt
  Y=data/v5b/eval/yamls/$d.yaml; S=data/v5b/eval/$d/SHA256SUMS
  for f in "$L" "$P" "$Y" "$S"; do [ -f "$f" ] || fail "missing $f"; done
  n=$(grep -c . "$L"); [ "$n" = "${EXPECT[$s]}" ] || fail "$s list has $n lines, expected ${EXPECT[$s]}"
  [ "$(sha "$L")" = "${LIST_SHA[$s]}" ] || fail "SEAL BROKEN: sha256($L) != lockbox value"
  [ "$(sha "$P")" = "${PODLIST_SHA[$s]}" ] || fail "SEAL BROKEN: sha256($P) != lockbox value"
  [ "$(sha "$Y")" = "${YAML_SHA[$s]}" ] || fail "SEAL BROKEN: sha256($Y) != lockbox value"
  [ "$(sha "$S")" = "${SUMS_SHA[$s]}" ] || fail "SEAL BROKEN: sha256($S) != lockbox value"
  bad=$(grep -vc "^$REPO/data/v5b/eval/$d/images/[^/]*$" "$L"); [ "$bad" = 0 ] || fail "$s: $bad list lines not under data/v5b/eval/$d/images/"
  nl=$(ls data/v5b/eval/$d/labels | wc -l); [ "$nl" = "$n" ] || fail "$s labels=$nl != list=$n"
  # the ROOTMAP must turn the local list into exactly the sealed pod list
  [ "$(sed "s#^$REPO/data/v5b/eval#$POD_EVAL#" "$L" | sha256sum | cut -c1-64)" = "${PODLIST_SHA[$s]}" ] \
    || fail "$s: ROOTMAP $ROOTMAP does not reproduce the sealed pod list"
  grep -q "^test: $REPO/$L\$" "$Y" || fail "$Y test: key does not point at $L"
  log "  ok $s: $n images, labels $nl, seal sha ok | list $L | yaml $Y | pod images -> $POD_EVAL/$d/images"
done
[ -f "$TAR" ] || fail "sealed tar missing: $TAR"
sz=$(stat -c %s "$TAR"); [ "$sz" = "$TAR_SIZE" ] || fail "SEAL BROKEN: tar size $sz != $TAR_SIZE"
if [ "$CHECK_SHA" = 1 ]; then
  [ "$(sha "$TAR")" = "$TAR_SHA" ] || fail "SEAL BROKEN: tar sha256 != lockbox value"; log "  ok tar sha256 $TAR_SHA"
else log "  ok tar size $sz (pass --check-sha to hash it)"; fi

# ---- 2. checkpoints (local proxies of the pod copies) + recipe consistency ---------
[ -f "$RECIPE" ] || fail "recipe missing: $RECIPE"
for r in "${RUNS[@]}"; do
  [ -f "${LCK[$r]}" ] || fail "local best.pt missing: ${LCK[$r]}"
  python - "$RECIPE" "${RNAME[$r]}" "${LCK[$r]}" <<'PY' || fail "recipe/$r checkpoint mismatch"
import json,sys; m=json.load(open(sys.argv[1]))["models"][sys.argv[2]]
assert m["checkpoint"]==sys.argv[3], (m["checkpoint"], sys.argv[3])
PY
  log "  ok ckpt $r (${RNAME[$r]}): local ${LCK[$r]} ($(stat -c %s "${LCK[$r]}") B) -> pod $SCR/smoke/runs_v3/$r/best.pt | infer-extra='${EXTRA[$r]}' | backbone ${BB[$r]}"
done
python - "$RECIPE" <<'PY' || fail "recipe model set is not {yolo_A,dino_B4,dino_B6,dino_B1}"
import json,sys; r=json.load(open(sys.argv[1])); assert set(r["models"])=={"yolo_A","dino_B4","dino_B6","dino_B1"}, set(r["models"])
print(f'  ok recipe {r["recipe"]} ({r["created_utc"]}): fusion={r["fusion"]["method"]} conf_type={r["fusion"]["conf_type"]} iou={r["fusion"]["iou_thr"]}; yolo_A ckpt {r["models"]["yolo_A"]["checkpoint"]}')
PY

# ---- 3. v4 DINO test preds that are REUSED (never recomputed) ----------------------
python - "${RUNS[@]}" <<'PY' || fail "v4 DINO test parquets incomplete -- they must be reused, not recomputed"
import sys, pandas as pd
exp={"sdis_test":4124,"boreal_test":1764,"forest_test":2871}
for r in sys.argv[1:]:
    for s,n in exp.items():
        p=f"reports/v4/test/eval_{r}_{s}.raw.parquet"; k=len(pd.read_parquet(p, columns=["image"]))
        assert k==n, (p,k,n)
print("  ok REUSE: reports/v4/test/eval_<run>_{sdis,boreal,forest}_test.raw.parquet present for all 3 runs (4124/1764/2871 rows)")
PY

# ---- 4. code paths + cluster-side prerequisites ------------------------------------
for f in cluster/v3/submit_dino_eval.py cluster/v3/dino_eval_entry.sh cluster/v3/dino_infer.py src/v3/det_modeling.py \
         src/v3/det_data.py cluster/v4/registry.py cluster/v5b/BUDGET_V5B.json scripts/v5b/fuse_test_preds.py \
         src/eval/eval_detection.py; do [ -f "$f" ] || fail "missing $f"; done
grep -q -- "--lora-merge" cluster/v3/dino_infer.py || fail "dino_infer.py lacks the v4 LoRA path"
grep -q -- "--dry-run" cluster/v3/submit_dino_eval.py || fail "submit_dino_eval.py lacks --dry-run"
grep -q "ROOTMAP_OVERRIDE" cluster/v3/dino_eval_entry.sh || fail "dino_eval_entry.sh lacks ROOTMAP_OVERRIDE"
grep -q "INFER_EXTRA" cluster/v3/dino_eval_entry.sh || fail "dino_eval_entry.sh lacks INFER_EXTRA"
grep -q '\.\${name}\.done' cluster/v4/yolo_eval_entry.sh || fail "yolo_eval_entry.sh marker convention changed; re-derive POD_MARKER"
[ -f "$VAL_EVIDENCE" ] || fail "no val-job evidence that eval_v5b is staged on /scratch ($VAL_EVIDENCE)"
ev_n=$(wc -l < "$VAL_EVIDENCE"); ev_err=$(grep -c '"error"' "$VAL_EVIDENCE"); ev_err=${ev_err:-0}
[ "$ev_n" = 1706 ] && [ "$ev_err" = 0 ] || fail "val evidence $VAL_EVIDENCE: rows=$ev_n errors=$ev_err (expected 1706/0)"
log "  ok staging evidence: $VAL_EVIDENCE (1706 rows, 0 errors) read $POD_EVAL/fireman_val_s5 -> eval_v5b.tar extracted at $POD_ROOT (marker $POD_MARKER)"
for r in "${RUNS[@]}"; do
  if [ -e "models/v3/evals/${r}_v5btest" ]; then
    [ "$DRY" = 1 ] && log "  WARNING: models/v3/evals/${r}_v5btest already exists (previous attempt?)" \
                   || fail "models/v3/evals/${r}_v5btest already exists -- a previous attempt downloaded preds; investigate before re-running"
  fi
done
python cluster/v4/registry.py status | python -c "
import json,sys,datetime as dt; d=json.load(sys.stdin)
dl=dt.datetime.fromisoformat(d['deadline_utc'].replace('Z','+00:00')); now=dt.datetime.fromisoformat(d['utc_now'].replace('Z','+00:00'))
print(f'  registry(v5b): active {d[\"active_nodes\"]}/{d[\"node_cap\"]} nodes ({d[\"active_nodes_by_type\"]}), {d[\"est_node_hours_spent\"]}/{d[\"node_hour_cap\"]} node-h, deadline {d[\"deadline_utc\"]} ({(dl-now).total_seconds()/3600:.1f} h left)')
sys.exit(0 if now<dl else 7)" || fail "past the v5b GPU window deadline"

# ---- 5. build the three job commands (one per checkpoint, 2 lists each) ------------
declare -A CMDS
for r in "${RUNS[@]}"; do
  CMD=(python cluster/v3/submit_dino_eval.py --eval-name "${r}_v5btest"
       --backbone-dir "${BB[$r]}" --ckpt "$SCR/smoke/runs_v3/$r/best.pt")
  for s in "${SETS[@]}"; do d=${SETDIR[$s]}; CMD+=(--extra-list "$REPO/data/v5b/eval/$d/lists/local/$d.txt=$s"); done
  CMD+=(--rootmap-override "$ROOTMAP" --width 1280 --height 736 --conf-floor 0.02)
  [ -n "${EXTRA[$r]}" ] && CMD+=(--infer-extra "${EXTRA[$r]}")
  CMDS[$r]=$(printf '%q ' "${CMD[@]}")
done

if [ "$DRY" = 1 ]; then
  log "[dry-run] would reserve: python cluster/v4/registry.py reserve --job $RES --nodes 1 --gpu-type L40S --track eval --note 'ONE-SHOT v5b final DINO B4/B6/B1 on H1+H3'"
  for r in "${RUNS[@]}"; do
    log "[dry-run] job for $r (sequential, inside the single reservation):"; echo "    ${CMDS[$r]}--dry-run"
    eval "${CMDS[$r]}--dry-run" || fail "dry-run submit failed for $r"
  done
  log "[dry-run] would then: gunzip models/v3/evals/<run>_v5btest/<set>_preds.jsonl.gz -> models/v5b/evals/<run>_test/<set>_preds.jsonl (rows must be 1949/327, 0 error rows)"
  log "[dry-run]   sweep: python -m src.eval.eval_detection --data-yaml data/v5b/eval/yamls/<setdir>.yaml --split test --preds-jsonl <preds> --out $OUTR/eval_<run>_<set>.json"
  log "[dry-run]   recall at pre-registered sdis-VAL confs (fixed-camera regime; both holdouts have 0 negatives -> recall only) -> $OUTR/dino_holdout_recall_at_sdisval_thresholds.json"
  log "[dry-run]   then: python scripts/v5b/fuse_test_preds.py --set <set> --yolo <S1_A|v4a_s2> for S5"
  log "DRY-RUN COMPLETE -- nothing submitted, no registry change, guard not armed, no holdout image opened"; exit 0
fi

# ---- 6. REAL: reserve once, arm the guard, three sequential jobs, release ---------
python cluster/v4/registry.py reserve --job "$RES" --nodes 1 --gpu-type L40S --track eval \
  --note "ONE-SHOT v5b final DINO B4/B6/B1 on H1 fireman_holdout_s10 + H3 sheffield_holdout" || fail "registry denied $RES (nothing submitted, guard not armed)"
{ echo "v5b DINO holdout eval started $(date -u +%FT%TZ) by $(whoami) on $(hostname)"
  echo "runs: ${RUNS[*]} | sets: ${SETS[*]} | reservation: $RES | recipe: $RECIPE"; } > "$GUARD"
log "once-guard armed: $GUARD"
FAILED=0; RCS=""
for r in "${RUNS[@]}"; do
  log "submitting $r -> $LOGD/finaltest_dino_$r.log"
  eval "${CMDS[$r]}" > "$LOGD/finaltest_dino_$r.log" 2>&1; rc=$?
  RCS="$RCS $r=$rc"; [ $rc = 0 ] || { log "job for $r exited rc=$rc"; FAILED=1; }
done
python cluster/v4/registry.py release --job "$RES" --outcome "finaltest_dino rc:$RCS"

# ---- 7. collect preds (hard-fail on wrong row count / error rows), sweep ------------
for r in "${RUNS[@]}"; do
  D=models/v5b/evals/${r}_test; mkdir -p "$D"
  for s in "${SETS[@]}"; do
    d=${SETDIR[$s]}; SRC=models/v3/evals/${r}_v5btest/${s}_preds.jsonl.gz
    [ -f "$SRC" ] && gunzip -cf "$SRC" > "$D/${s}_preds.jsonl"
    n=$(wc -l < "$D/${s}_preds.jsonl" 2>/dev/null); n=${n:-0}
    e=$(grep -c '"error"' "$D/${s}_preds.jsonl" 2>/dev/null); e=${e:-0}
    if [ "$n" != "${EXPECT[$s]}" ] || [ "$e" != 0 ]; then
      log "ERROR $r/$s: $n pred rows (expected ${EXPECT[$s]}), $e error rows -- see models/v3/evals/${r}_v5btest/shardlogs.txt (staged data missing at $POD_EVAL? marker $POD_MARKER)"
      FAILED=1; [ "$n" = 0 ] && rm -f "$D/${s}_preds.jsonl"; continue
    fi
    python -m src.eval.eval_detection --data-yaml "data/v5b/eval/yamls/$d.yaml" --split test \
      --preds-jsonl "$D/${s}_preds.jsonl" --out "$OUTR/eval_${r}_${s}.json" > "$LOGD/finaltest_dino_sweep_${r}_${s}.log" 2>&1 \
      && log "swept $r/$s -> $OUTR/eval_${r}_${s}.json (+ .raw.parquet)" || { log "SWEEP FAILED $r/$s"; FAILED=1; }
  done
  cp -f models/v3/evals/${r}_v5btest/shardlogs.txt "$D/" 2>/dev/null
done

# ---- 8. recall at the pre-registered sdis-VAL confs (recall-only sets; non-fatal) ----
THR=reports/v4/test/dino_test_thresholds.json
if [ -f "$THR" ]; then
python - "$THR" "$OUTR" "${RUNS[@]}" <<'PY' && log "recall@pre-registered confs -> $OUTR/dino_holdout_recall_at_sdisval_thresholds.json" || log "WARNING: threshold scoring failed (non-fatal)"
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "."); from src.eval.eval_detection import sweep_report
thr, outr, runs = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3:]
out = {"_doc": "box recall (IoU>=0.4) and image-level recall at the sdis-VAL neg-image-FP confs frozen in "
       "reports/v4/test/dino_test_thresholds.json (fixed-camera regime). Both holdouts have 0 negatives -> "
       "recall only (PREREGISTRATION amendment 09:35). No aerial-regime (FireMan-VAL) confs exist for the DINO components.", "runs": {}}
for r in runs:
    confs = {k: v["conf"] for k, v in thr["runs"][r]["sdis_val_source_only"].items()}
    out["runs"][r] = {"confs_sdis_val": confs}
    for s in ("firemanholdout", "sheffieldholdout"):
        p = f"{outr}/eval_{r}_{s}.raw.parquet"
        try: rows = pd.read_parquet(p).to_dict("records")
        except Exception as e: out["runs"][r][s] = {"error": str(e)}; continue
        rep = sweep_report(rows, 0.4, conf_grid=sorted(set(confs.values())))
        imx = np.array([max((q[4] for q in rw["preds"]), default=0.0) for rw in rows]); pos = np.array([len(rw["gt"]) > 0 for rw in rows])
        out["runs"][r][s] = {"n_images": len(rows), "n_pos": int(pos.sum()), "n_neg": int((~pos).sum()), "at": {
            k: {"conf": c, "box_recall": next(x["recall"] for x in rep["conf_sweep"] if x["conf"] == c),
                "tiny_recall": next(x["recall_by_size"]["tiny"] for x in rep["conf_sweep"] if x["conf"] == c),
                "image_recall": round(float((imx[pos] >= c).mean()), 4) if pos.any() else None} for k, c in confs.items()}}
json.dump(out, open(f"{outr}/dino_holdout_recall_at_sdisval_thresholds.json", "w"), indent=1)
PY
fi
echo "completed $(date -u +%FT%TZ) failed=$FAILED rc:$RCS" >> "$GUARD"
log "DONE (failed=$FAILED). Guard: $GUARD. Next: python scripts/v5b/fuse_test_preds.py --set firemanholdout --yolo <S1_A|v4a_s2> (and sheffieldholdout, sdistest, borealtest, foresttest)"
exit $FAILED
