"""Render models/v5b/teacher/TEACHER_POOL.md from teacher_pool_stats.json + raw job records + registry."""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path("/local/home/tianrgua/cvproject"); OUT = ROOT / "models/v5b/teacher"
S = json.loads((OUT / "teacher_pool_stats.json").read_text())
tm = json.loads((ROOT / "data/v5b/teacher/lists/TARGETS.json").read_text())
reg = json.loads((ROOT / "cluster/v5b/registry.json").read_text())["jobs"]
jobs = [json.loads(f.read_text()) for f in sorted((OUT / "raw").glob("*/job.json"))]
L = []
w = L.append
w(f"# V5b TEACHER prediction pool for arm D (teacher-assisted supervision) -- generated {S['generated_utc']}\n")
w("## Label type -- read first\n")
w("**Every row in `models/v5b/teacher/*__train.jsonl` is a MODEL-ASSISTED label** (a teacher detector's prediction at "
  "conf floor 0.02). These are **not author labels, not human-reviewed, not human-verified**; the statistics below are "
  "automated inspection, which is **not** human review. Arm D must record anything derived from this pool as "
  "`model-assisted` and must never present it as human annotation. Where this file compares teachers with "
  "\"author labels\" it means the public datasets' own annotations (pyro-sdis / D-Fire / Boreal YOLO txt), a separate "
  "annotation type that is reported separately and never merged into the teacher rows.\n")
w("## Scope -- TRAINING images only\n")
c = tm["counts"]
w(f"- Targets: **{c['total']:,}** images = E corpus `data/yolo/lists/local/expB_train.txt` ({c['E_pyro_sdis_train']:,} pyro-sdis TRAIN-site frames "
  f"[brison, serre-de-barre, courmettes, valbonne] + {c['E_dfire_train']:,} D-Fire) + Boreal TRAIN burns from `data/v4/yolo/E_ET_v1` "
  f"({c['T_boreal_karkkila']:,} karkkila + {c['T_boreal_evo']:,} evo + {c['T_boreal_karkkila_empty_clip']} karkkila Empty-Images non-smoke clip).")
w(f"- **Excluded on purpose: {c['excluded_ausmoke_in_E_ET_v1_T']:,} AuSmoke frames** that sit in E_ET_v1's T list -- {tm['exclusion_reason']}")
w("- Not scored: pyro-sdis VAL/TEST sites, Boreal heinola (aerial VAL) / ruokolahti (historical test), AuSmoke val/test units, FIgLib/lockboxes, "
  "any v5 acquisition. Leak check (path + basename) against every val/test list: "
  + ", ".join(f"{k.split('/')[-1]}={v['path_overlap']}/{v['basename_overlap']}" for k, v in tm["leak_check_vs_eval_lists"].items()) + " (all 0).")
w(f"- Lists + sha256: `data/v5b/teacher/lists/TARGETS.json`. Local/pod root map: `{tm['rootmap_local_to_pod']}`\n")
w("## Teachers, jobs, compute\n")
w("| teacher | checkpoint | inference | kraken jobs (wall) | rows / targets | unreadable | missing |\n|---|---|---|---|---|---|---|")
for t, st in S["teachers"].items():
    pre = {"dino_b4_lora_L": "b4L", "dino_b6_lora_Hplus": "b6H", "rfdetr_L": "rfdetrL"}[t]
    js = [j for j in jobs if j["eval_name"].startswith(pre)]
    jtxt = "; ".join(f"{j.get('job')} {j.get('final_state','?')} {j.get('wall_s',0)//60} min" for j in js) or "-"
    cv = st["coverage"]
    w(f"| {t} | {st['checkpoint']} | {st['inference']} | {jtxt} | {cv['rows']:,} / {cv['targets']:,} | {cv['unreadable_errors']} | {cv['missing']} |")
res = {k: v for k, v in reg.items() if v.get("track") == "teacher"}
nh = 0.0
from datetime import datetime
for k, v in res.items():
    if "released_utc" in v:
        nh += (datetime.fromisoformat(v["released_utc"].replace("Z", "+00:00")) - datetime.fromisoformat(v["reserved_utc"].replace("Z", "+00:00"))).total_seconds() / 3600
w(f"\nRegistry (track `teacher`, L40S g6e 1 node each): {len(res)} reservations, **{nh:.2f} node-h** reserved->released "
  f"({nh*8:.1f} GPU-h); max 3 concurrent. Raw per-job outputs + pod log tails: `models/v5b/teacher/raw/<eval>/`, /scratch copy `smoke/v5b/teacher/<eval>/`.\n")
w("## Score distributions (per-image max score; conf floor 0.02)\n")
for t, st in S["teachers"].items():
    h = st["max_score_hist"]; b = st["boxes"]
    w(f"**{t}** -- boxes >=0.02: {b['total_boxes_ge_floor']:,} (mean {b['boxes_per_image_mean']:.2f}/img, p50/p90/max {b['boxes_per_image_p50_p90_max']}); "
      f"box-score quantiles p10/p50/p90/p99 = {[round(x,3) for x in b['score_quantiles_p10_p50_p90_p99']]}")
    w(f"- max-score histogram: none={h['none(no box>=0.02)']:,}; " + "; ".join(f"[{h['edges'][i]},{h['edges'][i+1] if h['edges'][i+1] <= 1 else 1.0})={n:,}" for i, n in enumerate(h["counts"])))
    fr = st["images_firing_at_thr"]
    w("- images firing (max >= thr): " + "; ".join(f"@{th}: {v['all']:,} (sdis {v['pyro_sdis']:,} / dfire {v['dfire']:,} / boreal {v['boreal']:,})" for th, v in fr.items()) + "\n")
w("## Agreement between teachers (image level = max score >= thr on both; box level = IoU>=0.4 greedy match)\n")
for pair, d in S["agreement"].items():
    if pair == "3-way": continue
    t1, t2 = pair.split("|")
    w(f"**{t1} vs {t2}** (n={d['n_common']:,})\n")
    w("| thr | both | only " + t1 + " | only " + t2 + " | neither | kappa | Jaccard | " + t1 + " boxes matched by " + t2 + " | " + t2 + " boxes matched by " + t1 + " |\n|---|---|---|---|---|---|---|---|---|")
    for th, v in d["at_thr"].items():
        w(f"| {th} | {v['both_fire']:,} | {v[f'only_{t1}']:,} | {v[f'only_{t2}']:,} | {v['neither']:,} | {v['image_kappa']:.3f} | {v['image_jaccard']:.3f} | "
          f"{v[f'box_frac_{t1}_matched_by_{t2}']:.3f} | {v[f'box_frac_{t2}_matched_by_{t1}']:.3f} |")
    w("")
if "3-way" in S["agreement"]:
    d = S["agreement"]["3-way"]
    w(f"**3-way** (n={d['n_common']:,}): images where 3/2/1/0 teachers fire\n\n| thr | 3 of 3 | 2 of 3 | 1 of 3 | 0 of 3 | 3-of-3 by source (sdis/dfire/boreal) |\n|---|---|---|---|---|---|")
    for th, v in d["at_thr"].items():
        bs = v["by_source_3of3"]
        w(f"| {th} | {v['fire_3of3']:,} | {v['fire_2of3']:,} | {v['fire_1of3']:,} | {v['fire_0of3']:,} | {bs['pyro_sdis']:,} / {bs['dfire']:,} / {bs['boreal']:,} |")
    w("")
w("## Reference only: agreement with AUTHOR labels (dataset annotations -- NOT our human review)\n")
for t, st in S["teachers"].items():
    va = st["vs_author_labels"]
    w(f"**{t}** -- author-positive images {va['author_positive_images']:,}, author-empty {va['author_empty_images']:,}, author boxes {va['author_boxes']:,}\n")
    w("| thr | image recall on author-pos | fire rate on author-empty | box recall of author boxes | teacher boxes | matching author | box precision vs author |\n|---|---|---|---|---|---|---|")
    for th, v in va["at_thr"].items():
        w(f"| {th} | {v['img_recall_author_pos']:.3f} | {v['img_fire_rate_author_empty']:.3f} | {v['box_recall_author']:.3f} | {v['teacher_boxes']:,} | {v['teacher_boxes_matching_author']:,} | {v['teacher_box_precision_vs_author']:.3f} |")
    w("")
w("**In-sample caveat:** every teacher was TRAINED on these images (RF-DETR-L on the E corpus; DINO B4/B6 LoRA on E_ET incl. the Boreal train burns), "
  "so agreement with author labels here is memorisation-inflated and says nothing about generalisation; use it only to pick pool thresholds / to find "
  "author-label omissions worth a human look. The pool is still useful as DENSE soft supervision (scores + boxes on author-empty regions) but any claim about "
  "teacher quality must come from the val/calibration sets (reports/v4, reports/v5/CALIBRATION_*), not from this file.\n")
w("Caveats: pyro-sdis author labels have known omission flicker (reports/v5/AUDIT_ONSET_V5.md), D-Fire has no event metadata, Boreal boxes are "
  "mask-derived -- 'fire rate on author-empty' therefore mixes teacher false alarms with author omissions and must not be read as an FP rate. "
  "RF-DETR has no NMS (duplicates inflate its box counts; cf. reports/v3/rfdetr_eval.md). DINO LoRA scores saturate >=0.99 (v4), so compare at "
  "high thresholds too.\n")
w("## Files\n")
for t in S["teachers"]:
    w(f"- `models/v5b/teacher/{t}__train.jsonl` -- one row per target image (`{{image: LOCAL path, w, h, preds: [[x1n,y1n,x2n,y2n,score],...], teacher, label_type: model-assisted}}`; "
      f"unreadable images carry `error`); `missing__{t}.txt` lists any target without a row.")
w("- `models/v5b/teacher/teacher_pool_stats.json` -- all numbers above; `data/v5b/teacher/lists/` -- target lists; "
  "`cluster/v5b/{teacher_infer_entry.sh,submit_teacher_infer.py}`, `scripts/v5b/{teacher_pool_build_lists,build_teacher_pool,render_teacher_pool_md}.py`, `scripts/v5b/teacher_job.sh`.")
(OUT / "TEACHER_POOL.md").write_text("\n".join(L) + "\n"); print("->", OUT / "TEACHER_POOL.md")
