# V5b TEACHER prediction pool for arm D (teacher-assisted supervision) -- generated 2026-10-02T08:24:53Z

## Label type -- read first

**Every row in `models/v5b/teacher/*__train.jsonl` is a MODEL-ASSISTED label** (a teacher detector's prediction at conf floor 0.02). These are **not author labels, not human-reviewed, not human-verified**; the statistics below are automated inspection, which is **not** human review. Arm D must record anything derived from this pool as `model-assisted` and must never present it as human annotation. Where this file compares teachers with "author labels" it means the public datasets' own annotations (pyro-sdis / D-Fire / Boreal YOLO txt), a separate annotation type that is reported separately and never merged into the teacher rows.

## Scope -- TRAINING images only

- Targets: **40,516** images = E corpus `data/yolo/lists/local/expB_train.txt` (22,177 pyro-sdis TRAIN-site frames [brison, serre-de-barre, courmettes, valbonne] + 16,277 D-Fire) + Boreal TRAIN burns from `data/v4/yolo/E_ET_v1` (1,095 karkkila + 930 evo + 37 karkkila Empty-Images non-smoke clip).
- **Excluded on purpose: 10,904 AuSmoke frames** that sit in E_ET_v1's T list -- AuSmoke (via MultiNatSmoke) permission PENDING per reports/v5/PROTOCOL_V5.md S2 -- no new training use; scoring it for arm D would be a new training use. Re-run with excluded_ausmoke_T.txt if permission lands.
- Not scored: pyro-sdis VAL/TEST sites, Boreal heinola (aerial VAL) / ruokolahti (historical test), AuSmoke val/test units, FIgLib/lockboxes, any v5 acquisition. Leak check (path + basename) against every val/test list: sdis_val.txt=0/0, sdis_test.txt=0/0, dfire_test_raw.txt=0/0, val.txt=0/0, test.txt=0/0, val.txt=0/0, test.txt=0/0, gval.txt=0/0 (all 0).
- Lists + sha256: `data/v5b/teacher/lists/TARGETS.json`. Local/pod root map: `/local/home/tianrgua/cvproject/data/yolo=/scratch/scratch/tianrgua-sandbox/smoke_data_v2,/local/home/tianrgua/cvproject/data/v4/yolo/E_ET_v1=/scratch/scratch/tianrgua-sandbox/smoke/v4data/E_ET_v1`

## Teachers, jobs, compute

| teacher | checkpoint | inference | kraken jobs (wall) | rows / targets | unreadable | missing |
|---|---|---|---|---|---|---|
| dino_b4_lora_L | models/v3/runs/v4B4_lora_L/seg05/best.pt (pod runs_v3/v4B4_lora_L/best.pt, sha 87a9f175) | dino_infer.py 1280x736 LoRA r16, conf floor 0.02 | tianrgua-tchb4la-bnpo SUCCEEDED 9 min; tianrgua-tchb4lb-o38j SUCCEEDED 9 min | 40,516 / 40,516 | 0 | 0 |
| dino_b6_lora_Hplus | models/v3/runs/v4B6_lora_Hplus/seg10/best.pt (pod runs_v3/v4B6_lora_Hplus/best.pt, sha de235e28) | dino_infer.py 1280x736 LoRA r16, conf floor 0.02 | tianrgua-tchb6ha-wmd6 SUCCEEDED 30 min; tianrgua-tchb6hb-pft4 SUCCEEDED 30 min | 40,516 / 40,516 | 0 | 0 |
| rfdetr_L | models/v3/rfdetr/checkpoint_best_ema.pth (pod runs_v3/rfdetrL_1280_e48/, sha 2cca1fe8) | rfdetr_infer.py imgsz 1280, conf floor 0.02 (no NMS) | tianrgua-tchrfdetrlall-oekh SUCCEEDED 30 min | 40,516 / 40,516 | 0 | 0 |

Registry (track `teacher`, L40S g6e 1 node each): 5 reservations, **1.86 node-h** reserved->released (14.9 GPU-h); max 3 concurrent. Raw per-job outputs + pod log tails: `models/v5b/teacher/raw/<eval>/`, /scratch copy `smoke/v5b/teacher/<eval>/`.

## Score distributions (per-image max score; conf floor 0.02)

**dino_b4_lora_L** -- boxes >=0.02: 111,064 (mean 2.74/img, p50/p90/max [2.0, 6.0, 72]); box-score quantiles p10/p50/p90/p99 = [0.027, 0.215, 0.996, 1.0]
- max-score histogram: none=7,467; [0.02,0.1)=939; [0.1,0.3)=568; [0.3,0.5)=410; [0.5,0.7)=542; [0.7,0.9)=2,016; [0.9,0.95)=2,482; [0.95,0.99)=8,729; [0.99,1.0)=17,363
- images firing (max >= thr): @0.05: 32,472 (sdis 21,312 / dfire 9,134 / boreal 2,026); @0.1: 32,110 (sdis 21,150 / dfire 8,934 / boreal 2,026); @0.2: 31,780 (sdis 20,971 / dfire 8,783 / boreal 2,026); @0.3: 31,542 (sdis 20,821 / dfire 8,695 / boreal 2,026); @0.5: 31,132 (sdis 20,517 / dfire 8,589 / boreal 2,026); @0.7: 30,590 (sdis 20,084 / dfire 8,482 / boreal 2,024); @0.9: 28,574 (sdis 18,564 / dfire 7,987 / boreal 2,023); @0.95: 26,092 (sdis 16,730 / dfire 7,339 / boreal 2,023)

**dino_b6_lora_Hplus** -- boxes >=0.02: 117,449 (mean 2.90/img, p50/p90/max [2.0, 7.0, 54]); box-score quantiles p10/p50/p90/p99 = [0.027, 0.188, 0.996, 0.999]
- max-score histogram: none=7,237; [0.02,0.1)=951; [0.1,0.3)=616; [0.3,0.5)=456; [0.5,0.7)=546; [0.7,0.9)=1,761; [0.9,0.95)=2,281; [0.95,0.99)=8,896; [0.99,1.0)=17,772
- images firing (max >= thr): @0.05: 32,728 (sdis 21,435 / dfire 9,266 / boreal 2,027); @0.1: 32,328 (sdis 21,262 / dfire 9,040 / boreal 2,026); @0.2: 31,937 (sdis 21,045 / dfire 8,866 / boreal 2,026); @0.3: 31,712 (sdis 20,918 / dfire 8,768 / boreal 2,026); @0.5: 31,256 (sdis 20,598 / dfire 8,632 / boreal 2,026); @0.7: 30,710 (sdis 20,171 / dfire 8,513 / boreal 2,026); @0.9: 28,949 (sdis 18,808 / dfire 8,116 / boreal 2,025); @0.95: 26,668 (sdis 17,060 / dfire 7,584 / boreal 2,024)

**rfdetr_L** -- boxes >=0.02: 210,308 (mean 5.19/img, p50/p90/max [2.0, 11.0, 173]); box-score quantiles p10/p50/p90/p99 = [0.022, 0.037, 0.82, 0.932]
- max-score histogram: none=5,464; [0.02,0.1)=3,348; [0.1,0.3)=1,633; [0.3,0.5)=1,326; [0.5,0.7)=2,423; [0.7,0.9)=18,793; [0.9,0.95)=7,045; [0.95,0.99)=484; [0.99,1.0)=0
- images firing (max >= thr): @0.05: 32,763 (sdis 21,432 / dfire 9,307 / boreal 2,024); @0.1: 31,704 (sdis 20,916 / dfire 8,764 / boreal 2,024); @0.2: 30,709 (sdis 20,144 / dfire 8,541 / boreal 2,024); @0.3: 30,071 (sdis 19,577 / dfire 8,470 / boreal 2,024); @0.5: 28,745 (sdis 18,424 / dfire 8,308 / boreal 2,013); @0.7: 26,322 (sdis 16,514 / dfire 7,846 / boreal 1,962); @0.9: 7,529 (sdis 3,250 / dfire 2,906 / boreal 1,373); @0.95: 484 (sdis 46 / dfire 387 / boreal 51)

## Agreement between teachers (image level = max score >= thr on both; box level = IoU>=0.4 greedy match)

**dino_b4_lora_L vs dino_b6_lora_Hplus** (n=40,516)

| thr | both | only dino_b4_lora_L | only dino_b6_lora_Hplus | neither | kappa | Jaccard | dino_b4_lora_L boxes matched by dino_b6_lora_Hplus | dino_b6_lora_Hplus boxes matched by dino_b4_lora_L |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 32,039 | 433 | 689 | 7,355 | 0.912 | 0.966 | 0.804 | 0.772 |
| 0.1 | 31,742 | 368 | 586 | 7,820 | 0.928 | 0.971 | 0.836 | 0.811 |
| 0.2 | 31,444 | 336 | 493 | 8,243 | 0.939 | 0.974 | 0.868 | 0.850 |
| 0.3 | 31,251 | 291 | 461 | 8,513 | 0.946 | 0.977 | 0.889 | 0.873 |
| 0.5 | 30,817 | 315 | 439 | 8,945 | 0.948 | 0.976 | 0.915 | 0.901 |
| 0.7 | 30,225 | 365 | 485 | 9,441 | 0.943 | 0.973 | 0.934 | 0.922 |
| 0.9 | 27,809 | 765 | 1,140 | 10,802 | 0.886 | 0.936 | 0.946 | 0.925 |
| 0.95 | 24,923 | 1,169 | 1,745 | 12,679 | 0.842 | 0.895 | 0.937 | 0.911 |

**dino_b4_lora_L vs rfdetr_L** (n=40,516)

| thr | both | only dino_b4_lora_L | only rfdetr_L | neither | kappa | Jaccard | dino_b4_lora_L boxes matched by rfdetr_L | rfdetr_L boxes matched by dino_b4_lora_L |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 31,782 | 690 | 981 | 7,063 | 0.869 | 0.950 | 0.619 | 0.631 |
| 0.1 | 31,236 | 874 | 468 | 7,938 | 0.901 | 0.959 | 0.645 | 0.821 |
| 0.2 | 30,513 | 1,267 | 196 | 8,540 | 0.898 | 0.954 | 0.675 | 0.928 |
| 0.3 | 29,955 | 1,587 | 116 | 8,858 | 0.885 | 0.946 | 0.695 | 0.961 |
| 0.5 | 28,676 | 2,456 | 69 | 9,315 | 0.839 | 0.919 | 0.723 | 0.980 |
| 0.7 | 26,220 | 4,370 | 102 | 9,824 | 0.740 | 0.854 | 0.723 | 0.979 |
| 0.9 | 7,518 | 21,056 | 11 | 11,931 | 0.173 | 0.263 | 0.234 | 0.996 |
| 0.95 | 484 | 25,608 | 0 | 14,424 | 0.013 | 0.018 | 0.017 | 1.000 |

**dino_b6_lora_Hplus vs rfdetr_L** (n=40,516)

| thr | both | only dino_b6_lora_Hplus | only rfdetr_L | neither | kappa | Jaccard | dino_b6_lora_Hplus boxes matched by rfdetr_L | rfdetr_L boxes matched by dino_b6_lora_Hplus |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 31,862 | 866 | 901 | 6,887 | 0.859 | 0.948 | 0.601 | 0.638 |
| 0.1 | 31,264 | 1,064 | 440 | 7,748 | 0.888 | 0.954 | 0.628 | 0.824 |
| 0.2 | 30,547 | 1,390 | 162 | 8,417 | 0.891 | 0.952 | 0.663 | 0.931 |
| 0.3 | 29,977 | 1,735 | 94 | 8,710 | 0.876 | 0.943 | 0.684 | 0.963 |
| 0.5 | 28,685 | 2,571 | 60 | 9,200 | 0.832 | 0.916 | 0.713 | 0.981 |
| 0.7 | 26,245 | 4,465 | 77 | 9,729 | 0.735 | 0.853 | 0.714 | 0.980 |
| 0.9 | 7,516 | 21,433 | 13 | 11,554 | 0.166 | 0.260 | 0.229 | 0.996 |
| 0.95 | 484 | 26,184 | 0 | 13,848 | 0.013 | 0.018 | 0.017 | 1.000 |

**3-way** (n=40,516): images where 3/2/1/0 teachers fire

| thr | 3 of 3 | 2 of 3 | 1 of 3 | 0 of 3 | 3-of-3 by source (sdis/dfire/boreal) |
|---|---|---|---|---|---|
| 0.05 | 31,647 | 742 | 1,538 | 6,589 | 21,008 / 8,615 / 2,024 |
| 0.1 | 31,154 | 780 | 1,120 | 7,462 | 20,634 / 8,496 / 2,024 |
| 0.2 | 30,470 | 1,094 | 828 | 8,124 | 19,992 / 8,454 / 2,024 |
| 0.3 | 29,924 | 1,411 | 731 | 8,450 | 19,475 / 8,425 / 2,024 |
| 0.5 | 28,643 | 2,249 | 706 | 8,918 | 18,358 / 8,272 / 2,013 |
| 0.7 | 26,174 | 4,168 | 764 | 9,410 | 16,428 / 7,785 / 1,961 |
| 0.9 | 7,507 | 20,322 | 1,887 | 10,800 | 3,233 / 2,901 / 1,373 |
| 0.95 | 484 | 24,439 | 2,914 | 12,679 | 46 / 387 / 51 |

## Reference only: agreement with AUTHOR labels (dataset annotations -- NOT our human review)

**dino_b4_lora_L** -- author-positive images 29,222, author-empty 11,294, author boxes 33,555

| thr | image recall on author-pos | fire rate on author-empty | box recall of author boxes | teacher boxes | matching author | box precision vs author |
|---|---|---|---|---|---|---|
| 0.05 | 1.000 | 0.288 | 0.992 | 82,605 | 33,285 | 0.403 |
| 0.1 | 1.000 | 0.257 | 0.990 | 67,776 | 33,201 | 0.490 |
| 0.2 | 0.999 | 0.228 | 0.986 | 56,554 | 33,067 | 0.585 |
| 0.3 | 0.999 | 0.208 | 0.981 | 50,888 | 32,931 | 0.647 |
| 0.5 | 0.998 | 0.175 | 0.971 | 44,033 | 32,564 | 0.740 |
| 0.7 | 0.993 | 0.139 | 0.950 | 38,866 | 31,877 | 0.820 |
| 0.9 | 0.945 | 0.085 | 0.873 | 32,185 | 29,278 | 0.910 |
| 0.95 | 0.870 | 0.058 | 0.795 | 28,374 | 26,690 | 0.941 |

**dino_b6_lora_Hplus** -- author-positive images 29,222, author-empty 11,294, author boxes 33,555

| thr | image recall on author-pos | fire rate on author-empty | box recall of author boxes | teacher boxes | matching author | box precision vs author |
|---|---|---|---|---|---|---|
| 0.05 | 1.000 | 0.311 | 0.992 | 85,984 | 33,271 | 0.387 |
| 0.1 | 1.000 | 0.276 | 0.989 | 69,899 | 33,197 | 0.475 |
| 0.2 | 0.999 | 0.242 | 0.985 | 57,719 | 33,055 | 0.573 |
| 0.3 | 0.999 | 0.223 | 0.982 | 51,827 | 32,935 | 0.635 |
| 0.5 | 0.998 | 0.185 | 0.971 | 44,710 | 32,571 | 0.729 |
| 0.7 | 0.993 | 0.149 | 0.950 | 39,394 | 31,870 | 0.809 |
| 0.9 | 0.955 | 0.092 | 0.883 | 32,899 | 29,631 | 0.901 |
| 0.95 | 0.888 | 0.063 | 0.811 | 29,197 | 27,210 | 0.932 |

**rfdetr_L** -- author-positive images 29,222, author-empty 11,294, author boxes 33,555

| thr | image recall on author-pos | fire rate on author-empty | box recall of author boxes | teacher boxes | matching author | box precision vs author |
|---|---|---|---|---|---|---|
| 0.05 | 1.000 | 0.314 | 0.994 | 81,030 | 33,361 | 0.412 |
| 0.1 | 1.000 | 0.220 | 0.992 | 53,266 | 33,295 | 0.625 |
| 0.2 | 0.999 | 0.134 | 0.987 | 41,135 | 33,101 | 0.805 |
| 0.3 | 0.997 | 0.083 | 0.976 | 36,809 | 32,751 | 0.890 |
| 0.5 | 0.974 | 0.026 | 0.936 | 32,498 | 31,398 | 0.966 |
| 0.7 | 0.898 | 0.007 | 0.845 | 28,699 | 28,369 | 0.989 |
| 0.9 | 0.258 | 0.000 | 0.226 | 7,572 | 7,569 | 1.000 |
| 0.95 | 0.017 | 0.000 | 0.014 | 484 | 484 | 1.000 |

**In-sample caveat:** every teacher was TRAINED on these images (RF-DETR-L on the E corpus; DINO B4/B6 LoRA on E_ET incl. the Boreal train burns), so agreement with author labels here is memorisation-inflated and says nothing about generalisation; use it only to pick pool thresholds / to find author-label omissions worth a human look. The pool is still useful as DENSE soft supervision (scores + boxes on author-empty regions) but any claim about teacher quality must come from the val/calibration sets (reports/v4, reports/v5/CALIBRATION_*), not from this file.

Caveats: pyro-sdis author labels have known omission flicker (reports/v5/AUDIT_ONSET_V5.md), D-Fire has no event metadata, Boreal boxes are mask-derived -- 'fire rate on author-empty' therefore mixes teacher false alarms with author omissions and must not be read as an FP rate. RF-DETR has no NMS (duplicates inflate its box counts; cf. reports/v3/rfdetr_eval.md). DINO LoRA scores saturate >=0.99 (v4), so compare at high thresholds too.

## Files

- `models/v5b/teacher/dino_b4_lora_L__train.jsonl` -- one row per target image (`{image: LOCAL path, w, h, preds: [[x1n,y1n,x2n,y2n,score],...], teacher, label_type: model-assisted}`; unreadable images carry `error`); `missing__dino_b4_lora_L.txt` lists any target without a row.
- `models/v5b/teacher/dino_b6_lora_Hplus__train.jsonl` -- one row per target image (`{image: LOCAL path, w, h, preds: [[x1n,y1n,x2n,y2n,score],...], teacher, label_type: model-assisted}`; unreadable images carry `error`); `missing__dino_b6_lora_Hplus.txt` lists any target without a row.
- `models/v5b/teacher/rfdetr_L__train.jsonl` -- one row per target image (`{image: LOCAL path, w, h, preds: [[x1n,y1n,x2n,y2n,score],...], teacher, label_type: model-assisted}`; unreadable images carry `error`); `missing__rfdetr_L.txt` lists any target without a row.
- `models/v5b/teacher/teacher_pool_stats.json` -- all numbers above; `data/v5b/teacher/lists/` -- target lists; `cluster/v5b/{teacher_infer_entry.sh,submit_teacher_infer.py}`, `scripts/v5b/{teacher_pool_build_lists,build_teacher_pool,render_teacher_pool_md}.py`, `scripts/v5b/teacher_job.sh`.

## NEW-source extension for arm D2 -- generated 2026-10-02T09:33:32Z

**Model-assisted labels, same caveats as above.** Targets: the exact B-corpus TRAIN lists for the two new sources (`data/v5b/teacher/lists/train_new_sources.txt`, sha256 b4b463471f05475d; `TARGETS_NEW.json` leak check vs sdis val/test, D-Fire test, FireMan project val/holdout: 0 overlaps): FireMan burn-A project-TRAIN capped 600/video = 8,279 (author positives 6,707 / explicit negatives 1,572), Sheffield single session = 1,327 (positives 1,303 / explicit negatives 24). Images read from the arm-B stage on /scratch (`/local/home/tianrgua/cvproject/data/v5b/yolo/B_v1=/scratch/scratch/tianrgua-sandbox/smoke/v5bdata/B_v1`). Outputs: `models/v5b/teacher/<teacher>__train_new.jsonl` (image = local B_v1 path as listed).

| teacher | ckpt sha256 (pod-verified) | kraken job (wall) | rows / targets | unreadable | missing | fireman firing @0.5 / @0.9 (pos imgs) | fireman explicit-neg firing @0.5 / @0.9 | sheffield firing @0.5 / @0.9 (pos imgs) | sheffield explicit-neg firing @0.5 / @0.9 | author boxes matched IoU>=0.4 @0.5 (fireman / sheffield) |
|---|---|---|---|---|---|---|---|---|---|---|
| dino_b4_lora_L | 87a9f175effc183d (expect 87a9f175effc183d) | tianrgua-tchb4lnew-gyki SUCCEEDED 6 min | 9,606 / 9,606 | 0 | 0 | 6,670 / 6,640 (of 6,707) | 0 / 0 (of 1,572) | 1,303 / 1,265 (of 1,303) | 0 / 0 (of 24) | 0.9749 / 0.1052 |
| dino_b6_lora_Hplus | de235e28dd131019 (expect de235e28dd131019) | tianrgua-tchb6hnew-stqq SUCCEEDED 11 min | 9,606 / 9,606 | 0 | 0 | 6,662 / 6,645 (of 6,707) | 4 / 0 (of 1,572) | 1,303 / 1,280 (of 1,303) | 0 / 0 (of 24) | 0.9744 / 0.0982 |
| rfdetr_L | 2cca1fe8cd381b5d (expect 2cca1fe8cd381b5d) | tianrgua-tchrfdetrlnew-9drc SUCCEEDED 5 min | 9,606 / 9,606 | 0 | 0 | 6,441 / 2,788 (of 6,707) | 2 / 0 (of 1,572) | 1,223 / 512 (of 1,303) | 0 / 0 (of 24) | 0.943 / 0.0914 |

Image-level agreement (max score >= thr on both; n = images scored by both): dino_b4_lora_L|dino_b6_lora_Hplus@0.5: both 7,956, kappa 0.9889; dino_b4_lora_L|rfdetr_L@0.5: both 7,663, kappa 0.8926; dino_b6_lora_Hplus|rfdetr_L@0.5: both 7,663, kappa 0.894.

Used by: `scripts/v5b/build_corpus_v5b.py analyze-d2` / `build D2` (calibrated 3-teacher WBF, models/v5b/fusion/teacher_ensemble_d2.json).

## D3 extension -- UNLABELLED smoke-positive material (model-assisted; generated 2026-10-02T10:01:46Z)

Targets `data/v5b/teacher/lists/train_d3_sources.txt` (sha 745dd29eafc6fdfc; counts {"borealclip": 3370, "firesense": 530, "dfirefire": 944, "borealclip_no_smoke_clip_frames_scored_not_trained": 32, "borealclip_clips": {"smoke": 104, "no_smoke": 1}, "firesense_videos": 13, "total": 4844}); images staged from `data/v5b/stage/D3_src.tar` via DATA_TARS to `/scratch/scratch/tianrgua-sandbox/smoke/v5bdata/D3_src` (Boreal clip frames resized to max side 1920). No author boxes exist here; video-level labels only (D-Fire fire-only: smoke unknown).

| teacher | ckpt sha256 | job (wall) | rows / targets | unreadable | missing | borealclip firing @0.5 / @0.9 | firesense @0.5 / @0.9 | dfirefire @0.5 / @0.9 |
|---|---|---|---|---|---|---|---|---|
| dino_b4_lora_L | 87a9f175effc183d | tianrgua-tchb4ld3-fvig SUCCEEDED 3 min | 4,844 / 4,844 | 0 | 0 | 3,264 / 3,264 (of 3,370) | 504 / 441 (of 530) | 695 / 436 (of 944) |
| dino_b6_lora_Hplus | de235e28dd131019 | tianrgua-tchb6hd3-mt6q SUCCEEDED 7 min | 4,844 / 4,844 | 0 | 0 | 3,264 / 3,264 (of 3,370) | 471 / 438 (of 530) | 729 / 447 (of 944) |
| rfdetr_L | 2cca1fe8cd381b5d | tianrgua-tchrfdetrld3-i3h5 SUCCEEDED 2 min | 4,844 / 4,844 | 0 | 0 | 3,198 / 1,944 (of 3,370) | 337 / 83 (of 530) | 449 / 45 (of 944) |
