"""Freeze CORPUS_REP_v1 from reviewed scale (+pilot-c) i2i outputs.

Inclusion:
  accept_pos  -> image + ONE weak box label (class 0)
  accept_ctrl -> image + empty label (reviewed smoke-free; semantically a
                 true negative, unlike arm-B's invisible-ignore images)
  uncertain / reject_* -> EXCLUDED from supervision entirely (counted).

Weak box resolution (model-assisted, disclosed):
  blind stage-1 union box B and structural diff box D (1/8-scale change
  support). If both exist: IoU(B,D) >= 0.05 -> use B∪D; else trust neither
  -> demote to uncertain_box (excluded). If only B: use B. Boxes clipped to
  [0,1], must have 0.0004 <= area <= 0.70 else demoted.

Writes data/v6/synthREP/{images,labels}/ + manifest.jsonl + CORPUS_REP_v1.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/v6/synthREP"


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return i / (aa + bb - i)


def union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def resolve_box(rec):
    s1 = rec.get("stage1_blind") or {}
    bs = [b for b in (s1.get("boxes") or [])
          if isinstance(b, (list, tuple)) and len(b) == 4]
    B = None
    if bs:
        B = [min(b[0] for b in bs), min(b[1] for b in bs),
             max(b[2] for b in bs), max(b[3] for b in bs)]
    diff = rec.get("i2i_diff") or {}
    D = diff.get("diff_box")
    rerender = bool(diff.get("global_restyle"))
    if rerender:
        # whole scene re-rendered: the structural diff spans the frame and
        # carries no localization signal; the blind box is the only source
        box, src = (B, "blind_only_rerender") if B else (None, "none")
    elif B and D:
        box, src = (union(B, D), "blind_union_diff") if iou(B, D) >= 0.05 \
            else (None, "disagree")
    elif B:
        box, src = B, "blind_only"
    elif D:
        box, src = None, "diff_only"  # no independent smoke evidence
    else:
        box, src = None, "none"
    if box:
        box = [min(max(v, 0.0), 1.0) for v in box]
        area = (box[2] - box[0]) * (box[3] - box[1])
        if not (0.0004 <= area <= 0.70):
            box, src = None, f"{src}_bad_area"
    return box, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True,
                    help="review.jsonl:gen_dir pairs")
    args = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    man = (OUT / "manifest.jsonl").open("w")
    cnt = Counter()
    kept = []
    for spec in args.sets:
        rev, gdir = spec.split(":")
        gdir = ROOT / gdir
        for line in (ROOT / rev).open():
            r = json.loads(line)
            if r.get("error") or r.get("route") != "i2i":
                continue
            d = r.get("decision_v3") or r.get("decision")
            src_img = gdir / f"{r['hash']}.jpg"
            if not src_img.exists():
                continue
            rec = {"hash": r["hash"], "kind": r["kind"], "decision": d,
                   "bg": r.get("bg"), "visibility": r.get("visibility"),
                   "i2i_rerender": r.get("i2i_rerender"),
                   "review_src": rev, "gen_dir": str(gdir.name),
                   "label_source": None, "box": None}
            if d == "accept_pos":
                box, bsrc = resolve_box(r)
                rec["label_source"], rec["box"] = bsrc, box
                if not box:
                    cnt[f"excluded_pos_{bsrc}"] += 1
                    rec["decision"] = "uncertain_box"
                    man.write(json.dumps(rec) + "\n")
                    continue
                name = f"rep_{r['hash']}"
                shutil.copy2(src_img, OUT / "images" / f"{name}.jpg")
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                w, h = box[2] - box[0], box[3] - box[1]
                (OUT / "labels" / f"{name}.txt").write_text(
                    f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                cnt["pos"] += 1
                kept.append(name)
            elif d == "accept_ctrl":
                name = f"rep_{r['hash']}"
                shutil.copy2(src_img, OUT / "images" / f"{name}.jpg")
                (OUT / "labels" / f"{name}.txt").write_text("")
                cnt["neg_ctrl"] += 1
                kept.append(name)
            else:
                cnt[f"excluded_{d}"] += 1
            man.write(json.dumps(rec) + "\n")
    man.close()
    summary = {"corpus": "CORPUS_REP_v1", "counts": dict(cnt),
               "images": len(kept),
               "note": "labels are MODEL-ASSISTED weak boxes (blind review "
                       "∪ structural diff); uncertain/rejects excluded from "
                       "supervision, never auto-negative"}
    (OUT / "CORPUS_REP_v1.json").write_text(json.dumps(summary, indent=1))
    lst = OUT / "list_local.txt"
    lst.write_text("".join(f"{OUT}/images/{n}.jpg\n" for n in sorted(kept)))
    pod_root = "/scratch/scratch/tianrgua-sandbox/smoke/v6data/synthREP"
    (OUT / "list_pod.txt").write_text(
        "".join(f"{pod_root}/images/{n}.jpg\n" for n in sorted(kept)))
    print(json.dumps(summary, indent=1))
    print("list:", lst)


if __name__ == "__main__":
    main()
