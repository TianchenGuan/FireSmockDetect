"""V5b arm D: merge raw teacher shards -> models/v5b/teacher/<teacher>__train.jsonl
(v4 preds schema, LOCAL image paths) + TEACHER_POOL.md / teacher_pool_stats.json.

Everything produced here is MODEL-ASSISTED (teacher predictions). Nothing is
human-reviewed. "Author labels" below = the public datasets' own annotations
(pyro-sdis / D-Fire / Boreal YOLO txt), used only as a reference for agreement;
they are a separate annotation type and are never mixed into the teacher rows.

usage: python scripts/v5b/build_teacher_pool.py [--teacher name ...]
"""
from __future__ import annotations
import argparse, collections, gzip, json, statistics
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path("/local/home/tianrgua/cvproject")
LISTS = ROOT / "data/v5b/teacher/lists"
RAW = ROOT / "models/v5b/teacher/raw"
OUT = ROOT / "models/v5b/teacher"
TEACHERS = {  # teacher -> (raw dir prefix, checkpoint, infer config)
    "dino_b4_lora_L": ("b4L", "models/v3/runs/v4B4_lora_L/seg05/best.pt (pod runs_v3/v4B4_lora_L/best.pt, sha 87a9f175)", "dino_infer.py 1280x736 LoRA r16, conf floor 0.02"),
    "dino_b6_lora_Hplus": ("b6H", "models/v3/runs/v4B6_lora_Hplus/seg10/best.pt (pod runs_v3/v4B6_lora_Hplus/best.pt, sha de235e28)", "dino_infer.py 1280x736 LoRA r16, conf floor 0.02"),
    "rfdetr_L": ("rfdetrL", "models/v3/rfdetr/checkpoint_best_ema.pth (pod runs_v3/rfdetrL_1280_e48/, sha 2cca1fe8)", "rfdetr_infer.py imgsz 1280, conf floor 0.02 (no NMS)"),
}
POD2LOCAL = [("/scratch/scratch/tianrgua-sandbox/smoke_data_v2", str(ROOT / "data/yolo")),
             ("/scratch/scratch/tianrgua-sandbox/smoke/v4data/E_ET_v1", str(ROOT / "data/v4/yolo/E_ET_v1"))]
THRS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]
IOU_T = 0.4


def to_local(p):
    for s, d in POD2LOCAL:
        if p.startswith(s): return d + p[len(s):]
    return p

def source(p):
    return "boreal" if "/E_ET_v1/" in p else ("pyro_sdis" if "/pyro_sdis/" in p else "dfire")

def author_boxes(p):
    L = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
    if not L.exists(): return []
    out = []
    for r in L.read_text().splitlines():
        f = r.split()
        if len(f) < 5: continue
        cx, cy, w, h = map(float, f[1:5]); out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return out

def iou_mat(a, b):
    if not a or not b: return np.zeros((len(a), len(b)))
    a, b = np.asarray(a)[:, :4], np.asarray(b)[:, :4]
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)

def matched(a, b):
    """greedy one-to-one matches at IoU>=IOU_T -> (n_a_matched, n_b_matched)"""
    m = iou_mat(a, b)
    if m.size == 0: return 0, 0
    used_b, na = set(), 0
    for i in np.argsort(-m.max(axis=1) if m.shape[1] else []):
        order = np.argsort(-m[i])
        for j in order:
            if m[i, j] < IOU_T: break
            if j not in used_b: used_b.add(j); na += 1; break
    return na, len(used_b)

def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool); n = len(a)
    po = np.mean(a == b); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def load_teacher(prefix, targets):
    rows, srcfiles = {}, []
    for d in sorted(RAW.glob(prefix + "*")):
        for f in sorted(d.glob("*_preds.jsonl.gz")):
            n = 0
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    if not line.strip(): continue
                    r = json.loads(line); r["image"] = to_local(r["image"]); n += 1
                    rows[r["image"]] = r          # later dirs (retries) override
            srcfiles.append((str(f.relative_to(ROOT)), n))
    return rows, srcfiles


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--teacher", action="append", default=None); a = ap.parse_args()
    teachers = a.teacher or list(TEACHERS)
    targets = [l.strip() for l in (LISTS / "train_all.txt").read_text().splitlines() if l.strip()]
    tset = set(targets); tman = json.loads((LISTS / "TARGETS.json").read_text())
    src_of = {p: source(p) for p in targets}
    auth = {p: author_boxes(p) for p in targets}
    stats = {"generated_utc": datetime.now(timezone.utc).strftime("%FT%TZ"), "label_type": "model-assisted (teacher predictions); NOT human-reviewed; NOT author labels",
             "targets": tman["counts"], "iou_match": IOU_T, "thresholds": THRS, "teachers": {}, "agreement": {}, "author_label_reference": {}}
    maxs, boxes_at = {}, {}
    for t in teachers:
        prefix, ck, cfg = TEACHERS[t]
        rows, srcfiles = load_teacher(prefix, targets)
        extra = [p for p in rows if p not in tset]
        assert not extra, f"{t}: {len(extra)} rows for NON-target images, e.g. {extra[:2]} -- refusing"
        missing = [p for p in targets if p not in rows]
        errs = [p for p in targets if p in rows and "error" in rows[p]]
        outp = OUT / f"{t}__train.jsonl"
        with outp.open("w") as fh:
            for p in targets:
                if p not in rows: continue
                r = rows[p]
                rec = {"image": p, "teacher": t, "label_type": "model-assisted"}
                if "error" in r: rec["error"] = r["error"]
                else: rec.update(w=r.get("w"), h=r.get("h"), preds=r.get("preds", []))
                fh.write(json.dumps(rec) + "\n")
        (OUT / f"missing__{t}.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
        ok = [p for p in targets if p in rows and "error" not in rows[p]]
        ms = {p: (max((b[4] for b in rows[p]["preds"]), default=0.0) if p in rows and "error" not in rows[p] else None) for p in targets}
        maxs[t] = ms
        boxes_at[t] = {p: rows[p]["preds"] if p in rows and "error" not in rows[p] else [] for p in targets}
        allscores = np.array([b[4] for p in ok for b in rows[p]["preds"]])
        nbox = np.array([len(rows[p]["preds"]) for p in ok])
        bysrc = collections.Counter(src_of[p] for p in ok)
        hist_edges = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0001]
        mx = np.array([ms[p] for p in ok])
        st = {"checkpoint": ck, "inference": cfg, "raw_files": srcfiles,
              "coverage": {"targets": len(targets), "rows": len([p for p in targets if p in rows]), "scored_ok": len(ok), "unreadable_errors": len(errs),
                           "missing": len(missing), "scored_by_source": dict(bysrc), "complete": not missing and not errs},
              "boxes": {"total_boxes_ge_floor": int(nbox.sum()), "boxes_per_image_mean": float(nbox.mean()) if len(nbox) else None,
                        "boxes_per_image_p50_p90_max": [float(np.percentile(nbox, 50)), float(np.percentile(nbox, 90)), int(nbox.max())] if len(nbox) else None,
                        "score_quantiles_p10_p50_p90_p99": [float(np.percentile(allscores, q)) for q in (10, 50, 90, 99)] if len(allscores) else None},
              "max_score_hist": {"edges": hist_edges, "none(no box>=0.02)": int((mx == 0).sum()),
                                 "counts": np.histogram(mx[mx > 0], bins=hist_edges)[0].tolist()},
              "images_firing_at_thr": {str(th): {"all": int((mx >= th).sum()), **{s: int(sum(1 for p in ok if ms[p] >= th and src_of[p] == s)) for s in ("pyro_sdis", "dfire", "boreal")}} for th in THRS}}
        # reference: agreement with AUTHOR labels (dataset annotations; not our review)
        ref = {}
        apos = [p for p in ok if auth[p]]; aneg = [p for p in ok if not auth[p]]
        for th in THRS:
            tp_img = sum(1 for p in apos if ms[p] >= th); fp_img = sum(1 for p in aneg if ms[p] >= th)
            na = nm = ntb = ntm = 0
            for p in ok:
                tb = [b for b in boxes_at[t][p] if b[4] >= th]
                if auth[p] or tb:
                    m1, m2 = matched(auth[p], tb); na += len(auth[p]); nm += m1; ntb += len(tb); ntm += m2
            ref[str(th)] = {"img_recall_author_pos": round(tp_img / max(len(apos), 1), 4), "img_fire_rate_author_empty": round(fp_img / max(len(aneg), 1), 4),
                            "box_recall_author": round(nm / max(na, 1), 4), "teacher_boxes": ntb, "teacher_boxes_matching_author": ntm,
                            "teacher_box_precision_vs_author": round(ntm / max(ntb, 1), 4)}
        st["vs_author_labels"] = {"author_positive_images": len(apos), "author_empty_images": len(aneg), "author_boxes": sum(len(auth[p]) for p in ok), "at_thr": ref}
        stats["teachers"][t] = st
        print(f"{t}: rows {st['coverage']['rows']}/{len(targets)} ok {len(ok)} err {len(errs)} missing {len(missing)} -> {outp.relative_to(ROOT)}")
    # pairwise + 3-way agreement on images scored by all teachers involved
    done = [t for t in teachers if stats["teachers"][t]["coverage"]["scored_ok"] > 0]
    for i, t1 in enumerate(done):
        for t2 in done[i + 1:]:
            common = [p for p in targets if maxs[t1][p] is not None and maxs[t2][p] is not None]
            pair = {"n_common": len(common), "at_thr": {}}
            for th in THRS:
                f1 = [maxs[t1][p] >= th for p in common]; f2 = [maxs[t2][p] >= th for p in common]
                both = sum(x and y for x, y in zip(f1, f2)); o1 = sum(x and not y for x, y in zip(f1, f2)); o2 = sum(y and not x for x, y in zip(f1, f2))
                neither = len(common) - both - o1 - o2
                n1 = n1m = n2 = n2m = 0
                for p in common:
                    b1 = [b for b in boxes_at[t1][p] if b[4] >= th]; b2 = [b for b in boxes_at[t2][p] if b[4] >= th]
                    if b1 or b2:
                        m1, m2 = matched(b1, b2); n1 += len(b1); n1m += m1; n2 += len(b2); n2m += m2
                pair["at_thr"][str(th)] = {"both_fire": both, f"only_{t1}": o1, f"only_{t2}": o2, "neither": neither,
                                           "image_kappa": round(kappa(f1, f2), 4), "image_jaccard": round(both / max(both + o1 + o2, 1), 4),
                                           f"box_frac_{t1}_matched_by_{t2}": round(n1m / max(n1, 1), 4), f"box_frac_{t2}_matched_by_{t1}": round(n2m / max(n2, 1), 4)}
            stats["agreement"][f"{t1}|{t2}"] = pair
    if len(done) == 3:
        common = [p for p in targets if all(maxs[t][p] is not None for t in done)]
        tri = {"n_common": len(common), "at_thr": {}}
        for th in THRS:
            k = collections.Counter(sum(maxs[t][p] >= th for t in done) for p in common)
            tri["at_thr"][str(th)] = {"fire_3of3": k[3], "fire_2of3": k[2], "fire_1of3": k[1], "fire_0of3": k[0],
                                      "by_source_3of3": {s: sum(1 for p in common if src_of[p] == s and all(maxs[t][p] >= th for t in done)) for s in ("pyro_sdis", "dfire", "boreal")}}
        stats["agreement"]["3-way"] = tri
    (OUT / "teacher_pool_stats.json").write_text(json.dumps(stats, indent=1))
    print("-> models/v5b/teacher/teacher_pool_stats.json")


if __name__ == "__main__":
    main()
