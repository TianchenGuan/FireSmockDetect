"""V5b model-collaboration study on pyro-sdis VAL from SAVED predictions (no inference).

Grouped 2-fold by SITE (cabanelle = fold 0, marguerite = fold 1).  Every
calibration map, fusion hyper-parameter and learned-fusion weight is fit on one
fold and applied to the other; all headline numbers are pooled OUT-OF-FOLD.
Matching = src.eval.eval_detection.sweep_report semantics (IoU>=0.4, greedy by
score), re-implemented as a one-pass matcher (verified identical) so hundreds
of fusion configurations can be swept.  Writes
  reports/v5b/ensemble_dev_v5b.json, reports/v5b/ensemble_dev_tables.md,
  models/v5b/fusion/accurate_recipe.json
Usage: python scripts/v5b/ensemble_dev.py [--workers 24] [--quick]
"""
from __future__ import annotations

import argparse, itertools, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.eval_detection import SIZE_BUCKETS  # noqa: E402
from scripts.v4.compare_matched_fp import interp_at_fp  # noqa: E402
from scripts.v5b.fusion import (ScoreCalibrator, iou_matrix, union_nms, wbf,  # noqa: E402
                                FusionRecipe, fuse, image_score)

MODELS = {  # label -> (parquet, checkpoint, family, latency_ms_L40S_batch1, latency source)
    "yolo_A":   ("reports/v4/eval_v4a_s2_sdisval.raw.parquet", "models/trained/expV4A_ctrl_s2/seg05/best.pt",
                 "YOLO11m@1280 (v4 arm A, E corpus, seed 2)", 11.2, "models/v3/evals/yolo_latency/latency_gpu.json (expE weights, same arch)"),
    "dino_B4":  ("reports/v4/eval_v4B4_lora_L_sdis_val.raw.parquet", "models/v3/runs/v4B4_lora_L/seg05/best.pt",
                 "DINOv3 ViT-L LoRA r16 Faster R-CNN (v4 B4, E+T corpus)", 157.5, "FINAL_REPORT_V4 T13, LoRA merged"),
    "dino_B6":  ("reports/v4/eval_v4B6_lora_Hplus_sdis_val.raw.parquet", "models/v3/runs/v4B6_lora_Hplus/seg10/best.pt",
                 "DINOv3 ViT-H+ LoRA r16 Faster R-CNN (v4 B6, E+T corpus)", 351.6, "FINAL_REPORT_V4 T13, LoRA merged"),
    "rfdetr_L": ("reports/v3/eval_rfdetrL_e48_val.raw.parquet", "models/v3/rfdetr/checkpoint_best_ema.pth",
                 "RF-DETR-L@1280 e48 (v3, expB list)", 31.4, "models/v3/evals/rfdetrLe48lat/latency_gpu.json"),
    "yolo_expE": ("reports/eval_expE_val.raw.parquet", "models/trained/expE_mix_y11m_1280_v2/seg03/best.pt",
                  "YOLO11m@1280 (iter-2 expE, expB list, 60 ep)", 11.2, "models/v3/evals/yolo_latency/latency_gpu.json"),
    "dino_B1":  ("reports/v4/eval_v4B1_frozen_L_sdis_val.raw.parquet", "models/v3/runs/v4B1_frozen_L/seg03/best.pt",
                 "DINOv3 ViT-L frozen Faster R-CNN (v4 B1, E+T corpus)", 156.6, "FINAL_REPORT_V4 T13 (v3 B1 frozen reference)"),
}
FP_TARGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
FOLDS = {0: "cabanelle", 1: "marguerite"}
GRID = np.unique(np.concatenate([np.linspace(0, 1, 1001), np.linspace(0.99, 1, 201),
                                 np.linspace(0.999, 1, 101)])).round(6)
MATCH_IOU = 0.4

# ---------------------------------------------------------------- data (globals for fork workers)
D: dict = {}


def load_data():
    ref = pd.read_parquet(ROOT / MODELS["yolo_A"][0]).sort_values("image").reset_index(drop=True)
    images = ref["image"].tolist()
    fold = np.array([0 if "cabanelle" in p else 1 if "marguerite" in p else -1 for p in images])
    assert (fold >= 0).all()
    gts = [np.asarray([list(g) for g in g_], float).reshape(-1, 5) for g_ in ref["gt"]]
    preds = {}
    for m, (pq, *_rest) in MODELS.items():
        df = pd.read_parquet(ROOT / pq).sort_values("image").reset_index(drop=True)
        assert df["image"].tolist() == images, m
        for a, b in zip(df["gt"], ref["gt"]):
            assert len(a) == len(b) and (len(a) == 0 or np.allclose(np.stack(list(a)), np.stack(list(b)))), m
        arrs = []
        for p in df["preds"]:
            a = np.asarray([list(x) for x in p], float).reshape(-1, 5)
            if len(a):
                a[:, :4] = np.clip(a[:, :4], 0, 1)
            arrs.append(a)
        preds[m] = arrs
    is_neg = np.array([len(g) == 0 for g in gts])
    D.update(images=images, fold=fold, gts=gts, preds=preds, is_neg=is_neg)
    bucket = lambda a: next(i for i, (_, lo, hi) in enumerate(SIZE_BUCKETS) if lo <= a < hi)
    gt_n = np.zeros((2, 4), int)
    for f, g in zip(fold, gts):
        for a in g[:, 4]:
            gt_n[f, bucket(a)] += 1
    D["gt_n"] = gt_n
    return D


# ---------------------------------------------------------------- matching tables (one pass, == sweep_report)
def tables(fused: list[np.ndarray]) -> dict:
    sc, tp, bk, fb, bi = [], [], [], [], []
    neg_max, neg_fold, neg_img = [], [], []
    for i, p in enumerate(fused):
        g = D["gts"][i]
        if len(g) == 0:
            neg_max.append(p[:, 4].max() if len(p) else 0.0); neg_fold.append(D["fold"][i]); neg_img.append(i)
        if len(p) == 0:
            continue
        p = p[np.argsort(-p[:, 4], kind="stable")]
        ious = iou_matrix(p[:, :4], g[:, :4]) if len(g) else np.zeros((len(p), 0))
        matched = np.zeros(len(g), bool)
        for r in range(len(p)):
            if len(g):
                cand = np.where(~matched, ious[r], -1.0)
                j = int(np.argmax(cand)); best = cand[j]
            else:
                j, best = -1, 0.0
            sc.append(p[r, 4]); fb.append(D["fold"][i]); bi.append(i)
            if best >= MATCH_IOU and best > 0:
                matched[j] = True; tp.append(True)
                a = g[j, 4]; bk.append(next(b for b, (_, lo, hi) in enumerate(SIZE_BUCKETS) if lo <= a < hi))
            else:
                tp.append(False); bk.append(-1)
    return {"sc": np.array(sc), "tp": np.array(tp, bool), "bk": np.array(bk, int), "fb": np.array(fb, int),
            "bi": np.array(bi, int), "neg_max": np.array(neg_max), "neg_fold": np.array(neg_fold, int),
            "neg_img": np.array(neg_img, int), "gt_n": D["gt_n"].copy()}


def subset(t: dict, folds: list[int]) -> dict:
    mb = np.isin(t["fb"], folds); mn = np.isin(t["neg_fold"], folds)
    gt_n = t["gt_n"].copy(); gt_n[[f for f in (0, 1) if f not in folds]] = 0
    return {"sc": t["sc"][mb], "tp": t["tp"][mb], "bk": t["bk"][mb], "fb": t["fb"][mb], "bi": t["bi"][mb],
            "neg_max": t["neg_max"][mn], "neg_fold": t["neg_fold"][mn], "neg_img": t["neg_img"][mn], "gt_n": gt_n}


def concat(a: dict, b: dict) -> dict:
    return {k: np.concatenate([a[k], b[k]]) if k != "gt_n" else a[k] + b[k] for k in a}


def curve(t: dict, grid=GRID) -> list[dict]:
    order = np.argsort(-t["sc"], kind="stable")
    s = t["sc"][order]; tp = t["tp"][order]; bk = t["bk"][order]
    ctp = np.concatenate([[0], np.cumsum(tp)]); cfp = np.concatenate([[0], np.cumsum(~tp)])
    cb = [np.concatenate([[0], np.cumsum(tp & (bk == i))]) for i in range(4)]
    negs = np.sort(t["neg_max"]); n_neg = len(negs)
    gt_tot = t["gt_n"].sum(0); G = gt_tot.sum()
    ks = np.searchsorted(-s, -grid, side="right")
    nneg_ge = n_neg - np.searchsorted(negs, grid, side="left")
    out = []
    for gi, k in enumerate(ks):
        T, F = int(ctp[k]), int(cfp[k])
        out.append({"conf": float(grid[gi]),
                    "precision": round(T / (T + F), 4) if T + F else None,
                    "recall": round(T / G, 4) if G else None,
                    "neg_image_fp_rate": round(nneg_ge[gi] / n_neg, 4) if n_neg else None,
                    "recall_by_size": {n: (round(cb[i][k] / gt_tot[i], 4) if gt_tot[i] else None)
                                       for i, (n, _, _) in enumerate(SIZE_BUCKETS)}})
    return out


def matched(t: dict) -> dict:
    sw = curve(t)
    res = {}
    for p in FP_TARGETS:
        res[f"fp{p:.2f}"] = {"thr": interp_at_fp(sw, p, lambda s: s["conf"]),
                             "recall": interp_at_fp(sw, p, lambda s: s["recall"]),
                             "tiny": interp_at_fp(sw, p, lambda s: s["recall_by_size"]["tiny"]),
                             "small": interp_at_fp(sw, p, lambda s: s["recall_by_size"]["small"]),
                             "precision": interp_at_fp(sw, p, lambda s: s["precision"])}
    rs = [res[f"fp{p:.2f}"]["recall"] for p in FP_TARGETS]
    res["mean_recall"] = round(float(np.mean([r if r is not None else 0.0 for r in rs])), 4)
    res["n_neg"] = int(len(t["neg_max"])); res["n_gt"] = int(t["gt_n"].sum())
    return res


def mean_recall_w(t: dict, w_img: np.ndarray, fold: int) -> float:
    """Mean recall over FP_TARGETS for one fold's table under per-image resampling weights."""
    wb = w_img[t["bi"]]; wn = w_img[t["neg_img"]]
    order = np.argsort(-t["sc"], kind="stable"); s = t["sc"][order]; tp = t["tp"][order]; wbo = wb[order]
    ctp = np.concatenate([[0], np.cumsum(wbo * tp)])
    G = sum(w_img[i] * len(D["gts"][i]) for i in np.where(D["fold"] == fold)[0])
    negs_o = np.argsort(t["neg_max"]); negs = t["neg_max"][negs_o]; cwn = np.concatenate([[0], np.cumsum(wn[negs_o])])
    n_neg = wn.sum()
    ks = np.searchsorted(-s, -GRID, side="right"); kn = np.searchsorted(negs, GRID, side="left")
    rec = ctp[ks] / G; fp = (n_neg - cwn[kn]) / n_neg
    out = []
    for p in FP_TARGETS:   # linear interpolation of recall at the FP target along the (fp, rec) curve
        if p > fp.max() or p < fp.min():
            out.append(0.0); continue
        j = np.where((fp[:-1] >= p) & (fp[1:] <= p))[0]
        if len(j) == 0:
            out.append(float(rec[np.argmin(np.abs(fp - p))])); continue
        j = j[0]; f1, f2, r1, r2 = fp[j], fp[j + 1], rec[j], rec[j + 1]
        out.append(float(r1 if f1 == f2 else r1 + (p - f1) / (f2 - f1) * (r2 - r1)))
    return float(np.mean(out))


def bootstrap_diff(ta: dict, tb: dict, fold: int, n_boot: int = 500, seed: int = 0) -> dict:
    """95% percentile CI of mean_recall(ta) - mean_recall(tb), resampling IMAGES of `fold` with replacement."""
    rng = np.random.default_rng(seed)
    idx = np.where(D["fold"] == fold)[0]
    n_img = len(D["images"])
    diffs = []
    for _ in range(n_boot):
        w = np.zeros(n_img); draw = rng.choice(idx, size=len(idx), replace=True)
        np.add.at(w, draw, 1.0)
        diffs.append(mean_recall_w(ta, w, fold) - mean_recall_w(tb, w, fold))
    d = np.array(diffs)
    w1 = np.ones(n_img)
    return {"point_pts": round((mean_recall_w(ta, w1, fold) - mean_recall_w(tb, w1, fold)) * 100, 2),
            "ci95_pts": [round(float(np.percentile(d, 2.5)) * 100, 2), round(float(np.percentile(d, 97.5)) * 100, 2)],
            "p_gain_le_0": round(float((d <= 0).mean()), 3), "n_boot": n_boot}


# ---------------------------------------------------------------- calibration
def fit_calibrators(fit_fold: int | None) -> dict[str, ScoreCalibrator]:
    idx = np.where(D["is_neg"] & ((D["fold"] == fit_fold) if fit_fold is not None else True))[0]
    out = {}
    for m, arrs in D["preds"].items():
        nm = [arrs[i][:, 4].max() if len(arrs[i]) else 0.0 for i in idx]
        out[m] = ScoreCalibrator.fit(nm, {"fit_fold": FOLDS.get(fit_fold, "all_val")})
    return out


CAL: dict = {}  # (fit_fold) -> {model: calibrated preds list}
OOF_TBL: dict = {}  # (models, method) -> {eval_fold: OOF table of the fold-selected configuration}


def prepare_calibrated():
    for ff in (0, 1, None):
        cals = fit_calibrators(ff)
        CAL[ff] = {}
        for m, arrs in D["preds"].items():
            lst = []
            for a in arrs:
                b = a.copy()
                if len(b):
                    b[:, 4] = cals[m](b[:, 4])
                lst.append(b)
            CAL[ff][m] = lst
        CAL[("calibrators", ff)] = cals


def fuse_all(models: tuple[str, ...], method: str, params: dict, calib_fold) -> list[np.ndarray]:
    srcs = [CAL[calib_fold][m] for m in models]
    n = len(D["images"]); out = []
    for i in range(n):
        per = [s[i] for s in srcs]
        if method == "union":
            out.append(union_nms(per, iou_thr=params["iou_thr"]))
        elif method == "wbf":
            out.append(wbf(per, iou_thr=params["iou_thr"], skip_box_thr=params["skip_box_thr"],
                           conf_type=params["conf_type"]))
        elif method == "single":
            out.append(per[0])
        else:
            raise ValueError(method)
    return out


def _job(args):
    models, method, params = args
    res = {}
    for calib_fold in (0, 1):
        t = tables(fuse_all(models, method, params, calib_fold))
        res[calib_fold] = t
    return (models, method, json.dumps(params, sort_keys=True)), res


def evaluate_configs(jobs: list, workers: int) -> dict:
    """For each (models, method, params): in-sample objective per fit fold and OOF tables."""
    out = {}
    with Pool(workers) as pool:
        for key, res in pool.imap_unordered(_job, jobs, chunksize=1):
            # res[c] = tables for ALL images with calibration fit on fold c
            fit_obj = {c: matched(subset(res[c], [c])) for c in (0, 1)}          # in-sample on fold c
            oof_tbl = {c: subset(res[1 - c], [c]) for c in (0, 1)}                # eval fold c, calib from other
            out[key] = {"fit": fit_obj, "oof_tbl": oof_tbl}
    return out


def select_and_score(cfgs: dict, models: tuple, method: str) -> dict:
    """Per fold: pick params by in-sample objective on the OTHER fold, evaluate on this fold; pool."""
    keys = [k for k in cfgs if k[0] == models and k[1] == method]
    chosen = {}
    for c in (0, 1):
        other = 1 - c
        best = max(keys, key=lambda k: cfgs[k]["fit"][other]["mean_recall"])
        chosen[c] = best
    pooled = concat(cfgs[chosen[0]]["oof_tbl"][0], cfgs[chosen[1]]["oof_tbl"][1])
    both = max(keys, key=lambda k: cfgs[k]["fit"][0]["mean_recall"] + cfgs[k]["fit"][1]["mean_recall"])
    OOF_TBL[(models, method)] = {c: cfgs[chosen[c]]["oof_tbl"][c] for c in (0, 1)}
    by_fold = {FOLDS[c]: matched(cfgs[chosen[c]]["oof_tbl"][c]) for c in (0, 1)}
    return {"method": method, "models": list(models),
            "params_by_eval_fold": {FOLDS[c]: json.loads(chosen[c][2]) for c in (0, 1)},
            "params_all_val": json.loads(both[2]),
            "oof_pooled": matched(pooled),
            "oof_by_fold": by_fold,
            "oof_macro_mean": round(float(np.mean([v["mean_recall"] for v in by_fold.values()])), 4),
            "n_configs": len(keys)}


# ---------------------------------------------------------------- learned fusion
def anchor_features(models: tuple, calib_fold, iou_anchor=0.5):
    """Anchors = union_nms(0.5) of calibrated boxes; features per anchor; label = IoU>=0.4 with any GT."""
    feats, labels, folds, boxes, img_idx = [], [], [], [], []
    srcs = [CAL[calib_fold][m] for m in models]
    raw = [D["preds"][m] for m in models]
    for i in range(len(D["images"])):
        per = [s[i] for s in srcs]
        anc = union_nms(per, iou_thr=iou_anchor)
        if len(anc) == 0:
            continue
        g = D["gts"][i]
        lab = (iou_matrix(anc[:, :4], g[:, :4]).max(1) >= MATCH_IOU) if len(g) else np.zeros(len(anc), bool)
        cols = []
        for k in range(len(models)):
            c = per[k]; r = raw[k][i]
            if len(c):
                io = iou_matrix(anc[:, :4], c[:, :4]); hit = io >= iou_anchor
                cmax = np.where(hit, c[:, 4][None, :], 0.0).max(1)
                rmax = np.where(hit, r[:, 4][None, :], 0.0).max(1)
            else:
                cmax = np.zeros(len(anc)); rmax = np.zeros(len(anc))
            cols += [cmax, rmax]
        C = np.stack(cols, 1)
        cal_cols = C[:, 0::2]
        logfp = -np.log10(np.clip(1.0 - cal_cols, 1e-4, 1.0))  # 'FP decades' per model (0 = absent)
        w = anc[:, 2] - anc[:, 0]; h = anc[:, 3] - anc[:, 1]
        area = w * h
        X = np.column_stack([C, logfp, logfp.max(1), logfp.sum(1), (cal_cols > 0).sum(1), cal_cols.mean(1), cal_cols.max(1),
                             np.log10(area + 1e-7), np.log(w / (h + 1e-7) + 1e-7), area < 0.001])
        feats.append(X); labels.append(lab); folds.append(np.full(len(anc), D["fold"][i]))
        boxes.append(anc[:, :4]); img_idx.append(np.full(len(anc), i))
    return (np.concatenate(feats), np.concatenate(labels), np.concatenate(folds),
            np.concatenate(boxes), np.concatenate(img_idx))


def learned_fusion(models: tuple) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    out = {}
    fits = {c: anchor_features(models, c) for c in (0, 1)}  # calibration fit on fold c
    for name, mk in {"logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                     "hgb": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3,
                                                                  min_samples_leaf=50, random_state=0)}.items():
        pooled = None
        per_fold = {}
        for c in (0, 1):  # eval fold c; train on fold 1-c with calibration fit on 1-c
            Xtr, ytr, ftr, _, _ = fits[1 - c]
            tr = ftr == (1 - c)
            clf = mk().fit(Xtr[tr], ytr[tr])
            Xte, yte, fte, bte, ite = fits[1 - c]  # OOF: calibration from 1-c applied to fold c anchors
            te = fte == c
            prob = clf.predict_proba(Xte[te])[:, 1]
            fused = [np.zeros((0, 5)) for _ in D["images"]]
            for ii in np.unique(ite[te]):
                sel = ite[te] == ii
                fused[ii] = np.column_stack([bte[te][sel], prob[sel]])
            t = subset(tables(fused), [c])
            per_fold[FOLDS[c]] = matched(t)
            pooled = t if pooled is None else concat(pooled, t)
        out[name] = {"oof_pooled": matched(pooled), "oof_by_fold": per_fold,
                     "oof_macro_mean": round(float(np.mean([v["mean_recall"] for v in per_fold.values()])), 4),
                     "features": "per model: calibrated max, raw max, -log10(1-cal) ; max/sum log-FP ; agreement count ; mean/max cal ; log10 area ; log aspect ; tiny flag",
                     "n_train_anchors": {FOLDS[c]: int((fits[c][2] == c).sum()) for c in (0, 1)},
                     "pos_rate": {FOLDS[c]: round(float(fits[c][1][fits[c][2] == c].mean()), 4) for c in (0, 1)}}
    return out


# ---------------------------------------------------------------- image level
def recall_at_fp_exact(pos: np.ndarray, neg: np.ndarray, p: float) -> tuple[float, float]:
    """Largest recall with neg-image FP <= p: thr = (k+1)-th largest neg score, k = floor(p*n_neg); count > thr."""
    k = int(np.floor(p * len(neg)))
    negd = np.sort(neg)[::-1]
    thr = negd[k] if k < len(negd) else -np.inf
    return float((pos > thr).mean()), float(thr)


def image_level(models_all: tuple, chosen: tuple, wbf_params: dict) -> dict:
    from sklearn.linear_model import LogisticRegression
    is_neg = D["is_neg"]; fold = D["fold"]
    n = len(D["images"])
    def maxes(calib_fold, ms):
        return np.stack([[CAL[calib_fold][m][i][:, 4].max() if len(CAL[calib_fold][m][i]) else 0.0 for i in range(n)]
                         for m in ms], 1)
    raw = np.stack([[D["preds"][m][i][:, 4].max() if len(D["preds"][m][i]) else 0.0 for i in range(n)]
                    for m in models_all], 1)
    res = {}
    def report(score):  # pooled OOF scores (len n) -> recall at FP levels (pooled + by fold)
        r = {}
        for p in FP_TARGETS:
            rec, thr = recall_at_fp_exact(score[~is_neg], score[is_neg], p)
            r[f"fp{p:.2f}"] = {"recall": round(rec, 4), "thr": round(thr, 4)}
            for c in (0, 1):
                m = fold == c
                rc, _ = recall_at_fp_exact(score[~is_neg & m], score[is_neg & m], p)
                r[f"fp{p:.2f}"][FOLDS[c]] = round(rc, 4)
        r["mean_recall"] = round(float(np.mean([r[f"fp{p:.2f}"]["recall"] for p in FP_TARGETS])), 4)
        return r
    for k, m in enumerate(models_all):
        res[f"single:{m}"] = report(raw[:, k])
    for label, ms in (("chosen", chosen), ("all6", models_all)):
        oof = np.zeros((n, len(ms)))
        for c in (0, 1):
            oof[fold == c] = maxes(1 - c, ms)[fold == c]
        res[f"max_cal[{label}]"] = report(oof.max(1))
        res[f"mean_cal[{label}]"] = report(oof.mean(1))
        res[f"noisy_or[{label}]"] = report(1 - np.prod(1 - oof, 1))
        lr = np.zeros(n)
        for c in (0, 1):
            Xtr = maxes(1 - c, ms)[fold == 1 - c]; ytr = ~is_neg[fold == 1 - c]
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
            lr[fold == c] = clf.predict_proba(oof[fold == c])[:, 1]
        res[f"logreg_cal[{label}]"] = report(lr)
    fb = np.zeros(n)
    for c in (0, 1):
        fused = fuse_all(chosen, "wbf", wbf_params, 1 - c)
        for i in np.where(fold == c)[0]:
            fb[i] = fused[i][:, 4].max() if len(fused[i]) else 0.0
    res["fused_box_max[chosen,wbf]"] = report(fb)
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-json", default="reports/v5b/ensemble_dev_v5b.json")
    ap.add_argument("--recipe", default="models/v5b/fusion/accurate_recipe.json")
    args = ap.parse_args()
    t0 = time.time()
    load_data(); prepare_calibrated()
    print(f"loaded {len(D['images'])} imgs, neg {int(D['is_neg'].sum())}, folds "
          f"{ {FOLDS[c]: int((D['fold']==c).sum()) for c in (0,1)} }, gt_n {D['gt_n'].tolist()}", flush=True)
    R: dict = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "fp_targets": FP_TARGETS,
               "folds": {FOLDS[c]: {"n_images": int((D["fold"] == c).sum()), "n_neg": int((D["is_neg"] & (D["fold"] == c)).sum()),
                                    "n_gt": int(D["gt_n"][c].sum()), "n_gt_tiny": int(D["gt_n"][c][0])} for c in (0, 1)},
               "match_iou": MATCH_IOU, "grid_points": int(len(GRID))}

    # 1. single models: raw score (standard) and OOF-calibrated pooled
    R["single_raw"] = {}; R["single_cal_oof"] = {}
    for m in MODELS:
        R["single_raw"][m] = matched(tables(D["preds"][m]))
        R["single_raw"][m]["by_fold"] = {FOLDS[c]: matched(subset(tables(D["preds"][m]), [c])) for c in (0, 1)}
        tt = {c: subset(tables(CAL[1 - c][m]), [c]) for c in (0, 1)}
        R["single_cal_oof"][m] = matched(concat(tt[0], tt[1]))
    print("single models done", round(time.time() - t0), "s", flush=True)

    # 2./3. union and WBF sweeps with the model-addition protocol
    union_grid = [{"iou_thr": v} for v in (0.3, 0.4, 0.5, 0.6, 0.7)]
    if args.quick:
        wbf_grid = [{"iou_thr": i, "skip_box_thr": s, "conf_type": c} for i, s, c in
                    itertools.product((0.4, 0.55), (0.0, 0.8), ("avg", "max"))]
    else:
        wbf_grid = [{"iou_thr": i, "skip_box_thr": s, "conf_type": c} for i, s, c in
                    itertools.product((0.3, 0.4, 0.5, 0.6, 0.7), (0.0, 0.5, 0.8, 0.9),
                                      ("avg", "max", "box_and_model_avg", "absent_model_aware_avg"))]
    R["sweep_grids"] = {"union": union_grid, "wbf": wbf_grid,
                        "objective": "mean recall over FP targets, in-sample on the fitting fold"}
    current = ("yolo_A", "dino_B4")
    R["addition_protocol"] = []
    R["sets"] = {}
    def run_set(ms):
        jobs = [(ms, "union", p) for p in union_grid] + [(ms, "wbf", p) for p in wbf_grid]
        cfgs = evaluate_configs(jobs, args.workers)
        res = {"union": select_and_score(cfgs, ms, "union"), "wbf": select_and_score(cfgs, ms, "wbf")}
        res["wbf"]["all_configs_fit_obj"] = {k[2]: {FOLDS[c]: cfgs[k]["fit"][c]["mean_recall"] for c in (0, 1)}
                                             for k in cfgs if k[1] == "wbf"}
        return res
    R["sets"]["+".join(current)] = run_set(current)
    base = R["sets"]["+".join(current)]
    best_mean = max(base["wbf"]["oof_macro_mean"], base["union"]["oof_macro_mean"])
    R["addition_criterion"] = ("fold-averaged (macro) OOF mean recall over FP targets 1/2/5/10/20%, best of union/WBF; "
                               "keep candidate if gain >= 1.0 pt. Pooled-OOF is reported but not used for selection because "
                               "cross-site calibration offsets penalise any calibrated score when the two folds are pooled.")
    print(f"base {current}: union {base['union']['oof_macro_mean']} wbf {base['wbf']['oof_macro_mean']} (macro OOF) "
          f"({round(time.time()-t0)} s)", flush=True)
    for cand in ("dino_B6", "rfdetr_L", "yolo_expE", "dino_B1"):
        trial = current + (cand,)
        R["sets"]["+".join(trial)] = run_set(trial)
        mw = R["sets"]["+".join(trial)]["wbf"]["oof_macro_mean"]
        mu = R["sets"]["+".join(trial)]["union"]["oof_macro_mean"]
        gain = round((max(mw, mu) - best_mean) * 100, 2)
        keep = gain >= 1.0
        R["addition_protocol"].append({"candidate": cand, "set_tested": list(trial), "wbf_mean_recall": mw,
                                       "union_mean_recall": mu, "gain_pts_vs_current_best": gain, "kept": keep})
        print(f"+{cand}: union {mu} wbf {mw} gain {gain}pt keep={keep} ({round(time.time()-t0)} s)", flush=True)
        if keep:
            current = trial; best_mean = max(mw, mu)
    R["chosen_models"] = list(current)
    chosen_res = R["sets"]["+".join(current)]
    best_method = "wbf" if chosen_res["wbf"]["oof_macro_mean"] >= chosen_res["union"]["oof_macro_mean"] else "union"
    R["chosen_method"] = best_method

    # 4. learned fusion on the chosen set (and on all six for information)
    R["learned"] = {"chosen": learned_fusion(current)}
    if not args.quick:
        R["learned"]["all6"] = learned_fusion(tuple(MODELS))
    lb = max(R["learned"]["chosen"].items(), key=lambda kv: kv[1]["oof_macro_mean"])
    R["learned_beats_wbf_by_pts"] = round((lb[1]["oof_macro_mean"] - chosen_res["wbf"]["oof_macro_mean"]) * 100, 2)
    R["learned_best"] = lb[0]
    print("learned done", round(time.time() - t0), "s; best", lb[0], lb[1]["oof_pooled"]["mean_recall"], flush=True)

    # 5. image level
    wbf_params_all = chosen_res["wbf"]["params_all_val"]
    R["image_level"] = image_level(tuple(MODELS), current, wbf_params_all)
    print("image-level done", round(time.time() - t0), "s", flush=True)

    # 5a. bootstrap CIs (image resampling within fold): chosen fusion OOF vs best single model of that fold, vs B6, vs A
    R["bootstrap"] = {}
    for c in (0, 1):
        f = FOLDS[c]
        singles = {m: subset(tables(D["preds"][m]), [c]) for m in MODELS}
        best_single = max(MODELS, key=lambda m: R["single_raw"][m]["by_fold"][f]["mean_recall"])
        fus = OOF_TBL[(current, best_method)][c]
        lean = OOF_TBL[(("yolo_A", "dino_B4", "dino_B6"), "wbf")][c]
        R["bootstrap"][f] = {"best_single": best_single,
                             f"fusion_vs_{best_single}": bootstrap_diff(fus, singles[best_single], c),
                             "fusion_vs_dino_B6": bootstrap_diff(fus, singles["dino_B6"], c),
                             "fusion_vs_yolo_A": bootstrap_diff(fus, singles["yolo_A"], c),
                             "fusion_vs_lean(A+B4+B6 wbf)": bootstrap_diff(fus, lean, c),
                             "lean(A+B4+B6 wbf)_vs_dino_B6": bootstrap_diff(lean, singles["dino_B6"], c)}
    print("bootstrap done", round(time.time() - t0), "s", flush=True)

    # 5b. threshold transfer: threshold set for FP p on fold X -> realized neg-FP / recall on fold Y
    def transfer(tbl_all):
        out = {}
        for c in (0, 1):
            fit = matched(subset(tbl_all, [c])); ev = subset(tbl_all, [1 - c]); sw = curve(ev)
            out[f"fit_{FOLDS[c]}->eval_{FOLDS[1-c]}"] = {}
            for p in FP_TARGETS:
                thr = fit[f"fp{p:.2f}"]["thr"]
                if thr is None:
                    out[f"fit_{FOLDS[c]}->eval_{FOLDS[1-c]}"][f"fp{p:.2f}"] = None; continue
                pt = min(sw, key=lambda s_: abs(s_["conf"] - thr))
                out[f"fit_{FOLDS[c]}->eval_{FOLDS[1-c]}"][f"fp{p:.2f}"] = {"thr": thr, "realized_fp": pt["neg_image_fp_rate"], "recall": pt["recall"]}
        return out
    R["threshold_transfer"] = {m: transfer(tables(D["preds"][m])) for m in MODELS}
    _fused_in = fuse_all(current, best_method, chosen_res[best_method]["params_all_val"], None)
    R["threshold_transfer"]["fusion[chosen, all-VAL calibration]"] = transfer(tables(_fused_in))
    print("transfer done", round(time.time() - t0), "s", flush=True)

    # 6. recipe: calibration refit on ALL VAL negatives; params chosen on all VAL; thresholds from pooled OOF curve
    final_params = chosen_res[best_method]["params_all_val"]
    cals = CAL[("calibrators", None)]
    oof_thr = {k: v["thr"] for k, v in chosen_res[best_method]["oof_pooled"].items() if k.startswith("fp")}
    il = R["image_level"]
    il_best = max([k for k in il if k.startswith(("max_cal[chosen", "mean_cal[chosen", "noisy_or[chosen", "logreg_cal[chosen", "fused_box_max"))],
                  key=lambda k: il[k]["mean_recall"])
    il_method = {"max_cal": "max_calibrated", "mean_cal": "mean_calibrated", "noisy_or": "noisy_or",
                 "fused_box_max": "fused_box_max"}.get(il_best.split("[")[0], "max_calibrated")
    recipe = {
        "recipe": "accurate_v5b", "mode": "ACCURATE (full-frame PARALLEL inference of every component; selective execution is a later test)",
        "created_utc": R["created_utc"], "scenario": "fixed-camera (pyro-sdis)",
        "score_scale": "calibrated: per-model raw score -> neg-image-FP quantile on pyro-sdis VAL negatives (1-c = single-model neg-image FP); "
                       "fused score = %s on that scale" % best_method,
        "dev_protocol": "grouped 2-fold by site (cabanelle | marguerite); calibration + params fit out-of-fold; thresholds_oof from pooled OOF curve; "
                        "calibration maps in this file are REFIT on all 1,158 VAL negatives for deployment",
        "models": {m: {"family": MODELS[m][2], "checkpoint": MODELS[m][1], "imgsz": 1280, "preds_source": MODELS[m][0],
                       "min_saved_score": 0.02, "latency_ms_l40s_batch1": MODELS[m][3], "latency_source": MODELS[m][4],
                       "calibration": cals[m].to_dict(),
                       "raw_score_at_calibrated": {str(q): round(cals[m].inverse(q), 4) for q in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99)}}
                   for m in current},
        "fusion": {"method": best_method, **final_params, "weights": {m: 1.0 for m in current}},
        "thresholds_oof": oof_thr,
        "thresholds_oof_by_site": {f: {k: v["thr"] for k, v in chosen_res[best_method]["oof_by_fold"][f].items() if k.startswith("fp")}
                                   for f in FOLDS.values()},
        "threshold_note": "neg-FP quantile maps did not transfer between the two VAL sites (see threshold_transfer in ensemble_dev_v5b.json): "
                          "a deployed camera must have its operating threshold set on its own verified negatives; thresholds_oof is the pooled compromise.",
        "image_level": {"method": il_method, "selected_by": il_best,
                        "thresholds_oof": {k: v["thr"] for k, v in il[il_best].items() if k.startswith("fp")}},
        "oof_results": {"box_level": chosen_res[best_method]["oof_pooled"], "image_level": il[il_best]},
        "training_data_disclosure": "yolo_A: E corpus (pyro-sdis train sites + D-Fire); dino_B4/B6: E+T (T = Boreal train burns + AuSmoke train frames, "
                                    "licence PENDING per PROTOCOL_V5 §2 — existing v4 use disclosed, not extended); rfdetr_L/yolo_expE: expB list (E). "
                                    "No FIgLib material in any component (quarantined at v4 dedup). Fusion fits only VAL-negative score quantiles + 3 WBF params.",
    }
    Path(args.recipe).parent.mkdir(parents=True, exist_ok=True)
    Path(args.recipe).write_text(json.dumps(recipe, indent=1))
    # sanity: recipe fuse() on VAL (in-sample calibration) reproduces the all-val configuration + timing
    rec = FusionRecipe(recipe)
    times = []
    fused = []
    for i in range(len(D["images"])):
        pb = {m: D["preds"][m][i].tolist() for m in current}
        t1 = time.perf_counter(); fb = fuse(pb, rec); times.append((time.perf_counter() - t1) * 1000)
        fused.append(np.asarray(fb).reshape(-1, 5))
    R["recipe_insample_all_val"] = matched(tables(fused))
    R["fusion_latency_ms_cpu"] = {"mean": round(float(np.mean(times)), 3), "median": round(float(np.median(times)), 3),
                                  "p90": round(float(np.percentile(times, 90)), 3), "max": round(float(np.max(times)), 3),
                                  "n_images": len(times), "note": "scripts/v5b/fusion.fuse (numpy WBF + calibration), devbox CPU single thread, per image"}
    comp = {m: MODELS[m][3] for m in current}
    fl = R["fusion_latency_ms_cpu"]["median"]
    R["latency_table"] = {"components_ms": comp, "fusion_ms": fl,
                          "parallel_ms": round(max(comp.values()) + fl, 1), "serial_ms": round(sum(comp.values()) + fl, 1),
                          "note": "batch-1 1x L40S fp32 @1280 from reports (T13 / v3 latency json); parallel = max(components)+fusion assumes one GPU per component; selective execution is a later test"}
    R["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(R, indent=1, default=float))
    print("saved", args.out_json, args.recipe, "elapsed", R["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
