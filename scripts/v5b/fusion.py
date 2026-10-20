"""V5b ACCURATE-mode detection fusion (pyro-sdis fixed-camera scenario).

Importable by the demo / serving layer:

    from scripts.v5b.fusion import fuse, load_recipe
    boxes = fuse({"yolo_A": [[x1n, y1n, x2n, y2n, score], ...],
                  "dino_B4": [...]} )          # -> [[x1n, y1n, x2n, y2n, fused_score], ...]

Box format everywhere: normalized xyxy in [0, 1] plus a score, identical to the
`preds` column of the eval_detection .raw.parquet files.  Fused scores live on
the CALIBRATED scale: each model's raw confidence is first mapped through a
monotone piecewise-linear map to its negative-image-FP quantile on pyro-sdis
VAL negatives (c = fraction of VAL negative images whose max score is below
the raw score, so 1 - c = that model's single-model neg-image FP rate), then
boxes are combined with Weighted Boxes Fusion (WBF) or union+NMS as fixed in
the recipe JSON (models/v5b/fusion/accurate_recipe.json).

The WBF here is a dependency-free numpy re-implementation that mirrors
ensemble_boxes.weighted_boxes_fusion (ZFTurbo, MIT) semantics for the four
conf_type modes; scripts/v5b/test_fusion.py checks parity against the
pip package when it is installed.  Nothing in this module runs inference.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE = ROOT / "models" / "v5b" / "fusion" / "accurate_recipe.json"

Box = Sequence[float]  # [x1n, y1n, x2n, y2n, score]


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _as_array(boxes: Iterable[Box] | None) -> np.ndarray:
    """-> float64 array (N, 5); clips coords to [0,1], orders x1<x2, y1<y2."""
    if boxes is None:
        return np.zeros((0, 5), dtype=np.float64)
    arr = np.asarray([list(b)[:5] for b in boxes], dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 5), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(f"boxes must be (N,5) [x1,y1,x2,y2,score]; got {arr.shape}")
    xy = np.clip(arr[:, :4], 0.0, 1.0)
    x1 = np.minimum(xy[:, 0], xy[:, 2]); x2 = np.maximum(xy[:, 0], xy[:, 2])
    y1 = np.minimum(xy[:, 1], xy[:, 3]); y2 = np.maximum(xy[:, 1], xy[:, 3])
    out = np.stack([x1, y1, x2, y2, arr[:, 4]], axis=1)
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) xyxy arrays -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


# --------------------------------------------------------------------------
# calibration: raw score -> neg-image-FP quantile
# --------------------------------------------------------------------------
class ScoreCalibrator:
    """Monotone piecewise-linear map raw score -> calibrated score in [0,1].

    Fitted from the per-negative-image MAX raw score of one model: for a raw
    score s, c(s) = (mid-rank) fraction of negative images whose max score is
    below s, so 1 - c(s) is the single-model negative-image FP rate at
    threshold s on the fitting set.  Images with no saved prediction count
    with max score 0 (saved parquets keep conf >= 0.02)."""

    def __init__(self, x: Sequence[float], y: Sequence[float], meta: dict | None = None):
        x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("calibrator needs >=2 knots")
        if np.any(np.diff(x) < 0) or np.any(np.diff(y) < 0):
            raise ValueError("calibrator knots must be non-decreasing")
        self.x, self.y = x, y
        self.meta = dict(meta or {})

    @classmethod
    def fit(cls, neg_max_scores: Sequence[float], meta: dict | None = None) -> "ScoreCalibrator":
        s = np.sort(np.asarray(neg_max_scores, dtype=np.float64))
        n = len(s)
        if n == 0:
            raise ValueError("need at least one negative image to fit")
        vals, first = np.unique(s, return_index=True)
        counts = np.diff(np.append(first, n))
        below = first.astype(np.float64)
        mid = (below + 0.5 * counts) / n            # mid-rank ECDF at each knot
        if vals[0] == 0.0:                          # score 0 = "no detection": any real score is above all of them
            mid[0] = counts[0] / n
        else:
            vals = np.concatenate([[0.0], vals]); mid = np.concatenate([[0.0], mid])
        # scores above the worst negative keep their ordering: interpolate to (1.0, 1.0)
        # instead of collapsing to 1.0 (DINO heads saturate >= 0.99; ties would hurt fusion)
        if vals[-1] < 1.0:
            vals = np.append(vals, 1.0); mid = np.append(mid, 1.0)
        x, y = vals, np.maximum.accumulate(mid)
        y[-1] = 1.0
        m = {"n_neg": int(n), "max_neg_score": float(s[-1])}
        m.update(meta or {})
        return cls(x, y, m)

    def __call__(self, scores) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        return np.interp(s, self.x, self.y, left=0.0, right=1.0)

    def inverse(self, cal: float) -> float:
        """Smallest raw score whose calibrated value reaches `cal`."""
        return float(np.interp(cal, self.y, self.x))

    def to_dict(self) -> dict:
        return {"knots_raw": [round(float(v), 6) for v in self.x],
                "knots_cal": [round(float(v), 6) for v in self.y],
                "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "ScoreCalibrator":
        return cls(d["knots_raw"], d["knots_cal"], d.get("meta"))


# --------------------------------------------------------------------------
# fusion primitives (inputs: list of (N_i,5) arrays, one per model)
# --------------------------------------------------------------------------
def nms(boxes: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS on an (N,5) array; keeps the highest-score box per cluster."""
    if len(boxes) == 0:
        return boxes
    order = np.argsort(-boxes[:, 4], kind="stable")
    b = boxes[order]
    keep = []
    alive = np.ones(len(b), dtype=bool)
    ious = iou_matrix(b[:, :4], b[:, :4])
    for i in range(len(b)):
        if not alive[i]:
            continue
        keep.append(i)
        alive &= ious[i] <= iou_thr
        alive[i] = False
    return b[keep]


def union_nms(per_model: Sequence[np.ndarray], iou_thr: float = 0.5) -> np.ndarray:
    """OR/union baseline: concatenate (already calibrated) boxes, then NMS."""
    arrs = [a for a in per_model if len(a)]
    if not arrs:
        return np.zeros((0, 5))
    return nms(np.concatenate(arrs, axis=0), iou_thr)


CONF_TYPES = ("avg", "max", "box_and_model_avg", "absent_model_aware_avg")


def wbf(per_model: Sequence[np.ndarray], iou_thr: float = 0.55, skip_box_thr: float = 0.0,
        conf_type: str = "avg", weights: Sequence[float] | None = None) -> np.ndarray:
    """Weighted Boxes Fusion (Solovyev et al. 2021) — numpy port of
    ensemble_boxes.weighted_boxes_fusion for a single class.

    per_model[i] is an (N_i,5) array for model i (coords in [0,1]).  Returns an
    (K,5) array sorted by fused score desc.  Semantics kept identical to the
    reference: boxes with score < skip_box_thr are dropped; scores are
    pre-multiplied by the model weight; a box joins the cluster whose current
    fused box has the highest IoU if that IoU is > iou_thr; fused coords are
    the score-weighted mean; cluster confidence is then rescaled by
    conf_type ('avg': mean * min(M, n)/sum(w); 'max': max/ max(w);
    'box_and_model_avg', 'absent_model_aware_avg' as in the reference)."""
    if conf_type not in CONF_TYPES:
        raise ValueError(f"conf_type must be one of {CONF_TYPES}")
    M = len(per_model)
    w = np.ones(M) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(w) != M:
        raise ValueError("weights length must equal number of models")
    rows = []  # [score*w, w, model_idx, x1, y1, x2, y2]
    for i, arr in enumerate(per_model):
        if arr is None or len(arr) == 0:
            continue
        a = np.asarray(arr, dtype=np.float64)
        a = a[a[:, 4] >= skip_box_thr]
        a = a[(a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) > 0]
        if len(a) == 0:
            continue
        rows.append(np.column_stack([a[:, 4] * w[i], np.full(len(a), w[i]),
                                     np.full(len(a), i), a[:, :4]]))
    if not rows:
        return np.zeros((0, 5))
    allb = np.concatenate(rows, axis=0)
    allb = allb[np.argsort(-allb[:, 0], kind="stable")]

    clusters: list[list[int]] = []
    fused: list[np.ndarray] = []  # current fused xyxy per cluster
    for j in range(len(allb)):
        idx = -1
        if fused:
            ious = iou_matrix(np.asarray(fused), allb[j:j + 1, 3:7])[:, 0]
            k = int(np.argmax(ious))
            if ious[k] > iou_thr:
                idx = k
        if idx == -1:
            clusters.append([j]); fused.append(allb[j, 3:7].copy())
        else:
            clusters[idx].append(j)
            mem = allb[clusters[idx]]
            fused[idx] = (mem[:, 0:1] * mem[:, 3:7]).sum(0) / mem[:, 0].sum()

    out = np.zeros((len(clusters), 5))
    wsum = w.sum()
    for c, members in enumerate(clusters):
        mem = allb[members]
        n = len(mem)
        if conf_type == "max":
            conf = mem[:, 0].max() / w.max()
        else:
            conf = mem[:, 0].mean()
            if conf_type == "avg":
                conf = conf * min(M, n) / wsum
            elif conf_type == "box_and_model_avg":
                uniq = np.unique(mem[:, 2].astype(int))
                conf = conf * n / mem[:, 1].sum() * w[uniq].sum() / wsum
            else:  # absent_model_aware_avg
                uniq = np.unique(mem[:, 2].astype(int))
                absent = np.ones(M, dtype=bool); absent[uniq] = False
                conf = conf * n / (mem[:, 1].sum() + w[absent].sum())
        out[c, :4] = fused[c]
        out[c, 4] = conf
    return out[np.argsort(-out[:, 4], kind="stable")]


# --------------------------------------------------------------------------
# recipe
# --------------------------------------------------------------------------
class FusionRecipe:
    """Models + per-model calibration maps + fusion method/params."""

    def __init__(self, d: dict):
        self.raw = d
        self.name = d.get("recipe", "unnamed")
        self.models: dict[str, dict] = d["models"]
        self.order: list[str] = list(self.models.keys())
        self.calibrators = {m: ScoreCalibrator.from_dict(spec["calibration"])
                            for m, spec in self.models.items()}
        f = d["fusion"]
        self.method = f["method"]
        self.params = {k: v for k, v in f.items() if k not in ("method", "weights")}
        wd = f.get("weights") or {}
        self.weights = [float(wd.get(m, 1.0)) for m in self.order]
        self.thresholds = d.get("thresholds_oof", {})
        self.image_level = d.get("image_level", {})

    def calibrate(self, preds_by_model: dict[str, Iterable[Box]], strict: bool = True
                  ) -> list[np.ndarray]:
        unknown = set(preds_by_model) - set(self.order)
        missing = set(self.order) - set(preds_by_model)
        if strict and (unknown or missing):
            raise ValueError(f"model set mismatch: unknown={sorted(unknown)} "
                             f"missing={sorted(missing)}; recipe expects {self.order}")
        out = []
        for m in self.order:
            arr = _as_array(preds_by_model.get(m))
            if len(arr):
                arr[:, 4] = self.calibrators[m](arr[:, 4])
            out.append(arr)
        return out

    def fuse_arrays(self, cal: Sequence[np.ndarray]) -> np.ndarray:
        if self.method == "wbf":
            return wbf(cal, weights=self.weights, **self.params)
        if self.method == "union_nms":
            return union_nms(cal, iou_thr=float(self.params.get("iou_thr", 0.5)))
        raise ValueError(f"unknown fusion method {self.method}")

    def threshold_for_fp(self, fp: float) -> float | None:
        key = f"fp{fp:.2f}"
        v = self.thresholds.get(key)
        return None if v is None else float(v)


_RECIPE_CACHE: dict[str, FusionRecipe] = {}


def load_recipe(path: str | os.PathLike | None = None) -> FusionRecipe:
    p = Path(path or os.environ.get("SMOKE_FUSION_RECIPE", DEFAULT_RECIPE))
    key = str(p.resolve())
    if key not in _RECIPE_CACHE:
        _RECIPE_CACHE[key] = FusionRecipe(json.loads(p.read_text()))
    return _RECIPE_CACHE[key]


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def fuse(preds_by_model: dict[str, list[Box]], recipe: FusionRecipe | str | os.PathLike | None = None,
         min_score: float | None = None, strict: bool = True) -> list[list[float]]:
    """Fuse one image's per-model detections into one list of boxes.

    preds_by_model: {model_name: [[x1n, y1n, x2n, y2n, raw_score], ...]} with
      model names as in the recipe (e.g. "yolo_A", "dino_B4").  An empty list
      means the model ran and found nothing.  With strict=True (default) the
      key set must equal the recipe's model set — ACCURATE mode is full-frame
      PARALLEL execution of every component; selective execution is a later
      test and must not be silently emulated by omitting models.
    recipe: FusionRecipe, path, or None (-> models/v5b/fusion/accurate_recipe.json
      or $SMOKE_FUSION_RECIPE).
    min_score: optional cut on the fused CALIBRATED score (e.g.
      recipe.threshold_for_fp(0.05)).
    Returns [[x1n, y1n, x2n, y2n, fused_calibrated_score], ...] sorted by score desc.
    """
    rec = recipe if isinstance(recipe, FusionRecipe) else load_recipe(recipe)
    cal = rec.calibrate(preds_by_model, strict=strict)
    out = rec.fuse_arrays(cal)
    if min_score is not None and len(out):
        out = out[out[:, 4] >= min_score]
    return [[float(v) for v in row] for row in out]


def image_score(preds_by_model: dict[str, list[Box]], recipe: FusionRecipe | str | os.PathLike | None = None,
                strict: bool = True) -> float:
    """Image-level alert score on the calibrated scale (for alerting).

    method from recipe["image_level"]["method"]:
      'max_calibrated' (default): max over models of the calibrated max score;
      'mean_calibrated': mean over models; 'noisy_or': 1 - prod(1 - c_m);
      'fused_box_max': max fused box score from fuse()."""
    rec = recipe if isinstance(recipe, FusionRecipe) else load_recipe(recipe)
    method = rec.image_level.get("method", "max_calibrated")
    if method == "fused_box_max":
        boxes = fuse(preds_by_model, rec, strict=strict)
        return float(boxes[0][4]) if boxes else 0.0
    cal = rec.calibrate(preds_by_model, strict=strict)
    per = np.array([a[:, 4].max() if len(a) else 0.0 for a in cal])
    if method == "max_calibrated":
        return float(per.max())
    if method == "mean_calibrated":
        return float(per.mean())
    if method == "noisy_or":
        return float(1.0 - np.prod(1.0 - per))
    raise ValueError(f"unknown image_level method {method}")


__all__ = ["fuse", "image_score", "load_recipe", "FusionRecipe", "ScoreCalibrator",
           "wbf", "union_nms", "nms", "iou_matrix", "DEFAULT_RECIPE"]
