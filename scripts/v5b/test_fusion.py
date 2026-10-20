"""Unit tests for scripts/v5b/fusion.py.  Run: .venv/bin/python -m pytest scripts/v5b/test_fusion.py -q"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.v5b import fusion as F  # noqa: E402


def _recipe(method="wbf", conf_type="max", iou_thr=0.4, skip=0.0, models=("m1", "m2")):
    rng = np.random.default_rng(0)
    d = {"recipe": "test", "models": {}, "fusion": {"method": method, "iou_thr": iou_thr,
                                                     "skip_box_thr": skip, "conf_type": conf_type},
         "thresholds_oof": {"fp0.05": 0.95, "fp0.10": 0.90},
         "image_level": {"method": "max_calibrated"}}
    if method == "union_nms":
        d["fusion"] = {"method": "union_nms", "iou_thr": iou_thr}
    for m in models:
        cal = F.ScoreCalibrator.fit(np.concatenate([rng.random(300) * 0.6, np.zeros(50)]))
        d["models"][m] = {"calibration": cal.to_dict()}
    return F.FusionRecipe(d)


# ------------------------------------------------------------------ calibration
def test_calibrator_monotone_and_range():
    cal = F.ScoreCalibrator.fit([0.0, 0.1, 0.1, 0.3, 0.9])
    s = np.linspace(0, 1, 101)
    c = cal(s)
    assert np.all(np.diff(c) >= 0) and c[-1] == 1.0
    assert cal(0.0) == pytest.approx(1 / 5)             # one negative with no detection sits below any real score
    # 1 - c(s) ~ fraction of negatives with max score >= s (mid-rank at ties)
    assert abs((1 - cal(0.3)) - 1.5 / 5) < 1e-9        # 0.3 (half) + 0.9 above/at
    assert cal(0.9) == pytest.approx(4.5 / 5)
    assert cal(0.9) < cal(0.95) < cal(1.0) == 1.0       # ordering above the worst negative is preserved
    assert cal(-1.0) == 0.0 and cal(2.0) == 1.0
    cal2 = F.ScoreCalibrator.fit([0.2, 0.4])
    assert cal2(0.0) == 0.0 and cal2(0.1) < cal2(0.2) == pytest.approx(0.25)


def test_calibrator_roundtrip_json_and_inverse():
    cal = F.ScoreCalibrator.fit(np.random.default_rng(1).random(200))
    d = json.loads(json.dumps(cal.to_dict()))
    cal2 = F.ScoreCalibrator.from_dict(d)
    s = np.random.default_rng(2).random(50)
    assert np.allclose(cal(s), cal2(s), atol=1e-6)
    for q in (0.5, 0.9, 0.99):
        assert cal(cal.inverse(q)) == pytest.approx(q, abs=1e-6)


def test_calibrator_rejects_bad_knots():
    with pytest.raises(ValueError):
        F.ScoreCalibrator([0.0, 0.5, 0.4], [0.0, 0.5, 1.0])
    with pytest.raises(ValueError):
        F.ScoreCalibrator.fit([])


# ------------------------------------------------------------------ primitives
def test_iou_matrix_basic():
    a = np.array([[0, 0, 1, 1]]); b = np.array([[0.5, 0, 1.5, 1], [2, 2, 3, 3]])
    assert np.allclose(F.iou_matrix(a, b), [[1 / 3, 0.0]])
    assert F.iou_matrix(np.zeros((0, 4)), b).shape == (0, 2)


def test_nms_keeps_highest_and_separate():
    b = np.array([[0, 0, 1, 1, 0.5], [0.05, 0, 1.05, 1, 0.9], [0.5, 0.5, 0.6, 0.6, 0.7]])
    out = F.nms(b, 0.5)
    assert len(out) == 2 and out[0, 4] == 0.9 and out[1, 4] == 0.7


def test_union_nms_empty_and_single():
    assert F.union_nms([np.zeros((0, 5)), np.zeros((0, 5))]).shape == (0, 5)
    one = np.array([[0.1, 0.1, 0.2, 0.2, 0.3]])
    assert np.allclose(F.union_nms([one, np.zeros((0, 5))]), one)


def test_wbf_agreeing_boxes_merge_and_disagreeing_kept():
    m1 = np.array([[0.10, 0.10, 0.30, 0.30, 0.8]])
    m2 = np.array([[0.11, 0.10, 0.31, 0.30, 0.6], [0.7, 0.7, 0.9, 0.9, 0.5]])
    out = F.wbf([m1, m2], iou_thr=0.5, conf_type="avg")
    assert len(out) == 2
    # merged cluster: mean conf 0.7 * min(2,2)/2 = 0.7; lone box 0.5 * 1/2 = 0.25
    assert out[0, 4] == pytest.approx(0.7) and out[1, 4] == pytest.approx(0.25)
    # fused coordinates = score-weighted mean
    assert out[0, 0] == pytest.approx((0.10 * 0.8 + 0.11 * 0.6) / 1.4)
    out_max = F.wbf([m1, m2], iou_thr=0.5, conf_type="max")
    assert out_max[0, 4] == pytest.approx(0.8) and out_max[1, 4] == pytest.approx(0.5)


def test_wbf_skip_thr_weights_and_conf_types():
    m1 = np.array([[0.1, 0.1, 0.3, 0.3, 0.9]]); m2 = np.array([[0.1, 0.1, 0.3, 0.3, 0.2]])
    assert len(F.wbf([m1, m2], skip_box_thr=0.5)) == 1
    assert F.wbf([m1, m2], skip_box_thr=0.5, conf_type="avg")[0, 4] == pytest.approx(0.45)
    # weights: scores pre-multiplied; 'max' divides by max weight
    assert F.wbf([m1, m2], conf_type="max", weights=[2, 1])[0, 4] == pytest.approx(0.9)
    # box_and_model_avg with both models present == plain mean; absent_model_aware == mean when all present
    for ct in ("box_and_model_avg", "absent_model_aware_avg"):
        assert F.wbf([m1, m2], conf_type=ct)[0, 4] == pytest.approx(0.55)
    # lone box from one of three models: absent_model_aware -> 0.9*1/(1+2)
    assert F.wbf([m1, np.zeros((0, 5)), np.zeros((0, 5))], conf_type="absent_model_aware_avg")[0, 4] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        F.wbf([m1], conf_type="median")


@pytest.mark.skipif(pytest.importorskip("ensemble_boxes", reason="pip package absent") is None, reason="")
def test_wbf_parity_with_ensemble_boxes():
    import warnings
    from ensemble_boxes import weighted_boxes_fusion
    rng = np.random.default_rng(3)
    for _ in range(60):
        per = []
        for _m in range(rng.integers(1, 4)):
            n = rng.integers(0, 6)
            c = rng.random((n, 2)); s = rng.random((n, 2)) * 0.3 + 0.02
            per.append(np.column_stack([np.clip(c - s / 2, 0, 1), np.clip(c + s / 2, 0, 1), rng.random(n)]))
        w = list(rng.integers(1, 3, size=len(per)).astype(float))
        for ct in F.CONF_TYPES:
            mine = F.wbf(per, iou_thr=0.5, skip_box_thr=0.1, conf_type=ct, weights=w)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rb, rs, _ = weighted_boxes_fusion([p[:, :4].tolist() for p in per], [p[:, 4].tolist() for p in per],
                                                  [[0] * len(p) for p in per], weights=w, iou_thr=0.5,
                                                  skip_box_thr=0.1, conf_type=ct)
            assert len(rb) == len(mine)
            if len(rb):
                assert np.allclose(rb, mine[:, :4], atol=1e-5) and np.allclose(rs, mine[:, 4], atol=1e-5)


# ------------------------------------------------------------------ recipe + public API
def test_fuse_output_format_sorted_and_calibrated():
    rec = _recipe()
    out = F.fuse({"m1": [[0.1, 0.1, 0.3, 0.3, 0.9], [0.5, 0.5, 0.6, 0.6, 0.05]],
                  "m2": [[0.1, 0.1, 0.3, 0.3, 0.7]]}, rec)
    assert isinstance(out, list) and all(len(b) == 5 for b in out)
    assert all(isinstance(v, float) for b in out for v in b)
    scores = [b[4] for b in out]
    assert scores == sorted(scores, reverse=True) and all(0.0 <= s <= 1.0 for s in scores)
    assert len(out) == 2 and out[0][4] > 0.99                  # 0.9 raw > every synthetic negative (max 0.6)


def test_fuse_empty_and_strictness():
    rec = _recipe()
    assert F.fuse({"m1": [], "m2": []}, rec) == []
    with pytest.raises(ValueError):
        F.fuse({"m1": []}, rec)                       # missing model
    with pytest.raises(ValueError):
        F.fuse({"m1": [], "m2": [], "m3": []}, rec)   # unknown model
    assert F.fuse({"m1": [[0.1, 0.1, 0.2, 0.2, 0.9]]}, rec, strict=False)[0][4] > 0.99


def test_fuse_min_score_and_clipping():
    rec = _recipe()
    out = F.fuse({"m1": [[-0.1, 0.2, 1.3, 0.1, 0.9]], "m2": []}, rec)   # out-of-range + swapped y
    assert out[0][:4] == pytest.approx([0.0, 0.1, 1.0, 0.2])
    assert F.fuse({"m1": [[0.1, 0.1, 0.2, 0.2, 0.01]], "m2": []}, rec, min_score=0.5) == []


def test_union_recipe_and_image_score():
    rec = _recipe(method="union_nms", iou_thr=0.5)
    out = F.fuse({"m1": [[0.1, 0.1, 0.3, 0.3, 0.9]], "m2": [[0.11, 0.1, 0.31, 0.3, 0.3]]}, rec)
    assert len(out) == 1
    assert F.image_score({"m1": [], "m2": []}, rec) == 0.0
    s = F.image_score({"m1": [[0.1, 0.1, 0.3, 0.3, 0.9]], "m2": []}, rec)
    assert s > 0.99
    rec.image_level = {"method": "noisy_or"}
    assert 0.0 <= F.image_score({"m1": [[0.1, 0.1, 0.3, 0.3, 0.4]], "m2": [[0.6, 0.6, 0.7, 0.7, 0.4]]}, rec) <= 1.0
    rec.image_level = {"method": "fused_box_max"}
    assert F.image_score({"m1": [[0.1, 0.1, 0.3, 0.3, 0.9]], "m2": []}, rec) > 0.99
    assert rec.threshold_for_fp(0.05) == 0.95 and rec.threshold_for_fp(0.01) is None


def test_fuse_is_deterministic():
    rec = _recipe()
    pb = {"m1": [[0.1, 0.1, 0.3, 0.3, 0.9], [0.4, 0.4, 0.5, 0.5, 0.3]], "m2": [[0.12, 0.1, 0.3, 0.3, 0.7]]}
    assert F.fuse(pb, rec) == F.fuse(pb, rec)


@pytest.mark.skipif(not F.DEFAULT_RECIPE.exists(), reason="accurate_recipe.json not built yet")
def test_default_recipe_loads_and_is_consistent():
    rec = F.load_recipe()
    assert rec.method in ("wbf", "union_nms") and len(rec.order) >= 2
    thr = [rec.threshold_for_fp(p) for p in (0.01, 0.02, 0.05, 0.10, 0.20)]
    thr = [t for t in thr if t is not None]
    assert thr == sorted(thr, reverse=True), "fused-score thresholds must fall as the FP budget grows"
    pb = {m: [[0.4, 0.4, 0.45, 0.45, 0.99]] for m in rec.order}
    out = F.fuse(pb, rec)
    assert len(out) == 1 and out[0][4] > 0.9
    for m in rec.order:
        assert Path(ROOT / rec.models[m]["checkpoint"]).exists(), m
