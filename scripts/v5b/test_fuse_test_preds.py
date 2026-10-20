"""Unit test for scripts/v5b/fuse_test_preds.py on the pyro-sdis VAL parquets (validation side only).

  .venv/bin/python -m pytest scripts/v5b/test_fuse_test_preds.py -q      # assertions
  .venv/bin/python scripts/v5b/test_fuse_test_preds.py                   # same + summary table

Checks: (1) the fused VAL parquet with the recipe's own yolo (v4a_s2) reproduces ENSEMBLE_DEV's
recipe_insample_all_val recall @1/2/5/10/20% neg-image FP (same GRID/matching code as
scripts/v5b/ensemble_dev.py); (2) schema == eval_detection .raw.parquet (image/gt/preds);
(3) scripts/v4/compare_matched_fp.py and src.eval.eval_detection --preds-jsonl both consume
the outputs; (4) the S1_A (v5b control) substitution runs and is flagged.  Outputs land in
reports/v5b/val/eval_S5_ensemble_{v4a_s2,S1_A}_sdisval.*  (VAL artifacts, allowed).
"""
from __future__ import annotations

import functools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.v5b import ensemble_dev as ED  # noqa: E402  (reference metric implementation)
from scripts.v5b import fuse_test_preds as FT  # noqa: E402

PY = str(ROOT / ".venv/bin/python")
REF_JSON = ROOT / "reports/v5b/ensemble_dev_v5b.json"
FP_KEYS = ["fp0.01", "fp0.02", "fp0.05", "fp0.10", "fp0.20"]
TOL = 0.002   # grid resolution of the shared GRID (recall is piecewise-constant between grid points)


def _populate_D(ref_parquet: Path) -> None:
    """Mirror ensemble_dev.load_data() for the GT side only (fold/gts/gt_n), from one parquet."""
    ref = pd.read_parquet(ref_parquet).sort_values("image").reset_index(drop=True)
    images = ref["image"].tolist()
    fold = np.array([0 if "cabanelle" in p else 1 if "marguerite" in p else -1 for p in images])
    assert (fold >= 0).all()
    gts = [np.asarray([list(g) for g in g_], float).reshape(-1, 5) for g_ in ref["gt"]]
    bucket = lambda a: next(i for i, (_, lo, hi) in enumerate(ED.SIZE_BUCKETS) if lo <= a < hi)  # noqa: E731
    gt_n = np.zeros((2, 4), int)
    for f, g in zip(fold, gts):
        for a in g[:, 4]:
            gt_n[f, bucket(a)] += 1
    ED.D.clear()
    ED.D.update(images=images, fold=fold, gts=gts, gt_n=gt_n, is_neg=np.array([len(g) == 0 for g in gts]))


def _arrs(parquet: Path) -> list[np.ndarray]:
    df = pd.read_parquet(parquet).sort_values("image").reset_index(drop=True)
    assert df["image"].tolist() == ED.D["images"], parquet
    return [np.asarray([list(p)[:5] for p in p_], float).reshape(-1, 5) for p_ in df["preds"]]


def matched_fp(parquet: Path) -> dict:
    """ensemble_dev-style matched-FP recall table (IoU>=0.4, GRID interpolation)."""
    return ED.matched(ED.tables(_arrs(parquet)))


@functools.lru_cache(maxsize=None)
def fused(alias: str) -> dict:
    out = ROOT / "reports/v5b/val" / f"eval_S5_ensemble_{alias}_sdisval"
    rc = FT.main(["--set", "sdisval", "--yolo", alias, "--quiet"])
    assert rc == 0
    meta = json.loads(out.with_suffix(".json").read_text())
    return {"out": out, "meta": meta, "parquet": out.with_suffix(".raw.parquet"), "jsonl": out.with_suffix(".jsonl")}


@functools.lru_cache(maxsize=None)
def _setup() -> dict:
    _populate_D(ROOT / FT.SETS["sdisval"]["yolo"]["v4a_s2"])
    ref = json.loads(REF_JSON.read_text())
    singles = {m: ROOT / FT.default_sources("sdisval", "v4a_s2")[m] for m in ("yolo_A", "dino_B4", "dino_B6", "dino_B1")}
    singles["yolo_S1_A"] = ROOT / FT.resolve_yolo("sdisval", "S1_A")
    return {"ref": ref, "singles": singles}


# ---------------------------------------------------------------- tests
def test_schema_matches_eval_detection_parquet():
    f = fused("v4a_s2")
    df = pd.read_parquet(f["parquet"])
    assert list(df.columns) == ["image", "gt", "preds"] and len(df) == 7335
    ref = pd.read_parquet(ROOT / FT.SETS["sdisval"]["yolo"]["v4a_s2"]).sort_values("image").reset_index(drop=True)
    assert df["image"].tolist() == ref["image"].tolist()
    for a, b in zip(df["gt"], ref["gt"]):
        assert len(a) == len(b) and (len(a) == 0 or np.allclose(np.stack(list(a)), np.stack(list(b))))
    for p in df["preds"]:
        arr = np.asarray([list(x) for x in p], float).reshape(-1, 5)
        assert ((arr[:, :4] >= 0) & (arr[:, :4] <= 1)).all() and ((arr[:, 4] >= 0) & (arr[:, 4] <= 1)).all()
        assert (np.diff(arr[:, 4]) <= 1e-12).all()          # fused boxes sorted by score desc
    assert f["meta"]["yolo_component_matches_recipe"] is True
    assert sum(1 for _ in open(f["jsonl"])) == 7335


def test_fused_val_reproduces_ensemble_dev():
    s = _setup()
    got = matched_fp(fused("v4a_s2")["parquet"])
    exp = s["ref"]["recipe_insample_all_val"]
    for k in FP_KEYS:
        assert abs(got[k]["recall"] - exp[k]["recall"]) <= TOL, (k, got[k]["recall"], exp[k]["recall"])
    # the single models from this harness must equal ENSEMBLE_DEV T2 too (same parquets, same code)
    for m in ("yolo_A", "dino_B4", "dino_B6", "dino_B1"):
        single = matched_fp(s["singles"][m])
        for k in FP_KEYS:
            assert abs(single[k]["recall"] - s["ref"]["single_raw"][m][k]["recall"]) <= TOL, (m, k)


def test_S1_A_substitution_runs_and_is_flagged():
    f = fused("S1_A")
    assert f["meta"]["yolo_component_matches_recipe"] is False
    assert f["meta"]["components"]["yolo_A"]["parquet"].endswith("reports/v5b/val/eval_S1_A_sdisval.raw.parquet")
    assert len(pd.read_parquet(f["parquet"])) == 7335


def test_compare_matched_fp_consumes_fused_parquet():
    s = _setup(); f = fused("v4a_s2")
    out = ROOT / "reports/v5b/val/matched_fp_S5_ensemble_v4a_s2_sdisval.json"
    cmd = [PY, "scripts/v4/compare_matched_fp.py", "--model", f"S5_ensemble={f['parquet']}",
           "--model", f"yolo_A={s['singles']['yolo_A']}", "--model", f"dino_B6={s['singles']['dino_B6']}",
           "--top-fine", "--fp-targets", "0.05,0.10", "--out", str(out)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(out.read_text())
    assert "S5_ensemble" in json.dumps(d)


def test_eval_detection_consumes_fused_jsonl(tmp_path=Path("/tmp/fuse_selftest")):
    f = fused("v4a_s2"); tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "eval_S5_ensemble_v4a_s2_sdisval.json"
    cmd = [PY, "-m", "src.eval.eval_detection", "--data-yaml", FT.SETS["sdisval"]["yaml"], "--split", "val",
           "--preds-jsonl", str(f["jsonl"]), "--out", str(out)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads(out.read_text()); assert rep["n_images"] == 7335 and rep["n_negative_images"] == 1158
    a = pd.read_parquet(out.with_suffix(".raw.parquet")).sort_values("image").reset_index(drop=True)
    b = pd.read_parquet(f["parquet"]).sort_values("image").reset_index(drop=True)
    assert a["image"].tolist() == b["image"].tolist()
    for x, y in zip(a["preds"], b["preds"]):
        assert len(x) == len(y) and (len(x) == 0 or np.allclose(np.stack(list(x)), np.stack(list(y))))


# ---------------------------------------------------------------- summary table
def summary() -> None:
    s = _setup()
    rows = []
    for label, pq in (("yolo_A (v4a_s2, recipe)", s["singles"]["yolo_A"]), ("yolo S1_A (v5b ctrl)", s["singles"]["yolo_S1_A"]),
                      ("dino_B4", s["singles"]["dino_B4"]), ("dino_B6", s["singles"]["dino_B6"]), ("dino_B1", s["singles"]["dino_B1"]),
                      ("S5 fused [yolo=v4a_s2]", fused("v4a_s2")["parquet"]), ("S5 fused [yolo=S1_A]", fused("S1_A")["parquet"])):
        m = matched_fp(pq); rows.append((label, *(m[k]["recall"] for k in FP_KEYS), m["mean_recall"]))
    exp = s["ref"]["recipe_insample_all_val"]
    rows.append(("ENSEMBLE_DEV recipe_insample_all_val", *(exp[k]["recall"] for k in FP_KEYS), exp["mean_recall"]))
    print("sdis VAL matched neg-image-FP recall (IoU>=0.4, ensemble_dev GRID; 7,335 imgs, 1,158 neg)")
    print(f"{'system':38s} " + " ".join(f"{k[2:]:>7s}" for k in FP_KEYS) + "    mean")
    for r in rows:
        print(f"{r[0]:38s} " + " ".join(f"{v:7.4f}" for v in r[1:6]) + f"  {r[6]:6.4f}")
    for alias in ("v4a_s2", "S1_A"):
        bl = fused(alias)["meta"]["at_recipe_thresholds"]["box_level"]
        print(f"S5[{alias}] box recall at recipe thresholds_oof: " + "  ".join(f"{k}:{v['recall']} (fp {v['neg_image_fp_rate']})" for k, v in bl.items()))


if __name__ == "__main__":
    import pytest
    rc = pytest.main([__file__, "-q", "-x", "-p", "no:cacheprovider"])
    summary()
    raise SystemExit(rc)
