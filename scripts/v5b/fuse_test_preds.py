#!/usr/bin/env python
"""Fuse per-model .raw.parquet predictions into the S5 ACCURATE ensemble parquet.

Pure parquet in -> parquet out: never opens an image, never fits anything.  Applies the
FROZEN recipe (models/v5b/fusion/accurate_recipe.json: yolo_A + dino_B4 + dino_B6 + dino_B1,
calibrated WBF) via scripts/v5b/fusion.fuse()/image_score() to one evaluation set whose
four component parquets already exist, and writes

  <out>.raw.parquet   rows image / gt / preds  (preds = fused boxes [x1n,y1n,x2n,y2n,CALIBRATED score])
                      -> same schema as src.eval.eval_detection output, so
                         scripts/v4/compare_matched_fp.py --model S5=<out>.raw.parquet works as-is
  <out>.jsonl         {"image", "preds", "image_score"} per row -> `python -m src.eval.eval_detection
                      --preds-jsonl <out>.jsonl --data-yaml <yaml> --split <split>` re-sweeps it
                      (eval_detection's fixed CONF_GRID 0.05..0.7 is coarse on the calibrated scale;
                      use compare_matched_fp --top-fine or the recipe thresholds below)
  <out>.json          provenance (component parquets + sha256, recipe, yolo component) and recall at
                      the recipe's pre-registered calibrated thresholds (box level + image level)

Usage
  python scripts/v5b/fuse_test_preds.py --set firemanholdout --yolo S1_A
  python scripts/v5b/fuse_test_preds.py --set sdistest --yolo v4a_s2        # recipe's own yolo (v4 S1_A_ctrl)
  python scripts/v5b/fuse_test_preds.py --set sdisval  --yolo v4a_s2        # VAL reproduction (unit test)
  python scripts/v5b/fuse_test_preds.py --set sdistest --model yolo_A=path --model dino_B4=path ...

The recipe's yolo_A is v4a_s2 (models/trained/expV4A_ctrl_s2/seg05/best.pt). The v5b student control
S1_A (expV5bA_ctrl_s3) is a different checkpoint: fusing it is a substitution and is flagged in the
metadata (yolo_component_matches_recipe=false) and in the output name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.v5b.fusion import FusionRecipe, fuse, image_score, load_recipe  # noqa: E402

V4T, V5T, V4V, V5V = "reports/v4/test", "reports/v5b/test", "reports/v4", "reports/v5b/val"
DINO_RUN = {"dino_B4": "v4B4_lora_L", "dino_B6": "v4B6_lora_Hplus", "dino_B1": "v4B1_frozen_L"}
RECIPE_YOLO_ALIAS = "v4a_s2"           # recipe yolo_A checkpoint = models/trained/expV4A_ctrl_s2/seg05/best.pt
RECIPE_YOLO_CKPT = "models/trained/expV4A_ctrl_s2/seg05/best.pt"
# per set: DINO parquet pattern ({run}), named yolo parquets, eval yaml/split, expected rows, output dir
SETS = {
    "sdistest": dict(dino=f"{V4T}/eval_{{run}}_sdis_test.raw.parquet", n=4124, outdir=V5T,
                     yolo={"v4a_s2": f"{V4T}/eval_S1_A_ctrl_sdistest.raw.parquet"},
                     yaml="data/yolo/yamls/local/expB.yaml", split="test", role="H2 pyro-sdis TEST (second opening)"),
    "borealtest": dict(dino=f"{V4T}/eval_{{run}}_boreal_test.raw.parquet", n=1764, outdir=V5T,
                       yolo={"v4a_s2": f"{V4T}/eval_S1_A_ctrl_borealtest.raw.parquet"},
                       yaml="data/v4/eval/yamls/boreal_test.yaml", split="test", role="H4 historical (opened at v4 close)"),
    "foresttest": dict(dino=f"{V4T}/eval_{{run}}_forest_test.raw.parquet", n=2871, outdir=V5T,
                       yolo={"v4a_s2": f"{V4T}/eval_S1_A_ctrl_foresttest.raw.parquet"},
                       yaml="data/v4/eval/yamls/forest_test.yaml", split="test", role="H4 historical (permission pending)"),
    "firemanholdout": dict(dino=f"{V5T}/eval_{{run}}_firemanholdout.raw.parquet", n=1949, outdir=V5T, yolo={},
                           yaml="data/v5b/eval/yamls/fireman_holdout_s10.yaml", split="test",
                           role="H1 SEALED (recall-only, 0 negatives)"),
    "sheffieldholdout": dict(dino=f"{V5T}/eval_{{run}}_sheffieldholdout.raw.parquet", n=327, outdir=V5T, yolo={},
                             yaml="data/v5b/eval/yamls/sheffield_holdout.yaml", split="test",
                             role="H3 within-incident only (recall-only; S5 is H3-clean, v5b arms B/C/D are not)"),
    "sdisval": dict(dino=f"{V4V}/eval_{{run}}_sdis_val.raw.parquet", n=7335, outdir=V5V,
                    yolo={"v4a_s2": f"{V4V}/eval_v4a_s2_sdisval.raw.parquet"},
                    yaml="data/yolo/yamls/local/expB.yaml", split="val", role="VALIDATION (fusion fit here; in-sample)"),
}


def resolve_yolo(set_name: str, alias: str) -> str:
    cfg = SETS[set_name]
    if alias in cfg["yolo"]:
        return cfg["yolo"][alias]
    if Path(alias).exists():                       # explicit parquet path
        return alias
    return f"{cfg['outdir']}/eval_{alias}_{set_name}.raw.parquet"   # v5b naming eval_<sys>_<set>


def default_sources(set_name: str, yolo_alias: str) -> dict[str, str]:
    cfg = SETS[set_name]
    src = {"yolo_A": resolve_yolo(set_name, yolo_alias)}
    for m, run in DINO_RUN.items():
        src[m] = cfg["dino"].format(run=run)
    return src


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_parquet(path: str | Path):
    df = pd.read_parquet(path).sort_values("image").reset_index(drop=True)
    if list(df.columns) != ["image", "gt", "preds"]:
        raise SystemExit(f"{path}: columns {list(df.columns)} != ['image','gt','preds']")
    gts = [np.asarray([list(g) for g in g_], float).reshape(-1, 5) for g_ in df["gt"]]
    preds = [np.asarray([list(p)[:5] for p in p_], float).reshape(-1, 5) for p_ in df["preds"]]
    return df["image"].tolist(), gts, preds


def fuse_set(sources: dict[str, str], rec: FusionRecipe):
    """-> images, gts, fused boxes per image (np (K,5)), image scores, per-model raw preds."""
    missing = [f"{m}: {p}" for m, p in sources.items() if not Path(p).exists()]
    if missing:
        raise SystemExit("missing component parquet(s):\n  " + "\n  ".join(missing))
    if set(sources) != set(rec.order):
        raise SystemExit(f"model set {sorted(sources)} != recipe {rec.order} (ACCURATE = all components, strict)")
    images = gts = None
    preds: dict[str, list[np.ndarray]] = {}
    for m in rec.order:
        im, g, p = load_parquet(sources[m])
        if images is None:
            images, gts = im, g
        else:
            if im != images:
                a, b = set(im), set(images)
                raise SystemExit(f"{m}: image set differs from yolo_A ({len(a - b)} extra, {len(b - a)} missing)")
            for i, (x, y) in enumerate(zip(g, gts)):
                if x.shape != y.shape or (len(x) and not np.allclose(x, y, atol=1e-6)):
                    raise SystemExit(f"{m}: GT differs from yolo_A at row {i} ({images[i]})")
        preds[m] = p
    fused, scores = [], []
    for i in range(len(images)):
        pb = {m: preds[m][i].tolist() for m in rec.order}
        fused.append(np.asarray(fuse(pb, rec), float).reshape(-1, 5))
        scores.append(image_score(pb, rec))
    return images, gts, fused, np.asarray(scores, float), preds


def recall_at_recipe_thresholds(rows: list[dict], scores: np.ndarray, rec: FusionRecipe) -> dict:
    pos = np.array([len(r["gt"]) > 0 for r in rows])
    out = {"box_level": {}, "image_level": {}, "n_images": len(rows),
           "n_positive_images": int(pos.sum()), "n_negative_images": int((~pos).sum()),
           "note": "calibrated-scale thresholds from the recipe (thresholds_oof = pooled OOF sdis-VAL neg-image-FP "
                   "quantiles; fixed-camera regime). Sets with 0 negatives are recall-only."}
    thr = {k: float(v) for k, v in rec.thresholds.items()}
    if thr:
        try:
            from src.eval.eval_detection import sweep_report  # noqa: PLC0415  (imports torch)
            rep = sweep_report(rows, 0.4, conf_grid=sorted(set(thr.values())))
            by_conf = {s["conf"]: s for s in rep["conf_sweep"]}
            for k, c in thr.items():
                s = by_conf[c]
                out["box_level"][k] = {"thr": c, "recall": s["recall"], "tiny": s["recall_by_size"]["tiny"],
                                       "small": s["recall_by_size"]["small"], "precision": s["precision"],
                                       "neg_image_fp_rate": s["neg_image_fp_rate"]}
        except Exception as e:  # pragma: no cover
            out["box_level"] = {"error": f"{type(e).__name__}: {e}"}
    for k, c in (rec.image_level.get("thresholds_oof") or {}).items():
        c = float(c)
        out["image_level"][k] = {"thr": c,
                                 "pos_image_recall": round(float((scores[pos] >= c).mean()), 4) if pos.any() else None,
                                 "neg_image_fp_rate": round(float((scores[~pos] >= c).mean()), 4) if (~pos).any() else None}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=sorted(SETS), help="evaluation set (v5b naming)")
    ap.add_argument("--yolo", default="S1_A",
                    help="yolo_A source: alias (v4a_s2 = recipe's own; S1_A, S2_B, yolo_A... = reports/<dir>/eval_<alias>_<set>) or a parquet path")
    ap.add_argument("--model", action="append", default=[], help="override a component: name=path.raw.parquet")
    ap.add_argument("--recipe", default=None, help="recipe json (default models/v5b/fusion/accurate_recipe.json)")
    ap.add_argument("--out", default=None, help="output prefix (default <outdir>/eval_S5_ensemble_<yolo>_<set>)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    cfg = SETS[args.set]
    rec = load_recipe(args.recipe)
    sources = default_sources(args.set, args.yolo)
    for spec in args.model:
        name, path = spec.split("=", 1)
        sources[name] = path
    yolo_tag = args.yolo if not Path(args.yolo).exists() else Path(args.yolo).stem.replace(".raw", "")
    out = Path(args.out) if args.out else ROOT / cfg["outdir"] / f"eval_S5_ensemble_{yolo_tag}_{args.set}"
    out.parent.mkdir(parents=True, exist_ok=True)

    images, gts, fused, scores, _ = fuse_set(sources, rec)
    if len(images) != cfg["n"]:
        print(f"WARNING: {len(images)} rows, expected {cfg['n']} for {args.set}", file=sys.stderr)
    rows = [{"image": im, "gt": g.tolist(), "preds": f.tolist()} for im, g, f in zip(images, gts, fused)]
    pd.DataFrame(rows, columns=["image", "gt", "preds"]).to_parquet(out.with_suffix(".raw.parquet"))
    with open(out.with_suffix(".jsonl"), "w") as fh:
        for r, s in zip(rows, scores):
            fh.write(json.dumps({"image": r["image"], "preds": r["preds"], "image_score": float(s)}) + "\n")
    matches = args.yolo == RECIPE_YOLO_ALIAS or sources["yolo_A"] == cfg["yolo"].get(RECIPE_YOLO_ALIAS)
    meta = {
        "system": "S5 ACCURATE ensemble", "set": args.set, "role": cfg["role"], "data_yaml": cfg["yaml"], "split": cfg["split"],
        "recipe": {"path": str(args.recipe or "models/v5b/fusion/accurate_recipe.json"), "name": rec.name,
                   "created_utc": rec.raw.get("created_utc"), "fusion": rec.raw["fusion"],
                   "score_scale": rec.raw.get("score_scale"), "image_level_method": rec.image_level.get("method")},
        "components": {m: {"parquet": str(p), "sha256": sha256(p), "checkpoint_in_recipe": rec.models[m]["checkpoint"]}
                       for m, p in sources.items()},
        "yolo_alias": args.yolo, "yolo_component_matches_recipe": bool(matches), "recipe_yolo_checkpoint": RECIPE_YOLO_CKPT,
        "n_images": len(rows), "n_positive_images": int(sum(len(r["gt"]) > 0 for r in rows)),
        "n_negative_images": int(sum(len(r["gt"]) == 0 for r in rows)), "n_gt_boxes": int(sum(len(r["gt"]) for r in rows)),
        "n_fused_boxes": int(sum(len(r["preds"]) for r in rows)),
        "at_recipe_thresholds": recall_at_recipe_thresholds(rows, scores, rec),
        "outputs": {"raw_parquet": str(out.with_suffix(".raw.parquet")), "preds_jsonl": str(out.with_suffix(".jsonl"))},
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "no image opened; no fitting; fused scores are on the calibrated scale (1-c = single-model sdis-VAL neg-image FP)",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    if not args.quiet:
        print(f"S5 fused {args.set}: {len(rows)} rows ({meta['n_positive_images']} pos / {meta['n_negative_images']} neg), "
              f"{meta['n_fused_boxes']} fused boxes | yolo_A={args.yolo}"
              f"{'' if matches else '  ** NOT the recipe yolo (v4a_s2) -- substitution **'}")
        bl = meta["at_recipe_thresholds"]["box_level"]
        if bl and "error" not in bl:
            print("  box recall @recipe thr: " + "  ".join(f"{k}:{v['recall']}" for k, v in bl.items()))
        il = meta["at_recipe_thresholds"]["image_level"]
        if il:
            print("  image recall @recipe thr: " + "  ".join(f"{k}:{v['pos_image_recall']}" for k, v in il.items()))
        print(f"  -> {out}.raw.parquet / .jsonl / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
