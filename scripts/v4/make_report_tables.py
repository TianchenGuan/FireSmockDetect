"""Generate EVERY numeric table of the iteration-4 final report from saved artifacts.

Nothing here is typed by hand: each table is rendered from a JSON/CSV/MD artifact
under reports/ and cites its source path in the caption. Denominators (n_images,
n_neg, units/events) are stated in every caption. Missing artifacts degrade to a
"PENDING" placeholder (final TEST tables) or an explicit "artifact missing" note,
never to a fabricated number.

Inputs (all optional except seed_summary):
  reports/v4/seed_summary.json                      3-seed val matrix + selection rules
  reports/v4/matched_fp_{sdisval,borealval,forestval}.json   seed-0 YOLO arms (+expE_iter2)
  reports/v4/matched_fp_dino_{sdis,boreal,forest}_val.json   DINOv3 runs + YOLO refs
  reports/v4/dino_adapt_results.md                  param counts / training cost / latency tables
  reports/v4/replay/replay_{v4a,v4e,v4c}_figlib39.json, reports/v4/replay_{v4B4_lora_L,v4B6_lora_Hplus}_figlib39.json,
  reports/replay_expE_figlib.json (+ reports/v3/replay_B1_figlib_valmatched.json if present)
  reports/v4/trackd_v4_eval.json                    temporal / early-warning track
  reports/v4/COMPUTE_LEDGER.csv (+ cluster/v4/BUDGET_V4.json for the cap)
  reports/v4/test/{matched_fp_sdistest,yolo_test_at_preregistered_thresholds,dino_test_at_preregistered_thresholds,
                   forest_test_per_unit,dino_test_thresholds}.json   FINAL TEST (T25-T28; PENDING if absent)
  reports/v4/eval_*.raw.parquet                     optional: exact recall at val-selected confs
                                                    via the frozen matching code (no model is run)

Outputs: reports/v4/tables/T??_*.md (one file per table) + reports/v4/tables/ALL_TABLES.md

Usage:  .venv/bin/python scripts/v4/make_report_tables.py [--out reports/v4/tables]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports"
V4 = R / "v4"

FP_KEYS = ["fp0.01", "fp0.02", "fp0.05", "fp0.10", "fp0.20"]
FP_LABEL = {"fp0.01": "1%", "fp0.02": "2%", "fp0.05": "5%", "fp0.10": "10%", "fp0.20": "20%"}
CONF_KEYS = ["conf0.10", "conf0.20", "conf0.30", "conf0.50"]

ARMS = ["A_ctrl_E", "E_ET", "C_EGT", "D_curr", "B_Gonly", "expE_iter2"]
ARM_DESC = {
    "A_ctrl_E": "E only (existing-data control)",
    "E_ET": "E + T (target aerial/forest data)",
    "C_EGT": "E + G + T (G, T at 2x multiplicity)",
    "D_curr": "curriculum: G pretrain -> E+T (+15% G replay)",
    "B_Gonly": "G only (generic smoke, source-only selection)",
    "expE_iter2": "iteration-2 expE YOLO11m (E only, deployed candidate)",
}
ARM_CKPT = {"A_ctrl_E": "v4a", "E_ET": "v4e", "C_EGT": "v4c", "D_curr": "v4dada", "B_Gonly": "v4b"}
DINO_RUNS = ["v4B1_frozen_L", "v4B4_lora_L", "v4B5_full_L", "v4B6_lora_Hplus"]
DINO_REF_TO_CKPT = {"YOLO_A_ctrlE": "v4a", "YOLO_E_ET": "v4e", "YOLO_C_EGT": "v4c"}
# eval_<sys>_<split>.json system tokens -> the val-artifact names used for thresholds
TEST_SYS_TO_ARM = {"v4a": "A_ctrl_E", "v4e": "E_ET", "v4c": "C_EGT", "v4b": "B_Gonly", "v4dada": "D_curr",
                   "expE": "expE_iter2", "expE_y11m": "expE_iter2"}

MISSING: list[str] = []
NOTES: list[str] = []
TABLES: "OrderedDict[str, str]" = OrderedDict()


# ----------------------------------------------------------------------------- helpers
def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def load_json(p: Path):
    if not p.exists():
        MISSING.append(rel(p))
        return None
    return json.loads(p.read_text())


def f3(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}"


def pct(x, nd=1):
    return "n/a" if x is None else f"{100 * x:.{nd}f}%"


def ms(agg: dict | None, nd=3):
    """mean ± std (n) from an aggregate record."""
    if not agg or agg.get("mean") is None:
        return "n/a"
    n = agg.get("n", 0)
    if n <= 1:
        return f"{agg['mean']:.{nd}f} (n=1)"
    return f"{agg['mean']:.{nd}f} ± {agg['std']:.{nd}f} (n={n})"


def md_table(header: list[str], rows: list[list], align_first=True) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def emit(tid: str, title: str, caption: str, body: str, sources: list[str]):
    src = "\n".join(f"- `{s}`" for s in sources)
    text = f"### {tid}. {title}\n\n{caption}\n\n{body}\n\nSources:\n{src}\n"
    TABLES[tid] = text


def curve_edge(curve: list[dict], target: float):
    """Explain a None matched-FP cell from the model's own sweep curve."""
    if not curve:
        return "n/a"
    fps = [c["neg_image_fp_rate"] for c in curve if c.get("neg_image_fp_rate") is not None]
    if not fps:
        return "n/a"
    if target > max(fps):
        lo = min(curve, key=lambda c: c["conf"])
        return f"FP ceiling {100 * max(fps):.1f}% -> R>={lo['recall']:.3f}"
    if target < min(fps):
        return f"n/a (FP floor {100 * min(fps):.1f}%)"
    return "n/a"


def cell_rtc(entry: dict | None, curve: list[dict], target: float) -> str:
    if not entry or entry.get("recall") is None:
        return curve_edge(curve, target)
    return f"{entry['recall']:.3f} / {entry.get('tiny_recall', entry.get('tiny')):.3f} (c={entry['conf']:.3f})"


# --- exact recall at arbitrary confidences from saved predictions (frozen matching code)
_SWEEP = None


def _sweep_fn():
    global _SWEEP
    if _SWEEP is None:
        try:
            sys.path.insert(0, str(ROOT))
            from src.eval.eval_detection import sweep_report  # noqa
            _SWEEP = sweep_report
        except Exception as e:  # pragma: no cover
            NOTES.append(f"sweep_report import failed ({e!r}); recall-at-threshold falls back to nearest grid conf")
            _SWEEP = False
    return _SWEEP


_PQ_CACHE: dict = {}


def recall_at_confs_exact(parquet: Path, confs: list[float]):
    """{conf: sweep-entry} computed at exactly these confs from a .raw.parquet, or None."""
    if not parquet.exists():
        return None
    fn = _sweep_fn()
    if not fn:
        return None
    key = (str(parquet), tuple(sorted(set(round(c, 6) for c in confs))))
    if key in _PQ_CACHE:
        return _PQ_CACHE[key]
    import pandas as pd
    rows = pd.read_parquet(parquet).to_dict("records")
    rep = fn(rows, 0.4, conf_grid=sorted(set(round(c, 6) for c in confs)))
    out = {round(s["conf"], 6): s for s in rep["conf_sweep"]}
    out["_n"] = (rep["n_images"], rep["n_positive_images"], rep["n_negative_images"])
    _PQ_CACHE[key] = out
    return out


def nearest_rac(rac: dict, conf: float):
    """Nearest grid entry from a recall_at_conf dict {'conf0.xx': {...}}."""
    best = min(rac.items(), key=lambda kv: abs(float(kv[0][4:]) - conf))
    return float(best[0][4:]), best[1]


def interp_at_fp(sweep: list[dict], target: float, key):
    pts = [(s["neg_image_fp_rate"], key(s)) for s in sweep
           if s.get("neg_image_fp_rate") is not None and key(s) is not None]
    if not pts:
        return None
    fps = [p[0] for p in pts]
    if target > max(fps) or target < min(fps):
        return None
    for (f1, v1), (f2, v2) in zip(pts, pts[1:]):
        lo, hi = min(f1, f2), max(f1, f2)
        if lo <= target <= hi:
            if f1 == f2:
                return round(float(v1), 4)
            w = (target - f1) / (f2 - f1)
            return round(float(v1 + w * (v2 - v1)), 4)
    return None


def interp_at_conf(sweep: list[dict], conf: float):
    """Linear interpolation of (recall, tiny) vs conf on a sweep list."""
    pts = sorted((s["conf"], s["recall"], s["recall_by_size"]["tiny"]) for s in sweep)
    if conf <= pts[0][0]:
        return pts[0][1], pts[0][2], pts[0][0]
    if conf >= pts[-1][0]:
        return pts[-1][1], pts[-1][2], pts[-1][0]
    for a, b in zip(pts, pts[1:]):
        if a[0] <= conf <= b[0]:
            w = 0 if b[0] == a[0] else (conf - a[0]) / (b[0] - a[0])
            return a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2]), conf
    return None, None, conf


def parse_md_tables(md: Path) -> list[list[list[str]]]:
    """All pipe tables in a markdown file -> list of [rows], each row a list of cells."""
    if not md.exists():
        MISSING.append(rel(md))
        return []
    tables, cur = [], []
    for ln in md.read_text().splitlines():
        if ln.strip().startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                continue
            cur.append(cells)
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return tables


def find_md_table(tables, first_headers: list[str]):
    for t in tables:
        if t and [h.lower() for h in t[0][:len(first_headers)]] == [h.lower() for h in first_headers]:
            return t
    return None


# ----------------------------------------------------------------------------- 1. seed matrix
def denominators_from_matched(path: Path):
    d = load_json(path)
    if not d:
        return None, {}
    m = next(iter(d["models"].values()))
    return (m["n_images"], m["n_positive_images"], m["n_negative_images"]), d


def seed_tables():
    ss = load_json(V4 / "seed_summary.json")
    if not ss:
        return
    dens = {}
    for vs in ["sdisval", "borealval", "forestval"]:
        dens[vs], _ = denominators_from_matched(V4 / f"matched_fp_{vs}.json")

    def den_txt(vs, extra=""):
        d = dens.get(vs)
        if not d:
            return "denominators unavailable (matched_fp json missing)"
        return f"n_images={d[0]:,} (positives {d[1]:,}, negatives {d[2]:,}){extra}"

    arms = ss["arms"]
    order = [a for a in ["A_ctrl_E", "E_ET", "C_EGT", "D_curr", "B_Gonly"] if a in arms]

    # T01 sdis
    rows = []
    for a in order:
        ag = arms[a].get("sdisval", {}).get("agg", {})
        n = ag.get("fp0.10.recall", {}).get("n", 0)
        rows.append([a, ARM_DESC.get(a, ""), n] + [ms(ag.get(f"{k}.recall")) for k in FP_KEYS]
                    + [ms(ag.get("fp0.05.tiny")), ms(ag.get("fp0.10.tiny"))])
    emit("T01", "Transfer matrix, pyro-sdis VAL (fixed-camera source domain): recall at matched neg-image FP, 3 seeds",
         f"Cell = mean ± std over seeds (n seeds in parentheses; n=1 arms are single runs). Matching IoU>=0.4. "
         f"Denominators: {den_txt('sdisval')}; 1% neg-FP = ~{round(0.01 * dens['sdisval'][2]) if dens.get('sdisval') else 'n/a'} negative images. "
         "tiny = GT boxes <0.1% of image area.",
         md_table(["arm", "recipe", "seeds", "R@1%", "R@2%", "R@5%", "R@10%", "R@20%", "tiny R@5%", "tiny R@10%"], rows),
         ["reports/v4/seed_summary.json", "reports/v4/matched_fp_sdisval.json (denominators)"])

    # T02 boreal
    rows = []
    for a in order:
        ag = arms[a].get("borealval", {}).get("agg", {})
        n = max([v.get("n", 0) for v in ag.values()] or [0])
        rows.append([a, n] + [ms(ag.get(f"{k}.recall")) for k in FP_KEYS])
    nneg = dens["borealval"][2] if dens.get("borealval") else None
    emit("T02", "Transfer matrix, boreal AERIAL VAL (burn_heinola, 1 burn): recall at matched neg-image FP, 3 seeds",
         f"Cell = mean ± std (n seeds that reach the FP target; 'n/a' = no seed's curve reaches it — an FP ceiling: "
         f"the model never exceeds that FP even at its lowest confidence, so recall is at least its max, see T07). "
         f"Denominators: {den_txt('borealval')}; ONE independent burn; 1% neg-FP = ~{round(0.01 * nneg) if nneg else 'n/a'} negative images, "
         "so the 1% column has a resolution of ~0.5pt FP and a seed std of ~5pt is expected. Tiny recall is 0 for every arm "
         "(aerial plumes are never <0.1% area; boreal_val has 1 tiny + 5 small GT boxes) and is omitted.",
         md_table(["arm", "seeds", "R@1%", "R@2%", "R@5%", "R@10%", "R@20%"], rows),
         ["reports/v4/seed_summary.json", "reports/v4/matched_fp_borealval.json (denominators)"])

    # T03 forest
    rows = []
    for a in order:
        ag = arms[a].get("forestval", {}).get("agg", {})
        n = max([v.get("n", 0) for v in ag.values()] or [0])
        rows.append([a, n] + [ms(ag.get(f"{k}.recall")) for k in CONF_KEYS] + [ms(ag.get("conf0.20.tiny")), ms(ag.get("conf0.30.tiny"))])
    emit("T03", "Transfer matrix, AuSmoke FOREST VAL (new region, positives-only): recall at fixed confidence, 3 seeds",
         f"forest_val has NO negative images, so matched-FP is undefined; recall is reported at fixed YOLO confidences "
         f"(comparable across YOLO arms only — never against DINO scores). Denominators: {den_txt('forestval')}; "
         "4 units / ~3 independent events (Axis pair = same fixed camera). False-alarm behaviour on this domain is unmeasured.",
         md_table(["arm", "seeds", "R@conf0.10", "R@conf0.20", "R@conf0.30", "R@conf0.50", "tiny R@0.20", "tiny R@0.30"], rows),
         ["reports/v4/seed_summary.json", "reports/v4/matched_fp_forestval.json (denominators)"])

    # T04 selection
    sel = ss.get("selection", {})
    rows = []
    for k, v in sel.items():
        if isinstance(v, dict) and "mean" in v:
            rows.append([k, f"mean {v['mean']:.4f}, std {v['std']:.4f}, min {v['min']:.4f}, max {v['max']:.4f}, n={v['n']}"])
        elif isinstance(v, dict) and "ckpt" in v:
            rows.append([k, f"ckpt `{v['ckpt']}` (sdis R@10% = {v['recall']:.4f}; n_seeds={v['n_seeds']})"])
        else:
            rows.append([k, str(v)])
    emit("T04", "Pre-registered selection quantities (computed on VAL only)",
         "S2 rule: the v4 data arm with the higher 3-seed mean boreal-val R@1% neg-FP; tie <1pt -> E_ET. "
         "Median-seed rule: the seed whose sdis-val R@10% is the median of its arm is the one carried to TEST. "
         f"Denominators as in T01/T02 (boreal 1% = ~{round(0.01 * nneg) if nneg else 'n/a'} of {nneg} negatives).",
         md_table(["quantity", "value"], rows),
         ["reports/v4/seed_summary.json (selection block)", "reports/v4/PREREGISTRATION_V4.md (rules)"])

    # T05 per-seed detail
    rows = []
    for a in order:
        ps_s = arms[a].get("sdisval", {}).get("per_seed", {})
        ps_b = arms[a].get("borealval", {}).get("per_seed", {})
        ps_f = arms[a].get("forestval", {}).get("per_seed", {})
        for ck in sorted(set(ps_s) | set(ps_b) | set(ps_f)):
            s, b, f = ps_s.get(ck, {}), ps_b.get(ck, {}), ps_f.get(ck, {})
            g = lambda d, k, m: (d.get(k) or {}).get(m)
            rows.append([a, ck,
                         f3(g(s, "fp0.01", "recall")), f3(g(s, "fp0.10", "recall")), f3(g(s, "fp0.10", "tiny")),
                         f3(g(s, "fp0.05", "conf")), f3(g(s, "fp0.10", "conf")),
                         f3(g(b, "fp0.01", "recall")), f3(g(b, "fp0.05", "recall")), f3(g(b, "fp0.01", "conf")),
                         f3(g(f, "conf0.20", "recall")), f3(g(f, "conf0.30", "recall"))])
    emit("T05", "Per-seed detail (every checkpoint behind T01-T03) incl. the val-selected confidences used as thresholds",
         "conf@X% = the checkpoint's own confidence at X% neg-image FP on that val set (these are the SOURCE-ONLY regime "
         "thresholds from sdis_val and the TARGET-ADAPTED thresholds from boreal_val in PREREGISTRATION_V4.md). "
         f"Denominators: sdis {den_txt('sdisval')}; boreal {den_txt('borealval')}; forest {den_txt('forestval')}. 'n/a' = FP target not reachable.",
         md_table(["arm", "ckpt", "sdis R@1%", "sdis R@10%", "sdis tiny@10%", "sdis conf@5%", "sdis conf@10%",
                   "boreal R@1%", "boreal R@5%", "boreal conf@1%", "forest R@0.20", "forest R@0.30"], rows),
         ["reports/v4/seed_summary.json"])


# ----------------------------------------------------------------------------- 2. seed-0 matched-FP arms
def matched_fp_table(tid, title, path: Path, models_order: list[str], caption_extra: str, sources: list[str], desc: dict | None = None):
    d = load_json(path)
    if not d:
        emit(tid, title, f"ARTIFACT MISSING: `{rel(path)}`", "_no table_", [rel(path)])
        return None
    models = [m for m in models_order if m in d["models"]] + [m for m in d["models"] if m not in models_order]
    m0 = d["models"][models[0]]
    hdr = ["neg-image FP target"] + models
    rows = []
    for k in FP_KEYS:
        t = float(k[2:])
        rows.append([FP_LABEL[k]] + [cell_rtc(d["matched_fp"][m].get(k), d["models"][m]["curve"], t) for m in models])
    legend = ""
    if desc:
        legend = "\n\n" + md_table(["column", "system"], [[m, desc.get(m, "")] for m in models])
    emit(tid, title,
         f"Cell = recall / tiny-recall (conf at which this model sits at the FP target), interpolated on each model's own "
         f"curve with the frozen matching code (IoU>={d['match_iou']}). Denominators: n_images={m0['n_images']:,} "
         f"(positives {m0['n_positive_images']:,}, negatives {m0['n_negative_images']:,}); 1% FP = ~{round(0.01 * m0['n_negative_images'])} negative images. "
         + caption_extra,
         md_table(hdr, rows) + legend, sources)
    return d


def forest_regime_table(tid, title, forest_path: Path, sdis_path: Path, boreal_path: Path, models_order: list[str],
                        pq_for_model, sources: list[str], extra_caption=""):
    """Recall on a positives-only split at (a) fixed confs, (b) SOURCE-ONLY thresholds (sdis-val conf @5%/10%),
    (c) TARGET-ADAPTED thresholds (boreal-val conf @1%/5%)."""
    fd = load_json(forest_path)
    if not fd:
        emit(tid, title, f"ARTIFACT MISSING: `{rel(forest_path)}`", "_no table_", [rel(forest_path)])
        return
    sd = load_json(sdis_path) or {}
    bd = load_json(boreal_path) or {}
    models = [m for m in models_order if m in fd["models"]] + [m for m in fd["models"] if m not in models_order]
    m0 = fd["models"][models[0]]
    rac = fd["recall_at_conf"]
    # (a) fixed confs
    rows_a = []
    for ck in CONF_KEYS:
        rows_a.append([ck.replace("conf", "conf=")] + [
            (f"{rac[m][ck]['recall']:.3f} / {rac[m][ck]['tiny_recall']:.3f}" if ck in rac[m] else "n/a") for m in models])
    # (b)/(c) regimes
    regimes = [("SOURCE-ONLY: sdis-val conf @5% negFP", sd, "fp0.05"),
               ("SOURCE-ONLY: sdis-val conf @10% negFP", sd, "fp0.10"),
               ("TARGET-ADAPTED: boreal-val conf @1% negFP", bd, "fp0.01"),
               ("TARGET-ADAPTED: boreal-val conf @5% negFP", bd, "fp0.05")]
    rows_b = []
    approx_used = False
    for label, src, k in regimes:
        row = [label]
        for m in models:
            conf = ((src.get("matched_fp") or {}).get(m) or {}).get(k, {}).get("conf") if src else None
            if conf is None:
                row.append("n/a (no val threshold)")
                continue
            ex = recall_at_confs_exact(pq_for_model(m), [conf])
            if ex and round(conf, 6) in ex:
                s = ex[round(conf, 6)]
                row.append(f"{s['recall']:.3f} / {s['recall_by_size']['tiny']:.3f} (c={conf:.3f})")
            else:
                c_used, e = nearest_rac(rac[m], conf)
                approx_used = True
                row.append(f"{e['recall']:.3f} / {e['tiny_recall']:.3f} (c={conf:.3f}, grid≈{c_used:.2f})")
        rows_b.append(row)
    cap = (f"forest_val is positives-only (n_images={m0['n_images']:,}, negatives {m0['n_negative_images']}), so matched-FP is undefined. "
           "(a) recall / tiny-recall at fixed confidences (comparable only within one score family). "
           "(b) the two pre-registered threshold regimes: SOURCE-ONLY = each model's own sdis-val confidence at 5%/10% neg-FP applied "
           "unchanged; TARGET-ADAPTED = its boreal-val confidence at 1%/5% neg-FP (217 boreal negatives). "
           "Regime rows are computed at the exact confidence from the saved predictions with the frozen matching code where the "
           ".raw.parquet exists" + (" (rows marked grid≈ fall back to the nearest 0.01 grid point)" if approx_used else "") + ". " + extra_caption)
    emit(tid, title, cap, "**(a) fixed confidence**\n\n" + md_table(["threshold"] + models, rows_a)
         + "\n\n**(b) val-selected thresholds (pre-registered regimes)**\n\n" + md_table(["regime"] + models, rows_b), sources)


def seed0_tables():
    matched_fp_table("T06", "Seed-0 arms incl. D_curr and expE_iter2, pyro-sdis VAL: recall / tiny at matched neg-image FP",
                     V4 / "matched_fp_sdisval.json", ARMS,
                     "Seed 0 only (3-seed means in T01). D_curr is a single run.",
                     ["reports/v4/matched_fp_sdisval.json"], ARM_DESC)
    matched_fp_table("T07", "Seed-0 arms incl. D_curr, boreal AERIAL VAL (burn_heinola): recall at matched neg-image FP",
                     V4 / "matched_fp_borealval.json", ARMS,
                     "ONE independent burn. Tiny recall is structurally 0 on this set (1 tiny + 5 small GT boxes). "
                     "'FP ceiling' = the model never exceeds that neg-FP even at its lowest confidence.",
                     ["reports/v4/matched_fp_borealval.json"], ARM_DESC)
    forest_regime_table("T08", "Seed-0 arms incl. D_curr, AuSmoke FOREST VAL: recall at fixed confs and under both threshold regimes",
                        V4 / "matched_fp_forestval.json", V4 / "matched_fp_sdisval.json", V4 / "matched_fp_borealval.json", ARMS,
                        lambda m: V4 / f"eval_{ARM_CKPT.get(m, m)}_forestval.raw.parquet",
                        ["reports/v4/matched_fp_forestval.json", "reports/v4/matched_fp_sdisval.json (SOURCE-ONLY confs)",
                         "reports/v4/matched_fp_borealval.json (TARGET-ADAPTED confs)", "reports/v4/eval_<ckpt>_forestval.raw.parquet (exact recall at conf)"],
                        "4 units / ~3 independent events.")


# ----------------------------------------------------------------------------- 3. DINO adaptation
def dino_tables():
    order = DINO_RUNS + ["YOLO_A_ctrlE", "YOLO_E_ET", "YOLO_C_EGT", "expE_y11m_iter2", "v3_B1_frozenL_Eonly12ep", "v3_B3_top4L_Eonly12ep"]
    matched_fp_table("T09", "DINOv3 adaptation runs vs YOLO arms, pyro-sdis VAL: recall / tiny at matched neg-image FP",
                     V4 / "matched_fp_dino_sdis_val.json", order,
                     "v4B* = iteration-4 DINOv3 Faster R-CNN runs on the E+T corpus (B1 frozen ViT-L control, B4 LoRA r16 ViT-L, B5 full unfreeze ViT-L, "
                     "B6 LoRA r16 ViT-H+); YOLO_* = seed-0 iteration-4 YOLO arms; v3_* = iteration-3 E-only DINO runs. Compare ONLY at matched FP: "
                     "adapted DINO confidences saturate >=0.99 (see conf in cells) while YOLO sits at 0.5-0.8.",
                     ["reports/v4/matched_fp_dino_sdis_val.json"])
    matched_fp_table("T10", "DINOv3 adaptation runs vs YOLO arms, boreal AERIAL VAL: recall at matched neg-image FP",
                     V4 / "matched_fp_dino_boreal_val.json", order,
                     "ONE burn; differences among DINO runs are within a handful of images. Tiny recall structurally 0.",
                     ["reports/v4/matched_fp_dino_boreal_val.json"])

    def pq(m):
        if m in DINO_RUNS:
            return V4 / f"eval_{m}_forest_val.raw.parquet"
        if m in DINO_REF_TO_CKPT:
            return V4 / f"eval_{DINO_REF_TO_CKPT[m]}_forestval.raw.parquet"
        return V4 / "__none__.parquet"
    forest_regime_table("T11", "DINOv3 adaptation runs vs YOLO arms, AuSmoke FOREST VAL: fixed confs and both threshold regimes",
                        V4 / "matched_fp_dino_forest_val.json", V4 / "matched_fp_dino_sdis_val.json", V4 / "matched_fp_dino_boreal_val.json", order,
                        pq, ["reports/v4/matched_fp_dino_forest_val.json", "reports/v4/matched_fp_dino_sdis_val.json", "reports/v4/matched_fp_dino_boreal_val.json",
                             "reports/v4/eval_<run>_forest_val.raw.parquet"],
                        "Under SOURCE-ONLY thresholds the saturated (adapted) models transfer WORSE than frozen B1 even where their fixed-conf recall is higher: "
                        "this is a calibration/threshold-transfer effect, not a detection-quality ranking.")

    md = V4 / "dino_adapt_results.md"
    tabs = parse_md_tables(md)
    t_par = find_md_table(tabs, ["run", "backbone trainable / total"])
    t_cost = find_md_table(tabs, ["run", "status", "segments"])
    t_runs = find_md_table(tabs, ["run", "backbone", "adaptation"])
    if t_par and t_cost:
        cost = {r[0]: r for r in t_cost[1:]}
        runs = {r[0]: r for r in (t_runs[1:] if t_runs else [])}
        rows = []
        for r in t_par[1:]:
            run = r[0]
            c = cost.get(run, [""] * 8)
            rr = runs.get(run, [""] * 7)
            rows.append([run, rr[1] if rr else "", rr[2] if rr else "", r[1], r[2], r[3], r[4], c[3], c[5], c[6], c[7]])
        emit("T12", "DINOv3 runs: adaptation mode, parameter counts, training cost",
             "Parameter counts from checkpoints + HF config; node-hours = orchestrator-observed segment wall time on 8x L40S "
             "(incl. pod start and eval passes). best.pt selected by COCO mAP on a 1,500-image sdis-val subset. "
             "B5 trains at bs 2/GPU (2x the steps) and evaluates only at the end -> confounded vs B1/B4/B6 (see source).",
             md_table(["run", "backbone", "adaptation", "backbone trainable / total", "neck", "head", "ALL trainable",
                       "node-h (8xL40S)", "epochs (exposures)", "best COCO mAP @step", "OOM/retries"], rows),
             ["reports/v4/dino_adapt_results.md (tables 'Parameter counts', 'Training outcome / cost', 'Runs')"])
    else:
        emit("T12", "DINOv3 runs: parameter counts, training cost", f"ARTIFACT MISSING or table not found in `{rel(md)}`", "_no table_", [rel(md)])
    t_lat = find_md_table(tabs, ["system", "form", "mean ms"])
    if t_lat:
        rows = [r[:6] for r in t_lat[1:]]
        # append the CPU serving latency artifact if present
        lat = load_json(V4 / "dino3_backend_latency_B4_8thr.json")
        extra = ""
        if lat and "latency" in lat:
            L = lat["latency"]
            rows.append([f"v4B4_lora_L (serving backend `dino3`, CPU {L['torch_threads']} threads)", "LoRA unmerged",
                         f"{L['batch1_ms_mean']:.1f}", f"{L['batch1_ms_median']:.1f}", f"{L['batch1_ms_p90']:.1f}", f"{L['infer_ms_median']:.1f}"])
            extra = " Last row: devbox CPU (192 vCPU host, torch 8 threads) from the serving-backend verification."
        emit("T13", "Batch-1 latency (1x L40S fp32 unless stated; n=50 sdis_val frames; preprocessing included, JPEG decode excluded)",
             "v3 protocol. Merged form folds LoRA into the base linears (deploy form; outputs identical to 1e-6). "
             "expE YOLO11m reference on the same protocol: see source text (11.2 ms)." + extra,
             md_table(["system", "form", "mean ms", "median ms", "p90 ms", "infer-only median ms"], rows),
             ["reports/v4/dino_adapt_results.md (Batch-1 latency table)", "reports/v4/dino3_backend_latency_B4_8thr.json"])
    else:
        emit("T13", "Batch-1 latency", f"latency table not found in `{rel(md)}`", "_no table_", [rel(md)])


# ----------------------------------------------------------------------------- 4. FIgLib-39 replay
def replay_stats(d: dict):
    ev = d["events"]
    n = len(ev)
    det = {k: v for k, v in ev.items() if v.get("detected")}
    delays = {k: v["delay_from_first_visible_s"] for k, v in det.items()}
    r_at = {m: sum(1 for x in delays.values() if x is not None and 0 <= x <= m * 60) for m in (1, 3, 5)}
    return {"n": n, "detected": len(det), "missed": sorted(k for k, v in ev.items() if not v.get("detected")),
            "pre_visible": sorted(k for k, v in ev.items() if v.get("pre_visible_alert")),
            "median_delay": d["summary"].get("median_delay_s_among_detected"),
            "mean_delay": d["summary"].get("mean_delay_s_among_detected"),
            "r_at": r_at, "delays": delays, "cfg": d["summary"].get("config", {})}


def paired(a: dict, b: dict):
    both = sorted(set(a) & set(b))
    deltas = [a[k] - b[k] for k in both]
    return {"n_both": len(both), "earlier": sum(1 for x in deltas if x < 0), "tie": sum(1 for x in deltas if x == 0),
            "later": sum(1 for x in deltas if x > 0),
            "median": statistics.median(deltas) if deltas else None, "mean": statistics.mean(deltas) if deltas else None}


def figlib_tables():
    systems = [
        ("S0 expE_iter2 (deployed thresholds; paired-delay reference)", R / "replay_expE_figlib.json"),
        ("S1 A_ctrl TEST ckpt (v4a_s2)", V4 / "replay/replay_v4a_s2_figlib39.json"),
        ("S2 E_ET TEST ckpt (v4e_s1)", V4 / "replay/replay_v4e_s1_figlib39.json"),
        ("S2c C_EGT TEST ckpt (v4c_s1)", V4 / "replay/replay_v4c_s1_figlib39.json"),
        ("S3 D_curr TEST ckpt (v4dada_s1, post-hoc)", V4 / "replay/replay_v4dada_s1_figlib39.json"),
        ("A_ctrl_E seed0 (v4a)", V4 / "replay/replay_v4a_figlib39.json"),
        ("E_ET seed0 (v4e)", V4 / "replay/replay_v4e_figlib39.json"),
        ("C_EGT seed0 (v4c)", V4 / "replay/replay_v4c_figlib39.json"),
        ("S4p v4B4_lora_L (DINO, sdis-val-selected confs)", V4 / "replay_v4B4_lora_L_figlib39.json"),
        ("S4 v4B6_lora_Hplus (DINO, sdis-val-selected confs)", V4 / "replay_v4B6_lora_Hplus_figlib39.json"),
    ]
    opt = R / "v3/replay_B1_figlib_valmatched.json"
    if opt.exists():
        systems.append(("v3 B1 frozen ViT-L (iteration-3 reference)", opt))
    st = OrderedDict()
    for label, p in systems:
        if "TEST ckpt" in label and not p.exists():
            NOTES.append(f"FIgLib replay for TEST checkpoint missing: {rel(p)}")
            MISSING.append(rel(p)); continue
        d = load_json(p)
        if d:
            st[label] = (replay_stats(d), p)
    if not st:
        emit("T14", "FIgLib-39 replay", "ARTIFACT MISSING: no replay JSON found", "_no table_", [rel(p) for _, p in systems])
        return
    n = next(iter(st.values()))[0]["n"]
    rows = []
    for label, (s, p) in st.items():
        c = s["cfg"]
        thr = f"{c.get('single_frame_alert_conf', 'n/a')} / {c.get('seq_frame_conf', 'n/a')}"
        rows.append([label, thr, f"{s['detected']}/{s['n']}", len(s["pre_visible"]),
                     f"{s['r_at'][1]}/{s['n']} = {s['r_at'][1] / s['n']:.3f}", f"{s['r_at'][3]}/{s['n']} = {s['r_at'][3] / s['n']:.3f}",
                     f"{s['r_at'][5]}/{s['n']} = {s['r_at'][5] / s['n']:.3f}",
                     f3(s["median_delay"], 0) + " s", f3(s["mean_delay"], 0) + " s", len(s["missed"])])
    emit("T14", "FIgLib-39 early-detection replay (HISTORICAL set, used since iteration 1 — not a lockbox)",
         f"n_events = {n} HPWREN fire events (~80 frames each, 60 s cadence, 40 min before/after first visible smoke). "
         "detected = any alert at/after first-visible smoke OR a pre-visible alert that persists; pre-visible = alert before the "
         "labelled first-visible frame (unverifiable: early smoke or false alarm; NOT counted as false). R@N min = events with "
         "0 <= delay <= N min / n_events (pre-visible alerts enter at delay 0 s — the same rule as scripts/v4/figlib_replay_report.py). "
         "YOLO systems use the deployed 3-of-5 policy at 0.30 / 0.15; DINO runs use their own sdis-val confs at 10% / 20% neg-FP "
         "(the v3 threshold-transfer rule). Median/mean delay over detected events. 'TEST ckpt' rows are the exact checkpoints "
         "evaluated on the sealed test sets (median seeds; STATUS.md 03:20Z); seed0 rows are the earlier iteration-4 replays.",
         md_table(["system", "conf-single / conf-seq", "detected", "pre-visible", "R@1 min", "R@3 min", "R@5 min",
                   "median delay", "mean delay", "misses"], rows),
         [rel(p) for _, (s, p) in st.items()])

    ref_label = next(iter(st))
    ref = st[ref_label][0]
    rows = []
    for label, (s, p) in list(st.items())[1:]:
        pr = paired(s["delays"], ref["delays"])
        rows.append([label, pr["n_both"], pr["earlier"], pr["tie"], pr["later"],
                     (f"{pr['median']:+.0f} s" if pr["median"] is not None else "n/a"),
                     (f"{pr['mean']:+.0f} s" if pr["mean"] is not None else "n/a")])
    emit("T15", "FIgLib-39 paired detection delay vs expE (events detected by BOTH systems)",
         f"delta = system delay - expE delay per event (negative = earlier than expE); n_both = events detected by both of the {n}. "
         "Pre-visible detections carry delay 0 s on both sides. median = statistics.median (mean of the two middle values for even n; "
         "scripts/v4/figlib_replay_report.py uses the upper middle value, so its medians can differ by <=1 s).",
         md_table(["system", "n both", "earlier", "tie", "later", "median delta", "mean delta"], rows),
         [rel(p) for _, (s, p) in st.items()])

    # per-event matrix
    evs = sorted(set().union(*[set(load_json(p)["events"].keys()) for _, (s, p) in st.items()]))
    labels = list(st)
    short = [l.split(" (")[0] for l in labels]
    rows = []
    for e in evs:
        row = [e]
        for label in labels:
            s = st[label][0]
            if e in s["delays"]:
                d = s["delays"][e]
                row.append(("PRE " if e in s["pre_visible"] else "") + f"{d:.0f}")
            else:
                row.append("MISS")
        rows.append(row)
    emit("T16", "FIgLib-39 per-event detection delay (s from first-visible smoke; MISS = never alerted; PRE = pre-visible alert)",
         f"n_events = {n}. One row per event; a system's cell is its delay in seconds (0 for pre-visible alerts, prefixed PRE).",
         md_table(["event"] + short, rows), [rel(p) for _, (s, p) in st.items()])


# ----------------------------------------------------------------------------- 5. temporal track
def temporal_tables():
    rep = load_json(V4 / "trackd_v4_eval.json")
    if not rep:
        return
    order_hint = ["s1_expE", "s1_v4a_ctrlE", "s1_v4B1_dinoL", "s2_persist_3of5"]

    def sysorder(keys):
        return [k for k in order_hint if k in keys] + sorted(k for k in keys if k not in order_hint)

    A = rep["A_sdis_val_frame_level"]
    pc = rep.get("param_counts", {})
    rows = []
    for nme in sysorder(A["systems"]):
        e = A["systems"][nme]
        m = e["matched_fpr"]
        p = pc.get(nme, {}).get("n_params")
        rows.append([nme, f"{p:,}" if p else "0 (rule/raw)", f3(e["pr_auc"], 4), f3(e["roc_auc"], 4)]
                    + [f3(m[f"all_neg@{t}"]["recall_pos_frames"], 4) for t in ["0.005", "0.01", "0.02", "0.05", "0.1"]]
                    + [f3(m["all_neg@0.05"]["recall_tiny_frames"], 4)])
    emit("T17", "Early warning: pyro-sdis VAL frame-level recall at matched frame FPR (all-negative pool)",
         f"Frames {A['n_frames']:,} (positive {A['n_pos']:,} / negative {A['n_neg']:,}; clean-negative subset {A['n_clean_neg']:,}); "
         f"tiny-GT positive frames {A['n_tiny_pos_frames']:,}. s1_* = single-frame detector max score; s2_persist_3of5 = deployed persistence rule; "
         "tmp_* = learned causal temporal head (v3only / v4 = +AuSmoke extension / v4cap = capped 5-seq extension), t1_* = same module on the current frame only; "
         "_s0/_s1 = training seeds. Thresholds chosen on these val negatives.",
         md_table(["system", "params", "PR-AUC", "ROC-AUC", "R@FPR0.5%", "R@FPR1%", "R@FPR2%", "R@FPR5%", "R@FPR10%", "tinyR@FPR5%"], rows),
         ["reports/v4/trackd_v4_eval.json (A_sdis_val_frame_level)"])

    rows = []
    for nme in sysorder(A["systems"]):
        m = A["systems"][nme]["matched_fpr"]
        a, b = m["all_neg@0.01"], m["all_neg@0.05"]
        rows.append([nme, a["episodes"], f3(a["episode_detection_rate"], 4), a["median_onset_latency_frames"],
                     f3(b["episode_detection_rate"], 4), b["median_onset_latency_frames"]])
    dp = [[k, f3(v["fpr_all_neg"], 4), f3(v["fpr_clean_neg"], 4), f3(v["recall_pos_frames"], 4), f3(v["recall_tiny_frames"], 4)]
          for k, v in A["deployed_points"].items()]
    emit("T18", "Early warning: episode detection and onset latency on pyro-sdis VAL, plus the deployed discrete operating point",
         f"episodes = {next(iter(A['systems'].values()))['matched_fpr']['all_neg@0.01']['episodes']} positive smoke episodes in sdis_val sequences; "
         "latency in frames (29 s cadence) from episode start to first alert, median over detected episodes. Thresholds matched on frame FPR "
         f"over the {A['n_neg']:,} all-negative frames. Deployed point = 0.30 single / 0.15 seq, 3-of-5.",
         md_table(["system", "episodes", "det@FPR1%", "median latency (frames)", "det@FPR5%", "median latency (frames)"], rows)
         + "\n\n**Deployed discrete operating point (expE)**\n\n" + md_table(["point", "FPR all-neg", "FPR clean-neg", "recall pos frames", "recall tiny frames"], dp),
         ["reports/v4/trackd_v4_eval.json (A_sdis_val_frame_level)"])

    B = rep.get("B_v4_val_replay", {})
    parts = []
    srcs = B.get("sources", {})
    for g, v in srcs.items():
        wc = f"{v['wall_clock_h']:.3f} h" if v.get("wall_clock_h") else f"wall clock unknown ({v.get('wall_clock_note')})"
        head = (f"**{g}** — {v['sequences']} sequences / {v['frames']:,} frames, {wc}, truth_level={v['truth_level']}, "
                f"frame_truth={v['frame_truth']}, parent units={len(v['parent_units'])}")
        if v.get("notes"):
            head += "\n  - CAVEAT: " + "; ".join(v["notes"])
        rows = []
        for nme in sysorder(v["systems"]):
            r1 = v["systems"][nme].get("matched@fpr0.01") or {}
            r5 = v["systems"][nme].get("matched@fpr0.05") or {}
            rows.append([nme, f3(r1.get("threshold"), 4), pct(r1.get("frame_alert_rate")), pct(r1.get("clip_alert_rate")),
                         f3(r1.get("episodes_per_hour"), 2), pct(r5.get("frame_alert_rate")), pct(r5.get("clip_alert_rate"))])
        d = v.get("deployed_confirmed_or_provisional", {})
        rows.append(["DEPLOYED expE (prov OR conf, 0.30/0.15)", "-", pct(d.get("frame_alert_rate")), pct(d.get("clip_alert_rate")),
                     f3(d.get("episodes_per_hour"), 2), "-", "-"])
        parts.append(head + "\n\n" + md_table(["system", "thr@FPR1%", "frame alert @1%", "clip alert @1%", "episodes/h @1%", "frame alert @5%", "clip alert @5%"], rows))
    emit("T19", "Early warning: v4 VAL replay per source — the domain-shortcut evidence (thresholds from T17, sdis-val all-neg pool)",
         "For NEGATIVE footage (boreal/negative_footage = heinola non-smoke clips, firesense/negative_footage) every alert is false: "
         "the frame/clip alert rate IS the false-alarm rate. For positive sources the alert rate is a recall proxy only (boreal/positive is "
         "video-level truth; ausmoke/positive is positive from frame 0). Heads trained with the v4 AuSmoke extension (tmp_v4*, t1_v4*) fire on "
         "a large share of negative frames where v3-only heads fire on ~0% — the shortcut. NOTE: firesense negatives are also training "
         "negatives inside the YOLO corpora B_G/C_EGT/D_adapt (not inside these heads' data).",
         "\n\n".join(parts), ["reports/v4/trackd_v4_eval.json (B_v4_val_replay)"])

    C = rep.get("C_false_alert_episodes_per_hour", {})
    parts = []
    for pool, v in C.get("pools", {}).items():
        flag = " **[CONTAMINATED — see note]**" if v.get("contaminated") else ""
        head = f"**{pool}**{flag} — {v['sequences']} sequences / {v['frames']:,} frames / {v['wall_clock_h']:.3f} h wall clock. {v.get('note', '')}"
        rows = []
        for nme in sysorder(v["systems"]):
            a = v["systems"][nme].get("matched@fpr0.01") or {}
            b = v["systems"][nme].get("matched@fpr0.05") or {}
            rows.append([nme, f3(a.get("threshold"), 4), f3(a.get("episodes_per_hour"), 2), str(a.get("episodes_per_hour_ci95", "")),
                         f3(b.get("episodes_per_hour"), 2), str(b.get("episodes_per_hour_ci95", ""))])
        d = v.get("deployed_confirmed_or_provisional", {})
        rows.append(["DEPLOYED expE (0.30/0.15)", "-", f3(d.get("episodes_per_hour"), 2), str(d.get("episodes_per_hour_ci95", "")), "-", "-"])
        parts.append(head + "\n\n" + md_table(["system", "thr@FPR1%", "false episodes/h @FPR1%", "CI95", "false episodes/h @FPR5%", "CI95"], rows))
    emit("T20", "Early warning: false-alert EPISODES per hour on continuous negative footage",
         f"Episode = maximal run of alert-on frames; runs separated by < {C.get('cooldown_s', 'n/a')} s (deployed cooldown) merge. "
         "CI = exact Poisson 95%. Pools are tiny (minutes of footage) — read as order-of-magnitude.",
         "\n\n".join(parts), ["reports/v4/trackd_v4_eval.json (C_false_alert_episodes_per_hour)"])

    D = rep.get("D_yolo_plus_dino_sweep", {})
    parts = []
    for tag, dd in D.items():
        L = dd["latency_ms_per_frame"]
        sc = dd["sequence_coverage"]
        head = (f"**DINO source {tag}** (`{rep['D_dino_sources'][tag]['path']}`). Batch-1 L40S: YOLO11m expE {L['yolo11m_expE_L40S']} ms, "
                f"DINOv3-L {L['dinov3L_B1_L40S']} ms -> one DINO pass = {L['cost_ratio']}x one YOLO pass; N for <=20% added compute = {dd['n_for_20pct_added_cost']}. "
                f"Val sequences {sc['n_sequences']}, median length {sc['median_len_frames']} frames; fraction of sequences with ZERO DINO pass: "
                + ", ".join(f"{k} {v:.3f}" for k, v in sc["frac_sequences_with_zero_dino_pass"].items()))
        rows = []
        for key in sorted(dd["results"]):
            r = dd["results"][key]
            y = r["yolo_alone"]
            row = [key.replace("matched_fpr_", "FPR "), f"{y['episode_detection_rate']:.4f} ({y['episodes']} ep, {y['false_alert_frames']} FP fr / {y['n_negative_frames']})"]
            for N in ["N=20", "N=70"]:
                v = r["sweep"].get(N)
                if not v:
                    row.append("n/a")
                    continue
                row.append(f"{v['episode_detection_rate']:.4f} ({v['combo_minus_yolo_at_same_fp']:+.4f} vs YOLO@same FP; "
                           f"+{100 * v['added_compute_frac']:.0f}% compute; {'beats' if v['combo_beats_yolo_at_matched_fp'] else 'no'})")
            rows.append(row)
        parts.append(head + "\n\n" + md_table(["frame-FPR target", "YOLO alone: episode detection", "YOLO + DINO every 20th frame", "YOLO + DINO every 70th frame (<=20% compute)"], rows))
    emit("T21", "YOLO every frame + independent DINO pass every Nth frame (sdis VAL sequences)",
         "Episode detection over the 469 sdis_val positive episodes at thresholds matched on frame FPR (1,158 negative frames); "
         "'vs YOLO@same FP' = combo minus YOLO-alone re-thresholded to the combo's realised false-alert frames.",
         "\n\n".join(parts), ["reports/v4/trackd_v4_eval.json (D_yolo_plus_dino_sweep, D_dino_sources)"])


# ----------------------------------------------------------------------------- 6. compute ledger
def compute_tables():
    p = V4 / "COMPUTE_LEDGER.csv"
    if not p.exists():
        MISSING.append(rel(p))
        emit("T22", "Compute ledger", f"ARTIFACT MISSING: `{rel(p)}`", "_no table_", [rel(p)])
        return
    rows_all = list(csv.DictReader(open(p)))
    for r in rows_all:
        r["node_hours"] = float(r["node_hours"]); r["gpu_hours"] = float(r["gpu_hours"])
    external = [r for r in rows_all if r.get("in_scope", "True") == "False"]
    rows = [r for r in rows_all if r.get("in_scope", "True") != "False"]
    budget = load_json(ROOT / "cluster/v4/BUDGET_V4.json") or {}
    cap_nh, cap_gh = budget.get("max_allocation_node_hours"), budget.get("max_allocation_gpu_hours")
    snapshot = max(r["alloc_end_utc"] for r in rows)
    open_rows = [r for r in rows if r["record_type"] == "open-at-dump"]

    by_it = defaultdict(lambda: {"jobs": 0, "nh": 0.0, "gh": 0.0, "fail_nh": 0.0, "fail_jobs": 0, "gpu": defaultdict(float), "exact": 0, "approx": 0})
    for r in rows:
        b = by_it[r["iteration"]]
        b["jobs"] += 1; b["nh"] += r["node_hours"]; b["gh"] += r["gpu_hours"]; b["gpu"][r["gpu_type"]] += r["node_hours"]
        if r["kind"] == "training(failed/retry)" or r["outcome"].startswith("FAILED"):
            b["fail_nh"] += r["node_hours"]; b["fail_jobs"] += 1
        if r["attribution_method"].startswith("time"):
            b["exact"] += 1
        else:
            b["approx"] += 1
    trows = []
    tot = {"jobs": 0, "nh": 0.0, "gh": 0.0, "fail_nh": 0.0, "fail_jobs": 0}
    for it in sorted(by_it, key=int):
        b = by_it[it]
        gpu_txt = ", ".join(f"{k} {v:.1f}" for k, v in sorted(b["gpu"].items(), key=lambda kv: -kv[1]))
        attr = "time-window (exact)" if b["approx"] == 0 else ("name-based (approx)" if b["exact"] == 0 else f"mixed ({b['exact']} exact / {b['approx']} approx)")
        trows.append([it, b["jobs"], f"{b['nh']:.1f}", f"{b['gh']:.0f}", gpu_txt, f"{b['fail_nh']:.1f} ({b['fail_jobs']} jobs)", attr])
        for k in tot:
            tot[k] += b[k]
    trows.append(["ALL in-scope", tot["jobs"], f"{tot['nh']:.1f}", f"{tot['gh']:.0f}", "", f"{tot['fail_nh']:.1f} ({tot['fail_jobs']} jobs)", ""])
    if external:
        trows.append([f"external (other project; excluded)", len(external), f"{sum(r['node_hours'] for r in external):.1f}",
                      f"{sum(r['gpu_hours'] for r in external):.0f}", ", ".join(sorted({r['gpu_type'] for r in external})), "-",
                      "name-prefix: " + ", ".join(sorted({r['job_name'].rsplit('-', 1)[0] for r in external}))])
    emit("T22", "Compute accounting by iteration (scheduler-derived allocated hours; whole-node accounting)",
         f"Allocated node-hours = scheduler completion - start (queue wait excluded); GPU-hours = node-hours x GPUs/node. No pod utilization "
         f"telemetry exists (utilization column = n/a). Iterations 3/4 attributed by exact window start; 1 vs 2 by job-name pattern (approx). "
         f"{len(rows_all)} job records ({len(external)} flagged in_scope=False: same requester, other project, excluded from every iteration total); "
         f"ledger snapshot = {snapshot} ({len(open_rows)} job(s) still open at dump). "
         f"Iteration-4 ceiling: {cap_nh} node-h / {cap_gh} GPU-h (cluster/v4/BUDGET_V4.json). 'failed/retry' = FAILED outcome or training(failed/retry) kind.",
         md_table(["iteration", "jobs", "node-h", "GPU-h", "node-h by GPU type", "failed/retry node-h", "attribution"], trows),
         ["reports/v4/COMPUTE_LEDGER.csv", "cluster/v4/BUDGET_V4.json", "reports/v4/ledger/kraken_*_{raw.json,table.txt} (raw dumps)"])

    v4 = [r for r in rows if r["iteration"] == "4"]
    by_kind = defaultdict(lambda: [0, 0.0, 0.0])
    by_track = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for r in v4:
        k = by_kind[r["kind"]]; k[0] += 1; k[1] += r["node_hours"]; k[2] += r["gpu_hours"]
        t = by_track[r["track"]]; t[0] += 1; t[1] += r["node_hours"]; t[2] += r["gpu_hours"]
        if r["kind"] == "training(failed/retry)" or r["outcome"].startswith("FAILED"):
            t[3] += r["node_hours"]
    v4nh = sum(r["node_hours"] for r in v4); v4gh = sum(r["gpu_hours"] for r in v4)
    krows = [[k, v[0], f"{v[1]:.1f}", f"{v[2]:.0f}", f"{100 * v[1] / v4nh:.0f}%" if v4nh else "n/a"] for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1][1])]
    krows.append(["ALL v4", len(v4), f"{v4nh:.1f}", f"{v4gh:.0f}", f"{100 * v4nh / cap_nh:.0f}% of cap" if cap_nh else ""])
    trk = [[k, v[0], f"{v[1]:.1f}", f"{v[2]:.0f}", f"{v[3]:.1f}"] for k, v in sorted(by_track.items(), key=lambda kv: -kv[1][1])]
    emit("T23", "Iteration-4 compute by kind and by track",
         f"{len(v4)} in-scope iteration-4 jobs in the ledger snapshot ({snapshot}); {v4nh:.1f} node-h = {v4gh:.0f} GPU-h of the {cap_gh} GPU-h ceiling "
         f"({100 * v4gh / cap_gh:.0f}%). External same-requester jobs in the window are excluded (see T22). "
         + (f"{len(open_rows)} job(s) still open at dump are truncated. " if open_rows else "No job was open at the dump. "),
         "**by kind**\n\n" + md_table(["kind", "jobs", "node-h", "GPU-h", "share of v4 node-h"], krows)
         + "\n\n**by track**\n\n" + md_table(["track", "jobs", "node-h", "GPU-h", "failed/retry node-h"], trk),
         ["reports/v4/COMPUTE_LEDGER.csv", "cluster/v4/BUDGET_V4.json"])

    inc = [r for r in rows_all if not r["outcome"].startswith("SUCCEEDED") and not r["outcome"].startswith("FAILED")]
    irows = [[r["job_name"], (r["iteration"] if r.get("in_scope", "True") != "False" else "external"), r["track"], r["kind"], r["outcome"],
              r["created_utc"], r["alloc_start_utc"], r["alloc_end_utc"], f"{r['node_hours']:.2f}"] for r in inc]
    fail4 = [r for r in v4 if r["outcome"].startswith("FAILED")]
    ftrack = defaultdict(lambda: [0, 0.0])
    for r in fail4:
        ftrack[r["track"]][0] += 1; ftrack[r["track"]][1] += r["node_hours"]
    frows = [[k, v[0], f"{v[1]:.2f}"] for k, v in sorted(ftrack.items(), key=lambda kv: -kv[1][1])]
    emit("T24", "Compute incidents: non-terminal / stopped / never-allocated jobs, and iteration-4 FAILED jobs by track",
         f"Top: every ledger row whose outcome is neither SUCCEEDED nor FAILED ({len(inc)} rows; QUEUED(never allocated) = idle allocation request with 0 hours; "
         f"STOPPED = manually killed; RUNNING(open at dump) = truncated at the snapshot). Bottom: iteration-4 FAILED jobs ({len(fail4)} jobs, "
         f"{sum(r['node_hours'] for r in fail4):.2f} node-h) — the arm-D DDP crash loop (14 segments, exit 101, ~1.2 node-h per STATUS.md) sits in yolo-matrix.",
         md_table(["job", "iter", "track", "kind", "outcome", "created", "alloc start", "alloc end", "node-h"], irows)
         + "\n\n**iteration-4 FAILED jobs by track**\n\n" + md_table(["track", "failed jobs", "node-h"], frows),
         ["reports/v4/COMPUTE_LEDGER.csv"])


# ----------------------------------------------------------------------------- 7. FINAL TEST (one-shot)
TEST = V4 / "test"
# canonical roster: (id, label, key in matched_fp_sdistest.json, key in yolo_test json, key in dino_test json, role)
ROSTER = [
    ("S0",  "expE (iteration-2 YOLO11m candidate)",        "S0_expE",        "S0_expE",           None,              "deployment baseline (pre-registered)"),
    ("S0b", "expB (deployed default, YOLO11s@1280)",       "S0b_expB",       "S0b_expB_deployed", None,              "deployment baseline (pre-registered)"),
    ("S1",  "A_ctrl_E, median seed",                       "S1_A",           "S1_A_ctrl",         None,              "existing-data control (pre-registered)"),
    ("S2",  "E_ET, median seed",                           "S2_E_ET",        "S2_E_ET",           None,              "best v4 data arm by S2 rule (pre-registered)"),
    ("S2c", "C_EGT, median seed",                          "S2c_C_EGT",      "S2c_C_EGT",         None,              "matched control = the G-value comparison (pre-registered)"),
    ("S3",  "B_Gonly, seed 0",                             "S3_B_G",         "S3_B_Gonly",        None,              "completeness (pre-registered)"),
    ("S3",  "D_curr, median seed",                         "S3_D_curr",      "S3_D_curr_posthoc", None,              "POST-HOC addition (not a pre-registered candidate)"),
    ("S4",  "v4B6_lora_Hplus (DINOv3 LoRA-H+)",            "S4_B6_loraH",    None,                "v4B6_lora_Hplus",  "DINO rule pick: best sdis-val R@5% (pre-registered)"),
    ("S4c", "v4B1_frozen_L (DINOv3 frozen-L)",             "S4c_B1_frozenL", None,                "v4B1_frozen_L",    "DINO control (pre-registered)"),
    ("S4p", "v4B4_lora_L (DINOv3 LoRA-L)",                 "S4p_B4_loraL",   None,                "v4B4_lora_L",      "POST-HOC efficiency candidate (labelled)"),
]
REGIMES = [("source_only_sdis_val", "negfp_5pct", "SOURCE-ONLY sdis-val conf@5%"),
           ("source_only_sdis_val", "negfp_10pct", "SOURCE-ONLY sdis-val conf@10%"),
           ("source_only_sdis_val", "negfp_20pct", "SOURCE-ONLY sdis-val conf@20%"),
           ("target_adapted_boreal_val", "negfp_1pct", "TARGET-ADAPTED boreal-val conf@1%"),
           ("target_adapted_boreal_val", "negfp_5pct", "TARGET-ADAPTED boreal-val conf@5%")]
SPLIT_KEYS = {"sdistest": ("sdistest", "sdis_test"), "borealtest": ("borealtest", "boreal_test"), "foresttest": ("foresttest", "forest_test")}


def _test_cell(entry, with_fp=False, n_neg=None, with_tiny=True):
    if not isinstance(entry, dict) or entry.get("recall") is None:
        return str(entry) if entry else "n/a"
    txt = f"{entry['recall']:.3f}"
    if with_tiny and entry.get("tiny_recall") is not None:
        txt += f" / {entry['tiny_recall']:.3f}"
    txt += f" (c={entry['conf']:.4f}"
    if with_fp and entry.get("neg_image_fp_rate") is not None and n_neg:
        txt += f"; neg flagged {round(entry['neg_image_fp_rate'] * n_neg)}/{n_neg}"
    return txt + ")"


def _regime_entry(sysrow, yolo, dino, split, regime, point):
    _, _, _, ykey, dkey, _ = sysrow
    ysplit, dsplit = SPLIT_KEYS[split]
    if ykey and yolo:
        return ((yolo.get("results", {}).get(ykey) or {}).get(ysplit) or {}).get(regime, {}).get(point)
    if dkey and dino:
        return ((dino.get("results", {}).get(dkey) or {}).get(dsplit) or {}).get(regime, {}).get(point)
    return None


def _regime_table(yolo, dino, split, n_neg, with_tiny):
    hdr = ["system"] + [lab for _, _, lab in REGIMES]
    rows = []
    for sysrow in ROSTER:
        if (sysrow[3] and yolo and sysrow[3] in yolo.get("results", {})) or (sysrow[4] and dino and sysrow[4] in dino.get("results", {})):
            rows.append([f"{sysrow[0]} {sysrow[1]}"] + [_test_cell(_regime_entry(sysrow, yolo, dino, split, rg, pt), with_fp=bool(n_neg), n_neg=n_neg, with_tiny=with_tiny)
                                                        for rg, pt, _ in REGIMES])
    return md_table(hdr, rows)


def roster_table():
    """T28: exactly which systems/checkpoints went to TEST and how they were selected (from artifacts)."""
    ss = load_json(V4 / "seed_summary.json") or {}
    sel = ss.get("selection", {})
    yolo = load_json(TEST / "yolo_test_at_preregistered_thresholds.json")
    dthr = load_json(TEST / "dino_test_thresholds.json")
    rep_w = {}
    for ck in ["v4a_s2", "v4e_s1", "v4c_s1", "v4dada_s1"]:
        d = load_json(V4 / f"replay/replay_{ck}_figlib39.json")
        if d:
            rep_w[ck] = d["summary"].get("config", {}).get("weights")
    med = {arm: (sel.get(f"{arm}.median_seed(sdis R@10%)") or {}) for arm in ["A_ctrl_E", "E_ET", "C_EGT", "D_curr"]}
    ck_of = {"S1_A_ctrl": ("A_ctrl_E", "v4a_s2"), "S2_E_ET": ("E_ET", "v4e_s1"), "S2c_C_EGT": ("C_EGT", "v4c_s1"), "S3_D_curr_posthoc": ("D_curr", "v4dada_s1")}
    rows = []
    for sid, label, mkey, ykey, dkey, role in ROSTER:
        ckpt, basis = "", ""
        if ykey in ck_of:
            arm, ck = ck_of[ykey]
            m = med.get(arm, {})
            ckpt = f"`{ck}` -> {rep_w.get(ck, 'weights path in replay json')}"
            basis = f"median-seed rule: sdis-val R@10% {m.get('recall', 'n/a')} (n_seeds={m.get('n_seeds', 'n/a')})"
            if arm == "E_ET":
                basis += f"; S2 rule -> {sel.get('S2_pick')} ({sel.get('S2_rule', '')})"
        elif ykey == "S3_B_Gonly":
            ckpt, basis = "`v4b` (seed 0)", "only seed trained"
        elif ykey == "S0_expE":
            w = ((load_json(R / "replay_expE_figlib.json") or {}).get("summary", {}).get("config", {}) or {}).get("weights", "n/a")
            ckpt, basis = f"`{w}` (iteration-2 expE YOLO11m)", "deployment baseline; boreal/forest-val thresholds from reports/v4/eval_expE_{borealval,forestval}.json (v4evalr7)"
        elif ykey == "S0b_expB_deployed":
            ckpt, basis = "`models/expB_best.pt` (byte-identical to models/trained/expB_mix_y11s_1280/seg03/best.pt; RECONCILIATION_V4.md)", "deployed default; boreal/forest-val thresholds from reports/v4/eval_expB_{borealval,forestval}.json (v4evalr7)"
        elif dkey and dthr:
            r = dthr.get("runs", {}).get(dkey, {})
            ckpt = f"`{r.get('checkpoint', '')}` ({r.get('infer_extra') or 'plain'})"
            basis = dthr.get("roles", {}).get(dkey, "")
        thr = ""
        if ykey and yolo:
            t = yolo.get("thresholds", {}).get(ykey, {})
            so, ta = t.get("source_only_sdis_val", {}), t.get("target_adapted_boreal_val", {})
            fmt = lambda x: "n/a (val FP floor)" if x is None else str(x)
            thr = f"sdis@5/10/20%: {fmt(so.get('negfp_5pct'))} / {fmt(so.get('negfp_10pct'))} / {fmt(so.get('negfp_20pct'))}; boreal@1/5%: {fmt(ta.get('negfp_1pct'))} / {fmt(ta.get('negfp_5pct'))}"
        elif dkey and dthr:
            r = dthr.get("runs", {}).get(dkey, {})
            so, ta = r.get("sdis_val_source_only", {}), r.get("boreal_val_target_adapted", {})
            g = lambda d, k: (d.get(k) or {}).get("conf")
            thr = f"sdis@5/10/20%: {g(so, 'negfp_5pct')} / {g(so, 'negfp_10pct')} / {g(so, 'negfp_20pct')}; boreal@1/5%: {g(ta, 'negfp_1pct')} / {g(ta, 'negfp_5pct')}"
        rows.append([sid, label, ckpt, basis, thr, role])
    marker_y = (TEST / ".EVALUATED_ONCE").read_text().strip() if (TEST / ".EVALUATED_ONCE").exists() else "n/a"
    marker_d = (TEST / ".DINO_EVALUATED_ONCE").read_text().strip().replace("\n", " | ") if (TEST / ".DINO_EVALUATED_ONCE").exists() else "n/a"
    emit("T28", "FINAL TEST roster: systems, checkpoints, selection basis, fixed thresholds, provenance",
         f"Once-guards: YOLO `.EVALUATED_ONCE` = {marker_y}; DINO `.DINO_EVALUATED_ONCE` = {marker_d}. The DINO half was launched at 02:31Z by the "
         "pre-restart coordinator session with exactly the pre-registered systems/thresholds and was NOT re-run (STATUS.md 02:40Z, 03:00Z). "
         "D_curr is a POST-HOC addition: it was not a pre-registered candidate and its 3-seed median became available only after the test job launched; "
         "the shipped checkpoint coincidentally equals the median-seed pick (STATUS.md 03:00Z). Thresholds were fixed on VAL before any test prediction "
         "(reports/v4/test/dino_test_thresholds.json; yolo_test_at_preregistered_thresholds.json `thresholds`).",
         md_table(["id", "system", "checkpoint", "selection basis (val)", "fixed thresholds (conf)", "role"], rows),
         ["reports/v4/test/yolo_test_at_preregistered_thresholds.json", "reports/v4/test/dino_test_thresholds.json", "reports/v4/seed_summary.json (selection)",
          "reports/v4/replay/replay_{v4a_s2,v4e_s1,v4c_s1,v4dada_s1}_figlib39.json (weights)", "reports/v4/test/.EVALUATED_ONCE, .DINO_EVALUATED_ONCE", "STATUS.md 02:40Z/03:00Z/03:55Z"])


def test_tables():
    mfp = load_json(TEST / "matched_fp_sdistest.json")
    yolo = load_json(TEST / "yolo_test_at_preregistered_thresholds.json")
    dino = load_json(TEST / "dino_test_at_preregistered_thresholds.json")
    per_unit = load_json(TEST / "forest_test_per_unit.json")
    if not (mfp or yolo or dino):
        for tid, split, desc in [("T25", "sdistest", "pyro-sdis TEST"), ("T26", "borealtest", "boreal AERIAL TEST"), ("T27", "foresttest", "AuSmoke FOREST TEST")]:
            emit(tid, f"FINAL TEST — {desc}", f"**PENDING** — no test artifact under `reports/v4/test/` yet.", "_{{TEST:" + split + "}} — numbers pending_",
                 ["reports/v4/test/matched_fp_sdistest.json (absent)", "reports/v4/test/{yolo,dino}_test_at_preregistered_thresholds.json (absent)"])
        roster_table()
        return
    # ---- T25 sdis TEST
    body = ""
    if mfp:
        models = [m for _, _, m, _, _, _ in ROSTER if m in mfp["models"]] + [m for m in mfp["models"] if m not in {r[2] for r in ROSTER}]
        m0 = mfp["models"][models[0]]
        rows = [[FP_LABEL[k]] + [cell_rtc(mfp["matched_fp"][m].get(k), mfp["models"][m]["curve"], float(k[2:])) for m in models] for k in FP_KEYS]
        n_img, n_pos, n_neg = m0["n_images"], m0["n_positive_images"], m0["n_negative_images"]
        body += "**(a) matched neg-image FP — recall / tiny-recall (conf)**\n\n" + md_table(["neg-image FP target"] + models, rows)
        # deltas vs expE and vs A at 10%
        drow = []
        for m in models:
            e = mfp["matched_fp"][m].get("fp0.10", {}) or {}
            ref_e = (mfp["matched_fp"].get("S0_expE", {}).get("fp0.10") or {}).get("recall")
            ref_a = (mfp["matched_fp"].get("S1_A", {}).get("fp0.10") or {}).get("recall")
            if e.get("recall") is not None:
                drow.append([m, f"{e['recall']:.3f}", (f"{100 * (e['recall'] - ref_e):+.1f} pt" if ref_e is not None else "n/a"),
                             (f"{100 * (e['recall'] - ref_a):+.1f} pt" if ref_a is not None else "n/a")])
        body += "\n\n**(b) R@10% neg-FP deltas**\n\n" + md_table(["system", "R@10%", "vs S0_expE", "vs S1_A"], drow)
    else:
        n_img = n_pos = n_neg = None
    if yolo or dino:
        body += "\n\n**(c) recall / tiny at the pre-registered VAL-selected thresholds, with the FP realised on test negatives**\n\n" + _regime_table(yolo, dino, "sdistest", n_neg or 917, True)
    emit("T25", "FINAL TEST — pyro-sdis TEST (fixed-camera source domain; sites ferion + croix-augas, frozen since iteration 1)",
         f"One-shot evaluation. Denominators: n_images={n_img:,} (positives {n_pos:,}, negatives {n_neg:,}); 1% neg-FP = ~{round(0.01 * n_neg)} negative images. "
         "Matched-FP interpolated on each model's own curve (IoU>=0.4, --top-fine grid for saturated DINO scores). Compare across architectures only at matched FP." if n_img else "One-shot evaluation.",
         body, ["reports/v4/test/matched_fp_sdistest.json", "reports/v4/test/yolo_test_at_preregistered_thresholds.json", "reports/v4/test/dino_test_at_preregistered_thresholds.json"])
    # ---- T26 boreal TEST
    nb = None
    if yolo:
        e = ((yolo["results"].get("S1_A_ctrl") or {}).get("borealtest") or {}).get("source_only_sdis_val", {}).get("negfp_10pct", {})
        nb = (e.get("n_images"), e.get("n_pos"), e.get("n_neg"))
    elif dino:
        e = (dino["results"].get("v4B6_lora_Hplus") or {}).get("boreal_test", {})
        nb = (e.get("n_images"), e.get("n_positive_images"), e.get("n_negative_images"))
    emit("T26", "FINAL TEST — boreal AERIAL TEST (burn_ruokolahti, ONE burn, 9 DJI flights): recall at the pre-registered thresholds",
         f"Denominators: n_images={nb[0]:,} (positives {nb[1]:,}, negatives {nb[2]}) — ONE independent burn; with {nb[2]} negatives the false-alarm side is UNMEASURABLE "
         "('neg flagged x/2' is anecdotal). Tiny recall is undefined (no tiny GT boxes) and omitted. SOURCE-ONLY = sdis-val conf applied unchanged; "
         "TARGET-ADAPTED = boreal-val (heinola) conf. 'n/a(val FP floor)' = the model never reached that FP on boreal val.",
         _regime_table(yolo, dino, "borealtest", nb[2], False),
         ["reports/v4/test/yolo_test_at_preregistered_thresholds.json", "reports/v4/test/dino_test_at_preregistered_thresholds.json", "reports/v4/LOCKBOX_V4_FROZEN.md (composition)"])
    # ---- T27 forest TEST
    nf = None
    if yolo:
        e = ((yolo["results"].get("S1_A_ctrl") or {}).get("foresttest") or {}).get("source_only_sdis_val", {}).get("negfp_10pct", {})
        nf = (e.get("n_images"), e.get("n_pos"), e.get("n_neg"))
    elif dino:
        e = (dino["results"].get("v4B6_lora_Hplus") or {}).get("forest_test", {})
        nf = (e.get("n_images"), e.get("n_positive_images"), e.get("n_negative_images"))
    body = "**(a) whole set — recall / tiny-recall at the pre-registered thresholds**\n\n" + _regime_table(yolo, dino, "foresttest", None, True)
    if per_unit:
        hold = load_json(ROOT / "data/v4/manifests/forest_holdout_v4.json") or {}
        man = load_json(ROOT / "data/v4/eval/EVALTEST_MANIFEST.json") or {}
        units_meta = (man.get("forest_test") or {}).get("units", {})
        grp_of = {u: g for g, us in (hold.get("groups") or {}).items() for u in us}
        systems = list(per_unit["per_unit"].keys())
        units = sorted(next(iter(per_unit["per_unit"].values())).keys(), key=lambda u: (grp_of.get(u, ""), u))
        rows = []
        cur = None
        groups_order = []
        for u in units:
            g = grp_of.get(u, "?")
            if g not in groups_order:
                groups_order.append(g)
        for g in groups_order:
            us = [u for u in units if grp_of.get(u, "?") == g]
            is_aerial = any((units_meta.get(u) or {}).get("aerial_event") for u in us)
            for u in us:
                meta = units_meta.get(u) or {}
                flag = " **[staged TEST_RUN]**" if meta.get("staged_test_run") else ""
                n = next(iter(per_unit["per_unit"].values()))[u]["n"]
                rows.append([g + (" (aerial flight)" if is_aerial else ""), u + flag, n] + [f"{per_unit['per_unit'][s_][u]['recall']:.3f}" for s_ in systems])
            # event-level frame-weighted recall; for the aerial flight also excluding the staged unit
            def agg(us_):
                n_tot = sum(next(iter(per_unit["per_unit"].values()))[u]["n"] for u in us_)
                return [f"{sum(per_unit['per_unit'][s_][u]['recall'] * per_unit['per_unit'][s_][u]['n'] for u in us_) / n_tot:.3f}" for s_ in systems], n_tot
            vals, n_tot = agg(us)
            rows.append([f"**{g} — EVENT (frame-weighted, {len(us)} unit{'s' if len(us) > 1 else ''})**", "", n_tot] + [f"**{v}**" for v in vals])
            real = [u for u in us if not (units_meta.get(u) or {}).get("staged_test_run")]
            if len(real) != len(us):
                vals, n_tot = agg(real)
                rows.append([f"{g} — EVENT excl. staged TEST_RUN", "", n_tot] + vals)
        confs = {s_: next(iter(per_unit["per_unit"][s_].values()))["conf"] for s_ in systems}
        body += ("\n\n**(b) per unit and per independent EVENT at the TARGET-ADAPTED boreal-val-1% confidence** (conf per system: "
                 + ", ".join(f"{s_} {c:.4f}" for s_, c in confs.items()) + ")\n\n"
                 + md_table(["event (group)", "unit", "frames"] + systems, rows))
    emit("T27", "FINAL TEST — AuSmoke FOREST TEST (new region, ACT Australia; 9 units / 4 independent events incl. ONE DJI aerial flight)",
         f"Denominators: n_images={nf[0]:,} (positives {nf[1]:,}, negatives {nf[2]}) — positives-only, so false alarms are UNMEASURABLE on this domain and matched-FP is undefined; "
         "recall-at-threshold only. Independent events: g_20241021, g_20240326_fb769, g_20240517_fb884 (ground burns) and g_dji_20230922 (one aerial flight, 3 videos, "
         "one of them a staged test-run reported separately). Never pool the 9 units as 9 events; the EVENT rows are frame-weighted over a group's units.",
         body, ["reports/v4/test/yolo_test_at_preregistered_thresholds.json", "reports/v4/test/dino_test_at_preregistered_thresholds.json",
                "reports/v4/test/forest_test_per_unit.json", "data/v4/manifests/forest_holdout_v4.json (groups)", "data/v4/eval/EVALTEST_MANIFEST.json (aerial_event, staged_test_run)"])
    roster_table()


def aerial_val_test_table():
    """T29: aerial VAL (heinola) -> TEST (ruokolahti) for the exact TEST systems."""
    ss = load_json(V4 / "seed_summary.json") or {}
    yolo = load_json(TEST / "yolo_test_at_preregistered_thresholds.json")
    dino = load_json(TEST / "dino_test_at_preregistered_thresholds.json")
    if not (yolo or dino):
        return
    dbv = load_json(V4 / "matched_fp_dino_boreal_val.json") or {}
    per_seed = {}
    for arm, v in (ss.get("arms") or {}).items():
        for ck, e in (v.get("borealval") or {}).get("per_seed", {}).items():
            per_seed[ck] = e

    def val_from_parquet(pq):
        grid = [round(x / 100, 4) for x in range(1, 100)]
        ex = recall_at_confs_exact(pq, grid)
        if not ex:
            return None, None
        sw = [ex[c] for c in sorted(k for k in ex if k != "_n")]
        return interp_at_fp(sw, 0.01, lambda s_: s_["recall"]), interp_at_fp(sw, 0.05, lambda s_: s_["recall"])

    spec = [("S0 expE", None, V4 / "eval_expE_borealval.raw.parquet", "S0_expE", None),
            ("S0b expB (deployed)", None, V4 / "eval_expB_borealval.raw.parquet", "S0b_expB_deployed", None),
            ("S1 A_ctrl (v4a_s2)", "v4a_s2", None, "S1_A_ctrl", None),
            ("S2 E_ET (v4e_s1)", "v4e_s1", None, "S2_E_ET", None),
            ("S2c C_EGT (v4c_s1)", "v4c_s1", None, "S2c_C_EGT", None),
            ("S3 B_Gonly (v4b)", "v4b", None, "S3_B_Gonly", None),
            ("S3 D_curr (v4dada_s1, post-hoc)", "v4dada_s1", None, "S3_D_curr_posthoc", None),
            ("S4 B6 LoRA-H+", None, None, None, "v4B6_lora_Hplus"),
            ("S4c B1 frozen-L", None, None, None, "v4B1_frozen_L"),
            ("S4p B4 LoRA-L", None, None, None, "v4B4_lora_L")]
    rows = []
    for label, ck, pq, ykey, dkey in spec:
        if ck:
            e = per_seed.get(ck, {})
            v1, v5 = (e.get("fp0.01") or {}).get("recall"), (e.get("fp0.05") or {}).get("recall")
            vsrc = "seed_summary.json"
        elif pq:
            v1, v5 = val_from_parquet(pq)
            vsrc = rel(pq)
        else:
            m = (dbv.get("matched_fp") or {}).get(dkey, {})
            v1, v5 = (m.get("fp0.01") or {}).get("recall"), (m.get("fp0.05") or {}).get("recall")
            vsrc = "matched_fp_dino_boreal_val.json"
        sysrow = (None, None, None, ykey, dkey, None)
        so10 = _regime_entry(sysrow, yolo, dino, "borealtest", "source_only_sdis_val", "negfp_10pct")
        ta1 = _regime_entry(sysrow, yolo, dino, "borealtest", "target_adapted_boreal_val", "negfp_1pct")
        ta1_r = ta1.get("recall") if isinstance(ta1, dict) else None
        rows.append([label, f3(v1), f3(v5) if v5 is not None else "n/a (FP ceiling)", _test_cell(so10, with_fp=True, n_neg=2, with_tiny=False),
                     _test_cell(ta1, with_fp=True, n_neg=2, with_tiny=False),
                     (f"{100 * (ta1_r - v1):+.1f} pt" if (ta1_r is not None and v1 is not None) else "n/a"), vsrc])
    gv = load_json(V4 / "eval_v4a_s2_borealval.json") or {}
    gt = load_json(TEST / "eval_S1_A_ctrl_borealtest.json") or {}
    gsv = (gv.get("conf_sweep") or [{}])[0].get("gt_by_size", {})
    gst = (gt.get("conf_sweep") or [{}])[0].get("gt_by_size", {})
    emit("T29", "AERIAL: boreal VAL (heinola) -> boreal TEST (ruokolahti), exact TEST checkpoints",
         "VAL columns = recall at matched neg-image FP on heinola (n_images=1,123: 906 pos / 217 neg; 1% = ~2 negatives); TEST columns = recall at the "
         "pre-registered thresholds on ruokolahti (n_images=1,764: 1,762 pos / 2 neg; FP unmeasurable). Both are ONE burn each. "
         f"GT box sizes — val: {gsv}; test: {gst} (eval JSON gt_by_size; 'large' = >5% of image area): both are large-plume sets. "
         "The delta column compares test recall at the boreal-val-1% conf with val recall at 1% FP (same threshold, different burn). "
         "expE/expB val recall recomputed from their boreal-val parquet with the frozen matching code (v4evalr7).",
         md_table(["system", "VAL R@1% FP", "VAL R@5% FP", "TEST R @ sdis-val-10% conf", "TEST R @ boreal-val-1% conf", "TEST(ta1) - VAL(1%)", "val source"], rows),
         ["reports/v4/seed_summary.json", "reports/v4/matched_fp_dino_boreal_val.json", "reports/v4/eval_exp{E,B}_borealval.raw.parquet",
          "reports/v4/test/yolo_test_at_preregistered_thresholds.json", "reports/v4/test/dino_test_at_preregistered_thresholds.json",
          "reports/v4/eval_v4a_s2_borealval.json, reports/v4/test/eval_S1_A_ctrl_borealtest.json (gt_by_size)"])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(V4 / "tables"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    seed_tables()
    seed0_tables()
    dino_tables()
    figlib_tables()
    temporal_tables()
    compute_tables()
    test_tables()
    aerial_val_test_table()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    written = []
    for tid, text in TABLES.items():
        title = text.splitlines()[0][len(f"### {tid}. "):]
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower())[:60].strip("_")
        p = out / f"{tid}_{slug}.md"
        p.write_text(f"_generated {stamp} by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_\n\n" + text)
        written.append(p)
    # sweep stale table files from earlier runs (titles/slugs may change)
    keep = {p.name for p in written}
    for stale in out.glob("T[0-9][0-9]_*.md"):
        if stale.name not in keep:
            stale.unlink()
            print("removed stale", stale.name)
    idx = "\n".join(f"- [{tid}](#{tid.lower()}) {TABLES[tid].splitlines()[0][len(tid) + 6:]}" for tid in TABLES)
    miss = "\n".join(f"- `{m}`" for m in sorted(set(MISSING))) or "- none"
    notes = "\n".join(f"- {n}" for n in NOTES) or "- none"
    allmd = (f"# Iteration 4 — all report tables\n\n_generated {stamp} by `scripts/v4/make_report_tables.py`; every number is read from the "
             f"artifact(s) listed under each table. Re-run the script after new eval JSONs land (e.g. reports/v4/test/)._\n\n"
             f"## Index\n\n{idx}\n\n## Artifacts not found at generation time\n\n{miss}\n\n## Notes\n\n{notes}\n\n---\n\n"
             + "\n\n---\n\n".join(TABLES.values()))
    (out / "ALL_TABLES.md").write_text(allmd)
    print(f"{len(written)} tables -> {rel(out)}/ ; combined -> {rel(out / 'ALL_TABLES.md')}")
    for p in written:
        print("  ", p.name)
    if MISSING:
        print("missing artifacts:", *sorted(set(MISSING)), sep="\n  ")
    if NOTES:
        print("notes:", *NOTES, sep="\n  ")


if __name__ == "__main__":
    main()
