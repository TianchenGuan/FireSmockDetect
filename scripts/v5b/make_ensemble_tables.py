"""Render markdown tables for reports/v5b/ENSEMBLE_DEV_V5B.md from reports/v5b/ensemble_dev_v5b.json."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = json.load(open(ROOT / (sys.argv[1] if len(sys.argv) > 1 else "reports/v5b/ensemble_dev_v5b.json")))
FPS = ["fp0.01", "fp0.02", "fp0.05", "fp0.10", "fp0.20"]
HDR = "| system | 1% | 2% | 5% | 10% | 20% | mean |\n|---|---|---|---|---|---|---|"
FOLDS = ["cabanelle", "marguerite"]
SHORT = {"yolo_A": "A", "dino_B4": "B4", "dino_B6": "B6", "rfdetr_L": "RF", "yolo_expE": "expE", "dino_B1": "B1"}


def f(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def cell(d, k):
    return f"{f(d[k]['recall'])} / {f(d[k]['tiny'])}"


def row(name, d, params=None):
    s = f"| {name} | " + " | ".join(cell(d, k) for k in FPS) + f" | **{d['mean_recall']:.3f}**"
    return s + (f" <br>`{params}`" if params else "") + " |"


def pname(p):
    if "conf_type" in p:
        return f"{p['conf_type']},iou{p['iou_thr']},skip{p['skip_box_thr']}"
    return f"iou{p['iou_thr']}"


def setname(s):
    return "+".join(SHORT[m] for m in s.split("+"))


out = []
P = out.append
fo = R["folds"]
P("### T1. pyro-sdis VAL split by site (grouped folds)\n")
P("| fold | site | images | negative images | GT boxes | tiny GT (<0.1% area) | 1% FP = n neg images |\n|---|---|---|---|---|---|---|")
for c, s in enumerate(FOLDS):
    P(f"| {c} | {s} | {fo[s]['n_images']:,} | {fo[s]['n_neg']:,} | {fo[s]['n_gt']:,} | {fo[s]['n_gt_tiny']:,} | {fo[s]['n_neg']/100:.1f} |")
P(f"| all | both | 7,335 | 1,158 | {fo['cabanelle']['n_gt']+fo['marguerite']['n_gt']:,} | {fo['cabanelle']['n_gt_tiny']+fo['marguerite']['n_gt_tiny']:,} | 11.6 |\n")

P("### T2. Single models, own raw confidence, whole VAL (recall / tiny-recall at matched neg-image FP; IoU>=0.4)\n")
P(HDR)
for m, d in R["single_raw"].items():
    P(row(f"{SHORT[m]} ({m})", d))
P("")
for s in FOLDS:
    P(f"### T3-{s[:3]}. Held-out site = **{s}** — single models (raw conf) vs fusions fit on the other site (OOF)\n")
    P(HDR)
    for m, d in R["single_raw"].items():
        P(row(f"single {SHORT[m]}", d["by_fold"][s]))
    for st, d in R["sets"].items():
        for meth in ("union", "wbf"):
            v = d[meth]["oof_by_fold"][s]
            P(row(f"{'OR/union+NMS' if meth=='union' else 'WBF'} {setname(st)}", v, pname(d[meth]["params_by_eval_fold"][s])))
    for k, v in R["learned"]["chosen"].items():
        P(row(f"learned {k} {setname('+'.join(R['chosen_models']))}", v["oof_by_fold"][s]))
    P("")

P("### T4. Fold-averaged (macro) OOF mean recall — the selection statistic — and pooled OOF\n")
P("| model set | union macro | WBF macro | union pooled | WBF pooled | WBF params chosen on all VAL |\n|---|---|---|---|---|---|")
for st, d in R["sets"].items():
    P(f"| {setname(st)} | {d['union']['oof_macro_mean']:.4f} | {d['wbf']['oof_macro_mean']:.4f} | "
      f"{d['union']['oof_pooled']['mean_recall']:.4f} | {d['wbf']['oof_pooled']['mean_recall']:.4f} | `{pname(d['wbf']['params_all_val'])}` |")
best = {m: round(sum(R["single_raw"][m]["by_fold"][s]["mean_recall"] for s in FOLDS) / 2, 4) for m in R["single_raw"]}
P("| *single models (macro of per-site raw)* | " + ", ".join(f"{SHORT[m]} {v:.4f}" for m, v in best.items()) + " | | | | |")
for k, v in R["learned"]["chosen"].items():
    P(f"| learned {k} ({setname('+'.join(R['chosen_models']))}) | | {v['oof_macro_mean']:.4f} | | {v['oof_pooled']['mean_recall']:.4f} | |")
for k, v in R["learned"]["all6"].items():
    P(f"| learned {k} (all 6) | | {v['oof_macro_mean']:.4f} | | {v['oof_pooled']['mean_recall']:.4f} | |")
P("")

P("### T5. Model-addition protocol (keep if >= +1.0 pt macro-OOF mean recall over current best)\n")
P("| candidate added to | union macro | WBF macro | gain (pt) | kept |\n|---|---|---|---|---|")
for a in R["addition_protocol"]:
    P(f"| {setname('+'.join(a['set_tested'][:-1]))} + **{SHORT[a['candidate']]}** | {a['union_mean_recall']:.4f} | {a['wbf_mean_recall']:.4f} | {a['gain_pts_vs_current_best']:+.2f} | {'yes' if a['kept'] else 'no'} |")
P("")

P("### T6. Bootstrap (500 image resamples within the held-out site): difference in OOF mean recall, pts\n")
P("| held-out site | comparison | point | 95% CI | P(gain <= 0) |\n|---|---|---|---|---|")
for s in FOLDS:
    for k, v in R["bootstrap"][s].items():
        if k == "best_single":
            continue
        P(f"| {s} | {k.replace('fusion', 'ACCURATE fusion ('+setname('+'.join(R['chosen_models']))+' WBF)', 1) if k.startswith('fusion') else k} | {v['point_pts']:+.2f} | [{v['ci95_pts'][0]:+.2f}, {v['ci95_pts'][1]:+.2f}] | {v['p_gain_le_0']:.3f} |")
P("")

P("### T7. Image-level alert score (max over boxes) — recall of positive images at matched neg-image FP, WITHIN each site (thresholds per site)\n")
P("| method | site | 1% | 2% | 5% | 10% | 20% | mean |\n|---|---|---|---|---|---|---|---|")
il = R["image_level"]
for k, v in il.items():
    if k.startswith("single:") and SHORT[k.split(":")[1]] not in ("A", "B4", "B6", "RF"):
        continue
    for s in FOLDS:
        vals = [v[fp][s] for fp in FPS]
        P(f"| {k} | {s} | " + " | ".join(f"{x:.3f}" for x in vals) + f" | **{sum(vals)/5:.3f}** |")
P("")
P("### T7b. Image-level, POOLED over sites with one threshold (single models raw; fusions OOF-calibrated)\n")
P("| method | 1% | 2% | 5% | 10% | 20% | mean |\n|---|---|---|---|---|---|---|")
for k, v in il.items():
    P(f"| {k} | " + " | ".join(f"{v[fp]['recall']:.3f}" for fp in FPS) + f" | **{v['mean_recall']:.3f}** |")
P("")

P("### T8. Threshold transfer between sites: threshold set for target FP on one site -> realized neg-image FP on the other\n")
P("| system | direction | @1% | @2% | @5% | @10% | @20% |\n|---|---|---|---|---|---|---|")
for m, d in R["threshold_transfer"].items():
    for dirn, vals in d.items():
        P(f"| {SHORT.get(m, m)} | {dirn.replace('fit_', 'set on ').replace('->eval_', ' -> realized on ')} | " +
          " | ".join(f"{vals[fp]['realized_fp']:.3f}" if vals[fp] else "n/a" for fp in FPS) + " |")
P("")

lt = R["latency_table"]; fl = R["fusion_latency_ms_cpu"]
P("### T9. ACCURATE-mode latency, batch-1 1x L40S fp32 @1280 (component numbers from reports; fusion measured here)\n")
P("| component | ms | source |\n|---|---|---|")
rec = json.load(open(ROOT / "models/v5b/fusion/accurate_recipe.json"))
for m, ms in lt["components_ms"].items():
    P(f"| {m} | {ms} | {rec['models'][m]['latency_source']} |")
P(f"| fusion (calibration + WBF, numpy, CPU) | {fl['median']} median / {fl['p90']} p90 / {fl['max']} max | measured over {fl['n_images']:,} VAL images, devbox CPU |")
P(f"| **FULL-FRAME PARALLEL = max(components) + fusion** | **{lt['parallel_ms']}** | one GPU per component, all run on every frame |")
P(f"| serial (one GPU, components back to back) + fusion | {lt['serial_ms']} | |")
P("| not-yet-tested: selective execution (cheap model gates expensive ones) | — | later test |\n")

P("### T10. Recommended ACCURATE recipe (models/v5b/fusion/accurate_recipe.json)\n")
P("| field | value |\n|---|---|")
P(f"| models | {', '.join(rec['models'])} |")
P(f"| fusion | {rec['fusion']} |")
P(f"| calibration | per-model monotone map raw -> neg-image-FP quantile, refit on all 1,158 VAL negatives ({', '.join(str(len(s['calibration']['knots_raw'])) for s in rec['models'].values())} knots) |")
P(f"| raw score at calibrated 0.95 / 0.99 | " + "; ".join(f"{m}: {s['raw_score_at_calibrated']['0.95']} / {s['raw_score_at_calibrated']['0.99']}" for m, s in rec["models"].items()) + " |")
P(f"| fused-score thresholds (pooled OOF) | {rec['thresholds_oof']} |")
P(f"| fused-score thresholds by held-out site | {rec['thresholds_oof_by_site']} |")
P(f"| image-level | {rec['image_level']['method']} (selected by {rec['image_level']['selected_by']}); thresholds {rec['image_level']['thresholds_oof']} |")
ins = R["recipe_insample_all_val"]
P(f"| in-sample check (all-VAL calibration, whole VAL) recall/tiny | " + ", ".join(f"{fp[2:]}: {cell(ins, fp)}" for fp in FPS) + " |")
Path(ROOT / "reports/v5b/ensemble_dev_tables.md").write_text("\n".join(out) + "\n")
print("wrote reports/v5b/ensemble_dev_tables.md", len(out), "lines")
