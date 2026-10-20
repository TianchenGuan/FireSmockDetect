"""V5b report tables, generated from saved artifacts only (no model run, no cluster job).

Adapted from scripts/v4/make_report_tables.py (same emit()/md_table() conventions, same
matched-FP interpolation as scripts/v4/compare_matched_fp.py, same whole-node ledger logic as
scripts/v4/make_compute_ledger.py) but driven by the V5b artifact layout:

  T1  data sources & permissions        data/v5b/manifests/*_manifest.parquet, seg_v5b_manifest.json, CORPUS_B
  T2  corpus composition & budget       data/v5b/manifests/CORPUS_{B,C,D,D2,D3}_v1.json
  T3  model registry + v5b arms         models/v5b/MODEL_REGISTRY.json, cluster/v5b/registry.json, logs/v5b, b200 metrics
  T4  VAL matched-FP (3 sets)           reports/v5b/val/eval_<sys>_<set>.raw.parquet  (sweep_report, IoU>=0.4)
  T5  ensemble OOF                      reports/v5b/ensemble_dev_v5b.json
  T6  segmentation                      models/v5b/seg/segV5b_a/{MANIFEST.json, seg*/<cfg>/val_metrics_best.json, thr_sweep.json}
  T7  latency                           registry, ensemble latency_table, demo replay runs, SEGMENTATION_V5B.md (text)
  T8  FINAL TEST                        reports/v5b/test/eval_<sys>_<set>.{json,raw.parquet}  -> PENDING until they exist
  T9  compute by GPU type               fresh `kraken jobs list` dumps (scrubbed) -> reports/v5b/ledger, COMPUTE_LEDGER_V5B.csv

Every table lists its source artifacts. Missing artifacts are reported, never invented.

Usage:
  .venv/bin/python scripts/v5b/make_report_tables_v5b.py [--out reports/v5b/tables] [--no-dump] [--skip-sweeps]
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
import subprocess
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
R5B = ROOT / "reports/v5b"
MAN = ROOT / "data/v5b/manifests"
VAL = R5B / "val"
TEST = R5B / "test"
LEDGER = R5B / "ledger"

FP_KEYS = ["fp0.01", "fp0.02", "fp0.05", "fp0.10", "fp0.20"]
FP_LABEL = {"fp0.01": "1%", "fp0.02": "2%", "fp0.05": "5%", "fp0.10": "10%", "fp0.20": "20%"}
FP_TARGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
VAL_SETS = ["sdisval", "borealval", "firemanval"]
VAL_SET_DESC = {
    "sdisval": "pyro-sdis VAL (fixed camera; sites cabanelle + marguerite; 7,335 imgs / 1,158 negatives; 1% FP = ~12 negatives)",
    "borealval": "Boreal heinola aerial VAL (1,123 imgs / 217 negatives; 1% FP = ~2 negatives -> 1-2% rows are weakly supported)",
    "firemanval": "FireMan val_s5 (burn D video_015, every 5th frame; 1,706 imgs / 231 author negatives; one video = one unit; author boxes are large region boxes)",
}
TEST_SETS = ["firemanholdout", "sdistest", "sheffieldholdout", "borealtest", "foresttest"]
TEST_SET_DESC = {
    "firemanholdout": "H1 fireman_holdout_s10 (burn C Taivalkoski, 5 videos, 1,949 frames, 0 negatives -> RECALL ONLY)",
    "sdistest": "H2 pyro-sdis TEST (ferion + croix-augas, 4,124 imgs / 917 negatives; second opening after v4)",
    "sheffieldholdout": "H3 sheffield_holdout rec_0008 (327 frames, 0 negatives -> recall only; WITHIN-INCIDENT; CONTAMINATED for arms whose corpus holds Sheffield)",
    "borealtest": "H4 boreal ruokolahti TEST (1,764 imgs / 2 negatives -> FP unmeasurable; historical continuity)",
    "foresttest": "H4 AuSmoke forest TEST (2,871 imgs, positives only; permission pending -> shown, not used for decisions)",
}
DINO_RE = re.compile(r"(dino|lora|frozen|_B[1456](_|$)|b200|hr|S6)", re.I)
# pre-registered roster (reports/v5b/PREREGISTRATION_V5B.md) -> H3 contamination by training corpus
# (LOCKBOX_V5B_FROZEN.md: Sheffield rec_0008 sits in the train lists of arms B/C/D (and D2/D3 reuse the B corpus)).
# case-sensitive on purpose: a seed suffix like "_s3" must not be read as pre-registered system S3
H3_CONTAMINATED_RE = re.compile(r"(^(S2|S3|S4)([_-]|$))|((^|[_-])(arm_?[BCD]\d?|v5b_?[BCD]\d?|expV5b[BCD]\d?|[BCD]|D2|D3)([_-]|$))")
H3_CLEAN_RE = re.compile(r"(?i)(^S0b?([_-]|$)|^S1([_-]|$)|^S5([_-]|$)|^S6([_-]|$)|expB|expE|ctrl|accurate|ensemble|b200|dino)")

MISSING: list[str] = []
NOTES: list[str] = []
PENDING: list[str] = []
TABLES: "OrderedDict[str, str]" = OrderedDict()


# ----------------------------------------------------------------------------- helpers (v4 conventions)
def rel(p) -> str:
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


def fmt_int(x):
    return "n/a" if x is None else f"{int(x):,}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def emit(tid, title, caption, body, sources):
    src = "\n".join(f"- `{s}`" for s in sources)
    TABLES[tid] = f"### {tid}. {title}\n\n{caption}\n\n{body}\n\nSources:\n{src}\n"


def grep_first(path: Path, pattern: str):
    if not path.exists():
        return None
    for ln in path.read_text(errors="replace").splitlines():
        if re.search(pattern, ln):
            return ln.strip()
    return None


# ----------------------------------------------------------------------------- matched-FP machinery (compare_matched_fp.py logic)
_SWEEP = None


def _sweep_fn():
    global _SWEEP
    if _SWEEP is None:
        try:
            from src.eval.eval_detection import sweep_report  # noqa
            _SWEEP = sweep_report
        except Exception as e:  # pragma: no cover
            NOTES.append(f"sweep_report import failed ({e!r}); matched-FP tables degrade to the eval JSON grid")
            _SWEEP = False
    return _SWEEP


def interp_at_fp(sweep, target, key):
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


def interp_at_conf(sweep, conf):
    """(recall, tiny_recall) linearly interpolated vs conf on a sweep list (fallback when no parquet)."""
    pts = sorted((s["conf"], s["recall"], (s.get("recall_by_size") or {}).get("tiny")) for s in sweep)
    if conf <= pts[0][0]:
        return pts[0][1], pts[0][2]
    if conf >= pts[-1][0]:
        return pts[-1][1], pts[-1][2]
    for a, b in zip(pts, pts[1:]):
        if a[0] <= conf <= b[0]:
            w = 0 if b[0] == a[0] else (conf - a[0]) / (b[0] - a[0])
            t = None if (a[2] is None or b[2] is None) else a[2] + w * (b[2] - a[2])
            return a[1] + w * (b[1] - a[1]), t
    return None, None


def grid_for(sys_token: str):
    g = [round(x / 100, 4) for x in range(1, 100)]
    if DINO_RE.search(sys_token):
        g = sorted(set(g) | {round(0.991 + 0.001 * i, 4) for i in range(9)})
    return g


def _cache_key(parquet: Path, grid):
    st = parquet.stat()
    h = hashlib.sha1(f"{parquet}|{st.st_size}|{int(st.st_mtime)}|{len(grid)}|{grid[-1]}".encode()).hexdigest()[:16]
    return h


def curve_from_parquet(parquet: Path, grid, cache_dir: Path, skip_sweeps=False):
    """sweep_report on a .raw.parquet at the given conf grid; cached by parquet size/mtime."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"curve_{parquet.stem.replace('.raw', '')}_{_cache_key(parquet, grid)}.json"
    if cp.exists():
        return json.loads(cp.read_text())
    if skip_sweeps:
        NOTES.append(f"--skip-sweeps: no cached curve for {rel(parquet)}; table cell left n/a")
        return None
    fn = _sweep_fn()
    if not fn:
        return None
    import pandas as pd
    rows = pd.read_parquet(parquet).to_dict("records")
    rep = fn(rows, 0.4, conf_grid=grid)
    rep["source_parquet"] = rel(parquet)
    cp.write_text(json.dumps(rep))
    return rep


def recall_at_confs_exact(parquet: Path, confs):
    fn = _sweep_fn()
    if not fn or not parquet.exists():
        return None
    import pandas as pd
    rows = pd.read_parquet(parquet).to_dict("records")
    grid = sorted(set(round(c, 6) for c in confs))
    rep = fn(rows, 0.4, conf_grid=grid)
    return {round(s["conf"], 6): s for s in rep["conf_sweep"]} | {"_n": (rep["n_images"], rep["n_positive_images"], rep["n_negative_images"])}


def matched_fp_from_curve(rep):
    sw = rep["conf_sweep"]
    out = {}
    for t in FP_TARGETS:
        k = f"fp{t:.2f}"
        out[k] = {
            "conf": interp_at_fp(sw, t, lambda s: s["conf"]),
            "recall": interp_at_fp(sw, t, lambda s: s["recall"]),
            "precision": interp_at_fp(sw, t, lambda s: s["precision"]),
            "tiny_recall": interp_at_fp(sw, t, lambda s: s["recall_by_size"]["tiny"]),
        }
    return out


def discover_systems(folder: Path, sets):
    """{sys_token: {set: parquet_or_json_path}} from eval_<sys>_<set>.raw.parquet (or .json)."""
    found = defaultdict(dict)
    for p in sorted(folder.glob("eval_*")):
        m = re.match(r"eval_(.+)_(" + "|".join(sets) + r")\.(raw\.parquet|json)$", p.name)
        if not m:
            continue
        sysn, st, ext = m.groups()
        cur = found[sysn].get(st)
        if ext == "raw.parquet" or cur is None:
            found[sysn][st] = p
    return found


# ----------------------------------------------------------------------------- T1 data sources
def data_sources_table():
    import pandas as pd
    rows, srcs = [], []

    def card(name):
        p = MAN / f"{name}_CARD.md"
        srcs.append(rel(p))
        return p

    def mf(name):
        p = MAN / f"{name}_manifest.parquet"
        if not p.exists():
            MISSING.append(rel(p))
            return None
        srcs.append(rel(p))
        return pd.read_parquet(p)

    corp_b = load_json(MAN / "CORPUS_B_v1.json") or {}
    pc = (corp_b.get("counts") or {}).get("per_component", {})
    nb = (corp_b.get("counts") or {}).get("n_gt_boxes_by_source", {})
    reg = load_json(ROOT / "models/v5b/MODEL_REGISTRY.json") or {}
    e_recipe = next((m["train_data_recipe"] for m in reg.get("models", []) if m["id"] == "expB_y11s"), "")
    e_neg = re.search(r"incl\. ([\d,]+) negatives", e_recipe)
    srcs += ["data/v5b/manifests/CORPUS_B_v1.json", "models/v5b/MODEL_REGISTRY.json (train_data_recipe of expB_y11s)"]
    sdis_lic = grep_first(ROOT / "reports/FINAL_REPORT.md", r"(?i)^- pyro-sdis @.*\(") or "n/a"
    dfire_lic = grep_first(ROOT / "reports/FINAL_REPORT.md", r"(?i)^- D-Fire \(") or "n/a"
    sdis_lic = (re.search(r"\(([^)]+)\)", sdis_lic) or [None, sdis_lic])[1]
    dfire_lic = (re.search(r"\(([^)]+)\)", dfire_lic) or [None, dfire_lic])[1]
    srcs.append("reports/FINAL_REPORT.md (iteration-1 data section: licence lines)")
    rows.append(["E corpus = pyro-sdis TRAIN sites + D-Fire smoke (frozen expB_train.txt)", fmt_int(pc.get("E")),
                 f"{fmt_int(nb.get('sdis'))} + {fmt_int(nb.get('dfire'))}", e_neg.group(1) if e_neg else "n/a",
                 "4 sdis TRAIN sites + D-Fire (no units)", f"pyro-sdis: {sdis_lic}; D-Fire: {dfire_lic} (reports/FINAL_REPORT.md)", "inherited (v1-v4)",
                 "YES - every arm (A,B,C,D,D2,D3), x1", "sdis VAL/TEST sites held out"])

    bl = grep_first(ROOT / "data/v4/manifests/boreal_CARD.md", r"CC BY 4\.0")
    srcs.append("data/v4/manifests/boreal_CARD.md")
    rows.append(["Boreal Forest Fire (FGI) TRAIN burns evo + karkkila, Subset-A boxes", fmt_int(pc.get("boreal")), fmt_int(nb.get("boreal")),
                 "37 (karkkila Empty-Images clip)", "2 burns / 13 flights", (bl or "CC BY 4.0 (boreal_CARD.md)")[:70], "confirmed",
                 "YES - arms B,C,D,D2,D3 x2 (and v4 E+T models)", "heinola = aerial VAL; ruokolahti = historical TEST"])

    seg = load_json(MAN / "seg_v5b_manifest.json")
    if seg:
        srcs.append("data/v5b/manifests/seg_v5b_manifest.json")
        tr = [r for r in seg if r["role"] == "train"]
        va = [r for r in seg if r["role"] == "val"]
        rows.append(["Boreal Subset-C masks (SAM model-assisted + 27 human) + author negatives [SEG branch]",
                     f"{len(tr)} train / {len(va)} val", f"masks: {sum(1 for r in tr if r['has_mask'] and not r['is_negative'])} / {sum(1 for r in va if r['has_mask'] and not r['is_negative'])}",
                     f"{sum(1 for r in tr if r['is_negative'])} / {sum(1 for r in va if r['is_negative'])}",
                     f"train evo+karkkila {len({r['flight'] for r in tr})} flights; val heinola {len({r['flight'] for r in va})} flights",
                     "CC BY 4.0 (same source)", "confirmed", "YES - segV5b_a only (not detectors)", "heinola masks = VAL; ruokolahti masks (453) never staged"])

    fm = mf("fireman"); card("fireman")
    if fm is not None:
        rgb = fm[fm.modality == "rgb"]
        units = rgb.parent_unit.nunique()
        confl = rgb[rgb.label_conflict.fillna("") != ""]
        rows.append(["FireMan-UAV-RGBT (Oulu) RGB, Multiclass CVAT", fmt_int(len(rgb)), f"{int(rgb.n_smoke_boxes.sum()):,} on {int((rgb.n_smoke_boxes > 0).sum()):,} frames (incl. {int(confl.n_smoke_boxes.sum())} on {int((confl.n_smoke_boxes > 0).sum())} conflict frames; CARD says 69,232)",
                     fmt_int(rgb.is_explicit_negative.sum()), f"{units} burns / {rgb.video_id.nunique()} videos",
                     rgb.licence.iloc[0], rgb.permission_status.iloc[0],
                     f"YES - burn A train capped 600/video = {fmt_int(pc.get('fireman'))} lines (arms B,C,D,D2,D3)",
                     "burn D video_015 = aerial VAL (fireman_val_s5); burn C = H1 holdout (sealed); burn B = quarantine (Ruokolahti candidate)"])
        th = fm[fm.modality == "thermal"]
        rows.append(["FireMan thermal (no smoke supervision)", fmt_int(len(th)), "0 smoke boxes", "0", f"{th.video_id.nunique()} videos",
                     th.licence.iloc[0], th.permission_status.iloc[0], "NO (not exported)", "future RGB-T ablation only"])

    sh = mf("sheffield_uav"); card("sheffield_uav")
    if sh is not None:
        rows.append(["Sheffield UAV (ORDA, Roboflow re-deposit)", fmt_int(len(sh)), fmt_int(sh.n_smoke_boxes.sum()),
                     f"{int(sh.is_explicit_negative.sum())} (24 exported; 6 smoke-visible excluded)",
                     f"1 session / {sh.parent_unit.nunique()} recordings", sh.licence.iloc[0], sh.permission_status.iloc[0],
                     f"YES - ALL {fmt_int(pc.get('sheffield'))} exported frames x2 (arms B,C,D,D2,D3) incl. rec_0008",
                     "rec_0008 = H3 (CONTAMINATED for B/C/D/D2/D3; within-incident for everyone)"])

    fs = mf("firespot"); card("firespot")
    if fs is not None:
        rows.append(["FireSpot-CNX-r1 (NECTEC)", fmt_int(len(fs)), fmt_int(fs.n_smoke_boxes.sum()), fmt_int(fs.is_explicit_negative.sum()),
                     f"{fs.parent_unit.nunique()} sites / {fs.session_unit.nunique()} sessions", "UNSPECIFIED (no LICENSE anywhere; MNS 'CC BY 4.0' is an aggregator claim)",
                     fs.permission_status.iloc[0], "NO (guarded converter refuses); NOTE 2,862 MNS copies were in v4 G corpora (R3)", "none"])

    ws = mf("wsdataset"); card("wsdataset")
    if ws is not None:
        new = ws[ws.new_data == True]  # noqa: E712
        rows.append(["WSDataset (Kaggle gloryvu; MNS copies + GitHub demo)", f"{len(ws):,} rows ({len(new)} new files)",
                     f"{int(ws.n_smoke_boxes.sum()):,} (mask-derived, model-assisted; author boxes BLOCKED)",
                     f"{int(ws.is_explicit_negative.sum())} new author negatives", f"{ws.parent_unit.nunique()} units (66 videos recovered)",
                     "MIT declared on Kaggle page (text unverified); GitHub demo unlicensed", ws.permission_status.iloc[0],
                     "NO in v5b (MNS copies were v4 G material); original 18.4 GB archive BLOCKED (Kaggle token)", "none"])

    d3 = load_json(MAN / "CORPUS_D3_v1.json") or {}
    d3c = (d3.get("counts") or {}).get("per_component", {})
    srcs.append("data/v5b/manifests/CORPUS_D3_v1.json")
    rows.append(["UNLABELLED smoke-positive material for D3: Boreal train-burn Subset-B clip frames (104 smoke clips), FIRESENSE smoke_pos, D-Fire fire-only",
                 fmt_int(d3c.get("d3")), "0 author boxes -> 3,797 teacher pseudo boxes (model-assisted)", "0 (no teacher-silence negatives created)",
                 "3,370 + 530 + 944 target frames (TEACHER_POOL.md D3 section)", "Boreal CC BY 4.0; FIRESENSE / D-Fire inherited", "inherited",
                 "YES - arm D3 only (x1, pseudo boxes)", "none"])
    rows.append(["AuSmoke (via MultiNatSmoke) T list", "10,904", "-", "-", "-", "permission PENDING (PROTOCOL_V5 S2)", "pending",
                 "NO new use in v5b (v4 B1/B4/B6/E_ET models already trained on it - disclosed)", "AuSmoke forest TEST = H4 historical"])
    emit("T1", "Data sources, permissions and use in V5b training",
         "Counts are read from the V5b manifests (agent-built, nothing human-reviewed; `annotation_origin=author` everywhere except WSDataset masks = model-assisted "
         "and the D3 pseudo boxes = model-assisted). 'Used in training' names the arms whose train lists contain the source (data/v5b/manifests/CORPUS_*_v1.json). "
         "FireMan boxes are interpolated region boxes (presence-grade, not tight instances); Sheffield boxes are fragmented sub-boxes per plume.",
         md_table(["source", "frames (rows)", "smoke boxes", "explicit negatives", "units", "licence (verbatim/short)", "permission_status", "used in V5b training", "eval role"], rows),
         sorted(set(srcs)))


# ----------------------------------------------------------------------------- T2 corpus composition
def corpus_tables():
    arms = ["B", "C", "D", "D2", "D3"]
    rows, extra, srcs = [], [], []
    a_row = None
    for a in arms:
        p = MAN / f"CORPUS_{a}_v1.json"
        d = load_json(p)
        if not d:
            continue
        srcs.append(rel(p))
        b = d["budget"]
        if a_row is None:
            aa = b["arm_A"]
            a_row = ["A (control; expV5bA_ctrl_s3, seed-0 replicate)", fmt_int(aa["n_lines"]), "E x1", aa["epochs"], fmt_int(aa["exposures"]), "0.000%", "-", "-", "author only"]
        mult = d["multiplicities"]
        mtxt = ", ".join(f"{k} {v['n_raw']:,}x{v['multiplicity']}={v['n_effective']:,}" for k, v in mult.items())
        masked = d.get("masked") or {}
        pseudo = d.get("pseudo") or {}
        lo = d["counts"].get("per_label_origin", {})
        if a == "D3":
            dd = d.get("d3") or {}
            pseudo_txt = f"{mult.get('d3', {}).get('n_raw', 'n/a')} unlabelled imgs kept with pseudo boxes (rule: {str(dd.get('rule', ''))[:90]}...)"
        elif pseudo:
            pseudo_txt = f"{pseudo.get('pseudo_boxes', 'n/a')} boxes on {pseudo.get('images_with_pseudo', 'n/a')} imgs; by source {pseudo.get('boxes_by_source') or pseudo.get('images_by_source')}"
        else:
            pseudo_txt = "-"
        masked_txt = f"{masked.get('rects_filled')} rects on {masked.get('images_masked')} imgs ({masked.get('images_by_source')})" if masked else "-"
        rows.append([a, fmt_int(b["n_lines"]), mtxt, b["epochs"], fmt_int(b["exposures"]), f"{b['delta_vs_A_pct']:+.3f}%", masked_txt, pseudo_txt,
                     ", ".join(f"{k} {v:,}" for k, v in lo.items())])
        c = d["counts"]
        extra.append([a, fmt_int(c.get("fireman_explicit_negatives")), fmt_int(c.get("sheffield_explicit_negatives")),
                      fmt_int(c.get("new_source_effective_lines")), f3(c.get("new_source_effective_over_E_pct"), 1) + "%" if c.get("new_source_effective_over_E_pct") is not None else "n/a",
                      "; ".join(d.get("excluded", []))[:160] + ("..." if len("; ".join(d.get("excluded", []))) > 160 else "")])
    if a_row:
        rows.insert(0, a_row)
    emit("T2", "Training corpora per arm: lines, multiplicities, epochs, exposures (matched budget), masked / pseudo-label counts",
         "Exposures = lines x epochs, matched to arm A (38,454 x 60 = 2,307,240) within -0.3%/-0.65%. C = B + conservative ignore regions (teacher-agreed, "
         "GT-free rectangles filled with image mean); D = B + filtered teacher pseudo boxes on already-labelled E/Boreal images; D2 = D rules + looser all-source "
         "pseudo-labels (near-replicate of D: only 408 boxes, 0 on FireMan); D3 = B + teacher pseudo boxes on genuinely UNLABELLED smoke-positive material "
         "(the real teacher-to-student test). C/D are small by construction: on already-labelled data the teachers agree with the authors.",
         md_table(["arm", "train lines", "components raw x multiplicity = effective", "epochs", "exposures", "delta vs A", "masked (C)", "pseudo boxes (D/D2/D3)", "label origin"], rows)
         + "\n\n**new-source share and exclusions**\n\n" + md_table(["arm", "FireMan explicit negs", "Sheffield explicit negs", "new-source effective lines", "new-source / E", "excluded (from the manifest)"], extra),
         srcs)


# ----------------------------------------------------------------------------- T3 registry + v5b arms
def registry_tables():
    reg = load_json(ROOT / "models/v5b/MODEL_REGISTRY.json")
    rows = []
    if reg:
        for m in reg["models"]:
            v = (m.get("val") or [{}])[0]
            mfp = v.get("matched_fp") or {}
            r5, r10 = mfp.get("fp0.05") or {}, mfp.get("fp0.10") or {}
            valtxt = (f"R@5% {f3(r5.get('recall'))} (tiny {f3(r5.get('tiny'))}); R@10% {f3(r10.get('recall'))} (tiny {f3(r10.get('tiny'))}) [{v.get('artifact', '')}]"
                      if mfp else (v.get("note") or "none")[:90])
            fig = m.get("figlib39") or {}
            figtxt = f"{fig['n_detected']}/{fig['n_events']} (med {fig.get('median_delay_s')} s)" if fig.get("n_events") else "-"
            el = m.get("eligible_for") or {}
            prm = m.get("params") or {}
            rows.append([m["id"], m["family"], f"{prm.get('trainable_at_train_time', 'n/a')} / {prm.get('total', 'n/a')}" if isinstance(prm.get("total"), int) else "n/a",
                         (m.get("train_data_recipe") or "")[:70] + "...", valtxt, figtxt,
                         "yes" if m.get("exists") else "NO", "cluster-only" if m.get("cluster_only") else ("excluded" if m.get("excluded") else ("yes" if m.get("loads") else "no")),
                         m.get("backend") or "-", f"{f3(m.get('latency_cpu_measured_median_ms'), 0)} / {f3(m.get('latency_l40s_batch1_ms'), 1)}",
                         (m.get("licence_of_weights") or "")[:40], "/".join("Y" if el.get(k) else "n" for k in ("fast_mode", "accurate_mode", "teacher")),
                         ("; ".join(m.get("flags") or []))[:80]])
        emit("T3a", "Model registry (V5b phase 1): checkpoints that exist, load, and their frozen VAL numbers",
             f"Generated {reg.get('generated_utc')} by scripts/v5b/build_model_registry.py on the CPU devbox. Eligibility rules: fast = loads on CPU + CPU batch-1 <= 1000 ms; "
             "accurate = loads + L40S batch-1 <= 400 ms + licence permits internal serving; teacher = loads + licence permits derivative use + not FIgLib-trained + "
             "(sdis-VAL R@5% >= 0.202 or FIgLib-39 coverage > 30/39). Flags R1 (arm-A seeds all 0) and R2 (B6 shipped ckpt != run best) are in reports/v5b/RECONCILIATION_V5B.md.",
             md_table(["id", "family", "params trainable / total", "train recipe", "sdis-VAL matched-FP [artifact]", "FIgLib-39", "exists", "loads", "backend", "latency CPU / L40S ms", "licence", "fast/acc/teacher", "flags"], rows),
             ["models/v5b/MODEL_REGISTRY.json", "models/v5b/MODEL_REGISTRY.md", "reports/v5b/RECONCILIATION_V5B.md"])

    # v5b systems in training (not yet in the registry)
    rg = load_json(ROOT / "cluster/v5b/registry.json") or {"jobs": {}}
    arms = [("A", "expV5bA_ctrl_s3", "res-v5bactrls3", "armA_s3", None), ("B", "expV5bB_s0", "res-v5bbs0", "armB_s0", "B"), ("C", "expV5bC_s0", "res-v5bcs0", "armC_s0", "C"),
            ("D", "expV5bD_s0", "res-v5bds0", "armD_s0", "D"), ("D2", "expV5bD2_s0", "res-v5bd2s0", "armD2_s0", "D2"), ("D3", "expV5bD3_s0", "res-v5bd3s0", "armD3_s0", "D3")]
    rows2, srcs2 = [], ["cluster/v5b/registry.json", "logs/v5b/arm*.log (last 'submitted' line)"]
    now = datetime.now(timezone.utc)
    for arm, run, res, log, corp in arms:
        j = rg["jobs"].get(res, {})
        r0 = datetime.fromisoformat(j["reserved_utc"].replace("Z", "+00:00")) if j.get("reserved_utc") else None
        r1 = datetime.fromisoformat(j["released_utc"].replace("Z", "+00:00")) if j.get("released_utc") else now
        hours = f"{(r1 - r0).total_seconds() / 3600:.2f}" if r0 else "n/a"
        lp = ROOT / f"logs/v5b/{log}.log"
        seg = "n/a"
        if lp.exists():
            subs = [ln for ln in lp.read_text(errors="replace").splitlines() if "submitted" in ln]
            seg = subs[-1].strip()[:70] if subs else lp.read_text(errors="replace").splitlines()[-1].strip()[:70]
        else:
            MISSING.append(rel(lp))
        c = load_json(MAN / f"CORPUS_{corp}_v1.json") if corp else None
        if corp:
            srcs2.append(f"data/v5b/manifests/CORPUS_{corp}_v1.json")
        corpus = f"{c['budget']['n_lines']:,} lines / {c['budget']['epochs']} ep" if c else "38,454 lines / 60 ep (expB recipe)"
        rows2.append([arm, run, "YOLO11m COCO-init @1280 b48, seed 0" + (" (REPLICATE #4 of seed 0, R1)" if arm == "A" else ""), corpus, j.get("state", "n/a"), hours, seg])
    # B200 high-res run
    mp = ROOT / "models/v5b/b200/v5bB200_lora_L_1920/metrics.jsonl"
    b200 = "n/a"
    if mp.exists():
        srcs2.append(rel(mp))
        lines = [json.loads(x) for x in mp.read_text().splitlines() if x.strip()]
        evals = [x for x in lines if "eval" in x]
        last = [x for x in lines if "step" in x and "eval" not in x]
        best = max(evals, key=lambda x: x["eval"]["mAP"]) if evals else None
        b200 = (f"{len(evals)} epoch evals; last train step {last[-1]['step'] if last else 'n/a'} (of 12,660), "
                f"best 1,500-img sdis-val COCO mAP {best['eval']['mAP']:.4f} @ step {best['step']}" if evals else "no eval yet")
    j = rg["jobs"].get("res-b200dinohr", {})
    rows2.append(["S6", "v5bB200_lora_L_1920", "DINOv3-L LoRA r16 + SFP + Faster R-CNN @1920x1088, bs 4/GPU, 10 ep (B4 recipe)", "E + Boreal-train 40,516 imgs (no AuSmoke)",
                  j.get("state", "n/a"), f"{(now - datetime.fromisoformat(j['reserved_utc'].replace('Z', '+00:00'))).total_seconds() / 3600:.2f} (B200)" if j.get("reserved_utc") else "n/a", b200])
    emit("T3b", "V5b systems in training at generation time (not yet in the registry; numbers are training-time only)",
         "Reservation hours are registry wall-clock (reserve -> release/now), not scheduler allocation (see T9). 'last segment' = the last job the orchestrator submitted "
         "(segments resume from /scratch last.pt). The B200 run's COCO mAP is the training-time 1,500-image sdis-val subset metric (same as v4 B4's 0.4086 best), not the matched-FP protocol.",
         md_table(["system", "run", "recipe", "corpus", "registry state", "reserved node-h", "last segment / progress"], rows2), sorted(set(srcs2)))


# ----------------------------------------------------------------------------- T4 VAL matched-FP
def val_tables(derived: Path, skip_sweeps=False):
    found = discover_systems(VAL, VAL_SETS)
    curves, matched, confs = {}, {}, {}
    if not found:
        PENDING.append("T4 (no reports/v5b/val/eval_*_<set>.raw.parquet yet)")
    for st in VAL_SETS:
        rows, srcs = [], []
        for sysn in sorted(found):
            p = found[sysn].get(st)
            if p is None:
                rows.append([sysn] + ["PENDING"] * 5 + ["-"])
                continue
            srcs.append(rel(p))
            if p.suffix == ".parquet":
                rep = curve_from_parquet(p, grid_for(sysn), derived / "_cache", skip_sweeps)
            else:
                rep = load_json(p)
                NOTES.append(f"T4 {sysn}/{st}: only the eval JSON (10-point CONF_GRID) exists; matched-FP interpolated on that coarse grid")
            if not rep:
                rows.append([sysn] + ["n/a"] * 5 + ["-"])
                continue
            curves[(sysn, st)] = rep
            mf = matched_fp_from_curve(rep)
            matched.setdefault(st, {})[sysn] = {"n_images": rep["n_images"], "n_negative_images": rep["n_negative_images"], "n_positive_images": rep["n_positive_images"], "matched_fp": mf}
            cells = []
            for k in FP_KEYS:
                d = mf[k]
                if d["recall"] is None:
                    fps = [c["neg_image_fp_rate"] for c in rep["conf_sweep"] if c.get("neg_image_fp_rate") is not None]
                    cells.append(f"n/a (FP floor {100 * min(fps):.1f}%)" if fps and float(k[2:]) < min(fps) else "n/a")
                else:
                    cells.append(f"c={d['conf']:.3f} R {d['recall']:.3f} / tiny {f3(d['tiny_recall'])} / P {f3(d['precision'])}")
            gts = rep["conf_sweep"][0].get("gt_by_size", {}) if rep["conf_sweep"] else {}
            rows.append([sysn] + cells + [f"{rep['n_images']:,} imgs / {rep['n_negative_images']:,} neg; tiny GT {gts.get('tiny', 'n/a')}"])
            if st == "sdisval":
                confs.setdefault(sysn, {})["sdisval"] = {k: mf[k]["conf"] for k in ("fp0.05", "fp0.10", "fp0.20")}
            if st == "firemanval":
                confs.setdefault(sysn, {})["firemanval"] = {k: mf[k]["conf"] for k in ("fp0.05", "fp0.10")}
        tid = {"sdisval": "T4a", "borealval": "T4b", "firemanval": "T4c"}[st]
        emit(tid, f"VAL matched-FP: {VAL_SET_DESC[st]}",
             "Recall / tiny-recall (<0.1% area GT) / precision at matched negative-image FP, interpolated on each system's own confidence sweep "
             "(src.eval.eval_detection.sweep_report, IoU>=0.4, grid 0.01..0.99 step 0.01; +0.991..0.999 for DINO-family tokens, as scripts/v4/compare_matched_fp.py --top-fine). "
             "'tiny n/a' = the set has no tiny GT boxes (FireMan/Boreal author boxes are large plume regions). The eval JSONs record `imgsz: 1152`, which is "
             "eval_detection's CLI default when fed --preds-jsonl; the pod inference ran at IMGSZ=1280 (cluster/v4/yolo_eval_entry.sh default). "
             "FireMan VAL recall saturates for v1-era models -> it is NOT a faint-smoke benchmark (STATUS 10:57 UTC).",
             md_table(["system", "1% FP", "2% FP", "5% FP", "10% FP", "20% FP", "denominators"], rows) if rows else "_PENDING: no eval parquet for this set_",
             srcs or [f"reports/v5b/val/eval_<sys>_{st}.raw.parquet (none yet)"])
    (derived / "matched_fp_val_v5b.json").write_text(json.dumps({"match_iou": 0.4, "fp_targets": FP_TARGETS, "sets": matched}, indent=1))
    (derived / "preregistered_confs_v5b.json").write_text(json.dumps({
        "note": "per-system confidence at matched neg-image FP on VAL (PREREGISTRATION_V5B.md 'Thresholds per system'): sdis-VAL 5/10/20% (fixed camera), FireMan-VAL 5/10% (aerial). Computed from reports/v5b/val/*.raw.parquet by make_report_tables_v5b.py; VAL-only.",
        "systems": confs}, indent=1))
    return confs, curves


# ----------------------------------------------------------------------------- T5 ensemble
def ensemble_table():
    p = R5B / "ensemble_dev_v5b.json"
    d = load_json(p)
    if not d:
        PENDING.append("T5 (reports/v5b/ensemble_dev_v5b.json missing)")
        emit("T5", "Ensemble OOF", f"ARTIFACT MISSING: `{rel(p)}`", "_PENDING_", [rel(p)])
        return
    SHORT = {"yolo_A": "A", "dino_B4": "B4", "dino_B6": "B6", "rfdetr_L": "RF", "yolo_expE": "expE", "dino_B1": "B1"}
    rows = []
    for m, s in d["single_raw"].items():
        bf = s["by_fold"]
        macro = (bf["cabanelle"]["mean_recall"] + bf["marguerite"]["mean_recall"]) / 2
        rows.append([f"single {SHORT.get(m, m)} ({m})", f3(bf["cabanelle"]["mean_recall"]), f3(bf["marguerite"]["mean_recall"]), f3(macro, 4), f3(s["mean_recall"]),
                     " / ".join(f3(s[k]["recall"]) for k in FP_KEYS), "-"])
    for st, s in d["sets"].items():
        for meth in ("union", "wbf"):
            v = s[meth]
            bf = v["oof_by_fold"]
            prm = v.get("params_all_val") or {}
            ptxt = (f"{prm.get('conf_type')},iou{prm.get('iou_thr')},skip{prm.get('skip_box_thr')}" if "conf_type" in prm else f"iou{prm.get('iou_thr')}")
            rows.append([f"{'OR/union+NMS' if meth == 'union' else 'WBF'} {'+'.join(SHORT.get(x, x) for x in st.split('+'))}", f3(bf["cabanelle"]["mean_recall"]), f3(bf["marguerite"]["mean_recall"]),
                         f3(v["oof_macro_mean"], 4), f3((v.get("oof_pooled") or {}).get("mean_recall")), "-", ptxt])
    ch = "+".join(SHORT.get(x, x) for x in d["chosen_models"])
    add = [[a["candidate"], "+".join(SHORT.get(x, x) for x in a["set_tested"]), f3(a["union_mean_recall"], 4), f3(a["wbf_mean_recall"], 4), f"{a['gain_pts_vs_current_best']:+.2f}", "yes" if a["kept"] else "no"] for a in d["addition_protocol"]]
    boot = []
    for site, b in d["bootstrap"].items():
        for k, v in b.items():
            if isinstance(v, dict) and "point_pts" in v:
                boot.append([site, k, f"{v['point_pts']:+.2f}", f"[{v['ci95_pts'][0]:+.2f}, {v['ci95_pts'][1]:+.2f}]", f3(v["p_gain_le_0"])])
    lt = d.get("latency_table") or {}
    emit("T5", "Model collaboration on pyro-sdis VAL: site-grouped out-of-fold (OOF) mean recall over FP 1/2/5/10/20% (selection statistic = macro over the two held-out sites)",
         f"Fusions are fit on one VAL site and evaluated on the other (calibration maps + WBF params); single models need no fitting. Chosen ACCURATE recipe = {ch} {d['chosen_method'].upper()} "
         f"(macro {d['sets']['+'.join(d['chosen_models'])]['wbf']['oof_macro_mean']:.4f} vs best single B6 {max((s['by_fold']['cabanelle']['mean_recall'] + s['by_fold']['marguerite']['mean_recall']) / 2 for s in d['single_raw'].values()):.4f}). "
         f"Learned fusion (logreg) beats WBF by only {d['learned_beats_wbf_by_pts']:+.2f} pt (< 1 pt bar) -> not adopted. Pooled-OOF numbers are pessimistic because calibration does not transfer between sites "
         "(ENSEMBLE_DEV_V5B.md T8). Latency (batch-1 L40S fp32 @1280, from reports): components " + ", ".join(f"{SHORT.get(k, k)} {v} ms" for k, v in (lt.get("components_ms") or {}).items())
         + f"; fusion {lt.get('fusion_ms')} ms CPU; PARALLEL {lt.get('parallel_ms')} ms; serial {lt.get('serial_ms')} ms.",
         md_table(["system", "OOF cabanelle", "OOF marguerite", "macro", "whole-VAL / pooled", "whole-VAL R @1/2/5/10/20%", "WBF/union params (all-VAL fit)"], rows)
         + "\n\n**model-addition protocol (keep if >= +1.0 pt macro-OOF)**\n\n" + md_table(["candidate", "set tested", "union macro", "WBF macro", "gain (pt)", "kept"], add)
         + "\n\n**bootstrap (500 image resamples within the held-out site), difference in OOF mean recall (pt)**\n\n" + md_table(["held-out site", "comparison", "point", "95% CI", "P(gain<=0)"], boot),
         [rel(p), "reports/v5b/ENSEMBLE_DEV_V5B.md", "models/v5b/fusion/accurate_recipe.json"])


# ----------------------------------------------------------------------------- T6 segmentation
def seg_tables():
    run = ROOT / "models/v5b/seg/segV5b_a"
    man = load_json(run / "MANIFEST.json")
    if not man:
        PENDING.append("T6 (models/v5b/seg/segV5b_a/MANIFEST.json missing)")
        emit("T6", "Segmentation", "ARTIFACT MISSING", "_PENDING_", [rel(run / "MANIFEST.json")])
        return
    rows, srcs = [], [rel(run / "MANIFEST.json")]
    label = {"lora": "DINOv3-L LoRA r16 + SFP + FPN-lite", "frozen": "DINOv3-L frozen + SFP + FPN-lite", "r50": "ResNet-50 (ImageNet, frozen) + FPN-lite (baseline)"}
    for cfg, c in man["checkpoints"].items():
        s = c["val_metrics_at_best"]
        pcnt = c["param_counts"]["all"]
        rows.append([label.get(cfg, cfg), rel(ROOT / c["path"]), f"ep {c['best_epoch']} / step {c['best_step']}", f3(s["iou"]), f3(s["dice"]), f3(s["precision"]), f3(s["recall"]), f3(s["miou_pos"]),
                     f"{f3(s['iou_sam_model_assisted'])} ({s['n_sam_model_assisted']})", f"{f3(s['iou_manual_human'])} ({s['n_manual_human']})",
                     f"{f3(s['neg_frame_false_alarm'])} / {f3(s['neg_frame_fa_1pct'])}", f"{f3(s['box_recall_A_iou04'])} / {f3(s['box_recall_B_union_iou04'])} / {f3(s['box_recall_C_hit'])}",
                     f"{pcnt['trainable']:,} / {pcnt['total']:,}"])
    thr_rows = []
    for cfg in ("frozen", "lora", "r50"):
        tp = run / "seg01" / cfg / "thr_sweep.json"
        t = load_json(tp)
        if not t:
            continue
        srcs.append(rel(tp))
        pts = t if isinstance(t, list) else t.get("sweep") or t.get("rows") or []
        for e in pts:
            if isinstance(e, dict) and round(float(e.get("thr", -1)), 2) in (0.3, 0.5, 0.7):
                thr_rows.append([cfg, e["thr"], f3(e.get("iou")), f3(e.get("dice")), f3(e.get("precision")), f3(e.get("recall")), f3(e.get("neg_fa")), f3(e.get("neg_fa1")), f3(e.get("neg_fp_frac"), 4)])
    # per-epoch IoU from metrics.jsonl
    ep_rows = []
    for cfg, seg in (("lora", "seg01"), ("frozen", "seg02"), ("r50", "seg01")):
        mp = run / seg / cfg / "metrics.jsonl"
        if not mp.exists():
            continue
        srcs.append(rel(mp))
        ious = []
        for ln in mp.read_text().splitlines():
            try:
                j = json.loads(ln)
            except Exception:
                continue
            v = j.get("val") or j.get("eval") or {}
            if isinstance(v, dict) and "iou" in v:
                ious.append(v["iou"])
            elif "val_iou" in j:
                ious.append(j["val_iou"])
        if ious:
            ep_rows.append([cfg] + [f3(x) for x in ious[:10]])
    emit("T6", "Thin-smoke SEGMENTATION branch (segV5b_a): heinola aerial VAL, thr 0.5, best epoch selected on VAL (selection-optimistic)",
         f"Train = 665 Boreal TRAIN-burn frames (628 masks: 601 SAM model-assisted + 27 human; 37 author negatives), VAL = 382 heinola frames (329 SAM + 7 human + 46 author negatives). "
         f"{man['node_minutes_total']} node-minutes on 1 L40S node ({man['registry_job']}). IoU vs SAM masks measures agreement with SAM, not with humans; the 7 human masks show "
         "diffuse haze is under-segmented. Mask->box recall vs the authors' one-big-box labels (A: component bbox IoU>=0.4 / B: union bbox / C: localisation hit); the GT masks themselves "
         "reach only 0.887 / 0.946 / 1.000. Negative-frame false alarms are driven by ONE clear-cut flight (DJI_0024: 21/21 frames for every model) -> the map is an explanation "
         "overlay, not a stand-alone alarm.",
         md_table(["config", "checkpoint", "best", "IoU", "Dice", "P", "R", "mIoU pos", "IoU vs SAM (n)", "IoU vs human (n)", "neg-frame FA any / >=1%", "mask->box A / B / C", "params trainable / total"], rows)
         + ("\n\n**threshold sweep (selected rows)**\n\n" + md_table(["config", "thr", "IoU", "Dice", "P", "R", "neg FA any", "neg FA >=1%", "neg FP pixel frac"], thr_rows) if thr_rows else "")
         + ("\n\n**per-epoch heinola VAL IoU**\n\n" + md_table(["config"] + [f"ep{i}" for i in range(1, 11)], ep_rows) if ep_rows else ""),
         sorted(set(srcs)) + ["reports/v5b/SEGMENTATION_V5B.md", "data/v5b/manifests/seg_v5b_manifest.json"])


# ----------------------------------------------------------------------------- T7 latency
def latency_table():
    rows, srcs = [], []
    reg = load_json(ROOT / "models/v5b/MODEL_REGISTRY.json")
    if reg:
        srcs.append("models/v5b/MODEL_REGISTRY.json")
        for m in reg["models"]:
            if m.get("latency_cpu_measured_median_ms") is None and m.get("latency_l40s_batch1_ms") is None:
                continue
            rows.append([m["id"], m["family"][:45], "single model, batch 1 @1280", f3(m.get("latency_cpu_measured_median_ms"), 0), f3(m.get("latency_l40s_batch1_ms"), 1),
                         f"CPU measured by build_model_registry.py (8 threads); L40S: {(m.get('latency_l40s_source') or '-')[:60]}"])
    e = load_json(R5B / "ensemble_dev_v5b.json")
    if e and e.get("latency_table"):
        srcs.append("reports/v5b/ensemble_dev_v5b.json (latency_table, fusion_latency_ms_cpu)")
        lt = e["latency_table"]
        rows.append(["ACCURATE recipe A+B4+B6+B1", "calibrated WBF fusion", "PARALLEL = max(component) + fusion (one GPU per component)", "-", f3(lt["parallel_ms"], 1), lt["note"][:90]])
        rows.append(["ACCURATE recipe A+B4+B6+B1", "calibrated WBF fusion", "SERIAL on one GPU + fusion", "-", f3(lt["serial_ms"], 1), "components back to back"])
        fl = e["fusion_latency_ms_cpu"]
        rows.append(["fusion step only", "numpy WBF + calibration", "per image, devbox CPU", f"{fl['median']} (p90 {fl['p90']}, max {fl['max']})", "-", f"n={fl['n_images']}"])
    for rid in ("fast_sdis", "acc_sdis", "fast_heinola", "acc_heinola"):
        p = ROOT / f"data/v5b/demo/runs/{rid}.json"
        d = load_json(p)
        if not d:
            continue
        srcs.append(rel(p))
        s = d["summary"]
        per = s.get("per_model_latency_ms") or {}
        rows.append([f"demo replay {rid} ({d['mode'].upper()}: {d['model_version'][:50]})", "service /detect, CPU host shared with other agents", f"{s['n_frames']} frames, historical_replay",
                     f"{s['latency_ms']['median']:.0f} median / {s['latency_ms']['max']:.0f} max" + (" (" + ", ".join(f"{k} {v['median']:.0f}" for k, v in per.items()) + ")" if per else ""), "-", "server-reported wall; not a GPU figure"])
    seg_line = grep_first(R5B / "SEGMENTATION_V5B.md", r"CPU inference")
    if seg_line:
        srcs.append("reports/v5b/SEGMENTATION_V5B.md (text, section 0 'CPU inference')")
        m = re.search(r"([\d.]+[–-][\d.]+ s) per (\S+) frame with DINOv3-L at (\S+)", seg_line)
        m2 = re.search(r"([\d.]+) s for the R50", seg_line)
        rows.append(["segV5b_a frozen/LoRA (seg_infer.py)", "DINOv3-L + FPN-lite seg head", f"CPU, {m.group(2) if m else '2048x1080'} frame at {m.group(3) if m else '1280x736'}, 32 threads",
                     (m.group(1) if m else "see text") + " (text figure, not a JSON artifact)", "-", "SEGMENTATION_V5B.md section 0"])
        if m2:
            rows.append(["segV5b_a r50 baseline (seg_infer.py)", "ResNet-50 + FPN-lite", "CPU, same frame", f"{m2.group(1)} s (text figure)", "-", "SEGMENTATION_V5B.md section 0"])
    emit("T7", "Latency: single models (registry), ACCURATE ensemble (parallel / serial), demo replays on CPU, segmentation on CPU",
         "GPU numbers are batch-1 1x L40S fp32 @1280 copied from v3/v4 reports (no new GPU latency was measured in V5b); CPU numbers are devbox measurements under load. "
         "ACCURATE on CPU is 5.8 s/frame (2-model demo) to 12.3 s/frame (4-model recipe) -> ACCURATE is a GPU or offline configuration (DEMO_V5B.md section 5-6).",
         md_table(["system", "architecture", "configuration", "CPU ms", "L40S ms", "source / note"], rows), sorted(set(srcs)) + ["reports/v5b/DEMO_V5B.md"])


# ----------------------------------------------------------------------------- T8 FINAL TEST
def _h3_flag(sysn: str, roster: dict | None):
    if roster and sysn in roster.get("systems", {}):
        return bool(roster["systems"][sysn].get("h3_contaminated")), "roster"
    if H3_CLEAN_RE.search(sysn) and not H3_CONTAMINATED_RE.search(sysn):
        return False, "heuristic"
    if H3_CONTAMINATED_RE.search(sysn):
        return True, "heuristic"
    return None, "unknown"


def test_tables(confs: dict, derived: Path, skip_sweeps=False):
    found = discover_systems(TEST, TEST_SETS)
    roster = load_json(TEST / "ROSTER_V5B.json") if (TEST / "ROSTER_V5B.json").exists() else None
    stamp = (TEST / ".EVALUATED_ONCE").read_text().strip() if (TEST / ".EVALUATED_ONCE").exists() else None
    head = (f"One-shot evaluation ran at {stamp} (reports/v5b/test/.EVALUATED_ONCE)." if stamp else
            "**PENDING** - reports/v5b/test/ does not exist yet; the one-shot script cluster/v5b/run_final_test_eval.sh has not been run. Tables below are rendered from "
            "whatever exists and will fill in on re-run.")
    if not found:
        PENDING.append("T8a-T8e (no reports/v5b/test/eval_*.{json,raw.parquet})")
    # roster
    rrows = []
    for sysn in sorted(found):
        flag, how = _h3_flag(sysn, roster)
        c = confs.get(sysn) or {}
        rrows.append([sysn, ", ".join(sorted(found[sysn])), ("CONTAMINATED" if flag else ("clean" if flag is False else "UNKNOWN")) + f" ({how})",
                      ", ".join(f"{k[2:]}: {f3(v)}" for k, v in (c.get("sdisval") or {}).items()) or "no sdis-val parquet -> n/a",
                      ", ".join(f"{k[2:]}: {f3(v)}" for k, v in (c.get("firemanval") or {}).items()) or "no fireman-val parquet -> n/a"])
    emit("T8a", "FINAL TEST roster: systems found under reports/v5b/test, H3 contamination status, pre-registered VAL confidences",
         head + " Pre-registered confs (PREREGISTRATION_V5B.md 'Thresholds per system') are computed from the VAL parquets by this script "
         "(reports/v5b/tables/_derived/preregistered_confs_v5b.json). H3 contamination: arms B/C/D/D2/D3 trained on ALL Sheffield frames incl. rec_0008 "
         "(LOCKBOX_V5B_FROZEN.md); the heuristic maps pre-registered tokens S2/S3/S4 and arm names B/C/D/D2/D3 to CONTAMINATED - drop a reports/v5b/test/ROSTER_V5B.json "
         "{\"systems\": {token: {\"h3_contaminated\": bool}}} to override. NOTE: LOCKBOX wording 'the ACCURATE ensemble, which contains B' conflates v5b arm B with the v4 DINO "
         "B-series; the shipped recipe (yolo_A + dino_B4/B6/B1) holds no Sheffield frame and is therefore NOT contaminated by its training lists.",
         md_table(["system token", "sets present", "H3 status", "sdis-VAL conf @5/10/20%", "FireMan-VAL conf @5/10%"], rrows) if rrows else "_PENDING_",
         ["reports/v5b/test/eval_<sys>_<set>.{json,raw.parquet}", "reports/v5b/tables/_derived/preregistered_confs_v5b.json", "reports/v5b/LOCKBOX_V5B_FROZEN.md", "reports/v5b/PREREGISTRATION_V5B.md"])

    def recall_rows(st, with_fp_note):
        rows, srcs = [], []
        for sysn in sorted(found):
            p = found[sysn].get(st)
            if p is None:
                rows.append([sysn] + ["PENDING"] * 6)
                continue
            srcs.append(rel(p))
            c = confs.get(sysn) or {}
            want = [("sdis@5%", (c.get("sdisval") or {}).get("fp0.05")), ("sdis@10%", (c.get("sdisval") or {}).get("fp0.10")), ("sdis@20%", (c.get("sdisval") or {}).get("fp0.20")),
                    ("fireman@5%", (c.get("firemanval") or {}).get("fp0.05")), ("fireman@10%", (c.get("firemanval") or {}).get("fp0.10"))]
            cells, n = [], ""
            if p.suffix == ".parquet" and not skip_sweeps:
                ex = recall_at_confs_exact(p, [v for _, v in want if v is not None])
                n = f"{ex['_n'][0]:,} imgs / {ex['_n'][2]} neg" if ex else ""
                for lbl, cv in want:
                    if cv is None or not ex:
                        cells.append("n/a (no VAL conf)")
                        continue
                    s = ex.get(round(cv, 6))
                    img = f"; imgFP {s['neg_image_fp_rate']:.3f}" if (with_fp_note and s and s.get("neg_image_fp_rate") is not None and ex['_n'][2] > 0) else ""
                    cells.append(f"c={cv:.3f} R {s['recall']:.3f} (tiny {f3(s['recall_by_size']['tiny'])}{img})" if s else "n/a")
            else:
                rep = load_json(p) if p.suffix == ".json" else None
                if rep:
                    n = f"{rep['n_images']:,} imgs / {rep['n_negative_images']} neg (interpolated on CONF_GRID)"
                    for lbl, cv in want:
                        if cv is None:
                            cells.append("n/a (no VAL conf)")
                            continue
                        r, t = interp_at_conf(rep["conf_sweep"], cv)
                        cells.append(f"c={cv:.3f} R~{f3(r)} (tiny {f3(t)}) [interp]")
                else:
                    cells = ["n/a"] * 5
            flag, _ = _h3_flag(sysn, roster)
            rows.append([sysn + (" **CONTAMINATED**" if (st == "sheffieldholdout" and flag) else "")] + cells + [n])
        return rows, srcs

    # H1
    rows, srcs = recall_rows("firemanholdout", False)
    emit("T8b", f"FINAL TEST H1 - {TEST_SET_DESC['firemanholdout']}: recall at the pre-registered VAL confidences",
         head + " H1 has 0 explicit negatives (amendment 09:35 UTC) -> RECALL ONLY (image-level + box recall @ IoU>=0.4); no holdout FP claim. Author boxes are interpolated "
         "region boxes -> box IoU metrics are weak; one burn, 640x512 rectified RGB, every 10th frame. Exact recall at each conf is recomputed from the test parquet with the frozen matching code.",
         md_table(["system", "R @ sdis-val 5%", "R @ sdis-val 10%", "R @ sdis-val 20%", "R @ fireman-val 5%", "R @ fireman-val 10%", "denominators"], rows) if rows else "_PENDING_",
         srcs or ["reports/v5b/test/eval_<sys>_firemanholdout.raw.parquet (PENDING)"])
    # H2 matched-FP
    rows, srcs = [], []
    for sysn in sorted(found):
        p = found[sysn].get("sdistest")
        if p is None:
            rows.append([sysn] + ["PENDING"] * 5 + ["-"])
            continue
        srcs.append(rel(p))
        rep = curve_from_parquet(p, grid_for(sysn), derived / "_cache", skip_sweeps) if p.suffix == ".parquet" else load_json(p)
        if not rep:
            rows.append([sysn] + ["n/a"] * 5 + ["-"])
            continue
        mf = matched_fp_from_curve(rep)
        rows.append([sysn] + [f"c={f3(mf[k]['conf'])} R {f3(mf[k]['recall'])} / tiny {f3(mf[k]['tiny_recall'])} / P {f3(mf[k]['precision'])}" if mf[k]["recall"] is not None else "n/a" for k in FP_KEYS]
                    + [f"{rep['n_images']:,} imgs / {rep['n_negative_images']} neg"])
    emit("T8c", f"FINAL TEST H2 - {TEST_SET_DESC['sdistest']}: matched-FP recall / tiny / precision",
         head + " Deployment rule (PREREGISTRATION_V5B.md): replace the FAST default only if a candidate beats expB on H2 R@10% by >= 2 pt AND does not lose on H1 image-level recall "
         "AND passes integration tests. v4 comparison points: reports/v4/tables/T25_*.md (same split, same protocol).",
         md_table(["system", "1% FP", "2% FP", "5% FP", "10% FP", "20% FP", "denominators"], rows) if rows else "_PENDING_",
         srcs or ["reports/v5b/test/eval_<sys>_sdistest.raw.parquet (PENDING)"])
    # H3
    rows, srcs = recall_rows("sheffieldholdout", False)
    emit("T8d", f"FINAL TEST H3 - {TEST_SET_DESC['sheffieldholdout']}",
         head + " Rows marked CONTAMINATED must not enter any comparison or decision (train/test overlap: all 327 frames are in the B/C/D train lists). Even clean rows are "
         "within-incident generalisation (same burn, site, 12-minute session as rec_0003-0007). Author boxes are fragmented sub-boxes -> box recall at IoU 0.4 is convention-limited "
         "(sheffield_uav_CARD.md section 7: 91% of author boxes are >=50% inside a predicted box while IoU>=0.4 recall is 0.085 for expE).",
         md_table(["system", "R @ sdis-val 5%", "R @ sdis-val 10%", "R @ sdis-val 20%", "R @ fireman-val 5%", "R @ fireman-val 10%", "denominators"], rows) if rows else "_PENDING_",
         srcs or ["reports/v5b/test/eval_<sys>_sheffieldholdout.raw.parquet (PENDING)"])
    # H4 historical
    body = []
    allsrc = []
    for st in ("borealtest", "foresttest"):
        rows, srcs = recall_rows(st, True)
        allsrc += srcs
        body.append(f"**{TEST_SET_DESC[st]}**\n\n" + (md_table(["system", "R @ sdis-val 5%", "R @ sdis-val 10%", "R @ sdis-val 20%", "R @ fireman-val 5%", "R @ fireman-val 10%", "denominators"], rows) if rows else "_PENDING_"))
    emit("T8e", "FINAL TEST H4 - historical continuity: Boreal ruokolahti TEST and AuSmoke forest TEST at the pre-registered confidences",
         head + " Reported for continuity with v4 (reports/v4/tables/T26_*.md, T27_*.md). Ruokolahti has 2 negatives (FP unmeasurable); AuSmoke is positives-only and permission-pending -> numbers shown, not used for decisions. "
         "FIgLib-39 causal replay is a separate historical artifact (not produced by this script).",
         "\n\n".join(body), allsrc or ["reports/v5b/test/eval_<sys>_{borealtest,foresttest}.raw.parquet (PENDING)"])


# ----------------------------------------------------------------------------- T9 compute (kraken dumps -> ledger)
URL_RE = re.compile(r"https?://[^\s\"'<>\\]+")
AMZ_RE = re.compile(r"X-Amz-[A-Za-z-]+=[^&\s\"'<>\\]+")
SECRET_KEY_RE = re.compile(r"(?i)(url|token|secret|key|pass|cred|sign|auth)")


def scrub_obj(o):
    if isinstance(o, dict):
        if "name" in o and "value" in o and isinstance(o.get("value"), str) and SECRET_KEY_RE.search(str(o["name"])):
            return {**o, "value": "<scrubbed>"}
        return {k: scrub_obj(v) for k, v in o.items()}
    if isinstance(o, list):
        return [scrub_obj(v) for v in o]
    if isinstance(o, str):
        return AMZ_RE.sub("X-Amz-<scrubbed>", URL_RE.sub("<scrubbed-url>", o))
    return o


def dump_kraken(projects=("green", "obsidian", "obsidian-pu"), requester="tianrgua"):
    """Fresh `kraken jobs list` dumps, scrubbed IN MEMORY before anything is written (presigned URLs live in job env/args)."""
    LEDGER.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc)
    for proj in projects:
        try:
            raw = subprocess.run(["kraken", "-p", proj, "jobs", "list", "--json", "-r", requester, "-n", "2000"], capture_output=True, text=True, timeout=300).stdout
            txt = subprocess.run(["kraken", "-p", proj, "jobs", "list", "-r", requester, "-n", "2000"], capture_output=True, text=True, timeout=300).stdout
        except Exception as e:
            NOTES.append(f"kraken dump for {proj} failed ({e!r}); using the existing dump if any")
            continue
        try:
            d = json.loads(raw[raw.index("{"):]) if "{" in raw else {"results": []}
        except Exception as e:
            NOTES.append(f"kraken {proj} JSON parse failed ({e!r})")
            continue
        clean = scrub_obj(d)
        clean_txt = AMZ_RE.sub("X-Amz-<scrubbed>", URL_RE.sub("<scrubbed-url>", txt))
        ser = json.dumps(clean, indent=1)
        # guard: no scheme-bearing URL and no signature may survive (plain words like curl's %{http_code} are fine)
        assert "X-Amz-Sig" not in ser and not re.search(r"https?://", ser), f"scrub failed for {proj}"
        assert "X-Amz-Sig" not in clean_txt and not re.search(r"https?://", clean_txt), f"scrub failed for {proj} table"
        (LEDGER / f"kraken_{proj}_raw.json").write_text(ser)
        (LEDGER / f"kraken_{proj}_table.txt").write_text(clean_txt)
        (LEDGER / f"kraken_{proj}_dump.meta").write_text(json.dumps({"dumped_utc": stamp.isoformat(timespec="seconds"), "n_results": len(d.get("results", [])), "requester": requester, "scrub": "<scrubbed-url> / X-Amz-<scrubbed> / env values with url|token|secret|key|pass|cred|sign|auth -> <scrubbed>"}))
        print(f"  kraken {proj}: {len(d.get('results', []))} jobs dumped (scrubbed)")
    return stamp


GPUS_PER_NODE = {"g6e.48xlarge": (8, "L40S"), "g6.48xlarge": (8, "L4"), "g6.2xlarge": (1, "L4"), "p6-b200.48xlarge": (8, "B200"),
                 "p5en.48xlarge": (8, "H200"), "p5.48xlarge": (8, "H100"), "p4de.24xlarge": (8, "A100"), "m6i.": (0, "CPU-node"), "m7i.": (0, "CPU-node")}
IT3_START = datetime(2026, 9, 13, 8, 10, 0, tzinfo=timezone.utc)
IT4_START = datetime(2026, 9, 14, 8, 33, 10, tzinfo=timezone.utc)
IT5_START = datetime(2026, 10, 2, 0, 35, 0, tzinfo=timezone.utc)      # cluster/v5/BUDGET_V5.json window_start_utc
IT5B_START = datetime(2026, 10, 2, 7, 26, 24, tzinfo=timezone.utc)    # cluster/v5b/BUDGET_V5B.json window_start_utc
NEVER_ALLOCATED = {"tianrgua-sv4-dgnx"}
EXTERNAL_PREFIXES = ("probe-bindle",)
TZ = {"PDT": timezone(timedelta(hours=-7)), "PST": timezone(timedelta(hours=-8))}


def gpu_info(nodepool):
    for k, v in GPUS_PER_NODE.items():
        if k in (nodepool or ""):
            return v
    return (8, "unknown-8?")


def classify(name, start, created):
    n = name.replace("tianrgua-", "")
    if created is not None and created >= IT5B_START:
        it, how = "5b", "time(creation>=window_start)"
    elif created is not None and created >= IT5_START:
        it, how = "5", "time(creation>=window_start)"
    elif start is not None and start >= IT4_START:
        it, how = "4", "time(window_start)"
    elif start is not None and start >= IT3_START:
        it, how = "3", "time(window_start)"
    elif re.match(r"(vlm|grpo|sft|qwen|diff|synth|fetch|expe|trkd)", n):
        it, how = "2", "name-based(approx)"
    else:
        it, how = "1", "name-based(approx)"
    arm = None
    if re.match(r"v5bactrl", n): track, arm = "yolo-matrix", "A"
    elif re.match(r"v5bbs\d", n): track, arm = "yolo-matrix", "B"
    elif re.match(r"v5bcs\d", n): track, arm = "yolo-matrix", "C"
    elif re.match(r"v5bd3s\d", n): track, arm = "yolo-matrix", "D3"
    elif re.match(r"v5bd2s\d", n): track, arm = "yolo-matrix", "D2"
    elif re.match(r"v5bds\d", n): track, arm = "yolo-matrix", "D"
    elif re.match(r"tch", n): track = "teacher"
    elif re.match(r"sgseg", n): track = "seg"
    elif re.match(r"v5bb200p", n): track = "infra-probe"
    elif re.match(r"v5bdinohr", n): track = "b200-dino"
    elif re.match(r"v5bq", n): track = "b200-qwen"
    elif re.match(r"v5bval|v5bfinal|v5eval|v5val", n): track = "eval"
    elif re.match(r"d3v4", n): track = "dino-adapt"
    elif re.match(r"d3", n): track = "dino-v3"
    elif re.match(r"(tra|extxv4|trkd|trackd)", n): track = "temporal"
    elif re.match(r"v4eval|yoloeval|deb|.*val", n): track = "eval"
    elif re.match(r"v4(a|b|c|e|d)", n): track = "yolo-matrix"
    elif re.match(r"(grpo)", n): track = "grpo"
    elif re.match(r"(vlm|sft)", n): track = "vlm-sft"
    elif re.match(r"(diff|synth)", n): track = "diffusion"
    elif re.match(r"(rfd|det)", n): track = "rfdetr/deimv2"
    elif re.match(r"(probe|gputest|sv|s3|smoke|toy|inittst|diag|fetch)", n): track = "infra-probe"
    else: track = "yolo-train"
    if track == "eval" or re.match(r"(v4eval|yoloeval|deb|fetch|diag|lat)", n) or "eval" in n or n.endswith("val"):
        kind = "evaluation"
    elif track == "teacher":
        kind = "inference(teacher pool)"
    elif track == "infra-probe" or re.match(r"(probe|gputest|sv|s3|toy|inittst|smoke)", n):
        kind = "staging/probe"
    elif "precom" in n or "feats" in n or "extx" in n:
        kind = "feature-extraction"
    else:
        kind = "training"
    return it, track, kind, how, arm


def parse_table(path: Path):
    out = {}
    tre = re.compile(r"(\d{1,2}/\d{1,2})\s+(\d{1,2}:\d{2}[AP]M) (PDT|PST)")
    if not path.exists():
        return out
    for ln in path.read_text().splitlines()[1:]:
        parts = ln.split()
        if not parts or not parts[0].startswith("tianrgua-"):
            continue
        ts = tre.findall(ln)

        def cv(md, hm, tz):
            m, d = md.split("/")
            return datetime.strptime(f"2026-{int(m):02d}-{int(d):02d} {hm}", "%Y-%m-%d %I:%M%p").replace(tzinfo=TZ[tz]).astimezone(timezone.utc)
        out[parts[0]] = (cv(*ts[0]) if ts else None, cv(*ts[1]) if len(ts) > 1 else None)
    return out


def build_ledger(dump_utc: datetime, out_csv: Path):
    rows = []
    for rawf in sorted(glob.glob(str(LEDGER / "kraken_*_raw.json"))):
        proj = Path(rawf).stem.replace("kraken_", "").replace("_raw", "")
        tbl = parse_table(Path(rawf.replace("_raw.json", "_table.txt")))
        for j in json.load(open(rawf))["results"]:
            name = j["jobName"]; md = j["metadata"]; status = md["status"]
            created = datetime.fromisoformat(md["creationTime"].replace("Z", "+00:00"))
            start, end = tbl.get(name, (None, None))
            start_src = "table" if start else "creation(no start in table)"
            start = start or created
            if end is None and md.get("completionTime"):
                end = datetime.fromisoformat(md["completionTime"].replace("Z", "+00:00"))
            running = status in ("RUNNING", "RESTARTING", "STOPPING")
            never = status in ("PENDING", "QUEUED", "CREATED") or name in NEVER_ALLOCATED
            if never:
                end = start
            elif end is None:
                end = dump_utc if running else start
            nodes = (j.get("resourceConfig") or {}).get("nodeCount") or 1
            gpn, gtype = gpu_info(j.get("nodePool") or (j.get("resourceConfig") or {}).get("queue") or "")
            node_h = max(0.0, (end - start).total_seconds() / 3600.0)
            it, track, kind, how, arm = classify(name, start, created)
            external = name.startswith(EXTERNAL_PREFIXES)
            if external:
                it, track, kind, how = "0", "external(other project)", "external", "name-prefix"
            outcome = f"{status}(open at dump)" if running else (f"{status}(never allocated)" if never else status)
            if status == "FAILED" and kind == "training":
                kind = "training(failed/retry)"
            rows.append(dict(job_id=j["jobId"], job_name=name, project=proj, iteration=it, track=track, arm=arm or "", kind=kind, gpu_type=gtype, gpus_per_node=gpn,
                             nodes=nodes, gpus=gpn * nodes, created_utc=created.isoformat(timespec="seconds"), alloc_start_utc=start.isoformat(timespec="seconds"),
                             alloc_end_utc=end.isoformat(timespec="seconds"), queue_wait_min=round((start - created).total_seconds() / 60, 1),
                             node_hours=round(node_h, 3), gpu_hours=round(node_h * gpn * nodes, 2), outcome=outcome, start_source=start_src,
                             utilization="n/a (no pod telemetry)", attribution_method=how, in_scope=(not external),
                             record_type="exact(scheduler)" if not running else "open-at-dump"))
    rows.sort(key=lambda r: r["alloc_start_utc"])
    if rows:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def compute_tables(do_dump: bool):
    dump_utc = None
    if do_dump:
        dump_utc = dump_kraken()
    metas = sorted(LEDGER.glob("kraken_*_dump.meta"))
    if dump_utc is None:
        if metas:
            dump_utc = max(datetime.fromisoformat(json.loads(m.read_text())["dumped_utc"]) for m in metas)
        else:
            dump_utc = datetime.now(timezone.utc)
            NOTES.append("T9: no dump metadata; open jobs stamped with generation time")
    out_csv = R5B / "COMPUTE_LEDGER_V5B.csv"
    rows_all = build_ledger(dump_utc, out_csv)
    if not rows_all:
        PENDING.append("T9 (no kraken dumps under reports/v5b/ledger)")
        emit("T9a", "Compute", "no ledger rows", "_PENDING_", ["reports/v5b/ledger/"])
        return
    rows = [r for r in rows_all if r["in_scope"]]
    budget = load_json(ROOT / "cluster/v5b/BUDGET_V5B.json") or {}
    snap = dump_utc.isoformat(timespec="seconds")
    # by iteration x gpu type
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0])
    for r in rows:
        k = (r["iteration"], r["gpu_type"]); agg[k][0] += 1; agg[k][1] += r["node_hours"]; agg[k][2] += r["gpu_hours"]; agg[k][3] += r["record_type"] == "open-at-dump"
    order = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "5b": 6}
    trows = [[it, g, n, f"{nh:.2f}", f"{gh:.1f}", op] for (it, g), (n, nh, gh, op) in sorted(agg.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1]))]
    emit("T9a", "Compute by iteration and GPU type (scheduler-derived allocated hours; whole-node accounting)",
         f"Allocated node-hours = scheduler completion - start from the kraken text table (queue wait excluded); jobs RUNNING at the dump are truncated at {snap}. "
         f"GPU-hours = node-hours x GPUs/node from the nodepool (g6e = 8x L40S, p6-b200 = 8x B200). Iteration 5b = jobs CREATED >= {IT5B_START.isoformat()} "
         f"(cluster/v5b/BUDGET_V5B.json window_start); iteration 5 = created >= {IT5_START.isoformat()}; 3/4 by allocation start as in v4; 1 vs 2 by job-name pattern. "
         f"{len(rows_all)} job records across projects {sorted({r['project'] for r in rows_all})}; {len(rows_all) - len(rows)} external (other project) rows excluded. "
         "Presigned URLs/tokens were scrubbed from the dumps in memory before writing (<scrubbed-url>). No pod utilization telemetry exists.",
         md_table(["iteration", "GPU type", "jobs", "node-h", "GPU-h", "open at dump"], trows),
         ["reports/v5b/COMPUTE_LEDGER_V5B.csv", "reports/v5b/ledger/kraken_{green,obsidian,obsidian-pu}_{raw.json,table.txt,dump.meta}", "cluster/v5b/BUDGET_V5B.json", "cluster/v5/BUDGET_V5.json"])
    v5b = [r for r in rows if r["iteration"] == "5b"]
    by = lambda key: defaultdict(lambda: [0, 0.0, 0.0, 0.0])  # noqa: E731
    bt, bk, ba = by("t"), by("k"), by("a")
    for r in v5b:
        for dct, key in ((bt, r["track"]), (bk, r["kind"]), (ba, r["arm"] or "-")):
            d = dct[(key, r["gpu_type"])]; d[0] += 1; d[1] += r["node_hours"]; d[2] += r["gpu_hours"]
            if r["outcome"].startswith("FAILED") or r["kind"] == "training(failed/retry)":
                d[3] += r["node_hours"]
    nh = {g: sum(r["node_hours"] for r in v5b if r["gpu_type"] == g) for g in sorted({r["gpu_type"] for r in v5b})}
    gh = {g: sum(r["gpu_hours"] for r in v5b if r["gpu_type"] == g) for g in nh}
    cap_nh, cap_gh = budget.get("max_allocation_node_hours"), budget.get("max_allocation_gpu_hours")
    reg = load_json(ROOT / "cluster/v5b/registry.json") or {"jobs": {}}
    regh = defaultdict(float)
    for k, v in reg["jobs"].items():
        r0 = datetime.fromisoformat(v["reserved_utc"].replace("Z", "+00:00"))
        r1 = datetime.fromisoformat(v["released_utc"].replace("Z", "+00:00")) if v.get("released_utc") else dump_utc
        regh[v["gpu_type"]] += (r1 - r0).total_seconds() / 3600 * v["nodes"]
    emit("T9b", "Iteration 5b compute by GPU type, track, kind and arm (scheduler allocation) with the reservation-registry cross-check",
         f"{len(v5b)} iteration-5b jobs at the snapshot ({snap}; {sum(1 for r in v5b if r['record_type'] == 'open-at-dump')} still open -> truncated). "
         + "; ".join(f"{g}: {nh[g]:.2f} node-h = {gh[g]:.1f} GPU-h" for g in nh) + f". Window cap: {cap_nh} node-h / {cap_gh} GPU-h total "
         f"(max concurrent {budget.get('max_nodes_by_type')}). Registry (cluster/v5b/registry.json, reserve->release wall-clock, includes queue wait): "
         + ", ".join(f"{g} {h:.2f} node-h" for g, h in sorted(regh.items())) + ". The two differ by queue wait / pod start and by segments whose reservation outlives the job.",
         "**by track**\n\n" + md_table(["track", "GPU", "jobs", "node-h", "GPU-h", "failed node-h"], [[k[0], k[1], v[0], f"{v[1]:.2f}", f"{v[2]:.1f}", f"{v[3]:.2f}"] for k, v in sorted(bt.items(), key=lambda kv: -kv[1][1])])
         + "\n\n**by kind**\n\n" + md_table(["kind", "GPU", "jobs", "node-h", "GPU-h", "failed node-h"], [[k[0], k[1], v[0], f"{v[1]:.2f}", f"{v[2]:.1f}", f"{v[3]:.2f}"] for k, v in sorted(bk.items(), key=lambda kv: -kv[1][1])])
         + "\n\n**by arm (yolo-matrix only)**\n\n" + md_table(["arm", "GPU", "jobs", "node-h", "GPU-h", "failed node-h"], [[k[0], k[1], v[0], f"{v[1]:.2f}", f"{v[2]:.1f}", f"{v[3]:.2f}"] for k, v in sorted(ba.items()) if k[0] != "-"]),
         ["reports/v5b/COMPUTE_LEDGER_V5B.csv", "cluster/v5b/registry.json", "cluster/v5b/BUDGET_V5B.json"])
    inc = [r for r in v5b if not r["outcome"].startswith("SUCCEEDED")]
    emit("T9c", "Iteration 5b jobs that did not end SUCCEEDED (FAILED segments, STOPPED probes, RUNNING at dump)",
         "FAILED yolo-matrix segments are the seg-1 presigned-upload-expiry incident (STATUS 10:40 UTC): training resumed from /scratch last.pt, so their node-hours are real training time, not lost work.",
         md_table(["job", "track", "arm", "kind", "GPU", "outcome", "created", "alloc start", "alloc end", "node-h"],
                  [[r["job_name"], r["track"], r["arm"], r["kind"], r["gpu_type"], r["outcome"], r["created_utc"][5:16], r["alloc_start_utc"][5:16], r["alloc_end_utc"][5:16], f"{r['node_hours']:.2f}"] for r in inc]) if inc else "_none_",
         ["reports/v5b/COMPUTE_LEDGER_V5B.csv"])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R5B / "tables"))
    ap.add_argument("--no-dump", action="store_true", help="do not re-run kraken; reuse reports/v5b/ledger dumps")
    ap.add_argument("--skip-sweeps", action="store_true", help="only use cached matched-FP curves (no new sweep_report runs)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    derived = out / "_derived"; derived.mkdir(exist_ok=True)

    data_sources_table()
    corpus_tables()
    registry_tables()
    confs, _ = val_tables(derived, a.skip_sweeps)
    ensemble_table()
    seg_tables()
    latency_table()
    test_tables(confs, derived, a.skip_sweeps)
    compute_tables(not a.no_dump)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    written = []
    for tid, text in TABLES.items():
        title = text.splitlines()[0][len(f"### {tid}. "):]
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower())[:60].strip("_")
        p = out / f"{tid}_{slug}.md"
        p.write_text(f"_generated {stamp} by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_\n\n" + text)
        written.append(p)
    keep = {p.name for p in written}
    for stale in out.glob("T[0-9]*_*.md"):
        if stale.name not in keep:
            stale.unlink()
    idx = "\n".join(f"- [{tid}](#{tid.lower()}) {TABLES[tid].splitlines()[0][len(tid) + 6:]}" for tid in TABLES)
    miss = "\n".join(f"- `{m}`" for m in sorted(set(MISSING))) or "- none"
    pend = "\n".join(f"- {m}" for m in PENDING) or "- none"
    notes = "\n".join(f"- {n}" for n in dict.fromkeys(NOTES)) or "- none"
    (out / "ALL_TABLES.md").write_text(
        f"# Iteration 5b - all report tables\n\n_generated {stamp} by `scripts/v5b/make_report_tables_v5b.py`; every number is read from the artifact(s) listed under each table. "
        f"Re-run after new eval artifacts land (reports/v5b/val, reports/v5b/test) or to refresh the compute ledger._\n\n## Index\n\n{idx}\n\n## PENDING\n\n{pend}\n\n"
        f"## Artifacts not found at generation time\n\n{miss}\n\n## Notes\n\n{notes}\n\n---\n\n" + "\n\n---\n\n".join(TABLES.values()))
    print(f"{len(written)} tables -> {rel(out)}/ ; combined -> {rel(out / 'ALL_TABLES.md')}")
    for p in written:
        print("  ", p.name)
    if PENDING:
        print("PENDING:", *PENDING, sep="\n  ")
    if MISSING:
        print("missing artifacts:", *sorted(set(MISSING)), sep="\n  ")
    if NOTES:
        print("notes:", *dict.fromkeys(NOTES), sep="\n  ")


if __name__ == "__main__":
    main()
