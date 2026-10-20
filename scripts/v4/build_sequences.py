#!/usr/bin/env python3
"""
build_sequences.py — iteration-4 DATA PREP: chronological sequences for the
causal temporal head (Track D-style), from the new v4 sources. CPU only,
deterministic, resumable. Never touches data/v3/ or existing v4 manifests.

Sources -> sequences (a sequence = chronologically ordered frames of one
video/camera, cut at cadence gaps where a cadence is derivable):

  ausmoke    multinatsmoke AuSmoke subset (29 videos, per-frame smoke MASKS).
             Frame order = filename numbering `<unit>_<fidx:06d>-<+off:06d>.jpg`
             (fidx strictly monotone per unit; fidx-off constant per unit, so
             `off` is a per-unit frame-index offset in the ORIGINAL video
             timeline — original fps is NOT recorded, so `off` is an ORDER
             unit, not seconds). Gap cut: diff(fidx) > 20 x per-unit median
             stride (mirrors v3's 600s cut at 29s cadence ~ 20.7x).
             Boxes = mask -> connected-component tight boxes via the frozen
             converter in scripts/v4/build_corpus.py (mask_to_lines).
             Split: data/v4/manifests/forest_holdout_v4.json — train units ->
             train, val units -> val, TEST UNITS NEVER LOADED.

  boreal     Boreal-Forest-Fire 30-s 4K clips, video-level labels only.
             Frames extracted at 1 fps (cv2) into
             data/v4/sequences/boreal_frames/<clip_id>/ for the train+val
             burns of data/v4/manifests/aerial_split_v4.json only
             (train: evo+karkkila, val: heinola; ruokolahti=test NEVER read).
             One sequence per clip, ts_or_order = seconds.

  firesense  1-fps frames already on disk under
             data/v4/raw/firesense/frames/{smoke_pos,smoke_neg}/<video>/.
             Video-level labels only. One sequence per video, ts = t-index
             seconds. No unit split exists for firesense; policy (documented
             in the report): smoke_neg -> val (negative replay footage;
             CAVEAT: these same frames are background negatives inside the
             v4 YOLO train corpora), smoke_pos -> train (weak video-level
             positives, not box-supervised).

Outputs (all under data/v4/sequences/):
  sequences_v4.parquet   one row per frame
  ONSET_AUDIT.md         onset + label-consistency audit
  replay_val_v4.jsonl    VAL alert-replay package
  REPORT.md              counts / trust assessment
  boreal_frames/         extracted boreal frames
  audit_v4.json          machine-readable stats

Usage:  .venv/bin/python scripts/v4/build_sequences.py all
        (or: boreal-extract | build | audit | replay | report)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/local/home/tianrgua/cvproject")
RAW_MNS = REPO / "data/v4/raw/multinatsmoke"
RAW_BOREAL = REPO / "data/v4/raw/boreal"
RAW_FS = REPO / "data/v4/raw/firesense"
MAN = REPO / "data/v4/manifests"
OUT = REPO / "data/v4/sequences"
BOREAL_FRAMES = OUT / "boreal_frames"
V3_AUDIT = REPO / "data/v3/sequences/audit.json"

GAP_FACTOR = 20          # cut when diff(fidx) > GAP_FACTOR * median stride
MIN_SEQ_LEN = 3          # as in v3
JPEG_QUALITY = 92
BOREAL_TRAIN_BURNS = ("evo", "karkkila")
BOREAL_VAL_BURNS = ("heinola",)
N_WORKERS = 32

# ------------------------------------------------------------------ helpers


def load_mask_converter():
    """Import mask_to_lines from the frozen corpus builder (single source of
    truth for mask->box)."""
    spec = importlib.util.spec_from_file_location(
        "build_corpus", REPO / "scripts/v4/build_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.mask_to_lines


def yolo_lines_to_xyxyn(lines: list[str]) -> list[list[float]]:
    out = []
    for ln in lines:
        _, cx, cy, w, h = ln.split()
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        out.append([round(max(0.0, cx - w / 2), 6), round(max(0.0, cy - h / 2), 6),
                    round(min(1.0, cx + w / 2), 6), round(min(1.0, cy + h / 2), 6)])
    return out


def _convert_one(job):
    """Worker: (i, mask_abs, img_abs) -> (i, xyxyn boxes)."""
    i, mask_abs, img_abs = job
    lines = _convert_one.fn(mask_abs, img_abs)
    return i, yolo_lines_to_xyxyn(lines)


def _init_worker():
    _convert_one.fn = load_mask_converter()


# ------------------------------------------------------------------ ausmoke


def ausmoke_frames() -> pd.DataFrame:
    hold = json.loads((MAN / "forest_holdout_v4.json").read_text())
    df = pd.read_parquet(MAN / "multinatsmoke_manifest.parquet")
    au = df[df.subset == "AuSmoke"].copy()
    ext = au.rel_path.str.split("/").str[-1].str.extract(
        r"_(\d{6})-([+-]\d{6})\.jpg$")
    assert ext[0].notna().all(), "AuSmoke filename pattern mismatch"
    au["fidx"] = pd.to_numeric(ext[0])
    au["off"] = pd.to_numeric(ext[1])
    split_of = {u: "train" for u in hold["train"]}
    split_of.update({u: "val" for u in hold["val"]})
    test_units = set(hold["test"])
    au = au[~au.parent_unit.isin(test_units)]           # never load test units
    au = au[au.parent_unit.isin(split_of)].copy()
    au["split"] = au.parent_unit.map(split_of)
    # verify monotone order in 3 units (order = filename numbering)
    for u in sorted(au.parent_unit.unique())[:3]:
        f = np.sort(au.loc[au.parent_unit == u, "fidx"].values)
        assert (np.diff(f) > 0).all(), f"non-monotone fidx in {u}"
        assert au.loc[au.parent_unit == u, "fidx"].is_unique
    return au.sort_values(["parent_unit", "fidx"]).reset_index(drop=True)


def build_ausmoke_rows() -> tuple[list[dict], dict]:
    au = ausmoke_frames()
    conv = load_mask_converter()  # noqa: F841 (validated importable)
    # mask -> boxes in parallel, deterministic collection by index
    jobs = [(i, str(RAW_MNS / r.mask_rel_path), str(RAW_MNS / r.rel_path))
            for i, r in enumerate(au.itertuples())]
    boxes_by_i: dict[int, list] = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_init_worker) as ex:
        for i, boxes in ex.map(_convert_one, jobs, chunksize=64):
            boxes_by_i[i] = boxes

    rows, seq_meta = [], []
    n_cut, n_drop_short = 0, 0
    for unit, g in au.groupby("parent_unit", sort=True):
        g = g.sort_values("fidx")
        fidx = g.fidx.values
        med = float(np.median(np.diff(fidx))) if len(fidx) > 1 else 1.0
        thr = GAP_FACTOR * max(med, 1.0)
        # segment boundaries
        segs, start = [], 0
        for k in range(1, len(fidx)):
            if fidx[k] - fidx[k - 1] > thr:
                segs.append((start, k))
                start = k
                n_cut += 1
        segs.append((start, len(fidx)))
        seg_no = 0
        for s, e in segs:
            if e - s < MIN_SEQ_LEN:
                n_drop_short += 1
                continue
            seq_id = f"ausmoke_{unit}_{seg_no:03d}"
            seg_no += 1
            sub = g.iloc[s:e]
            for j, r in enumerate(sub.itertuples()):
                b = boxes_by_i[r.Index]
                rows.append(dict(
                    seq_id=seq_id, source="ausmoke", parent_unit=unit,
                    frame_idx=j, ts_or_order=int(r.off),
                    abs_path=str(RAW_MNS / r.rel_path),
                    label_kind="box", has_box_label=len(b) > 0,
                    n_boxes=len(b),
                    boxes=json.dumps(b) if b else None,
                    split=r.split, seq_label="smoke",
                    mask_frac=float(r.mask_fg_frac)))
            seq_meta.append(dict(seq_id=seq_id, unit=unit,
                                 split=sub.split.iloc[0],
                                 n=e - s, median_stride=med))
    stats = dict(n_units=int(au.parent_unit.nunique()),
                 n_frames=len(rows), n_sequences=len(seq_meta),
                 n_gap_cuts=n_cut, n_dropped_short=n_drop_short,
                 gap_rule=f"diff(fidx) > {GAP_FACTOR} x median stride",
                 seq_meta=seq_meta)
    return rows, stats


# ------------------------------------------------------------------- boreal


def boreal_clips() -> pd.DataFrame:
    df = pd.read_parquet(MAN / "boreal_manifest.parquet")
    vl = df[df.label_type == "videolabel"].copy()
    vl["burn"] = vl.parent_unit.str.replace("burn_", "", regex=False)
    keep = vl.burn.isin(BOREAL_TRAIN_BURNS + BOREAL_VAL_BURNS)
    vl = vl[keep].copy()                                   # ruokolahti never read
    vl["clip_id"] = vl.rel_path.str.split("/").str[-1].str.replace(
        ".mp4", "", regex=False)
    vl["split"] = np.where(vl.burn.isin(BOREAL_VAL_BURNS), "val", "train")
    return vl.sort_values("clip_id").reset_index(drop=True)


def _extract_clip(job):
    """Worker: extract 1-fps frames from one clip. Deterministic, resumable."""
    import cv2
    cv2.setNumThreads(2)
    src, clip_id = job
    out_dir = BOREAL_FRAMES / clip_id
    done = out_dir / ".done.json"
    if done.exists():
        d = json.loads(done.read_text())
        if all((out_dir / f).exists() for f in d["files"]):
            return clip_id, d["files"], d["fps"], True
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps and fps > 1, f"bad fps for {src}"
    targets, t = [], 0
    n_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while round(t * fps) < max(n_est, 1):
        targets.append((t, int(round(t * fps))))
        t += 1
    files, k, ti = [], 0, 0
    ok = True
    while ti < len(targets) and ok:
        ok = cap.grab()
        if not ok:
            break
        if k == targets[ti][1]:
            ok2, frame = cap.retrieve()
            if not ok2:
                break
            name = f"{clip_id}_t{targets[ti][0]:04d}.jpg"
            cv2.imwrite(str(out_dir / name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            files.append(name)
            ti += 1
        k += 1
    cap.release()
    done.write_text(json.dumps(dict(files=files, fps=fps)))
    return clip_id, files, fps, False


def cmd_boreal_extract() -> dict:
    clips = boreal_clips()
    jobs = [(str(RAW_BOREAL / r.rel_path), r.clip_id)
            for r in clips.itertuples()]
    per_clip = {}
    n_new = 0
    with ProcessPoolExecutor(max_workers=16) as ex:
        for clip_id, files, fps, cached in ex.map(_extract_clip, jobs):
            per_clip[clip_id] = dict(n_frames=len(files), fps=fps)
            n_new += 0 if cached else 1
    print(f"boreal: {len(per_clip)} clips, "
          f"{sum(v['n_frames'] for v in per_clip.values())} frames "
          f"({n_new} newly extracted)")
    return per_clip


def build_boreal_rows() -> tuple[list[dict], dict]:
    clips = boreal_clips()
    rows, seq_meta = [], []
    for r in clips.itertuples():
        out_dir = BOREAL_FRAMES / r.clip_id
        done = out_dir / ".done.json"
        assert done.exists(), f"run boreal-extract first ({r.clip_id})"
        files = json.loads(done.read_text())["files"]
        seq_id = f"boreal_{r.clip_id}"
        seq_label = "smoke" if r.has_smoke_label else "no_smoke"
        for j, name in enumerate(files):
            rows.append(dict(
                seq_id=seq_id, source="boreal", parent_unit=r.parent_unit,
                frame_idx=j, ts_or_order=j,
                abs_path=str(out_dir / name),
                label_kind="videolabel", has_box_label=False, n_boxes=0,
                boxes=None, split=r.split, seq_label=seq_label,
                mask_frac=None))
        seq_meta.append(dict(seq_id=seq_id, unit=r.parent_unit,
                             split=r.split, n=len(files),
                             seq_label=seq_label))
    stats = dict(n_clips=len(seq_meta), n_frames=len(rows),
                 n_sequences=len(seq_meta), seq_meta=seq_meta)
    return rows, stats


# ---------------------------------------------------------------- firesense


def firesense_frames() -> pd.DataFrame:
    df = pd.read_parquet(MAN / "firesense_manifest.parquet")
    fr = df[df.rel_path.str.startswith("frames/", na=False)].copy()
    fr["cat"] = fr.rel_path.str.split("/").str[1]
    fr = fr[fr.cat.isin(["smoke_pos", "smoke_neg"])].copy()
    fr["video"] = fr.rel_path.str.split("/").str[2]
    fr["t"] = fr.rel_path.str.extract(r"_t(\d+)\.jpg$")[0].astype(int)
    fr["split"] = np.where(fr.cat == "smoke_neg", "val", "train")
    return fr.sort_values(["video", "t"]).reset_index(drop=True)


def build_firesense_rows() -> tuple[list[dict], dict]:
    fr = firesense_frames()
    rows, seq_meta = [], []
    for video, g in fr.groupby("video", sort=True):
        g = g.sort_values("t")
        seq_id = f"firesense_{video}"
        cat = g.cat.iloc[0]
        seq_label = "smoke" if cat == "smoke_pos" else "no_smoke"
        for j, r in enumerate(g.itertuples()):
            rows.append(dict(
                seq_id=seq_id, source="firesense", parent_unit=r.parent_unit,
                frame_idx=j, ts_or_order=int(r.t),
                abs_path=str(RAW_FS / r.rel_path),
                label_kind="videolabel", has_box_label=False, n_boxes=0,
                boxes=None, split=r.split, seq_label=seq_label,
                mask_frac=None))
        seq_meta.append(dict(seq_id=seq_id, unit=g.parent_unit.iloc[0],
                             split=g.split.iloc[0], n=len(g),
                             seq_label=seq_label, cat=cat))
    stats = dict(n_videos=len(seq_meta), n_frames=len(rows),
                 n_sequences=len(seq_meta), seq_meta=seq_meta)
    return rows, stats


# ------------------------------------------------------------------- build


COLS = ["seq_id", "source", "parent_unit", "frame_idx", "ts_or_order",
        "abs_path", "label_kind", "has_box_label", "n_boxes", "boxes",
        "split", "seq_label", "mask_frac"]


def cmd_build():
    OUT.mkdir(parents=True, exist_ok=True)
    a_rows, a_stats = build_ausmoke_rows()
    b_rows, b_stats = build_boreal_rows()
    f_rows, f_stats = build_firesense_rows()
    df = pd.DataFrame(a_rows + b_rows + f_rows, columns=COLS)
    # spot-check paths exist (deterministic sample: every 500th row)
    missing = [p for p in df.abs_path.iloc[::500] if not Path(p).exists()]
    assert not missing, f"missing files e.g. {missing[:3]}"
    df.to_parquet(OUT / "sequences_v4.parquet", index=False)
    stats = dict(ausmoke=a_stats, boreal=b_stats, firesense=f_stats)
    (OUT / "audit_v4.json").write_text(json.dumps(stats, indent=1))
    print(df.groupby(["source", "split"]).agg(
        frames=("seq_id", "size"), seqs=("seq_id", "nunique")).to_string())
    return df, stats


# ------------------------------------------------------------------- audit


def seq_arrays(df: pd.DataFrame):
    for seq_id, g in df.groupby("seq_id", sort=True):
        g = g.sort_values("frame_idx")
        yield seq_id, g


def cmd_audit(df: pd.DataFrame, stats: dict):
    v3 = json.loads(V3_AUDIT.read_text()) if V3_AUDIT.exists() else None
    au = df[df.source == "ausmoke"]

    # ---- onset (AuSmoke: frame-level truth = non-empty smoke annotation.
    # In this corpus every shipped frame has a non-empty mask, so "annotated
    # positive" is evaluated from mask_frac > 0 to stay honest.)
    onset = {"ausmoke": {"train": dict(total=0, observed_onset=0,
                                       positive_from_start=0, ambiguous=0,
                                       negative_only=0),
                         "val": dict(total=0, observed_onset=0,
                                     positive_from_start=0, ambiguous=0,
                                     negative_only=0)}}
    onset_rows = []
    for seq_id, g in seq_arrays(au):
        pos = (g.mask_frac.values > 0)
        sp = g.split.iloc[0]
        d = onset["ausmoke"][sp]
        d["total"] += 1
        if pos[0]:
            d["positive_from_start"] += 1
            cat, onset_i = "positive_from_start", None
        elif pos.any():
            d["observed_onset"] += 1
            cat, onset_i = "observed_onset", int(np.argmax(pos))
        else:
            d["negative_only"] += 1
            cat, onset_i = "negative_only", None
        onset_rows.append(dict(seq_id=seq_id, split=sp, category=cat,
                               onset_frame_idx=onset_i,
                               onset_observed=cat == "observed_onset"))
    # boreal / firesense: video-level only -> never a frame onset
    vid_onset = {}
    for src in ("boreal", "firesense"):
        s = df[df.source == src]
        per = {}
        for sp in ("train", "val"):
            gg = s[s.split == sp]
            seqs = gg.groupby("seq_id").seq_label.first()
            per[sp] = dict(total=int(len(seqs)),
                           videolabel_positive_ambiguous=int((seqs == "smoke").sum()),
                           negative=int((seqs == "no_smoke").sum()),
                           observed_onset=0)
        vid_onset[src] = per

    # ---- label consistency (AuSmoke)
    n_pairs = n_flips_box = n_flips_ann = n_area_jump = 0
    seq_flicker_box, seq_area_jump = 0, 0
    per_seq = []
    for seq_id, g in seq_arrays(au):
        hb = g.has_box_label.values.astype(int)
        ann = (g.mask_frac.values > 0).astype(int)
        fr = g.mask_frac.values
        flips_b = int(np.abs(np.diff(hb)).sum())
        flips_a = int(np.abs(np.diff(ann)).sum())
        both = (fr[1:] > 0) & (fr[:-1] > 0)
        ratio = np.abs(np.log2(np.where(both, fr[1:] / np.maximum(fr[:-1], 1e-12), 1.0)))
        jumps = int((ratio > 1.0).sum())          # >2x area change frame-to-frame
        n_pairs += len(hb) - 1
        n_flips_box += flips_b
        n_flips_ann += flips_a
        n_area_jump += jumps
        seq_flicker_box += flips_b > 0
        seq_area_jump += jumps > 0
        per_seq.append(dict(seq_id=seq_id, split=g.split.iloc[0],
                            n=len(hb), box_flips=flips_b, ann_flips=flips_a,
                            area_jumps_gt2x=jumps))
    n_seqs_au = len(per_seq)
    cons = dict(
        adjacent_pairs=n_pairs,
        annotation_presence_flips=n_flips_ann,
        annotation_flip_rate=round(n_flips_ann / max(n_pairs, 1), 4),
        derived_box_presence_flips=n_flips_box,
        derived_box_flip_rate=round(n_flips_box / max(n_pairs, 1), 4),
        seq_with_derived_box_flicker=seq_flicker_box,
        seq_with_derived_box_flicker_pct=round(100 * seq_flicker_box / max(n_seqs_au, 1), 1),
        area_jumps_gt2x=n_area_jump,
        area_jump_rate=round(n_area_jump / max(n_pairs, 1), 4),
        seq_with_area_jump=seq_area_jump,
        seq_with_area_jump_pct=round(100 * seq_area_jump / max(n_seqs_au, 1), 1),
    )
    v3_cmp = None
    if v3:
        v3_cmp = {}
        for sp in ("train", "val"):
            lp = v3[sp]["label_patterns"]
            n = v3[sp]["n_sequences"]
            v3_cmp[sp] = dict(
                n_sequences=n, flicker=lp.get("flicker", 0),
                flicker_pct=round(100 * lp.get("flicker", 0) / n, 1))

    audit = dict(onset_ausmoke=onset["ausmoke"], onset_videolabel=vid_onset,
                 consistency_ausmoke=cons, v3_pyro_sdis_flicker=v3_cmp,
                 onset_rows=onset_rows, per_seq_consistency=per_seq)
    full = json.loads((OUT / "audit_v4.json").read_text())
    full["audit"] = audit
    (OUT / "audit_v4.json").write_text(json.dumps(full, indent=1))

    # ---- ONSET_AUDIT.md
    oa = onset["ausmoke"]
    # units whose first shipped frame carries offset 0 (dataset trimmed at onset)
    au_units = {}
    for u, g in au.groupby("parent_unit"):
        au_units[u] = int(g.ts_or_order.min())
    n_off0 = sum(1 for v in au_units.values() if v == 0)
    md = []
    md.append("# ONSET + LABEL-CONSISTENCY AUDIT — v4 sequences (iteration 4)\n")
    md.append("Generated deterministically by `scripts/v4/build_sequences.py audit`. "
              "Frame-level truth exists ONLY for AuSmoke (per-frame masks). "
              "Boreal and FIRESENSE are video-level labels: no frame onsets are "
              "invented for them.\n")
    md.append("## AuSmoke (multinatsmoke, mask-labelled)\n")
    md.append("| split | sequences | observed onset (bg->smoke) | positive from start "
              "(onset NOT observed) | negative only | ambiguous |")
    md.append("|---|---|---|---|---|---|")
    for sp in ("train", "val"):
        d = oa[sp]
        md.append(f"| {sp} | {d['total']} | {d['observed_onset']} | "
                  f"{d['positive_from_start']} | {d['negative_only']} | "
                  f"{d['ambiguous']} |")
    tot = {k: oa["train"][k] + oa["val"][k] for k in oa["train"]}
    md.append(f"| **total** | **{tot['total']}** | **{tot['observed_onset']}** | "
              f"**{tot['positive_from_start']}** | **{tot['negative_only']}** | "
              f"**{tot['ambiguous']}** |\n")
    md.append(f"**Sequences genuinely spanning a background->smoke transition: "
              f"{tot['observed_onset']} of {tot['total']}.**\n")
    md.append("Why: the AuSmoke corpus ships ONLY annotated smoke frames — all "
              f"{len(au)} kept frames have a non-empty mask (mask_fg_px > 0). "
              "The filename convention `<unit>_<fidx>-<+offset>` is FIgLib-like: "
              "offset >= 0 everywhere and offset==0 exists in "
              f"{n_off0}/{len(au_units)} kept units, i.e. the shipped segments "
              "start at (or after) first visible smoke. Pre-onset background "
              "frames were not shipped, so NO AuSmoke sequence observes an "
              "onset. Candidate 'first annotated frame' is always frame 0 -> "
              "flagged `onset_observed=false` for every sequence.\n")
    md.append("## Boreal (30-s clips, video-level labels only)\n")
    md.append("| split | clips | videolabel positive (frame onset UNKNOWN) | "
              "non-smoke | observed frame onset |")
    md.append("|---|---|---|---|---|")
    for sp in ("train", "val"):
        d = vid_onset["boreal"][sp]
        md.append(f"| {sp} | {d['total']} | {d['videolabel_positive_ambiguous']} | "
                  f"{d['negative']} | 0 (no frame labels — not invented) |")
    md.append("")
    md.append("## FIRESENSE (video-level labels only)\n")
    md.append("| split | videos | videolabel positive (frame onset UNKNOWN) | "
              "non-smoke | observed frame onset |")
    md.append("|---|---|---|---|---|")
    for sp in ("train", "val"):
        d = vid_onset["firesense"][sp]
        md.append(f"| {sp} | {d['total']} | {d['videolabel_positive_ambiguous']} | "
                  f"{d['negative']} | 0 (no frame labels — not invented) |")
    md.append("")
    md.append("## AuSmoke label consistency\n")
    md.append(f"- Adjacent frame pairs: {cons['adjacent_pairs']}")
    md.append(f"- Annotation-presence flips (mask nonempty <-> empty): "
              f"{cons['annotation_presence_flips']} "
              f"({100*cons['annotation_flip_rate']:.1f}% of pairs) — 0 by "
              "construction: the corpus contains no annotated-negative frames.")
    md.append(f"- Derived-box-presence flips (converter min-area filter): "
              f"{cons['derived_box_presence_flips']} "
              f"({100*cons['derived_box_flip_rate']:.2f}% of pairs); sequences "
              f"affected: {cons['seq_with_derived_box_flicker']} "
              f"({cons['seq_with_derived_box_flicker_pct']}%).")
    md.append(f"- Mask-area discontinuities (>2x frame-to-frame area change): "
              f"{cons['area_jumps_gt2x']} pairs "
              f"({100*cons['area_jump_rate']:.1f}%); sequences affected: "
              f"{cons['seq_with_area_jump']} ({cons['seq_with_area_jump_pct']}%).")
    if v3_cmp:
        md.append(f"\n**v3 pyro-sdis comparison**: v3 sequences with label "
                  f"flicker: train {v3_cmp['train']['flicker']}/"
                  f"{v3_cmp['train']['n_sequences']} "
                  f"({v3_cmp['train']['flicker_pct']}%), val "
                  f"{v3_cmp['val']['flicker']}/{v3_cmp['val']['n_sequences']} "
                  f"({v3_cmp['val']['flicker_pct']}%). AuSmoke's 0% presence "
                  "flicker is NOT better labelling — it is a positive-only "
                  "corpus, so presence flicker is structurally impossible. The "
                  "comparable instability signal here is mask-AREA "
                  "discontinuity, which affects "
                  f"{cons['seq_with_area_jump_pct']}% of sequences.\n")
    (OUT / "ONSET_AUDIT.md").write_text("\n".join(md) + "\n")
    print(f"wrote {OUT/'ONSET_AUDIT.md'}")
    return audit


# ------------------------------------------------------------------- replay


def cmd_replay(df: pd.DataFrame):
    """VAL alert-replay package: AuSmoke val sequences, ALL boreal heinola
    clips, boreal non-smoke clips from train burns (negative footage), and
    firesense smoke_neg videos (negative footage, train-contaminated)."""
    lines = []
    corp_note = ("frames used as background negatives in v4 YOLO train "
                 "corpora (e.g. CORPUS_B_G_v1: 292 firesense background_neg "
                 "lines) — false-alert numbers are OPTIMISTIC for models "
                 "trained on those corpora")

    def frames_payload(g, with_truth):
        out = []
        for r in g.sort_values("frame_idx").itertuples():
            fr = dict(abs_path=r.abs_path, ts_or_order=int(r.ts_or_order))
            if with_truth:
                fr["has_box"] = bool(r.has_box_label)
                fr["n_boxes"] = int(r.n_boxes)
                fr["boxes"] = json.loads(r.boxes) if r.boxes else []
            out.append(fr)
        return out

    # 1) AuSmoke val sequences (per-frame truth)
    au = df[(df.source == "ausmoke") & (df.split == "val")]
    for seq_id, g in seq_arrays(au):
        lines.append(dict(
            seq_id=seq_id, source="ausmoke",
            parent_unit=g.parent_unit.iloc[0], split="val",
            role="positive", seq_label="smoke", truth_level="frame_boxes",
            n_frames=len(g),
            notes=["positive from frame 0 — NO pre-onset segment exists; "
                   "contributes nothing to the negative-footage pool"],
            frames=frames_payload(g, True)))
    # 2) boreal heinola (val burn): all clips
    bo = df[df.source == "boreal"]
    for seq_id, g in seq_arrays(bo[bo.split == "val"]):
        lab = g.seq_label.iloc[0]
        lines.append(dict(
            seq_id=seq_id, source="boreal",
            parent_unit=g.parent_unit.iloc[0], split="val",
            role="positive" if lab == "smoke" else "negative_footage",
            seq_label=lab, truth_level="videolabel", n_frames=len(g),
            notes=[], frames=frames_payload(g, False)))
    # 3) boreal non-smoke clips from TRAIN burns -> negative footage
    for seq_id, g in seq_arrays(bo[(bo.split == "train")
                                   & (bo.seq_label == "no_smoke")]):
        lines.append(dict(
            seq_id=seq_id, source="boreal",
            parent_unit=g.parent_unit.iloc[0], split="train",
            role="negative_footage", seq_label="no_smoke",
            truth_level="videolabel", n_frames=len(g),
            notes=["clip from a TRAIN burn site (evo/karkkila) — background "
                   "seen at train time; weaker negative evidence than val-site "
                   "footage"],
            frames=frames_payload(g, False)))
    # 4) firesense smoke_neg videos -> negative footage
    fs = df[(df.source == "firesense") & (df.seq_label == "no_smoke")]
    for seq_id, g in seq_arrays(fs):
        lines.append(dict(
            seq_id=seq_id, source="firesense",
            parent_unit=g.parent_unit.iloc[0], split="val",
            role="negative_footage", seq_label="no_smoke",
            truth_level="videolabel", n_frames=len(g),
            notes=[corp_note], frames=frames_payload(g, False)))

    lines.sort(key=lambda d: (d["source"], d["seq_id"]))
    with open(OUT / "replay_val_v4.jsonl", "w") as f:
        for d in lines:
            f.write(json.dumps(d) + "\n")
    n_neg = sum(1 for d in lines if d["role"] == "negative_footage")
    print(f"wrote {OUT/'replay_val_v4.jsonl'}: {len(lines)} sequences "
          f"({n_neg} negative-footage)")
    return lines


# ------------------------------------------------------------------- report


def cmd_report(df: pd.DataFrame, audit: dict, replay_lines: list):
    g = df.groupby(["source", "split"]).agg(
        sequences=("seq_id", "nunique"), frames=("seq_id", "size"))
    md = []
    md.append("# v4 sequences — data-prep report (iteration 4, DATA PREP track)\n")
    md.append("Deliverables in this directory: `sequences_v4.parquet`, "
              "`ONSET_AUDIT.md`, `replay_val_v4.jsonl`, `boreal_frames/`, "
              "`audit_v4.json`. Built by `scripts/v4/build_sequences.py all` "
              "(CPU-only, deterministic; v3 data and v4 manifests untouched).\n")
    md.append("## Counts by source and split\n")
    md.append("| source | split | sequences | frames | wall-clock |")
    md.append("|---|---|---|---|---|")
    for (src, sp), r in g.iterrows():
        if src == "ausmoke":
            wall = "unknown (order units; original fps not recorded)"
        else:
            wall = f"{r.frames/3600:.2f} h @1fps"
        md.append(f"| {src} | {sp} | {r.sequences} | {r.frames} | {wall} |")
    md.append(f"| **total** |  | **{df.seq_id.nunique()}** | **{len(df)}** |  |\n")
    md.append("Splits: AuSmoke per `forest_holdout_v4.json` (test units never "
              "loaded); boreal per `aerial_split_v4.json` burn-site split "
              "(train=evo+karkkila, val=heinola, ruokolahti=test never read); "
              "FIRESENSE has no unit split — policy: smoke_neg->val (replay "
              "negative footage), smoke_pos->train (weak video-level "
              "positives).\n")
    oa = audit["onset_ausmoke"]
    cons = audit["consistency_ausmoke"]
    tot = {k: oa["train"][k] + oa["val"][k] for k in oa["train"]}
    md.append("## Onset (see ONSET_AUDIT.md)\n")
    md.append(f"- AuSmoke: **0 of {tot['total']} sequences observe a "
              "background->smoke onset** — every shipped frame is annotated "
              "positive; filename offsets (>=0, 0 = first annotated frame) "
              "show segments were trimmed at/after first visible smoke.")
    md.append("- Boreal + FIRESENSE positives: video-level labels only; frame "
              "onsets deliberately NOT invented.")
    md.append("- Consequence: **v4 adds NO observed-onset training signal**; "
              "onset supervision still comes only from v3 pyro-sdis sequences "
              "(train seqs with 'appears' pattern) and FIgLib-style replay "
              "eval.\n")
    md.append("## Label consistency (AuSmoke)\n")
    md.append(f"- Annotation-presence flicker: 0% (structurally — positive-only "
              "corpus), vs v3 pyro-sdis sequence flicker 29% train / 36% val. "
              "NOT comparable as label quality.")
    md.append(f"- Derived-box-presence flips (mask->box min-area filter): "
              f"{100*cons['derived_box_flip_rate']:.2f}% of adjacent pairs, "
              f"{cons['seq_with_derived_box_flicker_pct']}% of sequences.")
    md.append(f"- Mask-area discontinuities >2x: "
              f"{100*cons['area_jump_rate']:.1f}% of adjacent pairs, "
              f"{cons['seq_with_area_jump_pct']}% of sequences — the honest "
              "instability metric for this source.\n")
    n_negfoot = sum(1 for d in replay_lines if d["role"] == "negative_footage")
    n_negframes = sum(d["n_frames"] for d in replay_lines
                      if d["role"] == "negative_footage")
    md.append("## VAL alert-replay package (replay_val_v4.jsonl)\n")
    md.append(f"- {len(replay_lines)} sequences; negative-footage pool: "
              f"{n_negfoot} sequences / {n_negframes} frames "
              f"(~{n_negframes/3600:.2f} h @1fps).")
    md.append("- CONTAMINATION FLAGS carried per line: firesense smoke_neg "
              "frames are background negatives inside v4 YOLO train corpora "
              "(CORPUS_B_G_v1 et al.) -> optimistic false-alert numbers for "
              "models trained on them; the single karkkila non-smoke clip is "
              "from a TRAIN burn site.")
    md.append("- AuSmoke val sequences contribute NO pre-onset negative "
              "segments (none exist).\n")
    md.append("## Trust assessment\n")
    md.append("**Usable for supervised temporal training (frame-level):** "
              "AuSmoke train sequences (box labels from masks) — but "
              "positive-only: they teach smoke persistence/appearance, NOT the "
              "background->smoke transition; negatives must come from v3 "
              "pyro-sdis negatives / firesense / boreal no-smoke clips. "
              "aus_20240209_SVF_Arbo_SmokeMachine is synthetic-source smoke "
              "(train-only per holdout notes, never eval).")
    md.append("**Weak/sequence-level only:** boreal clips and firesense videos "
              "(videolabel) — usable for sequence-level supervision or "
              "alert-replay eval, NOT for frame-supervised training.")
    md.append("**Export-for-review:** AuSmoke frames with derived-box flips or "
              ">2x area jumps (listed in audit_v4.json per_seq_consistency); "
              "boreal positive clips if frame-level use is ever wanted "
              "(would need manual onset annotation).")
    md.append("**Val caveats:** AuSmoke val = 4 units but only ~3 independent "
              "events (axis fixed-cam pair shares one camera); boreal val = 1 "
              "burn (heinola) — treat aerial val numbers as anecdotal "
              "(tiny-N warning in aerial_split_v4.json).\n")
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    print(f"wrote {OUT/'REPORT.md'}")


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["boreal-extract", "build", "audit",
                                    "replay", "report", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd in ("boreal-extract", "all"):
        cmd_boreal_extract()
    if args.cmd in ("build", "all"):
        df, stats = cmd_build()
    elif args.cmd in ("audit", "replay", "report"):
        df = pd.read_parquet(OUT / "sequences_v4.parquet")
        stats = json.loads((OUT / "audit_v4.json").read_text())
    if args.cmd in ("audit", "all"):
        audit = cmd_audit(df, stats)
    if args.cmd in ("replay", "all"):
        replay_lines = cmd_replay(df)
    if args.cmd in ("report", "all"):
        full = json.loads((OUT / "audit_v4.json").read_text())
        audit = full["audit"]
        if args.cmd == "report":
            replay_lines = [json.loads(l) for l in
                            open(OUT / "replay_val_v4.jsonl")]
        cmd_report(df, audit, replay_lines)


if __name__ == "__main__":
    main()
