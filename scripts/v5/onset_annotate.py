#!/usr/bin/env python
"""onset_annotate.py — human interval annotation of smoke ONSET in an ordered frame sequence.

PROTOCOL_V5 §3: early warning is measured from FIRST CLEAR SMOKE, with the
annotated interval [last_clear_no_smoke, first_clear_smoke]. The three cases
the user insists on keeping apart are explicit here:
  * already smoke at start           -> onset_class = already_smoke_at_start
  * smoke seen only after a camera turn -> onset_class = visible_after_camera_turn
  * genuine first visible smoke        -> onset_class = true_first_visible
plus "no smoke observed" and "onset not observed (other)" for clips that never
show a transition. Model scores are ASSISTIVE ONLY and OFF by default: the tool
renders and records without any detector output (required for final-test
labelling), and the annotation JSON records whether scores/labels were shown.

Subcommands
  render    build data/v5/onset/<event_id>/{strip.jpg, sheet_NNN.jpg, frames_index.csv,
            render_meta.json[, scores.png]} from --frames-dir | --seq-json+--seq-id | --list
  annotate  write data/v5/onset/<event_id>/annotation.json (validated)
  summary   table of all annotations under --out (acceptance: >=10 independent
            true_first_visible events with review_status=human)

Examples
  python scripts/v5/onset_annotate.py render --event-id ev1 --frames-dir data/v4/sequences/boreal_frames/evo_1
  python scripts/v5/onset_annotate.py render --event-id ev2 --seq-json data/v3/sequences/sequences_val.json \
        --seq-id val_sdis-07_marguerite-282_20240313T161635 [--preds yoloA=reports/v4/eval_v4a_s2_sdisval.raw.parquet]
  python scripts/v5/onset_annotate.py annotate --event-id ev2 --last-clear-no-smoke-idx 18 \
        --first-clear-smoke-idx 20 --annotator "J. Doe" --review-status human --notes "..."
  python scripts/v5/onset_annotate.py annotate --event-id ev1 --starts-with-smoke --annotator ... --review-status human
  python scripts/v5/onset_annotate.py annotate --event-id evX --no-smoke-observed --annotator ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_common as rc  # noqa: E402

ONSET_CLASSES = ["true_first_visible", "visible_after_camera_turn", "already_smoke_at_start",
                 "no_smoke_observed", "onset_not_observed_other"]


# --------------------------------------------------------------------------
def build_source(a) -> tuple[rc.FrameSource, dict, list]:
    """Returns (FrameSource, meta, labels_or_None_list)."""
    labels = None
    if a.frames_dir:
        src = rc.FrameSource.from_dir(a.frames_dir, cadence_s=a.cadence_s)
        meta = dict(input_kind="frames_dir", sequence_ref=str(Path(a.frames_dir).resolve()))
    elif a.seq_json:
        if not a.seq_id:
            raise SystemExit("--seq-json needs --seq-id")
        seqs = json.load(open(a.seq_json))
        hit = [s for s in seqs if s["seq_id"] == a.seq_id]
        if not hit:
            raise SystemExit(f"seq_id {a.seq_id} not in {a.seq_json}")
        s = hit[0]
        if s.get("split", "").lower() == "test":
            raise SystemExit("refusing to render a TEST-split sequence outside the final-test procedure")
        paths = [f["image_path"] for f in s["frames"]]
        ts = [f["ts"] for f in s["frames"]]
        src = rc.FrameSource.from_paths(paths, ref=f"{a.seq_json}::{a.seq_id}", ts_override=ts)
        labels = [bool(f.get("smoke")) for f in s["frames"]]
        meta = dict(input_kind="seq_json", sequence_ref=f"{a.seq_json}::{a.seq_id}", site=s.get("site"),
                    camera=s.get("camera"), split=s.get("split"), label_pattern=s.get("label_pattern"))
    elif a.list:
        src = rc.open_source(a.list, cadence_s=a.cadence_s)
        meta = dict(input_kind="frame_list", sequence_ref=str(Path(a.list).resolve()))
    elif a.video:
        src = rc.FrameSource.from_video(a.video, sample_s=a.sample_s)
        meta = dict(input_kind="video", sequence_ref=str(Path(a.video).resolve()), fps=src.fps)
    else:
        raise SystemExit("need one of --frames-dir / --seq-json --seq-id / --list / --video")
    if not src.frames:
        raise SystemExit("no frames")
    return src, meta, labels


def cmd_render(a):
    out = Path(a.out) / a.event_id
    if out.exists() and (out / "annotation.json").exists() and not a.force:
        raise SystemExit(f"{out} already has annotation.json — use --force to re-render (annotation kept)")
    out.mkdir(parents=True, exist_ok=True)
    src, meta, labels = build_source(a)
    n = len(src.frames)
    # ---- assistive scores (optional) ----
    models = []
    scores = {}
    for spec in a.preds or []:
        name, path = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
        pm = rc.load_preds(path)
        col = []
        miss = 0
        for f in src.frames:
            s = rc.lookup_score(pm, f["path"]) if src.kind != "video" else None
            if s is None:
                miss += 1
            col.append(s)
        scores[name] = col
        models.append(dict(name=name, path=path, n_missing=miss))
    show_labels = bool(a.show_labels and labels is not None)
    # ---- frames_index.csv ----
    cols = ["idx", "rel_s", "rel_label", "abs_utc", "name", "path", "video_frame"]
    if show_labels:
        cols.append("dataset_label_smoke")
    for m in models:
        cols.append(f"score_{m['name']}")
    rows = []
    for i, f in enumerate(src.frames):
        r = dict(idx=i, rel_s=round(f["rel_s"], 3), rel_label=rc.fmt_rel(f["rel_s"]), abs_utc=rc.fmt_abs(f["abs_epoch"]),
                 name=f["name"], path=f["path"], video_frame="" if f["video_frame"] is None else f["video_frame"])
        if show_labels:
            r["dataset_label_smoke"] = int(labels[i])
        for m in models:
            v = scores[m["name"]][i]
            r[f"score_{m['name']}"] = "" if v is None else round(v, 4)
        rows.append(r)
    rc.write_csv(out / "frames_index.csv", rows, cols)
    # ---- thumbnails (single decode pass); --crop zooms into a normalized region ----
    thumbs = {}
    crop = None
    if a.crop:
        crop = [float(v) for v in a.crop.split(",")]
        if len(crop) != 4 or not all(0.0 <= v <= 1.0 for v in crop) or crop[0] >= crop[2] or crop[1] >= crop[3]:
            raise SystemExit("--crop must be x0,y0,x1,y1 normalized in [0,1] with x0<x1, y0<y1")
    for f, im in src.iter_images(max_side=None if crop else a.thumb_w):
        if crop:
            W, H = im.size
            im = im.crop((int(crop[0] * W), int(crop[1] * H), max(int(crop[0] * W) + 1, int(crop[2] * W)),
                          max(int(crop[1] * H) + 1, int(crop[3] * H))))
        thumbs[f["idx"]] = rc.fit_width(im, a.thumb_w)

    def tile(i, size=13):
        f = src.frames[i]
        l1 = f"#{i:04d} {rc.fmt_rel(f['rel_s'])}"
        extra = []
        if show_labels:
            extra.append("L+" if labels[i] else "L-")
        for m in models:
            v = scores[m["name"]][i]
            extra.append(f"{m['name'][:6]}={'na' if v is None else f'{v:.2f}'}")
        l2 = rc.fmt_abs(f["abs_epoch"]) or f["name"][:40]
        lines = [l1, l2] + ([" ".join(extra)] if extra else [])
        return rc.burn_text(thumbs[i], lines, size=size)

    # keyframe strip: K evenly spaced incl. first and last
    K = min(a.keyframes, n)
    if K == n:
        kidx = list(range(n))
    else:
        kidx = sorted(set(int(round(j * (n - 1) / (K - 1))) for j in range(K)))
    strip = rc.contact_sheet([tile(i) for i in kidx], cols=min(len(kidx), a.strip_cols), thumb_w=a.thumb_w,
                             title=f"{a.event_id}  keyframes {len(kidx)}/{n}  cadence~{src.observed()['cadence_s']}s  {'CROP ' + a.crop + '  ' if crop else ''}"
                                   f"{'SCORES SHOWN (assistive)' if models else 'NO MODEL SCORES'}  "
                                   f"{'labels shown' if show_labels else 'labels hidden'}")
    strip.save(out / "strip.jpg", quality=88, optimize=True)
    # all-frames sheets
    per = a.sheet_cols * a.sheet_rows
    n_sheets = (n + per - 1) // per
    for k in range(n_sheets):
        ids = list(range(k * per, min(n, (k + 1) * per)))
        sh = rc.contact_sheet([tile(i) for i in ids], cols=a.sheet_cols, thumb_w=a.thumb_w,
                              title=f"{a.event_id}  all frames sheet {k + 1}/{n_sheets}  (#{ids[0]:04d}-#{ids[-1]:04d})")
        sh.save(out / f"sheet_{k:03d}.jpg", quality=86, optimize=True)
    # score curve
    if models:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(8, min(24, n * 0.25)), 3.2), dpi=110)
        xs = list(range(n))
        for m in models:
            ys = [v if v is not None else float("nan") for v in scores[m["name"]]]
            ax.plot(xs, ys, marker="o", ms=3, lw=1.2, label=m["name"])
        if show_labels:
            for i, lb in enumerate(labels):
                if lb:
                    ax.axvspan(i - 0.5, i + 0.5, color="#f0c080", alpha=0.25, lw=0)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_xlabel("frame idx (relative time below)")
        ax.set_ylabel("max box conf")
        ticks = kidx
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{i}\n{rc.fmt_rel(src.frames[i]['rel_s'])[2:]}" for i in ticks], fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title(f"{a.event_id} — ASSISTIVE model scores (not ground truth)" + ("; shaded = dataset label smoke" if show_labels else ""), fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "scores.png")
        plt.close(fig)
    rmeta = dict(event_id=a.event_id, n_frames=n, observed=src.observed(), scores_shown=bool(models), score_models=models,
                 labels_shown=show_labels, keyframes=kidx, n_sheets=n_sheets, thumb_w=a.thumb_w, crop_xyxy_norm=crop,
                 first_frame_sha256_1MB=rc.sha256_file(src.frames[0]["path"], 1 << 20) if src.kind != "video" else rc.sha256_file(src.ref, 1 << 20),
                 rendered_utc=rc.utc_now(), **meta)
    rc.write_json(out / "render_meta.json", rmeta)
    print(f"[render] {a.event_id}: {n} frames, cadence~{rmeta['observed']['cadence_s']}s, strip {len(kidx)} keyframes, "
          f"{n_sheets} sheet(s), scores_shown={bool(models)} labels_shown={show_labels} -> {out}")


# --------------------------------------------------------------------------
def cmd_annotate(a):
    d = Path(a.out) / a.event_id
    fi = d / "frames_index.csv"
    rm = d / "render_meta.json"
    if not fi.exists() or not rm.exists():
        raise SystemExit(f"render {a.event_id} first ({d})")
    if (d / "annotation.json").exists() and not a.force:
        raise SystemExit("annotation.json exists — use --force to overwrite")
    frames = rc.read_csv(fi)
    meta = json.load(open(rm))
    n = len(frames)

    def ts(i):
        if i is None:
            return None
        f = frames[i]
        return dict(idx=i, rel_s=float(f["rel_s"]), abs_utc=f["abs_utc"] or None, name=f["name"])

    last_i = a.last_clear_no_smoke_idx
    first_i = a.first_clear_smoke_idx
    for v, nm in ((last_i, "last-clear-no-smoke-idx"), (first_i, "first-clear-smoke-idx")):
        if v is not None and not (0 <= v < n):
            raise SystemExit(f"--{nm} {v} out of range [0,{n - 1}]")
    modes = sum(bool(x) for x in (a.starts_with_smoke, a.no_smoke_observed, a.onset_not_observed))
    if modes > 1:
        raise SystemExit("choose at most one of --starts-with-smoke / --no-smoke-observed / --onset-not-observed")
    if a.starts_with_smoke:
        if last_i is not None:
            raise SystemExit("--starts-with-smoke is incompatible with --last-clear-no-smoke-idx")
        first_i = 0 if first_i is None else first_i
        onset_class, observed = "already_smoke_at_start", False
    elif a.no_smoke_observed:
        if last_i is not None or first_i is not None:
            raise SystemExit("--no-smoke-observed takes no indices")
        onset_class, observed = "no_smoke_observed", False
    elif a.onset_not_observed:
        if first_i is not None:
            raise SystemExit("--onset-not-observed: do not give --first-clear-smoke-idx (use the normal mode instead)")
        if not a.notes:
            raise SystemExit("--onset-not-observed needs --notes explaining why (gap, clip ends, occlusion, ...)")
        onset_class, observed = "onset_not_observed_other", False
    else:
        if last_i is None or first_i is None:
            raise SystemExit("normal mode needs BOTH --last-clear-no-smoke-idx and --first-clear-smoke-idx "
                             "(or one of --starts-with-smoke / --no-smoke-observed / --onset-not-observed)")
        if not last_i < first_i:
            raise SystemExit("last_clear_no_smoke_idx must be < first_clear_smoke_idx")
        if a.camera_turn_before_smoke:
            onset_class, observed = "visible_after_camera_turn", False
        else:
            onset_class, observed = "true_first_visible", True
    if a.camera_turn_before_smoke and onset_class not in ("visible_after_camera_turn", "already_smoke_at_start"):
        raise SystemExit("--camera-turn-before-smoke only makes sense with an observed first-smoke frame")
    unc = None
    bracket_s = None
    if last_i is not None and first_i is not None:
        inner = list(range(last_i + 1, first_i))
        bracket_s = round(float(frames[first_i]["rel_s"]) - float(frames[last_i]["rel_s"]), 3)
        unc = dict(start_idx=inner[0] if inner else None, end_idx=inner[-1] if inner else None, n_frames=len(inner),
                   duration_s=bracket_s, note="frames strictly between last clear no-smoke and first clear smoke; "
                   "the bracket [last_clear, first_clear] is the §3 interval")
    ann = dict(event_id=a.event_id, sequence_ref=meta.get("sequence_ref"), input_kind=meta.get("input_kind"),
               site=meta.get("site"), camera=meta.get("camera"), split=meta.get("split"), n_frames=n,
               cadence_s=meta["observed"]["cadence_s"], valid_observed_s=meta["observed"]["valid_s"],
               onset_class=onset_class, onset_observed=observed, starts_with_smoke=bool(a.starts_with_smoke),
               camera_turn_before_smoke=bool(a.camera_turn_before_smoke),
               last_clear_no_smoke_idx=last_i, last_clear_no_smoke_ts=ts(last_i),
               first_clear_smoke_idx=first_i, first_clear_smoke_ts=ts(first_i),
               uncertain_span=unc, bracket_s=bracket_s,
               usable_for_early_warning_eval=(onset_class == "true_first_visible"),
               scores_shown_during_annotation=bool(meta.get("scores_shown")), score_models=[m["name"] for m in meta.get("score_models", [])],
               dataset_labels_shown=bool(meta.get("labels_shown")),
               annotator=a.annotator, review_status=a.review_status, notes=a.notes or "",
               annotated_utc=rc.utc_now(), render_meta_sha_first_frame=meta.get("first_frame_sha256_1MB"))
    rc.write_json(d / "annotation.json", ann)
    print(json.dumps(ann, indent=2))


def cmd_summary(a):
    root = Path(a.out)
    rows = []
    for p in sorted(root.glob("*/annotation.json")):
        r = json.load(open(p))
        rows.append(r)
    from collections import Counter
    cls = Counter(r["onset_class"] for r in rows)
    human_tfv = [r for r in rows if r["onset_class"] == "true_first_visible" and r["review_status"] == "human"]
    indep = set((r.get("site"), r.get("camera"), (r.get("first_clear_smoke_ts") or {}).get("abs_utc", "")[:10]) for r in human_tfv)
    lines = [f"# Onset annotations under {root} ({len(rows)} events)", "",
             "| event_id | class | observed | last_clear idx/t | first_clear idx/t | bracket s | cam turn | scores shown | status | annotator |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lc = r["last_clear_no_smoke_ts"]
        fc = r["first_clear_smoke_ts"]
        lines.append(f"| {r['event_id']} | {r['onset_class']} | {r['onset_observed']} | "
                     f"{'' if lc is None else f'{lc[chr(105)+chr(100)+chr(120)]}/{rc.fmt_rel(lc[chr(114)+chr(101)+chr(108)+chr(95)+chr(115)])}'} | "
                     f"{'' if fc is None else f'{fc[chr(105)+chr(100)+chr(120)]}/{rc.fmt_rel(fc[chr(114)+chr(101)+chr(108)+chr(95)+chr(115)])}'} | "
                     f"{'' if r['bracket_s'] is None else r['bracket_s']} | {r['camera_turn_before_smoke']} | "
                     f"{r['scores_shown_during_annotation']} | {r['review_status']} | {r['annotator'][:30]} |")
    lines += ["", f"classes: {dict(cls)}",
              f"human-reviewed true_first_visible events: {len(human_tfv)} (independent site/camera/day: {len(indep)}) — "
              f"PROTOCOL_V5 §6 acceptance (>=10 independent, human-reviewed): {'MET' if len(indep) >= 10 else 'NOT MET'}",
              f"agent_provisional annotations are NOT counted toward acceptance."]
    text = "\n".join(lines)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("render")
    r.add_argument("--event-id", required=True)
    r.add_argument("--out", default="data/v5/onset")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--frames-dir")
    g.add_argument("--seq-json")
    g.add_argument("--list")
    g.add_argument("--video")
    r.add_argument("--seq-id")
    r.add_argument("--cadence-s", type=float)
    r.add_argument("--sample-s", type=float, default=1.0)
    r.add_argument("--preds", action="append", help="name=path (parquet/jsonl). ASSISTIVE ONLY; omit for blind labelling")
    r.add_argument("--show-labels", action="store_true", help="show dataset frame labels (seq-json only). Off by default")
    r.add_argument("--keyframes", type=int, default=12)
    r.add_argument("--strip-cols", type=int, default=6)
    r.add_argument("--sheet-cols", type=int, default=6)
    r.add_argument("--sheet-rows", type=int, default=6)
    r.add_argument("--thumb-w", type=int, default=320)
    r.add_argument("--crop", help="x0,y0,x1,y1 normalized: zoom every tile into this region (tiny plumes); recorded in render_meta")
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_render)
    an = sp.add_parser("annotate")
    an.add_argument("--event-id", required=True)
    an.add_argument("--out", default="data/v5/onset")
    an.add_argument("--last-clear-no-smoke-idx", type=int)
    an.add_argument("--first-clear-smoke-idx", type=int)
    an.add_argument("--starts-with-smoke", action="store_true", help="smoke already visible in frame 0 (onset NOT observed)")
    an.add_argument("--no-smoke-observed", action="store_true", help="no smoke anywhere in the sequence")
    an.add_argument("--onset-not-observed", action="store_true", help="smoke never becomes clearly visible / gap / clip ends (explain in --notes)")
    an.add_argument("--camera-turn-before-smoke", action="store_true", help="smoke first seen right after a pan/turn — NOT a true onset")
    an.add_argument("--annotator", required=True)
    an.add_argument("--review-status", required=True, choices=["human", "agent_provisional"])
    an.add_argument("--notes", default="")
    an.add_argument("--force", action="store_true")
    an.set_defaults(fn=cmd_annotate)
    s = sp.add_parser("summary")
    s.add_argument("--out", default="data/v5/onset")
    s.add_argument("--md")
    s.set_defaults(fn=cmd_summary)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
