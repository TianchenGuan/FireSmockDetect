#!/usr/bin/env python
"""negative_review.py — human no-smoke verification of candidate negative footage.

PROTOCOL_V5 §6: negatives count only when a HUMAN has verified no smoke;
detector silence is NOT evidence. This tool makes the human step cheap:

  sheets     build contact sheets (default 4 cols x 6 rows, timestamps burned
             in) for every unit in a manifest, write per-unit frame index,
             units.csv (metadata + valid observed seconds) and a verdicts.csv
             TEMPLATE that the reviewer fills by editing (verdict column).
  summarise  hours verified no-smoke per deployment scenario, counting ONLY
             verdict == no_smoke (uncertain / smoke / blank are excluded),
             independent-unit counts and the §6 acceptance check.

Manifest (CSV or JSON list) columns:
  unit*            unique id (used as directory name)
  path*            directory of frames | video file | .txt list of frame paths
  scenario*        fixed_camera | aerial | fixed_camera_like | other
  source           dataset name (boreal, firesense, ...)
  independent_unit grouping key for "independent units" (burn/site/flight/date)
  license          licence or permission status (e.g. CC-BY-4.0, PENDING)
  cadence_s        seconds between frames for frame dirs without parseable ts
  sample_s         video sampling period in seconds (default --sample-s)
  contamination    free text: e.g. "in v4 train corpora"
  notes            free text
  (* required)

Verdict CSV columns (fixed):
  unit, reviewer, verdict{no_smoke,smoke,uncertain}, first_smoke_frame_or_blank,
  notes, reviewed_utc

Usage:
  python scripts/v5/negative_review.py sheets --manifest M.csv --out data/v5/review
  python scripts/v5/negative_review.py summarise --review-dir data/v5/review [--verdicts F.csv] [--md out.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_common as rc  # noqa: E402

VERDICT_COLS = ["unit", "reviewer", "verdict", "first_smoke_frame_or_blank", "notes", "reviewed_utc"]
VALID_VERDICTS = {"no_smoke", "smoke", "uncertain"}
UNIT_COLS = ["unit", "scenario", "source", "independent_unit", "license", "input_kind", "path",
             "n_frames", "cadence_s", "ts_kind", "valid_observed_s", "span_s", "n_gaps_excluded",
             "excluded_s", "n_sheets", "contamination", "notes", "first_frame_sha256_1MB"]
FRAME_COLS = ["unit", "idx", "sheet", "row", "col", "rel_s", "rel_label", "abs_utc", "name", "path",
              "video_frame"]


def load_manifest(path: str):
    if path.endswith(".json"):
        rows = json.load(open(path))
    else:
        rows = rc.read_csv(path)
    for r in rows:
        for k in ("unit", "path", "scenario"):
            if not r.get(k):
                raise SystemExit(f"manifest row missing {k}: {r}")
    units = [r["unit"] for r in rows]
    if len(set(units)) != len(units):
        raise SystemExit("duplicate unit ids in manifest")
    return sorted(rows, key=lambda r: rc.natural_key(r["unit"]))


def build_unit(row: dict, out_dir: Path, cols: int, rows_per_sheet: int, thumb_w: int,
               sample_s: float, font_size: int, overwrite: bool) -> dict:
    unit = row["unit"]
    udir = out_dir / unit
    udir.mkdir(parents=True, exist_ok=True)
    cadence = float(row["cadence_s"]) if row.get("cadence_s") else None
    s_s = float(row["sample_s"]) if row.get("sample_s") else sample_s
    src = rc.open_source(row["path"], cadence_s=cadence, sample_s=s_s)
    if not src.frames:
        raise SystemExit(f"{unit}: no frames found at {row['path']}")
    per_sheet = cols * rows_per_sheet
    n_sheets = (len(src.frames) + per_sheet - 1) // per_sheet
    obs = src.observed()
    frame_rows = []
    tiles = []
    sheet_i = 0
    existing = all((udir / f"sheet_{k:03d}.jpg").exists() for k in range(n_sheets))
    render = overwrite or not existing
    for f, im in (src.iter_images(max_side=thumb_w) if render else ((f, None) for f in src.frames)):
        i = f["idx"]
        k, within = divmod(i, per_sheet)
        r_, c_ = divmod(within, cols)
        abs_utc = rc.fmt_abs(f["abs_epoch"])
        frame_rows.append(dict(unit=unit, idx=i, sheet=f"sheet_{k:03d}.jpg", row=r_, col=c_,
                               rel_s=round(f["rel_s"], 3) if f["rel_s"] is not None else "",
                               rel_label=rc.fmt_rel(f["rel_s"]), abs_utc=abs_utc, name=f["name"],
                               path=f["path"], video_frame="" if f["video_frame"] is None else f["video_frame"]))
        if render:
            im = rc.fit_width(im, thumb_w)
            l1 = f"{unit}  #{i:04d}  {rc.fmt_rel(f['rel_s'])}"
            l2 = abs_utc if abs_utc else f["name"][:int(thumb_w / (font_size * 0.6))]
            tiles.append(rc.burn_text(im, [l1, l2], size=font_size))
            if len(tiles) == per_sheet:
                _save_sheet(tiles, udir, sheet_i, cols, thumb_w, unit, len(src.frames), n_sheets)
                sheet_i += 1
                tiles = []
    if render and tiles:
        _save_sheet(tiles, udir, sheet_i, cols, thumb_w, unit, len(src.frames), n_sheets)
    rc.write_csv(udir / "frames_index.csv", frame_rows, FRAME_COLS)
    first = src.frames[0]
    sha = rc.sha256_file(first["path"], limit_bytes=1 << 20)
    meta = dict(unit=unit, scenario=row["scenario"], source=row.get("source", ""),
                independent_unit=row.get("independent_unit", unit), license=row.get("license", ""),
                input_kind=src.kind, path=row["path"], n_frames=len(src.frames),
                cadence_s=obs["cadence_s"], ts_kind=first["ts_kind"], valid_observed_s=obs["valid_s"],
                span_s=obs["span_s"], n_gaps_excluded=obs["n_gaps_excluded"], excluded_s=obs["excluded_s"],
                n_sheets=n_sheets, contamination=row.get("contamination", ""), notes=row.get("notes", ""),
                first_frame_sha256_1MB=sha)
    if src.kind == "video":
        meta["notes"] = (meta["notes"] + f" | video fps={src.fps:.3f} total_frames={getattr(src, 'n_video_frames', '?')} sampled_every={s_s}s").strip(" |")
    return meta


def _save_sheet(tiles, udir, k, cols, thumb_w, unit, n_frames, n_sheets):
    title = f"{unit}  sheet {k + 1}/{n_sheets}  ({n_frames} frames)  REVIEW: look for ANY smoke; faint plumes need the full-res frame (frames_index.csv)"
    sheet = rc.contact_sheet(tiles, cols, thumb_w, title=title)
    sheet.save(udir / f"sheet_{k:03d}.jpg", quality=88, optimize=True)


def merge_verdicts(vpath: Path, units: list):
    """Create template rows for new units; keep existing rows untouched."""
    existing = rc.read_csv(vpath) if vpath.exists() else []
    have = {r["unit"] for r in existing}
    rows = list(existing)
    for u in units:
        if u not in have:
            rows.append(dict(unit=u, reviewer="", verdict="", first_smoke_frame_or_blank="", notes="",
                             reviewed_utc=""))
    rows.sort(key=lambda r: rc.natural_key(r["unit"]))
    rc.write_csv(vpath, rows, VERDICT_COLS)
    return len(rows) - len(existing)


def cmd_sheets(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = load_manifest(a.manifest)
    if a.only:
        keep = set(a.only)
        man = [r for r in man if r["unit"] in keep]
    metas = []
    for r in man:
        m = build_unit(r, out, a.cols, a.rows, a.thumb_w, a.sample_s, a.font_size, a.overwrite)
        metas.append(m)
        print(f"[sheets] {m['unit']:<28} {m['input_kind']:<10} frames={m['n_frames']:>4} "
              f"valid={m['valid_observed_s']:>7.1f}s sheets={m['n_sheets']} scenario={m['scenario']}")
    # units.csv merge (replace rows for rebuilt units)
    upath = out / "units.csv"
    old = {r["unit"]: r for r in rc.read_csv(upath)} if upath.exists() else {}
    for m in metas:
        old[m["unit"]] = m
    rc.write_csv(upath, sorted(old.values(), key=lambda r: rc.natural_key(r["unit"])), UNIT_COLS)
    n_new = merge_verdicts(out / "verdicts.csv", [m["unit"] for m in metas])
    with open(out / "_runs.jsonl", "a") as f:
        f.write(json.dumps(dict(utc=rc.utc_now(), cmd="sheets", manifest=a.manifest, units=[m["unit"] for m in metas],
                                cols=a.cols, rows=a.rows, thumb_w=a.thumb_w)) + "\n")
    tot = sum(float(m["valid_observed_s"]) for m in metas)
    print(f"[sheets] {len(metas)} units, {tot / 3600:.3f} h candidate footage, {n_new} new verdict rows -> {out / 'verdicts.csv'}")
    print("[sheets] NOTE: nothing here is verified yet — a human must fill verdicts.csv (verdict in {no_smoke,smoke,uncertain}).")


def cmd_summarise(a):
    rd = Path(a.review_dir)
    units = {r["unit"]: r for r in rc.read_csv(rd / "units.csv")}
    vpath = Path(a.verdicts) if a.verdicts else rd / "verdicts.csv"
    verdicts = rc.read_csv(vpath) if vpath.exists() else []
    bad = [v for v in verdicts if v["verdict"] and v["verdict"] not in VALID_VERDICTS]
    if bad:
        raise SystemExit(f"invalid verdict values: {[(b['unit'], b['verdict']) for b in bad]}")
    vmap = {v["unit"]: v for v in verdicts}
    per = defaultdict(lambda: dict(units=0, no_smoke=0, smoke=0, uncertain=0, unreviewed=0,
                                   verified_s=0.0, candidate_s=0.0, indep_verified=set(), indep_all=set(),
                                   sources=set(), contaminated_verified_s=0.0, cadences=set()))
    unit_rows = []
    for u, m in sorted(units.items(), key=lambda kv: rc.natural_key(kv[0])):
        sc = m["scenario"]
        p = per[sc]
        p["units"] += 1
        vs = float(m["valid_observed_s"] or 0)
        p["candidate_s"] += vs
        p["indep_all"].add(m.get("independent_unit") or u)
        p["sources"].add(m.get("source", ""))
        p["cadences"].add(str(m.get("cadence_s", "")))
        v = vmap.get(u, {}).get("verdict", "")
        if v == "no_smoke":
            p["no_smoke"] += 1
            p["verified_s"] += vs
            p["indep_verified"].add(m.get("independent_unit") or u)
            if m.get("contamination"):
                p["contaminated_verified_s"] += vs
        elif v == "smoke":
            p["smoke"] += 1
        elif v == "uncertain":
            p["uncertain"] += 1
        else:
            p["unreviewed"] += 1
        unit_rows.append(dict(unit=u, scenario=sc, source=m.get("source", ""), verdict=v or "(blank)",
                              reviewer=vmap.get(u, {}).get("reviewer", ""), valid_observed_s=vs,
                              counted_h=round(vs / 3600, 4) if v == "no_smoke" else 0.0,
                              contamination=m.get("contamination", "")))
    lines = []
    lines.append(f"# Verified no-smoke footage summary (verdicts: {vpath})")
    lines.append("")
    lines.append("Counting rule: ONLY verdict == no_smoke contributes; uncertain / smoke / blank contribute 0. "
                 "Hours are PROTOCOL_V5 §3 valid observed time per unit (gaps > 3x median cadence excluded, units never concatenated). "
                 "Acceptance (§6) per scenario: >= 2.0 h verified AND >= 3 independent units AND licence recorded.")
    lines.append("")
    lines.append("| scenario | units | no_smoke | smoke | uncertain | unreviewed | candidate h | VERIFIED no-smoke h | of which contaminated h | independent units verified/all | cadence_s | acceptance |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for sc in sorted(per):
        p = per[sc]
        vh = p["verified_s"] / 3600
        acc = "MET" if (vh >= 2.0 and len(p["indep_verified"]) >= 3) else "NOT MET"
        lines.append(f"| {sc} | {p['units']} | {p['no_smoke']} | {p['smoke']} | {p['uncertain']} | {p['unreviewed']} | "
                     f"{p['candidate_s'] / 3600:.3f} | **{vh:.3f}** | {p['contaminated_verified_s'] / 3600:.3f} | "
                     f"{len(p['indep_verified'])}/{len(p['indep_all'])} | {','.join(sorted(p['cadences']))} | {acc} |")
        summary[sc] = dict(units=p["units"], no_smoke=p["no_smoke"], smoke=p["smoke"], uncertain=p["uncertain"],
                           unreviewed=p["unreviewed"], candidate_h=round(p["candidate_s"] / 3600, 4),
                           verified_no_smoke_h=round(vh, 4), contaminated_verified_h=round(p["contaminated_verified_s"] / 3600, 4),
                           independent_units_verified=len(p["indep_verified"]), independent_units_all=len(p["indep_all"]),
                           sources=sorted(p["sources"]), acceptance=acc)
    lines.append("")
    lines.append("| unit | scenario | source | verdict | reviewer | valid s | counted h | contamination |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in unit_rows:
        lines.append(f"| {r['unit']} | {r['scenario']} | {r['source']} | {r['verdict']} | {r['reviewer']} | {r['valid_observed_s']:.0f} | {r['counted_h']:.4f} | {r['contamination']} |")
    text = "\n".join(lines)
    print(text)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(text + "\n")
    if a.json:
        rc.write_json(a.json, dict(verdicts_file=str(vpath), generated_utc=rc.utc_now(), scenarios=summary, units=unit_rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sheets", help="build contact sheets + verdict template")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", default="data/v5/review")
    s.add_argument("--cols", type=int, default=4)
    s.add_argument("--rows", type=int, default=6)
    s.add_argument("--thumb-w", type=int, default=480)
    s.add_argument("--font-size", type=int, default=14)
    s.add_argument("--sample-s", type=float, default=1.0, help="video sampling period (s)")
    s.add_argument("--only", nargs="*", help="restrict to these unit ids")
    s.add_argument("--overwrite", action="store_true", help="re-render sheets even if present")
    s.set_defaults(fn=cmd_sheets)
    m = sp.add_parser("summarise", help="hours verified no-smoke by scenario")
    m.add_argument("--review-dir", default="data/v5/review")
    m.add_argument("--verdicts", help="alternative verdicts CSV (default <review-dir>/verdicts.csv)")
    m.add_argument("--md", help="write markdown summary here")
    m.add_argument("--json", help="write JSON summary here")
    m.set_defaults(fn=cmd_summarise)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
