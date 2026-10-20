#!/usr/bin/env python3
"""
build_eval_v5b.py -- V5b evaluation packaging. DATA PREP ONLY: no model, no prediction, no metric.

Builds under data/v5b/eval/ (images symlinked to the raw frames; labels copied from the YOLO export):
  fireman_val_s5       FireMan proposed_split=val (burn D, rgb_train_video_015, 8,530 RGB frames incl. 1,154
                       author explicit negatives) -> every 5th frame of the WHOLE video (frame_idx % 5 == 0;
                       the stride is applied to the video, not per class, so negatives keep the same cadence).
                       VALIDATION-side (thresholds / selection) -- NOT sealed.
  fireman_holdout_s10  FireMan proposed_split=holdout (burn C Taivalkoski 2023-08-26, 5 RGB videos, 19,458
                       exported frames, 0 explicit negatives) -> every 10th frame per video (frame_idx % 10 == 0).
                       SEALED final holdout H1.
  sheffield_holdout    Sheffield UAV rec_0008 (327 frames; same burn/site/session as its train recordings ->
                       within-incident only). Final holdout H3.
Per set: lists/{local,pod}/<set>.txt, SHA256SUMS (per image), yamls/<set>.yaml (val: and test: -> local list),
yamls/pod/<set>.yaml (pod paths). ONE dereferenced tar data/v5b/stage/eval_v5b.tar (top dir eval_v5b/, extracts
under /scratch/scratch/tianrgua-sandbox/smoke/v5bdata/). Manifest data/v5b/eval/EVAL_V5B_MANIFEST.json.
Overlap check (realpath AND sha256 of bytes) against every list under data/v5b/yolo/*/lists/local/ and
data/yolo/lists/local/. Lockbox note: reports/v5b/LOCKBOX_V5B_FROZEN.md.
"""
import glob
import hashlib
import json
import os
import re
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

ROOT = Path("/local/home/tianrgua/cvproject")
EVAL_ROOT = ROOT / "data/v5b/eval"
PKG = "eval_v5b"
POD_EXTRACT_DIR = "/scratch/scratch/tianrgua-sandbox/smoke/v5bdata"
POD_ROOT = f"{POD_EXTRACT_DIR}/{PKG}"
SCRIPT = "scripts/v5b/build_eval_v5b.py"
POLICY = ("Evaluated once at V5b close; never used for pseudo-labels, mining, fusion fitting, "
          "threshold tuning, or model selection.")
FROZEN_UTC = datetime.now(timezone.utc).isoformat(timespec="seconds")
FM = ROOT / "data/v5b/yolo/fireman"
SH = ROOT / "data/v5b/yolo/sheffield_uav"
HOLDOUT_CONFLICT = ("rgb/val/video_002 frame_000706 (presence_no_but_boxes) skipped by the YOLO export; "
                    "706 % 10 != 0 so the stride-10 selection is unaffected")

SETS = {
    "fireman_val_s5": dict(src_img=FM / "images/val", src_lbl=FM / "labels/val", stride=5, kind="fireman",
                           burn="fireman_burnD_overcast_clearcut_edge", sealed=False, holdout_id=None,
                           role="VALIDATION (aerial thresholds / selection) -- NOT sealed"),
    "fireman_holdout_s10": dict(src_img=FM / "images/holdout", src_lbl=FM / "labels/holdout", stride=10,
                                kind="fireman", burn="fireman_burnC_taivalkoski_2023-08-26", sealed=True,
                                holdout_id="H1", role="SEALED FINAL HOLDOUT H1"),
    "sheffield_holdout": dict(src_img=SH / "images/holdout", src_lbl=SH / "labels/holdout", stride=1,
                              kind="sheffield", burn="sheffield_uav_session_2022-07-21 (rec_0008)", sealed=True,
                              holdout_id="H3", role="FINAL HOLDOUT H3 (within-incident only)"),
}
FM_RE = re.compile(r"^(rgb_(?:train|val|test)_video_\d+)_frame_(\d+)\.jpg$")
SH_RE = re.compile(r"^DJI_\d+_(\d{4})_([WZ])_MP4-(\d+)_jpg\.rf\.[A-Za-z0-9]+\.jpg$")
REF_LIST_GLOBS = ["data/v5b/yolo/*/lists/local/*.txt", "data/yolo/lists/local/*.txt"]
# the only overlap that is allowed (and expected): Sheffield rec_0008 inside the B/C/D training lists
ALLOWED_OVERLAP_SET = "sheffield_holdout"
ALLOWED_OVERLAP_LIST_RE = re.compile(r"^data/v5b/yolo/[BCD]_v1/lists/local/(train|train_sheffield)\.txt$")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _sha_job(p):
    return p, sha256_file(p)


def sha256_many(paths, procs=96):
    paths = list(paths)
    with Pool(procs) as pool:
        return dict(pool.imap_unordered(_sha_job, paths, chunksize=32))


def _hdr_job(p):
    with Image.open(p) as im:  # header only -- pixels are never decoded
        return p, (im.size[0], im.size[1], im.format)


def headers_many(paths, procs=32):
    with Pool(procs) as pool:
        return dict(pool.imap_unordered(_hdr_job, list(paths), chunksize=16))


def select(cfg):
    """-> entries [(fname, abs_raw, label_lines, unit, frame_idx)], n_source_files, Counter(source frames/unit)."""
    entries, n_src, src_units = [], 0, Counter()
    for fn in sorted(os.listdir(cfg["src_img"])):
        n_src += 1
        if cfg["kind"] == "fireman":
            m = FM_RE.match(fn)
            assert m, fn
            unit, idx = m.group(1), int(m.group(2))
        else:
            m = SH_RE.match(fn)
            assert m, fn
            assert m.group(1) == "0008", f"non-rec_0008 frame in sheffield holdout export: {fn}"
            unit, idx = f"rec_{m.group(1)}_{m.group(2)}", int(m.group(3))
        src_units[unit] += 1
        if idx % cfg["stride"] != 0:
            continue
        link = cfg["src_img"] / fn
        assert link.is_symlink(), f"expected symlink: {link}"
        raw = os.path.realpath(link)
        assert os.path.isfile(raw), raw
        lbl = cfg["src_lbl"] / (os.path.splitext(fn)[0] + ".txt")
        assert lbl.is_file(), f"missing label file: {lbl}"
        lines = [l.strip() for l in lbl.read_text().splitlines() if l.strip()]
        for l in lines:
            parts = l.split()
            assert len(parts) == 5 and parts[0] == "0", (str(lbl), l)
            assert all(0.0 <= float(v) <= 1.0 for v in parts[1:]), (str(lbl), l)
        entries.append((fn, raw, lines, unit, idx))
    assert len({e[0] for e in entries}) == len(entries)
    assert len({e[1] for e in entries}) == len(entries)
    return entries, n_src, src_units


def materialize(name, cfg, entries, img_sha, img_hdr):
    d = EVAL_ROOT / name
    for sub in ("images", "labels", "lists/local", "lists/pod"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    local_lines, pod_lines, sums = [], [], []
    n_pos = n_neg = n_boxes = 0
    per_unit, res = defaultdict(Counter), Counter()
    for fn, raw, lines, unit, idx in entries:
        link = d / "images" / fn
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(raw, link)
        (d / "labels" / (os.path.splitext(fn)[0] + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        local_lines.append(str(link))
        pod_lines.append(f"{POD_ROOT}/{name}/images/{fn}")
        sums.append(f"{img_sha[raw]}  images/{fn}")
        W, H, fmt = img_hdr[raw]
        assert fmt == "JPEG", (raw, fmt)
        res[f"{W}x{H}"] += 1
        per_unit[unit]["images"] += 1
        if lines:
            n_pos += 1
            per_unit[unit]["positives"] += 1
        else:
            n_neg += 1
            per_unit[unit]["negatives"] += 1
        n_boxes += len(lines)
        per_unit[unit]["boxes"] += len(lines)
    lst = d / "lists/local" / f"{name}.txt"
    pod = d / "lists/pod" / f"{name}.txt"
    lst.write_text("\n".join(local_lines) + "\n")
    pod.write_text("\n".join(pod_lines) + "\n")
    (d / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    yd = EVAL_ROOT / "yamls"
    (yd / "pod").mkdir(parents=True, exist_ok=True)
    yp = yd / f"{name}.yaml"
    yp.write_text(f"# v5b {PKG} {name} -- {cfg['role']} -- generated by {SCRIPT}\n# {POLICY}\n"
                  f"path: {d}\nval: {lst}\ntest: {lst}\nnc: 1\nnames: [smoke]\n")
    ypp = yd / "pod" / f"{name}.yaml"
    podlst = f"{POD_ROOT}/{name}/lists/pod/{name}.txt"
    ypp.write_text(f"# v5b {PKG} {name} -- {cfg['role']} -- POD paths -- generated by {SCRIPT}\n# {POLICY}\n"
                   f"path: {POD_ROOT}/{name}\nval: {podlst}\ntest: {podlst}\nnc: 1\nnames: [smoke]\n")
    return {"role": cfg["role"], "sealed": cfg["sealed"], "holdout_id": cfg["holdout_id"], "burn": cfg["burn"],
            "stride": cfg["stride"],
            "stride_rule": (f"frame_idx % {cfg['stride']} == 0 on the original per-video frame index "
                            f"(whole video, not per class)" if cfg["stride"] > 1 else "all exported frames"),
            "images": len(local_lines), "positives": n_pos, "negatives": n_neg, "boxes": n_boxes,
            "resolutions": dict(res), "n_units": len(per_unit),
            "per_unit": {u: dict(c) for u, c in sorted(per_unit.items())},
            "lists": {"local": {"path": str(lst), "n": len(local_lines), "sha256": sha256_file(lst)},
                      "pod": {"path": str(pod), "pod_path": podlst, "n": len(pod_lines),
                              "sha256": sha256_file(pod)}},
            "sha256sums": {"path": str(d / "SHA256SUMS"), "sha256": sha256_file(d / "SHA256SUMS")},
            "yaml": {"path": str(yp), "sha256": sha256_file(yp)},
            "yaml_pod": {"path": str(ypp), "tar_member": f"{PKG}/yamls/{name}.yaml", "sha256": sha256_file(ypp)}}


def overlap_check(entries_by_set, eval_sha):
    ref_lists = sorted({p for g in REF_LIST_GLOBS for p in glob.glob(str(ROOT / g))})
    ref_paths, list_n = {}, {}
    for lp in ref_lists:
        rel = os.path.relpath(lp, ROOT)
        lines = [l.strip() for l in Path(lp).read_text().splitlines() if l.strip()]
        list_n[rel] = len(lines)
        for l in lines:
            ref_paths.setdefault(os.path.realpath(l), set()).add(rel)
    print(f"[overlap] {len(ref_lists)} reference lists, {sum(list_n.values())} lines, "
          f"{len(ref_paths)} unique realpaths -> hashing bytes ...", flush=True)
    ref_sha = sha256_many(ref_paths.keys())
    sha_to_lists = defaultdict(set)
    for p, s in ref_sha.items():
        sha_to_lists[s].update(ref_paths[p])
    out = {"reference_lists": list_n, "reference_unique_realpaths": len(ref_paths),
           "method": "realpath equality AND sha256 of file bytes (all reference images hashed)",
           "per_set": {}}
    violations = []
    for name, entries in entries_by_set.items():
        path_hits, sha_hits, n_path, n_sha = Counter(), Counter(), 0, 0
        for fn, raw, lines, unit, idx in entries:
            if raw in ref_paths:
                n_path += 1
                for l in ref_paths[raw]:
                    path_hits[l] += 1
            s = eval_sha[raw]
            if s in sha_to_lists:
                n_sha += 1
                for l in sha_to_lists[s]:
                    sha_hits[l] += 1
        for l in set(path_hits) | set(sha_hits):
            if not (name == ALLOWED_OVERLAP_SET and ALLOWED_OVERLAP_LIST_RE.match(l)):
                violations.append((name, l, path_hits[l], sha_hits[l]))
        out["per_set"][name] = {"path_overlap": n_path, "sha256_overlap": n_sha,
                                "path_hits_by_list": dict(sorted(path_hits.items())),
                                "sha256_hits_by_list": dict(sorted(sha_hits.items()))}
        print(f"[overlap] {name}: path_overlap={n_path} sha256_overlap={n_sha} "
              f"lists_hit={sorted(set(path_hits) | set(sha_hits))}", flush=True)
    assert not violations, f"UNEXPECTED OVERLAP: {violations}"
    return out


def h3_contamination(ov, n_h3):
    hits = ov["per_set"]["sheffield_holdout"]
    contaminated_lists = sorted(set(hits["path_hits_by_list"]) | set(hits["sha256_hits_by_list"]))
    arms = sorted({re.search(r"/([BCD])_v1/", l).group(1) for l in contaminated_lists})
    assert arms == ["B", "C", "D"], arms
    for l in contaminated_lists:
        assert hits["path_hits_by_list"][l] == n_h3 == hits["sha256_hits_by_list"][l], (l, hits)
    return {
        "status": "CONTAMINATED for arms B/C/D",
        "detail": (f"all {n_h3} rec_0008 frames are in the training lists of arms {'/'.join(arms)} "
                   f"(the Sheffield data agent's B_v1 corpus used ALL 1,327 exported Sheffield frames incl. "
                   f"val rec_0007 and holdout rec_0008); byte-identical (sha256) and same raw files (realpath)"),
        "contaminated_lists": contaminated_lists,
        "valid_for": ["A (expV5bA_ctrl_s3; trained on expB_train.txt only)", "expB", "expE",
                      "DINOv3-L LoRA (B200, run v5bB200_lora_L_1920) -- trains on "
                      "data/v5b/coco/instances_train_eb.json = E (38,454 pyro-sdis + D-Fire) + Boreal train "
                      "(2,062); 0 Sheffield and 0 FireMan frames (verified from the json)"],
        "reporting_rule": ("H3 numbers for B/C/D (and the ACCURATE ensemble, which contains B) must be labelled "
                           "CONTAMINATED (train/test overlap) and must not enter any comparison or decision; "
                           "even for uncontaminated arms H3 is within-incident generalisation only "
                           "(rec_0008 = same burn, site and 12-minute session as rec_0003-0007)."),
    }


def write_report(st):
    L = [f"# LOCKBOX V5B -- FROZEN (final holdouts H1 + H3, iteration 5b)", "",
         f"- Frozen (UTC): **{st['frozen_utc']}**",
         f"- Built by: `{SCRIPT}` (DATA PREP ONLY -- no model run, no prediction, no metric).",
         f"- Manifest: `{st['manifest']}`",
         f"- **{POLICY}**",
         f"- Pre-registration: `reports/v5b/PREREGISTRATION_V5B.md` (frozen 2026-10-02 09:30 UTC + 09:35 amendment).",
         f"- Timeline note: build stamp above is the devbox system clock; `PREREGISTRATION_V5B.md` file mtime = "
         f"{datetime.fromtimestamp(os.stat(ROOT / 'reports/v5b/PREREGISTRATION_V5B.md').st_mtime, timezone.utc).isoformat(timespec='seconds')} "
         f"(same clock) while its text states 09:30 / 09:35 UTC -> the pre-registration file predates this build; "
         f"the stated clock times in that file are not those of the system clock.",
         "", "## What is in the package", "",
         "| set | role | units | frames (stride) | positives | negatives | boxes | resolution |",
         "|---|---|---|---|---|---|---|---|"]
    for name in SETS:
        s = st[name]
        frames = (f"{s['images']} (every {s['stride']}th of {s['source_frames']} exported)" if s['stride'] > 1
                  else f"{s['images']} (all {s['source_frames']} exported)")
        L.append(f"| {name} | {s['role']} | {s['burn']} ({s['n_units']} video{'s' if s['n_units'] > 1 else ''}) | "
                 f"{frames} | {s['positives']} | {s['negatives']} | {s['boxes']} | "
                 f"{', '.join(f'{k}: {v}' for k, v in s['resolutions'].items())} |")
    L += ["", "### Caveats (read before interpreting any number)", "",
          "- **fireman_holdout_s10 (H1) = ONE burn** (burn C, Taivalkoski 2023-08-26 -- the only GPS/time-confirmed "
          "FireMan burn, via the DJI overlay on its paired thermal videos). 5 RGB videos, 640x512 rectified RGB at "
          "thermal geometry (low resolution). Frames are every 10th frame of each video to limit within-video "
          "correlation; the effective sample size is still 1 event.",
          "- H1 has **0 explicit negatives** (all 19,459 burn-C RGB frames are author smoke-positive) -> per the "
          "09:35 amendment H1 reports **RECALL ONLY** (image-level + box recall @ IoU>=0.4 at the pre-registered "
          "confs). No holdout-based false-positive claim will be made. Author boxes are interpolated/projected "
          "region boxes -> box IoU metrics are labelled weak.",
          f"- H1 source: 19,459 burn-C frames; the YOLO export holds 19,458 ({HOLDOUT_CONFLICT}).",
          "- Per-video H1 frames: " + ", ".join(f"{u} {c['images']}" for u, c in st['fireman_holdout_s10']['per_unit'].items()) + ".",
          "- **sheffield_holdout (H3) = within-incident only**: rec_0008 is the same prescribed burn, same site, "
          "same 12-minute UAV session as the train/val recordings rec_0003-0007 (unknown provenance, Roboflow "
          "re-deposit). It must never be reported as an independent site. 0 negatives -> recall-only.",
          f"- **H3 CONTAMINATION: {st['h3_contamination']['status']}.** {st['h3_contamination']['detail']}. "
          f"Lists: {', '.join('`' + l + '`' for l in st['h3_contamination']['contaminated_lists'])}. "
          f"Still valid for: {'; '.join(st['h3_contamination']['valid_for'])}. "
          f"{st['h3_contamination']['reporting_rule']}",
          "- **fireman_val_s5 is NOT sealed.** It is the aerial VALIDATION video (burn D, video_015, overcast "
          "clearcut edge, 958x768) used for the pre-registered aerial thresholds and the secondary selection "
          "criterion. Every 5th frame of the whole video (frame_idx % 5 == 0), so the "
          f"{st['fireman_val_s5']['negatives']} selected explicit negatives (of 1,154 in the video) keep the same "
          "cadence as the positives. One video -> one unit; its 'negatives' are one pre-/off-burn flight segment, "
          "not independent samples.",
          "- H2 (pyro-sdis TEST, `data/yolo/lists/local/sdis_test.txt`) is NOT part of this package; it stays "
          "frozen as in v1-v4 (opened once at v4 close -> second opening to be disclosed).",
          "- Burn identification for FireMan units A/B/D is agent visual judgement; only burn C (H1) is "
          "metadata-confirmed (`data/v5b/manifests/fireman_video_units.json`).",
          "", "## Overlap check against every training list", ""]
    ov = st["overlap_check"]
    L.append(f"- Reference: {len(ov['reference_lists'])} lists (`data/v5b/yolo/*/lists/local/*.txt`, "
             f"`data/yolo/lists/local/*.txt`), {sum(ov['reference_lists'].values())} lines, "
             f"{ov['reference_unique_realpaths']} unique images; sha256 of bytes computed for ALL of them.")
    for name in SETS:
        p = ov["per_set"][name]
        L.append(f"- `{name}`: path_overlap={p['path_overlap']}, sha256_overlap={p['sha256_overlap']}"
                 + (f" -- hits only in {sorted(set(p['path_hits_by_list']) | set(p['sha256_hits_by_list']))} "
                    f"(= the H3 contamination above)" if p['path_overlap'] or p['sha256_overlap'] else " (clean)"))
    L += ["", "## Seal (sha256)", "",
          f"- tar `{st['tar']['path']}` ({st['tar']['bytes']:,} bytes, {st['tar']['n_members']} members): "
          f"`{st['tar']['sha256']}`",
          f"  - pod extraction dir: `{POD_EXTRACT_DIR}/` -> `{POD_ROOT}/<set>/{{images,labels,lists,SHA256SUMS}}`, "
          f"`{POD_ROOT}/yamls/<set>.yaml` (pod paths)"]
    for name in SETS:
        s = st[name]
        L.append(f"- `{s['lists']['local']['path']}` ({s['lists']['local']['n']}): `{s['lists']['local']['sha256']}`")
        L.append(f"- `{s['lists']['pod']['path']}` ({s['lists']['pod']['n']}): `{s['lists']['pod']['sha256']}`")
        L.append(f"- `{s['sha256sums']['path']}` (per-image sha256): `{s['sha256sums']['sha256']}`")
        L.append(f"- `{s['yaml']['path']}`: `{s['yaml']['sha256']}`")
    L += ["", "## Rules", "",
          f"1. {POLICY}",
          "2. fireman_holdout_s10 and sheffield_holdout: no script other than the one-time V5b close evaluation may "
          "read the images listed above (this builder only symlinked them, copied label text, hashed bytes and "
          "read JPEG headers). Any per-image output of that evaluation is HISTORICAL afterwards.",
          "3. If any list/tar sha256 above changes, the seal is broken and the set must be re-frozen and "
          "re-declared before use.",
          "4. H3 results for arms B/C/D and for any ensemble containing them are CONTAMINATED and are reported "
          "only as such; H3 is within-incident generalisation for every system.",
          "5. H1 is recall-only (0 negatives); aerial FP rates come from FireMan VAL (fireman_val_s5, 1 video) and "
          "Boreal heinola VAL, both validation-side.",
          f"6. Layout mirrors `v4_evaltest` (images/ labels/ lists/{{local,pod}}/<set>.txt, SHA256SUMS, yaml with "
          f"val: and test:); consumers pass the list or yaml path explicitly."]
    rp = ROOT / "reports/v5b/LOCKBOX_V5B_FROZEN.md"
    rp.write_text("\n".join(L) + "\n")
    return str(rp)


def main():
    for name in SETS:
        assert not (EVAL_ROOT / name / "lists/local" / f"{name}.txt").exists(), \
            f"{name} already materialised -- refusing to re-freeze silently; remove it first"
    tar_path = ROOT / "data/v5b/stage" / f"{PKG}.tar"
    assert not tar_path.exists(), f"{tar_path} exists -- refusing to overwrite a frozen tar"
    sel = {}
    for name, cfg in SETS.items():
        entries, n_src, src_units = select(cfg)
        sel[name] = (entries, n_src, src_units)
        print(f"[select] {name}: {len(entries)} / {n_src} exported frames, units={dict(src_units)}", flush=True)
    assert sel["fireman_val_s5"][1] == 8530 and sel["fireman_holdout_s10"][1] == 19458 \
        and sel["sheffield_holdout"][1] == 327, "source export counts changed vs the data CARDs"
    all_raw = [e[1] for v in sel.values() for e in v[0]]
    assert len(set(all_raw)) == len(all_raw), "the three sets share raw frames"
    print(f"[sha] hashing {len(all_raw)} eval images + reading JPEG headers ...", flush=True)
    eval_sha = sha256_many(all_raw)
    hdr = headers_many(all_raw)
    st = {"package": PKG, "frozen_utc": FROZEN_UTC, "script": SCRIPT, "policy": POLICY,
          "pod_root": POD_ROOT, "pod_extract_dir": POD_EXTRACT_DIR, "data_prep_only": True,
          "sources": {"fireman": "data/v5b/yolo/fireman/{images,labels}/{val,holdout} "
                                 "(fireman_export_yolo.py; proposed_split from fireman_video_units.json)",
                      "sheffield_uav": "data/v5b/yolo/sheffield_uav/{images,labels}/holdout (rec_0008)",
                      "label_origin": "author annotations (FireMan CVAT smoke_region / smoke_presence=no; "
                                      "Sheffield Roboflow-user smoke boxes); nothing human-reviewed by us; "
                                      "no teacher prediction replaces any evaluation label"}}
    for name, cfg in SETS.items():
        entries, n_src, src_units = sel[name]
        st[name] = materialize(name, cfg, entries, eval_sha, hdr)
        st[name]["source_frames"] = n_src
        st[name]["source_frames_per_unit"] = dict(sorted(src_units.items()))
    st["fireman_val_s5"].update({"source_explicit_negatives": 1154, "source_box_positives": 7376,
                                 "independent_events": 1,
                                 "note": "stride on the whole video; negatives and positives share cadence"})
    st["fireman_holdout_s10"].update({"source_frames_burn": 19459, "export_conflict_skipped": HOLDOUT_CONFLICT,
                                      "independent_events": 1, "fp_claim": "none (0 negatives) -> recall only"})
    st["sheffield_holdout"].update({"independent_events": 0,
                                    "within_incident": "same burn/site/12-min session as rec_0003-0007 (train/val)"})
    assert st["fireman_val_s5"]["images"] == 1706 and st["fireman_holdout_s10"]["negatives"] == 0 \
        and st["sheffield_holdout"]["images"] == 327 and st["sheffield_holdout"]["negatives"] == 0
    st["overlap_check"] = overlap_check({n: sel[n][0] for n in SETS}, eval_sha)
    st["h3_contamination"] = h3_contamination(st["overlap_check"], st["sheffield_holdout"]["images"])
    print("[tar] writing dereferenced tar ...", flush=True)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w", dereference=True) as tf:
        for name in SETS:
            tf.add(EVAL_ROOT / name, arcname=f"{PKG}/{name}")
        for name in SETS:
            tf.add(EVAL_ROOT / "yamls/pod" / f"{name}.yaml", arcname=f"{PKG}/yamls/{name}.yaml")
    with tarfile.open(tar_path) as tf:
        members = tf.getmembers()
        n_img = sum(1 for m in members if m.isfile() and "/images/" in m.name)
        assert all(not m.issym() for m in members), "tar contains symlinks -- dereference failed"
    assert n_img == len(all_raw), (n_img, len(all_raw))
    st["tar"] = {"path": str(tar_path), "bytes": tar_path.stat().st_size, "sha256": sha256_file(tar_path),
                 "n_members": len(members), "n_image_files": n_img, "pod_extract_dir": POD_EXTRACT_DIR,
                 "kraken_data_tar_arg": f"v5b_eval=data/v5b/stage/{PKG}.tar",
                 "layout": f"{PKG}/<set>/{{images,labels,lists/{{local,pod}},SHA256SUMS}} + {PKG}/yamls/<set>.yaml"}
    mp = EVAL_ROOT / "EVAL_V5B_MANIFEST.json"
    st["manifest"] = str(mp)
    st["report"] = write_report(st)
    mp.write_text(json.dumps(st, indent=1))
    for name in SETS:
        s = st[name]
        print(f"[done] {name}: images={s['images']} pos={s['positives']} neg={s['negatives']} boxes={s['boxes']} "
              f"units={s['n_units']} res={s['resolutions']}")
    print(f"[done] tar {st['tar']['bytes']:,} B sha256={st['tar']['sha256'][:16]}... members={st['tar']['n_members']}")
    print(f"[done] manifest {mp}\n[done] report {st['report']}")


if __name__ == "__main__":
    main()
