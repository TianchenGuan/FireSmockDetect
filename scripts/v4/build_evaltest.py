"""Build the iteration-4 SEALED TEST eval package (v4_evaltest).

DATA PREP ONLY. This script never loads a model, never computes a prediction
and never reads a metric. It materialises ground-truth for the frozen TEST
units so that they can be evaluated ONCE at iteration-4 close.

  boreal_test = burn_ruokolahti (data/v4/manifests/aerial_split_v4.json "test")
  forest_test = the 9 AuSmoke units in forest_holdout_v4.json "test"
  sdis_test   = data/yolo/lists/local/sdis_test.txt (expB.yaml "test"); only its
                sha256 is recorded here -- its images are NOT touched/copied.

This is the mirror image of scripts/v4/build_evalval.py: that script hard-bans
the test units; this one hard-bans everything that is NOT a test unit and
asserts zero overlap with every train/val list by absolute path AND by sha256
of the image bytes.

Outputs
  data/v4/eval/boreal_test/{images,labels,lists/{local,pod},SHA256SUMS}
  data/v4/eval/forest_test/{images,labels,lists/{local,pod},SHA256SUMS}
      lists/{local,pod}/test.txt
  data/v4/eval/yamls/{boreal_test,forest_test}.yaml   (keys val: AND test: ->
      the same list file, so src.eval.eval_detection --split {val,test} works)
  data/v4/coco/instances_{boreal_test,forest_test}.json (file_name = ABSOLUTE
      pod path under /scratch/.../smoke/v4data/v4_evaltest/)
  data/v4/stage/v4_evaltest.tar  (dereferenced; top dir v4_evaltest/)
  data/v4/eval/EVALTEST_MANIFEST.json
  reports/v4/LOCKBOX_V4_FROZEN.md

Label policy = identical to build_evalval.py (converters imported from
scripts/v4/build_corpus.py):
  boreal_test: Subset-A box rows only. has_smoke box rows -> yolo_txt_to_lines;
      has_smoke=False box rows (Empty-Images with explicit Empty-Labels) ->
      verified-empty negatives. Subset-C rows are per-basename re-encodes of
      Subset-A frames -> skipped, counted. Subset-B videolabel .mp4 -> skipped.
      label_type != box rows (3 Subset-A frames with an empty txt in the
      positive-image folder, NOT in Empty-Images) -> excluded as ambiguous,
      exactly like the val builder's `unexpected_label_type_*` branch.
  forest_test: all mask rows -> mask_to_lines (same converter as corpus/val).
      Zero negatives exist (AuSmoke is positive-only).

Run: .venv/bin/python scripts/v4/build_evaltest.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path("/local/home/tianrgua/cvproject")
sys.path.insert(0, str(ROOT / "scripts/v4"))
from build_corpus import (LabelResolver, flat_name, mask_to_lines,  # noqa: E402
                          yolo_txt_to_lines)
from yolo_to_coco_eet import image_size  # noqa: E402

EVAL_ROOT = ROOT / "data/v4/eval"
PKG = "v4_evaltest"
POD_ROOT = f"/scratch/scratch/tianrgua-sandbox/smoke/v4data/{PKG}"
LIST_NAME = "test.txt"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
SCRIPT = "scripts/v4/build_evaltest.py"
FROZEN_UTC = datetime.now(timezone.utc).isoformat(timespec="seconds")
POLICY = ("Evaluated once at iteration-4 close; never used for training, mining, "
          "tuning, threshold selection, or model selection.")

AERIAL_SPLIT = json.loads((ROOT / "data/v4/manifests/aerial_split_v4.json").read_text())
HOLD = json.loads((ROOT / "data/v4/manifests/forest_holdout_v4.json").read_text())
BOREAL_TEST_UNITS = set(AERIAL_SPLIT["sources"]["boreal"]["test"])
BOREAL_NONTEST_UNITS = (set(AERIAL_SPLIT["sources"]["boreal"]["train"])
                        | set(AERIAL_SPLIT["sources"]["boreal"]["val"]))
FOREST_TEST_UNITS = list(HOLD["test"])
FOREST_NONTEST_UNITS = set(HOLD["train"]) | set(HOLD["val"])
UNIT_GROUP = {u: g for g, us in HOLD["groups"].items() for u in us}
STAGED_TEST_RUN_UNIT = "aus_DJI_20230922115131_0004_D_TEST_RUN_SMOKE"
AERIAL_EVENT_GROUP = "g_dji_20230922"

assert BOREAL_TEST_UNITS == {"burn_ruokolahti"}, BOREAL_TEST_UNITS
assert not BOREAL_TEST_UNITS & BOREAL_NONTEST_UNITS
assert len(FOREST_TEST_UNITS) == 9, FOREST_TEST_UNITS
assert not set(FOREST_TEST_UNITS) & FOREST_NONTEST_UNITS, "forest test/nontest overlap"
assert STAGED_TEST_RUN_UNIT in FOREST_TEST_UNITS
assert set(HOLD["groups"][AERIAL_EVENT_GROUP]) <= set(FOREST_TEST_UNITS)
assert set(HOLD["assignment"]["test"]) == {UNIT_GROUP[u] for u in FOREST_TEST_UNITS}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _sha_job(p):
    return p, sha256_file(p)


def sha256_many(paths, procs=64):
    paths = sorted(set(paths))
    out = {}
    with Pool(min(procs, os.cpu_count() or 1)) as pool:
        for p, h in pool.imap_unordered(_sha_job, paths, chunksize=32):
            out[p] = h
    return out


# ----------------------------------------------------------------------------
# entries: (abs_img, label_lines, unit) -- label_lines [] => negative
# ----------------------------------------------------------------------------
def build_boreal(excl: Counter, per_flight: dict):
    b = pd.read_parquet(ROOT / "data/v4/manifests/boreal_manifest.parquet")
    t = b[b.parent_unit.isin(BOREAL_TEST_UNITS)]
    assert len(t) == AERIAL_SPLIT["sources"]["boreal"]["frames_per_unit"]["burn_ruokolahti"]
    subA_basenames = {os.path.basename(p) for p in t.rel_path
                      if "Subset-A" in p and os.path.splitext(p)[1].lower() in IMG_EXTS}
    res = LabelResolver("boreal", ROOT / "data/v4/raw/boreal")
    entries = []
    for r in t.itertuples(index=False):
        if os.path.splitext(r.rel_path)[1].lower() not in IMG_EXTS:
            excl["boreal:non_image(videolabel_mp4)"] += 1
            continue
        if "Subset-C" in r.rel_path:
            if os.path.basename(r.rel_path) in subA_basenames:
                excl["boreal:subsetC_duplicate_of_subsetA"] += 1
            else:
                excl["boreal:subsetC_not_in_subsetA_skipped(label_type_none)"] += 1
            continue
        img = ROOT / "data/v4/raw/boreal" / r.rel_path
        if not img.exists():
            excl["boreal:missing_file"] += 1
            continue
        if str(r.label_type) != "box":
            excl[f"boreal:unexpected_label_type_{r.label_type}"] += 1
            continue
        found = res.find_box_label(img)
        if found is None:
            excl["boreal:label_not_found"] += 1
            continue
        fmt, ref = found
        assert fmt == "yolo", f"unexpected label fmt {fmt} for {img}"
        lines, _ = yolo_txt_to_lines(ref, {0})
        if bool(r.has_smoke_label):
            if not lines:
                excl["boreal:smoke_pos_converted_to_zero_boxes"] += 1
                continue
            entries.append((str(img), lines, str(r.parent_unit)))
            per_flight[str(r.camera_or_flight_id)]["positives"] += 1
        else:
            if lines:
                excl["boreal:hs_false_but_boxes_found"] += 1
                continue
            assert "Empty-Images" in r.rel_path and "Empty-Labels" in str(ref), ref
            entries.append((str(img), [], str(r.parent_unit)))  # verified-empty negative
            per_flight[str(r.camera_or_flight_id)]["negatives"] += 1
    return entries


def build_forest(excl: Counter):
    m = pd.read_parquet(ROOT / "data/v4/manifests/multinatsmoke_manifest.parquet")
    mt = m[m.parent_unit.isin(FOREST_TEST_UNITS)]
    assert len(mt) == HOLD["n_frames"]["test"], (len(mt), HOLD["n_frames"]["test"])
    raw = ROOT / "data/v4/raw/multinatsmoke"
    entries = []
    for r in mt.itertuples(index=False):
        img = raw / r.rel_path
        mask = raw / r.mask_rel_path
        if not img.exists() or not mask.exists():
            excl["forest:missing_file"] += 1
            continue
        lines = mask_to_lines(mask, img)
        if not lines:
            excl["forest:mask_converted_to_zero_boxes"] += 1
            continue
        entries.append((str(img), lines, str(r.parent_unit)))
    return entries


# ----------------------------------------------------------------------------
def materialize(name: str, source: str, entries, img_sha: dict):
    d = EVAL_ROOT / name
    for sub in ("images", "labels", "lists/local", "lists/pod"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    raw_root = ROOT / "data/v4/raw" / source
    local_lines, pod_lines, coco_images, coco_anns, sums = [], [], [], [], []
    ann_id, n_pos, n_neg, n_boxes = 1, 0, 0, 0
    per_unit = defaultdict(Counter)
    used = set()
    for i, (abs_img, lines, unit) in enumerate(sorted(entries)):
        rel = os.path.relpath(abs_img, raw_root)
        fname = flat_name(source, rel)
        assert fname not in used, f"flat name collision: {fname}"
        used.add(fname)
        link = d / "images" / fname
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(abs_img, link)
        (d / "labels" / (os.path.splitext(fname)[0] + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        local_lines.append(str(link))
        pod_lines.append(f"{POD_ROOT}/{name}/images/{fname}")
        sums.append(f"{img_sha[abs_img]}  images/{fname}")
        W, H = image_size(abs_img)
        img_id = i + 1
        coco_images.append({"id": img_id,
                            "file_name": f"{POD_ROOT}/{name}/images/{fname}",
                            "flat_file_name": fname, "width": W, "height": H,
                            "parent_unit": unit})
        per_unit[unit]["images"] += 1
        if lines:
            n_pos += 1
            per_unit[unit]["positives"] += 1
        else:
            n_neg += 1
            per_unit[unit]["negatives"] += 1
        for ln in lines:
            _, cx, cy, w, h = (float(v) for v in ln.split())
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            bw, bh = w * W, h * H
            coco_anns.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                              "bbox": [round(x1, 2), round(y1, 2),
                                       round(bw, 2), round(bh, 2)],
                              "area": round(bw * bh, 2), "iscrowd": 0,
                              "segmentation": []})
            ann_id += 1
            n_boxes += 1
            per_unit[unit]["boxes"] += 1
    (d / "lists/local" / LIST_NAME).write_text("\n".join(local_lines) + "\n")
    (d / "lists/pod" / LIST_NAME).write_text("\n".join(pod_lines) + "\n")
    (d / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    coco = {"info": {"description": f"smoke-detect v4 evaltest -- {name} (SEALED TEST)",
                     "conversion": SCRIPT, "frozen_utc": FROZEN_UTC, "policy": POLICY,
                     "path_convention": f"file_name = ABSOLUTE pod path under {POD_ROOT}"},
            "licenses": [],
            "categories": [{"id": 1, "name": "smoke", "supercategory": "none"}],
            "images": coco_images, "annotations": coco_anns}
    cj = ROOT / "data/v4/coco" / f"instances_{name}.json"
    cj.write_text(json.dumps(coco))
    yd = EVAL_ROOT / "yamls"
    yd.mkdir(exist_ok=True)
    yp = yd / f"{name}.yaml"
    lst = d / "lists/local" / LIST_NAME
    yp.write_text(
        f"# v4 evaltest {name} -- SEALED TEST -- generated by {SCRIPT}\n"
        f"# {POLICY}\n"
        f"path: {d}\n"
        f"val: {lst}\n"
        f"test: {lst}\n"
        f"nc: 1\nnames: [smoke]\n")
    return {"images": len(local_lines), "positives": n_pos, "negatives": n_neg,
            "boxes": n_boxes,
            "per_unit": {u: dict(c) for u, c in sorted(per_unit.items())},
            "lists": {
                "local": {"path": str(lst), "n": len(local_lines), "sha256": sha256_file(lst)},
                "pod": {"path": str(d / "lists/pod" / LIST_NAME), "pod_path":
                        f"{POD_ROOT}/{name}/lists/pod/{LIST_NAME}", "n": len(pod_lines),
                        "sha256": sha256_file(d / "lists/pod" / LIST_NAME)}},
            "sha256sums": {"path": str(d / "SHA256SUMS"), "sha256": sha256_file(d / "SHA256SUMS")},
            "yaml": {"path": str(yp), "sha256": sha256_file(yp)},
            "coco": str(cj), "coco_sha256": sha256_file(cj)}


# ----------------------------------------------------------------------------
def overlap_check(test_abs: list[str], test_sha: dict):
    """No test image may appear in any train/val list by path or by bytes."""
    list_globs = [
        "data/v4/yolo/*/lists/local/train*.txt",   # v4 training corpora
        "data/v4/yolo/*/lists/local/gval.txt",     # v4 in-corpus val (extra)
        "data/v4/eval/*_val/lists/local/val.txt",  # v4 val package
        "data/yolo/lists/local/*",                 # v1-v3 sdis / dfire lists
    ]
    list_files = sorted({f for g in list_globs for f in glob.glob(str(ROOT / g))})
    ref_lines, ref_real = set(), set()
    per_list = {}
    for f in list_files:
        ls = [ln.strip() for ln in Path(f).read_text().splitlines() if ln.strip()]
        per_list[os.path.relpath(f, ROOT)] = len(ls)
        for ln in ls:
            ref_lines.add(ln)
            ref_real.add(os.path.realpath(ln))
    missing = [p for p in ref_real if not os.path.exists(p)]
    assert not missing, f"{len(missing)} reference images missing, e.g. {missing[:3]}"

    test_real = {os.path.realpath(p) for p in test_abs}
    path_hits = sorted((set(test_abs) | test_real) & (ref_lines | ref_real))

    print(f"[overlap] hashing {len(ref_real)} unique train/val images ...", flush=True)
    ref_sha = sha256_many(ref_real)
    sha_to_ref = defaultdict(list)
    for p, h in ref_sha.items():
        sha_to_ref[h].append(p)
    sha_hits = sorted({(p, test_sha[p], sha_to_ref[test_sha[p]][0])
                       for p in test_abs if test_sha[p] in sha_to_ref})

    # secondary (cheap) check: test bytes vs sha256 recorded for every NON-test
    # row of every v4 source manifest (catches raw-corpus rows not in any list).
    # KNOWN RE-DISTRIBUTION: the public MultiNatSmoke release ships the Boreal
    # dataset byte-for-byte under MultiNatSmokeDataset/{Train,Test}/Boreal/ ->
    # manifest origin_class="dup_boreal", parent_unit="boreal_ruokolahti_DJI_00xx".
    # build_corpus.py drops every origin_class != "usable" row, so these copies
    # never enter a corpus; they are TEST content, not non-test rows. They are
    # recorded separately and asserted absent from every train/val list.
    manifest_hits, redistributions = [], {}
    for mp in sorted((ROOT / "data/v4/manifests").glob("*_manifest.parquet")):
        df = pd.read_parquet(mp)
        if "sha256" not in df.columns:
            continue
        if mp.name == "boreal_manifest.parquet":
            df = df[~df.parent_unit.isin(BOREAL_TEST_UNITS)]
        elif mp.name == "multinatsmoke_manifest.parquet":
            dup = df[(df.origin_class.astype(str) == "dup_boreal")
                     & df.parent_unit.astype(str).str.startswith("boreal_ruokolahti")]
            dup_files = [os.path.realpath(str(ROOT / "data/v4/raw/multinatsmoke" / rp))
                         for rp in dup.rel_path.dropna()]
            in_lists = [f for f in dup_files if f in ref_real]
            assert not in_lists, f"dup_boreal ruokolahti copies in train/val lists: {in_lists[:3]}"
            redistributions["multinatsmoke:dup_boreal:burn_ruokolahti"] = {
                "rows": int(len(dup)),
                "byte_identical_to_test_images": int(dup.sha256.isin({test_sha[p] for p in test_abs}).sum()),
                "multinatsmoke_split_official": {str(k): int(v) for k, v in
                                                 dup.split_official.value_counts().items()},
                "files_on_disk": int(sum(os.path.exists(f) for f in dup_files)),
                "in_any_train_val_list": len(in_lists),
                "excluded_from_corpora_by": "scripts/v4/build_corpus.py origin_class != usable",
                "implication": "any checkpoint trained on the PUBLIC MultiNatSmoke release is "
                               "contaminated for boreal_test (ruokolahti frames sit in its official Train split)"}
            df = df[~df.parent_unit.isin(FOREST_TEST_UNITS) & ~df.index.isin(dup.index)]
        shas = set(df.sha256.dropna())
        for p in test_abs:
            if test_sha[p] in shas:
                manifest_hits.append((p, mp.name))

    within = Counter(test_sha[p] for p in test_abs)
    within_dups = sum(c - 1 for c in within.values() if c > 1)
    res = {"reference_lists": per_list,
           "reference_unique_lines": len(ref_lines),
           "reference_unique_realpaths": len(ref_real),
           "test_images": len(test_abs),
           "path_overlap": len(path_hits), "sha256_overlap": len(sha_hits),
           "manifest_nontest_sha256_overlap": len(manifest_hits),
           "within_test_duplicate_bytes": within_dups,
           "known_redistributions": redistributions,
           "path_hits": path_hits[:20], "sha_hits": sha_hits[:20],
           "manifest_hits": manifest_hits[:20],
           "result": "PASS" if not (path_hits or sha_hits or manifest_hits) else "FAIL"}
    print(f"[overlap] path_overlap={len(path_hits)} sha256_overlap={len(sha_hits)} "
          f"manifest_nontest_sha256_overlap={len(manifest_hits)} "
          f"within_test_duplicate_bytes={within_dups} -> {res['result']}", flush=True)
    assert res["result"] == "PASS", json.dumps(res, indent=1)
    return res


def sdis_test_record():
    yml = ROOT / "data/yolo/yamls/local/expB.yaml"
    test_list = None
    for ln in yml.read_text().splitlines():
        if ln.strip().startswith("test:"):
            test_list = Path(ln.split(":", 1)[1].strip())
    assert test_list is not None and test_list.exists(), test_list
    lines = [ln for ln in test_list.read_text().splitlines() if ln.strip()]
    rec = {"defined_by": str(yml), "list": str(test_list), "n": len(lines),
           "sha256": sha256_file(test_list),
           "images_touched": False,
           "note": "pyro-sdis site-disjoint TEST (ferion, croix-augas) per "
                   "data/manifests/splits_frozen.json; already frozen since iteration 1. "
                   "Only the list sha256 is recorded here; images are not copied."}
    pod = ROOT / "data/yolo/lists/pod" / test_list.name
    if pod.exists():
        rec["pod_list"] = {"path": str(pod), "sha256": sha256_file(pod),
                           "n": len([l for l in pod.read_text().splitlines() if l.strip()])}
    sf = ROOT / "data/manifests/splits_frozen.json"
    if sf.exists():
        rec["splits_frozen_sha256"] = sha256_file(sf)
    return rec


# ----------------------------------------------------------------------------
def write_report(stats: dict):
    bt, ft, ov, sd = stats["boreal_test"], stats["forest_test"], stats["overlap_check"], stats["sdis_test"]
    L = []
    L.append("# LOCKBOX V4 -- FROZEN (sealed final TEST set, iteration 4)\n")
    L.append(f"- Frozen (UTC): **{stats['frozen_utc']}**")
    L.append(f"- Built by: `{SCRIPT}` (DATA PREP ONLY -- no model run, no prediction, no metric).")
    L.append(f"- Manifest: `{stats['manifest']}`")
    L.append(f"- **{POLICY}**\n")

    L.append("## What is in the sealed set\n")
    L.append("| set | units | frames | positives | negatives | boxes | independent events |")
    L.append("|---|---|---|---|---|---|---|")
    L.append(f"| boreal_test | burn_ruokolahti (1 unit, {len(bt['per_flight'])} DJI flights) | "
             f"{bt['images']} | {bt['positives']} | {bt['negatives']} | {bt['boxes']} | **1** (ONE burn) |")
    L.append(f"| forest_test | 9 AuSmoke units in 4 groups | {ft['images']} | {ft['positives']} | "
             f"{ft['negatives']} | {ft['boxes']} | **4** (3 ground burns + 1 aerial flight) |")
    L.append(f"| sdis_test | pyro-sdis sites ferion + croix-augas | {sd['n']} | - | - | - | "
             f"2 sites (frozen since iteration 1; images not part of this package) |\n")

    L.append("### Caveats (read before interpreting any number)\n")
    L.append("- **boreal_test = ONE burn** (burn_ruokolahti). Every frame is from the same prescribed "
             "burn, split only by DJI flight; per-frame metrics are heavily autocorrelated and the "
             "effective sample size is 1 event. Boreal has only 4 independent burns in total "
             "(train 2 / val 1 / test 1) -- treat aerial test numbers as anecdotal (see "
             "aerial_split_v4.json tiny_n_warning).")
    L.append(f"- boreal_test negatives: only **{bt['negatives']}** verified-empty frames (Subset-A "
             "Empty-Images with explicit Empty-Labels). Matched-FP anchoring within boreal_test is "
             "effectively impossible; false-alert rates must come from sdis_test / other negative pools.")
    L.append("- **forest_test = 4 independent events**, of which **1 is an aerial flight** "
             f"(`{AERIAL_EVENT_GROUP}`: 3 DJI videos from ONE flight on 2023-09-22) that **includes a "
             f"staged test-run** (`{STAGED_TEST_RUN_UNIT}`, staged_test_run=true). Report the aerial "
             "event separately from the 3 ground burns and report the staged test-run unit on its own; "
             "never pool the 9 units as 9 events.")
    L.append("- forest_test negatives: **0** (AuSmoke ships positive-only frames; every frame has "
             "mask_fg_px>0). Recall-only set; matched-FP is undefined within forest_test.")
    L.append("- Group provenance (forest_holdout_v4.json): several units are the SAME burn filmed by "
             "different cameras -> counted once per group below.\n")

    L.append("### forest_test units\n")
    L.append("| unit | group | frames | boxes | staged_test_run | aerial_event |")
    L.append("|---|---|---|---|---|---|")
    for u, rec in ft["units"].items():
        L.append(f"| {u} | {rec['group']} | {rec['images']} | {rec['boxes']} | "
                 f"{'**true**' if rec['staged_test_run'] else 'false'} | "
                 f"{rec['aerial_event'] or '-'} |")
    L.append("")
    L.append("Independent events (forest_test): " + ", ".join(
        f"`{g}` ({', '.join(us)})" for g, us in ft["independent_events"].items()) + "\n")

    L.append("### boreal_test flights\n")
    L.append("| camera_or_flight_id | positives | negatives |")
    L.append("|---|---|---|")
    for fl, c in bt["per_flight"].items():
        L.append(f"| {fl} | {c.get('positives', 0)} | {c.get('negatives', 0)} |")
    L.append("")

    L.append("## Exclusions applied (same policy as the val builder)\n")
    for k, v in sorted(stats["excluded"].items()):
        L.append(f"- `{k}`: {v}")
    L.append("")

    L.append("## Leak check (train/val vs sealed test)\n")
    L.append(f"- Result: **{ov['result']}** -- path_overlap={ov['path_overlap']}, "
             f"sha256_overlap={ov['sha256_overlap']}, "
             f"manifest_nontest_sha256_overlap={ov['manifest_nontest_sha256_overlap']}, "
             f"within_test_duplicate_bytes={ov['within_test_duplicate_bytes']}")
    L.append(f"- Reference: {len(ov['reference_lists'])} list files, "
             f"{ov['reference_unique_realpaths']} unique images (sha256 of bytes computed for all), "
             f"vs {ov['test_images']} sealed test images. Lists checked:")
    for f, n in ov["reference_lists"].items():
        L.append(f"  - `{f}` ({n})")
    L.append("")
    for k, r in ov.get("known_redistributions", {}).items():
        L.append(f"- **Known re-distribution** `{k}`: {r['rows']} rows in the public MultiNatSmoke "
                 f"release are byte-identical copies of burn_ruokolahti frames "
                 f"({r['byte_identical_to_test_images']} match sealed test images; MultiNatSmoke "
                 f"official split: {r['multinatsmoke_split_official']}). They are excluded from every "
                 f"corpus by `{r['excluded_from_corpora_by']}` and verified present in "
                 f"{r['in_any_train_val_list']} train/val lists. **Implication:** {r['implication']}. "
                 f"(The same holds for burn_heinola = boreal_val.)")
    L.append("")

    L.append("## SHA256\n")
    L.append(f"- tar `{stats['tar']['path']}` ({stats['tar']['bytes']:,} bytes): `{stats['tar']['sha256']}`")
    L.append(f"  - pod extraction prefix: `{POD_ROOT}/`")
    for name in ("boreal_test", "forest_test"):
        s = stats[name]
        L.append(f"- `{s['lists']['local']['path']}` ({s['lists']['local']['n']}): `{s['lists']['local']['sha256']}`")
        L.append(f"- `{s['lists']['pod']['path']}` ({s['lists']['pod']['n']}): `{s['lists']['pod']['sha256']}`")
        L.append(f"- `{s['sha256sums']['path']}` (per-image sha256): `{s['sha256sums']['sha256']}`")
        L.append(f"- `{s['coco']}`: `{s['coco_sha256']}`")
        L.append(f"- `{s['yaml']['path']}`: `{s['yaml']['sha256']}`")
    L.append(f"- `{sd['list']}` ({sd['n']}): `{sd['sha256']}`  (sdis_test; defined by `{sd['defined_by']}`; images untouched)")
    if "pod_list" in sd:
        L.append(f"- `{sd['pod_list']['path']}` ({sd['pod_list']['n']}): `{sd['pod_list']['sha256']}`")
    if "splits_frozen_sha256" in sd:
        L.append(f"- `data/manifests/splits_frozen.json`: `{sd['splits_frozen_sha256']}`")
    L.append("")

    L.append("## Rules\n")
    L.append(f"1. {POLICY}")
    L.append("2. No script other than the one-time iteration-4 close evaluation may read the images "
             "listed in the files above. Any per-image output produced by that evaluation is "
             "HISTORICAL afterwards (same treatment as the FIgLib lockboxes v1-v3).")
    L.append("3. If any list/tar sha256 above changes, the seal is broken and the set must be re-frozen "
             "and re-declared before use.")
    L.append("4. The package layout mirrors `v4_evalval` (images/ labels/ lists/{local,pod}/test.txt, "
             "COCO json with absolute pod paths); consumers pass the list path explicitly.")
    L.append("5. Any future corpus built from `multinatsmoke` MUST keep dropping `origin_class=dup_boreal` "
             "rows (they are byte-identical Boreal copies, incl. burn_ruokolahti); any model initialised "
             "from a checkpoint trained on the public MultiNatSmoke release must be declared "
             "CONTAMINATED for boreal_test.")
    rp = ROOT / "reports/v4/LOCKBOX_V4_FROZEN.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L) + "\n")
    return str(rp)


# ----------------------------------------------------------------------------
def main():
    for name in ("boreal_test", "forest_test"):
        assert not (EVAL_ROOT / name / "lists/local" / LIST_NAME).exists(), \
            f"{name} already materialised -- refusing to re-freeze silently; remove it first"
    excl = Counter()
    per_flight = defaultdict(Counter)
    boreal = build_boreal(excl, per_flight)
    forest = build_forest(excl)

    # hard non-test bans (inverse of build_evalval.py)
    for abs_img, _, unit in boreal:
        assert unit in BOREAL_TEST_UNITS and "ruokolahti" in abs_img.lower(), abs_img
        for u in BOREAL_NONTEST_UNITS:
            assert u.replace("burn_", "") not in abs_img.lower(), f"NON-TEST LEAK: {u} in {abs_img}"
    for abs_img, _, unit in forest:
        assert unit in FOREST_TEST_UNITS, abs_img
        for u in FOREST_NONTEST_UNITS:
            assert u not in abs_img, f"NON-TEST LEAK: {u} in {abs_img}"
    assert len({u for _, _, u in forest}) == 9, "not all 9 forest test units materialised"

    test_abs = [e[0] for e in boreal + forest]
    assert len(set(test_abs)) == len(test_abs)
    print(f"[sha] hashing {len(test_abs)} sealed test images ...", flush=True)
    test_sha = sha256_many(test_abs)
    ov = overlap_check(test_abs, test_sha)

    stats = {"package": PKG, "frozen_utc": FROZEN_UTC, "script": SCRIPT, "policy": POLICY,
             "pod_root": POD_ROOT, "data_prep_only": True,
             "sources": {"boreal": "data/v4/manifests/aerial_split_v4.json[sources.boreal.test]",
                         "forest": "data/v4/manifests/forest_holdout_v4.json[test]",
                         "sdis": "data/yolo/yamls/local/expB.yaml[test]"}}
    stats["boreal_test"] = materialize("boreal_test", "boreal", boreal, test_sha)
    stats["boreal_test"].update({
        "units": sorted(BOREAL_TEST_UNITS),
        "independent_events": 1,
        "caveat": "ONE burn (burn_ruokolahti); frames split only by DJI flight; tiny-N aerial split.",
        "per_flight": {k: dict(v) for k, v in sorted(per_flight.items())},
        "label_policy": "Subset-A human YOLO boxes; Empty-Images/Empty-Labels = verified-empty negatives; "
                        "Subset-C re-encodes and Subset-B videos skipped; label_type!=box rows excluded (ambiguous)."})
    stats["forest_test"] = materialize("forest_test", "multinatsmoke", forest, test_sha)
    fu = {}
    for u in FOREST_TEST_UNITS:
        c = stats["forest_test"]["per_unit"].get(u, {})
        g = UNIT_GROUP[u]
        fu[u] = {"group": g, "images": c.get("images", 0), "positives": c.get("positives", 0),
                 "negatives": c.get("negatives", 0), "boxes": c.get("boxes", 0),
                 "staged_test_run": u == STAGED_TEST_RUN_UNIT,
                 "aerial_event": AERIAL_EVENT_GROUP if g == AERIAL_EVENT_GROUP else None}
    stats["forest_test"].update({
        "units": fu,
        "independent_events": {g: [u for u in FOREST_TEST_UNITS if UNIT_GROUP[u] == g]
                               for g in HOLD["assignment"]["test"]},
        "n_independent_events": len(HOLD["assignment"]["test"]),
        "caveat": "4 independent events = 3 ground burns + 1 aerial flight (g_dji_20230922, 3 videos, "
                  f"incl. staged test-run {STAGED_TEST_RUN_UNIT}); 0 negatives (positive-only source).",
        "label_policy": "masks -> tight boxes via build_corpus.mask_to_lines (same as corpus/val)."})
    stats["excluded"] = dict(excl)
    stats["overlap_check"] = ov
    stats["sdis_test"] = sdis_test_record()

    stage = ROOT / "data/v4/stage"
    stage.mkdir(parents=True, exist_ok=True)
    tar_path = stage / f"{PKG}.tar"
    with tarfile.open(tar_path, "w", dereference=True) as tf:
        for name in ("boreal_test", "forest_test"):
            tf.add(EVAL_ROOT / name, arcname=f"{PKG}/{name}")
            tf.add(ROOT / "data/v4/coco" / f"instances_{name}.json",
                   arcname=f"{PKG}/instances_{name}.json")
    stats["tar"] = {"path": str(tar_path), "bytes": tar_path.stat().st_size,
                    "sha256": sha256_file(tar_path),
                    "pod_extract_dir": os.path.dirname(POD_ROOT)}
    stats["manifest"] = str(EVAL_ROOT / "EVALTEST_MANIFEST.json")
    stats["report"] = write_report(stats)
    Path(stats["manifest"]).write_text(json.dumps(stats, indent=2))

    summary = {k: {kk: stats[k][kk] for kk in ("images", "positives", "negatives", "boxes")}
               for k in ("boreal_test", "forest_test")}
    summary.update({"excluded": stats["excluded"], "overlap_check": ov["result"],
                    "tar": stats["tar"], "sdis_test": {"n": stats["sdis_test"]["n"],
                                                       "sha256": stats["sdis_test"]["sha256"]},
                    "manifest": stats["manifest"], "report": stats["report"]})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
