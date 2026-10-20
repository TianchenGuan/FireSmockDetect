#!/usr/bin/env python3
"""
build_corpus.py — iteration-4 frozen YOLO training-corpus builder (transfer matrix).

Turns audited per-source manifests (data/v4/manifests/<source>_manifest.parquet,
produced by the acquisition agents) into frozen YOLO list-file corpora under
data/v4/yolo/, with dedup vs the existing corpus and eval material.

Subcommands
  dedup                          cross-source + vs-existing exact (sha256) and
                                 perceptual (phash, hamming<=6) dedup.
                                 -> data/v4/manifests/dedup_v4.parquet
  aerial-split                   parent_unit-level 60/20/20 split for aerial
                                 sources (boreal). Seeded 0, deterministic.
                                 -> data/v4/manifests/aerial_split_v4.json (+ .txt)
  build --arm X --version vN     materialize a corpus (symlinked images, real
                                 label txts, local+pod lists/yamls, stage tar,
                                 frozen CORPUS manifest).
  --self-test                    fabricate a tiny world in /tmp and run all
                                 subcommands end-to-end with verification.

Arm composition (letters after the underscore = corpus components):
  E = existing corpus (expB train list: sdis train sites + D-Fire supplement)
  G = generic / non-target smoke  (dfs + mmodalfire + firesense smoke_neg
      backgrounds + multinatsmoke usable NON-forest portions)
  T = forest/aerial target        (boreal aerial-split TRAIN units +
      multinatsmoke usable forest/Australian portions)

  B_G     = G                    (generic only)
  C_EGT   = E + G + T            (source-balanced by list-line repetition)
  D_pre   = G                    (stage-1 pretrain corpus of arm D; frozen
                                  separately from B_G for auditability)
  D_adapt = E + T                (stage-2 adapt corpus of arm D; content matches
                                  E_ET, difference is the init checkpoint)
  E_ET    = E + T

Label policy (frozen):
  * positives: label_type==box AND has_smoke_label -> YOLO txt class 0 (smoke);
    VOC xml / COCO json converted; label_type==mask -> tight bboxes from
    connected components, label_origin="mask-derived".
  * background negatives: ONLY firesense smoke-negative video frames
    (rel_path contains "smoke_neg", has_smoke_label==False) -> empty label txt.
  * EXCLUDED: video-positive frames without boxes, fire-only images, images
    with no usable label (missing smoke labels are NOT negatives), dedup bans,
    multinatsmoke origin_class != usable, boreal val/test aerial units.
    Explicit-empty box annotations (e.g. boreal Empty-Labels) are NOT used as
    negatives per policy — counted+reported so this can be revisited.

Run with the repo venv:  .venv/bin/python scripts/v4/build_corpus.py <cmd> ...
CPU only. Deterministic (seed 0 everywhere).
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import imagehash
except ImportError:  # pragma: no cover
    imagehash = None
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# ----------------------------------------------------------------------------
# Constants / policy
# ----------------------------------------------------------------------------

V4_SOURCES = ["dfs", "firesense", "mmodalfire", "boreal", "multinatsmoke"]
# priority when the same content appears in several v4 sources (keep highest)
SOURCE_PRIORITY = {"boreal": 0, "multinatsmoke": 1, "dfs": 2, "firesense": 3, "mmodalfire": 4}

# multinatsmoke sub-source dirs considered forest/Australian target domain (T)
TARGET_SUBSOURCES = {"AuSmoke", "Boreal", "Forest_Fire"}
MULTINAT_SPLIT_DIRS = {"Train", "Test", "TestSmall", "TestMedium", "TestLarge"}

AERIAL_SOURCES = ["boreal"]  # sources split by parent_unit into train/val/test

ARM_COMPOSITIONS = {
    "B_G": ("G",),
    "C_EGT": ("E", "G", "T"),
    "D_pre": ("G",),
    "D_adapt": ("E", "T"),
    "E_ET": ("E", "T"),
}
BALANCED_ARMS = {"C_EGT"}

COMPONENT_SOURCES = {
    "E": [],
    "G": ["dfs", "mmodalfire", "firesense", "multinatsmoke"],
    "T": ["boreal", "multinatsmoke"],
}

# per-source YOLO-txt class ids that mean "smoke" (default {0})
SMOKE_CLASS_IDS = {"boreal": {0}}
DEFAULT_SMOKE_IDS = {0}

PHASH_MAX_HAMMING = 6
SEED = 0

POD_V4_PREFIX = "/scratch/scratch/tianrgua-sandbox/smoke/v4data"
POD_SDIS_VAL = "/mnt/local/smoke_data/lists/pod/sdis_val.txt"
POD_SDIS_TEST = "/mnt/local/smoke_data/lists/pod/sdis_test.txt"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------


@dataclass
class Paths:
    base: Path

    @property
    def manifests(self):
        return self.base / "data/manifests"

    @property
    def v4_manifests(self):
        return self.base / "data/v4/manifests"

    @property
    def v4_raw(self):
        return self.base / "data/v4/raw"

    @property
    def v4_yolo(self):
        return self.base / "data/v4/yolo"

    @property
    def v4_stage(self):
        return self.base / "data/v4/stage"

    @property
    def cache_dir(self):
        return self.v4_manifests / "_cache"

    @property
    def dedup_parquet(self):
        return self.v4_manifests / "dedup_v4.parquet"

    @property
    def dedup_meta(self):
        return self.v4_manifests / "dedup_v4_meta.json"

    @property
    def aerial_split_json(self):
        return self.v4_manifests / "aerial_split_v4.json"

    @property
    def splits_frozen(self):
        return self.manifests / "splits_frozen.json"

    @property
    def pyro_sdis_manifest(self):
        return self.manifests / "pyro_sdis_manifest.parquet"

    @property
    def dfire_manifest(self):
        return self.manifests / "dfire_manifest.parquet"

    @property
    def figlib_manifest(self):
        return self.manifests / "figlib_subset_manifest.parquet"

    @property
    def figlib_lockbox(self):
        return self.base / "data/v3/figlib_lockbox"

    @property
    def expB_local(self):
        return self.base / "data/yolo/lists/local/expB_train.txt"

    @property
    def expB_pod(self):
        return self.base / "data/yolo/lists/pod/expB_train.txt"

    @property
    def sdis_val_local(self):
        return self.base / "data/yolo/lists/local/sdis_val.txt"

    @property
    def sdis_test_local(self):
        return self.base / "data/yolo/lists/local/sdis_test.txt"


REPO_ROOT = Path("/local/home/tianrgua/cvproject")


def log(msg):
    print(f"[build_corpus] {msg}", flush=True)


def warn(msg):
    print(f"[build_corpus][WARN] {msg}", flush=True)


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def replace_last(s, old, new):
    i = s.rfind(old)
    return s if i < 0 else s[:i] + new + s[i + len(old):]


# ----------------------------------------------------------------------------
# Parallel hashing with an on-disk cache
# ----------------------------------------------------------------------------


def _sha_worker(path):
    try:
        return (path, sha256_file(path), "")
    except Exception as e:  # noqa: BLE001
        return (path, "", f"{type(e).__name__}: {e}")


def _phash_worker(path):
    if imagehash is None:
        return (path, "", "imagehash not installed")
    try:
        with Image.open(path) as im:
            h = str(imagehash.phash(im))
        return (path, h, "")
    except Exception as e:  # noqa: BLE001
        return (path, "", f"{type(e).__name__}: {e}")


class HashCache:
    """Cache of sha256/phash keyed by (path, size, mtime_ns)."""

    def __init__(self, path: Path, workers=None):
        self.path = path
        self.workers = workers or min(64, os.cpu_count() or 8)
        self._d = {}
        if path.exists():
            try:
                df = pd.read_parquet(path)
                for r in df.itertuples(index=False):
                    self._d[r.path] = {
                        "size": int(r.size), "mtime_ns": int(r.mtime_ns),
                        "sha256": r.sha256 or "", "phash": r.phash or "",
                    }
            except Exception as e:  # noqa: BLE001
                warn(f"hash cache unreadable ({e}); starting fresh")

    def _fresh(self, p, st):
        e = self._d.get(p)
        return e if e and e["size"] == st.st_size and e["mtime_ns"] == st.st_mtime_ns else None

    def get_many(self, paths, kind):
        """kind in {'sha256','phash'} -> dict path -> value ('' on error)."""
        assert kind in ("sha256", "phash")
        worker = _sha_worker if kind == "sha256" else _phash_worker
        out, todo, stats = {}, [], {}
        for p in paths:
            p = str(p)
            try:
                st = os.stat(p)
            except OSError:
                out[p] = ""
                continue
            stats[p] = st
            e = self._fresh(p, st)
            if e and e[kind]:
                out[p] = e[kind]
            else:
                todo.append(p)
        if todo:
            log(f"computing {kind} for {len(todo)} files ({self.workers} workers)...")
            t0 = time.time()
            ctx = get_context("fork")
            errs = 0
            with ctx.Pool(self.workers) as pool:
                for i, (p, v, err) in enumerate(
                    pool.imap_unordered(worker, todo, chunksize=32)
                ):
                    out[p] = v
                    if err:
                        errs += 1
                        if errs <= 5:
                            warn(f"{kind} failed for {p}: {err}")
                    else:
                        st = stats[p]
                        e = self._d.setdefault(
                            p, {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                                "sha256": "", "phash": ""}
                        )
                        e["size"], e["mtime_ns"] = st.st_size, st.st_mtime_ns
                        e[kind] = v
                    if (i + 1) % 10000 == 0:
                        log(f"  {kind}: {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)")
            if errs:
                warn(f"{kind}: {errs} files failed (treated as unreadable)")
            self.save()
        return out

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"path": p, "size": e["size"], "mtime_ns": e["mtime_ns"],
             "sha256": e["sha256"], "phash": e["phash"]}
            for p, e in self._d.items()
        ]
        pd.DataFrame(rows, columns=["path", "size", "mtime_ns", "sha256", "phash"]).to_parquet(
            self.path, index=False
        )


# ----------------------------------------------------------------------------
# phash hamming matching
# ----------------------------------------------------------------------------

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    v = np.ascontiguousarray(x).view(np.uint8).reshape(x.shape + (8,))
    return _POP[v].sum(axis=-1)


def hexes_to_u64(hexes):
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64)


def phash_hits(q, r, maxd=PHASH_MAX_HAMMING):
    """q, r: np.uint64 arrays. Returns list of (qi, ri, dist)."""
    hits = []
    if len(q) == 0 or len(r) == 0:
        return hits
    chunk = max(1, int(4_000_000 // max(1, len(r))))
    for s in range(0, len(q), chunk):
        blk = q[s:s + chunk, None] ^ r[None, :]
        d = _popcount64(blk)
        qi, ri = np.nonzero(d <= maxd)
        for a, b in zip(qi, ri):
            hits.append((s + int(a), int(b), int(d[a, b])))
    return hits


# ----------------------------------------------------------------------------
# v4 manifests
# ----------------------------------------------------------------------------

REQUIRED_COLS = [
    "source", "rel_path", "sha256", "parent_unit", "label_type",
    "has_smoke_label", "has_fire_label",
]


def resolve_v4_path(P: Paths, source, rel_path):
    for cand in (P.v4_raw / source / rel_path, P.base / rel_path):
        if cand.exists():
            return cand
    return P.v4_raw / source / rel_path  # canonical location (may be missing)


def load_v4_manifests(P: Paths, quiet=False):
    out = {}
    for s in V4_SOURCES:
        f = P.v4_manifests / f"{s}_manifest.parquet"
        if not f.exists():
            if not quiet:
                log(f"source '{s}': manifest missing ({f}) — skipping")
            continue
        df = pd.read_parquet(f)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            warn(f"source '{s}': manifest missing columns {missing} — SKIPPING source")
            continue
        df = df.reset_index(drop=True).copy()
        df["rel_path"] = df["rel_path"].astype(str)
        df["abs_path"] = [str(resolve_v4_path(P, s, rp)) for rp in df["rel_path"]]
        if "origin_class" not in df.columns:
            df["origin_class"] = "usable"
        df["origin_class"] = df["origin_class"].fillna("usable")
        if "n_smoke_boxes" not in df.columns:
            df["n_smoke_boxes"] = -1
        out[s] = df
        if not quiet:
            log(f"source '{s}': {len(df)} rows loaded")
    return out


def multinat_subsource(rel_path, origin=None):
    """Sub-dataset dir of a multinatsmoke image, e.g. AuSmoke / WSDataset."""
    known = TARGET_SUBSOURCES | {
        "aiformankind", "SmokeSeg", "WSDataset", "FIgLib", "FESB-MLID",
        "FireSpot", "firecam", "D-Fire", "SMOKE5K_real",
    }
    if origin and isinstance(origin, str):
        for k in known:
            if k.lower() in origin.lower():
                return k
    parts = Path(rel_path).parts
    for i, p in enumerate(parts):
        if p in MULTINAT_SPLIT_DIRS and i + 1 < len(parts):
            return parts[i + 1]
    for p in parts:
        if p in known:
            return p
    return None


# verified-empty policy: an image whose annotation file EXISTS and contains
# zero boxes is an author-verified negative (unlike a missing label, which is
# unknown). Off by default; enabled per-build via --verified-empty-negatives
# and recorded in the frozen manifest.
VERIFIED_EMPTY_AS_NEG = False

# AuSmoke forest holdout (data/v4/manifests/forest_holdout_v4.json): units
# reserved for the fresh forest new-region val/test — never enter training
# corpora. Loaded in cmd_build when the file exists.
FOREST_HOLDOUT_UNITS: set = set()


def train_eligibility(source, rel_path, row):
    """Return (kind, exclusion_reason). kind in {pos_box, pos_mask, neg_background, None}."""
    if os.path.splitext(rel_path)[1].lower() not in IMG_EXTS:
        return None, "non_image_file"  # e.g. firesense .avi video rows
    lt = str(row.label_type)
    hs = bool(row.has_smoke_label)
    hf = bool(row.has_fire_label)
    if source == "firesense" and "smoke_neg" in rel_path and not hs:
        return "neg_background", ""
    if lt == "box" and hs:
        return "pos_box", ""
    if lt == "mask" and hs:
        return "pos_mask", ""
    if hs:  # videolabel / none with smoke: no per-image boxes -> excluded
        return None, f"smoke_pos_without_boxes(label_type={lt})"
    if hf:
        return None, "fire_only"
    if lt == "box":
        if VERIFIED_EMPTY_AS_NEG:
            return "neg_background", ""
        return None, "explicit_empty_box_not_used(policy)"
    return None, "no_usable_label"


# ----------------------------------------------------------------------------
# existing-corpus / eval material
# ----------------------------------------------------------------------------


def load_eval_material(P: Paths):
    """Paths + descriptions of existing material.

    Returns dict with:
      eval_paths:  [(abs_path, desc)]  — sdis val/test sites + FIgLib (subset+lockbox)
      train_paths: [(abs_path, desc)]  — sdis train sites + D-Fire (all splits)
      notes: list of strings
    """
    notes = []
    eval_paths, train_paths = [], []

    splits = json.loads(P.splits_frozen.read_text())
    site_split = splits["site_split"]
    val_sites, test_sites = set(site_split["val"]), set(site_split["test"])

    sdis = pd.read_parquet(P.pyro_sdis_manifest)
    for r in sdis.itertuples(index=False):
        p = r.image_path
        ap = p if os.path.isabs(p) else str(P.base / p)
        if r.site in val_sites:
            eval_paths.append((ap, f"sdis_val:{r.site}"))
        elif r.site in test_sites:
            eval_paths.append((ap, f"sdis_test:{r.site}"))
        else:
            train_paths.append((ap, f"sdis_train:{r.site}"))

    dfire = pd.read_parquet(P.dfire_manifest)
    for r in dfire.itertuples(index=False):
        p = r.image_path
        ap = p if os.path.isabs(p) else str(P.base / p)
        train_paths.append((ap, f"dfire:{getattr(r, 'split_official', '?')}"))

    # FIgLib: iteration-1 subset manifest frames + opened lockbox frames (all local)
    n_fig = 0
    if P.figlib_manifest.exists():
        fig = pd.read_parquet(P.figlib_manifest)
        col = "frame_path" if "frame_path" in fig.columns else None
        if col:
            for p in fig[col].astype(str):
                ap = p if os.path.isabs(p) else str(P.base / p)
                if os.path.exists(ap):
                    eval_paths.append((ap, "figlib_subset"))
                    n_fig += 1
        else:
            notes.append("figlib_subset_manifest has no frame_path column — skipped")
    else:
        notes.append("figlib_subset_manifest.parquet missing — FIgLib subset NOT checked")
    n_lock = 0
    if P.figlib_lockbox.exists():
        for f in P.figlib_lockbox.rglob("*.jpg"):
            if "_tars" in f.parts:
                continue
            eval_paths.append((str(f), "figlib_lockbox"))
            n_lock += 1
    else:
        notes.append("figlib_lockbox dir missing — lockbox frames NOT checked")
    if n_fig + n_lock == 0:
        notes.append("NO local FIgLib frames found — FIgLib dedup SKIPPED entirely")
    else:
        notes.append(f"FIgLib frames checked: subset={n_fig}, lockbox={n_lock}")

    # drop non-existent files
    def keep_existing(pairs, name):
        ok = [(p, d) for p, d in pairs if os.path.exists(p)]
        if len(ok) != len(pairs):
            notes.append(f"{name}: {len(pairs) - len(ok)} listed files missing on disk")
        return ok

    eval_paths = keep_existing(eval_paths, "eval material")
    train_paths = keep_existing(train_paths, "train material")
    return {"eval_paths": eval_paths, "train_paths": train_paths, "notes": notes}


# ----------------------------------------------------------------------------
# dedup subcommand
# ----------------------------------------------------------------------------


def cmd_dedup(P: Paths, args):
    if imagehash is None:
        raise SystemExit("imagehash not available in this environment")
    mans = load_v4_manifests(P)
    if not mans:
        log("no v4 source manifests present yet — nothing to dedup (exit 0)")
        return 0

    cache = HashCache(P.cache_dir / "hash_cache.parquet")
    em = load_eval_material(P)
    for n in em["notes"]:
        log(f"eval-material note: {n}")

    eval_files = em["eval_paths"]
    train_files = em["train_paths"]
    eval_sha = {}
    train_sha = {}
    sha_eval = cache.get_many([p for p, _ in eval_files], "sha256")
    sha_train = cache.get_many([p for p, _ in train_files], "sha256")
    for p, d in eval_files:
        if sha_eval.get(p):
            eval_sha[sha_eval[p]] = f"{d}:{os.path.basename(p)}"
    for p, d in train_files:
        s = sha_train.get(p)
        if s and s not in eval_sha:
            train_sha[s] = f"{d}:{os.path.basename(p)}"

    # ---- per-row state -------------------------------------------------
    # rows: list of dicts (mutable verdicts)
    rows = []
    for src, df in mans.items():
        for r in df.itertuples(index=False):
            kind, _ = train_eligibility(src, r.rel_path, r)
            rows.append({
                "source": src, "rel_path": r.rel_path, "sha256": str(r.sha256),
                "parent_unit": "" if pd.isna(r.parent_unit) else str(r.parent_unit),
                "abs_path": r.abs_path,
                "origin_class": str(r.origin_class),
                "train_eligible": kind is not None,
                "phash_checked": False,
                "verdict": "keep", "reason": "",
            })

    def ban(row, reason):
        if row["verdict"] == "keep":
            row["verdict"] = "ban"
            row["reason"] = reason

    evalhit_units = {}  # (source, parent_unit) -> trigger description

    # 1) origin_class quarantine/dup (multinatsmoke)
    for row in rows:
        if row["origin_class"] != "usable":
            ban(row, f"origin_class:{row['origin_class']}")

    # 2) missing files
    for row in rows:
        if row["verdict"] == "keep" and not os.path.exists(row["abs_path"]):
            ban(row, "missing_file")

    # 3) exact sha vs eval material (ALL rows, even already-banned ones can
    #    trigger a group ban), then vs train material
    for row in rows:
        s = row["sha256"]
        if s in eval_sha:
            trig = f"exact_dup_eval:{eval_sha[s]}"
            ban(row, trig)
            if row["parent_unit"]:
                evalhit_units.setdefault(
                    (row["source"], row["parent_unit"]),
                    f"{trig} via {row['rel_path']}",
                )
        elif s in train_sha and row["verdict"] == "keep":
            ban(row, f"exact_dup_train_corpus:{train_sha[s]}")

    # 4) exact sha within v4 (across AND within sources): keep highest-priority
    by_sha = defaultdict(list)
    for row in rows:
        if row["verdict"] == "keep":
            by_sha[row["sha256"]].append(row)
    for s, grp in by_sha.items():
        if len(grp) < 2:
            continue
        grp.sort(key=lambda r: (SOURCE_PRIORITY.get(r["source"], 99), r["rel_path"]))
        keeper = grp[0]
        for r in grp[1:]:
            ban(r, f"exact_dup_v4:{keeper['source']}/{keeper['rel_path']}")

    # 5) perceptual near-dup — restricted to rows that could enter training
    #    (others cannot contaminate; exact-sha above still covered every row).
    cand = [r for r in rows if r["verdict"] == "keep" and r["train_eligible"]]
    for r in cand:
        r["phash_checked"] = True
    ph_cand = cache.get_many([r["abs_path"] for r in cand], "phash")
    for r in cand:
        if not ph_cand.get(r["abs_path"]):
            ban(r, "unreadable_image")
    cand = [r for r in cand if r["verdict"] == "keep"]

    ph_eval = cache.get_many([p for p, _ in eval_files], "phash")
    eval_ok = [(p, d) for p, d in eval_files if ph_eval.get(p)]
    if eval_ok:
        e_arr = hexes_to_u64([ph_eval[p] for p, _ in eval_ok])
        c_arr = hexes_to_u64([ph_cand[r["abs_path"]] for r in cand])
        log(f"phash matching: {len(cand)} v4 candidates vs {len(eval_ok)} eval frames ...")
        for qi, ri, d in phash_hits(c_arr, e_arr):
            row = cand[qi]
            ep, ed = eval_ok[ri]
            trig = f"neardup_eval:{ed}:{os.path.basename(ep)}:h={d}"
            ban(row, trig)
            if row["parent_unit"]:
                evalhit_units.setdefault(
                    (row["source"], row["parent_unit"]),
                    f"{trig} via {row['rel_path']}",
                )
    cand = [r for r in cand if r["verdict"] == "keep"]

    # 6) cross-source near-dup within v4 (lower priority source banned)
    srcs_sorted = sorted({r["source"] for r in cand}, key=lambda s: SOURCE_PRIORITY.get(s, 99))
    by_src = {s: [r for r in cand if r["source"] == s] for s in srcs_sorted}
    for i, lo in enumerate(srcs_sorted):
        hi_rows = [r for s in srcs_sorted[:i] for r in by_src[s] if r["verdict"] == "keep"]
        lo_rows = [r for r in by_src[lo] if r["verdict"] == "keep"]
        if not hi_rows or not lo_rows:
            continue
        lo_arr = hexes_to_u64([ph_cand[r["abs_path"]] for r in lo_rows])
        hi_arr = hexes_to_u64([ph_cand[r["abs_path"]] for r in hi_rows])
        log(f"phash cross-source: {lo} ({len(lo_rows)}) vs higher-priority ({len(hi_rows)}) ...")
        for qi, ri, d in phash_hits(lo_arr, hi_arr):
            hr = hi_rows[ri]
            ban(lo_rows[qi], f"neardup_v4:{hr['source']}/{hr['rel_path']}:h={d}")

    # 7) group-aware eval bans: whole parent_unit goes when any frame matched eval
    n_group = 0
    for row in rows:
        key = (row["source"], row["parent_unit"])
        if row["parent_unit"] and key in evalhit_units and row["verdict"] == "keep":
            ban(row, f"group_ban_eval:unit={row['parent_unit']} ({evalhit_units[key]})")
            n_group += 1

    # ---- write output ---------------------------------------------------
    P.v4_manifests.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)[
        ["sha256", "rel_path", "source", "verdict", "reason",
         "parent_unit", "train_eligible", "phash_checked"]
    ]
    out.to_parquet(P.dedup_parquet, index=False)

    def reason_cat(r):
        return r.split(":", 1)[0] if r else ""

    summary = {"created_utc": utcnow(), "sources": {}, "notes": em["notes"],
               "eval_frames_checked": len(eval_ok) if eval_ok else 0,
               "train_material_hashed": len(train_files),
               "group_banned_rows": n_group,
               "evalhit_units": {f"{k[0]}/{k[1]}": v for k, v in evalhit_units.items()},
               "phash_policy": "phash restricted to train-eligible keep rows; "
                               "exact sha256 checked for ALL rows"}
    print("\n=== DEDUP SUMMARY ===")
    for src, df in mans.items():
        sub = out[out["source"] == src]
        cats = Counter(reason_cat(r) for r in sub[sub.verdict == "ban"]["reason"])
        summary["sources"][src] = {
            "rows": len(sub), "keep": int((sub.verdict == "keep").sum()),
            "ban": int((sub.verdict == "ban").sum()),
            "ban_reasons": dict(cats),
        }
        print(f"  {src:15s} rows={len(sub):7d} keep={summary['sources'][src]['keep']:7d} "
              f"ban={summary['sources'][src]['ban']:7d}  {dict(cats)}")
    print(f"  eval-hit parent_units group-banned: {len(evalhit_units)} "
          f"(+{n_group} extra rows)")
    for n in em["notes"]:
        print(f"  note: {n}")
    P.dedup_meta.write_text(json.dumps(summary, indent=2))
    log(f"wrote {P.dedup_parquet} and {P.dedup_meta}")
    return 0


# ----------------------------------------------------------------------------
# aerial-split subcommand
# ----------------------------------------------------------------------------


def cmd_aerial_split(P: Paths, args):
    mans = load_v4_manifests(P)
    present = [s for s in AERIAL_SOURCES if s in mans]
    if not present:
        log(f"no aerial-source manifests present yet ({AERIAL_SOURCES}) — nothing to split")
        return 0

    result = {"created_utc": utcnow(), "seed": SEED, "ratio": [0.6, 0.2, 0.2],
              "policy": "split by ENTIRE parent_unit, greedy on unit count (not frames)",
              "sources": {}}
    lines = ["aerial_split_v4 — parent_unit assignment (seed 0)", ""]

    for src in present:
        df = mans[src]
        units = sorted(u for u in df["parent_unit"].astype(str).unique() if u and u != "nan")
        frames = df.groupby(df["parent_unit"].astype(str)).size().to_dict()
        n = len(units)
        rnd = random.Random(SEED)
        shuffled = list(units)
        rnd.shuffle(shuffled)
        tiny = n < 5
        if n == 0:
            warn(f"{src}: no parent_units in manifest — cannot split")
            continue
        if n == 1:
            warn(f"{src}: ONLY 1 unit — assigning to train; NO aerial val/test possible!")
            tr, va, te = shuffled, [], []
        elif n == 2:
            tr, va, te = shuffled[:1], shuffled[1:2], []
            warn(f"{src}: only 2 units — 1 train / 1 val / 0 test")
        else:
            n_val = max(1, round(0.2 * n))
            n_test = max(1, round(0.2 * n))
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train, n_val, n_test = 1, max(1, (n - 1) // 2), n - 1 - max(1, (n - 1) // 2)
            tr, va, te = (shuffled[:n_train],
                          shuffled[n_train:n_train + n_val],
                          shuffled[n_train + n_val:])
        if tiny:
            msg = (f"{src}: TINY-N LIMITATION — only {n} independent parent_units; "
                   f"60/20/20 by unit count is statistically weak. >=1 unit forced "
                   f"into val and test where possible. Treat aerial val/test numbers "
                   f"as anecdotal.")
            warn(msg.upper())
            result["sources"].setdefault(src, {})["tiny_n_warning"] = msg
        d = result["sources"].setdefault(src, {})
        d.update({
            "n_units": n, "train": sorted(tr), "val": sorted(va), "test": sorted(te),
            "frames_per_unit": {u: int(frames.get(u, 0)) for u in units},
        })
        lines.append(f"source: {src}  ({n} units)")
        lines.append(f"  {'unit':30s} {'frames':>7s}  assignment")
        for u in units:
            a = "train" if u in tr else ("val" if u in va else "test")
            lines.append(f"  {u:30s} {frames.get(u, 0):7d}  {a}")
        for split_name, us in (("train", tr), ("val", va), ("test", te)):
            nf = sum(frames.get(u, 0) for u in us)
            lines.append(f"  -> {split_name}: {len(us)} units, {nf} frames")
        lines.append("")

    P.v4_manifests.mkdir(parents=True, exist_ok=True)
    P.aerial_split_json.write_text(json.dumps(result, indent=2))
    table = "\n".join(lines) + "\n"
    (P.v4_manifests / "aerial_split_v4.txt").write_text(table)
    print(table)
    log(f"wrote {P.aerial_split_json} (+ .txt table)")
    return 0


# ----------------------------------------------------------------------------
# annotation converters
# ----------------------------------------------------------------------------


def clamp01(v):
    return min(1.0, max(0.0, v))


def yolo_txt_to_lines(txt_path, smoke_ids):
    lines, dropped = [], 0
    for raw in Path(txt_path).read_text().splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            dropped += 1
            continue
        if cls not in smoke_ids:
            dropped += 1
            continue
        if not (-0.05 <= cx <= 1.05 and -0.05 <= cy <= 1.05) or w <= 0 or h <= 0:
            dropped += 1
            continue
        # clip the box (not just the centre) into [0,1]
        x1, y1 = clamp01(cx - w / 2), clamp01(cy - h / 2)
        x2, y2 = clamp01(cx + w / 2), clamp01(cy + h / 2)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            dropped += 1
            continue
        lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return lines, dropped


def _img_size(img_path):
    with Image.open(img_path) as im:
        return im.size  # (W, H)


def voc_xml_to_lines(xml_path, img_path):
    root = ET.parse(xml_path).getroot()
    try:
        W = int(float(root.findtext("size/width") or 0))
        H = int(float(root.findtext("size/height") or 0))
    except ValueError:
        W = H = 0
    if W <= 0 or H <= 0:
        W, H = _img_size(img_path)
    lines = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if "smoke" not in name:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            xmin = float(bb.findtext("xmin"))
            ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax"))
            ymax = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        xmin, xmax = sorted((max(0.0, xmin), min(float(W), xmax)))
        ymin, ymax = sorted((max(0.0, ymin), min(float(H), ymax)))
        if xmax - xmin < 2 or ymax - ymin < 2:
            continue
        lines.append(
            f"0 {(xmin + xmax) / 2 / W:.6f} {(ymin + ymax) / 2 / H:.6f} "
            f"{(xmax - xmin) / W:.6f} {(ymax - ymin) / H:.6f}"
        )
    return lines


def coco_boxes_to_lines(entry, img_path):
    W, H, boxes = entry
    if not W or not H:
        W, H = _img_size(img_path)
    lines = []
    for x, y, w, h in boxes:
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(W), x + w), min(float(H), y + h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(
            f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
            f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
        )
    return lines


MASK_MERGE_THRESHOLD = 8   # if more raw components than this, close-merge fragments
MASK_MAX_BOXES = 50        # keep at most this many boxes (largest by area)


def mask_to_lines(mask_path, img_path):
    if cv2 is not None:
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        m = None
    if m is None:
        with Image.open(mask_path) as im:
            m = np.array(im.convert("L"))
    H, W = m.shape[:2]
    min_area = max(9, int(2e-5 * W * H))
    bw = (m >= 128).astype(np.uint8)
    if bw.sum() == 0:
        return []
    if cv2 is not None:
        n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        comps = [tuple(int(v) for v in stats[i][:5]) for i in range(1, n)]
        if len(comps) > MASK_MERGE_THRESHOLD:
            # fragmented segmentation (some multinatsmoke masks have 100s of
            # specks) -> morphological closing to merge nearby fragments
            k = max(3, int(0.01 * (W * W + H * H) ** 0.5) | 1)
            bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
            n, _, stats, _ = cv2.connectedComponentsWithStats(bw2, connectivity=8)
            comps = [tuple(int(v) for v in stats[i][:5]) for i in range(1, n)]
    else:  # crude fallback: single tight box over all foreground
        ys, xs = np.nonzero(bw)
        comps = [(int(xs.min()), int(ys.min()),
                  int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1),
                  int(bw.sum()))]
    comps = [c for c in comps if c[4] >= min_area and c[2] >= 2 and c[3] >= 2]
    comps.sort(key=lambda c: -c[4])
    lines = []
    for x, y, w, h, _area in comps[:MASK_MAX_BOXES]:
        lines.append(
            f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
        )
    return lines


class LabelResolver:
    """Find the annotation for an image inside one v4 source's raw tree."""

    SKIP_DIRS = {"_packages", "audit_sheets", "__pycache__"}

    def __init__(self, source, root: Path):
        self.source = source
        self.root = Path(root)
        self.smoke_ids = SMOKE_CLASS_IDS.get(source, DEFAULT_SMOKE_IDS)
        self._idx = {}     # ext -> stem -> [paths]
        self._coco = None  # basename -> (W, H, [xywh])

    def _index(self, ext):
        if ext in self._idx:
            return self._idx[ext]
        idx = defaultdict(list)
        if self.root.exists():
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = [d for d in dirnames
                               if d not in self.SKIP_DIRS and not d.startswith("_")]
                for fn in filenames:
                    if fn.lower().endswith(ext):
                        p = Path(dirpath) / fn
                        idx[p.stem].append(p)
        self._idx[ext] = idx
        return idx

    @staticmethod
    def _closest(cands, img_path):
        s = str(img_path)
        return max(cands, key=lambda c: (len(os.path.commonprefix([str(c), s])), -len(str(c))))

    def _coco_index(self):
        if self._coco is not None:
            return self._coco
        self._coco = {}
        if not self.root.exists():
            return self._coco
        for jf in self.root.rglob("*.json"):
            if jf.name.startswith("_") or any(d in jf.parts for d in self.SKIP_DIRS):
                continue
            try:
                if jf.stat().st_size > 200 * 1024 * 1024:
                    continue
                j = json.loads(jf.read_text())
            except Exception:  # noqa: BLE001
                continue
            if not (isinstance(j, dict) and "images" in j and "annotations" in j):
                continue
            smoke_cats = {c["id"] for c in j.get("categories", [])
                          if "smoke" in str(c.get("name", "")).lower()}
            imgs = {im["id"]: (im.get("file_name", ""), im.get("width"), im.get("height"))
                    for im in j["images"]}
            per_img = defaultdict(list)
            for a in j["annotations"]:
                if smoke_cats and a.get("category_id") not in smoke_cats:
                    continue
                bb = a.get("bbox")
                if bb and len(bb) == 4:
                    per_img[a["image_id"]].append(tuple(float(v) for v in bb))
            for iid, (fn, w, h) in imgs.items():
                if fn:
                    self._coco[os.path.basename(fn)] = (w, h, per_img.get(iid, []))
        return self._coco

    def find_box_label(self, img: Path):
        """Return (kind, ref) where kind in {yolo, voc, coco}; None if not found."""
        s = str(img)
        cands_txt = [img.with_suffix(".txt")]
        cands_xml = [img.with_suffix(".xml")]
        if "/images/" in s:
            cands_txt.append(Path(replace_last(s, "/images/", "/labels/")).with_suffix(".txt"))
            for ad in ("/annotations/", "/Annotations/", "/labels/"):
                cands_xml.append(Path(replace_last(s, "/images/", ad)).with_suffix(".xml"))
        for c in cands_txt:
            if c.exists():
                return ("yolo", c)
        for c in cands_xml:
            if c.exists():
                return ("voc", c)
        hits = self._index(".txt").get(img.stem)
        if hits:
            return ("yolo", self._closest(hits, img))
        hits = self._index(".xml").get(img.stem)
        if hits:
            return ("voc", self._closest(hits, img))
        entry = self._coco_index().get(img.name)
        if entry:
            return ("coco", entry)
        return None

    def find_mask(self, img: Path):
        s = str(img)
        exts = [".png", ".jpg", ".jpeg", ".bmp", img.suffix]
        if "/images/" in s:
            base = Path(replace_last(s, "/images/", "/masks/"))
            for ext in exts:
                for stem in (base.stem, base.stem + "_mask"):
                    c = base.with_name(stem + ext)
                    if c.exists():
                        return c
        for ext in (".png", ".jpg"):
            hits = [p for p in self._index(ext).get(img.stem, [])
                    if any(q.lower() == "masks" for q in p.parts)]
            if hits:
                return self._closest(hits, img)
        return None


def convert_row(resolver: LabelResolver, kind, img_abs: Path, n_expected,
                mask_hint=None):
    """kind in {pos_box, pos_mask}. Returns (lines, label_origin, ref, err)."""
    if kind == "pos_mask":
        mp = mask_hint if (mask_hint and Path(mask_hint).exists()) else resolver.find_mask(img_abs)
        if mp is None:
            return None, "", "", "mask_not_found"
        try:
            lines = mask_to_lines(mp, img_abs)
        except Exception as e:  # noqa: BLE001
            return None, "", str(mp), f"mask_error:{type(e).__name__}"
        if not lines:
            return None, "", str(mp), "mask_empty"
        # no n_expected cross-check for masks: manifest counts raw components,
        # conversion merges/filters fragments by design
        return lines, "mask-derived", str(mp), ""
    found = resolver.find_box_label(img_abs)
    if found is None:
        return None, "", "", "box_label_not_found"
    fmt, ref = found
    try:
        if fmt == "yolo":
            lines, _ = yolo_txt_to_lines(ref, resolver.smoke_ids)
        elif fmt == "voc":
            lines = voc_xml_to_lines(ref, img_abs)
        else:
            lines = coco_boxes_to_lines(ref, img_abs)
            ref = "coco-index"
    except Exception as e:  # noqa: BLE001
        return None, "", str(ref), f"{fmt}_error:{type(e).__name__}"
    if not lines:
        return None, "", str(ref), f"{fmt}_no_smoke_boxes"
    mismatch = ""
    if n_expected is not None and n_expected >= 0 and len(lines) != n_expected:
        mismatch = f"box_count_mismatch(manifest={n_expected},converted={len(lines)})"
    return lines, f"box-{fmt}", str(ref), mismatch


# ----------------------------------------------------------------------------
# build subcommand
# ----------------------------------------------------------------------------


@dataclass
class Entry:
    component: str      # E / G / T
    source: str         # v4 source or "existing"
    local_line: str     # absolute local image path
    pod_line: str       # pod image path
    label_lines: list = None   # None for E (labels live beside originals)
    label_origin: str = ""     # box-yolo / box-voc / box-coco / mask-derived / background_neg / existing
    link_name: str = ""        # corpus images/<link_name>


def flat_name(source, rel_path):
    name = f"{source}__" + rel_path.replace("/", "__").replace(" ", "_")
    if len(name.encode()) > 200:
        stem, ext = os.path.splitext(name)
        h = sha256_bytes(rel_path.encode())[:12]
        name = stem[:150] + "__" + h + ext
    return name


def verify_conversions(entries, n=5):
    """Print box-vs-image-size sanity for up to n conversions per (source, origin)."""
    groups = defaultdict(list)
    for e in entries:
        if e.label_lines:
            groups[(e.source, e.label_origin)].append(e)
    print("\n=== CONVERSION VERIFICATION (box coords vs image size) ===")
    for (src, origin), es in sorted(groups.items()):
        for e in es[:n]:
            W, H = _img_size(e.local_line)
            print(f"  [{src}/{origin}] {os.path.basename(e.local_line)}  img={W}x{H}")
            for ln in e.label_lines[:4]:
                c, cx, cy, w, h = ln.split()
                cx, cy, w, h = float(cx), float(cy), float(w), float(h)
                x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
                x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
                ok = (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1
                      and -1e-6 <= x1 < x2 <= W + 1e-6 and -1e-6 <= y1 < y2 <= H + 1e-6)
                print(f"      cls={c} cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f} "
                      f"-> px[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] "
                      f"{'OK' if ok else '!! OUT OF BOUNDS !!'}")
                if not ok:
                    raise RuntimeError(
                        f"conversion sanity FAILED for {e.local_line}: {ln}")
    print("=== verification done ===\n")


def parse_label_file(path):
    """Validate a YOLO label txt; returns number of boxes. Raises on bad content."""
    n = 0
    txt = Path(path).read_text()
    for ln in txt.splitlines():
        if not ln.strip():
            continue
        parts = ln.split()
        assert len(parts) == 5, f"{path}: bad line {ln!r}"
        assert parts[0] == "0", f"{path}: class != 0 in {ln!r}"
        vals = [float(v) for v in parts[1:]]
        assert all(0.0 <= v <= 1.0 for v in vals), f"{path}: out-of-range {ln!r}"
        assert vals[2] > 0 and vals[3] > 0, f"{path}: empty box {ln!r}"
        n += 1
    return n


def load_expB(P: Paths):
    splits = json.loads(P.splits_frozen.read_text())
    rec = splits.get("lists", {}).get("expB_train", {})
    local_lines = [l for l in P.expB_local.read_text().splitlines() if l.strip()]
    pod_lines = [l for l in P.expB_pod.read_text().splitlines() if l.strip()]
    if rec.get("n") is not None and rec["n"] != len(local_lines):
        raise SystemExit(
            f"expB integrity FAILURE: splits_frozen records n={rec['n']} but "
            f"{P.expB_local} has {len(local_lines)} lines")
    if len(local_lines) != len(pod_lines):
        raise SystemExit("expB local/pod list length mismatch")
    for a, b in zip(local_lines, pod_lines):
        if os.path.basename(a) != os.path.basename(b):
            raise SystemExit(f"expB local/pod basename mismatch: {a} vs {b}")
    file_sha = sha256_file(P.expB_local)
    if rec.get("sha256") and rec["sha256"] not in (
        file_sha, sha256_bytes("\n".join(local_lines).encode()),
        sha256_bytes(("\n".join(local_lines) + "\n").encode()),
    ):
        warn(f"expB list sha256 does not match splits_frozen record "
             f"(file={file_sha[:12]}..., recorded={rec['sha256'][:12]}...) — "
             f"line count matches; proceeding but INSPECT THIS")
    return local_lines, pod_lines


_CONV_RESOLVERS = None  # set before forking the conversion pool


def _conv_worker(job):
    idx, src, kind, img, n_exp, hint = job
    res = _CONV_RESOLVERS[src]
    return idx, convert_row(res, kind, Path(img), n_exp, hint)


def cmd_build(P: Paths, args):
    global VERIFIED_EMPTY_AS_NEG, FOREST_HOLDOUT_UNITS
    VERIFIED_EMPTY_AS_NEG = bool(getattr(args, "verified_empty_negatives", False))
    fh = P.v4_manifests / "forest_holdout_v4.json"
    if fh.exists():
        d = json.loads(fh.read_text())
        FOREST_HOLDOUT_UNITS = set(d["val"]) | set(d["test"])
        log(f"forest holdout loaded: {len(FOREST_HOLDOUT_UNITS)} units banned "
            f"from training")
    arm, version = args.arm, args.version
    comp = ARM_COMPOSITIONS[arm]
    corpus_name = f"{arm}_{version}"
    corpus_dir = P.v4_yolo / corpus_name
    frozen_path = P.v4_manifests / f"CORPUS_{corpus_name}.json"
    if frozen_path.exists() and not args.force:
        raise SystemExit(f"{frozen_path} already exists (FROZEN). Use --force to rebuild.")

    # --- prerequisites ----------------------------------------------------
    if not P.dedup_parquet.exists():
        raise SystemExit("dedup output missing — run `build_corpus.py dedup` first")
    dedup = pd.read_parquet(P.dedup_parquet)
    verdict = {(r.source, r.rel_path): (r.verdict, r.reason)
               for r in dedup.itertuples(index=False)}
    dedup_sources = set(dedup["source"].unique())

    mans = load_v4_manifests(P)
    needed = sorted({s for c in comp for s in COMPONENT_SOURCES[c]})
    missing = [s for s in needed if s not in mans]
    if missing:
        msg = f"arm {arm} needs sources {needed}; MISSING manifests: {missing}"
        if not args.allow_missing_sources:
            raise SystemExit(msg + "  (use --allow-missing-sources to build without them)")
        warn(msg + " — proceeding WITHOUT them (recorded in frozen manifest)")
    stale = [s for s in needed if s in mans and s not in dedup_sources]
    if stale:
        raise SystemExit(f"manifest(s) {stale} present but not covered by dedup output — "
                         f"rerun `build_corpus.py dedup`")
    for s in needed:
        if s in mans and s in dedup_sources:
            n_ded = int((dedup["source"] == s).sum())
            if n_ded != len(mans[s]):
                raise SystemExit(
                    f"dedup row count for '{s}' ({n_ded}) != current manifest "
                    f"({len(mans[s])}) — manifest changed, rerun dedup")

    aerial = None
    if "T" in comp and any(s in mans for s in AERIAL_SOURCES):
        if not P.aerial_split_json.exists():
            raise SystemExit("aerial split missing — run `build_corpus.py aerial-split` first")
        aerial = json.loads(P.aerial_split_json.read_text())

    # --- assemble entries ---------------------------------------------------
    entries = {c: [] for c in comp}
    excluded = Counter()
    mismatches = []
    resolvers = {s: LabelResolver(s, P.v4_raw / s) for s in mans}

    if "E" in comp:
        local_lines, pod_lines = load_expB(P)
        for ll, pl in zip(local_lines, pod_lines):
            entries["E"].append(Entry("E", "existing", ll, pl,
                                      None, "existing"))
        log(f"E: {len(entries['E'])} lines from expB train list (integrity OK)")

    pod_img_prefix = f"{POD_V4_PREFIX}/{corpus_name}/images"
    used_names = {}

    # phase 1: routing + eligibility; collect conversion jobs
    staged = []   # (order_idx, component, src, rel_path, abs_path, kind)
    jobs = []     # (order_idx, src, kind, abs_path, n_exp, mask_hint)
    order_idx = 0
    for src in needed:
        if src not in mans:
            continue
        for r in mans[src].itertuples(index=False):
            v, reason = verdict.get((src, r.rel_path), ("ban", "not_in_dedup"))
            if v != "keep":
                excluded[f"dedup:{reason.split(':', 1)[0]}"] += 1
                continue
            if str(r.origin_class) != "usable":
                excluded["origin_class"] += 1
                continue
            # component routing
            if src == "boreal":
                component = "T"
                unit = str(r.parent_unit)
                tr_units = (set(aerial["sources"].get("boreal", {}).get("train", []))
                            if aerial else set())
                if unit not in tr_units:
                    excluded["aerial_holdout(boreal_val_test)"] += 1
                    continue
            elif src == "multinatsmoke":
                sub = getattr(r, "subset", None)
                if not isinstance(sub, str) or not sub:
                    sub = multinat_subsource(r.rel_path, getattr(r, "origin", None))
                if sub is None:
                    excluded["unknown_multinat_subsource"] += 1
                    continue
                component = "T" if sub in TARGET_SUBSOURCES else "G"
                if (FOREST_HOLDOUT_UNITS
                        and str(r.parent_unit) in FOREST_HOLDOUT_UNITS):
                    excluded["forest_holdout(ausmoke_val_test)"] += 1
                    continue
            else:
                component = "G"
            if component not in entries:
                excluded[f"component_{component}_not_in_arm"] += 1
                continue
            kind, why = train_eligibility(src, r.rel_path, r)
            if kind is None:
                excluded[f"policy:{why}"] += 1
                continue
            staged.append((order_idx, component, src, r.rel_path, r.abs_path, kind))
            if kind != "neg_background":
                n_exp = int(r.n_smoke_boxes) if r.n_smoke_boxes is not None else -1
                mask_hint = getattr(r, "mask_rel_path", None)
                mask_hint = (str(resolve_v4_path(P, src, mask_hint))
                             if isinstance(mask_hint, str) and mask_hint else None)
                jobs.append((order_idx, src, kind, r.abs_path, n_exp, mask_hint))
            order_idx += 1

    # phase 2: parallel conversion (fork pool; resolvers pre-warmed and inherited)
    conv = {}
    if jobs:
        global _CONV_RESOLVERS
        for src in {j[1] for j in jobs if j[2] == "pos_box"}:
            resolvers[src]._index(".txt")  # warm before fork
        _CONV_RESOLVERS = resolvers
        nw = min(64, os.cpu_count() or 8)
        log(f"converting {len(jobs)} annotations ({nw} workers)...")
        t0 = time.time()
        with get_context("fork").Pool(nw) as pool:
            for i, (idx, result) in enumerate(
                pool.imap_unordered(_conv_worker, jobs, chunksize=16)
            ):
                conv[idx] = result
                if (i + 1) % 5000 == 0:
                    log(f"  converted {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)")
        _CONV_RESOLVERS = None

    # phase 3: assemble entries in deterministic (manifest) order
    for idx, component, src, rel_path, abs_path, kind in staged:
        if kind == "neg_background":
            lines, origin = [], "background_neg"
        else:
            lines, origin, ref, err = conv[idx]
            if lines is None:
                excluded[f"convert:{err}"] += 1
                continue
            if err:  # box-count mismatch (non-fatal)
                mismatches.append(f"{src}/{rel_path}: {err}")
        name = flat_name(src, rel_path)
        if name in used_names:
            name = flat_name(src, sha256_bytes(rel_path.encode())[:8] + "_" + rel_path)
        used_names[name] = str(abs_path)
        entries[component].append(Entry(
            component, src,
            str(corpus_dir / "images" / name),
            f"{pod_img_prefix}/{name}",
            lines, origin, name,
        ))

    counts_comp = {c: len(entries[c]) for c in comp}
    log(f"assembled: {counts_comp}; excluded: {dict(excluded)}")
    if mismatches:
        warn(f"{len(mismatches)} box-count mismatches vs manifest n_smoke_boxes; first 10:")
        for m in mismatches[:10]:
            warn(f"  {m}")
    empty = [c for c in comp if counts_comp[c] == 0]
    if empty and not args.allow_missing_sources:
        raise SystemExit(f"components with ZERO images: {empty} — refusing to freeze "
                         f"(use --allow-missing-sources to override)")
    if empty:
        warn(f"components with ZERO images: {empty}")

    # --- materialize ---------------------------------------------------------
    if corpus_dir.exists():
        if not args.force:
            raise SystemExit(f"{corpus_dir} exists. Use --force to rebuild.")
        shutil.rmtree(corpus_dir)
    (corpus_dir / "images").mkdir(parents=True)
    (corpus_dir / "labels").mkdir(parents=True)
    (corpus_dir / "lists/local").mkdir(parents=True)
    (corpus_dir / "lists/pod").mkdir(parents=True)
    (corpus_dir / "yamls").mkdir(parents=True)

    v4_entries = [e for c in comp for e in entries[c] if e.component != "E"]
    for e in v4_entries:
        target = used_names[e.link_name]
        os.symlink(target, corpus_dir / "images" / e.link_name)
        stem = os.path.splitext(e.link_name)[0]
        (corpus_dir / "labels" / (stem + ".txt")).write_text(
            "\n".join(e.label_lines) + ("\n" if e.label_lines else ""))

    # sample conversion verification (positives only; uses real image sizes)
    verify_conversions([e for e in v4_entries if e.label_lines], n=args.verify_n)

    # --- multiplicities (C_EGT balancing) -------------------------------------
    mult = {c: 1 for c in comp}
    if arm in BALANCED_ARMS:
        nz = {c: n for c, n in counts_comp.items() if n > 0}
        if nz:
            M = max(nz.values())
            for c in nz:
                mult[c] = max(1, M // nz[c])  # effective in (M/2, M]
        log(f"balancing multiplicities: {mult} "
            f"(effective: { {c: counts_comp[c] * mult[c] for c in comp} })")

    # --- lists -----------------------------------------------------------------
    pairs = []
    for c in comp:
        for e in entries[c]:
            pairs.extend([(e.local_line, e.pod_line)] * mult[c])
    rnd = random.Random(SEED)
    rnd.shuffle(pairs)
    lists = {}
    for rel, idx in (("lists/local/train.txt", 0), ("lists/pod/train.txt", 1)):
        p = corpus_dir / rel
        p.write_text("\n".join(x[idx] for x in pairs) + "\n")
        lists[rel] = {"n": len(pairs), "sha256": sha256_file(p)}
    # per-component provenance lists (unrepeated, unshuffled)
    for c in comp:
        for rel, idx in ((f"lists/local/train_{c}.txt", 0), (f"lists/pod/train_{c}.txt", 1)):
            p = corpus_dir / rel
            p.write_text("\n".join(getattr(e, ("local_line", "pod_line")[idx])
                                   for e in entries[c]) + ("\n" if entries[c] else ""))
            lists[rel] = {"n": len(entries[c]), "sha256": sha256_file(p)}

    # --- yamls -------------------------------------------------------------------
    local_yaml = (
        f"# {corpus_name} (iteration 4) — generated {utcnow()}\n"
        f"path: {corpus_dir}\n"
        f"train: {corpus_dir / 'lists/local/train.txt'}\n"
        f"val: {P.sdis_val_local}\n"
        f"test: {P.sdis_test_local}\n"
        f"nc: 1\nnames: [smoke]\n"
    )
    pod_root = f"{POD_V4_PREFIX}/{corpus_name}"
    pod_yaml = (
        f"# {corpus_name} (iteration 4, pod) — generated {utcnow()}\n"
        f"path: {pod_root}\n"
        f"train: {pod_root}/lists/pod/train.txt\n"
        f"val: {POD_SDIS_VAL}\n"
        f"test: {POD_SDIS_TEST}\n"
        f"nc: 1\nnames: [smoke]\n"
    )
    (corpus_dir / "yamls/local.yaml").write_text(local_yaml)
    (corpus_dir / "yamls/pod.yaml").write_text(pod_yaml)
    for kind, text in (("local", local_yaml), ("pod", pod_yaml)):
        d = P.v4_yolo / "yamls" / kind
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{corpus_name}.yaml").write_text(text)

    # --- verify corpus ---------------------------------------------------------
    n_boxes = 0
    for e in v4_entries:
        img = corpus_dir / "images" / e.link_name
        assert img.exists(), f"broken symlink: {img}"
        n_boxes += parse_label_file(
            corpus_dir / "labels" / (os.path.splitext(e.link_name)[0] + ".txt"))
    miss = [l for l in {x[0] for x in pairs} if not os.path.exists(l)]
    if miss:
        raise SystemExit(f"{len(miss)} local train-list lines do not resolve, e.g. {miss[:3]}")
    log(f"verified: {len(v4_entries)} v4 images+labels, {n_boxes} smoke boxes, "
        f"all {len(pairs)} train-list lines resolve locally")

    # --- stage tar (dereferenced: real image bytes) ------------------------------
    P.v4_stage.mkdir(parents=True, exist_ok=True)
    tar_path = P.v4_stage / f"{corpus_name}.tar"
    log(f"writing {tar_path} (dereferenced) ...")
    with tarfile.open(tar_path, "w", dereference=True) as tf:
        tf.add(corpus_dir, arcname=corpus_name)
    tar_info = {"path": str(tar_path), "bytes": tar_path.stat().st_size,
                "sha256": sha256_file(tar_path)}

    # --- frozen manifest -----------------------------------------------------------
    per_source = Counter(e.source for c in comp for e in entries[c])
    per_origin = Counter(e.label_origin for c in comp for e in entries[c])
    frozen = {
        "arm": arm, "version": version, "corpus": corpus_name,
        "created_utc": utcnow(), "seed": SEED,
        "composition": list(comp),
        "composition_note": ("components parsed from arm name; D_pre = stage-1 "
                             "pretrain corpus (G), D_adapt = stage-2 adapt corpus "
                             "(E+T; same content family as E_ET, differs by init)"),
        "verified_empty_negatives": VERIFIED_EMPTY_AS_NEG,
        "target_subsources": sorted(TARGET_SUBSOURCES),
        "counts": {
            "per_component": counts_comp,
            "per_source": dict(per_source),
            "per_label_origin": dict(per_origin),
            "n_smoke_boxes_v4": n_boxes,
            "n_train_lines": len(pairs),
        },
        "excluded": dict(excluded),
        "multiplicities": {
            c: {"n_raw": counts_comp[c], "multiplicity": mult[c],
                "n_effective": counts_comp[c] * mult[c]} for c in comp
        },
        "lists": lists,
        "yamls": {"local": str(corpus_dir / "yamls/local.yaml"),
                  "pod": str(corpus_dir / "yamls/pod.yaml")},
        "val": {"local": str(P.sdis_val_local), "pod": POD_SDIS_VAL},
        "tar": tar_info,
        "missing_sources": missing,
        "box_count_mismatches": {"n": len(mismatches), "examples": mismatches[:20]},
        "dedup": {"parquet": str(P.dedup_parquet),
                  "sha256": sha256_file(P.dedup_parquet)},
        "aerial_split": ({"path": str(P.aerial_split_json),
                          "sha256": sha256_file(P.aerial_split_json)}
                         if aerial else None),
        "pod_root": pod_root,
    }
    frozen_path.write_text(json.dumps(frozen, indent=2))
    log(f"FROZEN manifest: {frozen_path}")
    print(json.dumps({k: frozen[k] for k in
                      ("corpus", "composition", "counts", "multiplicities",
                       "excluded", "missing_sources")}, indent=2))
    return 0


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------


def _mkimg(path, size=(96, 72), color=(30, 60, 30), rect=None, seed=0, quality=95):
    rnd = np.random.default_rng(seed)
    arr = np.clip(rnd.normal(0, 12, (size[1], size[0], 3)) +
                  np.array(color)[None, None, :], 0, 255).astype(np.uint8)
    if rect:
        x1, y1, x2, y2 = rect
        arr[y1:y2, x1:x2] = (210, 210, 215)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=quality)
    return path


def _mkmask(path, size=(96, 72), rect=(20, 15, 55, 40)):
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    x1, y1, x2, y2 = rect
    arr[y1:y2, x1:x2] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


def _manifest_row(source, rel_path, abs_path, unit, label_type, hs, hf, n_smoke,
                  origin="selftest", origin_class=None):
    row = {
        "source": source, "rel_path": rel_path, "sha256": sha256_file(abs_path),
        "parent_unit": unit, "camera_or_flight_id": unit,
        "timestamp": "2026-09-14T00:00:00Z", "label_type": label_type,
        "has_smoke_label": hs, "has_fire_label": hf, "n_smoke_boxes": n_smoke,
        "license": "test", "origin": origin,
    }
    if origin_class is not None:
        row["origin_class"] = origin_class
    return row


def cmd_selftest(_P_unused, args):
    tmp = Path(tempfile.mkdtemp(prefix="v4_corpus_selftest_"))
    log(f"self-test world: {tmp}")
    P = Paths(base=tmp)
    raw = P.v4_raw
    fails = []

    def check(cond, msg):
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            fails.append(msg)

    # ---- existing corpus (fake) ----
    sdis_rows = []
    for i, (site, role) in enumerate(
        [("brison", "train"), ("brison", "train"), ("cabanelle", "val"), ("ferion", "test")]
    ):
        p = tmp / f"data/yolo/pyro_sdis/images/train/img_{site}_{i}.jpg"
        # structured content so phash survives re-encoding (noise-only would not)
        _mkimg(p, seed=100 + i, rect=(8 + 6 * i, 10, 34 + 6 * i, 34 + 4 * i))
        sdis_rows.append({"image_path": str(p), "site": site, "split_official": "train"})
    P.manifests.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sdis_rows).to_parquet(P.pyro_sdis_manifest, index=False)

    dfire_img = tmp / "data/raw/dfire/train/images/df0.jpg"
    _mkimg(dfire_img, seed=200, rect=(5, 5, 25, 20))
    pd.DataFrame([{"image_path": "data/raw/dfire/train/images/df0.jpg",
                   "split_official": "train"}]).to_parquet(P.dfire_manifest, index=False)

    fig_frame = tmp / "data/raw/figlib/ev1/1000_+0000.jpg"
    _mkimg(fig_frame, seed=300, rect=(40, 10, 70, 35))
    pd.DataFrame([{"frame_path": "data/raw/figlib/ev1/1000_+0000.jpg"}]).to_parquet(
        P.figlib_manifest, index=False)
    lock_frame = P.figlib_lockbox / "ev_lock/2000_+0060.jpg"
    _mkimg(lock_frame, seed=301, rect=(50, 20, 80, 45))

    # expB lists: 4 lines (the two train-site sdis imgs + dfire + one more train img)
    extra = tmp / "data/yolo/pyro_sdis/images/train/img_brison_extra.jpg"
    _mkimg(extra, seed=101)
    expB_local = [sdis_rows[0]["image_path"], sdis_rows[1]["image_path"],
                  str(dfire_img), str(extra)]
    expB_pod = ["/mnt/local/smoke_data/" + os.path.relpath(p, tmp / "data/yolo")
                if "pyro_sdis" in p else "/mnt/local/smoke_data/dfire/" + os.path.basename(p)
                for p in expB_local]
    P.expB_local.parent.mkdir(parents=True, exist_ok=True)
    P.expB_pod.parent.mkdir(parents=True, exist_ok=True)
    P.expB_local.write_text("\n".join(expB_local) + "\n")
    P.expB_pod.write_text("\n".join(expB_pod) + "\n")
    P.sdis_val_local.write_text(sdis_rows[2]["image_path"] + "\n")
    P.sdis_test_local.write_text(sdis_rows[3]["image_path"] + "\n")
    P.splits_frozen.write_text(json.dumps({
        "site_split": {"train": ["brison"], "val": ["cabanelle"], "test": ["ferion"]},
        "lists": {"expB_train": {"n": 4, "sha256": sha256_file(P.expB_local)}},
    }))

    # ---- v4 sources ----
    rows = defaultdict(list)

    # boreal: 4 units x 2 frames, YOLO labels; one evo frame = exact copy of figlib frame
    for u, unit in enumerate(["evo", "karkkila", "ruokolahti", "heinola"]):
        for k in range(2):
            rel = f"Subset-A/{unit}/images/{unit}_DJI_{u}{k}.jpg"
            p = raw / "boreal" / rel
            if unit == "evo" and k == 0:
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(fig_frame, p)  # exact FIgLib dup -> group ban of evo
            else:
                _mkimg(p, seed=400 + 10 * u + k, rect=(15, 12, 45, 33))
            lbl = raw / "boreal" / f"Subset-A/{unit}/labels/{unit}_DJI_{u}{k}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("0 0.3125 0.3125 0.3125 0.2917\n")
            rows["boreal"].append(_manifest_row(
                "boreal", rel, p, unit, "box", True, False, 1))

    # firesense: 2 smoke_neg (background), 2 smoke_pos (videolabel -> excluded)
    for k in range(2):
        rel = f"frames/smoke_neg/negVideo1/frame{k}.jpg"
        p = _mkimg(raw / "firesense" / rel, seed=500 + k)
        rows["firesense"].append(_manifest_row(
            "firesense", rel, p, "smoke_neg/negVideo1", "videolabel", False, False, 0))
        rel = f"frames/smoke_pos/posVideo1/frame{k}.jpg"
        p = _mkimg(raw / "firesense" / rel, seed=510 + k, rect=(20, 20, 50, 45))
        rows["firesense"].append(_manifest_row(
            "firesense", rel, p, "smoke_pos/posVideo1", "videolabel", True, False, 0))

    # mmodalfire: 1 videolabel frame -> excluded
    p = _mkimg(raw / "mmodalfire" / "extracted/vid1/f0.jpg", seed=600)
    rows["mmodalfire"].append(_manifest_row(
        "mmodalfire", "extracted/vid1/f0.jpg", p, "vid1", "videolabel", True, True, 0))

    # dfs: VOC xml; 2 smoke positives + 1 fire-only
    for k, (hs, hf) in enumerate([(True, False), (True, True), (False, True)]):
        rel = f"JPEGImages/dfs{k}.jpg"
        p = _mkimg(raw / "dfs" / rel, seed=700 + k, rect=(30, 18, 70, 50))
        xml = raw / "dfs" / f"Annotations/dfs{k}.xml"
        xml.parent.mkdir(parents=True, exist_ok=True)
        objs = ""
        if hs:
            objs += ("<object><name>smoke</name><bndbox><xmin>30</xmin><ymin>18</ymin>"
                     "<xmax>70</xmax><ymax>50</ymax></bndbox></object>")
        if hf:
            objs += ("<object><name>fire</name><bndbox><xmin>2</xmin><ymin>2</ymin>"
                     "<xmax>12</xmax><ymax>12</ymax></bndbox></object>")
        xml.write_text(f"<annotation><size><width>96</width><height>72</height>"
                       f"</size>{objs}</annotation>")
        rows["dfs"].append(_manifest_row(
            "dfs", rel, p, f"dfs_unit{k}", "box", hs, hf, 1 if hs else 0))

    # multinatsmoke: masks. 2 AuSmoke (T), 2 WSDataset (G, one near-dups sdis val),
    # 1 D-Fire dup (origin_class), 1 FIgLib quarantine
    def mns(rel, seed, unit, oc="usable", copy_of=None, quality=95):
        p = raw / "multinatsmoke" / rel
        if copy_of is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(copy_of) as im:
                im.save(p, quality=quality)  # re-encoded -> near-dup, not exact
        else:
            _mkimg(p, seed=seed, rect=(25, 15, 60, 42))
        m = raw / "multinatsmoke" / replace_last(rel, "/images/", "/masks/")
        m = m.with_suffix(".png")
        _mkmask(m, rect=(25, 15, 60, 42))
        rows["multinatsmoke"].append(_manifest_row(
            "multinatsmoke", rel, p, unit, "mask", True, False, 1,
            origin=rel.split("/")[1], origin_class=oc))
        return p

    mns("MultiNatSmokeDataset/Train/AuSmoke/images/au0.jpg", 800, "AuSmoke_ev0")
    mns("MultiNatSmokeDataset/Train/AuSmoke/images/au1.jpg", 801, "AuSmoke_ev1")
    mns("MultiNatSmokeDataset/Train/WSDataset/images/ws0.jpg", 810, "WS_ev0")
    mns("MultiNatSmokeDataset/Train/WSDataset/images/ws1.jpg", 811, "WS_ev1",
        copy_of=sdis_rows[2]["image_path"], quality=70)  # near-dup of sdis VAL
    mns("MultiNatSmokeDataset/Train/D-Fire/images/dfdup.jpg", 820, "DFire_ev0",
        oc="dup_dfire")
    mns("MultiNatSmokeDataset/Train/FIgLib/images/figq.jpg", 830, "FIgLib_ev0",
        oc="quarantine_figlib")

    P.v4_manifests.mkdir(parents=True, exist_ok=True)
    for s, rs in rows.items():
        pd.DataFrame(rs).to_parquet(P.v4_manifests / f"{s}_manifest.parquet", index=False)

    # ---- run subcommands ----
    print("\n--- self-test: dedup ---")
    cmd_dedup(P, args)
    ded = pd.read_parquet(P.dedup_parquet)
    dd = {(r.source, r.rel_path): (r.verdict, r.reason) for r in ded.itertuples(index=False)}

    v, r = dd[("boreal", "Subset-A/evo/images/evo_DJI_00.jpg")]
    check(v == "ban" and r.startswith("exact_dup_eval"), f"figlib exact dup banned ({r})")
    v, r = dd[("boreal", "Subset-A/evo/images/evo_DJI_01.jpg")]
    check(v == "ban" and r.startswith("group_ban_eval"), f"evo unit-mate group-banned ({r})")
    v, r = dd[("multinatsmoke", "MultiNatSmokeDataset/Train/D-Fire/images/dfdup.jpg")]
    check(v == "ban" and r.startswith("origin_class"), f"dup_dfire origin_class banned ({r})")
    v, r = dd[("multinatsmoke", "MultiNatSmokeDataset/Train/WSDataset/images/ws1.jpg")]
    check(v == "ban" and r.startswith("neardup_eval"), f"sdis-val near-dup banned ({r})")
    check(dd[("boreal", "Subset-A/karkkila/images/karkkila_DJI_10.jpg")][0] == "keep",
          "clean boreal frame kept")

    print("\n--- self-test: aerial-split ---")
    cmd_aerial_split(P, args)
    asp = json.loads(P.aerial_split_json.read_text())
    b = asp["sources"]["boreal"]
    check(b["n_units"] == 4 and len(b["val"]) >= 1 and len(b["test"]) >= 1,
          f"aerial split 4 units -> train={b['train']} val={b['val']} test={b['test']}")
    check("tiny_n_warning" in b, "tiny-n limitation documented")

    print("\n--- self-test: build C_EGT + B_G ---")
    for arm in ("C_EGT", "B_G"):
        a = argparse.Namespace(arm=arm, version="stest", force=True,
                               allow_missing_sources=False, verify_n=5)
        cmd_build(P, a)

    cj = json.loads((P.v4_manifests / "CORPUS_C_EGT_stest.json").read_text())
    cdir = P.v4_yolo / "C_EGT_stest"
    train_local = [l for l in (cdir / "lists/local/train.txt").read_text().splitlines() if l]
    train_pod = [l for l in (cdir / "lists/pod/train.txt").read_text().splitlines() if l]
    check(len(train_local) == len(train_pod) == cj["counts"]["n_train_lines"],
          f"local/pod list lengths match frozen manifest ({len(train_local)})")
    check(all(os.path.exists(l) for l in train_local), "every local train line resolves")
    check(all(l.startswith(("/mnt/local/", POD_V4_PREFIX)) for l in train_pod),
          "pod train lines use pod prefixes")
    nlab = 0
    for f in (cdir / "labels").glob("*.txt"):
        nlab += 1
        parse_label_file(f)
    check(nlab > 0, f"{nlab} label txts parse as valid YOLO")
    # expected component counts
    exp_T = 2 * len(set(b["train"]) - {"evo"}) + 2          # boreal train units + 2 AuSmoke
    exp_G = 2 + 1 + 2                                        # dfs smoke + WS mask kept + fs neg
    check(cj["counts"]["per_component"]["E"] == 4, "E count == 4")
    check(cj["counts"]["per_component"]["G"] == exp_G,
          f"G count == {exp_G} (got {cj['counts']['per_component']['G']})")
    check(cj["counts"]["per_component"]["T"] == exp_T,
          f"T count == {exp_T} (got {cj['counts']['per_component']['T']})")
    m = cj["multiplicities"]
    M = max(v["n_effective"] for v in m.values())
    check(all(v["n_effective"] * 2 > M for v in m.values() if v["n_raw"] > 0),
          f"C_EGT balance within 2x ({ {k: v['n_effective'] for k, v in m.items()} })")
    check(cj["counts"]["per_label_origin"].get("mask-derived", 0) >= 3,
          "mask-derived boxes present")
    check(cj["counts"]["per_label_origin"].get("background_neg", 0) == 2,
          "2 firesense background negatives")
    excl = cj["excluded"]
    check(excl.get("policy:smoke_pos_without_boxes(label_type=videolabel)", 0) >= 3,
          f"video-positives excluded ({excl})")
    check(excl.get("policy:fire_only", 0) == 1, "fire-only excluded")
    tarp = Path(cj["tar"]["path"])
    with tarfile.open(tarp) as tf:
        members = tf.getnames()
        imgs = [mm for mm in members if "/images/" in mm]
        info = tf.getmember(imgs[0])
        check(info.isfile() and info.size > 0,
              f"tar contains dereferenced image bytes ({len(imgs)} images)")
        check(any(mm.endswith("lists/pod/train.txt") for mm in members), "tar contains lists")
    bg = json.loads((P.v4_manifests / "CORPUS_B_G_stest.json").read_text())
    check(bg["composition"] == ["G"] and bg["counts"]["per_component"]["G"] == exp_G,
          "B_G arm = G only")

    print()
    if fails:
        print(f"SELF-TEST FAILED ({len(fails)} checks):")
        for f in fails:
            print(f"  - {f}")
        print(f"world kept at {tmp}")
        return 1
    print(f"SELF-TEST PASSED (world: {tmp})")
    if not args.keep_world:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run end-to-end self test in /tmp")
    ap.add_argument("--keep-world", action="store_true", help="keep self-test tmp dir")
    ap.add_argument("--base", default=str(REPO_ROOT), help="repo root override")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("dedup", help="exact + perceptual dedup of v4 sources")
    sub.add_parser("aerial-split", help="parent_unit 60/20/20 split for aerial sources")
    b = sub.add_parser("build", help="materialize a frozen arm corpus")
    b.add_argument("--arm", required=True, choices=sorted(ARM_COMPOSITIONS))
    b.add_argument("--version", default="v1")
    b.add_argument("--force", action="store_true", help="overwrite an existing frozen corpus")
    b.add_argument("--allow-missing-sources", action="store_true")
    b.add_argument("--verify-n", type=int, default=5)
    b.add_argument("--verified-empty-negatives", action="store_true",
                   help="treat explicit zero-box annotations as background "
                        "negatives (author-verified empties, e.g. boreal)")
    sub.add_parser("selftest", help="alias for --self-test")
    args = ap.parse_args(argv)

    P = Paths(base=Path(args.base))
    if args.self_test or args.cmd == "selftest":
        return cmd_selftest(P, args)
    if args.cmd == "dedup":
        return cmd_dedup(P, args)
    if args.cmd == "aerial-split":
        return cmd_aerial_split(P, args)
    if args.cmd == "build":
        return cmd_build(P, args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
