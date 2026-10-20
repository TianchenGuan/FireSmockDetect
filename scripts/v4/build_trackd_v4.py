"""Build v3-format temporal sequence manifests from the v4 sequence corpus, and
stage a resized frame tree for cluster feature extraction (Track D / v4).

Why a converter: src/v3/trackd_features.py consumes a JSON list of sequences
with per-frame {image_name, image_path, ts, smoke, boxes_xywhn}. The v4 data
prep track shipped a parquet + a replay jsonl instead. This maps one to the
other WITHOUT inventing any label.

TRAIN extension (frame-supervised only):
  - ausmoke split=train (22 seqs / 10,905 frames): per-frame mask-derived boxes,
    every frame positive -> PERSISTENCE supervision only. AuSmoke observes 0/26
    onsets (data/v4/sequences/ONSET_AUDIT.md).
  - boreal split=train & seq_label=no_smoke (1 clip / 32 frames): a video-level
    "no smoke" label DOES imply every frame is negative, so this is legitimate
    frame-level negative supervision.
EXCLUDED from train (reasons recorded in the manifest):
  - boreal/firesense video-level POSITIVES: no frame truth exists; per-frame
    positives would be fabricated (ONSET_AUDIT.md forbids this).
  - firesense smoke_neg frames: data prep assigned them to VAL as the only
    continuous negative-footage replay pool. Training on them would destroy
    that eval pool and add a second contamination layer (they are already
    background negatives inside v4 YOLO CORPUS_B_G_v1 / C_EGT_v1 / D_adapt_v1).

VAL REPLAY: all 139 sequences of replay_val_v4.jsonl, carrying role/truth_level
so the evaluator can refuse to use video-level positives as frame truth.

Frame staging: DEFAULT is byte-identical passthrough of the native frames (no
resize, no re-encode), so the pod extractor sees exactly the pixels the v4 YOLO
arms see. Measured devbox->S3 throughput is ~227 MB/s, which makes the ~16.5 GB
full-res upload cheap enough that there is no reason to trade fidelity for size.
  --prescale is kept as an option: ultralytics LetterBox performs exactly one
cv2.INTER_LINEAR resize from native size to the imgsz-fitted size, so
pre-resizing to that same fitted size is the same single operation (the fitted
shapes match exactly - scripts/v4/verify_resize_equiv.py). It was NOT used for
the shipped features because JPEG q95 recompression alone moved per-frame
max-conf by up to 0.050 and features by ~4% relative: avoidable noise.

Usage:
  python scripts/v4/build_trackd_v4.py manifests
  python scripts/v4/build_trackd_v4.py stage [--workers 64]
  python scripts/v4/build_trackd_v4.py tar
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEQ_PARQUET = ROOT / "data/v4/sequences/sequences_v4.parquet"
REPLAY = ROOT / "data/v4/sequences/replay_val_v4.jsonl"
OUT_DIR = ROOT / "data/v4/sequences/trackd"
STAGE_ROOT = ROOT / "data/v4/stage/trackd_v4_frames"
TAR_PATH = ROOT / "data/v4/stage/trackd_v4.tar"
IMGSZ = 1280


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def xyxyn_to_xywhn(b):
    x1, y1, x2, y2 = b
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, 0.0), max(y2 - y1, 0.0)]


def fitted_size(w, h, imgsz=IMGSZ):
    """ultralytics LetterBox target before stride padding. Never pre-UPscale
    (r>=1 -> passthrough; the pod-side letterbox does any upscaling)."""
    r = min(imgsz / h, imgsz / w)
    if r >= 1.0:
        return w, h
    return int(round(w * r)), int(round(h * r))


def rel_for(src, abs_path):
    p = Path(abs_path)
    if src in ("boreal", "firesense"):
        return "%s/%s/%s" % (src, p.parent.name, p.name)
    return "%s/%s" % (src, p.name)


def img_name(src, abs_path):
    return "%s__%s" % (src, Path(abs_path).name)


# ------------------------------------------------------------------ manifests
def build_manifests():
    df = pd.read_parquet(SEQ_PARQUET)
    replay = [json.loads(l) for l in open(REPLAY)]
    staged = {}

    def mk_frame(src, abs_path, ts, smoke, boxes_xyxyn):
        rel = rel_for(src, abs_path)
        staged[abs_path] = rel
        return {
            "image_name": img_name(src, abs_path),
            "image_path": str(STAGE_ROOT / rel),
            "ts": int(ts),
            "smoke": bool(smoke),
            "boxes_xywhn": [xyxyn_to_xywhn(b) for b in (boxes_xyxyn or [])],
        }

    # ---- TRAIN EXTENSION -------------------------------------------------
    train_seqs, excluded = [], []
    for (src, split, lab), g in df.groupby(["source", "split", "seq_label"]):
        keep = (src == "ausmoke" and split == "train") or \
               (src == "boreal" and split == "train" and lab == "no_smoke")
        if not keep:
            if split == "train":
                excluded.append({
                    "source": src, "split": split, "seq_label": lab,
                    "sequences": int(g.seq_id.nunique()), "frames": int(len(g)),
                    "reason": "video-level positive label only, no frame truth"
                              " -> frame supervision would be fabricated",
                })
            continue
        for seq_id, sg in g.groupby("seq_id"):
            sg = sg.sort_values("frame_idx")
            frames = []
            for r in sg.itertuples():
                bx = json.loads(r.boxes) if isinstance(r.boxes, str) and r.boxes else []
                frames.append(mk_frame(src, r.abs_path, r.ts_or_order,
                                       lab == "smoke", bx))
            train_seqs.append({
                "seq_id": "v4x_" + seq_id, "split": "train_ext", "source": src,
                "parent_unit": str(sg.parent_unit.iloc[0]), "seq_label": lab,
                "truth_level": ("frame_boxes" if src == "ausmoke"
                                else "videolabel_negative"),
                "onset_observed": False, "n_frames": len(frames),
                "frames": frames,
            })

    fs_neg = df[(df.source == "firesense") & (df.seq_label == "no_smoke")]
    excluded.append({
        "source": "firesense", "split": "val", "seq_label": "no_smoke",
        "sequences": int(fs_neg.seq_id.nunique()), "frames": int(len(fs_neg)),
        "reason": "data-prep policy assigns firesense smoke_neg to VAL as the "
                  "only continuous negative-footage replay pool; training on "
                  "them would destroy that eval pool and add a second "
                  "contamination layer (already in v4 YOLO CORPUS_B_G_v1)",
    })

    # ---- VAL REPLAY ------------------------------------------------------
    val_seqs = []
    for r in replay:
        src = r["source"]
        frame_truth = (r["truth_level"] == "frame_boxes"
                       or r["role"] == "negative_footage")
        frames = []
        for fr in r["frames"]:
            if r["role"] == "negative_footage":
                smoke = False
            elif r["truth_level"] == "frame_boxes":
                smoke = bool(fr.get("has_box"))
            else:
                smoke = True  # video-level positive: NOT frame truth
            frames.append(mk_frame(src, fr["abs_path"], fr["ts_or_order"],
                                   smoke, fr.get("boxes") or []))
        val_seqs.append({
            "seq_id": "v4r_" + r["seq_id"], "split": "val_replay", "source": src,
            "parent_unit": r["parent_unit"], "role": r["role"],
            "seq_label": r["seq_label"], "frame_truth": frame_truth,
            "truth_level": r["truth_level"], "onset_observed": False,
            "notes": r.get("notes", []), "n_frames": len(frames),
            "frames": frames,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sequences_v4_train_ext.json").write_text(json.dumps(train_seqs))
    (OUT_DIR / "sequences_v4_val_replay.json").write_text(json.dumps(val_seqs))
    plan = [{"src": k, "rel": v} for k, v in sorted(staged.items())]
    (OUT_DIR / "stage_plan.json").write_text(json.dumps(plan))

    def cnt(seqs, pred=None):
        ss = [s for s in seqs if (pred is None or pred(s))]
        return {"sequences": len(ss), "frames": sum(s["n_frames"] for s in ss)}

    man = {
        "built_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "sequences_v4.parquet": {"sha256": sha256(SEQ_PARQUET),
                                     "bytes": SEQ_PARQUET.stat().st_size},
            "replay_val_v4.jsonl": {"sha256": sha256(REPLAY),
                                    "bytes": REPLAY.stat().st_size},
        },
        "extractor": {
            "code": "src/v3/trackd_features.py (unchanged from v3)",
            "weights": "models/trained/expE_mix_y11m_1280_v2/seg05/best.pt",
            "imgsz": IMGSZ, "n_channels": 194, "grid": [12, 20],
        },
        "train_ext": {
            "total": cnt(train_seqs),
            "by_source": dict((s, cnt(train_seqs, lambda x, s=s: x["source"] == s))
                              for s in sorted(set(t["source"] for t in train_seqs))),
            "positive_frames": int(sum(sum(f["smoke"] for f in s["frames"])
                                       for s in train_seqs)),
            "onset_observed_sequences": 0,
        },
        "val_replay": {
            "total": cnt(val_seqs),
            "by_source_role": dict(
                (s + "/" + ro,
                 cnt(val_seqs, lambda x, s=s, ro=ro: x["source"] == s
                     and x["role"] == ro))
                for s in sorted(set(v["source"] for v in val_seqs))
                for ro in sorted(set(v["role"] for v in val_seqs
                                     if v["source"] == s))),
            "frame_truth_sequences": cnt(val_seqs, lambda x: x["frame_truth"]),
            "videolabel_only_sequences": cnt(val_seqs,
                                             lambda x: not x["frame_truth"]),
        },
        "excluded_from_train": excluded,
        "unique_frames_to_stage": len(plan),
        "ts_semantics": "ts = ts_or_order from the v4 parquet, read as SECONDS. "
                        "boreal/firesense really are 1 fps. AuSmoke's original "
                        "fps was never recorded (order units), so the dt "
                        "feature treats AuSmoke as ~1 fps: a documented "
                        "approximation, not measured truth.",
        "cadence_caveat": "v4 sequences are ~1 fps; pyro-sdis (v3) median "
                          "cadence is 29 s. A T<=5 causal window spans ~5 s on "
                          "v4 vs ~2 min on v3 - disclosed distribution shift.",
    }
    (OUT_DIR / "MANIFEST_trackd_v4.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))


# ------------------------------------------------------------------ staging
def _stage_one(job):
    src, rel, prescale = job
    dst = STAGE_ROOT / rel
    if dst.exists() and dst.stat().st_size > 0:
        return 0, 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = cv2.imread(src)
    if im is None:
        return 0, 1
    h, w = im.shape[:2]
    tw, th = fitted_size(w, h) if prescale else (w, h)
    if (tw, th) != (w, h):
        im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_LINEAR)
        ok = bool(cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95]))
    else:
        dst.write_bytes(Path(src).read_bytes())  # byte-identical passthrough
        ok = True
    return (1, 0) if ok else (0, 1)


def stage(workers, prescale=False):
    plan = json.loads((OUT_DIR / "stage_plan.json").read_text())
    jobs = [(p["src"], p["rel"], prescale) for p in plan]
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    done = err = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (d, e) in enumerate(ex.map(_stage_one, jobs, chunksize=32)):
            done += d
            err += e
            if (i + 1) % 4000 == 0:
                print("%d/%d written=%d err=%d" % (i + 1, len(jobs), done, err),
                      flush=True)
    files = list(STAGE_ROOT.rglob("*.jpg"))
    total = sum(f.stat().st_size for f in files)
    print("STAGE_DONE prescale=%s planned=%d written=%d errors=%d on_disk=%d "
          "bytes=%d (%.2f GB)"
          % (prescale, len(jobs), done, err, len(files), total, total / 1e9))
    assert err == 0, "%d unreadable frames" % err


def make_tar():
    link = STAGE_ROOT.parent / "trackd_v4"
    if not link.exists():
        os.symlink(STAGE_ROOT, link)
    subprocess.run(["tar", "chf", str(TAR_PATH), "-C", str(STAGE_ROOT.parent),
                    "trackd_v4"], check=True)
    print("TAR %s %.2f GB sha256=%s"
          % (TAR_PATH, TAR_PATH.stat().st_size / 1e9, sha256(TAR_PATH)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifests", "stage", "tar"])
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--prescale", action="store_true",
                    help="pre-resize to the imgsz-fitted size (lossy JPEG "
                         "re-encode; NOT used for the shipped v4 features)")
    a = ap.parse_args()
    if a.cmd == "manifests":
        build_manifests()
    elif a.cmd == "stage":
        stage(a.workers, a.prescale)
    else:
        make_tar()


if __name__ == "__main__":
    main()
