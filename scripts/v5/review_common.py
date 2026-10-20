"""Shared helpers for the V5 CPU review tools (negative_review.py,
onset_annotate.py, export_candidates.py).

Design rules (PROTOCOL_V5.md §3):
  * valid observed time = covered intervals only; a gap > 3x the sequence's
    median cadence is excluded; independent units are never concatenated.
  * detector scores are ASSISTIVE only; every renderer works without them.
  * everything deterministic: sorted inputs, fixed tie-breaks, no RNG.
"""
from __future__ import annotations

import csv
import datetime as _dt
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v"}

_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})")
# V5 Day-1: MONIMET phenocam names, e.g. lammi_ls_mixed_2020_01_10_110111.jpg (YYYY_MM_DD_HHMMSS, local UTC+2)
_ISO2_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})(\d{2})(?:\.|_|$)")
_TIDX_RE = re.compile(r"_t(\d{3,6})(?:\.|_|$)")
_FRAME_RE = re.compile(r"frame(\d+)(?:\.|_|$)")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def sha256_file(p: str | Path, limit_bytes: Optional[int] = None) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        if limit_bytes:
            h.update(f.read(limit_bytes))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------
def parse_ts_from_name(name: str) -> Tuple[Optional[float], str]:
    """Return (seconds, kind). kind in {iso_epoch, t_index, frame_index, none}.
    iso_epoch -> absolute epoch seconds (UTC-naive as written in the name);
    t_index   -> relative seconds from `_tNNNN` (1-fps extraction convention
                 used by data/v4/sequences/boreal_frames and firesense frames);
    frame_index -> raw video frame number (needs fps to become seconds)."""
    m = _ISO_RE.search(name) or _ISO2_RE.search(name)
    if m:
        y, mo, d, hh, mm, ss = map(int, m.groups())
        dt = _dt.datetime(y, mo, d, hh, mm, ss, tzinfo=_dt.timezone.utc)
        return dt.timestamp(), "iso_epoch"
    m = _TIDX_RE.search(name)
    if m:
        return float(int(m.group(1))), "t_index"
    m = _FRAME_RE.search(name)
    if m:
        return float(int(m.group(1))), "frame_index"
    return None, "none"


def fmt_rel(seconds: Optional[float]) -> str:
    if seconds is None:
        return "t=?"
    s = int(round(seconds))
    sign = "-" if s < 0 else "+"
    s = abs(s)
    if s >= 3600:
        return f"t={sign}{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"t={sign}{s // 60}:{s % 60:02d}"


def fmt_abs(epoch: Optional[float]) -> str:
    if epoch is None:
        return ""
    return _dt.datetime.fromtimestamp(epoch, _dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def valid_observed_seconds(ts: Sequence[float], gap_factor: float = 3.0,
                           cadence_hint: Optional[float] = None) -> Dict:
    """PROTOCOL_V5 §3 valid observed time for ONE unit.
    Each frame i covers [ts_i, ts_{i+1}) if that gap <= gap_factor x median
    cadence, otherwise it covers one median cadence (the gap is excluded).
    The last frame covers one median cadence. Returns dict with seconds,
    cadence, excluded gaps."""
    ts = [float(t) for t in ts]
    n = len(ts)
    if n == 0:
        return dict(valid_s=0.0, cadence_s=None, n_frames=0, n_gaps_excluded=0,
                    excluded_s=0.0, span_s=0.0)
    if n == 1:
        cad = cadence_hint or 1.0
        return dict(valid_s=cad, cadence_s=cad, n_frames=1, n_gaps_excluded=0,
                    excluded_s=0.0, span_s=0.0)
    diffs = np.diff(np.asarray(ts))
    med = float(np.median(diffs)) if cadence_hint is None else float(cadence_hint)
    if med <= 0:
        med = cadence_hint or 1.0
    lim = gap_factor * med
    valid = 0.0
    excl = 0.0
    n_excl = 0
    for d in diffs:
        if d <= lim:
            valid += float(d)
        else:
            valid += med
            excl += float(d) - med
            n_excl += 1
    valid += med  # last frame
    return dict(valid_s=round(valid, 3), cadence_s=round(med, 3), n_frames=n,
                n_gaps_excluded=n_excl, excluded_s=round(excl, 3),
                span_s=round(float(ts[-1] - ts[0]), 3), gap_limit_s=round(lim, 3))


# --------------------------------------------------------------------------
# frame sources
# --------------------------------------------------------------------------
class FrameSource:
    """Ordered frames from a directory of images, an explicit list, or a video.
    .frames: list of dict(idx, name, path, video_frame, rel_s, abs_epoch, ts_kind)
    .load(i) -> PIL.Image (RGB)."""

    def __init__(self, kind: str, ref: str, frames: List[dict], fps: Optional[float] = None,
                 cadence_s: Optional[float] = None):
        self.kind = kind          # frames_dir | frame_list | video
        self.ref = ref
        self.frames = frames
        self.fps = fps
        self.cadence_s = cadence_s
        self._cap = None

    # ---- constructors ----
    @classmethod
    def from_dir(cls, d: str | Path, cadence_s: Optional[float] = None):
        d = Path(d)
        files = sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXT],
                       key=lambda p: natural_key(p.name))
        return cls._from_files([str(p) for p in files], "frames_dir", str(d), cadence_s)

    @classmethod
    def from_paths(cls, paths: Sequence[str], ref: str = "frame_list",
                   cadence_s: Optional[float] = None, ts_override: Optional[Sequence[float]] = None):
        src = cls._from_files(list(paths), "frame_list", ref, cadence_s)
        if ts_override is not None:
            assert len(ts_override) == len(src.frames)
            t0 = float(ts_override[0])
            for f, t in zip(src.frames, ts_override):
                f["abs_epoch"] = float(t)
                f["rel_s"] = float(t) - t0
                f["ts_kind"] = "given_epoch"
        return src

    @classmethod
    def _from_files(cls, files: List[str], kind: str, ref: str, cadence_s: Optional[float]):
        frames = []
        kinds = []
        for i, p in enumerate(files):
            v, k = parse_ts_from_name(os.path.basename(p))
            kinds.append(k)
            frames.append(dict(idx=i, name=os.path.basename(p), path=p, video_frame=None,
                               rel_s=None, abs_epoch=None, ts_kind=k, _raw=v))
        # decide timestamp semantics for the whole unit
        if frames:
            dominant = max(set(kinds), key=kinds.count)
            if dominant == "iso_epoch" and all(k == "iso_epoch" for k in kinds):
                t0 = frames[0]["_raw"]
                for f in frames:
                    f["abs_epoch"] = f["_raw"]
                    f["rel_s"] = f["_raw"] - t0
            elif dominant == "t_index" and all(k == "t_index" for k in kinds):
                cad = cadence_s or 1.0
                for f in frames:
                    f["rel_s"] = f["_raw"] * cad
                    f["ts_kind"] = "t_index"
            elif dominant == "frame_index" and all(k == "frame_index" for k in kinds):
                # frameN = N-th extracted still; cadence_s = seconds per index step
                cad = cadence_s or 1.0
                for f in frames:
                    f["rel_s"] = f["_raw"] * cad
                    f["ts_kind"] = "frame_index_x_cadence" + ("" if cadence_s else "_ASSUMED_1s")
            else:
                # fall back to index x cadence (cadence_s required to be meaningful)
                cad = cadence_s or 1.0
                for f in frames:
                    f["rel_s"] = f["idx"] * cad
                    f["ts_kind"] = "index_x_cadence" + ("" if cadence_s else "_ASSUMED_1s")
        for f in frames:
            f.pop("_raw", None)
        return cls(kind, ref, frames, fps=None, cadence_s=cadence_s)

    @classmethod
    def from_video(cls, video: str | Path, sample_s: float = 1.0, max_frames: Optional[int] = None):
        import cv2  # local import: keep PIL-only paths cv2-free
        video = str(video)
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps <= 0:
            raise RuntimeError(f"video has no fps metadata: {video}")
        step = max(1, int(round(sample_s * fps)))
        targets = list(range(0, max(n, 1), step))
        if max_frames:
            targets = targets[:max_frames]
        frames = []
        for i, vf in enumerate(targets):
            frames.append(dict(idx=i, name=f"{Path(video).stem}@f{vf:06d}", path=video, video_frame=vf,
                               rel_s=vf / fps, abs_epoch=None, ts_kind="video_frame/fps"))
        src = cls("video", video, frames, fps=fps, cadence_s=step / fps)
        src.n_video_frames = n
        return src

    # ---- access ----
    def load(self, i: int, max_side: Optional[int] = None) -> Image.Image:
        f = self.frames[i]
        if self.kind == "video":
            return self._load_video_frame(f["video_frame"], max_side)
        im = Image.open(f["path"])
        if max_side:
            # JPEG draft mode = fast downscaled decode (4K -> thumbnail)
            try:
                im.draft("RGB", (max_side * 2, max_side * 2))
            except Exception:
                pass
        im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.BILINEAR)
        return im

    def iter_images(self, max_side: Optional[int] = None) -> Iterable[Tuple[dict, Image.Image]]:
        """Sequential iteration (efficient for video: single pass decode)."""
        if self.kind != "video":
            for i in range(len(self.frames)):
                yield self.frames[i], self.load(i, max_side)
            return
        import cv2
        cap = cv2.VideoCapture(self.ref)
        want = {f["video_frame"]: f for f in self.frames}
        last = max(want) if want else -1
        k = 0
        while k <= last:
            ok, bgr = cap.read()
            if not ok:
                break
            if k in want:
                im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if max_side and max(im.size) > max_side:
                    im.thumbnail((max_side, max_side), Image.BILINEAR)
                yield want[k], im
            k += 1
        cap.release()

    def _load_video_frame(self, vf: int, max_side: Optional[int]) -> Image.Image:
        import cv2
        cap = cv2.VideoCapture(self.ref)
        cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"cannot decode frame {vf} of {self.ref}")
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.BILINEAR)
        return im

    def rel_times(self) -> List[float]:
        return [f["rel_s"] for f in self.frames]

    def observed(self, gap_factor: float = 3.0) -> Dict:
        return valid_observed_seconds(self.rel_times(), gap_factor, cadence_hint=self.cadence_s)


def open_source(path: str, cadence_s: Optional[float] = None, sample_s: float = 1.0,
                max_frames: Optional[int] = None) -> FrameSource:
    p = Path(path)
    if p.is_dir():
        return FrameSource.from_dir(p, cadence_s=cadence_s)
    if p.suffix.lower() in VID_EXT:
        return FrameSource.from_video(p, sample_s=sample_s, max_frames=max_frames)
    if p.suffix.lower() in {".txt", ".lst"}:
        paths = [l.strip() for l in open(p) if l.strip() and not l.startswith("#")]
        return FrameSource.from_paths(paths, ref=str(p), cadence_s=cadence_s)
    raise ValueError(f"unsupported input {path}: need a directory, a video, or a .txt list")


# --------------------------------------------------------------------------
# detector scores (assistive only)
# --------------------------------------------------------------------------
def _open_maybe_gz(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_preds(path: str) -> Dict[str, list]:
    """Load saved detector outputs -> {abs_image_path: [[x1,y1,x2,y2,conf],...]}.
    Accepts *.raw.parquet (columns image, preds[, gt]) and *.jsonl[.gz]
    ({"image":..., "preds":[...]})."""
    out: Dict[str, list] = {}
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path, columns=["image", "preds"])
        for img, pr in zip(df["image"].tolist(), df["preds"].tolist()):
            out[str(img)] = [list(map(float, b)) for b in (pr if pr is not None else [])]
        return out
    with _open_maybe_gz(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[str(r["image"])] = [list(map(float, b)) for b in (r.get("preds") or [])]
    return out


def frame_score(preds: Optional[list]) -> float:
    """Frame-level score = max box confidence (0.0 when no boxes)."""
    if not preds:
        return 0.0
    return max(float(b[4]) for b in preds if len(b) >= 5) if any(len(b) >= 5 for b in preds) else 0.0


def lookup_score(pred_map: Dict[str, list], path: str) -> Optional[float]:
    """Match by absolute path first, then by basename (for relocated images).
    Returns None when the frame has no saved prediction at all."""
    if path in pred_map:
        return frame_score(pred_map[path])
    base = os.path.basename(path)
    if not hasattr(lookup_score, "_base_cache") or lookup_score._base_cache_src is not pred_map:
        lookup_score._base_cache = {os.path.basename(k): k for k in pred_map}
        lookup_score._base_cache_src = pred_map
    k = lookup_score._base_cache.get(base)
    return frame_score(pred_map[k]) if k is not None else None


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
_FONT_PATHS = [
    "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def get_font(size: int = 14) -> ImageFont.FreeTypeFont:
    for p in _FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def burn_text(im: Image.Image, lines: Sequence[str], size: int = 14, pos: str = "bottom",
              fill=(255, 255, 255), bg=(0, 0, 0)) -> Image.Image:
    """Burn text lines with an opaque background strip (never over-writes the
    image content area: the strip is appended below/above the thumbnail)."""
    font = get_font(size)
    lh = size + 4
    strip_h = lh * len(lines) + 4
    w, h = im.size
    out = Image.new("RGB", (w, h + strip_h), bg)
    if pos == "top":
        out.paste(im, (0, strip_h))
        y0 = 2
    else:
        out.paste(im, (0, 0))
        y0 = h + 2
    dr = ImageDraw.Draw(out)
    for i, t in enumerate(lines):
        dr.text((3, y0 + i * lh), t, font=font, fill=fill)
    return out


def contact_sheet(tiles: List[Image.Image], cols: int, thumb_w: int, title: Optional[str] = None,
                  pad: int = 4, bg=(24, 24, 24)) -> Image.Image:
    """Grid of equally sized tiles (tiles are already labelled). Rows = ceil(n/cols)."""
    if not tiles:
        raise ValueError("no tiles")
    tw = thumb_w
    th = max(t.size[1] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    title_h = 0
    font = None
    if title:
        font = get_font(16)
        title_h = 24
    W = cols * tw + (cols + 1) * pad
    H = rows * th + (rows + 1) * pad + title_h
    sheet = Image.new("RGB", (W, H), bg)
    if title:
        ImageDraw.Draw(sheet).text((pad, 4), title, font=font, fill=(230, 230, 230))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = title_h + pad + r * (th + pad)
        sheet.paste(t, (x, y))
    return sheet


def fit_width(im: Image.Image, w: int) -> Image.Image:
    if im.size[0] == w:
        return im
    h = max(1, int(round(im.size[1] * w / im.size[0])))
    return im.resize((w, h), Image.BILINEAR)


def write_csv(path: str | Path, rows: List[dict], columns: Sequence[str]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})


def read_csv(path: str | Path) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: str | Path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_json_default)
        f.write("\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(str(type(o)))
