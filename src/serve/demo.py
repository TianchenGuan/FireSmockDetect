"""Demo endpoints (iteration 5b).

  GET /demo                      the FAST / ACCURATE upload + historical-replay UI
  GET /demo/sequences            short allow-list of stored sequences (pyro-sdis VAL
                                 sites + Boreal heinola aerial-VAL clip) with per-frame
                                 timestamps, dataset labels and frame URLs
  GET /demo/frame/{seq}/{idx}    one stored frame (only frames in the allow-list)
  GET /demo/runs[/{run_id}]      replay runs saved by scripts/v5b/demo_replay.py

The UI (and demo_replay.py) send every replayed frame through the ordinary
POST /detect with source=historical_replay; this module never runs a model and
never changes /detect semantics. Labels served here are DATASET AUTHOR labels
(pyro-sdis frame flags/boxes; Boreal clip-level label), not human review done
in this project. ruokolahti (aerial FINAL TEST) and pyro-sdis TEST sites are
refused by construction.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()
STATIC = Path(__file__).parent / "static"
SDIS_SEQ_JSON = "data/v3/sequences/sequences_val.json"
BOREAL_FRAMES = "data/v4/sequences/boreal_frames"
BOREAL_PARQUET = "data/v4/sequences/sequences_v4.parquet"
ALWAYS = ("val_force-06_cabanelle-125_20240403T101630",)      # the V5 shadow-demo sequence
PATTERNS = ("appears", "appears_disappears", "disappears", "all_neg", "flicker", "all_pos")
_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,160}$")
_registry = None


def install(reg):
    global _registry
    _registry = reg


class DemoRegistry:
    def __init__(self, settings, root: Path):
        self.cfg = settings
        self.root = Path(root)
        self.seqs: dict[str, dict] = {}
        rd = Path(settings.demo_runs_dir)
        self.runs_dir = rd if rd.is_absolute() else self.root / rd
        self.warnings: list[str] = []
        self.built = False

    # ------------------------------------------------------------------ build
    def build(self):
        self.seqs = {}
        try:
            self._build_sdis()
        except Exception as e:
            self.warnings.append(f"sdis sequences unavailable: {type(e).__name__}: {e}"[:200])
        try:
            self._build_boreal()
        except Exception as e:
            self.warnings.append(f"boreal clips unavailable: {type(e).__name__}: {e}"[:200])
        self.built = True

    def _build_sdis(self):
        seqs = json.load(open(self.root / SDIS_SEQ_JSON))
        by_id = {s["seq_id"]: s for s in seqs}
        if self.cfg.demo_sdis_sequences.strip():
            wanted = [x.strip() for x in self.cfg.demo_sdis_sequences.split(",") if x.strip()]
        else:   # deterministic short list: per label pattern the shortest 6..20-frame sequence
            wanted = list(ALWAYS)
            for pat in PATTERNS:
                c = [s for s in seqs if s["label_pattern"] == pat and 6 <= s["n_frames"] <= 20
                     and s["seq_id"] not in wanted]
                if c:
                    wanted.append(min(c, key=lambda s: (s["n_frames"], s["seq_id"]))["seq_id"])
        for sid in wanted:
            s = by_id.get(sid)
            if s is None:
                self.warnings.append(f"sdis seq_id not found: {sid}")
                continue
            if s.get("split") != "val":
                raise ValueError(f"{sid}: split {s.get('split')!r} is not val (TEST sites are never served)")
            frames = sorted(s["frames"], key=lambda f: f["ts"])
            fr = [{"idx": i, "name": f["image_name"], "ts": float(f["ts"]), "path": f["image_path"],
                   "gt_smoke": bool(f["smoke"]), "gt_boxes_xywhn": f.get("boxes_xywhn", [])}
                  for i, f in enumerate(frames)]
            self.seqs[sid] = {
                "id": sid, "kind": "pyro-sdis VAL", "site": s["site"], "camera": s["camera"],
                "label_pattern": s["label_pattern"], "n_frames": len(fr), "duration_s": s["duration_s"],
                "cadence": "~29 s sampled (bursts 1-4 s); real per-frame timestamps",
                "role": "pyro-sdis VAL site: calibration / model selection (PROTOCOL_V5 s2) - never training",
                "gt_source": "pyro-sdis dataset author labels (frame smoke flag + boxes); not reviewed in this project",
                "frames": fr}

    def _build_boreal(self):
        labels = {}
        try:
            import pandas as pd
            df = pd.read_parquet(self.root / BOREAL_PARQUET, columns=["seq_id", "seq_label", "split"])
            labels = df.drop_duplicates("seq_id").set_index("seq_id").to_dict("index")
        except Exception as e:
            self.warnings.append(f"boreal labels unavailable: {type(e).__name__}"[:120])
        for clip in (x.strip() for x in self.cfg.demo_boreal_clips.split(",")):
            if not clip:
                continue
            if "ruokolahti" in clip.lower():
                raise ValueError("ruokolahti is the aerial FINAL TEST (historical): refused")
            if not clip.lower().startswith("heinola"):
                raise ValueError(f"{clip}: only heinola (aerial VAL) clips are served by the demo")
            d = self.root / BOREAL_FRAMES / clip
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if not files:
                raise FileNotFoundError(f"no frames in {d}")
            lab = labels.get(f"boreal_{clip}", {})
            fr = [{"idx": i, "name": p.name, "ts": 1_700_000_000.0 + i, "path": str(p),
                   "gt_smoke": None, "gt_boxes_xywhn": []} for i, p in enumerate(files)]
            sid = f"boreal_{clip}"
            self.seqs[sid] = {
                "id": sid, "kind": "Boreal heinola (aerial VAL)", "site": "heinola", "camera": clip,
                "label_pattern": str(lab.get("seq_label", "unknown")), "n_frames": len(fr),
                "duration_s": len(fr) - 1, "cadence": "1 fps extracted frames; synthetic epoch 1700000000+idx",
                "role": "Boreal heinola: AERIAL calibration/VAL - never training (PROTOCOL_V5 s2)",
                "gt_source": f"Boreal clip-level dataset label ({lab.get('seq_label', 'unknown')}); per-frame boxes not served",
                "frames": fr}

    # ------------------------------------------------------------------ views
    def public(self) -> dict:
        out = []
        for sid, s in self.seqs.items():
            pub = {k: v for k, v in s.items() if k != "frames"}
            pub["frames"] = [{"idx": f["idx"], "name": f["name"], "ts": f["ts"], "gt_smoke": f["gt_smoke"],
                              "gt_boxes_xywhn": f["gt_boxes_xywhn"], "url": f"/demo/frame/{sid}/{f['idx']}"}
                             for f in s["frames"]]
            out.append(pub)
        return {"sequences": out, "warnings": self.warnings,
                "note": ("stored historical frames; the UI replays them through POST /detect with "
                         "source=historical_replay - never a field run")}

    def frame_path(self, sid: str, idx: int) -> Path:
        s = self.seqs.get(sid)
        if s is None or not (0 <= idx < len(s["frames"])):
            raise HTTPException(404, "unknown sequence or frame index")
        return Path(s["frames"][idx]["path"])

    def runs(self) -> list[dict]:
        out = []
        if not self.runs_dir.is_dir():
            return out
        for p in sorted(self.runs_dir.glob("*.json")):
            try:
                r = json.load(open(p))
                out.append({"run_id": p.stem, "seq_id": r.get("seq_id"), "mode": r.get("mode"),
                            "n_frames": len(r.get("frames", [])), "created_utc": r.get("created_utc"),
                            "model_version": r.get("model_version")})
            except Exception as e:
                out.append({"run_id": p.stem, "error": f"{type(e).__name__}"})
        return out

    def run(self, run_id: str) -> dict:
        if not _ID_RE.match(run_id):
            raise HTTPException(422, "bad run id")
        p = self.runs_dir / f"{run_id}.json"
        if not p.is_file():
            raise HTTPException(404, "unknown run")
        return json.load(open(p))

    def status(self) -> dict:
        return {"sequences": len(self.seqs), "runs_dir": str(self.runs_dir), "built": self.built,
                "warnings": self.warnings}


def _reg() -> DemoRegistry:
    if _registry is None or not _registry.built:
        raise HTTPException(503, "demo registry not ready")
    return _registry


@router.get("/demo")
def demo_page():
    return FileResponse(STATIC / "demo.html")


@router.get("/demo/sequences")
def demo_sequences():
    return _reg().public()


@router.get("/demo/frame/{sid}/{idx}")
def demo_frame(sid: str, idx: int):
    if not _ID_RE.match(sid):
        raise HTTPException(422, "bad sequence id")
    p = _reg().frame_path(sid, idx)
    if not p.is_file():
        raise HTTPException(404, "frame file missing")
    return FileResponse(p, media_type="image/jpeg")


@router.get("/demo/runs")
def demo_runs():
    return {"runs": _reg().runs()}


@router.get("/demo/runs/{run_id}")
def demo_run(run_id: str):
    return _reg().run(run_id)
