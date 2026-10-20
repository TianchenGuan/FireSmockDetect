"""FAST / ACCURATE serving modes (iteration 5b).

FAST     = the single primary backend; /detect is byte-for-byte today's response
           (no new keys). Nothing in this module runs for a FAST request.
ACCURATE = N configured component models (SMOKE_ACCURATE_MODELS) run in PARALLEL
           THREADS on the same full-frame image. The response carries, per model,
           its raw boxes + scores, latency and its OWN alert state (own thresholds,
           own per-device history), plus fused boxes and ONE final alert.

Score scales differ between models (DINOv3 B4 saturates near 1.0: its val-selected
thresholds are 0.93/0.82 where YOLO expB uses 0.30/0.15), so boxes are fused on a
common scale. Two named scales exist:
  alert_equivalent (builtin): each model's raw confidence is mapped piecewise-
      linearly so that its own (conf_seq, conf_single) land on the FAST reference
      pair (settings.seq_frame_conf, settings.single_frame_alert_conf).
  calibrated (ensemble track, scripts/v5b/fusion.py + models/v5b/fusion/
      accurate_recipe.json): raw score -> neg-image-FP quantile on pyro-sdis VAL
      negatives; fused with the recipe's WBF; alert thresholds = recipe
      thresholds_oof at the configured target FP levels.
Neither is a calibrated probability of smoke; both are reported by name in
`score_semantics` / `fusion.scale`, and neither changes a model's own alert.

Fusion methods (settings.accurate_fusion):
  or       (default) union of all boxes + cross-model NMS on alert-equivalent
           scores; final alert = the most severe per-model alert.
  wbf      recipe applicable (component names == recipe model set, module
           importable) -> recipe WBF on the calibrated scale; otherwise builtin
           WBF on alert-equivalent scores (said so in fusion.fallback_reason).
  learned  recipe + module required; otherwise falls back to `or`, loudly.
The recipe file is loaded whenever present. Its method is adopted as the default
ONLY when it is applicable to the configured component set (a half-applied
recipe is worse than none); SMOKE_ACCURATE_FUSION set explicitly always wins.
"""
from __future__ import annotations

import importlib.util
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .alerting import AlertEngine
from .backends import AVAILABLE, load_backend

MODES = ("fast", "accurate")
FUSIONS = ("or", "wbf", "learned")
SEVERITY = {"none": 0, "watch": 1, "provisional": 2, "confirmed": 3}
_RECIPE_METHODS = {"wbf": "wbf", "union_nms": "or", "or": "or", "learned": "learned"}

_FLOAT_KEYS = {"conf_single", "conf_seq", "conf_floor", "lora_alpha", "tile_overlap", "merge_iou"}
_INT_KEYS = {"imgsz", "lora_r", "width", "height", "tile_size"}
_BOOL_KEYS = {"lora_merge"}
_STR_KEYS = {"version", "device", "backbone_dir"}
_BACKEND_KW = {"backbone_dir", "lora_r", "lora_alpha", "lora_merge", "width", "height",
               "tile_size", "tile_overlap", "merge_iou"}
_NAME_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")


@dataclass
class ModelSpec:
    name: str
    backend: str
    weights: str          # as written in the spec (resolved against ROOT at load)
    version: str
    conf_single: float
    conf_seq: float
    conf_floor: float
    imgsz: int
    device: str
    kw: dict = field(default_factory=dict)   # backend-specific load kwargs


def parse_specs(text: str, *, conf_single: float, conf_seq: float, conf_floor: float,
                imgsz: int, device: str) -> list[ModelSpec]:
    """Parse SMOKE_ACCURATE_MODELS. Grammar per entry (';'-separated):
         name=backend:weights[,key=value...]       backend omitted -> yolo
    Unknown keys, duplicate names, unknown backends -> ValueError (startup fails
    loudly rather than silently serving a different ensemble)."""
    specs: list[ModelSpec] = []
    for raw in (e.strip() for e in (text or "").split(";")):
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"accurate model spec {raw!r}: expected name=backend:weights[,key=value...]")
        name, rest = raw.split("=", 1)
        name = name.strip()
        if not name or set(name) - _NAME_OK:
            raise ValueError(f"accurate model name {name!r}: use [A-Za-z0-9_.-]")
        if any(s.name == name for s in specs):
            raise ValueError(f"duplicate accurate model name {name!r}")
        parts = [p.strip() for p in rest.split(",")]
        target = parts[0]
        if ":" in target:
            backend, weights = (x.strip() for x in target.split(":", 1))
        else:
            backend, weights = "yolo", target
        backend = backend.lower()
        if backend not in AVAILABLE:
            raise ValueError(f"accurate model {name!r}: unknown backend {backend!r}; available {AVAILABLE}")
        if not weights:
            raise ValueError(f"accurate model {name!r}: empty weights path")
        opts: dict = {}
        for p in parts[1:]:
            if "=" not in p:
                raise ValueError(f"accurate model {name!r}: option {p!r} is not key=value")
            k, v = (x.strip() for x in p.split("=", 1))
            if k in _FLOAT_KEYS:
                opts[k] = float(v)
            elif k in _INT_KEYS:
                opts[k] = int(v)
            elif k in _BOOL_KEYS:
                opts[k] = v.lower() in ("1", "true", "yes", "on")
            elif k in _STR_KEYS:
                opts[k] = v
            else:
                raise ValueError(f"accurate model {name!r}: unknown option {k!r}")
        specs.append(ModelSpec(
            name=name, backend=backend, weights=weights,
            version=opts.get("version") or weights,
            conf_single=float(opts.get("conf_single", conf_single)),
            conf_seq=float(opts.get("conf_seq", conf_seq)),
            conf_floor=float(opts.get("conf_floor", conf_floor)),
            imgsz=int(opts.get("imgsz", imgsz)),
            device=str(opts.get("device") or device),
            kw={k: opts[k] for k in _BACKEND_KW if k in opts}))
    return specs


# --------------------------------------------------------------------------- scores / geometry
def alert_equivalent(score: float, conf_seq: float, conf_single: float,
                     ref_seq: float, ref_single: float) -> float:
    """Piecewise-linear map: [0,conf_seq]->[0,ref_seq], [conf_seq,conf_single]->
    [ref_seq,ref_single], [conf_single,1]->[ref_single,1]. Identity when the
    model's thresholds equal the reference pair."""
    s = min(max(float(score), 0.0), 1.0)
    lo, hi = float(conf_seq), float(conf_single)
    if hi <= lo:
        hi = lo + 1e-6
    if s <= lo:
        out = (s / lo) * ref_seq if lo > 0 else ref_seq
    elif s <= hi:
        out = ref_seq + (s - lo) / (hi - lo) * (ref_single - ref_seq)
    else:
        out = ref_single + ((s - hi) / (1.0 - hi) * (1.0 - ref_single) if hi < 1.0 else 0.0)
    return round(min(max(out, 0.0), 1.0), 6)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _fused_dict(box, score, cls_id, cls_name, models: dict, method: str) -> dict:
    return {"score": round(float(score), 4), "cls_id": int(cls_id), "cls_name": cls_name,
            "source": f"fused_{method}",
            "bbox_xyxy": [round(float(v), 1) for v in box],
            "models": sorted(models), "raw_scores": {m: round(float(v), 4) for m, v in models.items()}}


def fuse_or(cands: list[dict], iou_thr: float) -> list[dict]:
    """Union + cross-model NMS on alert-equivalent scores; each kept box lists the
    models that support it (max raw score per model)."""
    out = []
    for c in sorted(cands, key=lambda c: -c["score_norm"]):
        for o in out:
            if o["cls_id"] == c["cls_id"] and iou(o["bbox"], c["bbox"]) >= iou_thr:
                o["models"][c["model"]] = max(o["models"].get(c["model"], 0.0), c["score_raw"])
                break
        else:
            out.append({"bbox": list(c["bbox"]), "score": c["score_norm"], "cls_id": c["cls_id"],
                        "cls_name": c["cls_name"], "models": {c["model"]: c["score_raw"]}})
    return [_fused_dict(o["bbox"], o["score"], o["cls_id"], o["cls_name"], o["models"], "or") for o in out]


def fuse_wbf(cands: list[dict], iou_thr: float, model_names: list[str], weights: dict | None = None) -> list[dict]:
    """Builtin weighted box fusion on alert-equivalent scores. Cluster by IoU,
    coordinates = score-weighted mean, fused score = sum over models of (weight x
    best member score) / sum of weights over ALL models (a silent model lowers
    the fused score, as in standard WBF 'avg')."""
    weights = weights or {}
    wsum_all = sum(float(weights.get(n, 1.0)) for n in model_names) or 1.0
    clusters: list[dict] = []
    for c in sorted(cands, key=lambda c: -c["score_norm"]):
        best, best_iou = None, iou_thr
        for cl in clusters:
            if cl["cls_id"] != c["cls_id"]:
                continue
            i = iou(cl["box"], c["bbox"])
            if i >= best_iou:
                best, best_iou = cl, i
        if best is None:
            clusters.append({"cls_id": c["cls_id"], "cls_name": c["cls_name"], "members": [c], "box": list(c["bbox"])})
        else:
            best["members"].append(c)
            ws = [m["score_norm"] * float(weights.get(m["model"], 1.0)) for m in best["members"]]
            tot = sum(ws) or 1.0
            best["box"] = [sum(w * m["bbox"][k] for w, m in zip(ws, best["members"])) / tot for k in range(4)]
    out = []
    for cl in clusters:
        per_model: dict = {}
        for m in cl["members"]:
            per_model[m["model"]] = max(per_model.get(m["model"], (0.0, 0.0)), (m["score_norm"], m["score_raw"]))
        score = sum(float(weights.get(mn, 1.0)) * sn for mn, (sn, _) in per_model.items()) / wsum_all
        out.append(_fused_dict(cl["box"], score, cl["cls_id"], cl["cls_name"],
                               {mn: sr for mn, (_, sr) in per_model.items()}, "wbf"))
    return sorted(out, key=lambda d: -d["score"])


# --------------------------------------------------------------------------- ensemble
class AccurateEnsemble:
    def __init__(self, settings, root: Path):
        self.cfg = settings
        self.root = Path(root)
        self.specs = parse_specs(settings.accurate_models, conf_single=settings.single_frame_alert_conf,
                                 conf_seq=settings.seq_frame_conf, conf_floor=settings.conf_floor,
                                 imgsz=settings.imgsz, device=settings.accurate_device or settings.device)
        self.enabled = bool(self.specs)
        self.ref_single = float(settings.single_frame_alert_conf)
        self.ref_seq = float(settings.seq_frame_conf)
        self.iou = float(settings.accurate_fusion_iou)
        self.models: dict[str, object] = {}
        self.alerts: dict[str, AlertEngine] = {}
        self.fused_alerts: AlertEngine | None = None
        self.fused_thresholds = {"conf_single": self.ref_single, "conf_seq": self.ref_seq, "scale": "alert_equivalent"}
        self.method = str(settings.accurate_fusion).lower()
        self.method_requested = self.method
        self.scale = "alert_equivalent"
        self.fallback_reason: str | None = None
        self.recipe: dict | None = None
        self.recipe_path: str | None = None
        self.recipe_meta: dict | None = None
        self.recipe_models: list[str] = []
        self.recipe_weights: dict = {}
        self.recipe_applicable = False
        self._ext = None                 # scripts/v5b/fusion.py module
        self._ext_path: str | None = None
        self._rec = None                 # FusionRecipe instance
        self._pool: ThreadPoolExecutor | None = None
        self._run_lock = threading.Lock()
        self._stat_lock = threading.Lock()
        self.requests = 0
        self.ext_failures = 0
        self._lat_sum = 0.0
        self._lat_max = 0.0

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    @property
    def label(self) -> str:
        return f"accurate:{self.method}[" + "+".join(s.version for s in self.specs) + "]"

    # ------------------------------------------------------------------ lifecycle
    def _resolve(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else self.root / q

    def _load_recipe(self):
        cfg = self.cfg
        rp = self._resolve(cfg.accurate_recipe_path)
        rmethod = None
        if rp.is_file():
            self.recipe = json.load(open(rp))
            self.recipe_path = str(rp)
            fus = self.recipe.get("fusion")
            models = self.recipe.get("models")
            self.recipe_models = list(models.keys()) if isinstance(models, dict) else []
            if isinstance(fus, dict):                      # ensemble-track format (accurate_recipe.json)
                rmethod = str(fus.get("method", "")).lower()
                self.recipe_weights = dict(fus.get("weights") or {})
                riou = fus.get("iou_thr")
                params = {k: v for k, v in fus.items() if k != "weights"}
            else:                                          # simple format {"method","iou","weights"}
                rmethod = str(self.recipe.get("method", "")).lower()
                self.recipe_weights = dict(self.recipe.get("weights") or {})
                riou = self.recipe.get("iou", self.recipe.get("iou_thr"))
                params = {"method": rmethod, "iou": riou}
            if riou is not None and "accurate_fusion_iou" not in cfg.model_fields_set:
                self.iou = float(riou)
            self.recipe_meta = {"name": self.recipe.get("recipe"), "scenario": self.recipe.get("scenario"),
                                "created_utc": self.recipe.get("created_utc"), "models": self.recipe_models,
                                "method": rmethod, "params": params, "score_scale": self.recipe.get("score_scale"),
                                "thresholds_oof": self.recipe.get("thresholds_oof"),
                                "threshold_note": self.recipe.get("threshold_note")}
        # external fusion module (ensemble track)
        fp = self.root / "scripts/v5b/fusion.py"
        if fp.is_file():
            try:
                spec = importlib.util.spec_from_file_location("v5b_fusion", fp)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if callable(getattr(mod, "fuse", None)) and hasattr(mod, "FusionRecipe"):
                    self._ext, self._ext_path = mod, str(fp)
            except Exception as e:  # a broken optional module must not take the service down
                self.fallback_reason = f"scripts/v5b/fusion.py import failed: {type(e).__name__}: {e}"[:300]
        names_match = bool(self.recipe_models) and set(self.names) == set(self.recipe_models)
        if self.recipe is not None and self._ext is not None and names_match and isinstance(self.recipe.get("fusion"), dict):
            try:
                self._rec = self._ext.FusionRecipe(self.recipe)
                self.recipe_applicable = True
            except Exception as e:
                self.fallback_reason = f"recipe rejected by fusion.py: {type(e).__name__}: {e}"[:300]
        elif self.recipe is not None and not names_match:
            self.fallback_reason = (f"recipe {self.recipe_meta and self.recipe_meta.get('name')} expects models "
                                    f"{self.recipe_models} (full-frame parallel execution of every component); "
                                    f"configured {self.names} -> recipe not applied")
        # method: explicit env wins; else an APPLICABLE recipe's method; else the file default
        if "accurate_fusion" not in cfg.model_fields_set and self.recipe_applicable and rmethod in _RECIPE_METHODS:
            self.method = _RECIPE_METHODS[rmethod]
        self.method_requested = self.method
        if self.method not in FUSIONS:
            raise ValueError(f"accurate_fusion {self.method!r} not in {FUSIONS}")
        # scale + fused-alert thresholds
        if self.method in ("wbf", "learned") and self.recipe_applicable:
            th = (self.recipe.get("thresholds_oof") or {})
            ks, kq = f"fp{cfg.accurate_fused_fp_single:.2f}", f"fp{cfg.accurate_fused_fp_seq:.2f}"
            if ks in th and kq in th:
                self.scale = "calibrated"
                self.fused_thresholds = {"conf_single": float(th[ks]), "conf_seq": float(th[kq]), "scale": "calibrated",
                                         "target_fp_single": cfg.accurate_fused_fp_single,
                                         "target_fp_seq": cfg.accurate_fused_fp_seq, "source": "recipe thresholds_oof"}
            else:
                self.recipe_applicable = False
                self.fallback_reason = f"recipe thresholds_oof lacks {ks}/{kq}; keys {sorted(th)}"
        if self.method == "learned" and self.scale != "calibrated":
            self.fallback_reason = (self.fallback_reason or "no applicable recipe / fusion module") + " -> 'learned' unavailable, using 'or'"
            self.method = "or"
        if self.method == "wbf" and self.scale != "calibrated":
            self.fallback_reason = (self.fallback_reason or "no applicable recipe / fusion module") + " -> builtin wbf on alert-equivalent scores"

    def start(self):
        if not self.enabled:
            return
        self._load_recipe()
        for s in self.specs:
            wp = self._resolve(s.weights)
            if not wp.is_file():
                raise FileNotFoundError(f"accurate model {s.name!r}: weights not found: {wp}")
            self.models[s.name] = load_backend(
                s.backend, weights_path=str(wp), device=s.device, imgsz=s.imgsz,
                conf_floor=s.conf_floor, tile_size=s.kw.get("tile_size", 0),
                tile_overlap=s.kw.get("tile_overlap", self.cfg.tile_overlap),
                merge_iou=s.kw.get("merge_iou", self.cfg.merge_iou),
                backbone_dir=s.kw.get("backbone_dir") or None,
                lora_r=s.kw.get("lora_r"), lora_alpha=s.kw.get("lora_alpha"),
                lora_merge=bool(s.kw.get("lora_merge", False)),
                width=s.kw.get("width") or None, height=s.kw.get("height") or None)
            self.alerts[s.name] = AlertEngine(self.cfg.model_copy(update={
                "single_frame_alert_conf": s.conf_single, "seq_frame_conf": s.conf_seq}))
        self.fused_alerts = AlertEngine(self.cfg.model_copy(update={
            "single_frame_alert_conf": self.fused_thresholds["conf_single"],
            "seq_frame_conf": self.fused_thresholds["conf_seq"]}))
        self._pool = ThreadPoolExecutor(max_workers=len(self.specs), thread_name_prefix="accurate")

    def stop(self):
        if self._pool:
            self._pool.shutdown(wait=False)
            self._pool = None

    @property
    def loaded(self) -> bool:
        return self.enabled and len(self.models) == len(self.specs) and self._pool is not None

    # ------------------------------------------------------------------ inference
    def _detect_one(self, name: str, img):
        return self.models[name].detect(img)

    def _calibrated(self, name: str, scores: list[float]) -> list[float] | None:
        if self._rec is None or name not in self._rec.calibrators or not scores:
            return None if self._rec is None else []
        import numpy as np
        return [round(float(v), 6) for v in self._rec.calibrators[name](np.asarray(scores, dtype=float))]

    def run(self, img, *, device_id: str, ts: float, phash) -> dict:
        if not self.loaded:
            raise RuntimeError("accurate ensemble not loaded")
        h, w = img.shape[:2]
        with self._run_lock:                      # one request at a time per model object
            t0 = time.perf_counter()
            futs = {s.name: self._pool.submit(self._detect_one, s.name, img) for s in self.specs}
            raw: dict[str, tuple] = {}
            errors: dict[str, str] = {}
            for name, f in futs.items():
                try:
                    raw[name] = f.result()
                except Exception as e:
                    errors[name] = f"{type(e).__name__}: {e}"[:300]
            wall_ms = round((time.perf_counter() - t0) * 1000, 1)
            if not raw:
                raise RuntimeError(f"all accurate models failed: {errors}")

            per_model: dict[str, dict] = {}
            cands: list[dict] = []
            for s in self.specs:
                if s.name in errors:
                    per_model[s.name] = {"backend": s.backend, "model_version": s.version, "error": errors[s.name],
                                         "detections": [], "n_smoke": 0, "max_smoke_score": 0.0,
                                         "max_alert_equivalent": 0.0, "alert": None, "latency_ms": None}
                    continue
                dets, timing = raw[s.name]
                det = self.models[s.name]
                smoke = [d for d in dets if d.cls_name.lower() == "smoke" or len(det.class_names) == 1]
                smoke_ids = {id(d) for d in smoke}
                max_raw = max((d.score for d in smoke), default=0.0)
                alert = self.alerts[s.name].update(device_id, ts, max_raw, phash=phash)
                cal = self._calibrated(s.name, [d.score for d in dets])
                items = []
                for k, d in enumerate(dets):
                    dd = d.as_dict()
                    dd["score_alert_equivalent"] = alert_equivalent(d.score, s.conf_seq, s.conf_single,
                                                                    self.ref_seq, self.ref_single)
                    if cal is not None:
                        dd["score_calibrated"] = cal[k]
                    items.append(dd)
                    if id(d) in smoke_ids:
                        cands.append({"bbox": [d.x1, d.y1, d.x2, d.y2], "score_raw": float(d.score),
                                      "score_norm": dd["score_alert_equivalent"], "model": s.name,
                                      "cls_id": d.cls_id, "cls_name": d.cls_name})
                per_model[s.name] = {
                    "backend": s.backend, "model_version": s.version, "weights": s.weights,
                    "thresholds": {"conf_single": s.conf_single, "conf_seq": s.conf_seq, "conf_floor": s.conf_floor},
                    "detections": items, "n_smoke": len(smoke),
                    "max_smoke_score": round(max_raw, 4),
                    "max_alert_equivalent": alert_equivalent(max_raw, s.conf_seq, s.conf_single, self.ref_seq, self.ref_single),
                    **({"max_calibrated": (max([c for c, d in zip(cal, dets) if id(d) in smoke_ids], default=0.0))}
                       if cal is not None else {}),
                    "alert": {k: alert.get(k) for k in ("state", "reason", "window_hits", "window_size",
                                                         "history_reset", "cooldown_active")},
                    "latency_ms": timing.get("latency_ms"), "timing": timing}

            fused, image_score, fusion_info = self._fuse(cands, per_model, w, h)
            states = {n: m["alert"]["state"] for n, m in per_model.items() if m.get("alert")}
            if fusion_info["method"] == "or":
                worst = max(states, key=lambda n: SEVERITY[states[n]])
                final = self._final_from_model(per_model[worst], worst, states)
            else:
                final = self.fused_alerts.update(device_id, ts, image_score, phash=phash)
                final["by_model"] = states
                final["decided_by"] = (f"fused AlertEngine on the {fusion_info['scale']} scale "
                                       f"({fusion_info['method']}; thresholds {self.fused_thresholds['conf_single']}/"
                                       f"{self.fused_thresholds['conf_seq']})")
                final["note"] = ("Smoke alert from a detector ensemble, not an independently confirmed wildfire "
                                 "report. Absence of alert on low-quality/missing input does not mean 'safe'.")
            with self._stat_lock:
                self.requests += 1
                self._lat_sum += wall_ms
                self._lat_max = max(self._lat_max, wall_ms)
            return {"detections": fused, "max_smoke_score": round(float(image_score), 4), "alert": final,
                    "models": per_model, "fusion": fusion_info, "latency_ms": wall_ms,
                    "full_pass_ms": max((m.get("latency_ms") or 0.0) for m in per_model.values()),
                    "errors": errors}

    def _final_from_model(self, m: dict, name: str, states: dict) -> dict:
        a = dict(m["alert"])
        a["reason"] = f"{name}: {a['reason']}"
        a["by_model"] = states
        a["decided_by"] = f"most severe per-model alert (or-fusion): {name}"
        a["note"] = ("Smoke alert from a detector ensemble, not an independently confirmed wildfire "
                     "report. Absence of alert on low-quality/missing input does not mean 'safe'.")
        return a

    def _fuse(self, cands, per_model, w, h):
        info = {"method": self.method, "method_requested": self.method_requested, "impl": "builtin",
                "scale": "alert_equivalent", "iou": self.iou, "recipe_path": self.recipe_path,
                "recipe_applied": False, "fallback_reason": self.fallback_reason,
                "normalization": ("alert_equivalent: per-model raw score mapped piecewise-linearly so "
                                  f"(conf_seq, conf_single) -> ({self.ref_seq}, {self.ref_single})"),
                "final_alert_thresholds": dict(self.fused_thresholds)}
        if self.method in ("wbf", "learned") and self.scale == "calibrated" and self._rec is not None:
            try:
                fused, score = self._fuse_recipe(per_model, w, h)
                info.update({"impl": self._ext_path, "scale": "calibrated", "recipe_applied": True,
                             "normalization": self.recipe_meta.get("score_scale") if self.recipe_meta else "calibrated"})
                return fused, score, info
            except Exception as e:
                with self._stat_lock:
                    self.ext_failures += 1
                info["fallback_reason"] = f"recipe fuse() failed: {type(e).__name__}: {e}"[:200] + " -> builtin"
                info["final_alert_thresholds"] = {"conf_single": self.ref_single, "conf_seq": self.ref_seq, "scale": "alert_equivalent"}
                if self.method == "learned":
                    info["method"] = "or"
        if info["method"] == "wbf":
            fused = fuse_wbf(cands, self.iou, self.names, self.recipe_weights or None)
        else:
            fused = fuse_or(cands, self.iou)
        score = max((f["score"] for f in fused if f["cls_name"].lower() == "smoke"), default=0.0)
        return fused, score, info

    def _fuse_recipe(self, per_model, w, h):
        """scripts/v5b/fusion.py: normalized xyxy + raw score per model (empty list =
        model ran and found nothing) -> fused boxes on the calibrated scale; image
        score per the recipe's image_level method."""
        pm = {}
        for n in self.names:
            m = per_model.get(n) or {}
            pm[n] = [[d["bbox_xyxy"][0] / w, d["bbox_xyxy"][1] / h, d["bbox_xyxy"][2] / w, d["bbox_xyxy"][3] / h, d["score"]]
                     for d in m.get("detections", []) if d["cls_name"].lower() == "smoke"]
        boxes = self._ext.fuse(pm, self._rec, strict=True)
        score = float(self._ext.image_score(pm, self._rec, strict=True))
        fused = []
        for b in boxes:
            px = [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
            support = {}
            for n, m in per_model.items():
                for d in m.get("detections", []):
                    if d["cls_name"].lower() == "smoke" and iou(px, d["bbox_xyxy"]) >= self.iou:
                        support[n] = max(support.get(n, 0.0), d["score"])
            fused.append(_fused_dict(px, b[4], 0, "smoke", support, self.method))
        return fused, score

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        base = {"enabled": self.enabled}
        if not self.enabled:
            base["models"] = []
            return base
        models = []
        for s in self.specs:
            m = {"name": s.name, "backend": s.backend, "model_version": s.version, "weights_path": str(self._resolve(s.weights)),
                 "conf_single": s.conf_single, "conf_seq": s.conf_seq, "conf_floor": s.conf_floor,
                 "imgsz": s.imgsz, "device": s.device, "loaded": s.name in self.models}
            det = self.models.get(s.name)
            if det is not None and hasattr(det, "info"):
                m["backend_info"] = det.info()
            models.append(m)
        with self._stat_lock:
            n = self.requests
            base.update({
                "label": self.label, "loaded": self.loaded, "models": models,
                "fusion": {"method": self.method, "method_requested": self.method_requested, "scale": self.scale,
                           "iou": self.iou, "recipe_path": self.recipe_path, "recipe_applied": self.recipe_applicable,
                           "recipe": self.recipe_meta, "external_module": self._ext_path,
                           "fallback_reason": self.fallback_reason, "external_failures": self.ext_failures,
                           "final_alert_thresholds": dict(self.fused_thresholds),
                           "reference_thresholds": {"conf_single": self.ref_single, "conf_seq": self.ref_seq}},
                "requests": n,
                "latency_ms_mean": round(self._lat_sum / n, 1) if n else None,
                "latency_ms_max": round(self._lat_max, 1) if n else None})
        return base
