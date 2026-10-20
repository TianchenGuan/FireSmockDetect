"""RF-DETR (Large) inference backend, satisfying the src/serve/backends.py contract.

Model: RF-DETR-Large fine-tuned on our frozen pyro-sdis+D-Fire train list
(iteration-3 Track A, run rfdetrL_1280_e48). Code + Large weights are
Apache-2.0 (see reports/v3/trackA_notes.md §1); the `rfdetr` package is used
from the pinned clone in third_party/rf-detr (commit a57ad234).

Contract implemented (frozen for v3):
    detect(img_bgr) -> (list[Detection], timing_dict)
    filter_by_conf(dets, conf, tile_conf_offset=None) -> list[Detection]
    class_names: dict[int, str]
All detections carry source="full": this backend has no tiling pathway
(tiling was rejected on the iteration-2 lockbox for the YOLO backend, and
RF-DETR's recipe resizes the whole frame to a square anyway).

PREPROCESSING NOTE (important for parity with training/eval): RF-DETR resizes
the full frame to resolution x resolution (aspect ratio NOT preserved,
antialias=False bilinear -- matching the Albumentations resize used in
training) and normalizes with ImageNet mean/std. The steps below mirror
rfdetr.detr.RFDETR.predict() exactly, minus the `supervision` return type, so
that this backend can run in environments without `supervision` installed and
so that boxes come back as plain floats. Equivalence with the upstream
predict() is checked by scripts/v3/verify_rfdetr_backend.py (max |box| delta
and identical scores on real val images).

Raw scores are NOT calibrated probabilities; thresholds are only meaningful
against the val-selected grid (reports/v3/rfdetr_eval.md).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.serve.detector import Detection

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RFDETR_SRC = _REPO_ROOT / "third_party" / "rf-detr" / "src"

# ImageNet statistics used by RF-DETR (rfdetr.detr.RFDETR.means/stds).
_MEANS = (0.485, 0.456, 0.406)
_STDS = (0.229, 0.224, 0.225)


def _ensure_rfdetr_importable() -> None:
    """Import `rfdetr` from site-packages, else from the pinned in-tree clone."""
    try:
        import rfdetr  # noqa: F401
    except ModuleNotFoundError:
        if not _RFDETR_SRC.exists():
            raise
        sys.path.insert(0, str(_RFDETR_SRC))
        import rfdetr  # noqa: F401


class RfDetrBackend:
    """Single-frame RF-DETR detector wrapper.

    Args:
        weights_path: an RF-DETR checkpoint (`checkpoint_best_ema.pth` /
            `checkpoint_best_total.pth` / `last.ckpt`). Loaded through
            `RFDETR.from_checkpoint`, which infers the variant + num_classes
            from the file itself.
        device: "cpu" or "cuda[:i]". The checkpoint records device="cuda"; we
            always override it explicitly so CPU serving works.
        imgsz: inference resolution (square). Must be divisible by
            patch_size * num_windows (= 32 for RF-DETR-Large) -> use 1280,
            the resolution the model was fine-tuned at.
        conf_floor: scores below this are dropped inside detect(); keep it low
            (0.02-0.05) so a threshold sweep can be done post-hoc.
        class_names: optional {id: name} override. Defaults to the checkpoint's
            own class names, else {0: "smoke"} for our single-class models.
    """

    def __init__(self, weights_path: str, device: str = "cpu", imgsz: int = 1280,
                 conf_floor: float = 0.05, class_names: dict | None = None,
                 trust_checkpoint: bool = True, **kw):
        _ensure_rfdetr_importable()
        from rfdetr.detr import RFDETR

        self.weights_path = str(weights_path)
        self.device = torch.device(device)
        self.imgsz = int(imgsz)
        self.conf_floor = float(conf_floor)
        # kept only so the contract matches SmokeDetector.filter_by_conf; this
        # backend never emits source="tile" detections.
        self.tile_conf_offset = float(kw.get("tile_conf_offset", 0.0))

        # trust_checkpoint=True: our own training artifacts, produced by the
        # pinned container (Lightning ckpts need full unpickling).
        self._rf = RFDETR.from_checkpoint(self.weights_path,
                                          trust_checkpoint=trust_checkpoint,
                                          device=str(device), resolution=self.imgsz)
        block = (getattr(self._rf.model_config, "patch_size", 16)
                 * getattr(self._rf.model_config, "num_windows", 1))
        if self.imgsz % block:
            raise ValueError(f"imgsz {self.imgsz} not divisible by patch_size*num_windows={block}")

        ctx = self._rf.model
        self._net = ctx.model
        self._net.to(self.device).eval()
        ctx.device = self.device
        self._postprocess = ctx.postprocess

        ckpt_names = list(getattr(ctx, "class_names", None) or [])
        if class_names:
            self.class_names = {int(k): str(v) for k, v in class_names.items()}
        elif ckpt_names:
            self.class_names = dict(enumerate(ckpt_names))
        else:
            self.class_names = {0: "smoke"}
        self._mean = torch.tensor(_MEANS, device=self.device).view(3, 1, 1)
        self._std = torch.tensor(_STDS, device=self.device).view(3, 1, 1)

    # ---------------------------------------------------------------- contract
    def filter_by_conf(self, dets: list[Detection], conf: float,
                       tile_conf_offset: float | None = None) -> list[Detection]:
        """Same semantics as SmokeDetector.filter_by_conf (no tiles here)."""
        off = self.tile_conf_offset if tile_conf_offset is None else tile_conf_offset
        return [d for d in dets
                if d.score >= (conf + off if d.source == "tile" else conf)]

    @torch.inference_mode()
    def detect(self, img: np.ndarray) -> tuple[list[Detection], dict]:
        """img: HWC uint8 BGR (cv2.imread order), same as SmokeDetector.detect."""
        t0 = time.perf_counter()
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        h, w = img.shape[:2]
        rgb = np.ascontiguousarray(img[:, :, ::-1])          # BGR -> RGB
        chw = torch.from_numpy(rgb.transpose(2, 0, 1))       # uint8 CHW view
        chw = chw.to(self.device, non_blocking=(self.device.type == "cuda"))
        x = chw.to(torch.get_default_dtype()).div_(255.0)
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(self.imgsz, self.imgsz),
            mode="bilinear", align_corners=False, antialias=False)
        x = (x - self._mean) / self._std
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_pre = time.perf_counter()

        preds = self._net(x)
        if isinstance(preds, tuple):
            preds = {"pred_boxes": preds[0], "pred_logits": preds[1]}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_inf = time.perf_counter()

        target_sizes = torch.tensor([[h, w]], device=self.device)
        res = self._postprocess(preds, target_sizes=target_sizes,
                                score_threshold=self.conf_floor)[0]
        keep = (res["scores"] > self.conf_floor).nonzero(as_tuple=True)[0]
        boxes = res["boxes"][keep].float().cpu().numpy()
        scores = res["scores"][keep].float().cpu().numpy()
        labels = res["labels"][keep].cpu().numpy()
        dets = [Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                          float(s), int(c),
                          self.class_names.get(int(c), str(int(c))), "full")
                for b, s, c in zip(boxes, scores, labels)]
        t1 = time.perf_counter()
        timing = {"latency_ms": round((t1 - t0) * 1000, 1),
                  "preprocess_ms": round((t_pre - t0) * 1000, 1),
                  "infer_ms": round((t_inf - t_pre) * 1000, 1),
                  "postprocess_ms": round((t1 - t_inf) * 1000, 1),
                  # contract parity with the yolo backend's timing keys
                  "full_pass_ms": round((t_inf - t0) * 1000, 1),
                  "tile_pass_ms": 0.0, "n_tiles": 0}
        return dets, timing
