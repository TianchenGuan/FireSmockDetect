"""KD trainer for V6 arm E: YOLO11n student + frozen YOLO detector teacher.

Response-level distillation on the raw detection-head maps (per-scale
(B, 4*reg_max+nc, H, W)): soft-BCE on class logits against the teacher's
sigmoid probabilities everywhere, plus temperature-KL on the DFL box
distributions where the teacher is confident. The ORIGINAL supervised
v8DetectionLoss is kept at full weight — KD is additive, so trusted labels
are never overridden by teacher errors (brief §6).

Teacher and student must share head geometry (reg_max=16, nc=1, 3 scales,
same imgsz) — true for YOLO11{n,s,m} pairs.

Config via environment (survives ultralytics DDP relaunch):
  KD_TEACHER   path to teacher .pt (required)
  KD_LAMBDA    overall KD weight            (default 1.0)
  KD_CLS_W     class-KD weight              (default 4.0)
  KD_BOX_W     box-DFL-KD weight            (default 1.0)
  KD_TAU       DFL KL temperature           (default 2.0)
  KD_FG_T      teacher prob to count as fg  (default 0.25)

This file must be importable by name in DDP subprocesses: the launcher adds
its directory to PYTHONPATH.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer


class KDTrainer(DetectionTrainer):
    def _setup_train(self, *a, **kw):
        super()._setup_train(*a, **kw)
        tpath = os.environ["KD_TEACHER"]
        self.kd_lambda = float(os.environ.get("KD_LAMBDA", "1.0"))
        self.kd_cls_w = float(os.environ.get("KD_CLS_W", "4.0"))
        self.kd_box_w = float(os.environ.get("KD_BOX_W", "1.0"))
        self.kd_tau = float(os.environ.get("KD_TAU", "2.0"))
        self.kd_fg_t = float(os.environ.get("KD_FG_T", "0.25"))
        from ultralytics import YOLO
        t = YOLO(tpath)
        t_nc = t.model.yaml.get("nc", getattr(t.model.model[-1], "nc", None))
        s_m = self.model.module if hasattr(self.model, "module") else self.model
        s_nc = s_m.yaml.get("nc", None)
        assert t_nc == s_nc, f"teacher nc={t_nc} != student nc={s_nc}"
        self.teacher = t.model.to(self.device).eval().float()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.kd_log = {"n": 0, "cls": 0.0, "box": 0.0}
        # wrap the criterion: model.loss() calls self.criterion(preds, batch)
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        if base_model.criterion is None:
            base_model.criterion = base_model.init_criterion()
        base_crit = base_model.criterion
        trainer = self

        class _KDCrit:
            def __init__(self):
                self.hyp = base_crit.hyp  # ultralytics inspects this

            def __getattr__(self, k):
                return getattr(base_crit, k)

            def __call__(self, preds, batch):
                loss, items = base_crit(preds, batch)
                kd = trainer._kd_term(preds, batch)
                return loss + kd * batch["img"].shape[0], items

        base_model.criterion = _KDCrit()

    @torch.no_grad()
    def _teacher_maps(self, imgs):
        out = self.teacher(imgs)
        # eval-mode Detect returns (decoded, raw_maps_list)
        if isinstance(out, (tuple, list)) and isinstance(out[1], (list, tuple)):
            return [m.float() for m in out[1]]
        return [m.float() for m in out]

    def _kd_term(self, preds, batch):
        maps_s = preds[1] if (isinstance(preds, tuple)) else preds
        maps_t = self._teacher_maps(batch["img"].float())
        reg = 16 * 4
        cls_tot, box_tot, fg_tot = 0.0, 0.0, 0
        n_anchor = 0
        for ms, mt in zip(maps_s, maps_t):
            if ms.shape[-2:] != mt.shape[-2:]:
                continue
            b, _, h, w = ms.shape
            ms = ms.float()                     # AMP: keep KD math in fp32
            s_box, s_cls = ms[:, :reg], ms[:, reg:]
            t_box, t_cls = mt[:, :reg], mt[:, reg:]
            p_t = t_cls.sigmoid()
            cls_tot = cls_tot + F.binary_cross_entropy_with_logits(
                s_cls, p_t, reduction="sum")
            n_anchor += s_cls.numel()
            fg = (p_t.max(1).values > self.kd_fg_t)          # (B,H,W)
            nfg = int(fg.sum())
            if nfg:
                tau = self.kd_tau
                sb = s_box.view(b, 4, 16, h, w).permute(0, 3, 4, 1, 2)[fg]
                tb = t_box.view(b, 4, 16, h, w).permute(0, 3, 4, 1, 2)[fg]
                kl = F.kl_div(F.log_softmax(sb / tau, -1),
                              F.softmax(tb / tau, -1),
                              reduction="sum") * tau * tau
                wgt = p_t.max(1).values[fg]                   # confidence weight
                box_tot = box_tot + (kl / max(sb.shape[0] * 4, 1)) * wgt.mean()
                fg_tot += nfg
        cls_kd = cls_tot / max(n_anchor, 1)
        box_kd = box_tot / max(len(maps_s), 1) if fg_tot else maps_s[0].sum() * 0
        kd = self.kd_lambda * (self.kd_cls_w * cls_kd + self.kd_box_w * box_kd)
        self.kd_log["n"] += 1
        self.kd_log["cls"] += float(cls_kd)
        self.kd_log["box"] += float(box_kd) if fg_tot else 0.0
        if self.kd_log["n"] % 200 == 0:
            print(f"KD[{self.kd_log['n']}] cls={self.kd_log['cls']/200:.4f} "
                  f"box={self.kd_log['box']/200:.4f} fg={fg_tot}", flush=True)
            self.kd_log["cls"] = self.kd_log["box"] = 0.0
        return kd
