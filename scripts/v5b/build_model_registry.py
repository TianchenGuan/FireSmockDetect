"""V5b phase 1 - verified model registry.

For every candidate checkpoint named in reports/v4/FINAL_REPORT_V4.md s13,
reports/FINAL_REPORT_V3.md, reports/FINAL_REPORT_V2.md and src/serve/backends.py:
  * path exists, byte size, sha256 (full + first 12)
  * train provenance read from the run directory (args.yaml / results.csv last
    epoch / status.json for YOLO; ckpt['args'] / metrics.jsonl / status.json for
    the DINOv3 detectors; status.json + tar listing for the Qwen adapters)
  * LOADS?  -- actually loaded on this CPU host:
      yolo   : ultralytics YOLO(path) + one 640px dummy predict, then
               src.serve.detector.SmokeDetector batch-1 latency @1280 on a real
               pyro-sdis VAL image (first line of data/yolo/lists/local/sdis_val.txt)
      dino3  : src.serve.backends.load_backend('dino3', ...).detect(real image)
      rfdetr : src.serve.backends.load_backend('rfdetr', ...).detect(real image)
      7B head / Qwen adapters: files verified only, recorded 'cluster-only'
  * val numbers copied from the frozen matched-FP artifacts (path recorded)
  * L40S batch-1 latency copied from the v3/v4 reports (path recorded)

Writes models/v5b/MODEL_REGISTRY.json and models/v5b/MODEL_REGISTRY.md.
Usage: python scripts/v5b/build_model_registry.py [--threads 8] [--lat-n 10]
       [--skip-load] [--only id,id,...] [--merge-existing] [--out-dir DIR]

V5b additions (phase 2, V5B_CANDIDATES below): YOLO arms A/B/C/D/D2/D3 (lineage from
data/v5b/manifests/CORPUS_*_v1.json; in-progress runs resolve to the latest segment),
B200 DINOv3-L LoRA 1920x1088 + 1280x736 control (dino3 backend; best_model.pt registered,
best.pt verified tensor-identical), SEG branch (scripts/v5b/seg_infer.py SmokeSegmenter),
Qwen3.5-27B bf16 LoRA 200-step SMOKE TEST (files only), ACCURATE ensemble recipe (composite).
  python scripts/v5b/build_model_registry.py --merge-existing --only <v5b ids>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "models" / "v5b"

# ----------------------------------------------------------------------------
# candidate table
# ----------------------------------------------------------------------------
LIC_YOLO = "AGPL-3.0 (ultralytics YOLO11 code + COCO init weights; fine-tune inherits)"
LIC_DINO = "DINOv3 License (custom Meta, 2025-08-19; models/v3/hf/*/LICENSE.md) - derivative weights inherit"
LIC_RFDETR = "Apache-2.0 (RF-DETR-Large code + weights; reports/v3/trackA_notes.md s1)"
LIC_QWEN = "Apache-2.0 (Qwen3.5 base, models/v2/PINS.md) - adapter inherits"
LIC_DEIM = "DEIMv2 License - NON-COMMERCIAL ONLY, weights included (reports/v3/trackA_notes.md) -> EXCLUDED"

E_RECIPE = ("corpus E = frozen expB_train list: pyro-sdis TRAIN sites + D-Fire smoke "
            "(38,454 imgs incl. 11,257 negatives); no FIgLib")
ET_RECIPE = ("corpus E+T (data/v4/manifests/CORPUS_E_ET_v1.json): E 38,454 + T 12,966 "
             "(Boreal train burns evo/karkkila + AuSmoke train units) = 51,420 imgs; no FIgLib")
EGT_RECIPE = ("corpus E+G+T (CORPUS_C_EGT_v1.json): E + G (WSDataset/FireSpot/FESB/firesense neg) "
              "+ T, G and T at 2x multiplicity; no FIgLib")
D_RECIPE = ("curriculum: stage-1 G-only pretrain 63 ep (expV4Dpre_g_s1) -> stage-2 E+T + 15% G "
            "replay 26 ep (CORPUS_D_adapt_v1.json); no FIgLib")

CANDIDATES = [
    dict(id="expB_y11s", family="YOLO11s", backend="yolo",
         path="models/expB_best.pt",
         identical_to="models/trained/expB_mix_y11s_1280/seg03/best.pt",
         run_dir="models/trained/expB_mix_y11s_1280/seg03",
         recipe=E_RECIPE + "; 60 ep, imgsz 1280, bs 64, init yolo11s.pt (COCO), iteration 1",
         val=[("reports/v3/matched_fp_val.json", "expB_y11s")],
         figlib="reports/replay_expB_figlib.json",
         l40s_ms=None, l40s_src="not measured for expB (YOLO11s); expE YOLO11m same protocol = 11.2 ms",
         licence=LIC_YOLO, role="deployed default (:8000)"),
    dict(id="expE_y11m", family="YOLO11m", backend="yolo",
         path="models/trained/expE_mix_y11m_1280_v2/seg05/best.pt",
         run_dir="models/trained/expE_mix_y11m_1280_v2/seg05",
         recipe=E_RECIPE + "; 60 ep, imgsz 1280, bs 48, init yolo11m.pt (COCO), iteration 2",
         val=[("reports/v3/matched_fp_val.json", "expE_y11m"),
              ("reports/v4/matched_fp_sdisval.json", "expE_iter2")],
         figlib="reports/replay_expE_figlib.json",
         l40s_ms=11.2, l40s_src="models/v3/evals/yolo_latency/latency_gpu_yolo.json (n=50, fp32)",
         cpu_reported=("reports/v3/latency_yolo11m_cpu8.json", 324.3),
         licence=LIC_YOLO, role="v4 deployment baseline S0"),
    dict(id="v4A_ctrl_s2", family="YOLO11m", backend="yolo",
         path="models/trained/expV4A_ctrl_s2/seg05/best.pt",
         run_dir="models/trained/expV4A_ctrl_s2/seg05",
         recipe=E_RECIPE + "; 60 ep, 1280, bs 48, init yolo11m.pt; arm A existing-data control, "
                "'median seed' pick (v4a_s2)",
         val=[("reports/v4/seed_summary.json", "A_ctrl_E/sdisval/per_seed/v4a_s2")],
         figlib="reports/v4/replay/replay_v4a_s2_figlib39.json",
         l40s_ms=None, l40s_src="not measured; same architecture/imgsz as expE (11.2 ms)",
         licence=LIC_YOLO, role="v4 S1"),
    dict(id="v4E_et_s1", family="YOLO11m", backend="yolo",
         path="models/v4/runs/expV4E_et_s1/seg05/best.pt",
         run_dir="models/v4/runs/expV4E_et_s1/seg05",
         recipe=ET_RECIPE + "; 45 ep, 1280, bs 48, init yolo11m.pt, seed 1 (median seed)",
         val=[("reports/v4/seed_summary.json", "E_ET/sdisval/per_seed/v4e_s1")],
         figlib="reports/v4/replay/replay_v4e_s1_figlib39.json",
         l40s_ms=None, l40s_src="not measured; same architecture/imgsz as expE (11.2 ms)",
         licence=LIC_YOLO, role="v4 S2 (best v4 data arm by S2 rule)"),
    dict(id="v4C_egt_s1", family="YOLO11m", backend="yolo",
         path="models/v4/runs/expV4C_egt_s1/seg05/best.pt",
         run_dir="models/v4/runs/expV4C_egt_s1/seg05",
         recipe=EGT_RECIPE + "; 25 ep, 1280, bs 48, init yolo11m.pt, seed 1 (median seed)",
         val=[("reports/v4/seed_summary.json", "C_EGT/sdisval/per_seed/v4c_s1")],
         figlib="reports/v4/replay/replay_v4c_s1_figlib39.json",
         l40s_ms=None, l40s_src="not measured; same architecture/imgsz as expE (11.2 ms)",
         licence=LIC_YOLO, role="v4 S2c"),
    dict(id="v4Dada_s1", family="YOLO11m", backend="yolo",
         path="models/v4/runs/expV4Dada_s1/seg03/best.pt",
         run_dir="models/v4/runs/expV4Dada_s1/seg03",
         recipe=D_RECIPE + "; 1280, bs 48, seed 1 (median seed); POST-HOC test addition in v4",
         val=[("reports/v4/seed_summary.json", "D_curr/sdisval/per_seed/v4dada_s1")],
         figlib="reports/v4/replay/replay_v4dada_s1_figlib39.json",
         l40s_ms=None, l40s_src="not measured; same architecture/imgsz as expE (11.2 ms)",
         licence=LIC_YOLO, role="v4 S3 D_curr (post-hoc)"),
    dict(id="rfdetrL_e48", family="RF-DETR-Large (DINOv2-S backbone, 1280)", backend="rfdetr",
         path="models/v3/rfdetr/checkpoint_best_ema.pth",
         run_dir="models/v3/rfdetr",
         recipe=E_RECIPE + "; 48 ep @1280 from rf-detr-large-2026.pth, EMA best epoch 37, "
                "job tianrgua-rfdetrl1280e48a1-gs9y (iteration 3)",
         val=[("reports/v3/matched_fp_val.json", "rfdetrL")],
         figlib="reports/v3/replay_rfdetrL_e48_figlib_valmatched.json",
         l40s_ms=31.4, l40s_src="models/v3/evals/rfdetrLe48lat/latency_gpu.json (n=50, fp32)",
         cpu_reported=("reports/v3/latency_rfdetr_cpu8.json", 1016.1),
         licence=LIC_RFDETR, role="v3 Track A; ensemble candidate (event coverage 33/39)"),
    dict(id="v4B1_frozen_L", family="DINOv3 ViT-L/16 frozen + SFP + Faster R-CNN", backend="dino3",
         path="models/v3/runs/v4B1_frozen_L/seg03/best.pt",
         run_dir="models/v3/runs/v4B1_frozen_L/seg03",
         backbone="models/v3/hf/dinov3-vitl16",
         recipe=ET_RECIPE + "; 10 ep, 1280x736, bs 4/GPU (32), head+neck only, lr 1e-4",
         val=[("reports/v4/matched_fp_dino_sdis_val.json", "v4B1_frozen_L")],
         figlib=None,
         l40s_ms=156.6, l40s_src="reports/v4/dino_adapt_results.md latency table (v3 B1 frozen-L reference, "
                                 "identical architecture; models/v3/evals/B1val/latency_gpu.json)",
         licence=LIC_DINO, role="v4 S4c DINO control"),
    dict(id="v4B4_lora_L", family="DINOv3 ViT-L/16 LoRA r16 a32 (q/k/v/o) + SFP + Faster R-CNN", backend="dino3",
         path="models/v3/runs/v4B4_lora_L/seg05/best.pt",
         run_dir="models/v3/runs/v4B4_lora_L/seg05",
         backbone="models/v3/hf/dinov3-vitl16",
         recipe=ET_RECIPE + "; 10 ep, 1280x736, bs 4/GPU (32), LoRA r16 a32, lr 1e-4 head/adapters, grad-ckpt",
         val=[("reports/v4/matched_fp_dino_sdis_val.json", "v4B4_lora_L")],
         figlib="reports/v4/replay_v4B4_lora_L_figlib39.json",
         l40s_ms=157.5, l40s_src="reports/v4/dino_adapt_results.md (LoRA merged 157.5 / unmerged 161.8; "
                                 "models/v4/evals/v4B4_lora_L_figlib/latency_gpu*.json)",
         cpu_reported=("reports/v4/dino3_backend_latency_B4_8thr.json", 5616.6),
         licence=LIC_DINO, role="v4 S4p post-hoc efficiency candidate"),
    dict(id="v4B5_full_L", family="DINOv3 ViT-L/16 full unfreeze + SFP + Faster R-CNN", backend="dino3",
         path="models/v3/runs/v4B5_full_L/seg11/best.pt",
         run_dir="models/v3/runs/v4B5_full_L/seg11",
         backbone="models/v3/hf/dinov3-vitl16",
         recipe=ET_RECIPE + "; 10 ep, 1280x736, bs 2/GPU (16, 2x steps), all blocks trainable, "
                "backbone lr 1e-5, end-only checkpoint (confounded vs B1/B4/B6)",
         val=[("reports/v4/matched_fp_dino_sdis_val.json", "v4B5_full_L")],
         figlib=None,
         l40s_ms=None, l40s_src="not measured; plain ViT-L form == B1/B4-merged (~157 ms expected)",
         licence=LIC_DINO, role="v4 ablation (not on test roster)"),
    dict(id="v4B6_lora_Hplus", family="DINOv3 ViT-H+/16 LoRA r16 a32 (q/k/v/o) + SFP + Faster R-CNN", backend="dino3",
         path="models/v3/runs/v4B6_lora_Hplus/seg10/best.pt",
         run_dir="models/v3/runs/v4B6_lora_Hplus/seg10",
         backbone="models/v3/hf/dinov3-vith16plus",
         recipe=ET_RECIPE + "; 10 ep, 1280x736, bs 4/GPU (32), LoRA r16 a32, lr 1e-4, grad-ckpt",
         val=[("reports/v4/matched_fp_dino_sdis_val.json", "v4B6_lora_Hplus")],
         figlib="reports/v4/replay_v4B6_lora_Hplus_figlib39.json",
         l40s_ms=351.6, l40s_src="reports/v4/dino_adapt_results.md (LoRA merged 351.6 / unmerged 357.2; "
                                 "models/v4/evals/v4B6_lora_Hplus_figlib/latency_gpu*.json)",
         licence=LIC_DINO, role="v4 S4 pre-registered DINO pick"),
    dict(id="v4B6_lora_Hplus_seg05_globalbest", family="DINOv3 ViT-H+/16 LoRA r16 a32 (q/k/v/o) + SFP + Faster R-CNN", backend="dino3",
         path="models/v3/runs/v4B6_lora_Hplus/seg05/best.pt",
         run_dir="models/v3/runs/v4B6_lora_Hplus/seg05",
         backbone="models/v3/hf/dinov3-vith16plus",
         recipe=ET_RECIPE + "; same run as v4B6_lora_Hplus, earlier checkpoint (step 8428 = the run's global best "
                "1500-img COCO mAP 0.4217); NOT the file v4 evaluated/tested",
         val=[], val_text="NOT evaluated under the unified matched-FP protocol (v4 evaluated seg10/best.pt = step 11714); "
                          "only training-time COCO mAP 0.4217 on the 1,500-img sdis-val subset",
         figlib=None, l40s_ms=351.6, l40s_src="same architecture as v4B6_lora_Hplus (merged 351.6 ms)",
         licence=LIC_DINO, role="ALTERNATE checkpoint surfaced by this audit; needs unified val eval before use"),
    dict(id="C1_7b_head_cached", family="DINOv3 ViT-7B/16 frozen (cached bf16 feats) + SFP + Faster R-CNN head",
         backend="none (cluster-only; cached-feature head, no serving backend)",
         path="models/v3/runs/C1_7b_head_cached_1280/seg04/best.pt",
         run_dir="models/v3/runs/C1_7b_head_cached_1280/seg04",
         backbone="models/v3/hf/dinov3-vit7b16",
         recipe=E_RECIPE + "; 6 ep, 1280x736, head trained on cached bf16 7B features, NO aug "
                "(iteration 3, I/O-bound cut)",
         val=[("reports/v3/matched_fp_val_trackBC.json", "C1_7b")],
         figlib="reports/v3/replay_C1_figlib_valmatched.json",
         l40s_ms=368.5, l40s_src="reports/v3/trackBC_results.md s3 (bf16 backbone, 1x L40S)",
         licence=LIC_DINO, cluster_only=True,
         role="v3 Track C; FROZEN TEST collapse (0.469/0.304 vs expE 0.526/0.435)"),
    dict(id="qwen3.5-9b_sft_lora", family="Qwen3.5-9B VLM + LoRA adapter (SFT)",
         backend="none (cluster-only VLM; no serving backend)",
         path="models/v2/runs/trkA_9b_lora/seg05/final_adapter.tgz",
         run_dir="models/v2/runs/trkA_9b_lora/seg05",
         base="models/v2/qwen3.5-9b", base_rev="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
         recipe="SFT on data/v2/sft (60k examples built from corpus E labels; no FIgLib), LoRA ~0.9% of 9.4B, "
                "vision frozen (FINAL_REPORT_V2)",
         val_text="val-2000 single operating point: P 0.766 / box recall 0.589 / tiny 0.460 / hard-neg img FP 0.330 "
                  "(reports/FINAL_REPORT_V2.md 'Full comparison'; preds models/v2/evals/eval9b_sft/preds.jsonl)",
         figlib=None, l40s_ms=7300.0, l40s_src="reports/FINAL_REPORT_V2.md (7.3 s/img, 1x L40S bf16)",
         licence=LIC_QWEN, cluster_only=True, role="v2 Track A"),
    dict(id="qwen3.5-9b_sft_grpo500", family="Qwen3.5-9B VLM + LoRA adapter (SFT -> GRPO 500 steps)",
         backend="none (cluster-only VLM; no serving backend)",
         path="models/v2/runs/grpo_9b_500/attempt02/final_adapter.tgz",
         run_dir="models/v2/runs/grpo_9b_500/attempt02",
         base="models/v2/qwen3.5-9b", base_rev="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
         recipe="GRPO 500 steps on top of the SFT adapter, reward from corpus-E boxes (reports/v2/grpo_feasibility.md); no FIgLib",
         val_text="val-2000 single operating point: P 0.719 / box recall 0.646 / tiny 0.548 / hard-neg img FP 0.413 "
                  "(reports/FINAL_REPORT_V2.md; preds models/v2/evals/eval9b_grpo/preds.jsonl)",
         figlib=None, l40s_ms=6500.0, l40s_src="reports/FINAL_REPORT_V2.md (6.5 s/img, 1x L40S bf16)",
         licence=LIC_QWEN, cluster_only=True, role="v2 best VLM"),
    dict(id="qwen3.5-27b_sft_qlora", family="Qwen3.5-27B VLM nf4 + QLoRA adapter (SFT)",
         backend="none (cluster-only VLM; no serving backend)",
         path="models/v2/runs/trkB_27b_qlora/seg10/final_adapter.tgz",
         run_dir="models/v2/runs/trkB_27b_qlora/seg10",
         base="models/v2/qwen3.5-27b", base_rev="fc05daec18b0a78c049392ed2e771dde82bdf654",
         recipe="SFT QLoRA (nf4, 0.75 MP) on data/v2/sft; no FIgLib",
         val_text="val-2000 single operating point: P 0.754 / box recall 0.438 / tiny 0.302 / hard-neg img FP 0.254 "
                  "(reports/FINAL_REPORT_V2.md; preds models/v2/evals/eval27b_sft_v2/preds.jsonl)",
         figlib=None, l40s_ms=17800.0, l40s_src="reports/FINAL_REPORT_V2.md (17.8 s/img, 1x L40S)",
         licence=LIC_QWEN, cluster_only=True, role="v2 Track B"),
    dict(id="deimv2_M_coco_pretrain", family="DEIMv2-M (DINOv3 distilled) COCO pretrain",
         backend="deimv2 (registered in backends.AVAILABLE, never implemented)",
         path="models/v3/pretrain/deimv2/deimv2_dinov3_m_coco.pth", run_dir=None,
         recipe="upstream COCO weights; sanity fine-tune models/v3/runs/deimv2M_sanity produced no checkpoint "
                "(DDP hang, reports/v3/trackA_notes.md)",
         val_text="none", figlib=None, l40s_ms=None, l40s_src=None,
         licence=LIC_DEIM, excluded=True, role="EXCLUDED (non-commercial licence)"),
]

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def sha256_file(p: Path, bufsize=1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(rel):
    return json.load(open(ROOT / rel))


def dig(d, path):
    for k in path.split("/"):
        d = d[k]
    return d


def yolo_provenance(run_dir: Path) -> dict:
    out = {}
    try:
        import yaml
        a = yaml.safe_load(open(run_dir / "args.yaml"))
        out["args"] = {k: a.get(k) for k in ("model", "data", "epochs", "imgsz", "batch", "seed",
                                             "optimizer", "lr0", "pretrained", "name")}
    except Exception as e:  # noqa
        out["args_error"] = repr(e)
    try:
        rows = list(csv.DictReader(open(run_dir / "results.csv")))
        last = {k.strip(): v.strip() for k, v in rows[-1].items()}
        out["results_last_epoch"] = {
            "epoch": last.get("epoch"), "n_rows": len(rows),
            "precision": last.get("metrics/precision(B)"), "recall": last.get("metrics/recall(B)"),
            "mAP50": last.get("metrics/mAP50(B)"), "mAP50-95": last.get("metrics/mAP50-95(B)")}
        out["results_note"] = "ultralytics training-time val (its own protocol), NOT the unified matched-FP eval"
    except Exception as e:  # noqa
        out["results_error"] = repr(e)
    try:
        st = json.load(open(run_dir / "status.json"))
        out["status"] = {k: st.get(k) for k in ("exp", "phase", "train_rc", "gpus", "host", "ts", "entry")}
        out["kraken_job_hint"] = st.get("host")
    except Exception as e:  # noqa
        out["status_error"] = repr(e)
    out["segment"] = run_dir.name
    return out


def dino_provenance(run_dir: Path, ck_args: dict | None, ck_meta: dict) -> dict:
    out = {"segment": run_dir.name}
    try:
        st = json.load(open(run_dir / "status.json"))
        out["status"] = {k: st.get(k) for k in ("run", "phase", "rc", "gpus", "host", "ts")}
        out["kraken_job_hint"] = st.get("host")
    except Exception as e:  # noqa
        out["status_error"] = repr(e)
    try:
        lines = [json.loads(l) for l in open(run_dir / "metrics.jsonl") if l.strip()]
        evals = [l for l in lines if "eval" in l]
        if evals:
            e = evals[-1]
            out["metrics_last_eval"] = {"step": e.get("step"), **{k: round(v, 4) for k, v in e["eval"].items()
                                                                 if isinstance(v, (int, float))}}
            out["metrics_note"] = "COCO mAP on 1,500-img sdis-val subset (training-time), NOT the unified eval"
    except Exception as e:  # noqa
        out["metrics_error"] = repr(e)
    if ck_args:
        out["ckpt_args"] = {k: ck_args.get(k) for k in ("backbone_dir", "feat_root", "embed_dim", "lora", "lora_alpha",
                                                        "unfreeze_all", "unfreeze_top", "grad_ckpt", "epochs", "bs",
                                                        "lr", "backbone_lr", "width", "height", "seed", "train_json",
                                                        "output_dir") if k in ck_args}
    out.update(ck_meta)
    return out


def qwen_provenance(run_dir: Path, tgz: Path) -> dict:
    out = {"segment": run_dir.name}
    try:
        st = json.load(open(run_dir / "status.json"))
        out["status"] = st
        out["kraken_job_hint"] = st.get("host")
    except Exception as e:  # noqa
        out["status_error"] = repr(e)
    try:
        with tarfile.open(tgz, "r:gz") as tf:
            names = tf.getnames()
        out["adapter_tar_n_members"] = len(names)
        out["adapter_tar_members_head"] = [n for n in names if not n.endswith("/")][:12]
        out["adapter_has_safetensors"] = any(n.endswith("adapter_model.safetensors") for n in names)
        out["adapter_has_config"] = any(n.endswith("adapter_config.json") for n in names)
    except Exception as e:  # noqa
        out["adapter_tar_error"] = repr(e)
    return out


def backbone_info(rel: str | None) -> dict | None:
    if not rel:
        return None
    d = ROOT / rel
    info = {"path": rel, "exists": d.is_dir()}
    if d.is_dir():
        pin = d / "REVISION_PIN.txt"
        info["revision_pin"] = pin.read_text().strip() if pin.exists() else None
        w = sorted(p for p in d.glob("*.safetensors"))
        info["n_weight_shards"] = len(w)
        info["weights_bytes"] = sum(p.stat().st_size for p in w)
        info["licence_file"] = (d / "LICENSE.md").exists() or (d / "LICENSE").exists()
    return info


def val_numbers(spec_list):
    out = []
    for spec in spec_list:
        rel, key = spec[0], spec[1]
        split = spec[2] if len(spec) > 2 else "pyro-sdis VAL (7,335 imgs; 1,158 neg; IoU>=0.4)"
        try:
            d = read_json(rel)
            if "matched_fp" in d:
                row = d["matched_fp"][key]
            else:
                row = dig(d["arms"], key)
            pick = {}
            for fp in ("fp0.01", "fp0.02", "fp0.05", "fp0.10", "fp0.20"):
                if fp in row:
                    r = row[fp]
                    pick[fp] = {"recall": r.get("recall"), "tiny": r.get("tiny_recall", r.get("tiny")),
                                "conf": r.get("conf")}
                    if r.get("small_recall") is not None:
                        pick[fp]["small"] = r.get("small_recall")
            out.append({"artifact": rel, "key": key, "split": split, "matched_fp": pick})
        except Exception as e:  # noqa
            out.append({"artifact": rel, "key": key, "error": repr(e)})
    return out


def figlib_numbers(rel):
    if not rel:
        return None
    try:
        s = read_json(rel)["summary"]
        return {"artifact": rel, "n_detected": s.get("n_detected"), "n_events": s.get("n_events"),
                "median_delay_s": s.get("median_delay_s_among_detected"),
                "conf_single": s["config"].get("single_frame_alert_conf"),
                "conf_seq": s["config"].get("seq_frame_conf"), "weights": s["config"].get("weights"),
                "note": "FIgLib-39 is a HISTORICAL eval set (CC BY-NC-ND, eval-only); never training"}
    except Exception as e:  # noqa
        return {"artifact": rel, "error": repr(e)}


def lat_stats(ms):
    ms = sorted(ms)
    return {"n": len(ms), "mean_ms": round(statistics.mean(ms), 1), "median_ms": round(statistics.median(ms), 1),
            "p90_ms": round(ms[min(len(ms) - 1, int(round(0.9 * (len(ms) - 1))))], 1),
            "min_ms": round(ms[0], 1), "max_ms": round(ms[-1], 1)}


# ----------------------------------------------------------------------------
# load tests
# ----------------------------------------------------------------------------
def load_yolo(path: Path, img, lat_n: int) -> dict:
    import numpy as np
    from ultralytics import YOLO
    res = {}
    t0 = time.perf_counter()
    m = YOLO(str(path))
    res["load_s"] = round(time.perf_counter() - t0, 2)
    res["params_total"] = int(sum(p.numel() for p in m.model.parameters()))
    res["class_names"] = {int(k): v for k, v in m.names.items()}
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    r = m.predict(dummy, imgsz=640, device="cpu", verbose=False)
    res["dummy_640_ok"] = len(r) == 1
    res["dummy_640_n_dets"] = int(len(r[0].boxes))
    from src.serve.detector import SmokeDetector
    be = SmokeDetector(str(path), device="cpu", imgsz=1280)
    for _ in range(2):
        be.detect(img)
    ms = []
    for _ in range(lat_n):
        dets, timing = be.detect(img)
        ms.append(float(timing["latency_ms"]))
    res["real_img_n_dets"] = len(dets)
    res["real_img_top_scores"] = [round(d.score, 4) for d in sorted(dets, key=lambda d: -d.score)[:3]]
    res["cpu_latency_1280"] = lat_stats(ms)
    return res


def load_backend_generic(name: str, path: Path, img, lat_n: int) -> dict:
    from src.serve.backends import load_backend
    res = {}
    t0 = time.perf_counter()
    be = load_backend(name, weights_path=str(path), device="cpu", imgsz=1280)
    res["load_s"] = round(time.perf_counter() - t0, 2)
    res["class_names"] = {int(k): v for k, v in be.class_names.items()}
    model = getattr(be, "_model", None) or getattr(be, "_rf", None)
    try:
        inner = model
        for _ in range(4):  # rfdetr: RFDETRLarge -> ModelContext -> Model -> LWDETR (nn.Module)
            if inner is not None and hasattr(inner, "parameters"):
                break
            inner = getattr(inner, "model", None)
        params = list(inner.parameters()) if inner is not None and hasattr(inner, "parameters") else []
        res["params_total"] = int(sum(p.numel() for p in params)) if params else None
        res["params_trainable_flag"] = int(sum(p.numel() for p in params if p.requires_grad)) if params else None
    except Exception as e:  # noqa
        res["params_error"] = repr(e)
    for attr in ("backbone_dir", "lora_r", "lora_alpha", "lora_modules", "n_lora_params", "ckpt_step",
                 "ckpt_best_map", "n_ckpt_tensors", "width", "height"):
        if hasattr(be, attr):
            v = getattr(be, attr)
            res[attr] = str(v) if isinstance(v, Path) else v
    be.detect(img)  # warm-up
    ms = []
    for _ in range(lat_n):
        dets, timing = be.detect(img)
        ms.append(float(timing["latency_ms"]))
    res["real_img_n_dets"] = len(dets)
    res["real_img_top_scores"] = [round(d.score, 4) for d in sorted(dets, key=lambda d: -d.score)[:3]]
    res["cpu_latency_1280"] = lat_stats(ms)
    return res


# ----------------------------------------------------------------------------
# eligibility (rules stated once, applied mechanically)
# ----------------------------------------------------------------------------
ELIG_RULES = {
    "fast_mode": "loads on CPU through a src.serve backend AND measured CPU batch-1 latency (8 threads, 1280) <= 1000 ms",
    "accurate_mode": "loads on CPU through a src.serve backend AND reported 1x L40S batch-1 latency <= 400 ms "
                     "(or same architecture as a measured sibling) AND licence permits internal research serving",
    "teacher": "loads AND licence permits derivative (pseudo-label) use AND not trained on FIgLib AND "
               "(sdis-VAL matched-FP R@5% >= YOLO A_ctrl 3-seed mean 0.202, or FIgLib-39 event coverage > expE 30/39); "
               "YOLO11 arms are the STUDENT family -> ensemble members, not teachers",
}
YOLO_A_R5 = 0.202


def eligibility(c, loads, cpu_ms, l40s_ms, val, figlib):
    if c.get("excluded"):
        return {"fast_mode": False, "accurate_mode": False, "teacher": False, "reason": "licence excluded"}
    if c.get("cluster_only") or not loads:
        return {"fast_mode": False, "accurate_mode": False, "teacher": False,
                "reason": "cluster-only / no serving backend" if c.get("cluster_only") else "failed to load"}
    fast = cpu_ms is not None and cpu_ms <= 1000.0
    acc = (l40s_ms is not None and l40s_ms <= 400.0) or c["family"].startswith("YOLO11") or c["id"] == "v4B5_full_L"
    r5 = None
    for v in val:
        r5 = r5 or (v.get("matched_fp", {}).get("fp0.05") or {}).get("recall")
    fig_ok = bool(figlib and figlib.get("n_detected") and figlib["n_detected"] > 30)
    teacher = (not c["family"].startswith("YOLO11")) and ((r5 is not None and r5 >= YOLO_A_R5) or fig_ok)
    return {"fast_mode": bool(fast), "accurate_mode": bool(acc), "teacher": bool(teacher),
            "reason": f"cpu_ms={cpu_ms}, l40s_ms={l40s_ms}, R@5%={r5}, figlib>{30}={fig_ok}"}



# ----------------------------------------------------------------------------
# V5b additions (phase 2): student arms A/B/C/D/D2/D3, B200 DINO high-res, segmentation,
# Qwen 27B smoke test, ACCURATE ensemble recipe. Built by build_v5b_entry(); merged with
# --merge-existing so the phase-1 entries above are kept verbatim.
# ----------------------------------------------------------------------------
LIC_R50 = "BSD-3-Clause (torchvision resnet50 IMAGENET1K_V2 init, frozen; models/v5b/seg/pretrained) - head trained here"
LIC_ENSEMBLE = ("composite: AGPL-3.0 (YOLO11m v4A_ctrl_s2) + DINOv3 License x3 (v4B4/v4B6/v4B1); serving inherits the "
                "most restrictive component licence (AGPL-3.0)")
SDIS_SPLIT = "pyro-sdis VAL (7,335 imgs; 1,158 neg; IoU>=0.4)"
BOREAL_SPLIT = "Boreal heinola aerial VAL (1,123 imgs; 217 neg; IoU>=0.4)"
FIREMAN_SPLIT = "FireMan-UAV-RGBT VAL video (1,706 imgs; 231 neg; IoU>=0.4)"
V5B_DATA_NOTE = ("V5b training data: Boreal UAV CC-BY-4.0, FireMan-UAV-RGBT CC-BY-4.0 (data/v5b/manifests/fireman_CARD.md), "
                 "Sheffield UAV CC-BY-4.0 (sheffield_uav_CARD.md); AuSmoke/FireSpot/WSDataset EXCLUDED (permission pending / blocked); "
                 "no FIgLib, no heinola/ruokolahti, no pyro-sdis TEST, no FireMan val/holdout in any V5b training corpus.")

A_V5B_RECIPE = (E_RECIPE + "; 60 ep, 1280, bs 48, init yolo11m.pt; V5b arm A control = exposure anchor "
                "(38,454 x 60 = 2,307,240 exposures)")
B_V5B_RECIPE = ("corpus B_v1 (data/v5b/manifests/CORPUS_B_v1.json): E 38,454 + Boreal train 2,062 (x2) + FireMan RGB TRAIN "
                "8,279 + Sheffield UAV 1,327 = 53,511 lines, author labels only; 43 ep (exposure-matched to A, -0.27%)")
C_V5B_RECIPE = ("corpus C_v1 (CORPUS_C_v1.json): B + conservative teacher IGNORE regions (148 imgs / 300 rects filled "
                "where >=2 teachers fire on unlabelled area); 43 ep")
D_V5B_RECIPE = ("corpus D_v1 (CORPUS_D_v1.json): B + filtered teacher PSEUDO-boxes (495 boxes on 495 imgs; teachers "
                "v4B4+v4B6(+rfdetrL) agreement; sdis 449 / dfire 43 / boreal 3); 43 ep")
D2_V5B_RECIPE = ("corpus D2_v1 (CORPUS_D2_v1.json): B + pseudo-boxes from calibrated WBF of the 3 teachers "
                 "(models/v5b/fusion/teacher_ensemble_d2.json; fused >=0.9918 or single >=0.98) on 408 imgs; 43 ep")
D3_V5B_RECIPE = ("corpus D3_v1 (CORPUS_D3_v1.json): B unchanged + 3,796 pseudo-boxed UNLABELLED imgs x1 "
                 "(borealclip 3,254 / firesense 350 / dfirefire 192) = 57,307 lines; 40 ep (exposure-matched, -0.65%)")
EB_RECIPE = ("corpus E + Boreal-train (data/v5b/coco/instances_train_eb.json = frozen v4 E_ET json minus every AuSmoke path, "
             "scripts/v5b/build_train_eb.py): pyro-sdis 22,177 / D-Fire 16,277 / Boreal 2,062 = 40,516 imgs "
             "(29,222 pos / 11,294 neg; 33,548 boxes); no FIgLib; 10 ep = 12,660 steps @ eff. bs 32")
SEG_RECIPE = ("data/v5b/manifests/seg_v5b_manifest.json: 665 Boreal TRAIN-burn (karkkila+evo) masked frames "
              "(SAM model-assisted + author masks; 37 verified-empty), heinola 382 frames = VAL; letterbox 1280x736; "
              "BCE + soft Dice; 10 ep")

def _yolo_v5b(id_, run_root, sysname, recipe, manifest, role, seed_note=None, arm_A=False, final_seg="seg05",
              flags=None, has_eval=True):
    val = [("reports/v5b/val/matched_fp_sdisval.json", sysname, SDIS_SPLIT),
           ("reports/v5b/val/matched_fp_borealval.json", sysname, BOREAL_SPLIT),
           ("reports/v5b/val/matched_fp_firemanval.json", sysname, FIREMAN_SPLIT)] if (sysname and has_eval) else []
    return dict(id=id_, v5b=True, kind="yolo", family="YOLO11m", backend="yolo", run_root=run_root, final_seg=final_seg,
                recipe=recipe, corpus_manifest=manifest, corpus_arm_A=arm_A, val=val,
                figlib=f"reports/v5b/replay/replay_{sysname}_figlib39.json" if (sysname and has_eval) else None,
                l40s_ms=None, l40s_src="not measured; same architecture/imgsz as expE (11.2 ms L40S)",
                licence=LIC_YOLO, role=role, seed_note=seed_note, flags=flags or [])

V5B_CANDIDATES = [
    _yolo_v5b("v5bA_ctrl_s3", "models/trained/expV5bA_ctrl_s3", "S1_A", A_V5B_RECIPE, "data/v5b/manifests/CORPUS_B_v1.json",
              role="V5b S1 arm A control (pre-registered)", arm_A=True,
              flags=["SEED PROVENANCE: seed-0 REPLICATE of expV4A_ctrl_s{0,1,2} (run_yolo_arm.sh cannot pass --seed; "
                     "args.yaml seed 0) - a 4th same-seed replicate, NOT an independent seed; differs via non-deterministic DDP"]),
    _yolo_v5b("v5bA_ctrl_s1", "models/v4/runs/expV5bA_ctrl_s1", None, A_V5B_RECIPE, "data/v5b/manifests/CORPUS_B_v1.json",
              role="V5b arm A seed-1 repeat (matched control for v5bB_s1)", arm_A=True, has_eval=False),
    _yolo_v5b("v5bA_ctrl_s2", "models/v4/runs/expV5bA_ctrl_s2", None, A_V5B_RECIPE, "data/v5b/manifests/CORPUS_B_v1.json",
              role="V5b arm A seed-2 repeat (matched control for v5bB_s2)", arm_A=True, has_eval=False),
    _yolo_v5b("v5bB_s0", "models/v4/runs/expV5bB_s0", "S2_B", B_V5B_RECIPE, "data/v5b/manifests/CORPUS_B_v1.json",
              role="V5b S2 arm B - SELECTED student arm (SELECTION_V5B.md: mean(sdis R@10%, FireMan R@10%) 0.6496)"),
    _yolo_v5b("v5bB_s1", "models/v4/runs/expV5bB_s1", None, B_V5B_RECIPE, "data/v5b/manifests/CORPUS_B_v1.json",
              role="V5b arm B seed-1 repeat", has_eval=False),
    _yolo_v5b("v5bB_s2", "models/v4/runs/expV5bB_s2", None, B_V5B_RECIPE, "data/v5b/manifests/CORPUS_B_v1.json",
              role="V5b arm B seed-2 repeat", has_eval=False),
    _yolo_v5b("v5bC_s0", "models/v4/runs/expV5bC_s0", "S3_C", C_V5B_RECIPE, "data/v5b/manifests/CORPUS_C_v1.json",
              role="V5b S3 arm C (ignore regions)"),
    _yolo_v5b("v5bD_s0", "models/v4/runs/expV5bD_s0", "S4_D", D_V5B_RECIPE, "data/v5b/manifests/CORPUS_D_v1.json",
              role="V5b S4 arm D (teacher pseudo-boxes)"),
    _yolo_v5b("v5bD2_s0", "models/v4/runs/expV5bD2_s0", "S4b_D2", D2_V5B_RECIPE, "data/v5b/manifests/CORPUS_D2_v1.json",
              role="V5b S4b arm D2 (fused-teacher pseudo-boxes)"),
    _yolo_v5b("v5bD3_s0", "models/v4/runs/expV5bD3_s0", "S4c_D3", D3_V5B_RECIPE, "data/v5b/manifests/CORPUS_D3_v1.json",
              role="V5b S4c arm D3 (pseudo-boxed unlabelled imgs)",
              flags=["40 ep (not 43): exposure-matched to A on 57,307 lines (2,292,280 vs 2,307,240 exposures, -0.65%)"]),
    dict(id="v5bB200_lora_L_1920", v5b=True, kind="dino3", backend="dino3",
         family="DINOv3 ViT-L/16 LoRA r16 (q/k/v/o) + SFP + Faster R-CNN @ 1920x1088",
         path="models/v5b/b200/v5bB200_lora_L_1920/best_model.pt",
         alt_paths=["models/v5b/b200/v5bB200_lora_L_1920/best.pt"],
         run_dir="models/v5b/b200/v5bB200_lora_L_1920", report_dir="reports/v5b/b200_dino/v5bB200_lora_L_1920",
         backbone="models/v3/hf/dinov3-vitl16", recipe=EB_RECIPE + "; 1920x1088 (120x68 patches = 8,160 tok), bs 4/GPU, "
         "lr 1e-4 head+LoRA, wd 1e-4, grad-ckpt; TRAINED ON 8x B200 (bf16), 3.18 node-h",
         val=[("reports/v5b/matched_fp_b200_dino_sdis_val.json", "v5bB200_lora_L_1920", SDIS_SPLIT)], figlib=None,
         l40s_ms=None, l40s_src="NOT measured on L40S; B200 batch-1 283.8 ms median (reports/v5b/b200_dino/v5bB200_lora_L_1920/"
         "latency_b200.json, n=50); same architecture as v4B4 (157.5 ms L40S @1280x736) but 2.2x tokens",
         accurate_ok=False, accurate_reason="L40S latency unmeasured at 1920x1088; B200 283.8 ms with B200 ~1.5x faster than "
         "L40S on the 1280 control (103.9 vs 157.5) -> L40S likely >400 ms; not eligible until measured",
         licence=LIC_DINO, role="V5b S6 B200 high-res DINO (FINAL: behind v4B4 at every FP on sdis VAL)",
         flags=["CONFOUNDED vs v4B4: resolution (1920x1088 vs 1280x736) AND corpus (E+Boreal 40,516 vs E+T 51,420; AuSmoke "
                "dropped) AND hardware (B200 vs L40S) - B200_DINO_HIGHRES.md s0",
                "best_model.pt (102 MB, trainable-only) registered; best.pt (305 MB) = same 238 model tensors + optimizer state"]),
    dict(id="v5bB200_lora_L_1280", v5b=True, kind="dino3", backend="dino3",
         family="DINOv3 ViT-L/16 LoRA r16 (q/k/v/o) + SFP + Faster R-CNN @ 1280x736",
         path="models/v5b/b200/v5bB200_lora_L_1280/best_model.pt",
         alt_paths=["models/v5b/b200/v5bB200_lora_L_1280/best.pt"],
         run_dir="models/v5b/b200/v5bB200_lora_L_1280", report_dir="reports/v5b/b200_dino/v5bB200_lora_L_1280",
         backbone="models/v3/hf/dinov3-vitl16", recipe=EB_RECIPE + "; 1280x736 (80x46 = 3,680 tok) same-corpus CONTROL for the "
         "1920 run, bs 4/GPU, lr 1e-4, wd 1e-4, grad-ckpt; TRAINED ON 8x B200 (bf16), 1.25 node-h",
         val=[("reports/v5b/matched_fp_b200_dino_sdis_val.json", "v5bB200_lora_L_1280_ctrl", SDIS_SPLIT)], figlib=None,
         l40s_ms=None, l40s_src="NOT measured on L40S; B200 batch-1 103.9 ms median (reports/v5b/b200_dino/v5bB200_lora_L_1280/"
         "latency_b200.json, n=50); identical architecture+geometry to v4B4 (157.5 ms L40S merged)",
         accurate_ok=True, accurate_reason="same architecture and geometry as measured sibling v4B4 (157.5 ms L40S <= 400)",
         licence=LIC_DINO, role="V5b B200 1280x736 same-corpus control (isolates resolution from corpus)",
         flags=["best_model.pt (102 MB, trainable-only) registered; best.pt (305 MB) = same 238 model tensors + optimizer state"]),
    dict(id="segV5b_a_frozen", v5b=True, kind="seg", backend="seg_infer (scripts/v5b/seg_infer.py SmokeSegmenter; demo /segment slot, NOT a src.serve detector backend)",
         family="SEG: DINOv3 ViT-L/16 frozen + SFP + FPNLite seg head (pixel smoke mask)",
         path="models/v5b/seg/segV5b_a/seg02/frozen/best.pt", run_dir="models/v5b/seg/segV5b_a/seg02",
         backbone="models/v3/hf/dinov3-vitl16", recipe=SEG_RECIPE + "; frozen backbone, head/neck lr 2e-4, bs 4x2 GPU = 8, 830 steps",
         val_text="heinola VAL (382 fr; thr 0.5) smoke-pixel IoU 0.568 / Dice 0.724 / P 0.842 / R 0.636; mask->box recall vs author "
                  "boxes 0.830 (comp IoU>=0.4) / 0.997 (localisation hit) [reports/v5b/SEGMENTATION_V5B.md s3]",
         figlib=None, l40s_ms=None, l40s_src="not measured on GPU (CPU demo path only)", licence=LIC_DINO,
         role="V5b SEG branch - RECOMMENDED for /segment overlay (best VAL IoU, smallest)",
         flags=["best.pt = best of 10 epoch evals on heinola VAL (selection-optimistic; final-epoch IoU 0.523); 329/336 VAL masks "
                "are SAM model-assisted; overlay only - must NOT raise alerts (clear-cut false positives, SEGMENTATION_V5B.md s7)"]),
    dict(id="segV5b_a_lora", v5b=True, kind="seg", backend="seg_infer (scripts/v5b/seg_infer.py SmokeSegmenter)",
         family="SEG: DINOv3 ViT-L/16 LoRA r16 (q/k/v/o) + SFP + FPNLite seg head",
         path="models/v5b/seg/segV5b_a/seg01/lora/best.pt", run_dir="models/v5b/seg/segV5b_a/seg01",
         backbone="models/v3/hf/dinov3-vitl16", recipe=SEG_RECIPE + "; LoRA r16 a32 lr 1e-4 + head/neck 2e-4, bs 2x4 GPU = 8, 830 steps",
         val_text="heinola VAL (382 fr; thr 0.5) IoU 0.551 / Dice 0.710 / P 0.855 / R 0.608; mask->box recall 0.804 / 0.994 "
                  "[SEGMENTATION_V5B.md s3]; frozen - LoRA = +0.012 mean dIoU (wash)",
         figlib=None, l40s_ms=None, l40s_src="not measured on GPU", licence=LIC_DINO, role="V5b SEG branch LoRA variant",
         flags=["LoRA overfits after epoch 2 with 665 train frames; best.pt = epoch 2 (step 166); selection-optimistic"]),
    dict(id="segV5b_a_r50", v5b=True, kind="seg", backend="seg_infer (scripts/v5b/seg_infer.py SmokeSegmenter)",
         family="SEG: ResNet-50 (ImageNet, frozen) + FPNLite seg head - baseline",
         path="models/v5b/seg/segV5b_a/seg01/r50/best.pt", run_dir="models/v5b/seg/segV5b_a/seg01", backbone=None,
         recipe=SEG_RECIPE + "; frozen torchvision resnet50 C2-C5, head lr 2e-4, bs 8x2 GPU = 16, 410 steps",
         val_text="heinola VAL (382 fr; thr 0.5) IoU 0.510 / Dice 0.675 / P 0.742 / R 0.620; mask->box recall 0.854 / 1.000 "
                  "[SEGMENTATION_V5B.md s3]; leaks on every negative frame at thr <= 0.6",
         figlib=None, l40s_ms=None, l40s_src="not measured on GPU", licence=LIC_R50, role="V5b SEG baseline",
         flags=["trunk weights NOT in best.pt: SmokeSegmenter reloads models/v5b/seg/pretrained/resnet50-11ad3fa6.pth"]),
    dict(id="qwen3.5-27b_bf16_lora_SMOKETEST", v5b=True, kind="qwen_smoke",
         backend="none (cluster-only VLM; no serving backend)",
         family="Qwen3.5-27B VLM bf16 + LoRA r16 a32 adapter - 200-step SMOKE TEST",
         path="models/v5b/b200/v5bB200_qwen27b_smoke/final_adapter.tgz", run_dir="models/v5b/b200/v5bB200_qwen27b_smoke",
         report_dir="reports/v5b/b200_qwen/v5bB200_qwen27b_smoke",
         base="models/v2/qwen3.5-27b", base_rev="fc05daec18b0a78c049392ed2e771dde82bdf654",
         recipe="SMOKE TEST ONLY: 200 steps of bf16 LoRA SFT (eff. batch 64 = 12,800 examples = 0.21 epoch) on data/v2/sft "
                "(corpus-E labels; no FIgLib), LoRA r16 a32 dropout 0.05 on 256 LM linears (vision frozen), 1 MP, lr 1e-4 cosine; "
                "8x B200 pure DDP, 7.69 s/step, 0.64 node-h (B200_QWEN27B_PLAN.md s7)",
         val_text="NOT MEANINGFUL (plumbing check): 200 val imgs, parse failures 72-88% per shard, img-recall 0.14-0.32 on 25-img "
                  "shards; loss 0.877 (step 5) -> 0.392 (step 200) [B200_QWEN27B_PLAN.md s7]",
         figlib=None, l40s_ms=None, l40s_src="B200 HF generate bs 1: 14.6-16.3 s/img @192 new tokens (not L40S)",
         licence=LIC_QWEN, cluster_only=True, role="SMOKE TEST ONLY - not a trained model; feasibility evidence for a 27B bf16 arm",
         flags=["SMOKE TEST ONLY: 200 steps (0.21 epoch) = plumbing/throughput measurement, NOT a trained model; never evaluate or "
                "serve as a candidate"]),
    dict(id="ensemble_accurate_v5b", v5b=True, kind="composite", backend="composite (src.serve fusion of 4 registered backends; parallel full-frame WBF)",
         family="ACCURATE ensemble: calibrated WBF of yolo_A + dino_B4 + dino_B6 + dino_B1",
         path="models/v5b/fusion/accurate_recipe.json",
         component_ids={"yolo_A": "v4A_ctrl_s2", "dino_B4": "v4B4_lora_L", "dino_B6": "v4B6_lora_Hplus", "dino_B1": "v4B1_frozen_L"},
         recipe="NO TRAINING: recipe fit on pyro-sdis VAL only (grouped 2-fold by site cabanelle|marguerite; per-model score -> "
                "neg-image-FP quantile calibration, WBF max/iou0.3/skip0.0, equal weights); components' corpora: A = E, "
                "B4/B6/B1 = E+T (v4; AuSmoke use disclosed, not extended)",
         val_text=None, figlib=None, l40s_ms=352.0,
         l40s_src="FULL-FRAME PARALLEL = max(component) + fusion = 351.6 (B6) + 0.4 ms (ENSEMBLE_DEV_V5B.md); serial 677.3 ms",
         licence=LIC_ENSEMBLE, role="V5b S5 ACCURATE mode recipe (pre-registered)",
         flags=["thresholds did not transfer between the two VAL sites (accurate_recipe.json threshold_note): a deployed camera "
                "needs its own verified-negative operating threshold; thresholds_oof is the pooled compromise"]),
]


def latest_segment(run_root: Path):
    segs = sorted(d for d in run_root.glob("seg*") if d.is_dir() and (d / "best.pt").exists())
    return segs[-1] if segs else None


def resolve_yolo_v5b(c):
    """Final seg best.pt if the run finished, else the latest seg with a best.pt (-> in progress)."""
    root = ROOT / c["run_root"]
    final = root / c.get("final_seg", "seg05")
    info = {"run_root": c["run_root"]}
    if (final / "best.pt").exists():
        try:
            ph = json.load(open(final / "status.json")).get("phase")
        except Exception:  # noqa
            ph = None
        if ph == "done":
            info.update(path=final / "best.pt", run_dir=final, in_progress=False)
            return info
    seg = latest_segment(root)
    if seg is None:
        info.update(path=final / "best.pt", run_dir=final, in_progress=True, note="no segment with best.pt yet")
        return info
    info.update(path=seg / "best.pt", run_dir=seg, in_progress=True)
    return info


def yolo_lineage(c, prov):
    d = read_json(c["corpus_manifest"])
    a = (prov.get("args") or {})
    r = (prov.get("results_last_epoch") or {})
    lin = {"seed_args_yaml": a.get("seed"), "epochs_args_yaml": a.get("epochs"),
           "epochs_completed": int(r["epoch"]) if r.get("epoch") else None,
           "imgsz": a.get("imgsz"), "batch": a.get("batch"), "init_weights": "yolo11m.pt (COCO)",
           "data_yaml_in_pod": a.get("data"), "data_use_note": V5B_DATA_NOTE}
    if c.get("corpus_arm_A"):
        b = d["budget"]["arm_A"]
        lin.update(corpus="E (frozen expB_train list)", corpus_manifest="data/yolo/lists/local/expB_train.txt "
                   f"(budget recorded in {c['corpus_manifest']} budget.arm_A)", n_train_lines=b["n_lines"],
                   epochs_planned=b["epochs"], exposures=b["exposures"], components={"E": b["n_lines"]},
                   yaml_pod="data/yolo/yamls/pod/expB.yaml")
    else:
        b = d["budget"]
        lin.update(corpus=d["corpus"], corpus_manifest=c["corpus_manifest"], corpus_created_utc=d.get("created_utc"),
                   n_train_lines=d["counts"]["n_train_lines"], epochs_planned=b["epochs"], exposures=b["exposures"],
                   exposures_delta_vs_A_pct=b.get("delta_vs_A_pct"), components=d["counts"]["per_component"],
                   multiplicities={k: v["multiplicity"] for k, v in d["multiplicities"].items()},
                   label_origin=d["counts"]["per_label_origin"],
                   train_list_pod_sha256_12=d["lists"]["lists/pod/train.txt"]["sha256"][:12],
                   yaml_pod=os.path.relpath(d["yamls"]["pod_copy"], ROOT), excluded=d.get("excluded"))
    if lin.get("epochs_completed") is not None and lin.get("n_train_lines"):
        lin["exposures_completed"] = lin["epochs_completed"] * lin["n_train_lines"]
    return lin


def b200_dino_provenance(run_dir: Path, ck_args, ck_meta, rep_dir: Path) -> dict:
    out = dino_provenance(run_dir, ck_args, ck_meta)
    out.pop("status_error", None)
    try:
        lines = [json.loads(l) for l in open(run_dir / "metrics.jsonl") if l.strip()]
        evals = [l for l in lines if "eval" in l]
        best = max(evals, key=lambda x: x["eval"]["mAP"])
        out["metrics_n_evals"] = len(evals)
        out["metrics_best_eval"] = {"step": best["step"], "mAP": round(best["eval"]["mAP"], 4)}
        tr = [l for l in lines if "loss" in l]
        out["metrics_last_train_step"] = tr[-1]["step"] if tr else None
        out["s_per_it_median"] = round(statistics.median(l["s_per_it"] for l in tr if "s_per_it" in l), 3) if tr else None
        out["peak_mem_gib"] = max(l.get("peak_mem_gib", 0) for l in tr) if tr else None
    except Exception as e:  # noqa
        out["metrics_error"] = repr(e)
    try:
        o = json.load(open(rep_dir / "orchestrator_summary.json"))
        st = o.get("states") or []
        out["kraken_job"] = o.get("job")
        out["final_state"] = o.get("final_state") or (st[-1][1] if st else None)
        out["running_utc"] = o.get("running_utc"); out["ended_utc"] = o.get("ended_utc") or o.get("acked_utc")
        out["node_seconds_running_est"] = o.get("node_seconds_running_est")
        out["image"] = o.get("image"); out["queue"] = o.get("queue")
        out["kraken_job_hint"] = o.get("job")
    except Exception as e:  # noqa
        out["orchestrator_error"] = repr(e)
    try:
        out["latency_b200"] = json.load(open(rep_dir / "latency_b200.json"))
    except Exception as e:  # noqa
        out["latency_b200_error"] = repr(e)
    return out


def qwen_smoke_provenance(tgz: Path, rep_dir: Path) -> dict:
    out = {"segment": "single job (no segments)"}
    try:
        with tarfile.open(tgz, "r:gz") as tf:
            names = tf.getnames()
            out["adapter_tar_n_members"] = len(names)
            out["adapter_tar_members_head"] = [n for n in names if not n.endswith("/")][:12]
            out["adapter_has_safetensors"] = any(n.endswith("adapter_model.safetensors") for n in names)
            out["adapter_has_config"] = any(n.endswith("adapter_config.json") for n in names)
            cfgn = [n for n in names if n.endswith("adapter_config.json")]
            if cfgn:
                cfg = json.load(tf.extractfile(cfgn[0]))
                out["adapter_config"] = {k: cfg.get(k) for k in ("r", "lora_alpha", "lora_dropout", "target_modules",
                                                                  "base_model_name_or_path", "peft_type")}
    except Exception as e:  # noqa
        out["adapter_tar_error"] = repr(e)
    for fn, key in (("params.json", "params"), ("load_check.json", "load_check")):
        try:
            out[key] = json.load(open(rep_dir / fn))
        except Exception as e:  # noqa
            out[key + "_error"] = repr(e)
    try:
        t = json.load(open(rep_dir / "trainer_state.json"))
        lh = t.get("log_history") or []
        out["trainer_state"] = {"global_step": t.get("global_step"), "max_steps": t.get("max_steps"),
                                "epoch_fraction": round(lh[-1].get("epoch", 0), 3) if lh else None,
                                "first_loss": (lh[0].get("loss"), lh[0].get("step")) if lh else None,
                                "last_loss": (lh[-1].get("loss"), lh[-1].get("step")) if lh else None}
    except Exception as e:  # noqa
        out["trainer_state_error"] = repr(e)
    try:
        o = json.load(open(rep_dir / "orchestrator_summary.json"))
        out["status"] = {"run": o.get("run"), "phase": o.get("final_state"), "rc": None, "gpus": 8,
                         "host": o.get("job"), "ts": o.get("ended_utc")}
        out["kraken_job_hint"] = o.get("job"); out["node_seconds_running_est"] = o.get("node_seconds_running_est")
        out["image"] = o.get("image")
    except Exception as e:  # noqa
        out["orchestrator_error"] = repr(e)
    return out


def seg_provenance(run_dir: Path, path: Path, ck_meta: dict) -> dict:
    out = {"segment": run_dir.name}
    try:
        st = json.load(open(run_dir / "status.json"))
        cfg = path.parent.name
        out["status"] = {"run": st.get("run"), "phase": st.get("phase"), "rc": st.get("rc"), "host": st.get("host"),
                         "ts": st.get("ts"), "config_state": (st.get("configs") or {}).get(cfg)}
        out["kraken_job_hint"] = st.get("host")
    except Exception as e:  # noqa
        out["status_error"] = repr(e)
    try:
        man = read_json("models/v5b/seg/segV5b_a/MANIFEST.json")
        out["manifest"] = {k: man.get(k) for k in ("built_utc", "node", "node_minutes_total", "kraken_jobs", "epochs", "selection")}
        out["data_tar_sha256_12"] = (man.get("data_tar") or {}).get("sha256", "")[:12]
    except Exception as e:  # noqa
        out["manifest_error"] = repr(e)
    try:
        out["val_metrics_best"] = {k: round(v, 4) for k, v in read_json(os.path.relpath(path.parent / "val_metrics_best.json", ROOT))["summary"].items()
                                   if isinstance(v, (int, float))}
    except Exception as e:  # noqa
        out["val_metrics_error"] = repr(e)
    out.update(ck_meta)
    return out


def load_seg(path: Path, img_bgr, lat_n: int, threads: int) -> dict:
    import importlib.util
    from PIL import Image
    spec = importlib.util.spec_from_file_location("v5b_seg_infer", ROOT / "scripts/v5b/seg_infer.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pil = Image.fromarray(img_bgr[:, :, ::-1].copy())
    res = {}
    t0 = time.perf_counter()
    seg = mod.SmokeSegmenter(str(path), device="cpu", threads=threads)
    res["load_s"] = round(time.perf_counter() - t0, 2)
    res.update(kind=seg.kind, width=seg.width, height=seg.height, backbone_dir=str(seg.backbone_dir),
               best_step=seg.meta.get("best_step"), param_counts=seg.meta.get("param_counts"))
    pc = seg.meta.get("param_counts") or {}
    res["params_total"] = (pc.get("all") or {}).get("total")
    res["params_trainable"] = (pc.get("all") or {}).get("trainable")
    seg.predict(pil)  # warm-up
    ms = []
    for _ in range(lat_n):
        t0 = time.perf_counter(); prob = seg.predict(pil); ms.append((time.perf_counter() - t0) * 1000)
    res["prob_map_shape"] = list(prob.shape); res["prob_max"] = round(float(prob.max()), 4)
    res["frac_px_gt_0.5"] = round(float((prob > 0.5).mean()), 5)
    res["cpu_latency_1280"] = lat_stats(ms)
    return res


def alt_path_check(rel: str, primary_rel: str, kind: str, img) -> dict:
    p = ROOT / rel
    out = {"path": rel, "exists": p.exists()}
    if not p.exists():
        return out
    out["size_bytes"] = p.stat().st_size; out["sha256_12"] = sha256_file(p)[:12]
    if kind == "dino3":
        try:
            import torch
            a = torch.load(p, map_location="cpu", weights_only=False)
            out["top_level_keys"] = [k for k in a.keys()]
            out["has_optimizer_state"] = "opt" in a
            from src.serve.backends import load_backend
            t0 = time.perf_counter()
            be = load_backend("dino3", weights_path=str(p), device="cpu", imgsz=1280)
            be.detect(img)
            out["loads"] = True; out["load_and_detect_s"] = round(time.perf_counter() - t0, 1)
            del be, a
        except Exception as e:  # noqa
            out["loads"] = False; out["error"] = repr(e)
    return out


def model_tensors_identical(p1: Path, p2: Path) -> bool | None:
    try:
        import torch
        a = torch.load(p1, map_location="cpu", weights_only=False)["model"]
        b = torch.load(p2, map_location="cpu", weights_only=False)["model"]
        return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    except Exception:  # noqa
        return None


def composite_check(c, existing_by_id: dict) -> dict:
    rec = read_json(c["path"])
    comps = []
    for name, m in rec["models"].items():
        p = ROOT / m["checkpoint"]
        rid = c["component_ids"].get(name)
        reg = existing_by_id.get(rid) or {}
        sha = sha256_file(p) if p.exists() else None
        comps.append({"component": name, "family": m.get("family"), "checkpoint": m["checkpoint"], "exists": p.exists(),
                      "sha256_12": sha[:12] if sha else None, "registry_id": rid,
                      "sha256_matches_registry_entry": (sha == reg.get("sha256")) if (sha and reg) else None,
                      "registry_entry_loads": reg.get("loads"),
                      "latency_ms_l40s_batch1": m.get("latency_ms_l40s_batch1"), "weight": rec["fusion"]["weights"].get(name)})
    out = {"recipe": rec.get("recipe"), "mode": rec.get("mode"), "created_utc": rec.get("created_utc"),
           "scenario": rec.get("scenario"), "dev_protocol": rec.get("dev_protocol"), "components": comps,
           "fusion": {k: rec["fusion"][k] for k in ("method", "conf_type", "iou_thr", "skip_box_thr")},
           "thresholds_oof": rec.get("thresholds_oof"), "threshold_note": rec.get("threshold_note"),
           "oof_box_level": rec.get("oof_results", {}).get("box_level"),
           "training_data_disclosure": rec.get("training_data_disclosure")}
    try:
        dev = read_json("reports/v5b/ensemble_dev_v5b.json")
        out["oof_macro_mean_recall_wbf"] = dev["sets"]["yolo_A+dino_B4+dino_B6+dino_B1"]["wbf"]["oof_macro_mean"]
        out["oof_macro_source"] = "reports/v5b/ensemble_dev_v5b.json sets/yolo_A+dino_B4+dino_B6+dino_B1/wbf/oof_macro_mean (= ENSEMBLE_DEV_V5B.md T4)"
        sr = dev.get("single_raw") or {}
        out["single_model_macro_oof"] = {k: (v.get("macro_mean") if isinstance(v, dict) else v) for k, v in sr.items()} if isinstance(sr, dict) else None
    except Exception as e:  # noqa
        out["oof_macro_error"] = repr(e)
    return out


def eligibility_v5b(c, e):
    kind = c["kind"]; cpu_ms = e.get("latency_cpu_measured_median_ms")
    if kind == "yolo":
        if e.get("in_progress"):
            return {"fast_mode": False, "accurate_mode": False, "teacher": False, "reason": "IN PROGRESS - not a finished checkpoint"}
        if not e["loads"]:
            return {"fast_mode": False, "accurate_mode": False, "teacher": False, "reason": "failed to load"}
        return {"fast_mode": bool(cpu_ms is not None and cpu_ms <= 1000.0), "accurate_mode": True, "teacher": False,
                "reason": f"cpu_ms={cpu_ms}; YOLO11m = same arch as expE (11.2 ms L40S); YOLO11 arms are the STUDENT family (never teacher)"}
    if kind == "dino3":
        if not e["loads"]:
            return {"fast_mode": False, "accurate_mode": False, "teacher": False, "reason": "failed to load"}
        r5 = None
        for v in e["val"]:
            r5 = r5 or (v.get("matched_fp", {}).get("fp0.05") or {}).get("recall")
        fig = e.get("figlib39") or {}
        fig_ok = bool(fig.get("n_detected") and fig["n_detected"] > 30)
        teacher = (r5 is not None and r5 >= YOLO_A_R5) or fig_ok
        return {"fast_mode": bool(cpu_ms is not None and cpu_ms <= 1000.0), "accurate_mode": bool(c.get("accurate_ok")),
                "teacher": bool(teacher), "reason": f"cpu_ms={cpu_ms}; accurate: {c.get('accurate_reason')}; R@5%={r5} vs {YOLO_A_R5}; figlib>30={fig_ok}"}
    if kind == "seg":
        return {"fast_mode": False, "accurate_mode": False, "teacher": False,
                "reason": "segmentation overlay via scripts/v5b/seg_infer.py (not a src.serve detector backend); no matched-FP val; "
                          "must not raise alerts on its own (SEGMENTATION_V5B.md s7)" + ("" if e["loads"] else "; FAILED TO LOAD")}
    if kind == "qwen_smoke":
        return {"fast_mode": False, "accurate_mode": False, "teacher": False, "reason": "SMOKE TEST ONLY (200 steps); cluster-only VLM, no serving backend"}
    if kind == "composite":
        comps = (e.get("composite") or {}).get("components") or []
        ok = bool(comps) and all(x["exists"] and x.get("registry_entry_loads") and x.get("sha256_matches_registry_entry") for x in comps)
        return {"fast_mode": False, "accurate_mode": bool(ok and (c.get("l40s_ms") or 1e9) <= 400.0), "teacher": False,
                "reason": f"components exist+load+sha-match registry={ok}; parallel L40S {c.get('l40s_ms')} ms <= 400; CPU = sum of components "
                          f"(~20 s) -> not fast; serving recipe, not a pseudo-label source (teachers are its DINO components individually)"}
    return {"fast_mode": False, "accurate_mode": False, "teacher": False, "reason": "unknown kind"}


def build_v5b_entry(c, args, img, img_path, existing_by_id):
    import torch
    kind = c["kind"]
    e = {"id": c["id"], "v5b_addition": True, "kind": kind, "family": c["family"], "backend": c["backend"], "role": c.get("role"),
         "licence_of_weights": c["licence"], "cluster_only": bool(c.get("cluster_only")), "excluded": False,
         "train_data_recipe": c["recipe"], "flags": list(c.get("flags") or [])}
    run_dir = None
    if kind == "yolo":
        r = resolve_yolo_v5b(c)
        e["path"] = os.path.relpath(r["path"], ROOT); e["in_progress"] = r["in_progress"]; run_dir = r["run_dir"]
        e["run_root"] = c["run_root"]
    else:
        e["path"] = c["path"]; run_dir = ROOT / c["run_dir"] if c.get("run_dir") else None
    p = ROOT / e["path"]
    e["exists"] = p.exists()
    print(f"\n=== {c['id']}  exists={e['exists']}  path={e['path']}", flush=True)
    if e["exists"]:
        e["size_bytes"] = p.stat().st_size
        t0 = time.perf_counter(); full = sha256_file(p)
        e["sha256"] = full; e["sha256_12"] = full[:12]; e["sha_s"] = round(time.perf_counter() - t0, 1)
    e["backbone"] = backbone_info(c.get("backbone"))
    if c.get("base"):
        bd = ROOT / c["base"]
        e["base_weights"] = {"local_snapshot": c["base"], "exists": bd.is_dir(), "revision_pin": c.get("base_rev"),
                             "n_safetensors": len(list(bd.glob("*.safetensors"))) if bd.is_dir() else 0,
                             "also_on_cluster": "/scratch/scratch/tianrgua-sandbox/models/" + Path(c["base"]).name}
    # ---- provenance + lineage
    ck_args, ck_meta = None, {}
    if kind == "yolo":
        e["provenance"] = yolo_provenance(run_dir) if run_dir and run_dir.exists() else {"segment": None, "error": "run dir missing"}
        try:
            e["training_lineage"] = yolo_lineage(c, e["provenance"])
        except Exception as ex:  # noqa
            e["training_lineage"] = {"error": repr(ex), "corpus_manifest": c.get("corpus_manifest")}
        if e["in_progress"]:
            pv = e["provenance"]; st = pv.get("status") or {}; rl = pv.get("results_last_epoch") or {}
            tgt = (pv.get("args") or {}).get("epochs")
            e["progress"] = {"latest_segment": pv.get("segment"), "last_epoch": rl.get("epoch"), "epochs_target": tgt,
                             "phase": st.get("phase"), "train_rc": st.get("train_rc"), "ts": st.get("ts"), "host": st.get("host")}
            e["flags"].insert(0, f"IN PROGRESS: latest {pv.get('segment')} epoch {rl.get('epoch')}/{tgt}, phase {st.get('phase')} "
                                 f"(rc {st.get('train_rc')}) at {st.get('ts')}; registered checkpoint = intermediate best.pt; not evaluated")
    elif kind == "dino3":
        if e["exists"]:
            try:
                ck = torch.load(p, map_location="cpu", weights_only=False)
                ck_args = ck.get("args", {}); sd = ck.get("model", {})
                ck_meta = {"ckpt_step": ck.get("step"), "ckpt_best_map_1500": ck.get("best_map"), "ckpt_n_tensors": len(sd),
                           "ckpt_params_stored": int(sum(v.numel() for v in sd.values() if hasattr(v, "numel"))),
                           "ckpt_lora_tensors": sum(1 for k in sd if ".lora_" in k), "ckpt_has_optimizer_state": "opt" in ck}
                del ck, sd
            except Exception as ex:  # noqa
                ck_meta = {"ckpt_read_error": repr(ex)}
        e["provenance"] = b200_dino_provenance(run_dir, ck_args, ck_meta, ROOT / c["report_dir"])
        a = ck_args or {}
        pv = e["provenance"]
        e["training_lineage"] = {
            "corpus": "E + Boreal-train (no AuSmoke)", "corpus_manifest": "data/v5b/coco/instances_train_eb.json",
            "corpus_manifest_exists": (ROOT / "data/v5b/coco/instances_train_eb.json").exists(),
            "n_train_images": 40516, "epochs": a.get("epochs"), "steps": pv.get("metrics_last_train_step"),
            "eff_batch": (a.get("bs") or 0) * 8, "exposures": (pv.get("metrics_last_train_step") or 0) * (a.get("bs") or 0) * 8,
            "seed": a.get("seed"), "geometry": f"{a.get('width')}x{a.get('height')}", "lora_r": a.get("lora"),
            "lora_alpha": a.get("lora_alpha") or "None -> 2r = 32", "lr": a.get("lr"), "wd": a.get("wd"), "grad_ckpt": a.get("grad_ckpt"),
            "hardware": "8x B200 (p6, obsidian-pu, ap-south-1), bf16 autocast, torch 2.8.0+cu128",
            "init": "models/v3/hf/dinov3-vitl16 (pin ea8dc286) frozen + SFP + Faster R-CNN head from scratch",
            "best_ckpt_step": pv.get("ckpt_step"), "best_ckpt_map_1500": pv.get("ckpt_best_map_1500"),
            "data_use_note": V5B_DATA_NOTE}
        mb = pv.get("metrics_best_eval") or {}
        if mb and pv.get("ckpt_step") is not None and mb.get("step") != pv.get("ckpt_step"):
            e["flags"].append(f"CHECKPOINT != RUN BEST: shipped step {pv.get('ckpt_step')} vs metrics best mAP {mb.get('mAP')} @ step {mb.get('step')}")
    elif kind == "seg":
        if e["exists"]:
            try:
                ck = torch.load(p, map_location="cpu", weights_only=False)
                ck_args = ck.get("args", {}); sd = ck.get("model", {})
                ck_meta = {"ckpt_step": ck.get("step"), "ckpt_best_step": ck.get("best_step"), "ckpt_best_iou": ck.get("best_iou"),
                           "ckpt_epoch": ck.get("epoch"), "ckpt_kind": ck.get("kind"), "ckpt_n_tensors": len(sd),
                           "ckpt_params_stored": int(sum(v.numel() for v in sd.values() if hasattr(v, "numel"))),
                           "ckpt_lora_tensors": sum(1 for k in sd if ".lora_" in k), "ckpt_input_wh": ck.get("input_wh")}
                del ck, sd
            except Exception as ex:  # noqa
                ck_meta = {"ckpt_read_error": repr(ex)}
        e["provenance"] = seg_provenance(run_dir, p, ck_meta)
        a = ck_args or {}
        is_r50 = (ck_meta.get("ckpt_kind") == "r50")
        steps = 410 if is_r50 else 830; effb = 16 if is_r50 else 8
        e["training_lineage"] = {
            "corpus": "seg_v5b (Boreal TRAIN burns karkkila+evo, masks)", "corpus_manifest": "data/v5b/manifests/seg_v5b_manifest.json",
            "n_train": 665, "n_val": 382, "epochs": a.get("epochs"), "steps": steps, "eff_batch": effb, "exposures": steps * effb,
            "seed": a.get("seed"), "geometry": f"{a.get('width')}x{a.get('height')}", "lr": a.get("lr"), "bs_per_gpu": a.get("bs"),
            "lora_r": None if is_r50 or ck_meta.get("ckpt_kind") == "dino_frozen" else a.get("lora"),
            "hardware": "1x g6e.48xlarge 8x L40S shared by the 3 configs (frozen GPUs 4,5; lora 0-3; r50 6,7), bf16",
            "best_ckpt": {"epoch": ck_meta.get("ckpt_epoch"), "step": ck_meta.get("ckpt_best_step"), "val_iou": ck_meta.get("ckpt_best_iou")},
            "data_tar_sha256_12": e["provenance"].get("data_tar_sha256_12"), "data_use_note": V5B_DATA_NOTE}
    elif kind == "qwen_smoke":
        e["provenance"] = qwen_smoke_provenance(p, ROOT / c["report_dir"]) if e["exists"] else {"segment": None}
        ts = (e["provenance"].get("trainer_state") or {})
        e["training_lineage"] = {"corpus": "data/v2/sft (v2 SFT jsonl built from corpus-E labels; no FIgLib)",
                                 "corpus_manifest": "data/v2/sft (see models/v2/PINS.md, reports/FINAL_REPORT_V2.md)",
                                 "steps": ts.get("global_step"), "eff_batch": 64, "examples_seen": (ts.get("global_step") or 0) * 64,
                                 "exposures": (ts.get("global_step") or 0) * 64,
                                 "epochs": ts.get("epoch_fraction"), "epoch_fraction": ts.get("epoch_fraction"), "seed": "trainer default (not recorded in trainer_state)",
                                 "hardware": "8x B200 bf16, pure DDP torchrun, sdpa attention",
                                 "lora": (e["provenance"].get("adapter_config") or {}),
                                 "base": f"{c['base']} pin {c['base_rev']}", "data_use_note": V5B_DATA_NOTE,
                                 "status": "SMOKE TEST ONLY - not a trained model"}
    elif kind == "composite":
        e["provenance"] = {"segment": "n/a (recipe json)"}
        try:
            e["composite"] = composite_check(c, existing_by_id)
        except Exception as ex:  # noqa
            e["composite"] = {"error": repr(ex)}
        comp = e.get("composite") or {}
        e["training_lineage"] = {"corpus": "none (no training; fusion fit on pyro-sdis VAL OOF)",
                                 "corpus_manifest": c["path"], "components": {x["component"]: x["registry_id"] for x in comp.get("components", [])},
                                 "component_lineage": "see each component's registry entry (A: corpus E 60 ep seed 0 (same-seed replicate); "
                                                      "B4/B6/B1: corpus E_ET_v1 10 ep seed 0)",
                                 "fit_protocol": comp.get("dev_protocol"), "seed": "n/a", "epochs": 0, "exposures": 0,
                                 "data_use_note": "fusion fit on pyro-sdis VAL only; no holdout touched"}
        ob = (comp.get("oof_box_level") or {})
        mac = comp.get("oof_macro_mean_recall_wbf")
        e["val_text"] = (f"sdis-VAL grouped 2-fold OOF (WBF A+B4+B6+B1): macro mean recall over FP 1/2/5/10/20% = "
                         f"{mac:.4f} (best single B6 0.3258) [ENSEMBLE_DEV_V5B.md T4]; pooled OOF box R@5% {ob.get('fp0.05', {}).get('recall')} / "
                         f"R@10% {ob.get('fp0.10', {}).get('recall')} [accurate_recipe.json oof_results.box_level]") if mac is not None else "OOF numbers unavailable"
    # ---- val / figlib / reported latency
    e["val"] = val_numbers(c["val"]) if c.get("val") else []
    if c.get("val_text") and "val_text" not in e:
        e["val_text"] = c["val_text"]
    if kind == "yolo" and not e["val"]:
        e["val_text"] = ("IN PROGRESS - not evaluated" if e.get("in_progress") else
                         "not yet evaluated under the unified matched-FP protocol (no reports/v5b/val/eval_* artifact for this seed)")
    e["figlib39"] = figlib_numbers(c.get("figlib"))
    if e["figlib39"] and e["figlib39"].get("weights") and os.path.normpath(e["figlib39"]["weights"]) != os.path.normpath(e["path"]):
        e["flags"].append(f"FIgLib replay artifact weights {e['figlib39']['weights']} != registered path")
    e["latency_l40s_batch1_ms"] = c.get("l40s_ms"); e["latency_l40s_source"] = c.get("l40s_src")
    lb = (e.get("provenance") or {}).get("latency_b200") or {}
    if lb:
        e["latency_b200_batch1_ms"] = lb.get("batch1_ms_median")
    # ---- load test
    e["loads"] = False; e["load_test"] = None
    if not e["exists"]:
        e["load_test"] = {"status": "file missing"}
    elif c.get("cluster_only"):
        e["load_test"] = {"status": "not attempted: cluster-only (files verified only)"}
    elif kind == "composite":
        comps = (e.get("composite") or {}).get("components") or []
        ok = bool(comps) and all(x["exists"] and x.get("registry_entry_loads") for x in comps)
        e["load_test"] = {"status": "ok" if ok else "component missing/failed",
                          "note": "composite: recipe json parsed; each component verified against its own registry entry (sha + loads)"}
        e["loads"] = ok
    elif args.skip_load:
        e["load_test"] = {"status": "skipped (--skip-load)"}
    else:
        try:
            if kind == "yolo":
                lt = load_yolo(p, img, args.lat_n)
            elif kind == "dino3":
                lt = load_backend_generic("dino3", p, img, args.lat_n_dino)
            elif kind == "seg":
                lt = load_seg(p, img, args.lat_n_dino, args.threads)
            else:
                lt = {"status": "no loader"}
            lt["status"] = lt.get("status", "ok"); lt["image"] = img_path
            geo = f"{lt.get('width')}x{lt.get('height')}" if lt.get("width") else "imgsz 1280"
            lt["protocol"] = f"CPU, torch {args.threads} threads, {geo}, batch 1, warm-up excluded, JPEG decode excluded"
            e["loads"] = lt["status"] == "ok"; e["load_test"] = lt
            print("   load", lt.get("status"), lt.get("cpu_latency_1280"), lt.get("real_img_top_scores", lt.get("prob_max")), flush=True)
        except Exception as ex:  # noqa
            e["load_test"] = {"status": "FAILED", "error": repr(ex), "traceback": traceback.format_exc()[-1500:]}
            print("   LOAD FAILED:", repr(ex), flush=True)
    cpu_ms = (e["load_test"] or {}).get("cpu_latency_1280", {}).get("median_ms") if e["loads"] else None
    e["latency_cpu_measured_median_ms"] = cpu_ms
    # ---- alternate files (B200 best.pt vs best_model.pt)
    if c.get("alt_paths") and e["exists"]:
        alts = []
        for rel in c["alt_paths"]:
            a = alt_path_check(rel, e["path"], kind, img) if not args.skip_load else {"path": rel, "exists": (ROOT / rel).exists()}
            if a.get("exists"):
                a["model_tensors_identical_to_registered"] = model_tensors_identical(ROOT / rel, p)
            alts.append(a)
        e["alt_paths"] = alts
    # ---- params
    lt = e["load_test"] or {}
    if kind == "yolo":
        e["params"] = {"total": lt.get("params_total"), "trainable_at_train_time": lt.get("params_total")}
    elif kind == "dino3":
        e["params"] = {"total": lt.get("params_total"), "trainable_at_train_time": (e.get("provenance") or {}).get("ckpt_params_stored"),
                       "note": "trainable = tensors stored in best_model.pt (LoRA/neck/head); total includes the frozen HF backbone"}
    elif kind == "seg":
        e["params"] = {"total": lt.get("params_total"), "trainable_at_train_time": lt.get("params_trainable"),
                       "note": "from param_counts stored in the checkpoint (frozen backbone included in total)"}
    elif kind == "qwen_smoke":
        pr = (e.get("provenance") or {}).get("params") or {}
        e["params"] = {"total": pr.get("total_params"), "trainable_at_train_time": pr.get("trainable_params"),
                       "note": f"LoRA {pr.get('trainable_pct')}% of 27.4B (reports/v5b/b200_qwen/.../params.json)"}
    elif kind == "composite":
        tot = 0
        for x in (e.get("composite") or {}).get("components") or []:
            t = ((existing_by_id.get(x["registry_id"]) or {}).get("params") or {}).get("total")
            tot += t or 0
        e["params"] = {"total": tot or None, "trainable_at_train_time": 0, "note": "sum of component totals; fusion has 0 learned params"}
    e["eligible_for"] = eligibility_v5b(c, e)
    if not e["exists"]:
        e["flags"].append("MISSING FILE")
    if e["exists"] and not e["loads"] and not c.get("cluster_only") and kind != "composite":
        e["flags"].append("FAILS TO LOAD")
    return e


def fmt_split(e, token):
    for v in e.get("val") or []:
        if token in v.get("artifact", "") or token in v.get("key", ""):
            if "error" in v:
                return f"ERR {v['error'][:30]}"
            mf = v["matched_fp"]
            def f(fp):
                r = (mf.get(fp) or {}).get("recall")
                return "n/a" if r is None else f"{r:.3f}"
            return f"{f('fp0.05')} / {f('fp0.10')}"
    return "—"


def render_v5b_sections(reg):
    L = ["", "## V5b additions (phase 2 - student arms, B200 DINO, SEG branch, Qwen smoke test, ACCURATE recipe)", ""]
    L.append("Val columns = box recall at matched neg-image FP 5% / 10% (IoU>=0.4) on pyro-sdis VAL (7,335; 1,158 neg), Boreal heinola aerial VAL "
             "(1,123; 217 neg) and FireMan-UAV-RGBT VAL (1,706; 231 neg) from reports/v5b/val/matched_fp_*.json (B200 DINO: reports/v5b/"
             "matched_fp_b200_dino_sdis_val.json); 'n/a' = FP level outside the model's achievable range. FIgLib-39 from reports/v5b/replay/. "
             "CPU latency measured here once (batch 1, 8 threads, checkpoint geometry, median of n). GPU = reported batch-1 (L40S unless marked B200).")
    L.append("")
    L.append("| id | family | lineage: corpus · ep · exposures · seed | sdis-VAL R@5% / R@10% | Boreal-VAL | FireMan-VAL | FIgLib-39 | exists / sha256[:12] / size | loads (CPU) | CPU ms measured / GPU reported | licence of weights | fast / accurate / teacher | flags |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in reg["models"]:
        if not e.get("v5b_addition"):
            continue
        lin = e.get("training_lineage") or {}
        ep = lin.get("epochs_completed", lin.get("epochs"))
        tgt = lin.get("epochs_planned")
        if isinstance(ep, float) and ep.is_integer():
            ep = int(ep)
        eps = f"{ep}/{tgt} ep" if (tgt and ep != tgt) else f"{ep} ep"
        expo = lin.get("exposures_completed", lin.get("exposures"))
        expo_s = f"{expo/1e6:.2f}M" if isinstance(expo, (int, float)) and expo >= 1e5 else (f"{expo:,}" if isinstance(expo, (int, float)) else "—")
        seed = lin.get("seed_args_yaml", lin.get("seed"))
        corp = lin.get("corpus", "—")
        lineage = f"{corp} · {eps} · {expo_s} · seed {seed}"
        fig = e.get("figlib39")
        figs = f"{fig['n_detected']}/{fig['n_events']} (med {fig['median_delay_s']:.0f} s)" if fig and fig.get("n_detected") is not None else "—"
        ex = f"{'yes' if e['exists'] else 'NO'} / {e.get('sha256_12','—')} / {e.get('size_bytes',0)/1e6:.1f} MB"
        if e.get("in_progress"):
            loads = ("yes" if e["loads"] else "**NO**") + " (IN PROGRESS)"
        else:
            loads = "yes" if e["loads"] else ("cluster-only" if e["cluster_only"] else "**NO**")
        cpu = e.get("latency_cpu_measured_median_ms")
        lat = f"{cpu:.0f} ms / " if cpu is not None else "— / "
        if e.get("latency_b200_batch1_ms") is not None:
            lat += f"{e['latency_b200_batch1_ms']:.1f} ms B200"
        elif e.get("latency_l40s_batch1_ms") is not None:
            lat += f"{e['latency_l40s_batch1_ms']:.1f} ms"
        else:
            lat += "—"
        if e.get("val"):
            sdis = fmt_split(e, "sdisval") if any("sdisval" in v.get("artifact", "") for v in e["val"]) else fmt_split(e, "b200_dino")
        else:
            sdis = (e.get("val_text") or "none")[:140]
        el = e["eligible_for"]
        els = " / ".join("Y" if el[k] else "n" for k in ("fast_mode", "accurate_mode", "teacher"))
        flags = "; ".join(f[:100] for f in e.get("flags", [])) or ""
        L.append(f"| {e['id']} | {e['family']} | {lineage} | {sdis} | {fmt_split(e, 'borealval')} | {fmt_split(e, 'firemanval')} | {figs} | {ex} | {loads} | {lat} | {e['licence_of_weights'][:60]} | {els} | {flags} |")
    L.append("")
    L.append("### Training lineage (V5b additions)")
    L.append("Which corpus manifest, how many epochs (completed/planned), image exposures (= train lines x epochs), seed. "
             "Exposure anchor = arm A: 38,454 lines x 60 ep = 2,307,240; arms B/C/D/D2 run 43 ep on 53,5xx lines, D3 40 ep on 57,307 (all within -0.7%). "
             + V5B_DATA_NOTE)
    for e in reg["models"]:
        if not e.get("v5b_addition"):
            continue
        lin = e.get("training_lineage") or {}
        if "error" in lin:
            L.append(f"- **{e['id']}**: lineage error {lin['error']}"); continue
        bits = [f"manifest `{lin.get('corpus_manifest')}`"]
        if lin.get("n_train_lines"): bits.append(f"{lin['n_train_lines']:,} lines")
        if lin.get("n_train_images"): bits.append(f"{lin['n_train_images']:,} imgs")
        if lin.get("n_train"): bits.append(f"{lin['n_train']} train / {lin.get('n_val')} val frames")
        if lin.get("components") and e["kind"] == "yolo":
            bits.append("components " + ", ".join(f"{k} {v:,}" for k, v in lin["components"].items()))
        if lin.get("multiplicities"):
            m = {k: v for k, v in lin["multiplicities"].items() if v != 1}
            if m: bits.append("multiplicity " + ", ".join(f"{k} x{v}" for k, v in m.items()))
        if lin.get("label_origin"): bits.append("labels " + ", ".join(f"{k} {v:,}" for k, v in lin["label_origin"].items()))
        ep = lin.get("epochs_completed", lin.get("epochs")); tgt = lin.get("epochs_planned")
        if isinstance(ep, float) and ep.is_integer():
            ep = int(ep)
        bits.append(f"epochs {ep}" + (f" of {tgt} planned" if tgt and ep != tgt else ""))
        if lin.get("steps"): bits.append(f"{lin['steps']:,} steps @ eff. bs {lin.get('eff_batch')}")
        expo = lin.get("exposures_completed", lin.get("exposures"))
        if isinstance(expo, (int, float)) and expo: bits.append(f"exposures {int(expo):,}" + (f" ({lin['exposures_delta_vs_A_pct']:+.2f}% vs A)" if lin.get("exposures_delta_vs_A_pct") is not None else ""))
        bits.append(f"seed {lin.get('seed_args_yaml', lin.get('seed'))}")
        for k in ("init_weights", "init", "geometry", "hardware", "lora_r", "lr", "epoch_fraction", "examples_seen", "status", "component_lineage", "fit_protocol"):
            if lin.get(k) not in (None, "", {}): bits.append(f"{k} {lin[k]}")
        if lin.get("yaml_pod"): bits.append(f"pod yaml `{lin['yaml_pod']}`")
        if lin.get("train_list_pod_sha256_12"): bits.append(f"pod train list sha {lin['train_list_pod_sha256_12']}")
        if e.get("progress"): bits.append(f"PROGRESS {e['progress']}")
        L.append(f"- **{e['id']}** (`{e['path']}`): " + " · ".join(str(b) for b in bits))
    # alt paths / composite details
    extra = []
    for e in reg["models"]:
        if not e.get("v5b_addition"):
            continue
        for a in e.get("alt_paths") or []:
            extra.append(f"- **{e['id']}** alternate file `{a['path']}`: exists={a.get('exists')} sha {a.get('sha256_12')} {a.get('size_bytes',0)/1e6:.1f} MB, "
                         f"loads via dino3 backend={a.get('loads')}, optimizer state={a.get('has_optimizer_state')}, model tensors identical to registered file={a.get('model_tensors_identical_to_registered')}")
        comp = e.get("composite")
        if comp and comp.get("components"):
            extra.append(f"- **{e['id']}** components: " + "; ".join(
                f"{x['component']} -> {x['registry_id']} (`{x['checkpoint']}`, sha {x['sha256_12']}, matches registry={x['sha256_matches_registry_entry']}, "
                f"loads={x['registry_entry_loads']}, {x['latency_ms_l40s_batch1']} ms L40S, w={x['weight']})" for x in comp["components"])
                + f"; fusion {comp.get('fusion')}; thresholds_oof {comp.get('thresholds_oof')}; macro OOF mean recall {comp.get('oof_macro_mean_recall_wbf')}")
    if extra:
        L.append(""); L.append("### V5b file-level notes"); L.extend(extra)
    return L


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lat-n", type=int, default=10)
    ap.add_argument("--lat-n-dino", type=int, default=3)
    ap.add_argument("--skip-load", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--render-only", action="store_true", help="re-render the .md from the existing .json")
    ap.add_argument("--merge-existing", action="store_true",
                    help="keep entries of the existing MODEL_REGISTRY.json whose id is not rebuilt in this run")
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="output directory (default models/v5b; use a scratch dir for dry runs)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.render_only:
        reg = json.load(open(out_dir / "MODEL_REGISTRY.json"))
        (out_dir / "MODEL_REGISTRY.md").write_text(render_md(reg))
        print("re-rendered", out_dir / "MODEL_REGISTRY.md")
        return
    existing = None
    if args.merge_existing and (OUT_DIR / "MODEL_REGISTRY.json").exists():
        existing = json.load(open(OUT_DIR / "MODEL_REGISTRY.json"))
    existing_by_id = {m["id"]: m for m in (existing or {}).get("models", [])}

    import torch
    import cv2
    torch.set_num_threads(args.threads)
    val_list = (ROOT / "data/yolo/lists/local/sdis_val.txt").read_text().splitlines()
    img_path = val_list[0].strip()
    img = cv2.imread(img_path)
    assert img is not None, img_path

    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        repo_git = git.stdout.strip() if git.returncode == 0 else None
    except Exception:  # noqa
        repo_git = None

    import ultralytics, transformers
    env = {"host": platform.node(), "python": sys.version.split()[0], "torch": torch.__version__,
           "ultralytics": ultralytics.__version__, "transformers": transformers.__version__,
           "cuda": torch.cuda.is_available(), "torch_threads": args.threads,
           "cpu_count": os.cpu_count(), "repo_git_head": repo_git or "not a git repository",
           "loadavg_1_5_15_at_start": [round(x, 1) for x in os.getloadavg()],
           "latency_caveat": "host shared with other V5b agents during measurement (see loadavg); CPU latencies are "
                             "upper bounds (compare the reported 8-thread numbers in latency_cpu_reported)"}

    only = set(x for x in args.only.split(",") if x)
    entries = []
    for c in CANDIDATES + V5B_CANDIDATES:
        if only and c["id"] not in only:
            continue
        if c.get("v5b"):
            entries.append(build_v5b_entry(c, args, img, img_path, existing_by_id))
            continue
        p = ROOT / c["path"]
        e = {"id": c["id"], "family": c["family"], "backend": c["backend"], "role": c.get("role"),
             "path": c["path"], "exists": p.exists(), "licence_of_weights": c["licence"],
             "cluster_only": bool(c.get("cluster_only")), "excluded": bool(c.get("excluded")),
             "train_data_recipe": c["recipe"]}
        print(f"\n=== {c['id']}  exists={e['exists']}", flush=True)
        if e["exists"]:
            e["size_bytes"] = p.stat().st_size
            t0 = time.perf_counter()
            full = sha256_file(p)
            e["sha256"] = full
            e["sha256_12"] = full[:12]
            e["sha_s"] = round(time.perf_counter() - t0, 1)
        if c.get("identical_to"):
            q = ROOT / c["identical_to"]
            e["identical_to"] = {"path": c["identical_to"], "exists": q.exists(),
                                 "sha256_matches": (q.exists() and sha256_file(q) == e.get("sha256"))}
        e["backbone"] = backbone_info(c.get("backbone"))
        if c.get("base"):
            bd = ROOT / c["base"]
            e["base_weights"] = {"local_snapshot": c["base"], "exists": bd.is_dir(),
                                 "revision_pin": c.get("base_rev"),
                                 "n_safetensors": len(list(bd.glob("*.safetensors"))) if bd.is_dir() else 0,
                                 "also_on_cluster": "/scratch/scratch/tianrgua-sandbox/models/" + Path(c["base"]).name}
        # provenance
        run_dir = ROOT / c["run_dir"] if c.get("run_dir") else None
        ck_args, ck_meta = None, {}
        if c["backend"] == "yolo" and run_dir:
            e["provenance"] = yolo_provenance(run_dir)
        elif c["backend"].startswith("dino3") or c["id"].startswith("C1"):
            if e["exists"]:
                try:
                    ck = torch.load(p, map_location="cpu", weights_only=False)
                    ck_args = ck.get("args", {})
                    sd = ck.get("model", {})
                    ck_meta = {"ckpt_step": ck.get("step"), "ckpt_best_map_1500": ck.get("best_map"),
                               "ckpt_n_tensors": len(sd),
                               "ckpt_params_stored": int(sum(v.numel() for v in sd.values() if hasattr(v, "numel"))),
                               "ckpt_lora_tensors": sum(1 for k in sd if ".lora_" in k)}
                    del ck, sd
                except Exception as ex:  # noqa
                    ck_meta = {"ckpt_read_error": repr(ex)}
            e["provenance"] = dino_provenance(run_dir, ck_args, ck_meta) if run_dir else ck_meta
        elif c["backend"] == "rfdetr" and run_dir:
            prov = {"provenance_md": c["run_dir"] + "/PROVENANCE.md"}
            try:
                prov["provenance_md_head"] = (run_dir / "PROVENANCE.md").read_text().splitlines()[:6]
            except Exception as ex:  # noqa
                prov["error"] = repr(ex)
            e["provenance"] = prov
        elif c["id"].startswith("qwen") and run_dir:
            e["provenance"] = qwen_provenance(run_dir, p) if e["exists"] else {"segment": run_dir.name}
        else:
            e["provenance"] = {}
        # val / figlib / reported latency
        e["val"] = val_numbers(c["val"]) if c.get("val") else []
        if c.get("val_text"):
            e["val_text"] = c["val_text"]
        e["figlib39"] = figlib_numbers(c.get("figlib"))
        e["latency_l40s_batch1_ms"] = c.get("l40s_ms")
        e["latency_l40s_source"] = c.get("l40s_src")
        if c.get("cpu_reported"):
            e["latency_cpu_reported"] = {"artifact": c["cpu_reported"][0], "median_ms": c["cpu_reported"][1],
                                         "protocol": "devbox CPU, torch 8 threads, imgsz 1280, batch 1"}
        # load test
        e["loads"] = False
        e["load_test"] = None
        if not e["exists"]:
            e["load_test"] = {"status": "file missing"}
        elif c.get("excluded"):
            e["load_test"] = {"status": "not attempted (licence excluded)"}
        elif c.get("cluster_only"):
            e["load_test"] = {"status": "not attempted: cluster-only (files verified only)"}
        elif args.skip_load:
            e["load_test"] = {"status": "skipped (--skip-load)"}
        else:
            try:
                if c["backend"] == "yolo":
                    lt = load_yolo(p, img, args.lat_n)
                elif c["backend"] == "dino3":
                    lt = load_backend_generic("dino3", p, img, args.lat_n_dino)
                elif c["backend"] == "rfdetr":
                    lt = load_backend_generic("rfdetr", p, img, args.lat_n)
                else:
                    lt = {"status": "no loader"}
                lt["status"] = lt.get("status", "ok")
                lt["image"] = img_path
                lt["protocol"] = f"CPU, torch {args.threads} threads, imgsz 1280, batch 1, warm-up excluded, JPEG decode excluded"
                e["loads"] = lt["status"] == "ok"
                e["load_test"] = lt
                print("   load", lt.get("status"), lt.get("cpu_latency_1280"), lt.get("real_img_top_scores"), flush=True)
            except Exception as ex:  # noqa
                e["load_test"] = {"status": "FAILED", "error": repr(ex), "traceback": traceback.format_exc()[-1500:]}
                print("   LOAD FAILED:", repr(ex), flush=True)
        cpu_ms = (e["load_test"] or {}).get("cpu_latency_1280", {}).get("median_ms") if e["loads"] else None
        e["latency_cpu_measured_median_ms"] = cpu_ms
        # params
        lt = e["load_test"] or {}
        if c["backend"] == "yolo":
            e["params"] = {"total": lt.get("params_total"), "trainable_at_train_time": lt.get("params_total")}
        elif c["backend"] == "rfdetr":
            e["params"] = {"total": lt.get("params_total"), "trainable_at_train_time": lt.get("params_total"),
                           "report": "35.6M all trainable (reports/FINAL_REPORT_V3.md)"}
        elif c["backend"] == "dino3":
            stored = (e.get("provenance") or {}).get("ckpt_params_stored")
            e["params"] = {"total": lt.get("params_total"), "trainable_at_train_time": stored,
                           "note": "trainable = tensors stored in best.pt (LoRA/neck/head; B5 = full); "
                                   "total includes the frozen HF backbone (models/v4/backbone_totals.json)"}
        elif c["id"].startswith("C1"):
            e["params"] = {"total": 6811000000, "trainable_at_train_time": (e.get("provenance") or {}).get("ckpt_params_stored"),
                           "note": "6.811B total incl. frozen 7B backbone (reports/v3/trackBC_results.md)"}
        elif c["id"].startswith("qwen3.5-9b"):
            e["params"] = {"total": 9400000000, "trainable_at_train_time": "~0.9% LoRA (FINAL_REPORT_V2)"}
        elif c["id"].startswith("qwen3.5-27b"):
            e["params"] = {"total": 27000000000, "trainable_at_train_time": "QLoRA adapter (FINAL_REPORT_V2)"}
        else:
            e["params"] = {}
        e["eligible_for"] = eligibility(c, e["loads"], cpu_ms, c.get("l40s_ms"), e["val"], e["figlib39"])
        e["flags"] = []
        if not e["exists"]:
            e["flags"].append("MISSING FILE")
        if e["exists"] and not e["loads"] and not (c.get("cluster_only") or c.get("excluded")):
            e["flags"].append("FAILS TO LOAD")
        if c["id"] == "v4A_ctrl_s2":
            e["flags"].append("SEED PROVENANCE: args.yaml records seed=0 for ALL THREE arm-A runs (expV4A_ctrl_s0/s1/s2); "
                              "cluster/v4/run_yolo_arm.sh has no --seed passthrough (run_v4_arm.sh/train_entry_v4.sh do). "
                              "The three A runs differ only through non-deterministic DDP training; 'median seed' = median of 3 same-seed replicates.")
        if c["id"] == "qwen3.5-9b_sft_grpo500":
            e["flags"].append("status.json snapshot says phase=training (stale); final_adapter.tgz present and used for eval9b_grpo per FINAL_REPORT_V2")
        if c["id"] == "C1_7b_head_cached":
            e["flags"].append("7B backbone snapshot IS present locally (models/v3/hf/dinov3-vit7b16, 26 GB, pin b8036775) as well as on cluster /scratch; "
                              "not loaded here by policy (cluster-only). Head expects cached 4096-d features; no serving backend.")
        if c["backend"] == "dino3":
            try:
                rs = read_json("models/v4/dino_run_summary.json")[c["run_dir"].split("/")[-2]]
                best = max(rs["evals"], key=lambda x: x["mAP"])
                pv = e.get("provenance") or {}
                e["run_summary"] = {"artifact": "models/v4/dino_run_summary.json", "best_mAP": rs.get("best_mAP"),
                                    "best_step": best["step"], "best_seg": best["seg"], "node_hours": rs.get("node_hours"),
                                    "jobs": [j["name"] for j in rs.get("jobs", [])]}
                if pv.get("ckpt_step") != best["step"] and not c["id"].endswith("_globalbest"):
                    e["flags"].append(f"CHECKPOINT != RUN BEST: shipped best.pt is step {pv.get('ckpt_step')} "
                                      f"(best_map {round(pv.get('ckpt_best_map_1500') or -1, 4)}) but the run's best "
                                      f"1500-img COCO mAP was {best['mAP']} @ step {best['step']} ({best['seg']}); the "
                                      f"best-checkpoint tracker did not persist across a segment resume. The step-"
                                      f"{pv.get('ckpt_step')} weights are what v4 evaluated/tested (same file).")
            except Exception as ex:  # noqa
                e["run_summary"] = {"error": repr(ex)}
        if c["id"] == "v4B5_full_L":
            e["flags"].append("confounded vs B1/B4/B6 (bs 2/GPU, 2x steps, end-only checkpoint); best.pt exists only in seg11")
        entries.append(e)

    if existing is not None:
        new_ids = {e["id"] for e in entries}
        kept = [m for m in existing["models"] if m["id"] not in new_ids]
        print(f"\nmerge: keeping {len(kept)} existing entries, adding/replacing {len(entries)}", flush=True)
        order = {c["id"]: i for i, c in enumerate(CANDIDATES + V5B_CANDIDATES)}
        entries = sorted(kept + entries, key=lambda m: order.get(m["id"], 10 ** 6))
        prev = existing.get("previous_generations", []) + [{"generated_utc": existing.get("generated_utc"),
                                                            "environment": existing.get("environment"),
                                                            "n_models": len(existing.get("models", []))}]
    else:
        prev = []
    reg = {"schema": "smoke-detect MODEL_REGISTRY v5b.2",
           "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "generator": "scripts/v5b/build_model_registry.py",
           "environment": env,
           "load_test_image": img_path,
           "load_test_image_note": "first line of data/yolo/lists/local/sdis_val.txt (pyro-sdis VAL site cabanelle; "
                                   "VAL = calibration/model-selection split per PROTOCOL_V5 s2; used here only as a load smoke-test)",
           "eligibility_rules": ELIG_RULES,
           "data_use_note": "No candidate was trained on FIgLib / lockbox / pyro-sdis TEST / Boreal ruokolahti / Boreal heinola. "
                            "FIgLib-39 numbers are historical eval-only replays.",
           "v5b_data_use_note": V5B_DATA_NOTE,
           "previous_generations": prev,
           "models": entries}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MODEL_REGISTRY.json").write_text(json.dumps(reg, indent=1, default=str))
    (out_dir / "MODEL_REGISTRY.md").write_text(render_md(reg))
    print("\nwrote", out_dir / "MODEL_REGISTRY.json", out_dir / "MODEL_REGISTRY.md")


def fmt_params(p):
    if not p or p.get("total") is None:
        return "n/a"
    t = p["total"]
    tr = p.get("trainable_at_train_time")
    def m(x):
        if isinstance(x, (int, float)):
            return f"{x/1e9:.2f}B" if x >= 1e9 else f"{x/1e6:.1f}M"
        return str(x) if x is not None else "?"
    return f"{m(tr)} / {m(t)}"


def fmt_val(e):
    if e.get("val"):
        v = e["val"][0]
        if "error" in v:
            return f"ERR {v['error'][:40]}"
        mf = v["matched_fp"]
        parts = []
        for fp, lab in (("fp0.05", "5%"), ("fp0.10", "10%")):
            if fp in mf and mf[fp]["recall"] is not None:
                parts.append(f"R@{lab} {mf[fp]['recall']:.3f} (tiny {mf[fp]['tiny']:.3f}, c={mf[fp]['conf']:.3f})")
        return "; ".join(parts) + f" [{v['artifact']}#{v['key']}]"
    return e.get("val_text", "none")


def render_md(reg):
    n_v5b = sum(1 for e in reg["models"] if e.get("v5b_addition"))
    L = [f"# MODEL_REGISTRY (V5b phase 1 + {n_v5b} V5b additions) — generated {reg['generated_utc']} by `{reg['generator']}`", ""]
    if reg.get("previous_generations"):
        pg = reg["previous_generations"][0]
        n_p1 = len(reg["models"]) - n_v5b
        L.append(f"Phase-1 entries ({n_p1}) kept verbatim from the {pg.get('generated_utc')} generation "
                 f"(host {(pg.get('environment') or {}).get('host')}, loadavg at start {(pg.get('environment') or {}).get('loadavg_1_5_15_at_start')}); "
                 f"V5b additions measured on {reg['generated_utc'][:10]} with --merge-existing ({len(reg['previous_generations'])} merge generation(s); "
                 f"loadavg at start of the last run {reg['environment'].get('loadavg_1_5_15_at_start')}).")
        L.append("")
    env = reg["environment"]
    L.append(f"Host `{env['host']}` ({env['cpu_count']} vCPU, CUDA={env['cuda']}), python {env['python']}, torch {env['torch']}, "
             f"ultralytics {env['ultralytics']}, transformers {env['transformers']}; repo git HEAD: {env['repo_git_head']}.")
    L.append(f"Load test image: `{reg['load_test_image']}` — {reg['load_test_image_note']}.")
    L.append("CPU latency = measured here, batch 1, imgsz 1280, torch 8 threads, median over n (warm-up excluded, decode excluded). "
             "L40S latency = copied from the v3/v4 reports (source column in JSON). Val = pyro-sdis VAL recall at matched neg-image FP "
             "(frozen artifacts; path in brackets). FIgLib-39 = historical eval-only replay.")
    L.append("")
    L.append("Eligibility rules: " + " | ".join(f"**{k}**: {v}" for k, v in reg["eligibility_rules"].items()))
    L.append("")
    L.append("| id | family | params trainable / total | train data recipe | sdis-VAL matched-FP [artifact] | FIgLib-39 | exists / sha256[:12] / size | loads (CPU) | backend | latency CPU measured / L40S reported | licence of weights | fast / accurate / teacher | flags |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in reg["models"]:
        if e.get("v5b_addition"):
            continue
        fig = e.get("figlib39")
        figs = f"{fig['n_detected']}/{fig['n_events']} (med {fig['median_delay_s']:.0f} s)" if fig and fig.get("n_detected") is not None else "—"
        ex = f"{'yes' if e['exists'] else 'NO'} / {e.get('sha256_12','—')} / {e.get('size_bytes',0)/1e6:.1f} MB"
        loads = "yes" if e["loads"] else ("cluster-only" if e["cluster_only"] else ("excluded" if e["excluded"] else "**NO**"))
        cpu = e.get("latency_cpu_measured_median_ms")
        lat = f"{cpu:.0f} ms / " if cpu is not None else "— / "
        lat += f"{e['latency_l40s_batch1_ms']:.1f} ms" if e.get("latency_l40s_batch1_ms") is not None else "—"
        el = e["eligible_for"]
        els = " / ".join("Y" if el[k] else "n" for k in ("fast_mode", "accurate_mode", "teacher"))
        flags = "; ".join(f[:90] for f in e.get("flags", [])) or ""
        L.append(f"| {e['id']} | {e['family']} | {fmt_params(e.get('params'))} | {e['train_data_recipe'][:150]} | {fmt_val(e)} | {figs} | {ex} | {loads} | {e['backend']} | {lat} | {e['licence_of_weights'][:60]} | {els} | {flags} |")
    if n_v5b:
        L.extend(render_v5b_sections(reg))
    L.append("")
    L.append("## Load failures / flags")
    any_flag = False
    for e in reg["models"]:
        if e.get("flags") or (e["exists"] and not e["loads"] and not e["cluster_only"] and not e["excluded"]) or not e["exists"]:
            any_flag = True
            L.append(f"- **{e['id']}**: " + ("; ".join(e.get("flags", [])) or (e.get("load_test") or {}).get("status", "")))
            if e.get("load_test", {}) and e["load_test"].get("status") == "FAILED":
                L.append(f"  - error: `{e['load_test'].get('error')}`")
    if not any_flag:
        L.append("- none")
    L.append("")
    L.append("## Provenance summary (per checkpoint)")
    for e in reg["models"]:
        pv = e.get("provenance") or {}
        bits = []
        if "args" in pv:
            a = pv["args"]; bits.append(f"args.yaml: epochs {a.get('epochs')}, imgsz {a.get('imgsz')}, batch {a.get('batch')}, seed {a.get('seed')}, data {a.get('data')}")
        if "results_last_epoch" in pv:
            r = pv["results_last_epoch"]; bits.append(f"results.csv last epoch {r['epoch']}: mAP50 {r['mAP50']} mAP50-95 {r['mAP50-95']} (training-time val)")
        if "ckpt_args" in pv:
            a = pv["ckpt_args"]; bits.append(f"ckpt args: backbone_dir {a.get('backbone_dir')}, lora {a.get('lora')}, unfreeze_all {a.get('unfreeze_all')}, epochs {a.get('epochs')}, bs {a.get('bs')}, seed {a.get('seed')}")
        if "provenance_md_head" in pv:
            bits.append("PROVENANCE.md: " + " ".join(l.strip("- ").strip() for l in pv["provenance_md_head"][2:5])[:260])
        if pv.get("ckpt_step") is not None:
            bits.append(f"shipped best.pt: step {pv['ckpt_step']}, train-time best_map {round(pv.get('ckpt_best_map_1500') or -1, 4)}, "
                        f"{pv.get('ckpt_n_tensors')} tensors / {pv.get('ckpt_params_stored')} params stored")
        if "metrics_last_eval" in pv:
            bits.append(f"metrics.jsonl last eval: {pv['metrics_last_eval']}")
        if "metrics_best_eval" in pv:
            bits.append(f"metrics.jsonl best eval: {pv['metrics_best_eval']} over {pv.get('metrics_n_evals')} evals; {pv.get('s_per_it_median')} s/it, peak {pv.get('peak_mem_gib')} GiB/GPU")
        if pv.get("kraken_job"):
            bits.append(f"kraken {pv['kraken_job']} {pv.get('final_state')} {pv.get('running_utc')} -> {pv.get('ended_utc')} ({pv.get('node_seconds_running_est')} s)")
        if pv.get("ckpt_best_step") is not None:
            bits.append(f"seg best.pt: epoch {pv.get('ckpt_epoch')} step {pv.get('ckpt_best_step')} val IoU {round(pv.get('ckpt_best_iou') or -1, 4)}, {pv.get('ckpt_n_tensors')} tensors / {pv.get('ckpt_params_stored')} params stored; status config {((pv.get('status') or {}).get('config_state'))}")
        if "trainer_state" in pv:
            bits.append(f"trainer_state: {pv['trainer_state']}; LoRA {pv.get('adapter_config')}")
        if "params" in pv and isinstance(pv.get("params"), dict) and "trainable_params" in pv["params"]:
            bits.append(f"params.json: {pv['params']}")
        if "status" in pv and isinstance(pv["status"], dict):
            s = pv["status"]; bits.append(f"status: phase {s.get('phase')}, rc {s.get('train_rc', s.get('rc'))}, host {s.get('host')}, ts {s.get('ts')}")
        if "adapter_tar_n_members" in pv:
            bits.append(f"adapter tgz: {pv['adapter_tar_n_members']} members, safetensors={pv.get('adapter_has_safetensors')}, adapter_config={pv.get('adapter_has_config')}")
        if e.get("backbone"):
            b = e["backbone"]; bits.append(f"backbone snapshot {b['path']} exists={b['exists']} pin={b.get('revision_pin')} shards={b.get('n_weight_shards')} ({(b.get('weights_bytes') or 0)/1e9:.2f} GB)")
        if e.get("base_weights"):
            b = e["base_weights"]; bits.append(f"base {b['local_snapshot']} exists={b['exists']} pin={b['revision_pin']} safetensors={b['n_safetensors']}; cluster {b['also_on_cluster']}")
        if e.get("identical_to"):
            bits.append(f"byte-identical to {e['identical_to']['path']}: {e['identical_to']['sha256_matches']}")
        L.append(f"- **{e['id']}** (`{e['path']}`, segment {pv.get('segment','—')}): " + " · ".join(bits))
    L.append("")
    L.append(reg["data_use_note"])
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
