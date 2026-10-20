"""Build training curves + qualitative figures for reports/v5b/SEGMENTATION_V5B.md
from fetched artifacts (models/v5b/seg/<run>/seg<NN>/<cfg>/{metrics.jsonl,
val_metrics_best.json, val_per_image.json, val_probs.tar}).

Qualitative panels (image | GT mask | predicted prob map | overlay) are built
locally from the dumped stride-4 prob maps (no re-inference needed).
  python scripts/v5b/seg_report_figs.py --run segV5b_a --seg 1 --main lora
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/v5b/stage/seg_v5b"
EX = ROOT / "reports/v5b/seg_examples"
PAL = {"lora": "#1f5fbf", "frozen": "#d9822b", "r50": "#5b5b5b"}


def load_metrics(d: Path):
    steps, loss, evals = [], [], []
    for ln in (d / "metrics.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if "eval" in r:
            evals.append(r)
        elif "loss" in r:
            steps.append(r["step"]); loss.append(r["loss"])
    return steps, loss, evals


def curves(run_dir: Path, cfgs, out_png: Path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for c in cfgs:
        d = run_dir / c
        if not (d / "metrics.jsonl").exists():
            continue
        steps, loss, evals = load_metrics(d)
        axs[0].plot(steps, loss, color=PAL.get(c, None), label=c, lw=1.4)
        ep = [e["epoch"] for e in evals]
        axs[1].plot(ep, [e["eval"]["iou"] for e in evals], "o-", color=PAL.get(c), label=f"{c} IoU", lw=1.4)
        axs[1].plot(ep, [e["eval"]["dice"] for e in evals], "s--", color=PAL.get(c), label=f"{c} Dice", lw=1.0, alpha=.7)
        axs[2].plot(ep, [e["eval"]["precision"] for e in evals], "o-", color=PAL.get(c), label=f"{c} P", lw=1.4)
        axs[2].plot(ep, [e["eval"]["recall"] for e in evals], "s--", color=PAL.get(c), label=f"{c} R", lw=1.0, alpha=.7)
    axs[0].set(title="train loss (BCE+Dice)", xlabel="step"); axs[0].legend()
    axs[1].set(title="heinola VAL smoke-pixel IoU / Dice", xlabel="epoch", ylim=(0, 1)); axs[1].legend(fontsize=7)
    axs[2].set(title="heinola VAL precision / recall @0.5", xlabel="epoch", ylim=(0, 1)); axs[2].legend(fontsize=7)
    for a in axs:
        a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def letterbox_mask(mask_png: Path | None, orig_w, orig_h, W=1280, H=736):
    r = min(W / orig_w, H / orig_h); nw, nh = round(orig_w * r), round(orig_h * r)
    if mask_png is None:
        return np.zeros((nh, nw), bool), r, nw, nh
    m = Image.open(mask_png).convert("L").resize((nw, nh), Image.NEAREST)
    return np.asarray(m) > 127, r, nw, nh


def panel(row, prob_dir: Path, out_png: Path, title: str, thr=0.5):
    img = Image.open(DATA / row["image"]).convert("RGB")
    ow, oh = img.size
    gt, r, nw, nh = letterbox_mask(DATA / row["mask"] if row.get("mask") else None, ow, oh)
    pr = np.asarray(Image.open(prob_dir / f"{row['id']}.png")).astype(np.float32) / 255.0
    pr = np.asarray(Image.fromarray((pr * 255).astype(np.uint8)).resize((1280, 736), Image.BILINEAR)) / 255.0
    pr = pr[:nh, :nw]
    im = np.asarray(img.resize((nw, nh), Image.BILINEAR)).astype(np.float32)
    ov = im.copy(); w = (0.6 * pr)[..., None]
    heat = np.zeros_like(im); heat[..., 0] = 255 * pr
    ov = ov * (1 - w) + heat * w
    pm = pr >= thr
    e = np.zeros_like(pm); e[1:, :] |= pm[1:, :] ^ pm[:-1, :]; e[:, 1:] |= pm[:, 1:] ^ pm[:, :-1]
    ov[e] = (0, 255, 255)
    ge = np.zeros_like(gt); ge[1:, :] |= gt[1:, :] ^ gt[:-1, :]; ge[:, 1:] |= gt[:, 1:] ^ gt[:, :-1]
    ov[ge] = (0, 255, 0)
    fig, axs = plt.subplots(1, 4, figsize=(20, 3.4))
    axs[0].imshow(im.astype(np.uint8)); axs[0].set_title(row["id"], fontsize=9)
    axs[1].imshow(gt, cmap="gray", vmin=0, vmax=1); axs[1].set_title(f"GT mask ({row['provenance']})", fontsize=9)
    axs[2].imshow(pr, cmap="magma", vmin=0, vmax=1); axs[2].set_title("predicted smoke probability", fontsize=9)
    axs[3].imshow(ov.clip(0, 255).astype(np.uint8)); axs[3].set_title(title, fontsize=9)
    for a in axs:
        a.axis("off")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--seg", type=int, default=1)
    ap.add_argument("--main", default="lora"); ap.add_argument("--cfgs", default="lora,frozen,r50")
    a = ap.parse_args()
    run_dir = ROOT / "models/v5b/seg" / a.run / f"seg{a.seg:02d}"
    EX.mkdir(parents=True, exist_ok=True)
    cfgs = a.cfgs.split(",")
    curves(run_dir, cfgs, EX / f"curves_{a.run}.png")
    d = run_dir / a.main
    with tarfile.open(d / "val_probs.tar") as tf:
        tf.extractall(d)
    probs = d / "val_probs"
    per = {r["id"]: r for r in json.loads((d / "val_per_image.json").read_text())}
    rows = {r["id"]: r for r in json.loads((DATA / "manifest.json").read_text()) if r["role"] == "val"}
    pos = sorted([r for r in per.values() if not r["is_negative"]], key=lambda r: -r["iou"])
    manual = [r for r in pos if r["provenance"] == "manual_human"]
    neg = sorted([r for r in per.values() if r["is_negative"]], key=lambda r: -r["fp"])
    picks = []
    # 6 successes / typical: 2 best, 2 median, 1 manual-mask, 1 negative handled well
    picks += [("good", pos[0]), ("good", pos[1])]
    mid = len(pos) // 2; picks += [("median", pos[mid]), ("median", pos[mid + 7])]
    if manual:
        picks.append(("manual_gt", sorted(manual, key=lambda r: -r["iou"])[len(manual) // 2]))
    picks.append(("negative_ok", neg[-1]))
    # 2 failure cases: worst positive (missed smoke) + worst negative (false alarm)
    picks += [("FAIL_missed", pos[-1]), ("FAIL_false_alarm", neg[0])]
    index = []
    for i, (tag, r) in enumerate(picks):
        out = EX / f"{a.run}_{a.main}_{i+1:02d}_{tag}_{r['id']}.png"
        panel(rows[r["id"]], probs, out, f"overlay (cyan=pred@0.5, green=GT)  IoU={r['iou']:.2f} fp_pix={r['fp']}")
        index.append({"file": out.name, "tag": tag, "id": r["id"], "iou": round(r["iou"], 3), "fp": r["fp"],
                      "fn": r["fn"], "provenance": r["provenance"], "n_comp": r["n_comp"]})
    (EX / f"index_{a.run}_{a.main}.json").write_text(json.dumps(index, indent=1))
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
