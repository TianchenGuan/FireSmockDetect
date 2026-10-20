"""Automated quality filter for black-background generated smoke assets (V6).

Checks (all automated — recorded as such):
  coverage      alpha support fraction within [0.004, 0.62]
  max_alpha     must reach 0.12 (else nothing visible)
  colored_obj   saturated opaque pixels (cigarettes, bowls, ground): if
                confined to the bottom 30%, crop above them; else reject
  flat_blob     >35% of support both near-opaque and locally flat -> reject
                (solid sticker-like blob, not volumetric smoke)
  near_dup      dhash on the alpha map within the same spec_id -> reject dups
Accepted assets keep their original RGB (luma compositing mode); the manifest
carries kind (smoke|confounder from the sidecar), family_id=spec_id:seed.

Usage:
  .venv/bin/python scripts/v6/filter_assets.py --src data/v6/gen/glm_assets \
      --out data/v6/assets/C_glm --route glm [--min-side 96]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def dhash(a: np.ndarray) -> int:
    g = cv2.resize(a, (9, 8), interpolation=cv2.INTER_AREA)
    d = g[:, 1:] > g[:, :-1]
    return int("".join("1" if b else "0" for b in d.flatten()), 2)


def check_one(png: Path, meta: dict, min_side: int):
    rgb = np.asarray(Image.open(png).convert("RGB")).astype(np.float32) / 255
    h, w = rgb.shape[:2]
    if min(h, w) < min_side:
        return None, "too_small", None
    luma = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    floor = float(np.percentile(luma, 8))        # generator bg is rarely pure black
    a = np.clip((luma - floor) * 1.8, 0, 1)
    sat = rgb.max(-1) - rgb.min(-1)

    colored = (sat > 0.28) & (a > 0.3)
    if colored.mean() > 0.01:
        ys = np.nonzero(colored.any(axis=1))[0]
        if ys.min() > 0.68 * h:          # confined to bottom -> crop above
            cut = max(1, ys.min() - int(0.02 * h))
            rgb, luma, a = rgb[:cut], luma[:cut], a[:cut]
            sat = sat[:cut]
            h = cut
            if (a > 0.05).mean() < 0.004:
                return None, "empty_after_crop", None
        else:
            return None, "colored_object", None

    sup = a > 0.05
    cov = float(sup.mean())
    if cov < 0.004:
        return None, "coverage_low", None
    if cov > 0.85:
        return None, "coverage_high", None
    if float(a.max()) < 0.12:
        return None, "too_faint", None

    lv = cv2.Laplacian((luma * 255).astype(np.float32), cv2.CV_32F)
    flat_opaque = (a > 0.85) & (np.abs(lv) < 2.0)
    if sup.sum() and flat_opaque.sum() / sup.sum() > 0.35:
        return None, "flat_blob", None

    # desaturate: smoke is near-achromatic; generator colour casts (purple
    # stage lighting) are artifacts that the luma compositor must not inherit
    g3 = np.repeat(luma[..., None], 3, -1)
    rgb = 0.25 * rgb + 0.75 * g3

    # tight-crop to support with a margin (smaller assets composite faster)
    ys, xs = np.nonzero(sup)
    m = int(0.04 * max(h, w)) + 2
    y0, y1 = max(0, ys.min() - m), min(h, ys.max() + m)
    x0, x1 = max(0, xs.min() - m), min(w, xs.max() + m)
    crop = (rgb[y0:y1, x0:x1] * 255).astype(np.uint8)
    stats = {"coverage": round(cov, 4), "max_alpha": round(float(a.max()), 3),
             "crop": [int(x0), int(y0), int(x1), int(y1)], "size": [w, h]}
    return crop, None, (stats, dhash(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--min-side", type=int, default=96)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    (out / "png").mkdir(parents=True, exist_ok=True)

    seen_hashes: dict[str, list[int]] = {}
    accepted, rejects = [], []
    for png in sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")):
        mj = png.with_suffix(".json")
        meta = json.loads(mj.read_text()) if mj.exists() else {}
        if meta.get("error"):
            rejects.append({"src": str(png), "reason": "gen_error"})
            continue
        crop, why, extra = check_one(png, meta, args.min_side)
        if why:
            rejects.append({"src": str(png), "reason": why,
                            "spec_id": meta.get("spec_id")})
            continue
        stats, hsh = extra
        spec = meta.get("spec_id", "unk")
        dups = seen_hashes.setdefault(spec, [])
        if any(bin(hsh ^ h0).count("1") <= 4 for h0 in dups):
            rejects.append({"src": str(png), "reason": "near_dup", "spec_id": spec})
            continue
        dups.append(hsh)
        name = f"{args.route}_{png.stem}.png"
        Image.fromarray(crop).save(out / "png" / name)
        kind = "confounder" if meta.get("kind") == "confounder" else "smoke"
        accepted.append({"asset": str(out / "png" / name), "mode": "luma",
                         "kind": kind, "spec_id": spec,
                         "seed": meta.get("seed"),
                         "family_id": f"{args.route}:{spec}:{meta.get('seed')}",
                         "generator": meta.get("generator") or meta.get("model"),
                         "review": "automated", **stats})
    with (out / "manifest.jsonl").open("w") as f:
        for a in accepted:
            f.write(json.dumps(a) + "\n")
    with (out / "rejects.jsonl").open("w") as f:
        for r in rejects:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in rejects:
        by[r["reason"]] = by.get(r["reason"], 0) + 1
    print(json.dumps({"route": args.route, "accepted": len(accepted),
                      "rejected": len(rejects), "reject_reasons": by}, indent=1))


if __name__ == "__main__":
    main()
