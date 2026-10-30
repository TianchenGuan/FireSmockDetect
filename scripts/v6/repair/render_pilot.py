"""Render the V6 repair pilot queue from prompt_bank_v2.yaml (v6-repair-1).

Routes (per CC_SMOKE_SYNTH_REPAIR.md + prereg A3):
  B full-scene t2i : P01–P06 x24 positives (visibility 12/8/4 per template),
                     N01–N03 x24 negatives/confounders        -> 216
  A i2i edit       : E01 x (48 bgs x 2 seeds) positives,
                     E02 x 48 matched controls (same bgs)     -> 144
Total 360. Factors are cycled deterministically (seeded) so each template
family covers position x wind x lighting rather than synonym/seed-only
variation. Exclusion list is appended as prose (GLM-Image diffusers 0.40
has NO negative_prompt parameter). The rendered prompt saved in the queue
and in each output's sidecar IS the exact string sent to the pipeline.

Backgrounds: 48 stratified from data/v6/background_pool_v1.json (TRAIN-role,
author no-smoke evidence only), copied to data/v6/repair/bgs/ and tarred.
Requested positions are generation constraints, NOT labels.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import tarfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BANK = Path(__file__).resolve().parent / "prompt_bank_v2.yaml"
OUTD = ROOT / "data/v6/repair"
BGN = {"pyro_sdis_train": 20, "dfire_train": 12, "fireman_burnA_train": 8,
       "boreal_train_karkkila_empty": 4, "sheffield_rec0005_train": 4}

AVOID = ("Avoid: studio smoke photography, incense or incense burners, "
         "candles, smoking pipes, nozzles, colored spotlights, saturated "
         "purple or magenta stage lighting, glowing smoke, explosive "
         "eruptions, mushroom-cloud explosions, giant cinematic wildfire, "
         "product photography, text, logos, watermarks.")

# 1024x576 (both /32) approximates the 16:9 monitoring aspect; i2i output
# size is derived from each background by the worker.
T2I_W, T2I_H = 1024, 576


def jhash(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def render(tmpl: str, **f) -> str:
    out = tmpl
    for k, v in f.items():
        out = out.replace("{%s}" % k, v)
    assert "{" not in out.split("<sop>")[0], f"unrendered placeholder: {out}"
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-avoid", action="store_true",
                    help="v1b: drop the appended exclusion list (GLM renders "
                         "negated concepts; controlled comparison vs v1)")
    ap.add_argument("--suffix", default="",
                    help="suffix for queue/manifest filenames (e.g. _b)")
    ap.add_argument("--skip-bgs", action="store_true",
                    help="reuse existing bgs/ and bgs.tar")
    ap.add_argument("--bank", default=str(BANK))
    ap.add_argument("--bg-luma-min", type=float, default=0.0,
                    help="drop candidate backgrounds darker than this mean "
                         "luma (night/indoor shots observed in the pool)")
    ap.add_argument("--bg-dir-name", default="bgs")
    ap.add_argument("--bg-exclude", default="",
                    help="comma-separated filename substrings to skip "
                         "(visually confirmed non-wildland pool entries)")
    args = ap.parse_args()
    global AVOID
    if args.no_avoid:
        AVOID = ""
    bank = yaml.safe_load(Path(args.bank).read_text())
    base_style = bank.get("base_style", "")
    T = {t["id"]: t for t in bank["templates"]}
    F = bank["factors"]
    VIS = F["visibility"]
    vis_quota = (["clearly_visible"] * 12 + ["small_discernible"] * 8
                 + ["faint_discernible"] * 4)

    jobs = []

    def add(route, tid, kind, prompt, seed, vis=None, factors=None, bg=None,
            bg_src=None, pair=None, **extra):
        rec = {"route": route, "spec_id": tid, "kind": kind,
               "prompt": (prompt + " " + AVOID).strip(), "seed": seed,
               "visibility": vis, "factors": factors, "bg": bg,
               "bg_src_path": bg_src, "pair": pair,
               "bank_version": bank["version"], **extra}
        rec["hash"] = jhash({k: rec[k] for k in
                             ("route", "spec_id", "prompt", "seed", "bg")})
        if route == "t2i":
            rec["width"], rec["height"] = T2I_W, T2I_H
        jobs.append(rec)

    # ---- Route B: full-scene t2i ----
    for tid in ["P01_fixed_dark_canopy", "P02_oblique_drone",
                "P03_sky_background", "P04_canopy_occlusion",
                "P05_crosswind_smolder", "P06_regional_variation"]:
        rng = random.Random("v6rp:" + tid)
        vq = vis_quota[:]
        rng.shuffle(vq)
        for i in range(24):
            f = {"position": F["position"][i % 4],
                 "wind_direction": F["wind_direction"][(i // 4) % 3],
                 "lighting": F["lighting"][(i // 12 + i) % 3],
                 "visibility": VIS[vq[i]]}
            tmpl = T[tid]["template"].replace("{base_style}", base_style)
            if "{forest_type}" in tmpl:
                f["forest_type"] = F["forest_type"][i % 4]
            add("t2i", tid, "smoke", render(tmpl, **f),
                seed=20000 + hash(tid) % 1000 + i, vis=vq[i],
                factors={k: v for k, v in f.items() if k != "visibility"})
    for tid in ["N01_clean_forest", "N02_fog", "N03_road_dust"]:
        for i in range(24):
            tmpl = T[tid]["template"].replace("{base_style}", base_style)
            f = {"lighting": F["lighting"][i % 3]} if "{lighting}" in tmpl else {}
            if "{forest_type}" in tmpl:
                f["forest_type"] = F["forest_type"][(i // 3) % 4]
            add("t2i", tid, "nonsmoke", render(tmpl, **f),
                seed=30000 + hash(tid) % 1000 + i, factors=f)

    # ---- Route A: i2i on eligible REAL TRAIN backgrounds ----
    pool = json.load((ROOT / "data/v6/background_pool_v1.json").open())["images"]
    bysrc = {}
    for im in pool:
        bysrc.setdefault(im["source"], []).append(im)
    rng = random.Random("v6rp:bgs")

    def bg_ok(path):
        if args.bg_luma_min <= 0:
            return True
        try:
            from PIL import Image as _Im
            import numpy as _np
            im = _Im.open(path).convert("L")
            im.thumbnail((96, 96))
            return float(_np.asarray(im).mean()) / 255 >= args.bg_luma_min
        except Exception:
            return False

    bgs = []
    for src, n in BGN.items():
        cand = bysrc.get(src, [])
        if not cand:
            print(f"WARN: source {src} empty; skipping")
            continue
        rng.shuffle(cand)
        excl = [e for e in args.bg_exclude.split(",") if e]
        keep = []
        for c in cand:
            if any(e in c["path"] for e in excl):
                continue
            if bg_ok(c["path"]):
                keep.append(c)
            if len(keep) >= n:
                break
        if len(keep) < n:
            print(f"WARN: {src} only {len(keep)}/{n} pass bg filter")
        bgs += keep
    bgdir = OUTD / args.bg_dir_name
    bgdir.mkdir(parents=True, exist_ok=True)
    for b in bgs:
        p = Path(b["path"])
        b["bgname"] = f"{b['source']}__{p.name}"
        if not args.skip_bgs:
            shutil.copy2(p, bgdir / b["bgname"])

    e01 = T["E01_local_smoke"]["template"].replace("{base_style}", base_style)
    e02 = T["E02_matched_control"]["template"].replace("{base_style}", base_style)
    vq48x2 = (["clearly_visible"] * 48 + ["small_discernible"] * 34
              + ["faint_discernible"] * 14)
    rng.shuffle(vq48x2)
    k = 0
    for bi, b in enumerate(bgs):
        for s in range(2):
            f = {"position": F["position"][(bi + s) % 4],
                 "wind_direction": F["wind_direction"][(bi * 2 + s) % 3],
                 "visibility": VIS[vq48x2[k]]}
            add("i2i", "E01_local_smoke", "smoke", render(e01, **f),
                seed=40000 + bi * 10 + s, vis=vq48x2[k],
                factors={x: f[x] for x in ("position", "wind_direction")},
                bg=b["bgname"], bg_src=b["path"], pair=b["bgname"])
            k += 1
        add("i2i", "E02_matched_control", "ctrl", e02,
            seed=41000 + bi, bg=b["bgname"], bg_src=b["path"],
            pair=b["bgname"])

    q = OUTD / f"q_pilot{args.suffix}.jsonl"
    with q.open("w") as fh:
        for j in jobs:
            fh.write(json.dumps(j) + "\n")
    if not args.skip_bgs:
        with tarfile.open(OUTD / f"{args.bg_dir_name}.tar", "w") as tf:
            for p in sorted(bgdir.glob("*")):
                tf.add(p, arcname=p.name)

    from collections import Counter
    stats = {"total": len(jobs),
             "by_route_kind": {f"{r}:{k}": n for (r, k), n in
                               Counter((j["route"], j["kind"]) for j in jobs).items()},
             "by_visibility": dict(Counter(j["visibility"] for j in jobs if j["visibility"])),
             "by_template": dict(Counter(j["spec_id"] for j in jobs)),
             "bgs": len(bgs), "bank_version": bank["version"],
             "avoid_clause": bool(AVOID)}
    (OUTD / f"PILOT_RENDER_MANIFEST{args.suffix}.json").write_text(json.dumps(
        {**stats, "bg_list": [b["bgname"] for b in bgs]}, indent=1, default=str))
    print(json.dumps(stats, indent=1, default=str))
    print("queue:", q, "bgs.tar:", OUTD / "bgs.tar")


if __name__ == "__main__":
    main()
