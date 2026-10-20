"""Build the Track-F synthetic-smoke pilot spec (iteration 2).

Selects 150 TRAIN-split pyro-sdis NEGATIVE backgrounds (sites brison,
serre-de-barre, courmettes, valbonne ONLY — never val/test/FIgLib/lockbox),
diverse cameras/times, seeded, and writes 3 generation jobs per background:

  (a) smoke      — faint localized smoke plume at a sampled region
                   (scale tiny/small alternating, low contrast)
  (b) control    — matched no-change / minor global edit (same pipeline
                   artifacts, NO smoke)
  (c) confounder — fog bank / dust haze / low cloud at the SAME region as (a)
                   (NO smoke; tests that synthetic positives are not just
                   "anything hazy")

Outputs:
  data/v2/synth/spec.jsonl          — 450 jobs, exact prompts/seeds/regions
  data/v2/synth/backgrounds.txt     — 150 devbox abs paths (tar build list)
  data/v2/synth/SPEC_META.json      — selection stats + contamination guard

IMPORTANT: the requested region/prompt is NOT a verified label. Every job
carries verified=false / label_status=unverified-synthetic; downstream
consumers must not treat region boxes as ground truth without review.

Prompt constraint: no single quotes / apostrophes anywhere (kraken YAML args
must stay single-quote-free, and we keep the whole chain clean).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2" / "synth"
SEED = 20260911
N_BG = 150
TRAIN_SITES = ["brison", "serre-de-barre", "courmettes", "valbonne"]
VALTEST_SITES = {"cabanelle", "marguerite", "ferion", "croix-augas"}
MODEL_REPO = "Qwen/Qwen-Image-Edit-2511"
MODEL_REV = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"

# generation params (recorded per job; 30 steps is a deliberate deviation
# from the model-card example value 40 to fit the 45-min segment budget on
# L40S — flagged in PILOT.md)
STEPS = 30
TRUE_CFG = 4.0
GUIDANCE = 1.0
NEG_PROMPT = " "

SMOKE_COLOR = ["pale gray", "light gray", "whitish gray", "light bluish gray"]
SMOKE_FAINT = ["extremely faint", "very faint", "faint and hard to see"]

CONFOUNDERS = [
    ("fog",
     "Add a faint low fog bank hugging the terrain {loc}. The fog is soft, "
     "horizontal, semi-transparent and low contrast. There is no smoke "
     "anywhere in the image. Keep everything else in the image exactly "
     "unchanged: same landscape, same lighting, same colors, same sky."),
    ("dust",
     "Add a faint distant dust haze over the terrain {loc}, a soft brownish "
     "semi-transparent veil staying close to the ground. There is no smoke "
     "anywhere in the image. Keep everything else in the image exactly "
     "unchanged: same landscape, same lighting, same colors, same sky."),
    ("cloud",
     "Add one small low white cloud just above the horizon {loc}, soft and "
     "semi-transparent. There is no smoke anywhere in the image. Keep "
     "everything else in the image exactly unchanged: same landscape, same "
     "lighting, same colors, same sky."),
]

CONTROLS = [
    ("nochange",
     "Reproduce this exact image with no changes at all. Do not add, remove "
     "or move any object. Keep the same landscape, same lighting, same "
     "colors, same sky, same textures."),
    ("minor",
     "Very slightly increase the overall brightness of this image, keeping "
     "every object, texture, color and the sky otherwise identical. Do not "
     "add or remove anything."),
]


def loc_desc(cx: float, cy: float) -> str:
    h = ("in the left part of the image" if cx < 1 / 3 else
         "in the middle of the image" if cx < 2 / 3 else
         "in the right part of the image")
    return (f"{h}, about {round(cx * 100)} percent from the left edge and "
            f"{round(cy * 100)} percent from the top of the image")


def sample_region(rng: random.Random, scale: str):
    if scale == "tiny":
        w = rng.uniform(0.025, 0.055)
        h = rng.uniform(0.04, 0.09)
    else:  # small
        w = rng.uniform(0.06, 0.12)
        h = rng.uniform(0.08, 0.16)
    cx = rng.uniform(0.12, 0.88)
    # plume base sits in the terrain band typical for these fixed hilltop
    # cameras (horizon usually mid-frame); NOT verified per image.
    cy = rng.uniform(0.38, 0.62)
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    return [round(v, 4) for v in (x1, y1, x2, y2)], cx, cy


def select_backgrounds(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    cand = df[df.site.isin(TRAIN_SITES) & (df.n_boxes == 0)
              & df.decode_ok & (~df.coord_violation)].copy()
    assert set(cand.site.unique()) <= set(TRAIN_SITES)
    cand = cand.sort_values("image_name").reset_index(drop=True)
    picked = []
    # per-camera even spacing over time = camera + season/hour diversity
    by_cam = {c: g.sort_values("datetime").reset_index(drop=True)
              for c, g in cand.groupby("camera_id")}
    cams = sorted(by_cam)
    # round-robin cameras; within a camera take evenly spaced timestamps
    quota = {c: 0 for c in cams}
    i = 0
    while sum(quota.values()) < N_BG:
        c = cams[i % len(cams)]
        if quota[c] < len(by_cam[c]):
            quota[c] += 1
        i += 1
        if i > 10000:
            raise RuntimeError("selection loop stuck")
    for c in cams:
        g, k = by_cam[c], quota[c]
        if k == 0:
            continue
        idx = [round(j * (len(g) - 1) / max(k - 1, 1)) for j in range(k)]
        idx = sorted(set(idx))
        while len(idx) < k:  # collisions on short cameras
            extra = rng.randrange(len(g))
            if extra not in idx:
                idx.append(extra)
        picked.append(g.iloc[sorted(idx)])
    sel = pd.concat(picked).reset_index(drop=True)
    assert len(sel) == N_BG
    return sel


def main() -> None:
    rng = random.Random(SEED)
    df = pd.read_parquet(ROOT / "data" / "manifests"
                         / "pyro_sdis_manifest.parquet")
    sel = select_backgrounds(df, rng)

    # contamination guard — hard fail on any val/test site or non-train image
    assert not set(sel.site) & VALTEST_SITES
    assert (sel.n_boxes == 0).all()
    for p in sel.image_path:
        assert "pyro_sdis" in p and Path(p).exists(), p

    OUT.mkdir(parents=True, exist_ok=True)
    jobs, seed_counter = [], 42000
    for bi, row in sel.iterrows():
        bg_rel = f"bg/{row.image_name}"
        scale = "tiny" if bi % 2 == 0 else "small"
        region, cx, cy = sample_region(rng, scale)
        loc = loc_desc(cx, cy)
        px = [round(region[0] * row.width), round(region[1] * row.height),
              round(region[2] * row.width), round(region[3] * row.height)]
        base = dict(background=bg_rel, background_src=row.image_path,
                    site=row.site, camera_id=row.camera_id,
                    datetime=str(row.datetime), width=int(row.width),
                    height=int(row.height), model_repo=MODEL_REPO,
                    model_revision=MODEL_REV, num_inference_steps=STEPS,
                    true_cfg_scale=TRUE_CFG, guidance_scale=GUIDANCE,
                    negative_prompt=NEG_PROMPT, verified=False,
                    label_status="unverified-synthetic",
                    note="requested region/prompt is a generation REQUEST, "
                         "not a verified label")

        faint = SMOKE_FAINT[bi % len(SMOKE_FAINT)]
        color = SMOKE_COLOR[bi % len(SMOKE_COLOR)]
        smoke_prompt = (
            f"Add one {faint} {scale} plume of {color} wildfire smoke rising "
            f"from the terrain {loc}. The plume is thin, wispy, "
            f"semi-transparent and very low contrast against the background, "
            f"slightly wider at its top than at its base. Keep everything "
            f"else in the image exactly unchanged: same landscape, same "
            f"lighting, same colors, same sky.")
        jobs.append(dict(job_id=f"bg{bi:03d}_smoke", kind="smoke",
                         scale=scale, region_norm=region, region_px=px,
                         loc_desc=loc, prompt=smoke_prompt,
                         seed=seed_counter, **base))
        seed_counter += 1

        cname, cprompt = CONTROLS[bi % len(CONTROLS)]
        jobs.append(dict(job_id=f"bg{bi:03d}_ctrl", kind="control",
                         control_type=cname, scale=None, region_norm=None,
                         region_px=None, loc_desc=None, prompt=cprompt,
                         seed=seed_counter, **base))
        seed_counter += 1

        fname, fprompt = CONFOUNDERS[bi % len(CONFOUNDERS)]
        jobs.append(dict(job_id=f"bg{bi:03d}_conf", kind="confounder",
                         confounder_type=fname, scale=scale,
                         region_norm=region, region_px=px, loc_desc=loc,
                         prompt=fprompt.format(loc=loc),
                         seed=seed_counter, **base))
        seed_counter += 1

    for j in jobs:  # single-quote guard for the whole YAML/shell chain
        assert "'" not in json.dumps(j), j["job_id"]

    spec = OUT / "spec.jsonl"
    with spec.open("w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    (OUT / "backgrounds.txt").write_text(
        "\n".join(sel.image_path.tolist()) + "\n")

    meta = dict(seed=SEED, n_backgrounds=len(sel), n_jobs=len(jobs),
                model_repo=MODEL_REPO, model_revision=MODEL_REV,
                train_sites_only=TRAIN_SITES,
                guard="all backgrounds are pyro-sdis TRAIN-site negatives "
                      "(n_boxes==0); val/test/FIgLib/lockbox untouched",
                per_site=sel.groupby("site").size().to_dict(),
                per_camera=sel.groupby("camera_id").size().to_dict(),
                per_kind={k: sum(1 for j in jobs if j["kind"] == k)
                          for k in ("smoke", "control", "confounder")},
                scales={s: sum(1 for j in jobs if j.get("scale") == s)
                        for s in ("tiny", "small")},
                steps=STEPS, true_cfg_scale=TRUE_CFG,
                labels_verified=False)
    (OUT / "SPEC_META.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"wrote {spec} ({len(jobs)} jobs)")


if __name__ == "__main__":
    main()
