"""Build a V6 synthetic corpus (arm B procedural, arm C/D generative).

Given an asset bank (manifest jsonl: {asset, mode, kind, family_id, ...}) and
the background pool, composes n_pos positives + n_neg negatives:
  negatives = 50% matched controls (ctrl of a positive, same degrade params),
              30% confounder composites (confounder-kind assets),
              20% plain degraded backgrounds.
All four streams run through the IDENTICAL degrade+JPEG pipeline. Positives
are ~45% faint-stratum. Invisible composites (post-resize) are retried with
new params up to 3x, then written as negatives-with-ignore flag (counted,
never positive).

Outputs:
  <out>/images/*.jpg  <out>/labels/*.txt  <out>/masks/*.png
  <out>/manifest.jsonl  — one record per written image, full provenance
  <out>/stats.json
  <out>/list_pod.txt — pod-side list (prefix substitution)

Usage:
  .venv/bin/python scripts/v6/build_corpus_v6.py --assets data/v6/assets/B/manifest.jsonl \
      --bg data/v6/background_pool_v1.json --n-pos 27197 --n-neg 11257 \
      --out data/v6/synthB --pod-prefix /scratch/scratch/tianrgua-sandbox/smoke/v6data/synthB \
      --seed 60 --workers 48
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v6"))
import composite as C  # noqa: E402


def make_placement(rng, faint, j):
    """Placement from an empirical real-plume anchor (x, y_base) when the
    background's source has labelled smoke statistics; uniform-in-terrain for
    aerial sources. Confounders go to the sky band (cloud) or anchor level
    (fog)."""
    anchor = j.get("anchor")
    kind = "smoke" if j["type"] == "pos" else "conf"
    base = C.sample_placement(rng, faint)         # scales/opacity/tint/etc.
    if j.get("policy") == "aerial":
        base.y_frac = rng.uniform(0.2, 0.85)
        base.y_mode = "bottom" if kind == "smoke" else "center"
        return base
    if anchor:
        ax, ay = anchor
        if kind == "smoke":
            base.x_frac = min(0.95, max(0.05, ax + rng.uniform(-0.05, 0.05)))
            base.y_frac = min(0.97, max(0.1, ay + rng.uniform(-0.03, 0.05)))
            base.y_mode = "bottom"
        else:
            if rng.random() < 0.5:                 # cloud in sky band
                base.y_frac = rng.uniform(0.05, max(0.06, ay * 0.6))
            else:                                  # fog near terrain level
                base.y_frac = min(0.95, ay + rng.uniform(-0.06, 0.04))
            base.y_mode = "center"
        return base
    return base                                     # legacy uniform fallback


def one_job(j):
    rng = random.Random(j["seed"])
    out_stem = Path(j["out_stem"])
    try:
        if j["type"] in ("pos", "conf"):
            tries, rec = 0, None
            while tries < 3:
                placement = make_placement(rng, j["faint"], j)
                degrade = C.sample_degrade(rng)
                rec = C.compose_one(j["bg"], j["asset"], j["mode"], placement,
                                    degrade, out_stem, rng,
                                    write_control=j["want_ctrl"])
                if rec["visible"] or j["type"] == "conf":
                    break
                tries += 1
            if j["type"] == "conf":
                # confounders are NEGATIVES: no boxes regardless of visibility
                Path(f"{out_stem}.txt").write_text("")
                rec["boxes_norm"] = []
                rec["label"] = "synth_confounder"
            elif rec["visible"]:
                rec["label"] = "synth_smoke_faint" if j["faint"] else "synth_smoke"
            else:
                # stayed invisible after retries: keep as ignore-flagged negative
                Path(f"{out_stem}.txt").write_text("")
                rec["label"] = "synth_invisible_ignore"
            rec.update(type=j["type"], family=j["family"], name=out_stem.name,
                       anchor=j.get("anchor"), policy=j.get("policy"))
            return rec
        else:  # plain degraded background
            degrade = C.sample_degrade(rng)
            import numpy as np
            from PIL import Image
            bg = np.asarray(Image.open(j["bg"]).convert("RGB")).astype("float32") / 255
            img = C.degrade_pair(bg, degrade, rng)
            img = C.jpeg_roundtrip(img, degrade.jpeg_q)
            out_stem.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((img * 255).astype("uint8")).save(f"{out_stem}.jpg", quality=92)
            Path(f"{out_stem}.txt").write_text("")
            return {"type": "plainneg", "bg": j["bg"], "label": "synth_plain_neg",
                    "degrade": C.asdict(degrade), "visible": False,
                    "boxes_norm": [], "name": out_stem.name,
                    "family": j["family"], "out": f"{out_stem}.jpg"}
    except Exception as e:
        return {"type": j["type"], "error": str(e)[:200], "name": out_stem.name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--n-pos", type=int, required=True)
    ap.add_argument("--n-neg", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pod-prefix", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--faint-frac", type=float, default=0.45)
    args = ap.parse_args()

    assets = [json.loads(l) for l in Path(args.assets).read_text().splitlines()]
    smoke_assets = [a for a in assets if a["kind"] in ("smoke", "copy_paste", "parametric")]
    conf_assets = [a for a in assets if a["kind"] == "confounder"]
    bgs = json.loads(Path(args.bg).read_text())
    bg_entries = bgs if isinstance(bgs, list) else bgs["images"]
    rng = random.Random(args.seed)

    # empirical plume-base anchors (x_center, y_bottom) from REAL train labels:
    # per-camera for pyro-sdis, one global pool for dfire.
    def site_of(name):
        parts = name.split("_")
        return "_".join(parts[:2]) if len(parts) >= 3 else "global"

    anchors_site, anchors_global = {}, {"pyro_sdis": [], "dfire": []}
    train_list = ROOT / "data/yolo/lists/local/expB_train.txt"
    for ip in train_list.read_text().splitlines():
        ip = ip.strip()
        if not ip:
            continue
        lp = Path(ip.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
        if not lp.exists() or lp.stat().st_size == 0:
            continue
        src = "pyro_sdis" if "pyro_sdis" in ip else "dfire"
        for line in lp.read_text().splitlines():
            f = line.split()
            if len(f) != 5:
                continue
            _, cx, cy, w, h = map(float, f)
            a = (cx, min(0.99, cy + h / 2))
            # dfire is a one-off web-image mix with no per-camera geometry:
            # keep only ground-plausible bases (y>=0.4) in its global pool,
            # else overlook-style backgrounds get plumes based in the clouds
            if src == "dfire" and a[1] < 0.4:
                continue
            anchors_global[src].append(a)
            if src == "pyro_sdis":
                anchors_site.setdefault(site_of(Path(ip).name), []).append(a)
    print(f"anchor pools: sdis sites={len(anchors_site)} "
          f"sdis={len(anchors_global['pyro_sdis'])} dfire={len(anchors_global['dfire'])}")

    AERIAL = ("boreal", "fireman", "sheffield")

    def bg_pick():
        e = rng.choice(bg_entries)
        p, src = e["path"], e.get("source", "")
        if any(a in src or a in p for a in AERIAL):
            return p, "aerial", None
        if "pyro_sdis" in p:
            pool = anchors_site.get(site_of(Path(p).name)) or anchors_global["pyro_sdis"]
            return p, "empirical", rng.choice(pool)
        return p, "empirical", rng.choice(anchors_global["dfire"]) if anchors_global["dfire"] else None

    out = ROOT / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    n_ctrl = int(args.n_neg * 0.5)
    n_conf = int(args.n_neg * 0.3)
    n_plain = args.n_neg - n_ctrl - n_conf
    jobs = []
    for i in range(args.n_pos):
        a = smoke_assets[i % len(smoke_assets)]
        bgp, policy, anchor = bg_pick()
        jobs.append({"type": "pos", "bg": bgp, "policy": policy,
                     "anchor": anchor,
                     "asset": a["asset"], "mode": a["mode"],
                     "family": a.get("family_id", a["asset"]),
                     "faint": rng.random() < args.faint_frac,
                     "want_ctrl": i < n_ctrl,
                     "seed": rng.randrange(1 << 30),
                     "out_stem": str(out / "images" / f"p{i:06d}")})
    for i in range(n_conf):
        a = conf_assets[i % len(conf_assets)] if conf_assets else None
        bgp, policy, anchor = bg_pick()
        if a is None:
            jobs.append({"type": "plain", "bg": bgp,
                         "family": "plain", "seed": rng.randrange(1 << 30),
                         "out_stem": str(out / "images" / f"c{i:06d}")})
            continue
        jobs.append({"type": "conf", "bg": bgp, "policy": policy,
                     "anchor": anchor,
                     "asset": a["asset"], "mode": a["mode"],
                     "family": a.get("family_id", a["asset"]),
                     "faint": rng.random() < 0.5, "want_ctrl": False,
                     "seed": rng.randrange(1 << 30),
                     "out_stem": str(out / "images" / f"c{i:06d}")})
    for i in range(n_plain):
        bgp, policy, anchor = bg_pick()
        jobs.append({"type": "plain", "bg": bgp,
                     "family": "plain", "seed": rng.randrange(1 << 30),
                     "out_stem": str(out / "images" / f"n{i:06d}")})

    print(f"composing {len(jobs)} jobs ({args.n_pos} pos, {n_ctrl} ctrl-neg, "
          f"{n_conf} conf-neg, {n_plain} plain-neg) with {args.workers} procs")
    recs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, rec in enumerate(ex.map(one_job, jobs, chunksize=16)):
            recs.append(rec)
            if k % 2000 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)

    # move labels/masks next to ultralytics convention; collect lists
    img_list, errors, invis = [], 0, 0
    with (out / "manifest.jsonl").open("w") as mf:
        for rec in recs:
            if rec.get("error"):
                errors += 1
                mf.write(json.dumps(rec) + "\n")
                continue
            name = rec["name"]
            ip = out / "images" / f"{name}.jpg"
            # side-written files -> labels/ masks/
            t = out / "images" / f"{name}.txt"
            if t.exists():
                t.rename(out / "labels" / f"{name}.txt")
            m = out / "images" / f"{name}_mask.png"
            if m.exists():
                m.rename(out / "masks" / f"{name}_mask.png")
            if rec.get("label") == "synth_invisible_ignore":
                invis += 1
            img_list.append(str(ip))
            # matched control written by compose_one
            cj = out / "images" / f"{name}_ctrl.jpg"
            ct = out / "images" / f"{name}_ctrl.txt"
            if cj.exists():
                ct.rename(out / "labels" / f"{name}_ctrl.txt")
                img_list.append(str(cj))
                mf.write(json.dumps({"type": "ctrl", "of": name,
                                     "label": "synth_matched_ctrl",
                                     "out": str(cj), "bg": rec["bg"]}) + "\n")
            mf.write(json.dumps(rec) + "\n")

    local_prefix = str(out)
    (out / "list_local.txt").write_text("\n".join(img_list) + "\n")
    (out / "list_pod.txt").write_text(
        "\n".join(p.replace(local_prefix, args.pod_prefix) for p in img_list) + "\n")
    # unique accounting (brief §4: composites != independent phenomena)
    uniq_bg = len({r.get("bg") for r in recs if r.get("bg")})
    uniq_assets = len({r.get("asset") for r in recs if r.get("asset")})
    pos_written = sum(1 for r in recs if r.get("label", "").startswith("synth_smoke"))
    stats = {"jobs": len(jobs), "written": len(img_list), "errors": errors,
             "positives": pos_written, "invisible_kept_neg": invis,
             "unique_backgrounds": uniq_bg, "unique_assets": uniq_assets,
             "n_ctrl": n_ctrl, "n_conf": n_conf, "n_plain": n_plain,
             "faint_frac": args.faint_frac, "seed": args.seed}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
