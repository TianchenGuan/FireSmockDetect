"""Assemble the V6 report package: curated per-dataset images (resized,
deterministic seed), synthetic samples with trained-box overlays, pilot
iteration contact sheets, and the Chinese reports -> one zip.

Usage: .venv/bin/python scripts/v6/repair/build_report_package.py
"""
from __future__ import annotations

import json
import random
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "deliverables/v6_report_package"
IMG = PKG / "images"
N_PER = 6
LONG = 1280
RNG = random.Random(99)


def put(src: Path, dst_dir: Path, name=None, box=None):
    dst_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    if box:  # normalized xyxy
        W, H = im.size
        d = ImageDraw.Draw(im)
        d.rectangle([box[0]*W, box[1]*H, box[2]*W, box[3]*H],
                    outline=(255, 40, 40), width=max(3, W//400))
    s = LONG / max(im.size)
    if s < 1:
        im = im.resize((round(im.size[0]*s), round(im.size[1]*s)), Image.LANCZOS)
    im.save(dst_dir / (name or src.name.replace(".png", ".jpg")), "JPEG", quality=85)


def from_list(lst, dst, n=N_PER, label_tag=True):
    lines = [l.strip() for l in Path(lst).read_text().splitlines() if l.strip()]
    for p in RNG.sample(lines, min(n, len(lines))):
        sp = Path(p)
        if not sp.exists():
            continue
        name = sp.name
        if label_tag:
            lab = Path(str(sp).replace("/images/", "/labels/")).with_suffix(".txt")
            tag = "pos" if (lab.exists() and lab.stat().st_size > 0) else "neg"
            name = f"{tag}__{name}"
        put(sp, dst, name)


def main():
    if PKG.exists():
        shutil.rmtree(PKG)
    IMG.mkdir(parents=True)

    # --- real datasets ---
    from_list(ROOT/"data/yolo/lists/local/expB_train.txt", IMG/"real_01_pyro_sdis_dfire_train")
    from_list(ROOT/"data/yolo/lists/local/sdis_val.txt", IMG/"real_02_sdis_val")
    from_list(ROOT/"data/yolo/lists/local/sdis_test.txt", IMG/"real_03_sdis_test_H2")
    from_list(ROOT/"data/v4/eval/boreal_test/lists/local/test.txt", IMG/"real_04_boreal_test")
    from_list(ROOT/"data/v5b/eval/fireman_holdout_s10/lists/local/fireman_holdout_s10.txt", IMG/"real_05_fireman_holdout")
    from_list(ROOT/"data/v5b/eval/sheffield_holdout/lists/local/sheffield_holdout.txt", IMG/"real_06_sheffield_holdout")
    from_list(ROOT/"data/v4/eval/forest_test/lists/local/test.txt", IMG/"real_07_forest_test")
    fs = sorted((ROOT/"data/v6/eval/firespot_eval/images").glob("*.jpg"))
    for p in RNG.sample(fs, N_PER):
        lab = ROOT/"data/v6/eval/firespot_eval/labels"/(p.stem+".txt")
        tag = "pos" if (lab.exists() and lab.stat().st_size > 0) else "neg"
        put(p, IMG/"real_08_firespot_HNEW", f"{tag}__{p.name}")

    # --- synthetic: arm B procedural (trained, negative result) ---
    bman = [json.loads(l) for l in (ROOT/"data/v6/synthB2/manifest.jsonl").open()]
    bpos = [r for r in bman if r.get("type") == "pos" and r.get("boxes_norm")]
    for r in RNG.sample(bpos, N_PER):
        b = r["boxes_norm"][0]
        box = [b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2]
        put(Path(r["out"]), IMG/"synth_01_armB_procedural", Path(r["out"]).name, box=box)

    # --- synthetic: CORPUS_REP (Crep) with weak boxes + matched controls ---
    rman = [json.loads(l) for l in (ROOT/"data/v6/synthREP/manifest.jsonl").open()]
    rpos = [r for r in rman if r["decision"] == "accept_pos" and r.get("box")]
    rctl = [r for r in rman if r["decision"] == "accept_ctrl"]
    for r in RNG.sample(rpos, N_PER):
        put(ROOT/f"data/v6/synthREP/images/rep_{r['hash']}.jpg",
            IMG/"synth_02_crep_accepted_pos", f"rep_{r['hash']}.jpg", box=r["box"])
    for r in RNG.sample(rctl, 3):
        put(ROOT/f"data/v6/synthREP/images/rep_{r['hash']}.jpg",
            IMG/"synth_03_crep_controls", f"ctrl_{r['hash']}.jpg")

    # --- discarded GLM black-bg assets + pilot iterations ---
    for p in RNG.sample(sorted((ROOT/"data/v6/assets/C_glm/png").glob("*.png")), 4):
        put(p, IMG/"gen_01_glm_blackbg_DISCARDED")
    sheets = {"gen_02_pilot_v1_FAILED_magenta": "/tmp/t2i_contact.jpg",
              "gen_03_pilot_v1c_final_t2i": "/tmp/t2i_c_contact.jpg",
              "gen_04_pilot_v1c_final_i2i_pairs": "/tmp/i2i_c_contact.jpg"}
    for dname, sp in sheets.items():
        if Path(sp).exists():
            put(Path(sp), IMG/dname, Path(sp).name)
    # bedrock comparison (never trained)
    bd = sorted((ROOT/"data/v6/gen").glob("cmp_t2i/*.png"))[:3] + \
         sorted((ROOT/"data/v6/gen").glob("cmp_inpaint/*.png"))[:3]
    for p in bd:
        put(p, IMG/"gen_05_bedrock_comparison_NOT_TRAINED")
    # chain panels (diagnostics)
    for p in sorted((ROOT/"reports/v6/repair/panels").glob("chain_*.png"))[:6]:
        put(p, IMG/"diag_chain_panels", p.name.replace(".png", ".jpg"))

    print("image folders:")
    for d in sorted(IMG.iterdir()):
        print(" ", d.name, len(list(d.glob("*"))))


if __name__ == "__main__":
    main()
