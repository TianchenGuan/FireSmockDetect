"""Generator-comparison scoring (V6, frozen scene specs).

Routes scored:
  sd35 / core  (Bedrock t2i black-bg assets)  — filter_assets acceptance,
               diversity (dhash spread), cost/accepted, latency
  inpaint      (Bedrock in-scene edit)        — outside-mask change, in-mask
               smoke support (difference-based), latency, cost
  glm          (local GLM-Image assets)       — same as t2i, cost = GPU time
Writes reports/v6/GENCMP_V6.md + .json. VLM plausibility joins later from
data/v6/review/*.jsonl (model-assisted, labelled as such).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def t2i_route(gen_dir: Path, accepted_manifest: Path, rejects: Path,
              model_prefix: str, price: float):
    metas = [json.loads(p.read_text()) for p in gen_dir.glob("*.json")]
    metas = [m for m in metas if (m.get("model") or "").startswith(model_prefix)
             or (m.get("generator") or "").startswith(model_prefix)]
    if not metas:
        return None
    acc = [json.loads(l) for l in accepted_manifest.read_text().splitlines()] \
        if accepted_manifest.exists() else []
    rej = [json.loads(l) for l in rejects.read_text().splitlines()] \
        if rejects.exists() else []
    hashes = {m["hash"] for m in metas if "hash" in m}

    def of_route(rows):
        return [r for r in rows
                if any(h in str(r.get("asset", "") or r.get("src", "")) for h in hashes)]

    acc_r, rej_r = of_route(acc), of_route(rej)
    lat = [m["latency_s"] for m in metas if m.get("latency_s")]
    n = len(metas)
    n_acc = len(acc_r)
    return {"generated": n, "accepted": n_acc,
            "acceptance": round(n_acc / max(n, 1), 3),
            "reject_reasons": {r["reason"]: sum(1 for x in rej_r if x["reason"] == r["reason"])
                               for r in rej_r},
            "median_latency_s": round(float(np.median(lat)), 1) if lat else None,
            "cost_per_generated_usd": price,
            "cost_per_accepted_usd": round(price * n / max(n_acc, 1), 3)}


def inpaint_route(gen_dir: Path, price: float):
    rows = []
    for j in gen_dir.glob("*.json"):
        m = json.loads(j.read_text())
        if m.get("error") or not m.get("params"):
            continue
        png = gen_dir / f"{m['hash']}.png"
        if not png.exists():
            continue
        bgp = m["params"].get("bg")
        box = m["params"].get("mask_box")
        if not bgp or not box:
            continue
        out = np.asarray(Image.open(png).convert("RGB")).astype(np.float32) / 255
        bg = Image.open(bgp).convert("RGB")
        bg = np.asarray(bg.resize((out.shape[1], out.shape[0]))).astype(np.float32) / 255
        diff = np.abs(out - bg).mean(-1)
        x0, y0, x1, y1 = box
        g = 24  # grown-mask margin
        ib = diff[max(0, y0 - g):y1 + g, max(0, x0 - g):x1 + g]
        mask = np.zeros_like(diff, bool)
        mask[max(0, y0 - g):y1 + g, max(0, x0 - g):x1 + g] = True
        outside = diff[~mask]
        rows.append({"hash": m["hash"], "kind": m["kind"],
                     "spec_id": m.get("spec_id"),
                     "latency_s": m.get("latency_s"),
                     "outside_change": round(float((outside > 0.08).mean()), 4),
                     "inmask_support": round(float((ib > 0.06).mean()), 4)})
    if not rows:
        return None, []
    pos = [r for r in rows if r["kind"] == "inpaint_pos"]
    neg = [r for r in rows if r["kind"] == "inpaint_neg"]
    ok_pos = [r for r in pos if r["outside_change"] < 0.02 and r["inmask_support"] > 0.03]
    ok_neg = [r for r in neg if r["outside_change"] < 0.02]
    lat = [r["latency_s"] for r in rows if r["latency_s"]]
    summ = {"generated": len(rows), "pos": len(pos), "neg_edits": len(neg),
            "pos_localized_ok": len(ok_pos), "neg_clean": len(ok_neg),
            "acceptance_pos": round(len(ok_pos) / max(len(pos), 1), 3),
            "median_outside_change": round(float(np.median([r["outside_change"] for r in rows])), 4),
            "median_latency_s": round(float(np.median(lat)), 1) if lat else None,
            "cost_per_generated_usd": price,
            "cost_per_accepted_usd": round(price * len(pos) / max(len(ok_pos), 1), 3),
            "note": "request mask != ground truth; labels derive from output "
                    "difference support (automated)"}
    return summ, rows


def main():
    res = {}
    acc = ROOT / "data/v6/assets/cmp_sdcore/manifest.jsonl"
    rej = ROOT / "data/v6/assets/cmp_sdcore/rejects.jsonl"
    res["sd35_large"] = t2i_route(ROOT / "data/v6/gen/cmp_t2i", acc, rej,
                                  "stability.sd3-5", 0.08)
    res["stable_image_core"] = t2i_route(ROOT / "data/v6/gen/cmp_t2i", acc, rej,
                                         "stability.stable-image-core", 0.04)
    summ, rows = inpaint_route(ROOT / "data/v6/gen/cmp_inpaint", 0.07)
    res["stable_image_inpaint"] = summ
    gl_acc = ROOT / "data/v6/assets/C_glm/manifest.jsonl"
    gl_rej = ROOT / "data/v6/assets/C_glm/rejects.jsonl"
    if (ROOT / "data/v6/gen/glm_assets").exists():
        res["glm_image"] = t2i_route(ROOT / "data/v6/gen/glm_assets", gl_acc,
                                     gl_rej, "glm", 0.0)
    out = ROOT / "reports/v6"
    out.mkdir(parents=True, exist_ok=True)
    (out / "GENCMP_V6.json").write_text(json.dumps(
        {"summary": res, "inpaint_rows": rows}, indent=2))
    lines = ["# V6 generator comparison (frozen scene specs)\n",
             "| route | generated | accepted | acceptance | median lat (s) | $/accepted |",
             "|---|---|---|---|---|---|"]
    for k, v in res.items():
        if not v:
            continue
        a = v.get("accepted", v.get("pos_localized_ok"))
        ar = v.get("acceptance", v.get("acceptance_pos"))
        lines.append(f"| {k} | {v['generated']} | {a} | {ar} | "
                     f"{v.get('median_latency_s')} | {v.get('cost_per_accepted_usd')} |")
    lines.append("\nBedrock outputs are comparison/demo only (AWS Service "
                 "Terms 50.5 ambiguity, prereg A1); the training corpus "
                 "generator is GLM-Image (MIT). All acceptance checks are "
                 "AUTOMATED; VLM plausibility (model-assisted) joins from "
                 "data/v6/review/.")
    (out / "GENCMP_V6.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
