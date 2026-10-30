"""Blind two-stage model-assisted review of repair-pilot images (V6, prereg A3).

Stage 1 (BLIND): the reviewer sees ONLY the full frame at detector input
scale (1280 long side). It is not told the class, the requested location, or
that smoke should be present. 'uncertain' is allowed. Negatives and matched
controls go through the identical prompt, so reviewer hallucination is
measured on the same footing.

Stage 2 (CROP): for frames where stage 1 said yes/uncertain (or where the
i2i diff has support), a crop review checks plume morphology and studio/
compositing artifacts.

i2i construction checks (local, no model): pixel diff vs the resized
original background -> change support fraction, change outside the blind
box, global-restyle score. Construction evidence only — NEVER the sole
smoke evidence (per CC_SMOKE_SYNTH_REPAIR.md).

Decision rules (automated, recorded per record):
  smoke:    stage1 yes + stage2 shape/artifact pass (+ i2i localized)
            -> accept_pos (weak label = union of stage-1 boxes; i2i also
            records diff support). stage1 uncertain -> uncertain bucket
            (never a hard positive, never auto-negative). stage1 no ->
            reject_invisible.
  nonsmoke: stage1 no -> accept_neg; yes/uncertain -> reject_smoke_in_neg
            (counts toward the generation-contamination/hallucination rate).
  ctrl:     stage1 no AND no global restyle -> accept_ctrl; else reject.

All verdicts are MODEL-ASSISTED (Claude Haiku 4.5 on Bedrock), never human
ground truth. Spend goes through the shared Bedrock ledger.

Usage:
  .venv/bin/python scripts/v6/repair/blind_review.py \
      --src data/v6/gen/glm_repair_pilot --account 000000000002 \
      --region us-east-1 --out data/v6/repair/review_pilot.jsonl
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/v6"))
from bedrock_gen import Creds, ledger  # noqa: E402

MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
PRICE_IN, PRICE_OUT = 1.0, 5.0
DET_LONG = 1280  # student detector input long side

SYS1 = """You are screening outdoor monitoring-camera images. You are NOT \
told whether any smoke is present; many images contain none. Answer ONLY one \
JSON object with keys:
 smoke ("yes"|"no"|"uncertain") - is combustion smoke visible anywhere?
 boxes (array of [x0,y0,x1,y1] normalized 0-1, max 3) - tightest regions \
containing that smoke; empty unless smoke is "yes" or "uncertain"
 where (short location phrase, "" if none)
 alt ("none"|"fog"|"cloud"|"dust"|"steam"|"glare"|"haze"|"other") - most \
likely NON-smoke explanation for any gray/white region
 scene_realism (1-5 int) - 5 = looks like a real outdoor monitoring photo, \
1 = obviously synthetic, stylized, or studio-lit"""

SYS2 = """You review a CROP from a synthetic image generated for \
smoke-detector training. Answer ONLY one JSON object with keys:
 plume_shape_ok (bool) - an irregular, widening smoke plume with translucent \
edges and a coherent interior (NOT a thin decorative stem, solid sticker, or \
geometric patch)
 artifacts (array from ["incense_stem","source_prop","pedestal_or_floor",\
"colored_stage_light","neon_tint","halo","hard_cutout_edge","checkerboard",\
"sticker_flat","glow_block","text_or_watermark","explosion_like",\
"no_visible_smoke"]) - empty if clean
 attached (bool or null) - plume base visually connected to terrain/canopy \
(null if the crop cannot show this)
 blend_ok (bool) - lighting, haze and color consistent with the scene
 quality (1-5 int) - usefulness as a smoke-detection training example"""


def to_det_jpeg(img: Image.Image, long_side=DET_LONG) -> bytes:
    s = long_side / max(img.size)
    if s < 1:
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    return buf.getvalue()


def ask(cl, sys_prompt: str, jpeg: bytes, extra_text: str = "Return the JSON."):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 400,
        "system": sys_prompt,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/jpeg",
             "data": base64.b64encode(jpeg).decode()}},
            {"type": "text", "text": extra_text}]}]})
    r = cl.invoke_model(modelId=MODEL, body=body)
    d = json.loads(r["body"].read())
    txt = "".join(b.get("text", "") for b in d["content"]
                  if b.get("type") == "text")
    usage = d.get("usage", {})
    try:
        return json.loads(txt[txt.index("{"):txt.rindex("}") + 1]), usage
    except Exception:
        return {"parse_error": txt[:200]}, usage


def i2i_diff(gen: Image.Image, bg_path: str):
    """Construction check: diff vs original background resized to gen size."""
    try:
        bg = Image.open(bg_path).convert("RGB").resize(gen.size, Image.LANCZOS)
    except Exception as e:
        return {"diff_error": str(e)[:120]}
    a = np.asarray(gen.convert("RGB"), np.float32) / 255
    b = np.asarray(bg, np.float32) / 255
    d = np.abs(a - b).mean(-1)
    sup = d > 0.06
    ys, xs = np.nonzero(sup)
    box = None
    if len(ys) > 50:
        h, w = d.shape
        box = [round(float(xs.min() / w), 3), round(float(ys.min() / h), 3),
               round(float(xs.max() / w), 3), round(float(ys.max() / h), 3)]
    return {"change_frac": round(float(sup.mean()), 4),
            "mean_change": round(float(d.mean()), 4),
            "p99_change": round(float(np.percentile(d, 99)), 4),
            "diff_box": box,
            # restyle = widespread moderate change (not one localized edit)
            "global_restyle": bool((d > 0.04).mean() > 0.35)}


def union_box(boxes):
    bs = [b for b in boxes or [] if isinstance(b, (list, tuple)) and len(b) == 4]
    if not bs:
        return None
    return [min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs)]


def decide(kind, s1, s2, diff):
    smoke = (s1 or {}).get("smoke")
    bad_art = {"incense_stem", "source_prop", "pedestal_or_floor",
               "colored_stage_light", "neon_tint", "explosion_like",
               "text_or_watermark"}
    if kind == "smoke":
        if smoke == "no":
            return "reject_invisible"
        if smoke == "uncertain":
            return "uncertain"
        if s2:
            arts = set(s2.get("artifacts") or [])
            if arts & bad_art:
                return "reject_artifact"
            if s2.get("plume_shape_ok") is False:
                return "reject_morphology"
        if diff and diff.get("global_restyle"):
            return "reject_global_restyle"
        return "accept_pos"
    if kind == "nonsmoke":
        return "accept_neg" if smoke == "no" else "reject_smoke_in_neg"
    if kind == "ctrl":
        if smoke != "no":
            return "reject_smoke_in_ctrl"
        if diff and diff.get("global_restyle"):
            return "reject_global_restyle"
        return "accept_ctrl"
    return "unknown_kind"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--account", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    src = ROOT / args.src
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if outp.exists():
        for l in outp.read_text().splitlines():
            try:
                seen.add(json.loads(l)["hash"])
            except Exception:
                pass
    metas = []
    for mj in sorted(src.glob("*.json")):
        m = json.loads(mj.read_text())
        if m.get("error") or m["hash"] in seen:
            continue
        if (src / f"{m['hash']}.jpg").exists():
            metas.append(m)
    if args.nshards > 1:
        metas = [m for i, m in enumerate(metas) if i % args.nshards == args.shard]
    if args.limit:
        metas = metas[:args.limit]
    print(f"{len(metas)} images to review")
    if not metas:
        return 0

    est = round(len(metas) * 2 * (2600 * PRICE_IN + 300 * PRICE_OUT) / 1e6
                * 1.2, 2)
    res_id = f"blindrev-{int(time.time())}"
    if ledger(["reserve", "--id", res_id, "--tag", "review",
               "--est-usd", str(est), "--model", MODEL,
               "--account", args.account, "--region", args.region,
               "--n", str(len(metas)), "--note", "repair pilot blind review"]):
        print("LEDGER DENIED"); return 3

    creds = Creds(args.account)
    import boto3
    tin = tout = n_ok = 0
    with outp.open("a") as f:
        for i, m in enumerate(metas):
            img = Image.open(src / f"{m['hash']}.jpg")
            c = creds.get()
            sess = boto3.Session(aws_access_key_id=c["AccessKeyId"],
                                 aws_secret_access_key=c["SecretAccessKey"],
                                 aws_session_token=c["SessionToken"])
            cl = sess.client("bedrock-runtime", region_name=args.region)
            rec = {"hash": m["hash"], "kind": m.get("kind"),
                   "spec_id": m.get("spec_id"), "route": m.get("route"),
                   "visibility": m.get("visibility"), "bg": m.get("bg"),
                   "reviewer": MODEL, "review_kind": "model_assisted",
                   "det_long_side": DET_LONG}
            try:
                s1, u = ask(cl, SYS1, to_det_jpeg(img))
                tin += u.get("input_tokens", 0); tout += u.get("output_tokens", 0)
                rec["stage1_blind"] = s1

                diff = None
                if m.get("bg_src_path"):
                    diff = i2i_diff(img, m["bg_src_path"])
                    rec["i2i_diff"] = diff

                ub = union_box(s1.get("boxes")) or (diff or {}).get("diff_box")
                s2 = None
                if ub and s1.get("smoke") in ("yes", "uncertain"):
                    w, h = img.size
                    mx, my = 0.15 * (ub[2] - ub[0]) + 0.02, 0.15 * (ub[3] - ub[1]) + 0.02
                    x0, y0 = max(0, (ub[0] - mx)) * w, max(0, (ub[1] - my)) * h
                    x1, y1 = min(1, (ub[2] + mx)) * w, min(1, (ub[3] + my)) * h
                    if x1 - x0 > 16 and y1 - y0 > 16:
                        crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
                        s2, u2 = ask(cl, SYS2, to_det_jpeg(crop, 800))
                        tin += u2.get("input_tokens", 0)
                        tout += u2.get("output_tokens", 0)
                        rec["stage2_crop"] = s2
                rec["weak_box_union"] = ub
                rec["decision"] = decide(m.get("kind"), s1, s2, diff)
                n_ok += 1
            except Exception as e:
                rec["error"] = str(e)[:300]
                if "Throttl" in str(e) or "TooMany" in str(e):
                    time.sleep(12)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            time.sleep(0.4)
            if i % 25 == 0:
                print(f"  {i}/{len(metas)} tok_in={tin}", flush=True)

    actual = round((tin * PRICE_IN + tout * PRICE_OUT) / 1e6, 2)
    ledger(["settle", "--id", res_id, "--actual-usd", str(actual),
            "--note", f"blind-reviewed={n_ok} tin={tin} tout={tout}"])
    print(f"DONE reviewed={n_ok} actual=${actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
