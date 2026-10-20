"""Model-assisted structured review of synthetic images (V6).

Sends each image (optionally paired with its pre-edit original) to an
image-capable Claude on Bedrock runtime and records a STRUCTURED verdict.
This is AUTOMATED/MODEL-ASSISTED review — never recorded as human review.

Spend is tracked in the shared Bedrock ledger at Anthropic list price
(input+output tokens estimated per call, settled with measured usage).

Usage:
  .venv/bin/python scripts/v6/vlm_review.py --route sd35_assets \
      --glob 'data/v6/gen/sd35_assets/*.png' --meta-suffix .json \
      --account 000000000002 --region us-east-1 --limit 400 \
      --out data/v6/review/sd35_assets.jsonl
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v6"))
from bedrock_gen import Creds, ledger  # noqa: E402

MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
# list price USD/MTok (checked against reports/v6/private pricing when routing
# agent lands; adjust via --price-in/--price-out if different)
PRICE_IN, PRICE_OUT = 1.0, 5.0

SYS = """You review SYNTHETIC images produced for smoke-detector training.
Answer ONLY a single JSON object, no prose, with keys:
 smoke_present (bool) — is anything a human would call smoke visible?
 matches_request (bool) — does it match the stated request?
 density ("none"|"very_faint"|"faint"|"moderate"|"dense")
 artifacts (array from ["halo","hard_cutout_edge","unrealistic_texture",
   "flame_visible","text_or_watermark","global_scene_change","duplicate_plume",
   "wrong_object"]) — empty if clean
 confounder ("none"|"fog"|"cloud"|"dust"|"steam"|"other")
 plausible_in_scene (bool) — physically plausible placement/lighting?
 quality (1-5 int) — usefulness as training data for FAINT smoke detection
"""


def review_one(cl, img_path: Path, request_desc: str, orig_path: Path | None):
    content = []
    if orig_path and orig_path.exists():
        content.append({"type": "text", "text": "ORIGINAL (before edit):"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(orig_path.read_bytes()).decode()}})
        content.append({"type": "text", "text": "EDITED result:"})
    mt = "image/png" if img_path.suffix == ".png" else "image/jpeg"
    content.append({"type": "image", "source": {
        "type": "base64", "media_type": mt,
        "data": base64.b64encode(img_path.read_bytes()).decode()}})
    content.append({"type": "text",
                    "text": f"Request was: {request_desc}\nReturn the JSON."})
    body = json.dumps({"anthropic_version": "bedrock-2023-05-31",
                       "max_tokens": 300, "system": SYS,
                       "messages": [{"role": "user", "content": content}]})
    r = cl.invoke_model(modelId=MODEL, body=body)
    d = json.loads(r["body"].read())
    txt = "".join(b.get("text", "") for b in d["content"]
                  if b.get("type") == "text")
    usage = d.get("usage", {})
    try:
        j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    except Exception:
        j = {"parse_error": txt[:200]}
    return j, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--meta-suffix", default=".json",
                    help="sidecar with 'prompt' next to each image")
    ap.add_argument("--orig-key", default=None,
                    help="sidecar key holding the pre-edit original path")
    ap.add_argument("--account", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--price-in", type=float, default=PRICE_IN)
    ap.add_argument("--price-out", type=float, default=PRICE_OUT)
    args = ap.parse_args()

    imgs = sorted(Path(p) for p in glob.glob(args.glob))
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if outp.exists():
        for l in outp.read_text().splitlines():
            try:
                seen.add(json.loads(l)["image"])
            except Exception:
                pass
    todo = [p for p in imgs if str(p) not in seen]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(imgs)} images, {len(todo)} to review")
    if not todo:
        return 0

    est = round(len(todo) * (2600 * args.price_in + 250 * args.price_out) / 1e6
                * 1.2, 2)
    res_id = f"vlmrev-{args.route}-{int(time.time())}"
    if ledger(["reserve", "--id", res_id, "--tag", "review",
               "--est-usd", str(est), "--model", MODEL,
               "--account", args.account, "--region", args.region,
               "--n", str(len(todo)), "--note", f"VLM review {args.route}"]):
        print("LEDGER DENIED"); return 3

    creds = Creds(args.account)
    import boto3
    tin = tout = n_ok = 0
    with outp.open("a") as f:
        for i, p in enumerate(todo):
            meta = {}
            mp = p.with_suffix(args.meta_suffix)
            if mp.exists():
                try:
                    meta = json.loads(mp.read_text())
                except Exception:
                    pass
            desc = meta.get("prompt") or meta.get("request") or "synthetic smoke training image"
            orig = Path(meta[args.orig_key]) if args.orig_key and meta.get(args.orig_key) else None
            c = creds.get()
            sess = boto3.Session(aws_access_key_id=c["AccessKeyId"],
                                 aws_secret_access_key=c["SecretAccessKey"],
                                 aws_session_token=c["SessionToken"])
            cl = sess.client("bedrock-runtime", region_name=args.region)
            try:
                j, usage = review_one(cl, p, desc, orig)
                tin += usage.get("input_tokens", 0)
                tout += usage.get("output_tokens", 0)
                n_ok += 1
                f.write(json.dumps({"image": str(p), "route": args.route,
                                    "review": j, "reviewer": MODEL,
                                    "review_kind": "model_assisted"}) + "\n")
                f.flush()
            except Exception as e:
                f.write(json.dumps({"image": str(p), "error": str(e)[:300]}) + "\n")
                f.flush()
                if "Throttling" in str(e):
                    time.sleep(10)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} tok_in={tin}", flush=True)

    actual = round((tin * args.price_in + tout * args.price_out) / 1e6, 2)
    ledger(["settle", "--id", res_id, "--actual-usd", str(actual),
            "--note", f"reviewed={n_ok} tin={tin} tout={tout}"])
    print(f"DONE reviewed={n_ok} actual=${actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
