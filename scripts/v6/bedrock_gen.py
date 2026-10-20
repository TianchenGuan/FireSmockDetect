"""Resumable Bedrock image-generation worker for V6.

Design:
  - a QUEUE jsonl of generation items, each with a stable content hash;
    outputs land in data/v6/gen/<route>/<hash>.png + .json — items whose
    output json already exists are SKIPPED (no double spend on resume).
  - ledger integration: reserve estimated cost for the remaining batch
    BEFORE submitting (scripts/v6/bedrock_ledger.py, exit 3 = stop), settle
    with actual successful count after.
  - per-call boto3 client built from explicit ada creds for ONE account
    (no AWS_PROFILE races); creds re-minted when close to expiry.
  - bounded retries with exponential backoff on throttling; content-filtered
    or validation-failed items are recorded, never retried blindly.

Request schemas (model-native, verified by tiny probes before bulk):
  stability sd3/core/ultra: {prompt, negative_prompt?, mode, aspect_ratio,
      output_format, seed} -> {images:[b64], seeds, finish_reasons}
  stability inpaint service: {prompt, image:b64, mask:b64, grow_mask?, seed,
      output_format, negative_prompt?} (mask polarity configurable — TEST IT)
  amazon nova-canvas: taskType TEXT_IMAGE | INPAINTING with
      imageGenerationConfig; INPAINTING uses inPaintingParams{image, maskImage,
      text, negativeText} (mask polarity configurable — TEST IT)

Usage:
  .venv/bin/python scripts/v6/bedrock_gen.py --queue data/v6/gen/q_sd35_assets.jsonl \
      --route sd35_assets --account 000000000002 --region us-west-2 \
      --price-per-image 0.08 --ledger-tag gencmp --workers 4 [--limit 20]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ADA = "ada"
LEDGER = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/v6/bedrock_ledger.py")]


def item_hash(it: dict) -> str:
    basis = json.dumps({k: it[k] for k in sorted(it) if k not in ("hash",)},
                       sort_keys=True)
    return hashlib.sha256(basis.encode()).hexdigest()[:16]


class Creds:
    def __init__(self, account: str):
        self.account = account
        self.exp = 0.0
        self.lock = threading.Lock()
        self.c = None

    def get(self):
        with self.lock:
            if time.time() > self.exp - 600:
                p = subprocess.run(
                    [ADA, "credentials", "print", "--account", self.account,
                     "--provider INTERNAL-PROVIDER",
                     "--role", "INTERNAL-ADMIN-ROLE",
                     "--format", "json"],
                    capture_output=True, text=True, timeout=90, check=True)
                self.c = json.loads(p.stdout)
                from datetime import datetime, timezone
                self.exp = datetime.strptime(
                    self.c["Expiration"], "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc).timestamp()
            return self.c


def client_for(creds: Creds, region: str):
    import boto3
    c = creds.get()
    sess = boto3.Session(aws_access_key_id=c["AccessKeyId"],
                         aws_secret_access_key=c["SecretAccessKey"],
                         aws_session_token=c["SessionToken"])
    return sess.client("bedrock-runtime", region_name=region)


def build_body(it: dict) -> bytes:
    m = it["model"].removeprefix("us.").removeprefix("global.")
    if m.startswith("stability."):
        if it["kind"].startswith("inpaint"):
            body = {"prompt": it["prompt"], "image": it["image_b64"],
                    "mask": it["mask_b64"], "output_format": "png",
                    "seed": it.get("seed", 0)}
            if it.get("grow_mask") is not None:
                body["grow_mask"] = it["grow_mask"]
        else:
            body = {"prompt": it["prompt"], "mode": "text-to-image",
                    "aspect_ratio": it.get("aspect_ratio", "1:1"),
                    "output_format": "png", "seed": it.get("seed", 0)}
        if it.get("negative"):
            body["negative_prompt"] = it["negative"]
        return json.dumps(body).encode()
    if m.startswith("amazon.nova-canvas"):
        if it["kind"].startswith("inpaint"):
            body = {"taskType": "INPAINTING",
                    "inPaintingParams": {
                        "image": it["image_b64"],
                        "maskImage": it["mask_b64"],
                        "text": it["prompt"],
                        **({"negativeText": it["negative"]}
                           if it.get("negative") else {})},
                    "imageGenerationConfig": {
                        "numberOfImages": 1, "quality": "standard",
                        "seed": it.get("seed", 0)}}
        else:
            body = {"taskType": "TEXT_IMAGE",
                    "textToImageParams": {
                        "text": it["prompt"],
                        **({"negativeText": it["negative"]}
                           if it.get("negative") else {})},
                    "imageGenerationConfig": {
                        "numberOfImages": 1, "quality": "standard",
                        "width": it.get("width", 1024),
                        "height": it.get("height", 1024),
                        "seed": it.get("seed", 0)}}
        return json.dumps(body).encode()
    raise ValueError(f"no builder for {m}")


def parse_images(model: str, payload: dict) -> tuple[list[bytes], str | None]:
    model = model.removeprefix("us.").removeprefix("global.")
    if model.startswith("stability."):
        fr = payload.get("finish_reasons") or [None]
        if fr and fr[0] not in (None, "SUCCESS", "null"):
            return [], str(fr[0])
        return [base64.b64decode(b) for b in payload.get("images", [])], None
    if model.startswith("amazon.nova-canvas"):
        if payload.get("error"):
            return [], str(payload["error"])
        return [base64.b64decode(b) for b in payload.get("images", [])], None
    return [], "unknown-model"


def ledger(args_list) -> int:
    return subprocess.run(LEDGER + args_list).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--account", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--price-per-image", type=float, required=True)
    ap.add_argument("--ledger-tag", default="gen")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rpm", type=float, default=30.0,
                    help="client-side rate cap across workers")
    args = ap.parse_args()

    outdir = ROOT / "data/v6/gen" / args.route
    outdir.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in Path(args.queue).read_text().splitlines()
             if l.strip()]
    for it in items:
        it["hash"] = it.get("hash") or item_hash(it)
    todo = [it for it in items if not (outdir / f"{it['hash']}.json").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(items)} queued, {len(todo)} to run "
          f"({len(items)-len(todo)} already done)")
    if not todo:
        return 0

    est = round(len(todo) * args.price_per_image * 1.10, 2)
    res_id = f"{args.route}-{int(time.time())}"
    rc = ledger(["reserve", "--id", res_id, "--tag", args.ledger_tag,
                 "--est-usd", str(est), "--model", todo[0]["model"],
                 "--account", args.account, "--region", args.region,
                 "--n", str(len(todo)), "--note", f"route {args.route}"])
    if rc != 0:
        print("LEDGER DENIED — not submitting"); return 3

    creds = Creds(args.account)
    done, filtered, failed = [], [], []
    lock = threading.Lock()
    interval = 60.0 / max(args.rpm, 1e-3)
    tick = {"t": 0.0}

    def pace():
        with lock:
            now = time.time()
            wait = max(0.0, tick["t"] + interval - now)
            tick["t"] = max(now, tick["t"] + interval)
        if wait > 0:
            time.sleep(wait)

    def run_one(it):
        cl = client_for(creds, args.region)
        body = build_body(it)
        for attempt in range(5):
            pace()
            try:
                t0 = time.time()
                r = cl.invoke_model(modelId=it["model"], body=body)
                payload = json.loads(r["body"].read())
                rid = r["ResponseMetadata"].get("RequestId", "")
                imgs, filt = parse_images(it["model"], payload)
                rec = {"hash": it["hash"], "model": it["model"],
                       "kind": it["kind"], "spec_id": it.get("spec_id"),
                       "request_id": rid, "latency_s": round(time.time() - t0, 2),
                       "filtered": filt, "n_images": len(imgs),
                       "prompt": it["prompt"], "params": {
                           k: it.get(k) for k in ("seed", "aspect_ratio",
                                                  "grow_mask", "mask_polarity",
                                                  "bg", "mask_box")}}
                if imgs:
                    (outdir / f"{it['hash']}.png").write_bytes(imgs[0])
                (outdir / f"{it['hash']}.json").write_text(json.dumps(rec))
                with lock:
                    (filtered if filt else done).append(it["hash"])
                return
            except Exception as e:
                msg = str(e)
                if "Throttling" in msg or "TooManyRequests" in msg:
                    time.sleep(min(60, 2 ** attempt * 3))
                    continue
                rec = {"hash": it["hash"], "model": it["model"],
                       "kind": it["kind"], "error": msg[:400]}
                (outdir / f"{it['hash']}.json").write_text(json.dumps(rec))
                with lock:
                    failed.append((it["hash"], msg[:120]))
                return
        with lock:
            failed.append((it["hash"], "throttled-out"))

    threads = []
    q = list(todo)

    def worker():
        while True:
            with lock:
                if not q:
                    return
                it = q.pop()
            run_one(it)

    for _ in range(min(args.workers, len(todo))):
        t = threading.Thread(target=worker, daemon=True)
        t.start(); threads.append(t)
    for t in threads:
        t.join()

    actual = round((len(done) + len(filtered)) * args.price_per_image, 2)
    ledger(["settle", "--id", res_id, "--actual-usd", str(actual),
            "--note", f"ok={len(done)} filtered={len(filtered)} "
                      f"failed={len(failed)}"])
    print(f"DONE route={args.route}: ok={len(done)} filtered={len(filtered)} "
          f"failed={len(failed)} actual=${actual}")
    for h, m in failed[:10]:
        print("  fail", h, m)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
