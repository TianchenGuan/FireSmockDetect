"""GLM-Image pod worker (one GPU): shard of a prompt queue -> jpgs + meta.

Defensive against pipeline-signature drift: introspects GlmImagePipeline's
__call__ and passes only supported kwargs; logs the signature once.
Writes each image immediately (resume = skip existing meta files).

env: SHARD (int), NSHARDS, QUEUE (jsonl path), OUTDIR, WEIGHTS (model dir),
     IMGSZ (default 1024), STEPS (optional), DEADLINE_EPOCH (stop early).
"""
from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

import torch


def log(*a):
    print(f"[shard {os.environ.get('SHARD','?')}]", *a, flush=True)


def main():
    shard = int(os.environ["SHARD"])
    nshards = int(os.environ["NSHARDS"])
    outdir = Path(os.environ["OUTDIR"])
    outdir.mkdir(parents=True, exist_ok=True)
    deadline = float(os.environ.get("DEADLINE_EPOCH", "0")) or None
    imgsz = int(os.environ.get("IMGSZ", "1024"))
    steps = os.environ.get("STEPS")

    items = [json.loads(l) for l in
             Path(os.environ["QUEUE"]).read_text().splitlines() if l.strip()]
    mine = [it for i, it in enumerate(items) if i % nshards == shard]
    todo = [it for it in mine
            if not (outdir / f"{it['hash']}.json").exists()]
    log(f"{len(mine)} assigned, {len(todo)} todo")
    if not todo:
        return

    import diffusers
    log("diffusers", diffusers.__version__, "torch", torch.__version__)
    Pipe = getattr(diffusers, "GlmImagePipeline")
    t0 = time.time()
    pipe = Pipe.from_pretrained(os.environ["WEIGHTS"],
                                torch_dtype=torch.bfloat16)
    try:
        pipe.to("cuda")
        resident = True
    except torch.cuda.OutOfMemoryError:
        log("OOM on full residency -> cpu offload")
        pipe.enable_model_cpu_offload()
        resident = False
    log(f"pipeline loaded in {time.time()-t0:.0f}s resident={resident}")
    sig = set(inspect.signature(pipe.__call__).parameters)
    log("call params:", sorted(sig))

    for n, it in enumerate(todo):
        if deadline and time.time() > deadline:
            log("deadline reached, stopping"); break
        kw = {"prompt": it["prompt"]}
        if "height" in sig:
            kw["height"] = it.get("height", imgsz)
            kw["width"] = it.get("width", imgsz)
        if steps and "num_inference_steps" in sig:
            kw["num_inference_steps"] = int(steps)
        if it.get("negative") and "negative_prompt" in sig:
            kw["negative_prompt"] = it["negative"]
        if "generator" in sig:
            kw["generator"] = torch.Generator("cuda").manual_seed(
                int(it.get("seed", 0)))
        t1 = time.time()
        try:
            img = pipe(**kw).images[0]
            lat = round(time.time() - t1, 1)
            img = img.convert("RGB")
            img.save(outdir / f"{it['hash']}.jpg", quality=95)
            (outdir / f"{it['hash']}.json").write_text(json.dumps(
                {"hash": it["hash"], "spec_id": it.get("spec_id"),
                 "kind": it.get("kind"), "prompt": it["prompt"],
                 "seed": it.get("seed", 0), "latency_s": lat,
                 "size": img.size, "generator": "zai-org/GLM-Image",
                 "shard": shard}))
            if n % 5 == 0:
                log(f"{n}/{len(todo)} last={lat}s")
        except Exception as e:
            (outdir / f"{it['hash']}.json").write_text(json.dumps(
                {"hash": it["hash"], "error": str(e)[:300]}))
            log("ERROR", it["hash"], str(e)[:160])
    log("worker done")


if __name__ == "__main__":
    main()
