"""Pod-side worker: Qwen-Image-Edit-2511 synthetic smoke generation.

One process per GPU (launched 8x by synth_gen_entry.sh). Loads the pipeline
from /scratch (staged weights, offline), takes spec rows [jobs_from, jobs_to)
sharded round-robin across workers, edits the matching background image, and
writes PNG + per-image provenance JSON. Stops cleanly at --budget-s so the
segment fits the presigned-URL lifetime; whatever finished gets uploaded.

VRAM: bf16 components total ~57.7 GB > 48 GB L40S -> enable_model_cpu_offload
per worker (peak ~= transformer 40.9 GB + activations).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--backgrounds", required=True, help="dir containing bg/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--shard", required=True, help="i/n")
    ap.add_argument("--device-index", type=int, required=True)
    ap.add_argument("--jobs-from", type=int, default=0)
    ap.add_argument("--jobs-to", type=int, default=10**9)
    ap.add_argument("--budget-s", type=float, default=2100.0)
    ap.add_argument("--done-file", default=None,
                    help="file of job_ids already generated (skip)")
    args = ap.parse_args()

    done_ids = set()
    if args.done_file:
        done_ids = {l.strip() for l in open(args.done_file) if l.strip()}

    t0 = time.time()
    import torch
    from PIL import Image
    from diffusers import QwenImageEditPlusPipeline

    wi, wn = (int(x) for x in args.shard.split("/"))
    rows = [json.loads(l) for l in open(args.spec) if l.strip()]
    rows = [(k, r) for k, r in enumerate(rows)
            if args.jobs_from <= k < args.jobs_to
            and r["job_id"] not in done_ids]
    mine = [(gk, r) for k, (gk, r) in enumerate(rows) if k % wn == wi]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"worker {wi}/{wn} dev{args.device_index}: {len(mine)} jobs, "
          f"budget {args.budget_s}s", flush=True)

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_model_cpu_offload(gpu_id=args.device_index)
    t_load = time.time() - t0
    print(f"worker {wi}: pipeline loaded+offload-hooked in {t_load:.0f}s",
          flush=True)

    done, timings = 0, []
    for gidx, r in mine:
        if time.time() - t0 > args.budget_s:
            print(f"worker {wi}: budget reached after {done} jobs", flush=True)
            break
        jid = r["job_id"]
        if (out / f"{jid}.png").exists():
            done += 1
            continue
        img = Image.open(Path(args.backgrounds) / r["background"]).convert(
            "RGB")
        g = torch.Generator(device="cpu").manual_seed(int(r["seed"]))
        t1 = time.time()
        with torch.inference_mode():
            res = pipe(image=[img], prompt=r["prompt"],
                       negative_prompt=r["negative_prompt"], generator=g,
                       true_cfg_scale=float(r["true_cfg_scale"]),
                       guidance_scale=float(r["guidance_scale"]),
                       num_inference_steps=int(r["num_inference_steps"]),
                       num_images_per_prompt=1)
        dt = time.time() - t1
        im = res.images[0]
        im.save(out / f"{jid}.png")
        prov = dict(r)
        prov.update(spec_index=gidx, output_size=[im.width, im.height],
                    gen_time_s=round(dt, 1), worker=wi,
                    model_path=args.model_path,
                    torch_dtype="bfloat16", offload="model_cpu_offload",
                    verified=False, screening="none")
        (out / f"{jid}.json").write_text(json.dumps(prov) + "\n")
        timings.append(dt)
        done += 1
        print(f"worker {wi}: {jid} ok {dt:.0f}s ({done}/{len(mine)})",
              flush=True)

    (out / f"worker{wi}_done.json").write_text(json.dumps(dict(
        worker=wi, assigned=len(mine), done=done, load_s=round(t_load, 1),
        mean_gen_s=round(sum(timings) / max(len(timings), 1), 1),
        timings=[round(t, 1) for t in timings])) + "\n")
    return 0 if done > 0 or len(mine) == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
