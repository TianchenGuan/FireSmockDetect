"""Replay stored frames against the server, in chronological order, as a
HISTORICAL REPLAY. Causal: sends frames strictly in time order.

Every request this client makes carries the form field
    source=historical_replay
(hard-wired, not an option) plus a run tag in `metadata`, so the server-side
shadow log and /healthz counts can never confuse a replay with a live upload.
Replay output is never a field run and must not be described as one.

Usage (directory of frames, FIgLib or 1-fps clip):
  python -m src.serve.replay_client --frames data/raw/figlib/<event> \
      --server http://127.0.0.1:8000 --device-id drone-01 \
      --out logs/replay/<event>.jsonl [--fps 0.33] [--realtime]
Usage (pyro-sdis sequence JSON with explicit per-frame ts):
  python -m src.serve.replay_client --sequence-json data/v3/sequences/sequences_val.json \
      --seq-id val_force-06_cabanelle-125_20240403T101630 --server ... --out ...

Frame ordering (--frames): sorts by FIgLib +/- second offset parsed from the
filename when present, else by filename; capture_ts is synthesized from the
offset (or idx/fps) on top of --epoch-start. NOTE: pyro-sdis image names
contain "-NN" tokens that match the FIgLib offset pattern, so sdis sequences
must be replayed via --sequence-json (real `ts` per frame), not --frames.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

OFFSET_RE = re.compile(r"([+-]\d+)")
SOURCE = "historical_replay"   # never anything else from this client


def frame_key(p: Path):
    m = OFFSET_RE.search(p.stem)
    return (0, int(m.group(1))) if m else (1, p.name)


def frame_offset_s(p: Path, idx: int, fps: float):
    m = OFFSET_RE.search(p.stem)
    return int(m.group(1)) if m else idx / fps


def make_run_tag(label: str) -> str:
    return f"replay-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{label}"


def build_form(device_id: str, capture_ts: float, run_tag: str, extra: dict | None = None) -> dict:
    """The exact multipart data fields sent with every replayed frame."""
    meta = {"run_tag": run_tag, "source": SOURCE, "replay_client": "src.serve.replay_client"}
    if extra:
        meta.update(extra)
    return {"device_id": device_id, "capture_ts": capture_ts,
            "source": SOURCE, "metadata": json.dumps(meta, separators=(",", ":"))}


def load_sequence(seq_json: str, seq_id: str) -> list[tuple[Path, float]]:
    """(path, ts) pairs from a data/v3/sequences/*.json entry, in ts order."""
    seqs = json.load(open(seq_json))
    hit = [s for s in seqs if s["seq_id"] == seq_id]
    if not hit:
        raise SystemExit(f"seq_id {seq_id!r} not found in {seq_json}")
    frames = sorted(((Path(f["image_path"]), float(f["ts"])) for f in hit[0]["frames"]),
                    key=lambda x: x[1])
    return frames


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", help="directory of chronological frames")
    src.add_argument("--sequence-json", help="data/v3/sequences/sequences_*.json")
    ap.add_argument("--seq-id", help="seq_id inside --sequence-json")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--device-id", default="replay-drone")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=1 / 60,
                    help="assumed rate when filenames carry no time offsets")
    ap.add_argument("--epoch-start", type=float, default=1_700_000_000.0)
    ap.add_argument("--realtime", action="store_true",
                    help="sleep between frames to mimic live upload cadence (still a replay)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--run-tag", default=None, help="default: replay-<utc>-<label>")
    args = ap.parse_args()

    if args.sequence_json:
        if not args.seq_id:
            raise SystemExit("--seq-id is required with --sequence-json")
        pairs = load_sequence(args.sequence_json, args.seq_id)
        t0 = pairs[0][1]
        items = [(p, ts - t0, ts) for p, ts in pairs]       # (path, offset, capture_ts)
        label = args.seq_id
    else:
        frames = sorted([p for p in Path(args.frames).iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}], key=frame_key)
        items = []
        for i, fp in enumerate(frames):
            off = frame_offset_s(fp, i, args.fps)
            items.append((fp, off, args.epoch_start + off))
        label = Path(args.frames).name
    if args.limit:
        items = items[: args.limit]
    run_tag = args.run_tag or make_run_tag(label)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"HISTORICAL REPLAY (source={SOURCE}) run_tag={run_tag} "
          f"n_frames={len(items)} server={args.server} device_id={args.device_id}")
    print("  -> stored frames re-posted in time order; NOT a live field run.")

    first_alert = None
    prev_off = None
    with out.open("w") as fh:
        for i, (fp, off, cts) in enumerate(items):
            if args.realtime and prev_off is not None:
                time.sleep(max(0.0, min(off - prev_off, 5.0)))
            prev_off = off
            with fp.open("rb") as img:
                r = requests.post(f"{args.server}/detect",
                                  files={"image": (fp.name, img, "image/jpeg")},
                                  data=build_form(args.device_id, cts, run_tag,
                                                  {"frame": fp.name}),
                                  timeout=120)
            r.raise_for_status()
            rec = r.json()
            rec["frame"] = fp.name
            rec["frame_path"] = str(fp)
            rec["time_offset_s"] = off
            rec["source"] = SOURCE
            rec["run_tag"] = run_tag
            fh.write(json.dumps(rec) + "\n")
            state = rec["alert"]["state"]
            if first_alert is None and state in ("provisional", "confirmed"):
                first_alert = (fp.name, off, state)
            print(f"[{i+1}/{len(items)}] {fp.name} off={off:>7.1f} "
                  f"max_smoke={rec['max_smoke_score']:.3f} alert={state}")

    print("\nSummary (historical replay, not a live run):")
    if first_alert:
        print(f"  first alert: {first_alert[2]} at frame {first_alert[0]} "
              f"(offset {first_alert[1]}s from first frame; FIgLib-named offsets are "
              f"relative to first-visible-smoke)")
    else:
        print("  no alert raised during replay (missed event if smoke present; "
              "silence is not evidence of no smoke)")
    print(f"  run_tag: {run_tag}")
    print(f"  results: {out}")


if __name__ == "__main__":
    main()
