"""Pilot review dashboard (prereg A3): seeded-random + best/median/worst +
rejected examples per route/class, with bg / output / detector-size panes.

Selection is deterministic (seeded), NOT hand-picked; the random section is
listed FIRST per the repair brief. Scores come from the model-assisted review
(stage2 quality, else stage1 scene_realism); decisions from blind_review.py.

Usage:
  .venv/bin/python scripts/v6/repair/build_dashboard.py \
      --review data/v6/repair/review_pilot_b.jsonl \
      --src data/v6/gen/glm_repair_pilot_b \
      --queue data/v6/repair/q_pilot_b.jsonl \
      --out reports/v6/repair/dashboard_b
"""
from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]


def thumb(src: Path, dst: Path, long_side=520):
    im = Image.open(src).convert("RGB")
    s = long_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)))
    im.save(dst, "JPEG", quality=82)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = ROOT / args.src
    out = ROOT / args.out
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    q = {json.loads(l)["hash"]: json.loads(l)
         for l in (ROOT / args.queue).open()}
    recs = [json.loads(l) for l in (ROOT / args.review).open()]
    recs = [r for r in recs if not r.get("error")]

    def score(r):
        s2 = r.get("stage2_crop") or {}
        s1 = r.get("stage1_blind") or {}
        return s2.get("quality") or s1.get("scene_realism") or 0

    groups = {}
    for r in recs:
        groups.setdefault((r.get("route"), r.get("kind")), []).append(r)

    def cell(r):
        h = r["hash"]
        tp = out / "thumbs" / f"{h}.jpg"
        if not tp.exists() and (src / f"{h}.jpg").exists():
            thumb(src / f"{h}.jpg", tp)
        bg_html = ""
        bgp = q.get(h, {}).get("bg_src_path")
        if bgp and Path(bgp).exists():
            bt = out / "thumbs" / f"{h}_bg.jpg"
            if not bt.exists():
                thumb(Path(bgp), bt)
            bg_html = f'<img src="thumbs/{h}_bg.jpg"><div class="c">original bg</div>'
        s1 = r.get("stage1_blind") or {}
        s2 = r.get("stage2_crop") or {}
        cap = (f'{r.get("spec_id")} | {r.get("visibility") or "-"} | '
               f'blind={s1.get("smoke")} alt={s1.get("alt")} '
               f'realism={s1.get("scene_realism")} | '
               f'arts={",".join((s2.get("artifacts") or [])[:3]) or "-"} | '
               f'<b>{r.get("decision_v3") or r.get("decision")}</b>')
        return (f'<div class="cell">{bg_html}<img src="thumbs/{h}.jpg">'
                f'<div class="c">{html.escape(cap, quote=False)}</div></div>')

    sec = []
    for (route, kind), rs in sorted(groups.items(), key=lambda x: str(x[0])):
        rs_s = sorted(rs, key=score)
        rng = random.Random(f"dash:{route}:{kind}")
        rand = rng.sample(rs, min(8, len(rs)))
        best = rs_s[-4:][::-1]
        mid = rs_s[len(rs_s) // 2 - 2: len(rs_s) // 2 + 2]
        worst = rs_s[:4]
        rejected = [r for r in rs if str(r.get("decision_v3") or r.get("decision", "")).startswith("reject")]
        rej = rng.sample(rejected, min(8, len(rejected)))
        from collections import Counter
        dc = Counter(r.get("decision_v3") or r.get("decision") for r in rs)
        sec.append(f"<h2>{route} / {kind} — n={len(rs)}</h2>"
                   f"<p>decisions: {dict(dc)}</p>")
        for title, items in [("seeded-random", rand), ("best", best),
                             ("median", mid), ("worst", worst),
                             (f"rejected (of {len(rejected)})", rej)]:
            sec.append(f"<h3>{title}</h3><div class='row'>"
                       + "".join(cell(r) for r in items) + "</div>")

    (out / "index.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>V6 repair pilot dashboard</title><style>
body{{background:#161616;color:#ddd;font:14px sans-serif;margin:20px}}
.row{{display:flex;flex-wrap:wrap;gap:8px}}
.cell{{width:300px;background:#222;padding:6px;border-radius:6px}}
.cell img{{width:100%;display:block;border-radius:4px}}
.c{{font-size:11px;color:#aaa;padding:3px 1px}}
h2{{border-top:1px solid #444;padding-top:14px}}</style>
<h1>V6 repair pilot — model-assisted review dashboard</h1>
<p>Selection is seeded-random + score-sorted, not hand-picked. Review is
MODEL-ASSISTED (blind stage-1 at detector scale; crop stage-2), never human
ground truth. Source review: {html.escape(args.review)}</p>
{"".join(sec)}""")
    print("wrote", out / "index.html")


if __name__ == "__main__":
    main()
