"""Aggregate blind-review results into quality counts with denominators.

Usage:
  .venv/bin/python scripts/v6/repair/review_stats.py \
      data/v6/repair/review_pilot_b.jsonl data/v6/repair/review_pilot_c.jsonl
Prints a markdown comparison (and JSON) per variant: blind-visibility rates
by intended tier, negative/control contamination+hallucination, decision mix,
artifact frequencies, i2i restyle rates.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def load(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def pct(n, d):
    return f"{100*n/d:.0f}% ({n}/{d})" if d else "—"


def stats(recs, name):
    recs = [r for r in recs if not r.get("error")]
    out = {"variant": name, "n": len(recs)}
    print(f"\n## {name} (n={len(recs)} reviewed)")

    print("\n| group | blind smoke=yes | uncertain | no |")
    print("|---|---|---|---|")
    rows = []
    for route in ("t2i", "i2i"):
        for kind in ("smoke", "nonsmoke", "ctrl"):
            g = [r for r in recs if r.get("route") == route and r.get("kind") == kind]
            if not g:
                continue
            c = Counter((r.get("stage1_blind") or {}).get("smoke") for r in g)
            rows.append((f"{route}/{kind}", c))
            print(f"| {route}/{kind} (n={len(g)}) | {pct(c.get('yes',0),len(g))} "
                  f"| {pct(c.get('uncertain',0),len(g))} | {pct(c.get('no',0),len(g))} |")
    pos = [r for r in recs if r.get("kind") == "smoke"]
    print("\n| intended visibility | blind yes | uncertain | no |")
    print("|---|---|---|---|")
    for vis in ("clearly_visible", "small_discernible", "faint_discernible"):
        g = [r for r in pos if r.get("visibility") == vis]
        c = Counter((r.get("stage1_blind") or {}).get("smoke") for r in g)
        print(f"| {vis} (n={len(g)}) | {pct(c.get('yes',0),len(g))} | "
              f"{pct(c.get('uncertain',0),len(g))} | {pct(c.get('no',0),len(g))} |")

    print("\n| group | decision mix |")
    print("|---|---|")
    for route in ("t2i", "i2i"):
        for kind in ("smoke", "nonsmoke", "ctrl"):
            g = [r for r in recs if r.get("route") == route and r.get("kind") == kind]
            if not g:
                continue
            c = Counter(r.get("decision") for r in g)
            print(f"| {route}/{kind} | " + ", ".join(
                f"{k} {pct(v,len(g))}" for k, v in c.most_common()) + " |")

    arts = Counter(a for r in recs for a in
                   ((r.get("stage2_crop") or {}).get("artifacts") or []))
    print("\nstage-2 artifact counts:", dict(arts.most_common(8)))
    s1r = [(r.get("stage1_blind") or {}).get("scene_realism") for r in recs]
    s1r = [x for x in s1r if isinstance(x, (int, float))]
    if s1r:
        print(f"scene_realism mean {sum(s1r)/len(s1r):.2f} (n={len(s1r)})")
    i2 = [r for r in recs if r.get("i2i_diff")]
    if i2:
        gr = [r for r in i2 if (r["i2i_diff"] or {}).get("global_restyle")]
        print(f"i2i global_restyle: {pct(len(gr), len(i2))}")
    acc_pos = [r for r in pos if r.get("decision") == "accept_pos"]
    print(f"\nACCEPTED positives: {pct(len(acc_pos), len(pos))}; "
          f"accepted negatives/controls: "
          f"{pct(sum(1 for r in recs if str(r.get('decision','')).startswith('accept') and r.get('kind') != 'smoke'), sum(1 for r in recs if r.get('kind') != 'smoke'))}")
    return out


def main():
    for p in sys.argv[1:]:
        stats(load(p), Path(p).stem)


if __name__ == "__main__":
    main()
