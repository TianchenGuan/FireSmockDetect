"""Decision rule v3 (recorded calibration, applied before corpus build):
global_restyle no longer rejects POSITIVES (a re-rendered scene with valid
visible smoke is a valid training image; the flag is kept as i2i_rerender and
the pos/ctrl 'pair' claim is downgraded to style-matched same-background).
Uniform gates for all routes: positives need blind yes + morphology pass +
no studio artifacts + scene_realism>=3; controls/negatives need blind no +
scene_realism>=3. Uncertain stays its own bucket (never hard pos, never neg).
"""
import json, sys
from collections import Counter

BAD = {"incense_stem","source_prop","pedestal_or_floor","colored_stage_light",
       "neon_tint","explosion_like","text_or_watermark"}

def decide3(r):
    s1 = r.get("stage1_blind") or {}
    s2 = r.get("stage2_crop") or {}
    diff = r.get("i2i_diff") or {}
    smoke, realism = s1.get("smoke"), s1.get("scene_realism") or 0
    if r.get("kind") == "smoke":
        if smoke == "no":
            return "reject_invisible"
        if smoke == "uncertain":
            return "uncertain"
        if set(s2.get("artifacts") or []) & BAD:
            return "reject_artifact"
        if s2.get("plume_shape_ok") is False:
            return "reject_morphology"
        if realism < 3:
            return "reject_low_realism"
        return "accept_pos"
    # nonsmoke / ctrl
    if smoke != "no":
        return "reject_smoke_in_neg"
    if realism < 3:
        return "reject_low_realism"
    return "accept_neg" if r.get("kind") == "nonsmoke" else "accept_ctrl"

for var in sys.argv[1:]:
    out = []
    for l in open(f"data/v6/repair/review_pilot_{var}_v2.jsonl"):
        r = json.loads(l)
        if not r.get("error"):
            r["decision_v3"] = decide3(r)
            r["i2i_rerender"] = bool((r.get("i2i_diff") or {}).get("global_restyle"))
        out.append(r)
    with open(f"data/v6/repair/review_pilot_{var}_v3.jsonl","w") as f:
        for r in out:
            f.write(json.dumps(r)+"\n")
    recs = [r for r in out if not r.get("error")]
    print(f"\n== {var} decision_v3 ==")
    for route in ("t2i","i2i"):
        for kind in ("smoke","nonsmoke","ctrl"):
            g=[r for r in recs if r.get("route")==route and r.get("kind")==kind]
            if not g: continue
            c=Counter(r["decision_v3"] for r in g)
            acc=sum(v for k,v in c.items() if k.startswith("accept"))
            print(f" {route}/{kind} n={len(g)} accept={acc} ({100*acc/len(g):.0f}%) | {dict(c.most_common())}")
