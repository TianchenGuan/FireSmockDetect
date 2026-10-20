"""Unit tests for src/train/grpo_reward.py (standalone, no trl needed).

Run: .venv/bin/python src/train/test_grpo_reward.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grpo_reward import (CLIP_HI, CLIP_LO, reward_fn,  # noqa: E402
                         smoke_reward)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# fixtures: GT in normalized 0-1 coords
GT_POS_SMALL = [[0.50, 0.50, 0.60, 0.60]]            # one small smoke box
GT_POS_MULTI = [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4],
                [0.5, 0.5, 0.6, 0.6], [0.7, 0.7, 0.8, 0.8]]
GT_NEG = []

ALWAYS_NEG = '{"smoke_present": false, "boxes": []}'
ALWAYS_POS_BIGBOX = '{"smoke_present": true, "boxes": [[0,0,1000,1000]]}'
PERFECT_POS = '{"smoke_present": true, "boxes": [[500,500,600,600]]}'
PERFECT_NEG = ALWAYS_NEG
VERBOSE_POS = ('Let me analyse the scene carefully. There appears to be a '
               'faint plume rising near the ridge line, consistent with '
               'early wildfire smoke. Therefore my answer is '
               '{"smoke_present": true, "boxes": [[500,500,600,600]]} '
               'based on the visual evidence described above.')

# 1. always-negative output
r = smoke_reward(ALWAYS_NEG, GT_POS_SMALL)
check("always-neg on positive image = -1 (wrong present)",
      r["reward"] == -1.0, f"got {r['reward']}")
r = smoke_reward(ALWAYS_NEG, GT_NEG)
check("always-neg on negative image = +1 (correct present)",
      r["reward"] == 1.0, f"got {r['reward']}")

# 2. always-positive one-big-box output
r = smoke_reward(ALWAYS_POS_BIGBOX, GT_NEG)
check("always-pos-bigbox on negative image = -1.5 (wrong + unmatched box)",
      r["reward"] == -1.5, f"got {r['reward']}")
r_big = smoke_reward(ALWAYS_POS_BIGBOX, GT_POS_SMALL)
check("always-pos-bigbox on positive image = +0.5 (present ok, box IoU<0.4)",
      r_big["reward"] == 0.5, f"got {r_big['reward']}")

# 3. perfect output
r_perf = smoke_reward(PERFECT_POS, GT_POS_SMALL)
check("perfect on positive = +1.5 (present + matched box)",
      r_perf["reward"] == 1.5, f"got {r_perf['reward']}")
check("perfect strictly beats big-box on positive",
      r_perf["reward"] > r_big["reward"])
r = smoke_reward(PERFECT_NEG, GT_NEG)
check("perfect on negative = +1.0", r["reward"] == 1.0, f"got {r['reward']}")

# 4. verbose explanation must NOT beat a correct compact answer
r_verb = smoke_reward(VERBOSE_POS, GT_POS_SMALL)
check("verbose correct <= compact correct",
      r_verb["reward"] <= r_perf["reward"],
      f"verbose {r_verb['reward']} vs compact {r_perf['reward']}")
check("verbose correct still parsed and rewarded",
      r_verb["parse_ok"] and r_verb["reward"] == r_perf["reward"],
      f"got {r_verb}")

# 5. clipping
many_bad = json.dumps({"smoke_present": True,
                       "boxes": [[0, 0, 100, 100]] * 8})  # 8 unmatched boxes
r = smoke_reward(many_bad, GT_NEG)
check("floor clip at -2", r["reward"] == CLIP_LO, f"got {r['reward']}")
all_match = json.dumps({"smoke_present": True, "boxes": [
    [100, 100, 200, 200], [300, 300, 400, 400],
    [500, 500, 600, 600], [700, 700, 800, 800]]})
r = smoke_reward(all_match, GT_POS_MULTI)  # 1 + 4*0.5 = 3 -> clip 2
check("ceiling clip at +2", r["reward"] == CLIP_HI, f"got {r['reward']}")

# 6. unparseable / degenerate output
r = smoke_reward("I think maybe possibly cannot decide", GT_POS_SMALL)
check("unparseable = -1", r["reward"] == -1.0 and not r["parse_ok"],
      f"got {r}")
r = smoke_reward("", GT_NEG)
check("empty = -1", r["reward"] == -1.0, f"got {r['reward']}")

# 7. IoU boundary: exactly 0.4 must count as matched
#    pred [0,0,400,1000] vs gt [0,0,1.0,1.0] -> inter 0.4, union 1.0 -> 0.4
r = smoke_reward('{"smoke_present": true, "boxes": [[0,0,400,1000]]}',
                 [[0.0, 0.0, 1.0, 1.0]])
check("IoU==0.4 counts as match (+1.5)", r["reward"] == 1.5,
      f"got {r['reward']}")

# 8. box flag inconsistency: boxes present but flag false on negative image
r = smoke_reward('{"smoke_present": false, "boxes": [[0,0,1000,1000]]}',
                 GT_NEG)
check("boxes imply present (eval convention): wrong on neg = -1.5",
      r["reward"] == -1.5, f"got {r['reward']}")

# 9. trl-style batched call, conversational completions + reward log
comps = [
    [{"role": "assistant", "content": PERFECT_POS}],
    [{"role": "assistant", "content": ALWAYS_NEG}],
    [{"role": "assistant", "content": [{"type": "text",
                                        "text": VERBOSE_POS}]}],
    [{"role": "assistant", "content": ALWAYS_POS_BIGBOX}],
]
gts = [GT_POS_SMALL, GT_POS_SMALL, GT_POS_SMALL, GT_NEG]
with tempfile.TemporaryDirectory() as td:
    os.environ["GRPO_REWARD_LOG"] = str(Path(td) / "rl.jsonl")
    out = reward_fn(prompts=[None] * 4, completions=comps,
                    completion_ids=[None] * 4, gt_boxes=gts)
    check("batched rewards", out == [1.5, -1.0, 1.5, -1.5], f"got {out}")
    lines = [json.loads(l) for l in
             open(os.environ["GRPO_REWARD_LOG"]).read().splitlines()]
    check("reward log has 4 rows with fields",
          len(lines) == 4 and all("pred_present" in l and "reward" in l
                                  for l in lines))
    del os.environ["GRPO_REWARD_LOG"]

# 10. group non-degeneracy sanity: the 4 canonical policies produce a
#     non-constant reward vector on a positive image (GRPO advantage != 0)
rs = [smoke_reward(t, GT_POS_SMALL)["reward"]
      for t in (PERFECT_POS, ALWAYS_NEG, ALWAYS_POS_BIGBOX, "garbage")]
check("reward separates behaviours on positives", len(set(rs)) >= 3,
      f"got {rs}")

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
    sys.exit(1)
print("ALL TESTS PASSED")
