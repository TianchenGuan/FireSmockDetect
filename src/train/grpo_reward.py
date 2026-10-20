"""Verifiable reward for GRPO on the smoke-detection JSON task (iteration 2).

Parses the model's JSON answer with the SAME parser used at eval time
(src/eval/eval_vlm.py::parse_answer) so training reward and eval metrics
agree on what counts as a valid prediction.

Reward per completion (constants below, then clipped to [-2, 2]):
  +1.0  correct smoke_present   (pred_present == gt_present)
  -1.0  wrong smoke_present     (also applied when output is unparseable)
  +0.5  per GT box matched by a predicted box (greedy 1-1, IoU >= 0.4)
  -0.5  per predicted box left unmatched (hallucinated / badly placed box)

Notes:
- pred_present uses the eval convention: smoke_present flag OR any box.
- Boxes are [x1,y1,x2,y2] normalized to 0-1 (parse_answer divides the
  model's 0-1000 integers by 1000; gt_boxes in the probe jsonl are 0-1).
- No length/verbosity term: a verbose answer containing the same correct
  JSON scores exactly the same as the compact answer (never more).

trl entry point: `reward_fn(prompts, completions, gt_boxes=..., **kw)`
returns list[float]; extra dataset columns arrive as per-row lists.
If $GRPO_REWARD_LOG is set, one JSON line per completion is appended there
(step, reward, pred_present, n_pred_boxes, parse_ok) for post-hoc analysis.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# parse_answer/iou live in src/eval/eval_vlm.py; on pods the file is fetched
# side-by-side into the same dir as this file.
_CAND = [Path(__file__).resolve().parents[1] / "eval",  # devbox repo layout
         Path(__file__).resolve().parent]               # pod: flat /tmp/job
for _d in _CAND:
    if (_d / "eval_vlm.py").exists() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
from eval_vlm import iou, parse_answer  # noqa: E402

IOU_THR = 0.4
R_PRESENT_CORRECT = 1.0
R_PRESENT_WRONG = -1.0        # also the score for unparseable output
BOX_MATCH_BONUS = 0.5         # per GT box matched
UNMATCHED_PRED_PENALTY = 0.5  # per predicted box with no GT match
CLIP_LO, CLIP_HI = -2.0, 2.0


def smoke_reward(text: str, gt_boxes: list) -> dict:
    """Score one completion string against GT boxes ([x1,y1,x2,y2] in 0-1).

    Returns {"reward": float, "pred_present": bool|None,
             "n_pred_boxes": int, "parse_ok": bool}.
    """
    parsed = parse_answer(text)
    if parsed is None:
        return {"reward": R_PRESENT_WRONG, "pred_present": None,
                "n_pred_boxes": 0, "parse_ok": False}
    gt_present = len(gt_boxes) > 0
    pred_boxes = parsed["boxes"]
    pred_present = bool(parsed["smoke_present"]) or bool(pred_boxes)
    r = R_PRESENT_CORRECT if pred_present == gt_present else R_PRESENT_WRONG
    # greedy 1-1 matching, same protocol as eval_vlm.py
    matched = set()
    n_matched = 0
    for pb in pred_boxes:
        best_j, best = -1, 0.0
        for j, g in enumerate(gt_boxes):
            if j in matched:
                continue
            v = iou(pb, g[:4])
            if v > best:
                best_j, best = j, v
        if best >= IOU_THR:
            matched.add(best_j)
            n_matched += 1
    n_unmatched_pred = len(pred_boxes) - n_matched
    r += BOX_MATCH_BONUS * n_matched
    r -= UNMATCHED_PRED_PENALTY * n_unmatched_pred
    return {"reward": max(CLIP_LO, min(CLIP_HI, r)),
            "pred_present": pred_present,
            "n_pred_boxes": len(pred_boxes), "parse_ok": True}


def _completion_text(completion) -> str:
    """Extract text from a trl completion (str, or conversational
    [{"role": "assistant", "content": str-or-blocks}])."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        msg = completion[-1]
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # structured blocks
            return "".join(p.get("text", "") for p in content
                           if isinstance(p, dict))
    return ""


def reward_fn(prompts=None, completions=None, completion_ids=None,
              gt_boxes=None, **kwargs) -> list:
    """trl GRPOTrainer reward function. `gt_boxes` is the per-row dataset
    column (list of GT box lists, one per completion in the batch)."""
    if gt_boxes is None:
        raise ValueError("dataset must provide a 'gt_boxes' column")
    state = kwargs.get("trainer_state")
    step = getattr(state, "global_step", -1) if state is not None else -1
    log_path = os.environ.get("GRPO_REWARD_LOG")
    rewards = []
    logs = []
    for i, comp in enumerate(completions):
        text = _completion_text(comp)
        res = smoke_reward(text, gt_boxes[i] or [])
        rewards.append(res["reward"])
        if log_path:
            logs.append(json.dumps({
                "step": step, "i": i, "reward": res["reward"],
                "gt_present": bool(gt_boxes[i]),
                "pred_present": res["pred_present"],
                "n_pred_boxes": res["n_pred_boxes"],
                "parse_ok": res["parse_ok"],
                "text_head": text[:200]}))
    if log_path and logs:
        with open(log_path, "a") as fh:
            fh.write("\n".join(logs) + "\n")
    return rewards


reward_fn.__name__ = "smoke_json_reward"  # name shown in trl metrics
