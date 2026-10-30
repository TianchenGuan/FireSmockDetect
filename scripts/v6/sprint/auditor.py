"""Training-data quality auditor (sprint protocol, stage A blind).

Implements the user-specified auditor prompt. Stage A sees ONLY the
detector-input-sized frame (no boxes, no provenance). training_utility is
always "unmeasured". Outputs are MODEL-ASSISTED, never human ground truth;
'uncertain' is never auto-converted to negative.

Usage:
  .venv/bin/python scripts/v6/sprint/auditor.py --manifest <jsonl> \
      --account 000000000002 --region us-east-1 --out <out.jsonl>
Manifest lines: {"id":..., "image": abspath, ...extra fields passed through}
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/v6"))
from bedrock_gen import Creds, ledger  # noqa: E402

MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
PRICE_IN, PRICE_OUT = 1.0, 5.0
DET_LONG = 1280

SYS_A = """你是烟雾检测训练数据的质量审核员。你的职责是检查可见证据、样本适用性\
和标签一致性。你不能仅凭一张图片判断它会不会提升模型。

任务边界：
- 沿用项目既定的烟雾类别定义，不擅自把"烟雾检测"改成"必须证明是野火"。
- 只依据提供的图像证据。生成意图、文件名、已有标签和其他模型分数都不是事实依据。
- 不要求看到火焰或烟源；允许遮挡、小烟和薄烟。
- 烟与云、雾、蒸汽或灰尘无法可靠区分时，必须允许 uncertain。
- 不以美观、清晰度或电影感作为训练质量标准。
- no_visible_smoke 只表示本次观察未见烟，不自动授权将图片作为负样本。
- 画面过暗、过小、遮挡严重或存在模糊烟状区域时，优先输出 uncertain。

只返回一个 JSON 对象，字段：
 semantic_label: "smoke" | "no_visible_smoke" | "uncertain"
 visibility: "clear" | "small_identifiable" | "faint_identifiable" | "unjudgeable"
 approximate_location: 简短方位描述（如"左侧中景山脊上方"），无则 ""；不要编造精确框
 visual_evidence: 一至两句具体可见证据
 alternative_explanations: 可能的云、雾、蒸汽、灰尘、光照或图像伪影（数组，可为空）
 scene_issues: 明显不合理的物体、烟体形态、融合或照明问题（数组，可为空）
 training_utility: 必须为 "unmeasured" """


def to_jpeg(img: Image.Image, long_side=DET_LONG) -> bytes:
    s = long_side / max(img.size)
    if s < 1:
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    return buf.getvalue()


def ask(cl, jpeg: bytes):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 500,
        "system": SYS_A,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/jpeg",
             "data": base64.b64encode(jpeg).decode()}},
            {"type": "text", "text": "请按规定返回 JSON。"}]}]})
    r = cl.invoke_model(modelId=MODEL, body=body)
    d = json.loads(r["body"].read())
    txt = "".join(b.get("text", "") for b in d["content"]
                  if b.get("type") == "text")
    u = d.get("usage", {})
    try:
        return json.loads(txt[txt.index("{"):txt.rindex("}") + 1]), u
    except Exception:
        return {"parse_error": txt[:200]}, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--account", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="auditA")
    args = ap.parse_args()
    items = [json.loads(l) for l in Path(args.manifest).open() if l.strip()]
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if outp.exists():
        for l in outp.open():
            try:
                seen.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [it for it in items if it["id"] not in seen]
    print(f"{len(todo)} to audit")
    if not todo:
        return 0
    est = round(len(todo) * (2600 * PRICE_IN + 350 * PRICE_OUT) / 1e6 * 1.2, 2)
    rid = f"{args.tag}-{int(time.time())}"
    if ledger(["reserve", "--id", rid, "--tag", "review", "--est-usd", str(est),
               "--model", MODEL, "--account", args.account,
               "--region", args.region, "--n", str(len(todo)),
               "--note", f"sprint auditor stage-A {args.tag}"]):
        print("LEDGER DENIED"); return 3
    creds = Creds(args.account)
    import boto3
    tin = tout = n = 0
    with outp.open("a") as f:
        for i, it in enumerate(todo):
            c = creds.get()
            cl = boto3.Session(
                aws_access_key_id=c["AccessKeyId"],
                aws_secret_access_key=c["SecretAccessKey"],
                aws_session_token=c["SessionToken"]).client(
                    "bedrock-runtime", region_name=args.region)
            rec = dict(it)
            try:
                j, u = ask(cl, to_jpeg(Image.open(it["image"])))
                tin += u.get("input_tokens", 0); tout += u.get("output_tokens", 0)
                rec["stageA"] = j
                rec["review_kind"] = "model_assisted"
                n += 1
            except Exception as e:
                rec["error"] = str(e)[:300]
                if "Throttl" in str(e):
                    time.sleep(10)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(0.4)
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    actual = round((tin * PRICE_IN + tout * PRICE_OUT) / 1e6, 2)
    ledger(["settle", "--id", rid, "--actual-usd", str(actual),
            "--note", f"audited={n}"])
    print(f"DONE {n} actual=${actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
