"""Peek at a running V5b seg job's self-logged artifacts on S3 (pods have no logs).
  python scripts/v5b/seg_job_status.py --run segV5b_a --seg 1 [--console lora]"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster/v3"))
from orchestrate_dino import ada_session, BUCKET  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--seg", type=int, default=1)
ap.add_argument("--console", default=None); ap.add_argument("--log", action="store_true")
ap.add_argument("--tail", type=int, default=3000)
a = ap.parse_args()
sess, _ = ada_session(); s3 = sess.client("s3", region_name="us-east-1")
op = f"v5b/runs/{a.run}/seg{a.seg:02d}"
objs = s3.list_objects_v2(Bucket=BUCKET, Prefix=op).get("Contents", [])
for o in objs:
    print(f"{o['Size']:>10} {o['LastModified'].strftime('%H:%M:%S')} {o['Key'][len(op)+1:]}")
def get(k):
    try:
        return s3.get_object(Bucket=BUCKET, Key=f"{op}/{k}")["Body"].read().decode(errors="replace")
    except Exception as e:
        return f"<{type(e).__name__}>"
print("STATUS:", get("status.json")[:1500])
if a.console:
    print(f"--- console {a.console} ---"); print(get(f"{a.console}/console_tail.txt")[-a.tail:])
if a.log:
    print("--- entry log ---"); print(get("entry_log.txt")[-a.tail:])
