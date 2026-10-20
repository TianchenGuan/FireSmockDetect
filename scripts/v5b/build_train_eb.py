"""V5b: E + Boreal-train COCO json for the B200 high-res DINOv3 run.

Takes the frozen v4 E+T json (data/v4/coco/instances_train_eet.json, 51,420
images: E = pyro_sdis train + dfire, T = Boreal train + AuSmoke) and DROPS every
AuSmoke entry (path contains '/multinatsmoke__' ... 'AuSmoke'), keeping
E + Boreal train. Paths are rewritten from the green-cluster /scratch mirror
to the baked locations inside the ap-south-1 image smoke-dino3-b200:v1:
  /scratch/scratch/tianrgua-sandbox/smoke_data_v2/<x>          -> /opt/smoke_data/<x>
  /scratch/scratch/tianrgua-sandbox/smoke/v4data/E_ET_v1/images/<x> -> /opt/smoke_data/v4data/E_ET_v1/images/<x>
Annotations are copied verbatim (ids unchanged). Writes
data/v5b/coco/instances_train_eb.json + instances_train_eb.stats.json.
"""
import json, collections, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/v4/coco/instances_train_eet.json"
OUT = ROOT / "data/v5b/coco/instances_train_eb.json"
MAP = [("/scratch/scratch/tianrgua-sandbox/smoke_data_v2/", "/opt/smoke_data/"),
       ("/scratch/scratch/tianrgua-sandbox/smoke/v4data/E_ET_v1/images/",
        "/opt/smoke_data/v4data/E_ET_v1/images/")]


def is_ausmoke(p: str) -> bool:
    return "multinatsmoke__" in p or "AuSmoke" in p or "ausmoke" in p.lower()


def main():
    d = json.loads(SRC.read_text())
    keep, dropped, src_counts = [], 0, collections.Counter()
    for im in d["images"]:
        p = im["file_name"]
        if is_ausmoke(p):
            dropped += 1
            continue
        new = None
        for s, t in MAP:
            if p.startswith(s):
                new = t + p[len(s):]
                break
        if new is None:
            sys.exit(f"unmapped path: {p}")
        src = ("boreal" if "/v4data/" in new else
               "pyro_sdis" if "/pyro_sdis/" in new else
               "dfire" if "/dfire_smoke/" in new else "other")
        src_counts[src] += 1
        im = dict(im, file_name=new)
        keep.append(im)
    ids = {im["id"] for im in keep}
    anns = [a for a in d["annotations"] if a["image_id"] in ids]
    out = {k: v for k, v in d.items() if k not in ("images", "annotations")}
    out["images"], out["annotations"] = keep, anns
    out.setdefault("info", {})
    out["info"] = dict(out["info"], description="V5b E + Boreal-train (AuSmoke removed) — pod paths under /opt (smoke-dino3-b200:v1)",
                       source=str(SRC.relative_to(ROOT)))
    OUT.write_text(json.dumps(out))
    pos = sum(1 for im in keep if im["id"] in {a["image_id"] for a in anns})
    stats = {"source": str(SRC.relative_to(ROOT)), "source_images": len(d["images"]),
             "source_annotations": len(d["annotations"]), "dropped_ausmoke_images": dropped,
             "images": len(keep), "annotations": len(anns), "positive_images": pos,
             "negative_images": len(keep) - pos, "by_source": dict(src_counts)}
    (OUT.with_suffix(".stats.json")).write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
