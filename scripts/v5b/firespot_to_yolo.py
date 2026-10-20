"""GUARDED converter: FireSpot-CNX-r1 labels -> data/v5b/yolo/firespot/{images,labels}/ (class 0 = smoke;
explicit negatives get empty label files; images symlinked). FireSpot-CNX-r1 ships NO licence text
(repo has no LICENSE, archive has none; the 'CC BY 4.0' in MultiNatSmoke's README is an aggregator claim),
so permission_status = pending and training use is NOT allowed (PROTOCOL_V5 s2). This script refuses to
run unless --permission-confirmed <evidence-string> is given (e.g. 'author email 2026-10-xx, see CARD').
Boxes are already YOLO (xc yc w h normalised); 14 boxes slightly exceed [0,1] and are clipped."""
import argparse, os, sys, pandas as pd
ROOT = "/local/home/tianrgua/cvproject"
ap = argparse.ArgumentParser(); ap.add_argument("--permission-confirmed", default=None); ap.add_argument("--split", default="train,val", help="proposed_split values to export (holdout never exported by default)")
a = ap.parse_args()
if not a.permission_confirmed:
    sys.exit("REFUSED: FireSpot permission_status=pending (no licence text in repo/archive). Pass --permission-confirmed '<evidence>' once written permission/licence is on file and recorded in the CARD.")
man = pd.read_parquet(f"{ROOT}/data/v5b/manifests/firespot_manifest.parquet")
sel = man[man.proposed_split.isin(a.split.split(",")) & (man.dup_class != "internal_exact_dup") & (man.permission_status == "confirmed")]
if sel.empty: sys.exit("REFUSED: manifest permission_status is not 'confirmed'; update the manifest first.")
out = f"{ROOT}/data/v5b/yolo/firespot"; os.makedirs(f"{out}/images", exist_ok=True); os.makedirs(f"{out}/labels", exist_ok=True)
for r in sel.itertuples():
    src = f"{ROOT}/data/v5b/raw/firespot/{r.rel_path}"; stem = os.path.basename(r.rel_path)[:-4]
    dst = f"{out}/images/{stem}.jpg"
    if not os.path.lexists(dst): os.symlink(src, dst)
    lines = []
    for ln in open(f"{ROOT}/data/v5b/raw/firespot/dataset/labels/{stem}.txt"):
        p = ln.split()
        if len(p) == 5:
            xc, yc, w, h = map(float, p[1:]); x1, y1, x2, y2 = max(0, xc - w / 2), max(0, yc - h / 2), min(1, xc + w / 2), min(1, yc + h / 2)
            lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    open(f"{out}/labels/{stem}.txt", "w").write("\n".join(lines) + ("\n" if lines else ""))
print("exported", len(sel), "images to", out, "evidence:", a.permission_confirmed)
