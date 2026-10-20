"""V5b arm D: build the TEACHER target image lists (TRAINING data only).

Targets = E corpus (data/yolo/lists/local/expB_train.txt: pyro-sdis train sites
+ D-Fire, 38,454) + Boreal TRAIN burns (evo + karkkila incl. the karkkila
Empty-Images non-smoke clip) from data/v4/yolo/E_ET_v1 (2,062).
EXCLUDED on purpose: the 10,904 AuSmoke frames that are also in E_ET_v1's T
list -- PROTOCOL_V5 S2: AuSmoke permission PENDING, no NEW training use.
Never includes any val/test/holdout list (asserted below by path AND basename).

Outputs (data/v5b/teacher/lists/):
  train_E.txt, train_T_boreal.txt, train_all.txt      local paths
  dino_shard_a.txt, dino_shard_b.txt                   local paths, interleaved halves
  rfdetr_all_pod.txt                                   POD paths (rfdetr_infer has a 1-prefix root-map)
  TARGETS.json                                         counts, sha256, exclusions, leak check
"""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path

ROOT = Path("/local/home/tianrgua/cvproject")
OUT = ROOT / "data/v5b/teacher/lists"; OUT.mkdir(parents=True, exist_ok=True)
LOCAL_YOLO = str(ROOT / "data/yolo")
LOCAL_EET = str(ROOT / "data/v4/yolo/E_ET_v1")
POD_YOLO = "/scratch/scratch/tianrgua-sandbox/smoke_data_v2"
POD_EET = "/scratch/scratch/tianrgua-sandbox/smoke/v4data/E_ET_v1"
ROOTMAP = f"{LOCAL_YOLO}={POD_YOLO},{LOCAL_EET}={POD_EET}"

def lines(p): return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def to_pod(p):
    for s, d in (spec.split("=", 1) for spec in ROOTMAP.split(",")):
        if p.startswith(s): return d + p[len(s):]
    raise ValueError(p)

E = lines(ROOT / "data/yolo/lists/local/expB_train.txt")
T_all = lines(ROOT / "data/v4/yolo/E_ET_v1/lists/local/train_T.txt")
T = [p for p in T_all if "/images/boreal__" in p]
T_excl = [p for p in T_all if p not in set(T)]
assert len(E) == 38454 and len(T) == 2062 and len(T_excl) == 10904, (len(E), len(T), len(T_excl))
assert all(p.startswith((LOCAL_YOLO + "/pyro_sdis/images/train/", LOCAL_YOLO + "/pyro_sdis/images/val/",
                         LOCAL_YOLO + "/dfire_smoke/images/train/")) for p in E)
# pyro-sdis rows come from the HF train AND val folders but ONLY the frozen TRAIN sites (site-disjoint split)
TRAIN_SITES = ("brison", "serre-de-barre", "courmettes", "valbonne")
for p in E:
    if "/pyro_sdis/" in p:
        site = Path(p).name.split("_")[1].rsplit("-", 1)[0]   # <partner>_<site>-<azimuth>_<ts>.jpg
        assert site in TRAIN_SITES, (p, site)
assert all(p.startswith(LOCAL_EET + "/images/boreal__") for p in T)
assert all(("evo" in Path(p).name.lower()) or ("karkkila" in Path(p).name.lower()) for p in T), "non evo/karkkila boreal frame"
assert not any(k in p.lower() for p in T for k in ("heinola", "ruokolahti"))
assert all(("ausmoke" in p.lower()) for p in T_excl)
ALL = E + T
assert len(set(ALL)) == len(ALL)

# leak check against every val/test/holdout list we know of (path and basename)
EVAL_LISTS = [ROOT / "data/yolo/lists/local/sdis_val.txt", ROOT / "data/yolo/lists/local/sdis_test.txt",
              ROOT / "data/yolo/lists/local/dfire_test_raw.txt",
              ROOT / "data/v4/eval/boreal_val/lists/local/val.txt", ROOT / "data/v4/eval/boreal_test/lists/local/test.txt",
              ROOT / "data/v4/eval/forest_val/lists/local/val.txt", ROOT / "data/v4/eval/forest_test/lists/local/test.txt",
              ROOT / "data/v4/yolo/B_G_v1/lists/local/gval.txt"]
leak = {}
allset, allbase = set(ALL), {Path(p).name for p in ALL}
for L in EVAL_LISTS:
    ev = lines(L); leak[str(L.relative_to(ROOT))] = {"n": len(ev),
        "path_overlap": sum(p in allset for p in ev), "basename_overlap": sum(Path(p).name in allbase for p in ev)}
bad = {k: v for k, v in leak.items() if v["path_overlap"] or v["basename_overlap"]}
assert not bad, f"LEAK: {bad}"
missing = [p for p in ALL if not os.path.exists(p)]
assert not missing, f"{len(missing)} target images missing locally, e.g. {missing[:2]}"

(OUT / "train_E.txt").write_text("\n".join(E) + "\n")
(OUT / "train_T_boreal.txt").write_text("\n".join(T) + "\n")
(OUT / "train_all.txt").write_text("\n".join(ALL) + "\n")
(OUT / "dino_shard_a.txt").write_text("\n".join(ALL[0::2]) + "\n")
(OUT / "dino_shard_b.txt").write_text("\n".join(ALL[1::2]) + "\n")
(OUT / "rfdetr_all_pod.txt").write_text("\n".join(to_pod(p) for p in ALL) + "\n")
(OUT / "excluded_ausmoke_T.txt").write_text("\n".join(T_excl) + "\n")
man = {
  "purpose": "V5b arm D teacher-pool TARGETS -- TRAINING images only; model-assisted labels will be produced on these",
  "counts": {"E_pyro_sdis_train": sum("/pyro_sdis/" in p for p in E), "E_dfire_train": sum("/dfire_smoke/" in p for p in E),
             "T_boreal_karkkila": sum("Karkkila-Images" in p for p in T), "T_boreal_evo": sum("Evo-Images" in p for p in T),
             "T_boreal_karkkila_empty_clip": sum("Empty-Images" in p for p in T), "total": len(ALL),
             "excluded_ausmoke_in_E_ET_v1_T": len(T_excl)},
  "exclusion_reason": "AuSmoke (via MultiNatSmoke) permission PENDING per reports/v5/PROTOCOL_V5.md S2 -- no new training use; "
                      "scoring it for arm D would be a new training use. Re-run with excluded_ausmoke_T.txt if permission lands.",
  "sources": {"E": "data/yolo/lists/local/expB_train.txt", "T": "data/v4/yolo/E_ET_v1/lists/local/train_T.txt (boreal__ rows only)"},
  "rootmap_local_to_pod": ROOTMAP,
  "leak_check_vs_eval_lists": leak,
  "files": {f.name: {"n": len(lines(f)), "sha256": sha(f)} for f in sorted(OUT.glob("*.txt"))},
}
(OUT / "TARGETS.json").write_text(json.dumps(man, indent=1))
print(json.dumps(man["counts"]), "\nleak check OK (0 overlaps) ->", OUT)
