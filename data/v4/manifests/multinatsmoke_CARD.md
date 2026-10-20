# multinatsmoke (AusSmoke + MultiNatSmoke) — dataset card (iteration 4)

Acquired 2026-09-14 by data-acquisition agent. Raw data: `/local/home/tianrgua/cvproject/data/v4/raw/multinatsmoke/`
Manifest: `/local/home/tianrgua/cvproject/data/v4/manifests/multinatsmoke_manifest.parquet` (70,816 rows, one per image; 389 rows are mask-only, `image_present=False`).

## Provenance / download

- Paper: "AusSmoke meets MultiNatSmoke: a fully-labelled diverse smoke segmentation dataset", Li, Zhao, Zhu, Ji, Wilson, Yebra, Barnes (Bushfire Research Centre of Excellence, Australian National University), WACV 2026, pp. 7996-8006. PDF: https://openaccess.thecvf.com/content/WACV2026/papers/Li_AusSmoke_meets_MultiNatSmoke_a_fully-labelled_diverse_smoke_segmentation_dataset_WACV_2026_paper.pdf
- Code/docs: https://github.com/henryzhao0615/MultiNatSmoke
- Data: https://huggingface.co/datasets/hongjinzhao0615/MultiNatSmoke — single file `MultiNatSmokeDataset.zip`, 43,677,465,490 bytes, HF revision `5b7a4e0f7a094d7e27381a5154ad71be99f58137` (lastModified 2026-05-29). Downloaded 2026-09-14 via hf_hub (xet), exact size match; extracted to `raw/multinatsmoke/MultiNatSmokeDataset/` (42 GB, 163,324 files).
- The `Forest_Fire` subset ships masks only; its images require a separate Kaggle download (`kutaykutlu/forest-fire`) via the repo's `code/lib/curate_forest_fire.py`. NOT downloaded (subset quarantined anyway, see below).

## License / terms

- **No license anywhere for the aggregate or for AusSmoke itself**: GitHub license API returns 404 (no LICENSE file in repo tree); HF dataset has no license tag in its metadata. The README only requests citation of the WACV paper and of the constituent datasets. => AusSmoke = "unspecified / all-rights-reserved by default, publicly released for research benchmark use". Flagged as the main license blocker for the prize subset.
- Per-source licenses as stated in the MultiNatSmoke README table (verbatim column): FIgLib "–", Smoke5K "–", SmokeSeg "–", AI-for-Mankind "CC BY-NC-SA 4.0", Firecam "CC BY-NC-SA 4.0", Boreal Forest Fire "CC BY 4.0", D-Fire "CC0 1.0 (Public Domain)", WSDataset "MIT", FireSpot "CC BY 4.0", FESB-MLID "–", Forest Fire "–".
- FIgLib/HPWREN terms (recorded 2026-09-10 in figlib_subset_manifest): publicly available; use requires a credit reference to https://hpwren.ucsd.edu/ in derivative work; data provided as-is.
- Note: AI-for-Mankind and Firecam are **non-commercial** (CC BY-NC-SA) — both are quarantined here anyway. Internal research prototype use of the usable subsets (WSDataset MIT, FireSpot CC BY 4.0) is fine; AuSmoke and FESB-MLID have no explicit grant.

## Composition and origin classification (CONTAMINATION-CRITICAL)

Structure: `MultiNatSmokeDataset/{Train,Test,TestSmall,TestMedium,TestLarge}/<subset>/{images,masks,info.txt}`.
TestSmall/TestMedium/TestLarge are **byte-identical subsets of Test** (verified by sha256: 5,716+1,900+3,351 images, all contained in Test) — a smoke-size partition for evaluation. They carry **no manifest rows**; only Train+Test are canonical.

Per-subset classification (`origin_class` column; verified by me from filenames, info.txt, the paper, and sha256 cross-checks — not taken on trust):

| subset | imgs (Train+Test) | derivable units | origin_class | evidence |
|---|---|---|---|---|
| AuSmoke | 15,248 | **29 videos** | usable | authors' own ACT-Australia collection (paper sec 3.1) |
| WSDataset | 11,539 | 8 groups (~4 cameras) | usable | Vietnam (Lam Dong) VMOKED; rtsp/camera filenames; greenhouse-terrain visually confirmed; 0 sha overlap with anything else |
| FireSpot | 2,865 | 29 site groups | usable | Thailand (Chiang Mai) towers, CC BY 4.0 |
| FESB-MLID | 251 | 67 | usable (205) / dup_dfire (46) | Croatia FESB; **46 images byte-identical to local D-Fire raw** |
| FIgLib | 12,435 | 391 events | quarantine_figlib | filenames = `<event>__<epoch>_<offset>`; all 391 events present in our figlib_events_index.csv (525 events); 1,945 frames byte-identical to local FIgLib holdings |
| SmokeSeg | 6,144 | 185 events | quarantine_figlib | same FIgLib event__epoch filenames (FoSp/SmokeSeg is FIgLib-sourced; paper confirms) |
| aiformankind | 2,191 | n/a (cuid names) | quarantine_figlib | info.txt: "selected from FIgLib"; sample OSD text shows hpwren.ucsd.edu |
| firecam | 1,661 | 72 camera-days | quarantine_figlib | filenames `<hpwren-camera>__<timestamp>`; 36 HPWREN camera ids; 19 frames byte-identical to local FIgLib |
| SMOKE5K_real | 1,458 | n/a (cuid names) | quarantine_figlib (1,432) / dup_dfire (26) | paper: Smoke5K annotations "built upon the FIgLib data"; shares byte-identical files with FIgLib(114), aiformankind(11), firecam(2), FESB(27+26), D-Fire |
| D-Fire | 10,525 | n/a | dup_dfire | **100% (10,525/10,525) byte-identical (sha256) to our local D-Fire raw**; prefixes WEB/AoF/PublicDataset |
| Boreal | 6,110 | 30 DJI videos | dup_boreal | `Boreal[_subsetA/-Forest-Fire]_<town>_DJI_NNNN_frameN`; 4 Finnish towns (evo, heinola, karkkila, ruokolahti) |
| Forest_Fire | 389 (masks only) | 389 | quarantine_unknown | images not shipped; Kaggle re-upload of USTC video frames, no license |

Totals: usable 29,857 / quarantine_figlib 23,863 / dup_dfire 10,597 / dup_boreal 6,110 / quarantine_unknown 389.
**Only `origin_class == usable` may enter training**, subject to the license caveats above and the AuSmoke split warning below.

## Label semantics (audited by opening ~45 image+mask pairs across all 12 subsets)

- **One binary smoke segmentation mask per image** (`masks/<stem>.png`, single-channel 0/255, same WxH as the image). 49 masks in FESB-MLID are lossy **.jpg** (thresholded at >127 for stats). A few FESB/FireSpot masks contain anti-aliased intermediate values (thresholded).
- **Smoke only. There is no fire/flame class anywhere.** I verified an image with clearly visible night-time flames (D-Fire `PublicDataset00000.jpg`) where only the illuminated smoke is masked, flames unmasked => `has_fire_label=False` everywhere; fire presence is UNKNOWN, not negative.
- **Essentially no negatives**: 70,812 of 70,816 masks are non-empty (only 4 empty, all FIgLib). FireSpot's original 1,179 non-smoke images were not included (its `-0`-suffixed files here still have non-empty masks). The dataset is positives-only — unusable as a negative/confounder source.
- Mask provenance (paper sec 3.3): existing masks retained for SmokeSeg/Smoke5K (manually verified); D-Fire boxes -> SAM box-prompted masks; unannotated images -> pseudo-labels from a trained smoke segmenter + self-training; manual review; ~7K low-quality re-annotated with interactive SAM. => `origin = model-assisted` for most subsets; `original-human` for SmokeSeg/SMOKE5K_real; `unknown` for FESB-MLID.
- Observed mask quality: correctly localized in all ~45 inspected cases (I chased three suspected misplacements at full resolution — all were actually on-target). Boundaries vary: generous/loose blobs around faint plumes (AuSmoke, FireSpot — mask often several times wider than the visible thin plume), blocky staircase edges on Boreal (pseudo-label upsampling), under-coverage of thin diffuse smoke (Boreal dense-smoke frames). Connected components per mask: median 1, 90th pct 34, max 2,548 — many masks are speckly; `n_smoke_boxes` column = 8-connectivity component count of the mask (NOT boxes; no box annotations exist).
- Smoke-size profile (fg fraction, median): firecam 0.0006, FIgLib 0.0018, WSDataset 0.0025, SmokeSeg 0.0029, AuSmoke 0.0030, FireSpot 0.014, D-Fire 0.086, Boreal 0.17. AuSmoke/FIgLib-family = small/faint early smoke (relevant to our objective); Boreal/D-Fire = large dense smoke.

## AusSmoke (the prize subset) — details

- 15,248 frames from **only 29 videos/events** (ACT, Australia: prescribed burns + fire-tower/handheld/PTZ/drone collections around Canberra: Goorooyaroo, Bonshaw, Yarralumla, Black Mountain, Cleland, Mount Stromlo Axis Q6318-LE PTZ, DJI drone flights, Panasonic HC-VX1/Sony FDR-AX53 handhelds). Frames per video: min 25 / median 497 / max 1,448 at ~1fps or 60s-per-frame; adjacent frames are highly correlated — **independent units = 29, not 15,248**.
- **The official Train/Test split is frame-level WITHIN the same 29 videos** (all 29 units appear in both Train and Test) => their benchmark numbers involve near-duplicate train/test frames; for our use, any split must be grouped by `parent_unit`.
- Resolutions 3840x2160 (47%), 1920x1080 (33%), others; filename = `<video>_<frameidx>-<+offset>.jpg` (FIgLib-style offset, 0 = ignition/first-visible-smoke in most units).
- Confounders/caveats inside AuSmoke: one unit is a **smoke machine** (`20240209_SVF_Arbo_SmokeMachine`, 127 frames — artificial smoke source in real scene); one unit is a drone "TEST_RUN_SMOKE"; one unit name has a typo'd date (`17240209_IanTanner_Cleland...`); Axis PTZ frames carry OSD text banners ("Not enough power for both camera and pan tilt motors", timestamps); some scenes are urban vistas of Canberra (buildings, lake) rather than forest.
- Label quality on faint plumes: masks are correctly placed but generously oversized relative to the thin plume (SAM/pseudo-label style). Fine for segmentation-ish supervision or box derivation with dilation tolerance; NOT tight ground truth.

## Counts (honest)

- Unique image files (Train+Test): **70,427** (unique sha256: **70,001** — 426 exact-duplicate files, incl. cross-subset dups FIgLib<->SMOKE5K_real<->firecam<->aiformankind, FESB<->D-Fire).
- **306 rows (147 sha) are byte-identical across Train and Test** (internal leakage in the official split, mostly FIgLib-family).
- Independent units: AuSmoke **29** videos; Boreal **30** videos (two filename families, `Boreal_subsetA_*` and `Boreal-Forest-Fire_*`, are different exports of the SAME videos — 1,416 frame stems appear in both with different bytes => internal near-dups); WSDataset **8** derivable groups (~2 named cameras + 1 rtsp cam + ~4 videos + 1 unknown-frames bucket); FireSpot **29** two-token site groups (160 at 3-token granularity; semantics unverified); FIgLib 391 events / 117 cameras; SmokeSeg 185 events; firecam 72 camera-days; D-Fire/aiformankind/SMOKE5K_real: no unit metadata derivable (parent_unit = per-image; treat as ungroupable).
- Usable subtotal: 29,857 images across 29+8+29 grouped units + 205 FESB stills/short-seqs.

## Contamination vs our existing corpus

- **pyro-sdis**: 0 sha256 matches (all 33,636 local pyro-sdis images hashed and compared).
- **D-Fire**: entire D-Fire subset (10,525) + 46 FESB + 26 SMOKE5K images byte-identical to local D-Fire raw (`dup_of_local_dfire`). D-Fire is already train-only in E — do not double count; note MNS adds *masks* for these D-Fire images (model-assisted SAM-from-box), potentially useful as extra supervision on images we already train on.
- **FIgLib (eval-only for us!)**: MNS FIgLib+SmokeSeg cover 404 FIgLib events; **65 of our 89 opened eval events** (39-event replay set + lockbox v2 + v3) appear, with 2,626 frames of which 1,929 are byte-identical to our local eval frames. aiformankind/firecam/SMOKE5K_real add more HPWREN imagery in cuid/other names that sha-matching only partially catches (crops/re-encodes). **Training anything on the quarantine_figlib portion would invalidate all FIgLib early-detection evals.** Hence subset-level quarantine, not just sha-level.
- Cross-flags in manifest: `dup_of_local_dfire`, `dup_of_local_figlib`, `sha_in_train_and_test`, `sha_dup_within_dataset`.

## Blockers / open items

1. **AuSmoke license unspecified** (no LICENSE in GitHub or HF; citation requested). Publicly released for benchmark use; OK-ish for internal research prototype but no explicit grant — decide/contact authors (henryzhao0615@gmail.com) before any broader use.
2. FESB-MLID and Forest_Fire have no stated license; Forest_Fire images not shipped (Kaggle download needed) — left quarantined.
3. WSDataset "frame_NNNN" files (2,484) and camera_16 files have no derivable capture time; the 08-15-2022 suffix on 5,395 files is an export artifact (identical second on all) — not a timestamp.
4. No negatives anywhere — this dataset cannot supply hard negatives/confounders; pair with our existing negative pools.
5. Central near-dup pass should still run FIgLib-family phash vs our eval frames (crops/re-encodes evade sha), and AuSmoke/WSDataset/FireSpot vs web-scraped D-Fire (low prior, 0 sha hits).
