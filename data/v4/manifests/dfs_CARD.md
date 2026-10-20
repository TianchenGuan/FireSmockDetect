# DFS (DFS-FIRE-SMOKE-Dataset) — Source Card (iteration 4)

**Status: BLOCKED — no data acquired.** `data/v4/raw/dfs/` is intentionally empty.
No manifest parquet was produced (nothing to hash/row). This card records provenance,
what was verified, and the exact blocker, audited 2026-09-14.

## Provenance

- Repo: https://github.com/siyuanwu/DFS-FIRE-SMOKE-Dataset (default branch `main`;
  repo contains only README.md, .gitattributes, and Figure/ example PNGs — no data, no code).
- Paper: Wu, Zhang, Liu, Li, "A dataset for fire and smoke object detection",
  Multimedia Tools and Applications, 2022 (Springer). DOI 10.1007/s11042-022-13580-x.
  (Springer page is bot-challenged from this host; not fetched.)
- README claims: 9,462 images, classes `fire` / `smoke` / `other` (confounder class for
  fire-like objects), VOC-format boxes, images "named by the fire size".
- GitHub issue #3 (author reply): annotated with **labelme**, converted to VOC XML;
  users report the XMLs do not open directly in labelImg (non-standard VOC details).
- GitHub issue #6: **no official train/val/test partition** was published (unanswered).
- GitHub issue #9 (unanswered): asks the source of "gasoline burning" / "styrene" burn
  images — suggests some imagery is from lab/controlled-burn footage; provenance of the
  photos themselves is not documented anywhere in the repo.

## Download URLs (official, from README)

1. Baidu Pan: https://pan.baidu.com/s/1-kV1h78BwuNe-A86cDUuvQ password `pnxx` — **ALIVE**
2. OneDrive: https://1drv.ms/u/s!AnnO2n5F9u0IehApsqCBnnYvXvE?e=8nroFp — **DEAD**

## Exact blocker

- OneDrive link: returns `accessDenied` ("The sharing link no longer exists, or you do not
  have permission to access it") via the OneDrive shares API (both `api.onedrive.com` and
  the migrated-to-SharePoint `my.microsoftpersonalcontent.com` Badger flow), and a real
  Chromium session renders "Sorry, you cannot access this document." GitHub issues #7/#8
  show it has been dead since at least 2025-06 with no author fix (latest user complaint
  2026-01-12).
- Baidu link: alive and the extraction code `pnxx` is accepted anonymously (verified via
  `share/verify` API, errno=0, and in a real browser session). Share = folder `火灾`
  ("fire"), shared 2022-09-29 by user `ws***65`, permanent validity. Full anonymous
  browsing of the tree works, but **download does not**: the web client spins on
  "正在获取下载链接" (fetching download link) and never yields a link for an anonymous
  session; Baidu Pan requires a logged-in Baidu account (registration requires a Chinese
  phone number) and, for files of this size, the proprietary netdisk client.
  Per task policy: **Baidu-only = blocker.** Unauthorized third-party mirrors
  (Kaggle/Roboflow re-uploads, pan-download proxies) were NOT used.
- No author-linked mirror exists on Zenodo, figshare, or Hugging Face (searched all three;
  no DFS hits). Author's other GitHub repos contain nothing dataset-related.

## What the live Baidu share contains (enumerated anonymously in browser, 2026-09-14)

```
火灾/
├── 平台/          (platform — code/tooling, not enumerated)
├── 依赖库/        (dependency libraries, not enumerated)
└── 数据集/        (datasets)
    ├── 常规数据集/   (standard datasets)
    │   ├── 2025-10/            (folder, modified 2026-09-02)
    │   ├── Smoke100k/          (folder, modified 2026-09-02)
    │   ├── Smoke100k(1)/       (folder, modified 2026-09-02)
    │   ├── DFS.rar        742.2 MB   modified 2026-09-02 19:59
    │   └── fireLabel.rar  369.4 MB   modified 2026-09-02 19:59
    └── 实际场景数据/  (real-scene data, modified 2026-09-02, not enumerated)
```

`DFS.rar` (742.2 MB) is presumably the 9,462-image dataset + VOC XMLs. Note the share
also bundles third-party data (Smoke100k is a different, synthetic smoke dataset by
Cheng et al.) — if this share is ever retrieved, only DFS.rar should be treated as DFS.

## License / terms

**No license found anywhere.** GitHub API reports `license: None`; no LICENSE/terms file
in the repo; README only requests citation of three papers (Wu et al. MTAP 2022;
Liu et al. CVIDL 2021; Sun et al. JRTIP 2023). Springer's page could not be checked from
this host. Treat as: all-rights-reserved academic release, citation requested, no explicit
grant for redistribution or commercial use. For our internal research prototype this is a
gray area even if we could download it — flag for legal review if it is ever acquired.

## Label semantics

**NOT independently audited** — no label files could be opened (deliverable #3 impossible
without data). From README + issues only (unverified):
- Box annotations, VOC-style XML (converted from labelme; reportedly not fully
  labelImg-compatible).
- 3 classes: `fire`, `smoke`, `other` (`other` = fire-colored/bright confounders such as
  lights/sun/reflections — this confounder class was the main reason DFS was shortlisted).
- Whether smoke is exhaustively annotated on fire images (or fire-only boxes imply
  unlabeled smoke) is UNKNOWN.
- Image-level counts per class: UNKNOWN (README gives only the 9,462 total).

## Counts

- Unique images: 0 acquired (README claims 9,462; unverified).
- Independent units/events: 0 acquired. Note: DFS images are web/real-scene photos
  "named by the fire size"; per-event grouping is unlikely to be derivable even with the
  data — expect per-image units with unknown inter-image correlation.

## Contamination vs pyro-sdis / D-Fire / FIgLib

Not checkable without files (no sha256s, no filenames). Prior expectation: DFS is
general fire/smoke web imagery; overlap risk with D-Fire (also web-scraped fire/smoke) is
plausible and MUST be checked centrally if DFS is ever acquired. Overlap with FIgLib
(fixed HPWREN towers) and pyro-sdis unlikely but unverified.

## Options to unblock (for orchestrator)

1. Open a GitHub issue / email the authors asking to restore the OneDrive link
   (issues #7/#8 already ask; no response in ~15 months — low odds).
2. Someone with a Baidu account + netdisk client downloads `DFS.rar` (742.2 MB) and
   `fireLabel.rar` from the live share and drops them in `data/v4/raw/dfs/`.
3. Drop DFS from iteration 4.
