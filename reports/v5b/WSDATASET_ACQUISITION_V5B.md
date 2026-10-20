# V5b phase 1 — WSDataset acquisition report (data agent, 2026-10-02)

Full card: `data/v5b/manifests/wsdataset_CARD.md` (provenance, verbatim licence evidence, counts, overlap, diagnostics).

## Outcome
- **BLOCKED (original)**: Kaggle `gloryvu/wildfire-smoke-detection` (18.4 GB, v7) needs a Kaggle login/API token
  (download URL 302 → `/account/login`; `~/.kaggle/kaggle.json` absent; no account created). No author hosting elsewhere
  (GitHub VMOKED README dataset link is empty; HF/Zenodo: none). Full file listing WAS obtained without credentials
  (57,194 files): 11,539 author-boxed smoke frames (68 videos) + 13,095 (+218) explicit nonsmoke frames (59+1 videos) +
  YOLO txt for 16,171 images — i.e. the archive contains ~13k author negatives we do not hold.
- **Substitute manifested**: `data/v5b/manifests/wsdataset_manifest.parquet` (11,753 rows).
  * 11,539 MultiNatSmoke copies of the positives → `dup_multinat`, `new_data=False`, model-assisted masks (NOT author boxes),
    proposed_split = train (inherited v4 G role: B_G_v1 x1, C_EGT_v1 x2). 66 original video units recovered from the Kaggle
    listing (+200 ambiguous frames); 0 overlap with any eval/held-out set or FIgLib (sha + phash≤6).
  * 214 author-labelled frames from the author's GitHub demo (new download, 426 MB on disk / 101.1 MB unique new = 129 files):
    126 explicit negatives (120 image-level `nonsmoke_00` = same camera view as MNS smoke_14/14_1 → train-inherit;
    6 detection negatives `non_smoke_71`, new scene → val) and 88 smoke frames of which 85 are phash-identical resized duplicates of MNS frames (3 same-scene near-dups)
    (`dup_multinat_resized`, not new).
- **Licence**: MIT is (a) declared by the author on the Kaggle page metadata (`"licenseNameNullable":"MIT"`) and (b) repeated
  second-hand in the MultiNatSmoke README; the MIT text itself was not read (archive blocked); GitHub demo has NO licence.
  permission_status = pending for all rows (PROTOCOL_V5 §2 unchanged).
- **YOLO**: `data/v5b/yolo/wsdataset/{images,labels}` — 11,545 label files (11,539 mask-derived via v4 `mask_to_lines`, named
  conversion; 6 empty author negatives), images symlinked; image-level frames have no box files.
- **Diagnostic (not review)**: expE_mix (never trained on WSDataset), CPU, 1280: box recall of derived boxes 95/225 = 42.2%
  (IoU≥0.4, conf 0.2; camera_4 group 1/38 boxes yet 35/38 images fire → box geometry mismatch, needs a human look); 79.5% of
  images fire; explicit negatives with any det ≥0.3: 28/126 = 22.2% (hazy forest view). Visual audit sheets:
  `reports/v5b/wsdataset_audit_sheet_{mns,github}.jpg`.

## Human actions that unblock
1. Provide a Kaggle API token (or download the archive yourself) → 18.4 GB single archive (allowed), author boxes + 13k negatives,
   licence file verifiable. 2. Or e-mail the authors (PTIT) for a direct copy / licence confirmation.

## Artifacts
- data/v5b/manifests/wsdataset_manifest.parquet, wsdataset_units.csv, wsdataset_build_summary.json, wsdataset_CARD.md,
  wsdataset_sanity_check.json, wsdataset_provenance/ (Kaggle API JSON + full file listing, GitHub README/API, MNS README, Elsevier API)
- data/v5b/raw/wsdataset_github_demo/ (sparse clone @731b65b), data/v5b/yolo/wsdataset/
- scripts/v5b/wsdataset_build_manifest.py, wsdataset_sanity_check.py, phash_fill.py (+ data/v5b/manifests/_cache/phash_v5b.parquet:
  phash for the 43,704 v4-cache rows that lacked one — pyro-sdis train, D-Fire)
- reports/v5b/wsdataset_audit_sheet_mns.jpg, wsdataset_audit_sheet_github.jpg, this report
