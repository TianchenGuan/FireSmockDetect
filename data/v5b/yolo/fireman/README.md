# FireMan-UAV-RGBT YOLO export (class 0 = smoke)
Source: data/v5b/manifests/fireman_manifest.parquet. Boxes = author smoke_region (CVAT, source semi-auto/manual). Empty txt = author image-level smoke_presence=no. Image-level positives without boxes have no txt (image_level_only.txt). Quarantine/thermal/conflict rows not exported.
{
 "holdout": {
  "box_files": 19458,
  "empty_neg_files": 0,
  "image_level_only_no_file": 0,
  "unlabeled_no_file": 0,
  "conflict_skipped": 1,
  "boxes": 19458
 },
 "train": {
  "box_files": 19858,
  "empty_neg_files": 3396,
  "image_level_only_no_file": 0,
  "unlabeled_no_file": 0,
  "conflict_skipped": 9,
  "boxes": 21654
 },
 "val": {
  "box_files": 7376,
  "empty_neg_files": 1154,
  "image_level_only_no_file": 0,
  "unlabeled_no_file": 0,
  "conflict_skipped": 0,
  "boxes": 7376
 }
}