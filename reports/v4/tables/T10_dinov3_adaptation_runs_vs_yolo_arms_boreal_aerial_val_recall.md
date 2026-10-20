_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T10. DINOv3 adaptation runs vs YOLO arms, boreal AERIAL VAL: recall at matched neg-image FP

Cell = recall / tiny-recall (conf at which this model sits at the FP target), interpolated on each model's own curve with the frozen matching code (IoU>=0.4). Denominators: n_images=1,123 (positives 906, negatives 217); 1% FP = ~2 negative images. ONE burn; differences among DINO runs are within a handful of images. Tiny recall structurally 0.

| neg-image FP target | v4B1_frozen_L | v4B4_lora_L | v4B5_full_L | v4B6_lora_Hplus | YOLO_A_ctrlE | YOLO_E_ET | YOLO_C_EGT |
|---|---|---|---|---|---|---|---|
| 1% | 0.974 / 0.000 (c=0.498) | 0.973 / 0.000 (c=0.868) | 0.963 / 0.000 (c=0.568) | 0.964 / 0.000 (c=0.888) | 0.640 / 0.000 (c=0.408) | 0.931 / 0.000 (c=0.068) | 0.934 / 0.000 (c=0.084) |
| 2% | 0.975 / 0.000 (c=0.406) | 0.983 / 0.000 (c=0.496) | 0.963 / 0.000 (c=0.406) | 0.966 / 0.000 (c=0.837) | 0.674 / 0.000 (c=0.377) | FP ceiling 1.4% -> R>=0.954 | 0.935 / 0.000 (c=0.073) |
| 5% | 0.982 / 0.000 (c=0.056) | 0.983 / 0.000 (c=0.146) | 0.968 / 0.000 (c=0.031) | 0.972 / 0.000 (c=0.511) | 0.783 / 0.000 (c=0.257) | FP ceiling 1.4% -> R>=0.954 | 0.936 / 0.000 (c=0.052) |
| 10% | FP ceiling 7.8% -> R>=0.982 | 0.986 / 0.000 (c=0.038) | FP ceiling 5.1% -> R>=0.968 | 0.980 / 0.000 (c=0.043) | 0.801 / 0.000 (c=0.215) | FP ceiling 1.4% -> R>=0.954 | 0.942 / 0.000 (c=0.025) |
| 20% | FP ceiling 7.8% -> R>=0.982 | FP ceiling 12.4% -> R>=0.987 | FP ceiling 5.1% -> R>=0.968 | FP ceiling 12.0% -> R>=0.985 | 0.828 / 0.000 (c=0.165) | FP ceiling 1.4% -> R>=0.954 | FP ceiling 11.5% -> R>=0.943 |

Sources:
- `reports/v4/matched_fp_dino_boreal_val.json`
