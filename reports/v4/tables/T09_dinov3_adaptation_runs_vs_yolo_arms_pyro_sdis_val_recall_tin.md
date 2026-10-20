_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T09. DINOv3 adaptation runs vs YOLO arms, pyro-sdis VAL: recall / tiny at matched neg-image FP

Cell = recall / tiny-recall (conf at which this model sits at the FP target), interpolated on each model's own curve with the frozen matching code (IoU>=0.4). Denominators: n_images=7,335 (positives 6,177, negatives 1,158); 1% FP = ~12 negative images. v4B* = iteration-4 DINOv3 Faster R-CNN runs on the E+T corpus (B1 frozen ViT-L control, B4 LoRA r16 ViT-L, B5 full unfreeze ViT-L, B6 LoRA r16 ViT-H+); YOLO_* = seed-0 iteration-4 YOLO arms; v3_* = iteration-3 E-only DINO runs. Compare ONLY at matched FP: adapted DINO confidences saturate >=0.99 (see conf in cells) while YOLO sits at 0.5-0.8.

| neg-image FP target | v4B1_frozen_L | v4B4_lora_L | v4B5_full_L | v4B6_lora_Hplus | YOLO_A_ctrlE | YOLO_E_ET | YOLO_C_EGT | expE_y11m_iter2 | v3_B1_frozenL_Eonly12ep | v3_B3_top4L_Eonly12ep |
|---|---|---|---|---|---|---|---|---|---|---|
| 1% | 0.060 / 0.027 (c=0.986) | 0.085 / 0.051 (c=0.997) | n/a (FP floor 1.4%) | 0.097 / 0.038 (c=0.997) | 0.053 / 0.015 (c=0.801) | 0.046 / 0.006 (c=0.819) | 0.041 / 0.002 (c=0.812) | 0.059 / 0.013 (c=0.799) | 0.071 / 0.029 (c=0.967) | 0.033 / 0.017 (c=0.995) |
| 2% | 0.099 / 0.049 (c=0.980) | 0.115 / 0.073 (c=0.996) | 0.094 / 0.039 (c=0.999) | 0.133 / 0.064 (c=0.996) | 0.104 / 0.040 (c=0.768) | 0.073 / 0.013 (c=0.800) | 0.090 / 0.019 (c=0.781) | 0.092 / 0.026 (c=0.775) | 0.100 / 0.042 (c=0.958) | 0.079 / 0.044 (c=0.992) |
| 5% | 0.175 / 0.104 (c=0.968) | 0.210 / 0.144 (c=0.993) | 0.209 / 0.131 (c=0.998) | 0.227 / 0.144 (c=0.994) | 0.204 / 0.112 (c=0.717) | 0.161 / 0.065 (c=0.746) | 0.172 / 0.078 (c=0.735) | 0.190 / 0.097 (c=0.721) | 0.173 / 0.082 (c=0.938) | 0.154 / 0.089 (c=0.986) |
| 10% | 0.301 / 0.200 (c=0.941) | 0.311 / 0.222 (c=0.988) | 0.316 / 0.231 (c=0.996) | 0.340 / 0.252 (c=0.990) | 0.309 / 0.201 (c=0.661) | 0.247 / 0.132 (c=0.691) | 0.266 / 0.164 (c=0.684) | 0.309 / 0.201 (c=0.655) | 0.275 / 0.161 (c=0.903) | 0.264 / 0.159 (c=0.976) |
| 20% | 0.464 / 0.341 (c=0.868) | 0.478 / 0.392 (c=0.973) | 0.488 / 0.413 (c=0.992) | 0.494 / 0.416 (c=0.980) | 0.483 / 0.367 (c=0.547) | 0.415 / 0.294 (c=0.566) | 0.419 / 0.330 (c=0.587) | 0.480 / 0.364 (c=0.535) | 0.457 / 0.329 (c=0.802) | 0.423 / 0.280 (c=0.945) |

Sources:
- `reports/v4/matched_fp_dino_sdis_val.json`
