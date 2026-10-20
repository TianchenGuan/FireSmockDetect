_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T25. FINAL TEST — pyro-sdis TEST (fixed-camera source domain; sites ferion + croix-augas, frozen since iteration 1)

One-shot evaluation. Denominators: n_images=4,124 (positives 3,207, negatives 917); 1% neg-FP = ~9 negative images. Matched-FP interpolated on each model's own curve (IoU>=0.4, --top-fine grid for saturated DINO scores). Compare across architectures only at matched FP.

**(a) matched neg-image FP — recall / tiny-recall (conf)**

| neg-image FP target | S0_expE | S0b_expB | S1_A | S2_E_ET | S2c_C_EGT | S3_B_G | S3_D_curr | S4_B6_loraH | S4c_B1_frozenL | S4p_B4_loraL |
|---|---|---|---|---|---|---|---|---|---|---|
| 1% | 0.107 / 0.034 (c=0.766) | 0.118 / 0.026 (c=0.759) | 0.130 / 0.054 (c=0.759) | 0.100 / 0.025 (c=0.756) | 0.116 / 0.025 (c=0.732) | 0.019 / 0.001 (c=0.549) | 0.122 / 0.040 (c=0.719) | 0.051 / 0.005 (c=0.997) | 0.068 / 0.033 (c=0.985) | 0.080 / 0.020 (c=0.996) |
| 2% | 0.172 / 0.081 (c=0.727) | 0.171 / 0.052 (c=0.721) | 0.170 / 0.084 (c=0.732) | 0.128 / 0.042 (c=0.736) | 0.150 / 0.047 (c=0.705) | 0.025 / 0.001 (c=0.508) | 0.170 / 0.071 (c=0.672) | 0.090 / 0.016 (c=0.996) | 0.115 / 0.059 (c=0.978) | 0.127 / 0.039 (c=0.994) |
| 5% | 0.256 / 0.154 (c=0.671) | 0.258 / 0.115 (c=0.655) | 0.273 / 0.175 (c=0.660) | 0.237 / 0.136 (c=0.654) | 0.214 / 0.110 (c=0.648) | 0.037 / 0.005 (c=0.416) | 0.271 / 0.149 (c=0.561) | 0.203 / 0.071 (c=0.991) | 0.192 / 0.103 (c=0.963) | 0.209 / 0.090 (c=0.990) |
| 10% | 0.350 / 0.257 (c=0.599) | 0.354 / 0.206 (c=0.577) | 0.385 / 0.287 (c=0.564) | 0.310 / 0.204 (c=0.587) | 0.318 / 0.214 (c=0.554) | 0.059 / 0.010 (c=0.306) | 0.362 / 0.235 (c=0.456) | 0.363 / 0.189 (c=0.980) | 0.322 / 0.179 (c=0.929) | 0.329 / 0.182 (c=0.980) |
| 20% | 0.506 / 0.406 (c=0.457) | 0.501 / 0.364 (c=0.409) | 0.549 / 0.441 (c=0.409) | 0.424 / 0.324 (c=0.459) | 0.448 / 0.349 (c=0.413) | 0.098 / 0.018 (c=0.160) | 0.478 / 0.361 (c=0.311) | 0.558 / 0.376 (c=0.943) | 0.514 / 0.325 (c=0.820) | 0.509 / 0.343 (c=0.941) |

**(b) R@10% neg-FP deltas**

| system | R@10% | vs S0_expE | vs S1_A |
|---|---|---|---|
| S0_expE | 0.350 | +0.0 pt | -3.6 pt |
| S0b_expB | 0.354 | +0.4 pt | -3.1 pt |
| S1_A | 0.385 | +3.6 pt | +0.0 pt |
| S2_E_ET | 0.310 | -4.0 pt | -7.6 pt |
| S2c_C_EGT | 0.318 | -3.1 pt | -6.7 pt |
| S3_B_G | 0.059 | -29.1 pt | -32.7 pt |
| S3_D_curr | 0.362 | +1.3 pt | -2.3 pt |
| S4_B6_loraH | 0.363 | +1.4 pt | -2.2 pt |
| S4c_B1_frozenL | 0.322 | -2.8 pt | -6.3 pt |
| S4p_B4_loraL | 0.329 | -2.1 pt | -5.7 pt |

**(c) recall / tiny at the pre-registered VAL-selected thresholds, with the FP realised on test negatives**

| system | SOURCE-ONLY sdis-val conf@5% | SOURCE-ONLY sdis-val conf@10% | SOURCE-ONLY sdis-val conf@20% | TARGET-ADAPTED boreal-val conf@1% | TARGET-ADAPTED boreal-val conf@5% |
|---|---|---|---|---|---|
| S0 expE (iteration-2 YOLO11m candidate) | 0.182 / 0.086 (c=0.7210; neg flagged 22/917) | 0.274 / 0.174 (c=0.6546; neg flagged 56/917) | 0.428 / 0.332 (c=0.5354; neg flagged 125/917) | 0.464 / 0.370 (c=0.4983; neg flagged 159/917) | 0.670 / 0.578 (c=0.2615; neg flagged 328/917) |
| S0b expB (deployed default, YOLO11s@1280) | 0.167 / 0.050 (c=0.7231; neg flagged 17/917) | 0.264 / 0.121 (c=0.6503; neg flagged 49/917) | 0.397 / 0.244 (c=0.5323; neg flagged 123/917) | 0.424 / 0.275 (c=0.4983; neg flagged 137/917) | 0.553 / 0.429 (c=0.3477; neg flagged 220/917) |
| S1 A_ctrl_E, median seed | 0.175 / 0.087 (c=0.7283; neg flagged 19/917) | 0.255 / 0.161 (c=0.6727; neg flagged 41/917) | 0.413 / 0.310 (c=0.5392; neg flagged 106/917) | 0.476 / 0.368 (c=0.4883; neg flagged 138/917) | 0.660 / 0.568 (c=0.2558; neg flagged 298/917) |
| S2 E_ET, median seed | 0.155 / 0.065 (c=0.7154; neg flagged 28/917) | 0.229 / 0.128 (c=0.6603; neg flagged 44/917) | 0.347 / 0.241 (c=0.5534; neg flagged 111/917) | 0.718 / 0.672 (c=0.0483; neg flagged 514/917) | n/a(val FP floor) |
| S2c C_EGT, median seed | 0.113 / 0.022 (c=0.7344; neg flagged 7/917) | 0.181 / 0.075 (c=0.6768; neg flagged 33/917) | 0.294 / 0.191 (c=0.5767; neg flagged 74/917) | 0.535 / 0.448 (c=0.2983; neg flagged 274/917) | 0.711 / 0.671 (c=0.0705; neg flagged 535/917) |
| S3 B_Gonly, seed 0 | 0.036 / 0.005 (c=0.4231; neg flagged 45/917) | 0.051 / 0.008 (c=0.3380; neg flagged 77/917) | 0.091 / 0.017 (c=0.1885; neg flagged 160/917) | 0.032 / 0.002 (c=0.4583; neg flagged 38/917) | 0.048 / 0.006 (c=0.3558; neg flagged 68/917) |
| S3 D_curr, median seed | 0.112 / 0.033 (c=0.7292; neg flagged 7/917) | 0.189 / 0.085 (c=0.6530; neg flagged 21/917) | 0.303 / 0.177 (c=0.5258; neg flagged 58/917) | 0.588 / 0.492 (c=0.1683; neg flagged 280/917) | 0.622 / 0.538 (c=0.1303; neg flagged 317/917) |
| S4 v4B6_lora_Hplus (DINOv3 LoRA-H+) | 0.143 / 0.040 (c=0.9939; neg flagged 30/917) | 0.225 / 0.083 (c=0.9901; neg flagged 56/917) | 0.367 / 0.192 (c=0.9801; neg flagged 92/917) | 0.651 / 0.495 (c=0.8883; neg flagged 269/917) | 0.808 / 0.705 (c=0.5115; neg flagged 464/917) |
| S4c v4B1_frozen_L (DINOv3 frozen-L) | 0.170 / 0.090 (c=0.9677; neg flagged 32/917) | 0.280 / 0.154 (c=0.9407; neg flagged 79/917) | 0.459 / 0.286 (c=0.8680; neg flagged 150/917) | 0.664 / 0.471 (c=0.4983; neg flagged 370/917) | 0.822 / 0.678 (c=0.0558; neg flagged 642/917) |
| S4p v4B4_lora_L (DINOv3 LoRA-L) | 0.145 / 0.047 (c=0.9934; neg flagged 22/917) | 0.239 / 0.112 (c=0.9881; neg flagged 54/917) | 0.382 / 0.228 (c=0.9730; neg flagged 112/917) | 0.620 / 0.459 (c=0.8683; neg flagged 273/917) | 0.838 / 0.729 (c=0.1458; neg flagged 564/917) |

Sources:
- `reports/v4/test/matched_fp_sdistest.json`
- `reports/v4/test/yolo_test_at_preregistered_thresholds.json`
- `reports/v4/test/dino_test_at_preregistered_thresholds.json`
