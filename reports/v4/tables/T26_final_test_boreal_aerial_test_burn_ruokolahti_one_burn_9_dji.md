_generated 2026-09-15T03:55Z by scripts/v4/make_report_tables.py from saved artifacts — do not edit by hand_

### T26. FINAL TEST — boreal AERIAL TEST (burn_ruokolahti, ONE burn, 9 DJI flights): recall at the pre-registered thresholds

Denominators: n_images=1,764 (positives 1,762, negatives 2) — ONE independent burn; with 2 negatives the false-alarm side is UNMEASURABLE ('neg flagged x/2' is anecdotal). Tiny recall is undefined (no tiny GT boxes) and omitted. SOURCE-ONLY = sdis-val conf applied unchanged; TARGET-ADAPTED = boreal-val (heinola) conf. 'n/a(val FP floor)' = the model never reached that FP on boreal val.

| system | SOURCE-ONLY sdis-val conf@5% | SOURCE-ONLY sdis-val conf@10% | SOURCE-ONLY sdis-val conf@20% | TARGET-ADAPTED boreal-val conf@1% | TARGET-ADAPTED boreal-val conf@5% |
|---|---|---|---|---|---|
| S0 expE (iteration-2 YOLO11m candidate) | 0.720 (c=0.7210; neg flagged 0/2) | 0.877 (c=0.6546; neg flagged 0/2) | 0.917 (c=0.5354; neg flagged 0/2) | 0.920 (c=0.4983; neg flagged 0/2) | 0.928 (c=0.2615; neg flagged 0/2) |
| S0b expB (deployed default, YOLO11s@1280) | 0.782 (c=0.7231; neg flagged 0/2) | 0.886 (c=0.6503; neg flagged 0/2) | 0.918 (c=0.5323; neg flagged 0/2) | 0.921 (c=0.4983; neg flagged 0/2) | 0.928 (c=0.3477; neg flagged 0/2) |
| S1 A_ctrl_E, median seed | 0.839 (c=0.7283; neg flagged 0/2) | 0.906 (c=0.6727; neg flagged 0/2) | 0.922 (c=0.5392; neg flagged 0/2) | 0.924 (c=0.4883; neg flagged 0/2) | 0.928 (c=0.2558; neg flagged 0/2) |
| S2 E_ET, median seed | 0.916 (c=0.7154; neg flagged 0/2) | 0.922 (c=0.6603; neg flagged 0/2) | 0.925 (c=0.5534; neg flagged 0/2) | 0.934 (c=0.0483; neg flagged 0/2) | n/a(val FP floor) |
| S2c C_EGT, median seed | 0.916 (c=0.7344; neg flagged 0/2) | 0.921 (c=0.6768; neg flagged 0/2) | 0.925 (c=0.5767; neg flagged 0/2) | 0.932 (c=0.2983; neg flagged 0/2) | 0.934 (c=0.0705; neg flagged 0/2) |
| S3 B_Gonly, seed 0 | 0.788 (c=0.4231; neg flagged 0/2) | 0.865 (c=0.3380; neg flagged 0/2) | 0.933 (c=0.1885; neg flagged 0/2) | 0.762 (c=0.4583; neg flagged 0/2) | 0.854 (c=0.3558; neg flagged 0/2) |
| S3 D_curr, median seed | 0.921 (c=0.7292; neg flagged 0/2) | 0.923 (c=0.6530; neg flagged 0/2) | 0.924 (c=0.5258; neg flagged 0/2) | 0.931 (c=0.1683; neg flagged 0/2) | 0.933 (c=0.1303; neg flagged 0/2) |
| S4 v4B6_lora_Hplus (DINOv3 LoRA-H+) | 0.928 (c=0.9939; neg flagged 0/2) | 0.929 (c=0.9901; neg flagged 0/2) | 0.931 (c=0.9801; neg flagged 0/2) | 0.934 (c=0.8883; neg flagged 0/2) | 0.934 (c=0.5115; neg flagged 0/2) |
| S4c v4B1_frozen_L (DINOv3 frozen-L) | 0.927 (c=0.9677; neg flagged 0/2) | 0.929 (c=0.9407; neg flagged 0/2) | 0.933 (c=0.8680; neg flagged 0/2) | 0.934 (c=0.4983; neg flagged 0/2) | 0.937 (c=0.0558; neg flagged 0/2) |
| S4p v4B4_lora_L (DINOv3 LoRA-L) | 0.931 (c=0.9934; neg flagged 0/2) | 0.932 (c=0.9881; neg flagged 0/2) | 0.933 (c=0.9730; neg flagged 0/2) | 0.934 (c=0.8683; neg flagged 0/2) | 0.935 (c=0.1458; neg flagged 1/2) |

Sources:
- `reports/v4/test/yolo_test_at_preregistered_thresholds.json`
- `reports/v4/test/dino_test_at_preregistered_thresholds.json`
- `reports/v4/LOCKBOX_V4_FROZEN.md (composition)`
