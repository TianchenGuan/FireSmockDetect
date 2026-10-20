### Headline (heinola VAL, 382 frames: 329 SAM-mask positives + 7 manual-mask positives + 46 author negatives; thr 0.5; metrics on valid letterboxed pixels at stride 4)

| config | best ep/step | smoke IoU (all) | Dice | precision | recall | mIoU pos frames | IoU vs SAM masks (329) | IoU vs manual masks (7) | mIoU manual (7) | neg FP pixel frac | neg frame FA (any comp) | neg frame FA (comp>=1%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DINOv3-L + LoRA r16 + SFP + FPN-lite head | 166 | 0.551 | 0.710 | 0.855 | 0.608 | 0.591 | 0.556 | 0.442 | 0.445 | 0.0047 | 0.457 | 0.065 |
| DINOv3-L frozen + SFP + FPN-lite head | 166 | 0.568 | 0.724 | 0.842 | 0.636 | 0.604 | 0.571 | 0.474 | 0.476 | 0.0027 | 0.457 | 0.000 |
| ResNet-50 (ImageNet, frozen) + FPN-lite/U-Net-style head (baseline) | 123 | 0.510 | 0.675 | 0.742 | 0.620 | 0.534 | 0.514 | 0.453 | 0.450 | 0.0064 | 1.000 | 0.109 |

### Mask -> box recall vs the authors' boxes (336 boxed heinola frames; A: any component bbox IoU>=0.4; B: union bbox IoU>=0.4; C: >=30% of predicted pixels inside the GT box and >=10% of the box covered)

| config | A comp IoU>=0.4 | B union IoU>=0.4 | C hit | n boxes | GT-mask reference A / B / C |
|---|---|---|---|---|---|
| DINOv3-L + LoRA r16 + SFP + FPN-lite head | 0.804 | 0.860 | 0.994 | 336 | 0.887 / 0.946 / 1.000 |
| DINOv3-L frozen + SFP + FPN-lite head | 0.830 | 0.878 | 0.997 | 336 | 0.887 / 0.946 / 1.000 |
| ResNet-50 (ImageNet, frozen) + FPN-lite/U-Net-style head (baseline) | 0.854 | 0.878 | 1.000 | 336 | 0.887 / 0.946 / 1.000 |

### Parameters / training

| config | backbone trainable / total | neck | head | all trainable | steps | s/it (last) | epochs evaluated |
|---|---|---|---|---|---|---|---|
| DINOv3-L + LoRA r16 + SFP + FPN-lite head | 3,145,728 / 306,275,328 | 7,805,184 | 2,297,217 | 13,248,129 | 830 | 0.33 | 10 |
| DINOv3-L frozen + SFP + FPN-lite head | 0 / 303,129,600 | 7,805,184 | 2,297,217 | 10,102,401 | 830 | 0.28 | 10 |
| ResNet-50 (ImageNet, frozen) + FPN-lite/U-Net-style head (baseline) | 0 / 23,508,032 | - | 5,410,177 | 5,410,177 | 410 | 0.14 | 10 |

### Per-epoch heinola VAL IoU

| config | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | ep9 | ep10 |
|---|---|---|---|---|---|---|---|---|---|---|
| lora | 0.522 | 0.551 | 0.506 | 0.484 | 0.523 | 0.507 | 0.520 | 0.526 | 0.526 | 0.529 |
| frozen | 0.481 | 0.568 | 0.484 | 0.499 | 0.519 | 0.517 | 0.511 | 0.515 | 0.523 | 0.523 |
| r50 | 0.444 | 0.496 | 0.510 | 0.495 | 0.498 | 0.479 | 0.490 | 0.490 | 0.493 | 0.494 |

### Per-flight IoU on heinola (main configs; positives only, mean per-image IoU)

| flight | n pos | n neg | lora mIoU | frozen mIoU | r50 mIoU | lora neg FA | frozen neg FA | r50 neg FA |
|---|---|---|---|---|---|---|---|---|
| heinola_DJI_0024 | 0 | 21 | - | - | - | 1.00 | 1.00 | 1.00 |
| heinola_DJI_0025 | 79 | 0 | 0.522 | 0.562 | 0.513 | - | - | - |
| heinola_DJI_0028 | 5 | 25 | 0.551 | 0.571 | 0.556 | 0.00 | 0.00 | 1.00 |
| heinola_DJI_0029 | 42 | 0 | 0.836 | 0.836 | 0.698 | - | - | - |
| heinola_DJI_0030 | 81 | 0 | 0.601 | 0.624 | 0.557 | - | - | - |
| heinola_DJI_0033 | 90 | 0 | 0.555 | 0.564 | 0.535 | - | - | - |
| heinola_DJI_0036 | 39 | 0 | 0.538 | 0.489 | 0.345 | - | - | - |
