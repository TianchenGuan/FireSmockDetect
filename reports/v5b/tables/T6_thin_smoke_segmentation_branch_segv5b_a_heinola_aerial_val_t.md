_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

### T6. Thin-smoke SEGMENTATION branch (segV5b_a): heinola aerial VAL, thr 0.5, best epoch selected on VAL (selection-optimistic)

Train = 665 Boreal TRAIN-burn frames (628 masks: 601 SAM model-assisted + 27 human; 37 author negatives), VAL = 382 heinola frames (329 SAM + 7 human + 46 author negatives). 19.5 node-minutes on 1 L40S node (res-segv5b-a). IoU vs SAM masks measures agreement with SAM, not with humans; the 7 human masks show diffuse haze is under-segmented. Mask->box recall vs the authors' one-big-box labels (A: component bbox IoU>=0.4 / B: union bbox / C: localisation hit); the GT masks themselves reach only 0.887 / 0.946 / 1.000. Negative-frame false alarms are driven by ONE clear-cut flight (DJI_0024: 21/21 frames for every model) -> the map is an explanation overlay, not a stand-alone alarm.

| config | checkpoint | best | IoU | Dice | P | R | mIoU pos | IoU vs SAM (n) | IoU vs human (n) | neg-frame FA any / >=1% | mask->box A / B / C | params trainable / total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DINOv3-L LoRA r16 + SFP + FPN-lite | models/v5b/seg/segV5b_a/seg01/lora/best.pt | ep 2 / step 166 | 0.551 | 0.710 | 0.855 | 0.608 | 0.591 | 0.556 (329) | 0.442 (7) | 0.457 / 0.065 | 0.804 / 0.860 / 0.994 | 13,248,129 / 316,377,729 |
| DINOv3-L frozen + SFP + FPN-lite | models/v5b/seg/segV5b_a/seg02/frozen/best.pt | ep 2 / step 166 | 0.568 | 0.724 | 0.842 | 0.636 | 0.604 | 0.571 (329) | 0.474 (7) | 0.457 / 0.000 | 0.830 / 0.878 / 0.997 | 10,102,401 / 313,232,001 |
| ResNet-50 (ImageNet, frozen) + FPN-lite (baseline) | models/v5b/seg/segV5b_a/seg01/r50/best.pt | ep 3 / step 123 | 0.510 | 0.675 | 0.742 | 0.620 | 0.534 | 0.514 (329) | 0.453 (7) | 1.000 / 0.109 | 0.854 / 0.878 / 1.000 | 5,410,177 / 28,918,209 |

**threshold sweep (selected rows)**

| config | thr | IoU | Dice | P | R | neg FA any | neg FA >=1% | neg FP pixel frac |
|---|---|---|---|---|---|---|---|---|
| frozen | 0.3 | 0.572 | 0.728 | 0.819 | 0.655 | 0.457 | 0.000 | 0.0046 |
| frozen | 0.5 | 0.567 | 0.724 | 0.842 | 0.635 | 0.457 | 0.000 | 0.0027 |
| frozen | 0.7 | 0.558 | 0.716 | 0.862 | 0.613 | 0.457 | 0.000 | 0.0016 |
| lora | 0.3 | 0.553 | 0.712 | 0.831 | 0.623 | 0.500 | 0.391 | 0.0064 |
| lora | 0.5 | 0.550 | 0.709 | 0.854 | 0.606 | 0.457 | 0.065 | 0.0047 |
| lora | 0.7 | 0.541 | 0.702 | 0.876 | 0.586 | 0.457 | 0.000 | 0.0030 |
| r50 | 0.3 | 0.508 | 0.674 | 0.703 | 0.647 | 1.000 | 0.174 | 0.0090 |
| r50 | 0.5 | 0.508 | 0.674 | 0.742 | 0.618 | 1.000 | 0.109 | 0.0064 |
| r50 | 0.7 | 0.504 | 0.670 | 0.780 | 0.588 | 0.913 | 0.087 | 0.0046 |

**per-epoch heinola VAL IoU**

| config | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | ep9 | ep10 |
|---|---|---|---|---|---|---|---|---|---|---|
| lora | 0.522 | 0.551 | 0.506 | 0.484 | 0.523 | 0.507 | 0.520 | 0.526 | 0.526 | 0.529 |
| frozen | 0.481 | 0.568 | 0.484 | 0.499 | 0.519 | 0.517 | 0.511 | 0.515 | 0.523 | 0.523 |
| r50 | 0.444 | 0.496 | 0.510 | 0.495 | 0.498 | 0.479 | 0.490 | 0.490 | 0.493 | 0.494 |

Sources:
- `models/v5b/seg/segV5b_a/MANIFEST.json`
- `models/v5b/seg/segV5b_a/seg01/frozen/thr_sweep.json`
- `models/v5b/seg/segV5b_a/seg01/lora/metrics.jsonl`
- `models/v5b/seg/segV5b_a/seg01/lora/thr_sweep.json`
- `models/v5b/seg/segV5b_a/seg01/r50/metrics.jsonl`
- `models/v5b/seg/segV5b_a/seg01/r50/thr_sweep.json`
- `models/v5b/seg/segV5b_a/seg02/frozen/metrics.jsonl`
- `reports/v5b/SEGMENTATION_V5B.md`
- `data/v5b/manifests/seg_v5b_manifest.json`
