# SMOKE-EDGE release — YOLO11n smoke detector (model A, v6)

Offline-capable compact smoke detector. Single class ("smoke"), tuned for
early faint plumes on fixed lookout cameras; also works on aerial views.

## Files

| file | size | role |
|---|---|---|
| `lockbox_A/best.pt` | 5.5 MB | PyTorch checkpoint (YOLO11n, 2.59 M params, imgsz 1280, trained 60 ep on pyro-sdis+D-Fire train, seed 0) |
| `lockbox_A/model.onnx` | 11.1 MB | ONNX (fp32, no NMS baked) — parity vs torch verified (76/76 boxes, raw maxΔ 0.016) |
| `lockbox_A/model_int8.tflite` | **3.6 MB** | INT8 PTQ (calibration: train/val only) — the edge artifact |
| `lockbox_A/model_fp16.tflite` | 5.7 MB | fp16 fallback (accuracy identical to INT8) |
| `infer.py` | — | one-command offline inference (images / folder / video) |
| `demo/` | — | real-sample demo outputs with boxes + RUN_REPORT.json |
| `failure_examples/` | — | representative misses / false positives (val) |
| `lockbox_A/SHA256SUMS` | — | artifact hashes |

## Install & run

```bash
pip install numpy pillow opencv-python-headless ai-edge-litert   # tflite path
# (.pt path additionally needs: pip install ultralytics; .onnx needs onnxruntime)
python infer.py --model lockbox_A/model_int8.tflite --source your_images/ --conf 0.62
python infer.py --model lockbox_A/best.pt --source video.mp4 --out out/
```

No network access needed at runtime.

## Model choice & thresholds (frozen from the INT8 artifact's own val curve)

| tolerated false-alarm rate (fraction of no-smoke frames that alarm) | conf |
|---|---|
| ~2 % | 0.73 |
| ~5 % | 0.68 |
| ~10 % | 0.62 |

Use `model_int8.tflite` for size/speed. At the 2 %/5 % operating points the
INT8 file matches fp32 accuracy; at the 10 % point it loses 2.1 pt recall
(graph-conversion effect — fp16 TFLite is accuracy-identical to INT8, so
quantization itself is lossless here; use ONNX/`best.pt` if that margin
matters and size does not).

## Validated performance (the delivered files themselves)

Primary val = pyro-sdis VAL (7,335 imgs / 1,158 author-negative), metric =
box recall (IoU≥0.4) at matched negative-image FP:

| artifact | R@2% | R@5% | R@10% |
|---|---|---|---|
| best.pt (fp32) | 0.107 | 0.190 | 0.310 |
| model_int8.tflite | 0.111 | 0.194 | 0.289 |

Frozen one-shot TEST (pyro-sdis TEST, 4,124 imgs): fp32 R@5% 0.243 /
R@10% **0.344** — best nano model of the v6 matrix (seed-0; A seed spread on
val 0.310–0.341). Other sources (one-shot, fp32): boreal_test R≈0.93
(saturated), fireman aerial image-recall 0.692, forest 0.083 (hard domain),
sheffield 0.193 (weak fragmented labels). FireSpot H-NEW: not evaluated
(fetch failures), pending.

## Measured hardware (test machine)

x86_64 Linux devbox, CPU only, batch=1, input 1280×1280:

| path | raw forward (median) | end-to-end incl. preprocess+NMS |
|---|---|---|
| TFLite INT8 | 45.8 ms | ~290–325 ms |
| TFLite fp16 | 90.4 ms | — |
| ONNX fp32 | 116 ms | ~320 ms |

**On-device (phone/drone) latency, memory and power are UNMEASURED** — run
`infer.py` on the target device to get real numbers; it prints an
end-to-end report.

## Known limitations

- Misses concentrate on very small, low-contrast distant plumes and on
  camera viewpoints unseen in training; false alarms are valley fog, clouds
  touching ridgelines, glare.
- Synthetic training data (procedural and generative) did NOT improve this
  model; it is trained on real data only (full study: project reports v6).
- Server-side default remains the larger expB_y11s model (unchanged).

Licences: model weights AGPL-3.0 (ultralytics lineage); training data
licences remain with the dataset owners.
