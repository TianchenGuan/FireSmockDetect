# rfdetrL_1280_e48 final artifacts (iteration 3, Track A)

- Source: green g6e job `tianrgua-rfdetrl1280e48a1-gs9y` (SUCCEEDED, 5h34m wall,
  48/48 epochs), run dir `/scratch/scratch/tianrgua-sandbox/smoke/runs_v3/rfdetrL_1280_e48`.
  Fetched with `python cluster/v3/fetch_run.py --run rfdetrL_1280_e48 --files
  checkpoint_best_ema.pth --files metrics.csv --files console.log`.
- `checkpoint_best_ema.pth` — EMA weights, best val mAP50:95 (epoch 37, 0.4701),
  141,649,947 bytes, sha256 `2cca1fe8cd381b5da3fa4919c118da59b789ee22f9eac8ed562a3f11000743d0`.
- `metrics.csv` — full Lightning metrics log (1,249 rows).
- `val_curve_training_protocol.csv` — per-epoch val curve (torchmetrics COCO
  protocol, RF-DETR's own loop; NOT our eval protocol).
- `train_loss_by_epoch.csv` — mean train loss per epoch.
- `console.log`, `scratch_listing.txt` — job console + run-dir listing (other
  checkpoints, incl. last.ckpt / checkpoint_best_total.pth, stay on /scratch).
- Model + weights license: Apache-2.0 (RF-DETR-Large); code from the pinned
  clone third_party/rf-detr @ a57ad2345c07c7822d110ef963b3bc330beb9937.
