# smoke-det3:v1 — iteration 3 Track A detection container (RF-DETR + DEIMv2)
# Base: same verified pytorch 2.8.0 cu128 runtime as iterations 1/2 (py3.11).
# Everything needed offline is baked: both repos (pinned clones), pretrain
# weights, COCO jsons, curl (presigned uploads), entry script fallback.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# pods have no internet; also RF_HOME points weight cache at baked weights
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RF_HOME=/opt/rfhome \
    PYTHONUNBUFFERED=1

# RF-DETR @ a57ad2345c07c7822d110ef963b3bc330beb9937 (Apache-2.0) + train extras
COPY rf-detr /opt/src/rf-detr
RUN pip install --no-cache-dir "/opt/src/rf-detr[train]" \
    && python -c "import rfdetr, pytorch_lightning, torchmetrics, torch; print('torch', torch.__version__)"

# DEIMv2 @ 1d2ca42171570c713e78fc6a766ec5104b7f4724 (custom NON-COMMERCIAL
# license — benchmark/evaluation use only, see LICENSE.md in the copy) + deps
# (requirements.txt minus its torch/torchvision pins: we keep base 2.8.0/0.23).
COPY DEIMv2 /opt/DEIMv2
RUN pip install --no-cache-dir faster-coco-eval PyYAML tensorboard scipy calflops

# Pretrain weights (rf-detr-large-2026.pth md5 5cb72153541cbcb9aa6efa26222acc75,
# verified against src/rfdetr/assets/model_weights.py registry at pin)
COPY rfhome/ /opt/rfhome/
COPY deimv2_ckpts/ /opt/DEIMv2/ckpts/

# Frozen-split COCO conversions (scripts/v3/yolo_to_coco.py, verified via
# contact sheets reports/v3/coco_verify_*.jpg)
COPY data/ /opt/data/

COPY det_entry.sh train_rfdetr.py /opt/

# pods run as uid 1001 — gdown-downloaded files are mode 600 on the devbox
# and COPY preserves that (caused PermissionError in job deimv2msanitya1)
RUN chmod -R a+rX /opt/DEIMv2 /opt/rfhome /opt/data /opt/det_entry.sh /opt/train_rfdetr.py

# rf-detr's matcher REQUIRES torch_linear_assignment; on CUDA it uses a
# Triton kernel whose driver shim JIT-compiles a C module at runtime ->
# needs a C compiler (job rfdetrlsanitya1 failed: "Failed to find C
# compiler"). Runtime base ships none; add gcc late to keep pip layers cached.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
