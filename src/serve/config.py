"""Service configuration. All values overridable via SMOKE_* env vars."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "SMOKE_", "protected_namespaces": ()}

    # model
    backend: str = "yolo"   # see src/serve/backends.py AVAILABLE
    # file defaults now match the DEPLOYED configuration (expB@1280). Before
    # 2026-09-14 they pointed at the contaminated pyronear baseline and expB
    # was env-only, so a bare launch silently served the contaminated model
    # (reports/v4/RECONCILIATION_V4.md D1).
    weights_path: str = "models/expB_best.pt"
    model_version: str = "expB-mix-y11s-1280@seg03"
    imgsz: int = 1280
    device: str = "cpu"  # "cuda:0" when GPU serving
    # tiled inference: 0 disables. tile inference runs alongside a full-image
    # pass and merges via NMS; improves small/distant smoke recall.
    tile_size: int = 0
    tile_overlap: float = 0.2
    merge_iou: float = 0.55

    # detection thresholds (selected on validation data; see reports/)
    conf_floor: float = 0.05        # discard below this before any logic
    single_frame_alert_conf: float = 0.30   # provisional warning from one frame
    seq_frame_conf: float = 0.15    # frame "hit" threshold for persistence logic
    seq_window: int = 5             # look at last N frames of this device
    seq_hits_required: int = 3      # >=K hits in window -> confirmed alert
    cooldown_s: float = 60.0        # min seconds between repeated confirmed alerts
    stale_after_s: float = 600.0    # reset device history after this gap
    scene_change_hamming: int = 20  # phash distance to treat as new scene (camera moved)

    # dino3 backend (iteration 4, LoRA checkpoints) -- ALL optional. Defaults
    # mean "auto-detect from the checkpoint" and leave every other backend's
    # behavior unchanged; see src/serve/backend_dino3.py.
    dino_backbone_dir: str = ""      # "" = auto: ckpt args / neck dim -> models/v3/hf/<snapshot>
    lora_r: int | None = None        # None = auto from ckpt lora_A keys; explicit must match
    lora_alpha: float | None = None  # None = ckpt training value (2*r default)
    lora_merge: bool = False         # fold LoRA into base linears after load (deployment form)
    det_width: int = 0               # 0 = ckpt-recorded letterbox geometry (1280x736)
    det_height: int = 0

    # shadow / bypass recording (iteration 5) -- ALL optional, default OFF.
    # When shadow_backend is non-empty, /detect ALSO runs this second model on
    # the same image in a dedicated FIFO worker thread and appends one JSONL
    # record per request to shadow_log_path. The HTTP response is produced by
    # the primary model only and is unchanged (scripts/v5/test_shadow_service.py).
    shadow_backend: str = ""            # "" = off; else a name in backends.AVAILABLE
    shadow_weights_path: str = ""       # required when shadow_backend is set
    shadow_model_version: str = ""      # "" = basename of shadow_weights_path
    shadow_imgsz: int = 1280
    shadow_device: str = ""             # "" = same as `device`
    shadow_conf_floor: float = 0.05
    shadow_conf_single: float = 0.30    # shadow AlertEngine single-frame threshold
    shadow_conf_seq: float = 0.15       # shadow AlertEngine sequence-hit threshold
    shadow_log_path: str = "data/v5/shadow/shadow_log.jsonl"
    shadow_queue_max: int = 256         # requests waiting for the shadow model; beyond -> dropped (counted)
    # dino3-specific shadow overrides; "" / None = auto (same rules as dino_* above)
    shadow_dino_backbone_dir: str = ""
    shadow_lora_r: int | None = None
    shadow_lora_alpha: float | None = None
    shadow_lora_merge: bool = False
    shadow_det_width: int = 0
    shadow_det_height: int = 0

    # FAST / ACCURATE modes (iteration 5b) -- ALL optional. File default mode is
    # "fast" = exactly today's single primary backend and response contract.
    # ACCURATE runs every model in `accurate_models` in parallel threads on the
    # same full-frame image and returns per-model evidence + fused boxes + one
    # final alert (src/serve/modes.py). /detect form field `mode` overrides the
    # default per request; /healthz lists both modes' model versions.
    mode: str = "fast"                  # default mode: "fast" | "accurate"
    # ';'-separated component specs:  name=backend:weights[,key=value...]
    #   keys: version conf_single conf_seq conf_floor imgsz device
    #         backbone_dir lora_r lora_alpha lora_merge width height tile_size
    #   backend omitted -> yolo. Thresholds omitted -> this file's FAST values.
    # e.g. "v4a_s2=yolo:models/trained/expV4A_ctrl_s2/seg05/best.pt,version=expV4A_ctrl_s2@seg05;"
    #      "dino3_b4=dino3:models/v3/runs/v4B4_lora_L/seg05/best.pt,conf_single=0.93,conf_seq=0.82"
    accurate_models: str = ""           # "" = ACCURATE mode not configured (requests -> 422)
    accurate_fusion: str = "or"         # "or" | "wbf" | "learned"; a recipe file overrides this
                                        # default unless SMOKE_ACCURATE_FUSION is set explicitly
    accurate_recipe_path: str = "models/v5b/fusion/accurate_recipe.json"  # loaded if present
    accurate_fusion_iou: float = 0.5    # cross-model box matching IoU (or / builtin wbf); recipe iou_thr overrides unless set
    # target neg-image-FP levels (pooled OOF keys fpX.XX in the recipe's thresholds_oof)
    # that set the fused CALIBRATED alert thresholds when the recipe is applied
    accurate_fused_fp_single: float = 0.05
    accurate_fused_fp_seq: float = 0.10
    accurate_device: str = ""           # "" = same as `device` (per-spec device= wins)

    # demo UI (iteration 5b; GET /demo). Replays go through /detect with
    # source=historical_replay; nothing here changes /detect semantics.
    demo_sdis_sequences: str = ""       # ','-separated seq_ids from data/v3/sequences/sequences_val.json; "" = auto short list
    demo_boreal_clips: str = "heinola_103"   # ','-separated clip dirs under data/v4/sequences/boreal_frames (heinola = aerial VAL)
    demo_runs_dir: str = "data/v5b/demo/runs"  # saved replay runs (scripts/v5b/demo_replay.py) listed by /demo/runs

    # server
    max_upload_mb: int = 25
    host: str = "127.0.0.1"         # private by default
    port: int = 8000


settings = Settings()
