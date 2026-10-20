"""Per-device alert state.

Single qualifying frame -> PROVISIONAL warning (never gated on history).
Recent consistent detections -> CONFIRMED (strengthened) alert.
History is advisory only: it can upgrade an alert, never suppress the
single-frame provisional path. These are smoke alerts, not confirmed
wildfire reports.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field


@dataclass
class FrameRecord:
    ts: float               # capture timestamp (epoch seconds, client-supplied)
    max_score: float        # best smoke score in frame (0 if none)
    hit: bool               # max_score >= seq_frame_conf
    phash: object = None    # imagehash of frame, for scene-change detection


@dataclass
class DeviceState:
    frames: deque = field(default_factory=lambda: deque(maxlen=64))
    last_confirmed_ts: float = float("-inf")
    resets: int = 0


class AlertEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self._devices: dict[str, DeviceState] = {}
        self._lock = threading.Lock()

    def _get(self, device_id: str) -> DeviceState:
        if device_id not in self._devices:
            self._devices[device_id] = DeviceState()
        return self._devices[device_id]

    def update(self, device_id: str, ts: float, max_smoke_score: float,
               phash=None) -> dict:
        cfg = self.cfg
        with self._lock:
            st = self._get(device_id)

            reset_reason = None
            if st.frames:
                last = st.frames[-1]
                if ts - last.ts > cfg.stale_after_s:
                    reset_reason = "stale_history"
                elif ts < last.ts:
                    reset_reason = "timestamp_regression"
                elif (phash is not None and last.phash is not None
                      and (phash - last.phash) > cfg.scene_change_hamming):
                    reset_reason = "scene_change"
            if reset_reason:
                st.frames.clear()
                st.resets += 1

            hit = max_smoke_score >= cfg.seq_frame_conf
            st.frames.append(FrameRecord(ts, max_smoke_score, hit, phash))

            window = list(st.frames)[-cfg.seq_window:]
            hits = sum(1 for f in window if f.hit)

            # single-frame provisional path — independent of history length
            provisional = max_smoke_score >= cfg.single_frame_alert_conf
            confirmed = (hits >= cfg.seq_hits_required
                         and len(window) >= cfg.seq_hits_required)

            in_cooldown = (ts - st.last_confirmed_ts) < cfg.cooldown_s
            if confirmed and not in_cooldown:
                st.last_confirmed_ts = ts

            if confirmed:
                state = "confirmed"
                reason = f"{hits}/{len(window)} recent frames >= {cfg.seq_frame_conf}"
            elif provisional:
                state = "provisional"
                reason = f"single frame score {max_smoke_score:.3f} >= {cfg.single_frame_alert_conf}"
            elif hit:
                state = "watch"
                reason = f"weak detection {max_smoke_score:.3f} (below single-frame threshold)"
            else:
                state = "none"
                reason = "no qualifying detection"

            return {
                "state": state,
                "reason": reason,
                "window_hits": hits,
                "window_size": len(window),
                "history_reset": reset_reason,
                "cooldown_active": bool(confirmed and in_cooldown),
                "note": ("Smoke alert from a detector, not an independently "
                         "confirmed wildfire report. Absence of alert on "
                         "low-quality/missing input does not mean 'safe'."),
            }

    def stats(self) -> dict:
        with self._lock:
            return {d: {"frames_buffered": len(s.frames), "resets": s.resets}
                    for d, s in self._devices.items()}
