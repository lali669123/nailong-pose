"""Debounced pose recognition state machine with sliding-window probability smoothing."""

from __future__ import annotations

from collections import deque
import time
from typing import Any


class PoseStateMachine:
    """State machine managing IDLE -> CHARGING -> TRIGGERED (pulse) -> COOLDOWN -> IDLE.

    Features:
    - Fixed-length sliding window probability smoothing.
    - Confidence threshold gating and idle-label filtering.
    - Configurable hold-to-charge duration (hold_seconds).
    - Transient dropout tolerance (grace_seconds) while charging.
    - Single-frame TRIGGERED pulse followed by cooldown suppression.
    """

    def __init__(
        self,
        hold_seconds: float = 0.45,
        cooldown_seconds: float = 2.5,
        conf_threshold: float = 0.72,
        grace_seconds: float = 0.15,
        window_size: int = 5,
        idle_label: str = "idle",
    ) -> None:
        self.hold_seconds = max(0.0, float(hold_seconds))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.conf_threshold = float(conf_threshold)
        self.grace_seconds = max(0.0, float(grace_seconds))
        self.window_size = max(1, int(window_size))
        self.idle_label = str(idle_label)

        self._window: deque[dict[str, float]] = deque(maxlen=self.window_size)
        self._state: str = "IDLE"
        self._candidate: str | None = None
        self._charge_start: float | None = None
        self._miss_since: float | None = None
        self._cooldown_until: float | None = None
        self._last_triggered: str | None = None

    def reset(self) -> None:
        """Reset smoothing buffer and return cleanly to IDLE."""
        self._window.clear()
        self._state = "IDLE"
        self._candidate = None
        self._charge_start = None
        self._miss_since = None
        self._cooldown_until = None
        self._last_triggered = None

    def _push_and_smooth(
        self,
        pred_label: str | None,
        confidence: float,
        probs: dict[str, float] | None,
    ) -> tuple[str | None, float]:
        """Push current frame probabilities into the sliding window and compute smoothed top-1."""
        if probs is not None:
            frame_probs = {str(k): max(0.0, float(v)) for k, v in probs.items()}
        elif pred_label is not None:
            clamped = max(0.0, min(1.0, float(confidence)))
            frame_probs = {str(pred_label): clamped}
        else:
            frame_probs = {}

        self._window.append(frame_probs)

        totals: dict[str, float] = {}
        for entry in self._window:
            for label, prob in entry.items():
                totals[label] = totals.get(label, 0.0) + prob

        if not totals:
            return None, 0.0

        n = float(len(self._window))
        best_label = max(totals, key=lambda k: totals[k])
        best_conf = totals[best_label] / n
        return best_label, best_conf

    def _make_result(
        self,
        state: str,
        candidate: str | None,
        confidence: float,
        charge_progress: float,
        cooldown_remaining: float,
        triggered_pose: str | None,
    ) -> dict[str, Any]:
        return {
            "state": state,
            "candidate": candidate,
            "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
            "charge_progress": round(max(0.0, min(1.0, float(charge_progress))), 4),
            "cooldown_remaining": round(max(0.0, float(cooldown_remaining)), 3),
            "triggered_pose": triggered_pose,
        }

    def _trigger(self, pose_id: str, confidence: float, now: float) -> dict[str, Any]:
        self._state = "COOLDOWN"
        self._cooldown_until = now + self.cooldown_seconds
        self._last_triggered = pose_id
        self._candidate = None
        self._charge_start = None
        self._miss_since = None
        return self._make_result(
            state="TRIGGERED",
            candidate=pose_id,
            confidence=confidence,
            charge_progress=1.0,
            cooldown_remaining=self.cooldown_seconds,
            triggered_pose=pose_id,
        )

    def update(
        self,
        pred_label: str | None,
        confidence: float,
        probs: dict[str, float] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Advance state machine by one frame and return the state snapshot."""
        ts = time.monotonic() if now is None else float(now)
        smooth_label, smooth_conf = self._push_and_smooth(pred_label, confidence, probs)

        is_active_pose = (
            smooth_label is not None
            and smooth_label != self.idle_label
            and smooth_conf >= self.conf_threshold
        )

        # 1. COOLDOWN handling
        if self._state == "COOLDOWN":
            remaining = (self._cooldown_until or ts) - ts
            if remaining > 0.0:
                return self._make_result(
                    state="COOLDOWN",
                    candidate=self._last_triggered,
                    confidence=smooth_conf if (smooth_label and smooth_label != self.idle_label) else 0.0,
                    charge_progress=0.0,
                    cooldown_remaining=remaining,
                    triggered_pose=None,
                )
            # Cooldown expired -> return to IDLE and evaluate current frame
            self._state = "IDLE"
            self._cooldown_until = None
            self._last_triggered = None

        # 2. IDLE handling
        if self._state == "IDLE":
            if is_active_pose and smooth_label is not None:
                if self.hold_seconds <= 0.0:
                    return self._trigger(smooth_label, smooth_conf, ts)
                self._state = "CHARGING"
                self._candidate = smooth_label
                self._charge_start = ts
                self._miss_since = None
                return self._make_result(
                    state="CHARGING",
                    candidate=self._candidate,
                    confidence=smooth_conf,
                    charge_progress=0.0,
                    cooldown_remaining=0.0,
                    triggered_pose=None,
                )

            reported_conf = smooth_conf if (smooth_label and smooth_label != self.idle_label) else 0.0
            return self._make_result(
                state="IDLE",
                candidate=None,
                confidence=reported_conf,
                charge_progress=0.0,
                cooldown_remaining=0.0,
                triggered_pose=None,
            )

        # 3. CHARGING handling
        if is_active_pose and smooth_label == self._candidate:
            self._miss_since = None
            elapsed = ts - (self._charge_start if self._charge_start is not None else ts)
            if self.hold_seconds <= 0.0 or elapsed >= self.hold_seconds:
                return self._trigger(self._candidate, smooth_conf, ts)
            progress = elapsed / self.hold_seconds
            return self._make_result(
                state="CHARGING",
                candidate=self._candidate,
                confidence=smooth_conf,
                charge_progress=progress,
                cooldown_remaining=0.0,
                triggered_pose=None,
            )

        if is_active_pose and smooth_label is not None and smooth_label != self._candidate:
            # Switched cleanly to another confident pose -> restart charge for new candidate
            if self.hold_seconds <= 0.0:
                return self._trigger(smooth_label, smooth_conf, ts)
            self._candidate = smooth_label
            self._charge_start = ts
            self._miss_since = None
            return self._make_result(
                state="CHARGING",
                candidate=self._candidate,
                confidence=smooth_conf,
                charge_progress=0.0,
                cooldown_remaining=0.0,
                triggered_pose=None,
            )

        # Not currently active while CHARGING -> apply transient grace window
        if self._miss_since is None:
            self._miss_since = ts

        miss_elapsed = ts - self._miss_since
        if miss_elapsed <= self.grace_seconds:
            charge_start = self._charge_start if self._charge_start is not None else ts
            frozen_elapsed = max(0.0, self._miss_since - charge_start)
            frozen_progress = (
                min(0.99, frozen_elapsed / self.hold_seconds)
                if self.hold_seconds > 0.0
                else 0.0
            )
            return self._make_result(
                state="CHARGING",
                candidate=self._candidate,
                confidence=smooth_conf if (smooth_label and smooth_label != self.idle_label) else 0.0,
                charge_progress=frozen_progress,
                cooldown_remaining=0.0,
                triggered_pose=None,
            )

        # Grace window exceeded -> fall back to IDLE
        self._state = "IDLE"
        self._candidate = None
        self._charge_start = None
        self._miss_since = None
        reported_conf = smooth_conf if (smooth_label and smooth_label != self.idle_label) else 0.0
        return self._make_result(
            state="IDLE",
            candidate=None,
            confidence=reported_conf,
            charge_progress=0.0,
            cooldown_remaining=0.0,
            triggered_pose=None,
        )
