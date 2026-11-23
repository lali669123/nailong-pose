"""FastAPI backend server for Nailong Pose Studio (MJPEG stream, 20Hz WS, REST APIs)."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import csv
from datetime import datetime
import logging
import os
from pathlib import Path
import random
import threading
import time
from typing import Any

from src.config_loader import (
    PROJECT_ROOT,
    RUNTIME_ROOT,
    ensure_runtime_dirs,
    load_chinese_font,
    load_config,
)

# Ensure runtime directories and .cache env vars before any third-party imports / static mounts
ensure_runtime_dirs()

import base64
import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from src.capture import compose_match_photo
from src.esp32_bridge import Esp32AudioBridge
from src.features import FEATURE_DIM, SKELETON_EDGES, extract_features, get_feature_names
from src.state_machine import PoseStateMachine

logger = logging.getLogger(__name__)

WEB_DIR = PROJECT_ROOT / "web"
ASSETS_DIR = PROJECT_ROOT / "assets"
CAPTURES_DIR = RUNTIME_ROOT / "captures"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLES_PATH = RUNTIME_ROOT / "data" / "samples.csv"
MODEL_PATH = MODELS_DIR / "pose_clf.joblib"
CONFUSION_IMG_PATH = MODELS_DIR / "eval_confusion.png"


class CollectStartRequest(BaseModel):
    pose_id: str = Field(..., min_length=1)
    frames: int = Field(default=80, ge=1, le=2000)
    countdown: float = Field(default=3.0, ge=0.0, le=30.0)


class GameStartRequest(BaseModel):
    countdown: float = Field(default=3.0, ge=0.0, le=15.0)


class SaveCertificateRequest(BaseModel):
    data_url: str = Field(..., min_length=32, max_length=15_000_000)


def _ensure_ascii_mediapipe_root(mp_module: Any) -> str | None:
    """Work around MediaPipe C++ non-ASCII (Chinese) path bug on Windows.

    MediaPipe's SolutionBase computes ``root_path`` from ``solution_base.__file__``
    and passes it to C++ file APIs that fail on non-ASCII Windows paths (e.g.
    ``<app_dir>\\奶龙家人鉴定\\_internal\\mediapipe\\...``). When non-ASCII characters
    are present, redirect ``solution_base.__file__`` to an ASCII-only mirror of
    ``mediapipe/modules`` (preferring 8.3 short path, else ``ProgramData``/``Public``).
    """
    import shutil
    from mediapipe.python import solution_base

    sb_file = os.path.abspath(solution_base.__file__)
    orig_root = os.sep.join(sb_file.split(os.sep)[:-3])
    if orig_root.isascii():
        return orig_root

    # 1. Try Windows 8.3 short path first (zero copy when enabled on drive)
    if os.name == "nt":
        try:
            import ctypes

            buf = ctypes.create_unicode_buffer(1024)
            n = ctypes.windll.kernel32.GetShortPathNameW(orig_root, buf, len(buf))
            if 0 < n < len(buf) and buf.value.isascii():
                short_root = buf.value.rstrip("\\/")
                if os.path.isdir(os.path.join(short_root, "mediapipe", "modules")):
                    solution_base.__file__ = os.path.join(
                        short_root, "mediapipe", "python", "solution_base.py"
                    )
                    logger.info("Redirected MediaPipe root via 8.3 short path: %s", short_root)
                    return short_root
        except Exception as exc:
            logger.debug("GetShortPathNameW unavailable: %s", exc)

    # 2. Mirror mediapipe/modules (~15MB) into an always-ASCII writable directory
    src_modules = Path(orig_root) / "mediapipe" / "modules"
    if not src_modules.is_dir():
        return None

    candidates = [
        Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "NailongPose" / "mp_root",
        Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "NailongPose" / "mp_root",
    ]
    for ascii_root in candidates:
        if not str(ascii_root).isascii():
            continue
        try:
            dst_modules = ascii_root / "mediapipe" / "modules"
            sentinel = dst_modules / "pose_landmark" / "pose_landmark_cpu.binarypb"
            if not sentinel.is_file():
                dst_modules.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_modules, dst_modules, dirs_exist_ok=True)
            solution_base.__file__ = str(ascii_root / "mediapipe" / "python" / "solution_base.py")
            logger.info("Redirected MediaPipe root to ASCII cache: %s", ascii_root)
            return str(ascii_root)
        except Exception as exc:
            logger.warning("Failed preparing ASCII MediaPipe cache at %s: %s", ascii_root, exc)
    return None


def _render_placeholder_frame(
    width: int,
    height: int,
    headline: str = "摄像头未连接 / 等待设备就绪",
    subline: str = "请检查摄像头权限或断开占用程序，系统将自动重连（预览模式仍可正常试玩）",
) -> np.ndarray:
    """Create a crisp dark-themed placeholder BGR frame when no camera is available."""
    img = Image.new("RGB", (width, height), (15, 23, 42))
    draw = ImageDraw.Draw(img)

    # Subtle cyber grid
    for x in range(0, width, 40):
        draw.line([(x, 0), (x, height)], fill=(23, 33, 54), width=1)
    for y in range(0, height, 40):
        draw.line([(0, y), (width, y)], fill=(23, 33, 54), width=1)

    # Center card
    pad_x, pad_y = max(24, width // 10), max(24, height // 5)
    draw.rounded_rectangle(
        [pad_x, pad_y, width - pad_x, height - pad_y],
        radius=20,
        fill=(30, 41, 59),
        outline=(250, 204, 21),
        width=3,
    )

    title_font = load_chinese_font(max(18, width // 26), bold=True)
    sub_font = load_chinese_font(max(13, width // 42), bold=False)

    draw.text(
        (width // 2, height // 2 - 22),
        headline,
        fill=(250, 204, 21),
        font=title_font,
        anchor="mm",
    )
    draw.text(
        (width // 2, height // 2 + 24),
        subline,
        fill=(203, 213, 225),
        font=sub_font,
        anchor="mm",
    )

    rgb = np.array(img, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _encode_jpeg(frame_bgr: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        return b""
    return buf.tobytes()


class StudioEngine:
    """Thread-safe controller owning camera capture, pose inference, collection, and model state."""

    def __init__(
        self,
        samples_path: Path | None = None,
        model_path: Path | None = None,
        captures_dir: Path | None = None,
        config_path: Path | None = None,
        confusion_img_path: Path | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._csv_lock = threading.Lock()
        self._train_lock = threading.Lock()

        self.samples_path: Path = samples_path or SAMPLES_PATH
        self.model_path: Path = model_path or MODEL_PATH
        self.captures_dir: Path = captures_dir or CAPTURES_DIR
        self.config_path: Path | None = config_path
        self.confusion_img_path: Path = confusion_img_path or CONFUSION_IMG_PATH

        self.config: dict[str, Any] = load_config(self.config_path)
        self.poses_by_id: dict[str, dict[str, Any]] = {
            str(p["id"]): p for p in (self.config.get("poses") or [])
        }

        app_cfg = self.config.get("app") or {}
        self.release_mode: bool = str(app_cfg.get("mode", "studio")).strip().lower() == "release"
        self.camera_index: int = int(app_cfg.get("camera_index", 0))
        self.frame_width: int = int(app_cfg.get("frame_width", 640))
        self.frame_height: int = int(app_cfg.get("frame_height", 480))
        self.flip_horizontal: bool = bool(app_cfg.get("flip_horizontal", True))
        self._hidden_pity_counts: dict[str, int] = {}

        rec_cfg = self.config.get("recognition") or {}
        self.state_machine = PoseStateMachine(
            hold_seconds=float(rec_cfg.get("hold_seconds", 0.45)),
            cooldown_seconds=float(rec_cfg.get("cooldown_seconds", 2.5)),
            conf_threshold=float(rec_cfg.get("confidence_threshold", 0.72)),
            grace_seconds=float(rec_cfg.get("grace_seconds", 0.15)),
            window_size=int(rec_cfg.get("smoothing_window", 5)),
            idle_label=str(rec_cfg.get("idle_label", "idle")),
        )

        esp_cfg = self.config.get("esp32") or {}
        self.esp32 = Esp32AudioBridge(
            enabled=bool(esp_cfg.get("enabled", False)),
            host=str(esp_cfg.get("host", "192.168.1.100")),
            port=int(esp_cfg.get("port", 80)),
            timeout=float(esp_cfg.get("timeout_seconds", 0.5)),
        )

        self._model_bundle: dict[str, Any] | None = None
        self.reload_model()

        # Frame & real-time telemetry state
        self._loading_bgr: np.ndarray = _render_placeholder_frame(
            self.frame_width,
            self.frame_height,
            headline="正在连接摄像头，请稍候...",
            subline="首次启动正在加载 AI 骨骼引擎，画面即将就绪",
        )
        self._placeholder_bgr: np.ndarray = _render_placeholder_frame(
            self.frame_width,
            self.frame_height,
        )
        self._latest_frame_bgr: np.ndarray = self._loading_bgr.copy()
        self._latest_jpeg: bytes = _encode_jpeg(self._loading_bgr)
        self._camera_available: bool = False

        self._fps: float = 0.0
        self._mode: str = "infer"
        self._person_detected: bool = False
        self._skeleton: list[dict[str, Any]] = []
        self._sm_state: dict[str, Any] = {
            "state": "IDLE",
            "candidate": None,
            "confidence": 0.0,
            "charge_progress": 0.0,
            "cooldown_remaining": 0.0,
            "triggered_pose": None,
        }

        # Game session state (stopped -> countdown -> running)
        self._game: dict[str, Any] = {
            "active": False,
            "phase": "stopped",
            "countdown_left": 0.0,
        }
        self._game_started_at: float = 0.0
        self._game_countdown_total: float = 3.0

        # Collection state
        self._collect: dict[str, Any] = {
            "active": False,
            "phase": "idle",
            "pose_id": None,
            "countdown_left": 0.0,
            "recorded": 0,
            "target": 80,
        }
        self._collect_session_id: str | None = None
        self._collect_started_at: float = 0.0
        self._collect_countdown_total: float = 3.0
        self._session_counter: int = 0

        # Triggered event sequence for lossless WS fan-out
        self._event_seq: int = 0
        self._last_event: dict[str, Any] | None = None

        # Background thread control
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    @property
    def current_event_seq(self) -> int:
        with self._lock:
            return self._event_seq

    def set_release_mode(self, release: bool) -> None:
        """Toggle studio vs release mode at runtime (e.g. from CLI flags)."""
        with self._lock:
            self.release_mode = bool(release)
            app_cfg = dict(self.config.get("app") or {})
            app_cfg["mode"] = "release" if self.release_mode else "studio"
            self.config["app"] = app_cfg

    def _resolve_hidden_variant(self, pose_id: str) -> tuple[str, bool, bool]:
        """Roll configured hidden variant (with optional pity guarantee) for pose_id.

        Must be called while holding ``self._lock``.
        Returns ``(final_pose_id, rolled_hidden, pity_triggered)``.
        """
        pose_info = self.poses_by_id.get(pose_id) or {}
        hv = pose_info.get("hidden_variant")
        if not isinstance(hv, dict):
            return pose_id, False, False

        target_id = str(hv.get("pose_id") or "").strip()
        if not target_id or target_id not in self.poses_by_id:
            return pose_id, False, False

        prob = max(0.0, min(1.0, float(hv.get("probability", 0.15))))
        pity_after = max(0, int(hv.get("pity_after", 0)))
        misses = self._hidden_pity_counts.get(pose_id, 0)

        if pity_after > 0 and (misses + 1) >= pity_after:
            self._hidden_pity_counts[pose_id] = 0
            return target_id, True, True

        if random.random() < prob:
            self._hidden_pity_counts[pose_id] = 0
            return target_id, True, False

        self._hidden_pity_counts[pose_id] = misses + 1
        return pose_id, False, False

    def refresh_config(self) -> dict[str, Any]:
        """Reload poses.yaml metadata (e.g. after TTS generation)."""
        cfg = load_config(self.config_path)
        with self._lock:
            app_cfg = dict(cfg.get("app") or {})
            app_cfg["mode"] = "release" if self.release_mode else "studio"
            cfg["app"] = app_cfg
            self.config = cfg
            self.poses_by_id = {str(p["id"]): p for p in (cfg.get("poses") or [])}
        return cfg

    def reload_model(self) -> bool:
        """Load or hot-reload the serialized pose classifier bundle from disk."""
        if not self.model_path.is_file() or self.model_path.stat().st_size == 0:
            with self._lock:
                self._model_bundle = None
            return False

        try:
            bundle = joblib.load(self.model_path)
            if not isinstance(bundle, dict) or "model" not in bundle or "classes" not in bundle:
                raise ValueError("Invalid model bundle format")
            if int(bundle.get("feature_dim", FEATURE_DIM)) != FEATURE_DIM:
                raise ValueError(
                    f"Model feature_dim={bundle.get('feature_dim')} != expected {FEATURE_DIM}"
                )
            with self._lock:
                self._model_bundle = bundle
                self.state_machine.reset()
            logger.info(
                "Loaded pose classifier '%s' (%d classes) from %s",
                bundle.get("model_name"),
                len(bundle.get("classes") or []),
                self.model_path,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load pose model %s: %s", self.model_path, exc)
            with self._lock:
                self._model_bundle = None
            return False

    def get_model_info(self) -> dict[str, Any]:
        """Build model summary for GET /api/config."""
        with self._lock:
            bundle = self._model_bundle

        confusion_url = (
            "/models/eval_confusion.png"
            if self.confusion_img_path.is_file() and self.confusion_img_path.stat().st_size > 0
            else ""
        )
        if bundle is None:
            return {
                "loaded": False,
                "model_name": None,
                "classes": [],
                "metrics": None,
                "trained_at": None,
                "confusion_matrix_url": confusion_url,
            }

        metrics = bundle.get("metrics") or {}
        return {
            "loaded": True,
            "model_name": bundle.get("model_name"),
            "classes": list(bundle.get("classes") or []),
            "metrics": metrics,
            "best_accuracy": metrics.get("best_accuracy"),
            "best_macro_f1": metrics.get("best_macro_f1"),
            "cv_strategy": metrics.get("cv_strategy"),
            "comparisons": metrics.get("comparisons"),
            "trained_at": bundle.get("trained_at"),
            "confusion_matrix_url": confusion_url,
        }

    def get_sample_counts(self) -> dict[str, int]:
        """Count valid recorded rows per pose label in samples.csv."""
        counts: dict[str, int] = {}
        with self._csv_lock:
            if not self.samples_path.is_file() or self.samples_path.stat().st_size == 0:
                return counts
            try:
                with open(self.samples_path, "r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        label = str(row.get("label") or "").strip()
                        if label:
                            counts[label] = counts.get(label, 0) + 1
            except Exception as exc:
                logger.warning("Failed reading sample counts from %s: %s", self.samples_path, exc)
        return counts

    def append_sample_row(
        self,
        feat_vec: np.ndarray,
        pose_id: str,
        session_id: str,
    ) -> None:
        """Append one extracted 46-dim feature vector row to samples.csv."""
        feature_names = get_feature_names()
        header = ["timestamp", "session_id", "label", *feature_names]
        ts_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
        row = [ts_iso, session_id, pose_id, *[f"{float(v):.6f}" for v in feat_vec.tolist()]]

        with self._csv_lock:
            self.samples_path.parent.mkdir(parents=True, exist_ok=True)
            need_header = (
                not self.samples_path.is_file() or self.samples_path.stat().st_size == 0
            )
            with open(self.samples_path, "a", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                if need_header:
                    writer.writerow(header)
                writer.writerow(row)

    def delete_samples(self, pose_id: str) -> tuple[int, dict[str, int]]:
        """Remove all rows matching pose_id from samples.csv atomically."""
        deleted = 0
        with self._csv_lock:
            if not self.samples_path.is_file() or self.samples_path.stat().st_size == 0:
                return 0, {}

            kept_rows: list[list[str]] = []
            fieldnames: list[str] = []
            with open(self.samples_path, "r", encoding="utf-8", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader, None)
                if header:
                    fieldnames = header
                    label_idx = header.index("label") if "label" in header else 2
                    for row in reader:
                        if not row:
                            continue
                        row_label = row[label_idx].strip() if label_idx < len(row) else ""
                        if row_label == pose_id:
                            deleted += 1
                        else:
                            kept_rows.append(row)

            if fieldnames:
                tmp_file = self.samples_path.with_suffix(".csv.tmp")
                with open(tmp_file, "w", encoding="utf-8", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(fieldnames)
                    writer.writerows(kept_rows)
                os.replace(tmp_file, self.samples_path)

        return deleted, self.get_sample_counts()

    def start_collect(
        self,
        pose_id: str,
        frames: int = 80,
        countdown: float = 3.0,
    ) -> dict[str, Any]:
        """Enter countdown/recording collection mode for a configured pose_id."""
        with self._lock:
            if self.release_mode:
                raise PermissionError("当前为精简发布版（Release 模式），已关闭采集与训练功能")
            if pose_id not in self.poses_by_id:
                raise KeyError(f"未知姿势 ID：{pose_id}")
            pose_info = self.poses_by_id[pose_id]
            if not pose_info.get("trainable", True):
                parent_id = pose_info.get("parent_pose_id") or ""
                parent_name = (self.poses_by_id.get(parent_id) or {}).get("name") or parent_id
                raise ValueError(
                    f"「{pose_info.get('name', pose_id)}」为隐藏款，无需单独录制（触发「{parent_name}」时会自动按概率抽取）"
                )

            self._session_counter += 1
            session_id = f"{pose_id}_{int(time.time())}_{self._session_counter}"
            now = time.monotonic()

            self._mode = "collect"
            self._game = {
                "active": False,
                "phase": "stopped",
                "countdown_left": 0.0,
            }
            self.state_machine.reset()
            self._sm_state = {
                "state": "IDLE",
                "candidate": None,
                "confidence": 0.0,
                "charge_progress": 0.0,
                "cooldown_remaining": 0.0,
                "triggered_pose": None,
            }
            self._collect_session_id = session_id
            self._collect_started_at = now
            self._collect_countdown_total = max(0.0, float(countdown))
            phase = "countdown" if self._collect_countdown_total > 0.0 else "recording"
            self._collect = {
                "active": True,
                "phase": phase,
                "pose_id": pose_id,
                "countdown_left": round(self._collect_countdown_total, 2),
                "recorded": 0,
                "target": int(frames),
            }
            return {
                "status": "started",
                "pose_id": pose_id,
                "session_id": session_id,
                "frames": int(frames),
                "countdown": self._collect_countdown_total,
            }

    def stop_collect(self) -> dict[str, Any]:
        """Abort collection mode and return to inference mode."""
        with self._lock:
            recorded = int(self._collect.get("recorded", 0))
            pose_id = self._collect.get("pose_id")
            self._mode = "infer"
            self._collect_session_id = None
            self._collect = {
                "active": False,
                "phase": "idle",
                "pose_id": None,
                "countdown_left": 0.0,
                "recorded": 0,
                "target": int(self._collect.get("target", 80)),
            }
            self.state_machine.reset()

        counts = self.get_sample_counts()
        return {
            "status": "stopped",
            "pose_id": pose_id,
            "recorded": recorded,
            "sample_counts": counts,
            "sample_total": sum(counts.values()),
        }

    def start_game(self, countdown: float = 3.0, now: float | None = None) -> dict[str, Any]:
        """Start recognition game with an optional preparation countdown."""
        with self._lock:
            if self._collect.get("active"):
                raise RuntimeError("正在录制动作数据，请先停止录制后再开始游戏")
            ts = time.monotonic() if now is None else float(now)
            cd = max(0.0, float(countdown))
            self._game_started_at = ts
            self._game_countdown_total = cd
            self.state_machine.reset()
            self._sm_state = {
                "state": "IDLE",
                "candidate": None,
                "confidence": 0.0,
                "charge_progress": 0.0,
                "cooldown_remaining": 0.0,
                "triggered_pose": None,
            }
            if cd > 0.0:
                self._game = {
                    "active": True,
                    "phase": "countdown",
                    "countdown_left": round(cd, 2),
                }
            else:
                self._game = {
                    "active": True,
                    "phase": "running",
                    "countdown_left": 0.0,
                }
            return {
                "status": "started",
                "game": dict(self._game),
            }

    def stop_game(self) -> dict[str, Any]:
        """Stop recognition game and return to safe standby preview mode."""
        with self._lock:
            self._game = {
                "active": False,
                "phase": "stopped",
                "countdown_left": 0.0,
            }
            self.state_machine.reset()
            self._sm_state = {
                "state": "IDLE",
                "candidate": None,
                "confidence": 0.0,
                "charge_progress": 0.0,
                "cooldown_remaining": 0.0,
                "triggered_pose": None,
            }
            return {
                "status": "stopped",
                "game": dict(self._game),
            }

    def get_game_state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._game)

    def trigger_pose(
        self,
        pose_id: str,
        confidence: float = 0.98,
        frame_bgr: np.ndarray | None = None,
        allow_hidden_roll: bool = True,
    ) -> dict[str, Any]:
        """Fire a pose trigger event: roll hidden variant, compose poster, notify ESP32, and publish to WS clients."""
        with self._lock:
            if pose_id not in self.poses_by_id:
                raise KeyError(f"未知姿势 ID：{pose_id}")

            rolled_hidden = False
            pity_triggered = False
            final_pose_id = pose_id
            if allow_hidden_roll:
                final_pose_id, rolled_hidden, pity_triggered = self._resolve_hidden_variant(pose_id)

            pose_copy = dict(self.poses_by_id[final_pose_id])
            snapshot_frame = (
                frame_bgr.copy()
                if frame_bgr is not None
                else self._latest_frame_bgr.copy()
            )

        capture_url = ""
        try:
            filename = compose_match_photo(
                frame_bgr=snapshot_frame,
                pose_info=pose_copy,
                confidence=confidence,
                output_dir=self.captures_dir,
            )
            capture_url = f"/captures/{filename}"
        except Exception as exc:
            logger.warning("Failed to compose match photo for '%s': %s", final_pose_id, exc)

        self.esp32.play_async(final_pose_id)

        is_hidden = bool(pose_copy.get("is_hidden") or final_pose_id == "10_rainbow")
        event = {
            "pose_id": str(pose_copy["id"]),
            "name": str(pose_copy.get("name", final_pose_id)),
            "subtitle": str(pose_copy.get("subtitle") or ""),
            "speech": str(pose_copy.get("speech") or ""),
            "image_url": str(pose_copy.get("image_url") or ""),
            "audio_url": str(pose_copy.get("audio_url") or ""),
            "color": str(pose_copy.get("color") or "#FACC15"),
            "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
            "capture_url": capture_url,
            "esp32_enabled": bool(self.esp32.enabled),
            "is_hidden": is_hidden,
            "rolled_from": pose_id if rolled_hidden else "",
            "pity_triggered": bool(pity_triggered),
        }

        with self._lock:
            self._event_seq += 1
            self._last_event = event

        return event

    def list_captures(self, limit: int = 24) -> list[dict[str, str]]:
        """List up to `limit` most recent captured match posters in reverse chronological order."""
        if not self.captures_dir.is_dir():
            return []

        items: list[tuple[float, str, Path]] = []
        for entry in self.captures_dir.iterdir():
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            items.append((mtime, entry.name, entry))

        items.sort(key=lambda item: (item[0], item[1]), reverse=True)
        result: list[dict[str, str]] = []
        for mtime, name, _ in items[:limit]:
            created_at = datetime.fromtimestamp(mtime).astimezone().isoformat(timespec="seconds")
            result.append(
                {
                    "filename": name,
                    "url": f"/captures/{name}",
                    "created_at": created_at,
                }
            )
        return result

    def get_latest_jpeg(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    def get_ws_snapshot(self, last_seen_seq: int) -> tuple[dict[str, Any], int]:
        """Build a 20Hz WebSocket status payload and advance per-client event sequence."""
        with self._lock:
            new_seq = last_seen_seq
            triggered_event: dict[str, Any] | None = None
            state_str = str(self._sm_state.get("state", "IDLE"))

            if self._event_seq > last_seen_seq and self._last_event is not None:
                triggered_event = dict(self._last_event)
                new_seq = self._event_seq
                state_str = "TRIGGERED"

            payload = {
                "ts": round(time.time(), 3),
                "fps": round(self._fps, 1),
                "mode": self._mode,
                "has_model": self._model_bundle is not None,
                "person_detected": self._person_detected,
                "skeleton": list(self._skeleton),
                "state": state_str,
                "candidate": self._sm_state.get("candidate"),
                "confidence": float(self._sm_state.get("confidence", 0.0)),
                "charge_progress": float(self._sm_state.get("charge_progress", 0.0)),
                "cooldown_remaining": float(self._sm_state.get("cooldown_remaining", 0.0)),
                "triggered_event": triggered_event,
                "collect": dict(self._collect),
                "game": dict(self._game),
            }
            return payload, new_seq

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        pose_detector: Any | None = None,
        now: float | None = None,
        already_mirrored: bool = False,
    ) -> None:
        """Process one camera frame: mirror, pose extraction, collect/infer, and JPEG encode."""
        ts = time.monotonic() if now is None else float(now)

        if self.flip_horizontal and not already_mirrored:
            frame_bgr = cv2.flip(frame_bgr, 1)

        feat_vec: np.ndarray | None = None
        skeleton: list[dict[str, Any]] = []

        if pose_detector is not None:
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = pose_detector.process(frame_rgb)
                if results and results.pose_landmarks:
                    feat_vec, skeleton = extract_features(results.pose_landmarks.landmark)
            except Exception as exc:
                logger.debug("Pose processing error: %s", exc)

        person_detected = feat_vec is not None
        jpeg_bytes = _encode_jpeg(frame_bgr)

        trigger_request: tuple[str, float] | None = None
        sample_to_append: tuple[np.ndarray, str, str] | None = None

        with self._lock:
            self._latest_frame_bgr = frame_bgr
            if jpeg_bytes:
                self._latest_jpeg = jpeg_bytes
            self._person_detected = person_detected
            self._skeleton = skeleton

            if self._mode == "collect" and self._collect.get("active"):
                elapsed = max(0.0, ts - self._collect_started_at)
                countdown_left = (
                    max(0.0, self._collect_countdown_total - elapsed)
                    if self._collect_countdown_total > 0.0
                    else 0.0
                )
                self._collect["countdown_left"] = round(countdown_left, 2)

                if countdown_left > 0.0:
                    self._collect["phase"] = "countdown"
                else:
                    self._collect["phase"] = "recording"
                    pose_id = str(self._collect.get("pose_id") or "")
                    session_id = str(self._collect_session_id or pose_id)
                    if feat_vec is not None and pose_id:
                        sample_to_append = (feat_vec.copy(), pose_id, session_id)
                        self._collect["recorded"] = int(self._collect["recorded"]) + 1
                        if self._collect["recorded"] >= int(self._collect["target"]):
                            # Completed target frame count -> return to infer mode
                            self._mode = "infer"
                            self._collect_session_id = None
                            self._collect = {
                                "active": False,
                                "phase": "idle",
                                "pose_id": None,
                                "countdown_left": 0.0,
                                "recorded": 0,
                                "target": int(self._collect["target"]),
                            }
                            self.state_machine.reset()
            else:
                # Game countdown / running gate
                game_phase = str(self._game.get("phase", "stopped"))
                if game_phase == "countdown":
                    elapsed = max(0.0, ts - self._game_started_at)
                    cd_left = max(0.0, self._game_countdown_total - elapsed)
                    self._game["countdown_left"] = round(cd_left, 2)
                    if cd_left <= 0.0:
                        self._game["phase"] = "running"
                        self._game["active"] = True
                        self._game["countdown_left"] = 0.0
                        self.state_machine.reset()
                        game_phase = "running"

                # Inference mode (only active when game phase is 'running')
                bundle = self._model_bundle
                if game_phase == "running" and bundle is not None and feat_vec is not None:
                    try:
                        model = bundle["model"]
                        classes = bundle["classes"]
                        proba = model.predict_proba(feat_vec.reshape(1, -1))[0]
                        probs_dict = {
                            str(cls): float(p) for cls, p in zip(classes, proba)
                        }
                        top_label = max(probs_dict, key=probs_dict.get)
                        top_conf = probs_dict[top_label]
                        sm_out = self.state_machine.update(
                            top_label,
                            top_conf,
                            probs=probs_dict,
                            now=ts,
                        )
                    except Exception as exc:
                        logger.warning("Model inference error: %s", exc)
                        sm_out = self.state_machine.update(None, 0.0, probs=None, now=ts)
                else:
                    if game_phase != "running":
                        self.state_machine.reset()
                        sm_out = {
                            "state": "IDLE",
                            "candidate": None,
                            "confidence": 0.0,
                            "charge_progress": 0.0,
                            "cooldown_remaining": 0.0,
                            "triggered_pose": None,
                        }
                    else:
                        sm_out = self.state_machine.update(None, 0.0, probs=None, now=ts)

                self._sm_state = sm_out
                if game_phase == "running" and sm_out.get("triggered_pose"):
                    trigger_request = (
                        str(sm_out["triggered_pose"]),
                        float(sm_out.get("confidence", 0.0)),
                    )

        # Perform disk I/O outside self._lock so WS & MJPEG never stall
        if sample_to_append is not None:
            vec, pid, sid = sample_to_append
            try:
                self.append_sample_row(vec, pid, sid)
            except Exception as exc:
                logger.warning("Failed to append sample row: %s", exc)

        if trigger_request is not None:
            pid, conf = trigger_request
            if pid in self.poses_by_id:
                self.trigger_pose(pid, confidence=conf, frame_bgr=frame_bgr, allow_hidden_roll=True)

    def _open_camera(self) -> cv2.VideoCapture | None:
        if os.environ.get("NAILONG_DISABLE_CAMERA") == "1":
            return None

        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        for backend in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(self.camera_index, backend)
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                    ok, _ = cap.read()
                    if ok:
                        logger.info(
                            "Opened camera index=%d (backend=%s, %dx%d)",
                            self.camera_index,
                            backend,
                            self.frame_width,
                            self.frame_height,
                        )
                        return cap
                if cap is not None:
                    cap.release()
            except Exception as exc:
                logger.debug("Camera open attempt (backend=%s) failed: %s", backend, exc)
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        return None

    def _worker_loop(self) -> None:
        # 1. Open camera FIRST so live video appears immediately (< 1s)
        cap: cv2.VideoCapture | None = self._open_camera()

        # 2. Initialize MediaPipe Pose asynchronously in background thread so video never stalls
        pose_holder: dict[str, Any] = {"detector": None}
        pose_lock = threading.Lock()

        def _init_mediapipe() -> None:
            if os.environ.get("NAILONG_DISABLE_MEDIAPIPE") == "1":
                return
            try:
                import mediapipe as mp

                _ensure_ascii_mediapipe_root(mp)
                detector = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                with pose_lock:
                    if self._stop_event.is_set():
                        detector.close()
                    else:
                        pose_holder["detector"] = detector
                logger.info("MediaPipe Pose initialized successfully.")
            except Exception as exc:
                logger.warning("MediaPipe Pose unavailable, running degraded mode: %s", exc)

        mp_thread = threading.Thread(target=_init_mediapipe, name="nailong-mp-init", daemon=True)
        mp_thread.start()

        last_retry_ts = time.monotonic()
        frame_counter = 0
        fps_window_start = time.monotonic()
        consecutive_failures = 0

        try:
            while not self._stop_event.is_set():
                loop_now = time.monotonic()

                if cap is None:
                    if loop_now - last_retry_ts >= 3.0:
                        last_retry_ts = loop_now
                        cap = self._open_camera()

                frame_bgr: np.ndarray | None = None
                if cap is not None:
                    ok, raw_frame = cap.read()
                    if ok and raw_frame is not None and raw_frame.size > 0:
                        frame_bgr = raw_frame
                        consecutive_failures = 0
                        self._camera_available = True
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= 15:
                            logger.warning("Camera read failed repeatedly; falling back to placeholder.")
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = None
                            self._camera_available = False
                            last_retry_ts = loop_now

                if frame_bgr is None:
                    self._camera_available = False
                    # Placeholder frame is already properly oriented (do not mirror text)
                    self.process_frame(
                        self._placeholder_bgr.copy(),
                        pose_detector=None,
                        now=loop_now,
                        already_mirrored=True,
                    )
                    self._stop_event.wait(0.05)
                else:
                    with pose_lock:
                        active_detector = pose_holder["detector"]
                    self.process_frame(
                        frame_bgr,
                        pose_detector=active_detector,
                        now=loop_now,
                        already_mirrored=False,
                    )

                frame_counter += 1
                elapsed_fps = time.monotonic() - fps_window_start
                if elapsed_fps >= 0.5:
                    with self._lock:
                        self._fps = frame_counter / elapsed_fps
                    frame_counter = 0
                    fps_window_start = time.monotonic()
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            with pose_lock:
                det = pose_holder["detector"]
                pose_holder["detector"] = None
            if det is not None:
                try:
                    det.close()
                except Exception:
                    pass

    def start(self) -> None:
        """Start the background camera + pose worker thread (idempotent)."""
        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="nailong-studio-worker",
                daemon=True,
            )
            self._worker_thread.start()

    def stop(self) -> None:
        """Signal the background worker to stop and wait briefly for clean teardown."""
        self._stop_event.set()
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.5)
        self._worker_thread = None


studio = StudioEngine()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    studio.start()
    try:
        yield
    finally:
        studio.stop()


app = FastAPI(
    title=str((studio.config.get("app") or {}).get("title") or "Nailong Pose Studio"),
    lifespan=lifespan,
)

# Mount static asset directories
WEB_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")
app.mount("/models", StaticFiles(directory=str(MODELS_DIR)), name="models")


@app.get("/", response_model=None)
def index() -> FileResponse | HTMLResponse:
    index_html = WEB_DIR / "index.html"
    if index_html.is_file():
        return FileResponse(str(index_html))
    return HTMLResponse(
        "<h1>Nailong Pose Studio</h1><p>Backend is running. Frontend index.html not found yet.</p>"
    )


@app.get("/video_feed")
async def video_feed() -> StreamingResponse:
    async def frame_generator():
        while True:
            jpeg = studio.get_latest_jpeg()
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.033)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _ensure_studio_mode() -> None:
    if studio.release_mode:
        raise HTTPException(
            status_code=403,
            detail="当前为精简发布版（Release 模式），已关闭采集与训练功能",
        )


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    counts = studio.get_sample_counts()
    cfg = studio.config
    app_cfg = dict(cfg.get("app") or {})
    app_cfg["mode"] = "release" if studio.release_mode else "studio"
    app_cfg["release_mode"] = bool(studio.release_mode)
    return {
        "app": app_cfg,
        "recognition": cfg.get("recognition") or {},
        "esp32": cfg.get("esp32") or {},
        "poses": cfg.get("poses") or [],
        "skeleton_edges": [list(edge) for edge in SKELETON_EDGES],
        "model_info": studio.get_model_info(),
        "sample_counts": counts,
        "sample_total": sum(counts.values()),
        "game": studio.get_game_state(),
    }


@app.post("/api/game/start")
def api_game_start(req: GameStartRequest | None = None) -> dict[str, Any]:
    countdown = 3.0 if req is None else req.countdown
    try:
        return studio.start_game(countdown=countdown)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/game/stop")
def api_game_stop() -> dict[str, Any]:
    return studio.stop_game()


@app.get("/api/captures")
def api_captures() -> list[dict[str, str]]:
    return studio.list_captures(limit=24)


@app.post("/api/captures/save_certificate")
def api_save_certificate(req: SaveCertificateRequest) -> dict[str, str]:
    raw = req.data_url.strip()
    if "," in raw:
        _, b64_part = raw.split(",", 1)
    else:
        b64_part = raw
    try:
        img_bytes = base64.b64decode(b64_part, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="证书图片数据无效") from exc
    if not img_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(status_code=400, detail="仅支持保存 PNG 格式证书")

    studio.captures_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_purity_certificate.png"
    dest = studio.captures_dir / filename
    dest.write_bytes(img_bytes)
    return {
        "filename": filename,
        "url": f"/captures/{filename}",
    }


@app.post("/api/collect/start")
def api_collect_start(req: CollectStartRequest) -> dict[str, Any]:
    _ensure_studio_mode()
    try:
        return studio.start_collect(
            pose_id=req.pose_id,
            frames=req.frames,
            countdown=req.countdown,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        detail = exc.args[0] if exc.args else str(exc)
        raise HTTPException(status_code=404, detail=detail) from exc


@app.post("/api/collect/stop")
def api_collect_stop() -> dict[str, Any]:
    _ensure_studio_mode()
    return studio.stop_collect()


@app.delete("/api/samples/{pose_id}")
def api_delete_samples(pose_id: str) -> dict[str, Any]:
    _ensure_studio_mode()
    deleted, counts = studio.delete_samples(pose_id)
    return {
        "pose_id": pose_id,
        "deleted": deleted,
        "sample_counts": counts,
        "sample_total": sum(counts.values()),
    }


@app.post("/api/train")
def api_train() -> dict[str, Any]:
    _ensure_studio_mode()
    with studio._lock:
        if studio._collect.get("active"):
            raise HTTPException(status_code=409, detail="正在录制动作数据，请先停止录制后再训练模型")

    if not studio._train_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="模型正在训练中，请稍候")

    try:
        from src.train import train_and_evaluate

        with studio._csv_lock:
            metrics = train_and_evaluate(
                samples_path=studio.samples_path,
                model_path=studio.model_path,
                plot_path=studio.confusion_img_path,
                config_path=studio.config_path,
            )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during model training")
        raise HTTPException(status_code=500, detail=f"模型训练失败：{exc}") from exc
    finally:
        studio._train_lock.release()

    studio.reload_model()
    return metrics


@app.post("/api/tts/generate")
def api_tts_generate() -> dict[str, Any]:
    _ensure_studio_mode()
    from src.tts_gen import generate_missing_audios

    results = generate_missing_audios(force=False, config_path=studio.config_path)
    studio.refresh_config()
    generated = sum(1 for r in results if r.get("status") == "generated")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    errors = sum(1 for r in results if r.get("status") == "error")
    return {
        "results": results,
        "generated": generated,
        "skipped": skipped,
        "errors": errors,
    }


@app.post("/api/preview/{pose_id}")
def api_preview(pose_id: str, roll: bool = False) -> dict[str, Any]:
    try:
        return studio.trigger_pose(
            pose_id=pose_id,
            confidence=0.98,
            allow_hidden_roll=bool(roll),
        )
    except KeyError as exc:
        detail = exc.args[0] if exc.args else str(exc)
        raise HTTPException(status_code=404, detail=detail) from exc


@app.websocket("/ws")
async def ws_status(websocket: WebSocket) -> None:
    await websocket.accept()
    last_seen_seq = studio.current_event_seq
    try:
        while True:
            payload, last_seen_seq = studio.get_ws_snapshot(last_seen_seq)
            await websocket.send_json(payload)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.debug("WebSocket connection closed: %s", exc)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Nailong Pose Studio server")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--release",
        action="store_true",
        help="Start in simplified Release mode (hides training/collection UI & locks training APIs)",
    )
    mode_group.add_argument(
        "--studio",
        action="store_true",
        help="Start in Studio mode (with data collection & model training)",
    )
    parser.add_argument("--host", type=str, default=None, help="Bind host")
    parser.add_argument("--port", type=int, default=None, help="Bind port")
    args = parser.parse_args()

    if args.release:
        studio.set_release_mode(True)
    elif args.studio:
        studio.set_release_mode(False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app_cfg = studio.config.get("app") or {}
    host = args.host or str(app_cfg.get("host", "127.0.0.1"))
    port = args.port or int(app_cfg.get("port", 8000))
    logger.info("Starting Nailong Pose Server in '%s' mode on %s:%d", app_cfg.get("mode", "studio"), host, port)
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
