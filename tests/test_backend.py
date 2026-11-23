"""Comprehensive unit & integration tests for Nailong Pose Studio backend modules."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any

from fastapi import HTTPException
import numpy as np
from PIL import Image
import pytest

from src.capture import POSTER_HEIGHT, POSTER_WIDTH, compose_match_photo
from src.config_loader import PROJECT_ROOT, ensure_runtime_dirs, load_config
from src.esp32_bridge import Esp32AudioBridge
from src.features import FEATURE_DIM, SKELETON_EDGES, get_feature_names
import src.server as server_mod
from src.state_machine import PoseStateMachine
from src.tts_gen import generate_missing_audios


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_synthetic_landmarks(arm_raise: float = 0.0, jitter: float = 0.0, seed: int = 0) -> list[dict[str, float]]:
    """Build 33 MediaPipe-compatible landmarks with high visibility and configurable arm height."""
    rng = np.random.default_rng(seed)
    lms = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99} for _ in range(33)]
    coords = {
        0: (0.50, 0.20),   # nose
        11: (0.42, 0.35),  # left shoulder
        12: (0.58, 0.35),  # right shoulder
        13: (0.36, 0.48 - arm_raise * 0.25),  # left elbow
        14: (0.64, 0.48 - arm_raise * 0.25),  # right elbow
        15: (0.32, 0.60 - arm_raise * 0.45),  # left wrist
        16: (0.68, 0.60 - arm_raise * 0.45),  # right wrist
        19: (0.31, 0.62 - arm_raise * 0.45),  # left index
        20: (0.69, 0.62 - arm_raise * 0.45),  # right index
        23: (0.44, 0.60),  # left hip
        24: (0.56, 0.60),  # right hip
        25: (0.44, 0.78),  # left knee
        26: (0.56, 0.78),  # right knee
        27: (0.44, 0.93),  # left ankle
        28: (0.56, 0.93),  # right ankle
        31: (0.43, 0.96),  # left foot
        32: (0.57, 0.96),  # right foot
    }
    for idx, (x, y) in coords.items():
        dx = float(rng.normal(0.0, jitter)) if jitter > 0 else 0.0
        dy = float(rng.normal(0.0, jitter)) if jitter > 0 else 0.0
        lms[idx] = {
            "x": x + dx,
            "y": y + dy,
            "z": 0.0,
            "visibility": 0.98,
        }
    return lms


class _FakePoseDetector:
    """Minimal stub mimicking mp.solutions.pose.Pose.process(rgb)."""

    def __init__(self, landmarks: list[dict[str, float]] | None) -> None:
        self.landmarks = landmarks

    def process(self, _rgb: np.ndarray) -> Any:
        if self.landmarks is None:
            return SimpleNamespace(pose_landmarks=None)
        return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=self.landmarks))


# ---------------------------------------------------------------------------
# 1. config_loader tests
# ---------------------------------------------------------------------------


def test_ensure_runtime_dirs_pins_cache_to_project() -> None:
    dirs = ensure_runtime_dirs()
    cache_root = PROJECT_ROOT / ".cache"

    for key in ("mpl", "tmp", "data", "models", "captures", "assets_images", "assets_audio"):
        assert dirs[key].is_dir()

    assert Path(os.environ["MPLCONFIGDIR"]).resolve() == (cache_root / "mpl").resolve()
    assert Path(os.environ["TMP"]).resolve() == (cache_root / "tmp").resolve()
    assert Path(os.environ["TEMP"]).resolve() == (cache_root / "tmp").resolve()


def test_load_config_preserves_real_images_and_generates_idle_placeholder() -> None:
    real_files = [
        PROJECT_ROOT / "assets" / "images" / name
        for name in (
            "01_kungfu.jpg",
            "05_think.jpg",
            "08_balance.jpg",
            "10_rainbow.png",
        )
    ]
    hashes_before = {p: _sha256(p) for p in real_files}

    cfg = load_config()
    poses = cfg["poses"]
    assert len(poses) == 11

    by_id = {p["id"]: p for p in poses}

    # All 10 real images remain untouched & marked has_real_image=True
    for p, digest in hashes_before.items():
        assert _sha256(p) == digest

    for pid in ("01_kungfu", "05_think", "07_point_laugh", "08_balance", "10_rainbow"):
        assert by_id[pid]["has_real_image"] is True
        assert by_id[pid]["image_url"].startswith("/assets/images/")
        assert by_id[pid]["has_audio"] is True

    # Mixed .wav and .mp3 audio paths both work cleanly
    assert by_id["07_point_laugh"]["audio_url"] == "/assets/audio/07_point_laugh.wav"
    assert by_id["01_kungfu"]["audio_url"] == "/assets/audio/01_kungfu.mp3"
    assert by_id["10_rainbow"]["audio_url"] == "/assets/audio/10_rainbow.mp3"

    # idle.png auto-generated (600x600) and marked placeholder (has_real_image=False)
    idle_img = PROJECT_ROOT / "assets" / "images" / "idle.png"
    assert idle_img.is_file()
    with Image.open(idle_img) as im:
        assert im.size == (600, 600)
    assert by_id["idle"]["has_real_image"] is False
    assert by_id["idle"]["image_url"] == "/assets/images/idle.png"
    assert by_id["idle"]["has_audio"] is False
    assert by_id["idle"]["audio_url"] == ""


# ---------------------------------------------------------------------------
# 2. state_machine tests
# ---------------------------------------------------------------------------


def test_state_machine_full_lifecycle_and_single_frame_pulse() -> None:
    sm = PoseStateMachine(
        hold_seconds=0.45,
        cooldown_seconds=2.5,
        conf_threshold=0.72,
        grace_seconds=0.15,
        window_size=5,
        idle_label="idle",
    )

    # Idle stays IDLE
    r0 = sm.update("idle", 0.95, probs={"idle": 0.95, "01_kungfu": 0.05}, now=10.0)
    assert r0["state"] == "IDLE"
    assert r0["candidate"] is None
    assert r0["triggered_pose"] is None

    # Fill window with confident 01_kungfu so smoothed prob >= 0.72 -> starts CHARGING
    t = 10.10
    r = sm.update("01_kungfu", 0.95, probs={"idle": 0.05, "01_kungfu": 0.95}, now=t)
    # After 1 idle + 1 kungfu, avg kungfu = 0.50 (< 0.72), still IDLE
    assert r["state"] == "IDLE"

    # Push more kungfu frames until CHARGING starts
    charge_start_t = None
    for i in range(1, 10):
        t += 0.05
        r = sm.update("01_kungfu", 0.96, probs={"idle": 0.04, "01_kungfu": 0.96}, now=t)
        if r["state"] == "CHARGING":
            charge_start_t = t
            break

    assert charge_start_t is not None
    assert r["candidate"] == "01_kungfu"
    assert r["charge_progress"] == 0.0
    assert r["triggered_pose"] is None

    # Mid-charge (0.25s elapsed < 0.45s)
    r_mid = sm.update(
        "01_kungfu", 0.96, probs={"idle": 0.04, "01_kungfu": 0.96}, now=charge_start_t + 0.25
    )
    assert r_mid["state"] == "CHARGING"
    assert 0.5 < r_mid["charge_progress"] < 0.65
    assert r_mid["triggered_pose"] is None

    # Reach hold_seconds (0.45s) -> single-frame TRIGGERED pulse
    trigger_t = charge_start_t + 0.46
    r_trig = sm.update(
        "01_kungfu", 0.97, probs={"idle": 0.03, "01_kungfu": 0.97}, now=trigger_t
    )
    assert r_trig["state"] == "TRIGGERED"
    assert r_trig["triggered_pose"] == "01_kungfu"
    assert r_trig["charge_progress"] == 1.0
    assert r_trig["cooldown_remaining"] == pytest.approx(2.5, abs=0.01)

    # Very next frame -> COOLDOWN (triggered_pose MUST be None!)
    r_cd1 = sm.update(
        "01_kungfu", 0.97, probs={"idle": 0.03, "01_kungfu": 0.97}, now=trigger_t + 0.05
    )
    assert r_cd1["state"] == "COOLDOWN"
    assert r_cd1["triggered_pose"] is None
    assert r_cd1["cooldown_remaining"] == pytest.approx(2.45, abs=0.02)

    # Still in cooldown at +2.0s
    r_cd2 = sm.update(
        "01_kungfu", 0.97, probs={"idle": 0.03, "01_kungfu": 0.97}, now=trigger_t + 2.0
    )
    assert r_cd2["state"] == "COOLDOWN"
    assert r_cd2["triggered_pose"] is None

    # After 2.5s cooldown expires -> can charge again (or IDLE if idle pose)
    r_after = sm.update(
        "idle", 0.99, probs={"idle": 0.99, "01_kungfu": 0.01}, now=trigger_t + 2.55
    )
    assert r_after["state"] in ("IDLE", "CHARGING")
    # Push enough idle frames to clear window -> IDLE
    for k in range(5):
        r_idle = sm.update("idle", 0.99, probs={"idle": 0.99}, now=trigger_t + 2.6 + 0.05 * k)
    assert r_idle["state"] == "IDLE"
    assert r_idle["cooldown_remaining"] == 0.0


def test_state_machine_grace_period_and_candidate_switch() -> None:
    sm = PoseStateMachine(
        hold_seconds=0.45,
        cooldown_seconds=2.5,
        conf_threshold=0.72,
        grace_seconds=0.15,
        window_size=1,
        idle_label="idle",
    )

    # Start charging pose A
    assert sm.update("01_kungfu", 0.9, now=1.0)["state"] == "CHARGING"
    assert sm.update("01_kungfu", 0.9, now=1.2)["charge_progress"] > 0.4

    # Transient dropout within 0.15s grace -> stays CHARGING, does NOT reset
    r_drop1 = sm.update(None, 0.0, now=1.25)
    assert r_drop1["state"] == "CHARGING"
    assert r_drop1["candidate"] == "01_kungfu"

    r_drop2 = sm.update(None, 0.0, now=1.35)  # 0.10s since miss_since (1.25)
    assert r_drop2["state"] == "CHARGING"
    assert r_drop2["triggered_pose"] is None

    # Recover before grace expires -> resumes charging and triggers cleanly!
    r_rec = sm.update("01_kungfu", 0.92, now=1.38)
    assert r_rec["state"] == "CHARGING"

    # Drop for > 0.15s -> falls back to IDLE
    sm.update(None, 0.0, now=1.40)  # miss_since = 1.40
    r_expired = sm.update(None, 0.0, now=1.56)  # 0.16s > 0.15s
    assert r_expired["state"] == "IDLE"
    assert r_expired["candidate"] is None
    assert r_expired["charge_progress"] == 0.0

    # Candidate switch resets charge progress cleanly
    sm.update("01_kungfu", 0.92, now=2.0)
    sm.update("01_kungfu", 0.92, now=2.25)
    r_switch = sm.update("02_arms_crossed", 0.95, now=2.30)
    assert r_switch["state"] == "CHARGING"
    assert r_switch["candidate"] == "02_arms_crossed"
    assert r_switch["charge_progress"] == 0.0


def test_state_machine_sliding_window_suppresses_single_frame_spike() -> None:
    sm = PoseStateMachine(conf_threshold=0.72, window_size=5, idle_label="idle")
    for i in range(4):
        sm.update("idle", 0.95, probs={"idle": 0.95, "06_shocked": 0.05}, now=float(i) * 0.05)

    # Single-frame false spike should be smoothed down to ~0.23 and stay IDLE
    r_spike = sm.update("06_shocked", 0.95, probs={"idle": 0.05, "06_shocked": 0.95}, now=0.25)
    assert r_spike["state"] == "IDLE"
    assert r_spike["triggered_pose"] is None


# ---------------------------------------------------------------------------
# 3. tts_gen tests
# ---------------------------------------------------------------------------


def test_tts_gen_skips_existing_real_mp3_and_wav() -> None:
    audio_dir = PROJECT_ROOT / "assets" / "audio"
    sample_files = [
        audio_dir / "01_kungfu.mp3",
        audio_dir / "07_point_laugh.wav",
        audio_dir / "10_rainbow.mp3",
    ]
    before_hashes = {p: _sha256(p) for p in sample_files}

    results = generate_missing_audios(force=False)
    assert len(results) == 10
    assert all(item["status"] == "skipped" for item in results)

    for p, digest in before_hashes.items():
        assert _sha256(p) == digest


# ---------------------------------------------------------------------------
# 4. esp32_bridge tests
# ---------------------------------------------------------------------------


def test_esp32_bridge_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def _fake_get(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr("src.esp32_bridge.requests.get", _fake_get)
    bridge = Esp32AudioBridge(enabled=False, host="192.168.1.100", port=80, timeout=0.2)
    assert bridge.play_async("01_kungfu") is False
    assert called is False


def test_esp32_bridge_enabled_async_and_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_urls: list[str] = []
    done = threading.Event()

    def _fake_get(url: str, timeout: float = 0.5) -> Any:
        recorded_urls.append(url)
        done.set()
        raise ConnectionError("Simulated unreachable ESP32")

    monkeypatch.setattr("src.esp32_bridge.requests.get", _fake_get)
    bridge = Esp32AudioBridge(enabled=True, host="192.168.1.100", port=8080, timeout=0.3)

    # Must return True immediately and NEVER raise despite ConnectionError inside thread
    assert bridge.play_async("01_kungfu") is True
    assert done.wait(timeout=2.0)
    assert recorded_urls == ["http://192.168.1.100:8080/play?id=01_kungfu"]


# ---------------------------------------------------------------------------
# 5. capture tests (4K + RGBA transparent PNG)
# ---------------------------------------------------------------------------


def test_compose_match_photo_handles_4k_and_rgba_png_without_black_background(
    tmp_path: Path,
) -> None:
    cfg = load_config()
    by_id = {p["id"]: p for p in cfg["poses"]}

    # Synthetic camera frame (640x480 BGR)
    frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_bgr[:, :] = (90, 60, 40)

    # A) Real 4K JPEG (05_think.jpg: 3840x3616)
    fname_4k = compose_match_photo(frame_bgr, by_id["05_think"], 0.94, tmp_path)
    poster_4k = tmp_path / fname_4k
    assert poster_4k.is_file()
    with Image.open(poster_4k) as im:
        assert im.size == (POSTER_WIDTH, POSTER_HEIGHT)
        assert im.mode == "RGB"

    # B) Real 10_rainbow.png
    fname_rainbow = compose_match_photo(frame_bgr, by_id["10_rainbow"], 0.99, tmp_path)
    poster_rainbow = tmp_path / fname_rainbow
    assert poster_rainbow.is_file()
    with Image.open(poster_rainbow) as im:
        assert im.size == (POSTER_WIDTH, POSTER_HEIGHT)

    # C) Explicit transparent RGBA PNG -> transparent corners MUST be warm bright, NOT black!
    rgba_path = tmp_path / "transparent_nailong.png"
    rgba_img = Image.new("RGBA", (600, 600), (0, 0, 0, 0))
    # Draw an opaque yellow square only in the center (200..400), leaving borders 100% transparent
    for y in range(200, 400):
        for x in range(200, 400):
            rgba_img.putpixel((x, y), (250, 204, 21, 255))
    rgba_img.save(rgba_path, format="PNG")

    custom_pose = {
        "id": "10_rainbow_rgba",
        "name": "透明通道彩虹奶蛙",
        "subtitle": "RGBA 透明底测试",
        "speech": "绝不能出现黑底！",
        "image_path": str(rgba_path),
        "color": "#C084FC",
    }
    fname_rgba = compose_match_photo(frame_bgr, custom_pose, 0.96, tmp_path)
    poster_rgba = tmp_path / fname_rgba
    assert poster_rgba.is_file()

    with Image.open(poster_rgba) as im:
        assert im.size == (POSTER_WIDTH, POSTER_HEIGHT)
        # Inspect a pixel inside the right panel's transparent border area
        # Right panel spans x in [664..1246], y in [74..518]; check bottom-right inner corner
        sample_px = im.getpixel((1200, 480))
        # Warm cream backdrop should be bright (well above black)
        assert sample_px[0] > 210 and sample_px[1] > 200 and sample_px[2] > 180


# ---------------------------------------------------------------------------
# 6. server / StudioEngine / REST & WS integration tests
# ---------------------------------------------------------------------------


def test_import_server_does_not_start_camera_thread() -> None:
    assert server_mod.studio._worker_thread is None


def test_studio_collect_delete_train_preview_and_ws_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_samples = tmp_path / "samples.csv"
    isolated_model = tmp_path / "pose_clf.joblib"
    isolated_captures = tmp_path / "captures"
    isolated_confusion = tmp_path / "eval_confusion.png"

    engine = server_mod.StudioEngine(
        samples_path=isolated_samples,
        model_path=isolated_model,
        captures_dir=isolated_captures,
        confusion_img_path=isolated_confusion,
    )
    monkeypatch.setattr(server_mod, "studio", engine)

    # 1. GET /api/config
    cfg_resp = server_mod.api_config()
    assert cfg_resp["skeleton_edges"] == [list(e) for e in SKELETON_EDGES]
    assert cfg_resp["model_info"]["loaded"] is False
    assert cfg_resp["sample_total"] == 0
    assert len(cfg_resp["poses"]) == 11

    # 2. POST /api/train with no samples -> HTTP 400 with Chinese message
    with pytest.raises(HTTPException) as exc_info:
        server_mod.api_train()
    assert exc_info.value.status_code == 400
    assert len(str(exc_info.value.detail)) > 0

    # 3. Collect 8 frames each for 'idle' and '01_kungfu' across distinct sessions
    dummy_frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    for pose_id, arm_raise in (("idle", 0.0), ("01_kungfu", 1.0)):
        for session_idx in range(2):
            start_res = server_mod.api_collect_start(
                server_mod.CollectStartRequest(pose_id=pose_id, frames=6, countdown=0.0)
            )
            assert start_res["status"] == "started"
            assert start_res["session_id"].startswith(pose_id)

            base_now = time.monotonic()
            for f_idx in range(6):
                lms = _make_synthetic_landmarks(
                    arm_raise=arm_raise,
                    jitter=0.005,
                    seed=session_idx * 100 + f_idx,
                )
                engine.process_frame(
                    dummy_frame.copy(),
                    pose_detector=_FakePoseDetector(lms),
                    now=base_now + f_idx * 0.05,
                )

            # Should auto-return to infer mode after hitting target=6
            snap, _ = engine.get_ws_snapshot(engine.current_event_seq)
            assert snap["mode"] == "infer"
            assert snap["collect"]["active"] is False

    # Verify CSV schema strictly matches SPEC: timestamp,session_id,label,<46 features>
    with open(isolated_samples, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["timestamp", "session_id", "label", *get_feature_names()]
    assert len(rows[0]) == 3 + FEATURE_DIM
    assert len(rows) == 1 + 24  # header + 12 idle + 12 01_kungfu

    # Verify unique session_ids across the 4 sessions
    session_ids = {r[1] for r in rows[1:]}
    assert len(session_ids) == 4

    # 4. POST /api/collect/stop works cleanly
    server_mod.api_collect_start(
        server_mod.CollectStartRequest(pose_id="02_arms_crossed", frames=20, countdown=3.0)
    )
    stop_res = server_mod.api_collect_stop()
    assert stop_res["status"] == "stopped"
    assert stop_res["sample_total"] == 24

    # 5. POST /api/train trains model, saves confusion matrix, and hot-loads model
    train_res = server_mod.api_train()
    assert train_res["best_model"] in ("RandomForest", "MLP")
    assert train_res["best_accuracy"] >= 0.8
    assert isolated_model.is_file()
    assert isolated_confusion.is_file()
    assert engine.get_model_info()["loaded"] is True

    # 6. POST /api/preview/{pose_id} generates capture & pushes single-shot WS triggered_event
    seq_before = engine.current_event_seq
    preview_res = server_mod.api_preview("10_rainbow")
    assert preview_res["pose_id"] == "10_rainbow"
    assert preview_res["capture_url"].startswith("/captures/")
    assert preview_res["audio_url"] == "/assets/audio/10_rainbow.mp3"
    assert preview_res["esp32_enabled"] is False

    # WS snapshot delivers triggered_event exactly once
    snap1, seq1 = engine.get_ws_snapshot(seq_before)
    assert snap1["state"] == "TRIGGERED"
    assert snap1["triggered_event"]["pose_id"] == "10_rainbow"
    assert snap1["has_model"] is True

    snap2, seq2 = engine.get_ws_snapshot(seq1)
    assert snap2["triggered_event"] is None
    assert seq2 == seq1

    # 7. GET /api/captures lists the new poster
    captures = server_mod.api_captures()
    assert len(captures) == 1
    assert captures[0]["url"] == preview_res["capture_url"]

    # 8. DELETE /api/samples/{pose_id} deletes only that class
    del_res = server_mod.api_delete_samples("01_kungfu")
    assert del_res["deleted"] == 12
    assert del_res["sample_counts"] == {"idle": 12}
    assert del_res["sample_total"] == 12

    # 9. POST /api/tts/generate skips all existing audios cleanly
    tts_res = server_mod.api_tts_generate()
    assert tts_res["skipped"] == 10
    assert tts_res["errors"] == 0


def test_studio_graceful_camera_degradation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NAILONG_DISABLE_CAMERA", "1")
    monkeypatch.setenv("NAILONG_DISABLE_MEDIAPIPE", "1")

    engine = server_mod.StudioEngine(
        samples_path=tmp_path / "samples.csv",
        model_path=tmp_path / "pose_clf.joblib",
        captures_dir=tmp_path / "captures",
    )
    engine.start()
    try:
        time.sleep(0.15)
        jpeg = engine.get_latest_jpeg()
        # Valid JPEG SOI + EOI markers
        assert jpeg.startswith(b"\xff\xd8")
        assert jpeg.endswith(b"\xff\xd9")
    finally:
        engine.stop()
        assert engine._worker_thread is None


def test_video_feed_yields_mjpeg_chunk() -> None:
    async def _read_first_chunk() -> bytes:
        resp = await server_mod.video_feed()
        chunk = await anext(resp.body_iterator)
        return chunk

    chunk = asyncio.run(_read_first_chunk())
    assert b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" in chunk
    assert b"\xff\xd8" in chunk


def test_hidden_variant_roll_pity_and_train_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_samples = tmp_path / "samples.csv"
    isolated_model = tmp_path / "pose_clf.joblib"
    isolated_captures = tmp_path / "captures"
    isolated_confusion = tmp_path / "eval_confusion.png"

    engine = server_mod.StudioEngine(
        samples_path=isolated_samples,
        model_path=isolated_model,
        captures_dir=isolated_captures,
        confusion_img_path=isolated_confusion,
    )
    monkeypatch.setattr(server_mod, "studio", engine)

    # 1. Attempting to collect hidden variant 10_rainbow must return HTTP 400
    with pytest.raises(HTTPException) as exc_info:
        server_mod.api_collect_start(
            server_mod.CollectStartRequest(pose_id="10_rainbow", frames=10, countdown=0.0)
        )
    assert exc_info.value.status_code == 400

    # 2. Preview defaults to roll=False -> always returns exact 04_belly_laugh
    monkeypatch.setattr(server_mod.random, "random", lambda: 0.0)
    exact_preview = server_mod.api_preview("04_belly_laugh")
    assert exact_preview["pose_id"] == "04_belly_laugh"
    assert exact_preview["is_hidden"] is False
    assert exact_preview["rolled_from"] == ""

    # 3. With roll=True and random()=0.0 (< 0.15) -> hits hidden variant 10_rainbow immediately
    lucky_roll = server_mod.api_preview("04_belly_laugh", roll=True)
    assert lucky_roll["pose_id"] == "10_rainbow"
    assert lucky_roll["is_hidden"] is True
    assert lucky_roll["rolled_from"] == "04_belly_laugh"
    assert lucky_roll["pity_triggered"] is False
    assert lucky_roll["capture_url"].startswith("/captures/")

    # 4. Pity guarantee: force random()=0.99 (> 0.15) -> first 7 rolls normal, 8th guaranteed 10_rainbow
    monkeypatch.setattr(server_mod.random, "random", lambda: 0.99)
    for _ in range(7):
        ev = engine.trigger_pose("04_belly_laugh", confidence=0.95, allow_hidden_roll=True)
        assert ev["pose_id"] == "04_belly_laugh"
        assert ev["pity_triggered"] is False

    pity_ev = engine.trigger_pose("04_belly_laugh", confidence=0.95, allow_hidden_roll=True)
    assert pity_ev["pose_id"] == "10_rainbow"
    assert pity_ev["is_hidden"] is True
    assert pity_ev["rolled_from"] == "04_belly_laugh"
    assert pity_ev["pity_triggered"] is True

    # 5. Even if legacy 10_rainbow rows exist in samples.csv, training automatically excludes them
    rng = np.random.default_rng(7)
    for label, offset in (("idle", 0.0), ("04_belly_laugh", 2.0), ("10_rainbow", 4.0)):
        for i in range(8):
            vec = np.full(FEATURE_DIM, offset, dtype=np.float32) + rng.normal(0.0, 0.05, size=FEATURE_DIM)
            engine.append_sample_row(vec, label, f"{label}_s{i % 2}")

    train_res = server_mod.api_train()
    assert "10_rainbow" not in train_res["classes"]
    assert train_res["excluded_labels"] == ["10_rainbow"]
    assert sorted(train_res["classes"]) == ["04_belly_laugh", "idle"]


def test_release_mode_locks_training_endpoints_and_allows_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NAILONG_MODE", "release")
    engine = server_mod.StudioEngine(
        samples_path=tmp_path / "samples.csv",
        model_path=tmp_path / "pose_clf.joblib",
        captures_dir=tmp_path / "captures",
        confusion_img_path=tmp_path / "eval_confusion.png",
    )
    monkeypatch.setattr(server_mod, "studio", engine)

    cfg_resp = server_mod.api_config()
    assert cfg_resp["app"]["mode"] == "release"
    assert cfg_resp["app"]["release_mode"] is True

    for call in (
        lambda: server_mod.api_collect_start(server_mod.CollectStartRequest(pose_id="01_kungfu")),
        lambda: server_mod.api_collect_stop(),
        lambda: server_mod.api_delete_samples("01_kungfu"),
        lambda: server_mod.api_train(),
        lambda: server_mod.api_tts_generate(),
    ):
        with pytest.raises(HTTPException) as exc_info:
            call()
        assert exc_info.value.status_code == 403

    # Preview & captures continue to work normally in release mode
    ev = server_mod.api_preview("03_rub_belly")
    assert ev["pose_id"] == "03_rub_belly"
    assert ev["capture_url"].startswith("/captures/")
    assert len(server_mod.api_captures()) == 1

    # Game start/stop also remain available in release mode
    g_start = server_mod.api_game_start(server_mod.GameStartRequest(countdown=3.0))
    assert g_start["game"]["phase"] == "countdown"
    g_stop = server_mod.api_game_stop()
    assert g_stop["game"]["phase"] == "stopped"


def test_game_start_countdown_and_stop_gates_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_samples = tmp_path / "samples.csv"
    isolated_model = tmp_path / "pose_clf.joblib"
    isolated_captures = tmp_path / "captures"
    isolated_confusion = tmp_path / "eval_confusion.png"

    engine = server_mod.StudioEngine(
        samples_path=isolated_samples,
        model_path=isolated_model,
        captures_dir=isolated_captures,
        confusion_img_path=isolated_confusion,
    )
    monkeypatch.setattr(server_mod, "studio", engine)

    # Train a quick 2-class model
    dummy_frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    for pose_id, arm_raise in (("idle", 0.0), ("01_kungfu", 1.0)):
        server_mod.api_collect_start(
            server_mod.CollectStartRequest(pose_id=pose_id, frames=8, countdown=0.0)
        )
        t0 = 100.0
        for i in range(8):
            lms = _make_synthetic_landmarks(arm_raise=arm_raise, jitter=0.004, seed=i)
            engine.process_frame(dummy_frame.copy(), pose_detector=_FakePoseDetector(lms), now=t0 + i * 0.05)

    server_mod.api_train()
    assert engine.get_game_state()["phase"] == "stopped"

    # While stopped, feeding kungfu frames MUST NOT charge or trigger
    kungfu_lms = _make_synthetic_landmarks(arm_raise=1.0, jitter=0.0, seed=1)
    seq0 = engine.current_event_seq
    for i in range(12):
        engine.process_frame(dummy_frame.copy(), pose_detector=_FakePoseDetector(kungfu_lms), now=200.0 + i * 0.1)
    snap_stopped, _ = engine.get_ws_snapshot(seq0)
    assert snap_stopped["game"]["phase"] == "stopped"
    assert snap_stopped["state"] == "IDLE"
    assert snap_stopped["triggered_event"] is None

    # Start game with 3s countdown -> during countdown MUST NOT trigger
    start_res = engine.start_game(countdown=3.0, now=300.0)
    assert start_res["game"]["phase"] == "countdown"
    engine.process_frame(dummy_frame.copy(), pose_detector=_FakePoseDetector(kungfu_lms), now=301.5)
    snap_cd, _ = engine.get_ws_snapshot(seq0)
    assert snap_cd["game"]["phase"] == "countdown"
    assert snap_cd["game"]["countdown_left"] == pytest.approx(1.5, abs=0.05)
    assert snap_cd["state"] == "IDLE"

    # Once countdown expires (>= 303.0) -> auto-transitions to running and triggers!
    for i in range(15):
        engine.process_frame(dummy_frame.copy(), pose_detector=_FakePoseDetector(kungfu_lms), now=303.1 + i * 0.08)
    snap_run, _ = engine.get_ws_snapshot(seq0)
    assert snap_run["game"]["phase"] == "running"
    assert snap_run["triggered_event"] is not None
    assert snap_run["triggered_event"]["pose_id"] == "01_kungfu"

    # Stop game resets state cleanly
    stop_res = engine.stop_game()
    assert stop_res["game"]["phase"] == "stopped"
    snap_end, _ = engine.get_ws_snapshot(engine.current_event_seq)
    assert snap_end["state"] == "IDLE"
    assert snap_end["game"]["active"] is False
