"""TTS audio generator using edge-tts for poses missing local audio files."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from typing import Any

import edge_tts

from src.config_loader import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)


async def _synthesize_one(
    text: str,
    out_path: Path,
    voice: str,
    rate: str,
    pitch: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    communicator = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
    )
    await communicator.save(str(out_path))


async def generate_missing_audios_async(
    force: bool = False,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Generate TTS audio files for triggerable poses that lack audio (async)."""
    cfg = load_config(config_path)
    tts_cfg = cfg.get("tts") or {}
    voice = str(tts_cfg.get("voice") or "zh-CN-XiaoyiNeural")
    rate = str(tts_cfg.get("rate") or "+10%")
    pitch = str(tts_cfg.get("pitch") or "+8Hz")

    results: list[dict[str, Any]] = []
    for pose in cfg.get("poses") or []:
        pose_id = str(pose.get("id", ""))
        if not pose.get("triggerable", False):
            continue

        speech = str(pose.get("speech") or "").strip()
        audio_rel = str(pose.get("audio") or "").strip()
        if not speech or not audio_rel:
            continue

        audio_path = PROJECT_ROOT / audio_rel
        if audio_path.is_file() and audio_path.stat().st_size > 0 and not force:
            results.append(
                {
                    "pose_id": pose_id,
                    "path": str(audio_path),
                    "audio_url": "/" + audio_rel.lstrip("/").replace("\\", "/"),
                    "status": "skipped",
                    "reason": "already_exists",
                }
            )
            continue

        try:
            await _synthesize_one(
                text=speech,
                out_path=audio_path,
                voice=voice,
                rate=rate,
                pitch=pitch,
            )
            results.append(
                {
                    "pose_id": pose_id,
                    "path": str(audio_path),
                    "audio_url": "/" + audio_rel.lstrip("/").replace("\\", "/"),
                    "status": "generated",
                }
            )
            logger.info("Generated TTS audio for %s -> %s", pose_id, audio_path)
        except Exception as exc:
            logger.warning("Failed to generate TTS audio for %s: %s", pose_id, exc)
            results.append(
                {
                    "pose_id": pose_id,
                    "path": str(audio_path),
                    "audio_url": "",
                    "status": "error",
                    "error": str(exc),
                }
            )

    return results


def generate_missing_audios(
    force: bool = False,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Generate TTS audio for triggerable poses with non-empty speech.

    Existing MP3/WAV files are never overwritten unless force=True.
    Safe to call from both synchronous CLI scripts and active asyncio event loops.
    """
    coro = generate_missing_audios_async(force=force, config_path=config_path)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Already inside a running event loop: run in an isolated worker thread
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate missing Nailong pose TTS audio files via edge-tts."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing audio files instead of skipping them.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    results = generate_missing_audios(force=args.force)
    for item in results:
        print(f"{item['pose_id']:>18} | {item['status']:<9} | {item['path']}")


if __name__ == "__main__":
    main()
