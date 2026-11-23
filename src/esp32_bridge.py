"""Non-blocking HTTP audio trigger bridge for optional ESP32 hardware playback."""

from __future__ import annotations

import logging
import threading
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)


class Esp32AudioBridge:
    """Dispatches fire-and-forget HTTP GET triggers to an ESP32 speaker on a daemon thread."""

    def __init__(
        self,
        enabled: bool,
        host: str,
        port: int = 80,
        timeout: float = 0.5,
    ) -> None:
        self.enabled = bool(enabled)
        self.host = str(host or "").strip()
        self.port = int(port)
        self.timeout = max(0.05, float(timeout))

    def build_url(self, pose_id: str) -> str:
        """Build the ESP32 play endpoint URL for the given pose_id."""
        safe_id = quote(str(pose_id), safe="")
        return f"http://{self.host}:{self.port}/play?id={safe_id}"

    def _send_request(self, url: str, pose_id: str) -> None:
        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code >= 400:
                logger.warning(
                    "ESP32 audio trigger for '%s' returned HTTP %s (%s)",
                    pose_id,
                    resp.status_code,
                    url,
                )
        except Exception as exc:
            logger.warning("ESP32 audio trigger failed for '%s' (%s): %s", pose_id, url, exc)

    def play_async(self, pose_id: str) -> bool:
        """Send an asynchronous HTTP GET trigger if enabled. Never raises or blocks."""
        if not self.enabled or not self.host or not pose_id:
            return False

        url = self.build_url(pose_id)
        try:
            worker = threading.Thread(
                target=self._send_request,
                args=(url, str(pose_id)),
                name=f"esp32-play-{pose_id}",
                daemon=True,
            )
            worker.start()
            return True
        except Exception as exc:
            logger.warning("Failed to spawn ESP32 worker thread for '%s': %s", pose_id, exc)
            return False
