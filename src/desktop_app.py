"""Standalone Windows desktop launcher for Nailong Pose Release Edition.

Starts the embedded FastAPI + Uvicorn server in strict Release mode on a free
localhost port, then opens a native Edge WebView2 desktop window (pywebview).
If WebView2 is unavailable on an older PC, gracefully falls back to opening the
default web browser with a small Tkinter control window.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def _configure_frozen_stdio_and_logging() -> Path:
    """Prevent crashes when running under PyInstaller --windowed (stdout/stderr=None)."""
    if getattr(sys, "frozen", False):
        runtime_root = Path(sys.executable).resolve().parent
    else:
        runtime_root = Path(__file__).resolve().parent.parent

    cache_dir = runtime_root / ".cache"
    tmp_dir = cache_dir / "tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["NAILONG_MODE"] = "release"

    log_path = cache_dir / "app.log"
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    return runtime_root


def _find_free_port(preferred: int = 18765) -> int:
    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", candidate))
                return int(s.getsockname()[1])
            except OSError:
                continue
    return 18765


def _wait_server_ready(url: str, timeout: float = 18.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.15)
    return False


def _run_browser_fallback(url: str, title: str) -> None:
    """Fallback when pywebview/WebView2 cannot initialize."""
    webbrowser.open(url)
    try:
        import tkinter as tk

        root = tk.Tk()
        root.title(title)
        root.geometry("420x180")
        root.resizable(False, False)
        root.configure(bg="#0f172a")

        lbl = tk.Label(
            root,
            text="奶龙家人鉴定已在浏览器打开！\n\n保持此窗口开启即可游玩\n关闭本窗口将自动退出程序并释放摄像头",
            fg="#fef08a",
            bg="#0f172a",
            font=("Microsoft YaHei", 11, "bold"),
            justify="center",
        )
        lbl.pack(expand=True, padx=16, pady=(18, 8))

        btn_frame = tk.Frame(root, bg="#0f172a")
        btn_frame.pack(pady=(0, 16))
        tk.Button(
            btn_frame,
            text="重新打开页面",
            command=lambda: webbrowser.open(url),
            bg="#facc15",
            fg="#0f172a",
            font=("Microsoft YaHei", 10, "bold"),
            padx=12,
            pady=4,
        ).pack(side="left", padx=8)
        tk.Button(
            btn_frame,
            text="退出程序",
            command=root.destroy,
            bg="#f43f5e",
            fg="#ffffff",
            font=("Microsoft YaHei", 10, "bold"),
            padx=12,
            pady=4,
        ).pack(side="left", padx=8)
        root.mainloop()
    except Exception:
        while True:
            time.sleep(1.0)


def main() -> None:
    runtime_root = _configure_frozen_stdio_and_logging()
    logger = logging.getLogger("nailong.desktop")
    logger.info("Launching Nailong Pose Desktop Release (runtime_root=%s)", runtime_root)

    import uvicorn
    from src.server import app, studio

    studio.set_release_mode(True)
    port = _find_free_port(18765)
    base_url = f"http://127.0.0.1:{port}"
    logger.info("Starting embedded Uvicorn server at %s", base_url)

    uv_config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_config=None,
        access_log=False,
        lifespan="on",
    )
    uv_server = uvicorn.Server(uv_config)
    server_thread = threading.Thread(target=uv_server.run, name="nailong-uvicorn", daemon=True)
    server_thread.start()

    ready = _wait_server_ready(f"{base_url}/api/config", timeout=18.0)
    logger.info("Embedded server ready=%s", ready)

    title = "奶龙家人鉴定 · 奶蛙动作识别"
    try:
        import webview

        # Allow downloading matched posters and purity certificates cleanly
        webview.settings["ALLOW_DOWNLOADS"] = True
        webview.create_window(
            title=title,
            url=base_url,
            width=1360,
            height=850,
            min_size=(1040, 680),
            background_color="#070a14",
            confirm_close=False,
        )
        webview.start(gui="edgechromium", debug=False)
    except Exception as exc:
        logger.warning("pywebview unavailable (%s), falling back to system browser", exc)
        _run_browser_fallback(base_url, title)
    finally:
        logger.info("Shutting down embedded server and camera worker...")
        uv_server.should_exit = True
        try:
            studio.stop()
        except Exception:
            pass
        server_thread.join(timeout=3.0)
        logger.info("Clean exit complete.")


if __name__ == "__main__":
    main()
