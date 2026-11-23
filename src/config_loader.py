"""Runtime directory setup, YAML config loading, and placeholder image generation."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

logger = logging.getLogger(__name__)


def _resolve_roots() -> tuple[Path, Path]:
    """Return (bundle_root, runtime_root) supporting both source and PyInstaller frozen builds."""
    if getattr(sys, "frozen", False):
        bundle_root = Path(
            getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)
        ).resolve()
        runtime_root = Path(sys.executable).resolve().parent.resolve()
        return bundle_root, runtime_root
    root = Path(__file__).resolve().parent.parent.resolve()
    return root, root


PROJECT_ROOT, RUNTIME_ROOT = _resolve_roots()

_PLACEHOLDER_TAG_KEY = "nailong_placeholder"
_PLACEHOLDER_TAG_VAL = "true"

_WINDOWS_FONT_CANDIDATES_BOLD = (
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
)

_WINDOWS_FONT_CANDIDATES_REGULAR = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
)


def ensure_runtime_dirs() -> dict[str, Path]:
    """Create required runtime directories and pin cache/tmp env vars to .cache/."""
    is_frozen = bool(getattr(sys, "frozen", False))
    cache_root = RUNTIME_ROOT / ".cache"
    mpl_dir = cache_root / "mpl"
    tmp_dir = cache_root / "tmp"

    dirs = {
        "cache": cache_root,
        "mpl": mpl_dir,
        "tmp": tmp_dir,
        "data": RUNTIME_ROOT / "data",
        "models": PROJECT_ROOT / "models",
        "captures": RUNTIME_ROOT / "captures",
        "assets_images": PROJECT_ROOT / "assets" / "images",
        "assets_audio": PROJECT_ROOT / "assets" / "audio",
    }
    writable_keys = {"cache", "mpl", "tmp", "captures"} if is_frozen else set(dirs.keys())
    for key, path in dirs.items():
        if key in writable_keys:
            path.mkdir(parents=True, exist_ok=True)

    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)

    return dirs


def parse_hex_color(
    color_str: str | None,
    fallback: tuple[int, int, int] = (250, 204, 21),
) -> tuple[int, int, int]:
    """Convert a '#RRGGBB' hex color string to an (R, G, B) tuple."""
    if not color_str or not isinstance(color_str, str):
        return fallback
    cleaned = color_str.strip().lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join(ch * 2 for ch in cleaned)
    if len(cleaned) != 6:
        return fallback
    try:
        return (
            int(cleaned[0:2], 16),
            int(cleaned[2:4], 16),
            int(cleaned[4:6], 16),
        )
    except ValueError:
        return fallback


def load_chinese_font(
    size: int,
    bold: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a Chinese-capable TrueType font on Windows with safe fallback."""
    candidates = (
        _WINDOWS_FONT_CANDIDATES_BOLD if bold else _WINDOWS_FONT_CANDIDATES_REGULAR
    )
    for font_path in candidates:
        if font_path.is_file():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap CJK/Latin text cleanly to fit within max_width pixels."""
    if not text:
        return []
    lines: list[str] = []
    current = ""
    for ch in text:
        if ch == "\n":
            lines.append(current)
            current = ""
            continue
        candidate = current + ch
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def _is_placeholder_image(img_path: Path) -> bool:
    """Detect whether an image file was auto-generated as a placeholder card."""
    if not img_path.is_file() or img_path.stat().st_size == 0:
        return False
    suffix = img_path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        return False
    try:
        with Image.open(img_path) as img:
            if suffix == ".png":
                return img.info.get(_PLACEHOLDER_TAG_KEY) == _PLACEHOLDER_TAG_VAL
            comment = img.info.get("comment")
            if isinstance(comment, bytes):
                comment = comment.decode("utf-8", errors="ignore")
            return comment == f"{_PLACEHOLDER_TAG_KEY}={_PLACEHOLDER_TAG_VAL}"
    except Exception:
        return False


def _generate_placeholder_image(img_path: Path, pose: dict[str, Any]) -> None:
    """Generate a 600x600 Nailong-styled placeholder card without overwriting existing files."""
    if img_path.exists() and img_path.stat().st_size > 0:
        return

    img_path.parent.mkdir(parents=True, exist_ok=True)
    size = 600
    accent = parse_hex_color(pose.get("color"), fallback=(250, 204, 21))

    # Warm cyber-dark gradient background blended softly with pose accent color
    canvas = Image.new("RGB", (size, size), (24, 27, 38))
    draw = ImageDraw.Draw(canvas)
    for y in range(size):
        t = y / (size - 1)
        r = int(26 * (1 - t) + (18 + accent[0] * 0.14) * t)
        g = int(31 * (1 - t) + (22 + accent[1] * 0.14) * t)
        b = int(46 * (1 - t) + (34 + accent[2] * 0.14) * t)
        draw.line([(0, y), (size, y)], fill=(min(255, r), min(255, g), min(255, b)))

    # Inner rounded card with accent frame
    margin = 28
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=32,
        fill=(32, 37, 52),
        outline=accent,
        width=4,
    )

    # Top accent pill
    pose_id = str(pose.get("id", "pose")).upper()
    badge_font = load_chinese_font(20, bold=True)
    badge_text = f"NAILONG · {pose_id}"
    badge_bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
    badge_w = (badge_bbox[2] - badge_bbox[0]) + 36
    badge_x0 = (size - badge_w) // 2
    draw.rounded_rectangle(
        [badge_x0, 56, badge_x0 + badge_w, 96],
        radius=20,
        fill=accent,
    )
    draw.text(
        (size // 2, 76),
        badge_text,
        fill=(18, 22, 32),
        font=badge_font,
        anchor="mm",
    )

    # Cute Nailong mascot emblem in center-top
    cx, cy = size // 2, 222
    draw.ellipse([cx - 84, cy - 84, cx + 84, cy + 84], fill=(254, 240, 138), outline=accent, width=5)
    # Little horns/ears
    draw.polygon([(cx - 48, cy - 64), (cx - 20, cy - 102), (cx - 4, cy - 72)], fill=(250, 204, 21))
    draw.polygon([(cx + 48, cy - 64), (cx + 20, cy - 102), (cx + 4, cy - 72)], fill=(250, 204, 21))
    # Eyes & blush
    draw.ellipse([cx - 34, cy - 22, cx - 16, cy - 4], fill=(30, 34, 46))
    draw.ellipse([cx + 16, cy - 22, cx + 34, cy - 4], fill=(30, 34, 46))
    draw.ellipse([cx - 56, cy + 4, cx - 32, cy + 20], fill=(251, 146, 146))
    draw.ellipse([cx + 32, cy + 4, cx + 56, cy + 20], fill=(251, 146, 146))
    # Happy smile arc
    draw.arc([cx - 28, cy - 4, cx + 28, cy + 36], start=15, end=165, fill=(30, 34, 46), width=4)

    # Pose Chinese name & subtitle
    name_font = load_chinese_font(38, bold=True)
    sub_font = load_chinese_font(22, bold=False)
    hint_font = load_chinese_font(20, bold=False)

    name = str(pose.get("name", pose_id))
    subtitle = str(pose.get("subtitle", ""))
    hint = str(pose.get("hint", ""))

    draw.text((size // 2, 356), name, fill=(255, 255, 255), font=name_font, anchor="mm")
    if subtitle:
        draw.text((size // 2, 404), subtitle, fill=accent, font=sub_font, anchor="mm")

    # Divider line
    draw.line([(96, 434), (size - 96, 434)], fill=(71, 85, 105), width=2)

    # Wrapped hint text
    hint_lines = _wrap_text(draw, hint, hint_font, max_width=size - 120)
    y_cursor = 462
    for line in hint_lines[:3]:
        draw.text((size // 2, y_cursor), line, fill=(203, 213, 225), font=hint_font, anchor="mm")
        y_cursor += 30

    if img_path.suffix.lower() == ".png":
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text(_PLACEHOLDER_TAG_KEY, _PLACEHOLDER_TAG_VAL)
        canvas.save(img_path, format="PNG", pnginfo=pnginfo)
    else:
        canvas.save(
            img_path,
            format="JPEG",
            quality=92,
            comment=f"{_PLACEHOLDER_TAG_KEY}={_PLACEHOLDER_TAG_VAL}".encode("utf-8"),
        )

    logger.info("Generated placeholder pose image at %s", img_path)


def _to_asset_url(rel_path: str) -> str:
    cleaned = rel_path.strip().replace("\\", "/").lstrip("/")
    return f"/{cleaned}" if cleaned else ""


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load poses.yaml, ensure runtime dirs, generate missing placeholder images, and enrich pose metadata."""
    ensure_runtime_dirs()

    cfg_file = config_path or (PROJECT_ROOT / "config" / "poses.yaml")
    with open(cfg_file, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    poses = cfg.get("poses") or []
    enriched_poses: list[dict[str, Any]] = []

    for raw_pose in poses:
        pose = dict(raw_pose)
        image_rel = str(pose.get("image") or "").strip()
        audio_rel = str(pose.get("audio") or "").strip()

        if image_rel:
            img_path = PROJECT_ROOT / image_rel
            if not img_path.is_file() or img_path.stat().st_size == 0:
                _generate_placeholder_image(img_path, pose)
            has_real_image = img_path.is_file() and img_path.stat().st_size > 0 and not _is_placeholder_image(img_path)
            image_url = _to_asset_url(image_rel)
        else:
            has_real_image = False
            image_url = ""

        if audio_rel:
            audio_path = PROJECT_ROOT / audio_rel
            has_audio = audio_path.is_file() and audio_path.stat().st_size > 0
            audio_url = _to_asset_url(audio_rel) if has_audio else ""
        else:
            has_audio = False
            audio_url = ""

        pose["image_url"] = image_url
        pose["audio_url"] = audio_url
        pose["has_real_image"] = has_real_image
        pose["has_audio"] = has_audio
        pose["trainable"] = bool(pose.get("trainable", True))
        pose["is_hidden"] = bool(pose.get("is_hidden", pose.get("id") == "10_rainbow"))
        pose["parent_pose_id"] = str(pose.get("parent_pose_id") or "")
        hv = pose.get("hidden_variant")
        if isinstance(hv, dict) and hv.get("pose_id"):
            pose["hidden_variant"] = {
                "pose_id": str(hv["pose_id"]),
                "probability": max(0.0, min(1.0, float(hv.get("probability", 0.15)))),
                "pity_after": max(0, int(hv.get("pity_after", 8))),
            }
        else:
            pose["hidden_variant"] = None
        enriched_poses.append(pose)

    app_cfg = dict(cfg.get("app") or {})
    env_mode = os.environ.get("NAILONG_MODE", "").strip().lower()
    raw_mode = env_mode or str(app_cfg.get("mode") or "studio").strip().lower()
    app_cfg["mode"] = "release" if raw_mode == "release" else "studio"
    cfg["app"] = app_cfg
    cfg["poses"] = enriched_poses
    return cfg
