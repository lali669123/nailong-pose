"""Dual-column match poster composition (1280x680) with 4K & RGBA transparency support."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.config_loader import PROJECT_ROOT, load_chinese_font, parse_hex_color

logger = logging.getLogger(__name__)

POSTER_WIDTH = 1280
POSTER_HEIGHT = 680

# Warm cream backdrop used behind right-panel Nailong illustrations so RGBA PNGs never get black backgrounds
_WARM_CARD_BG = (255, 251, 235)


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, size[0], size[1]], radius=radius, fill=255)
    return mask


def _prepare_live_panel(frame_bgr: np.ndarray | None, panel_size: tuple[int, int]) -> Image.Image:
    """Convert BGR OpenCV frame to RGB and center-crop fit into the left panel."""
    w, h = panel_size
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        fallback = Image.new("RGB", (w, h), (30, 41, 59))
        draw = ImageDraw.Draw(fallback)
        font = load_chinese_font(28, bold=True)
        draw.text((w // 2, h // 2), "镜头定格预览", fill=(148, 163, 184), font=font, anchor="mm")
        return fallback

    if frame_bgr.ndim == 2:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
    elif frame_bgr.shape[2] == 4:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(rgb)
    return ImageOps.fit(pil_img, (w, h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _resolve_pose_image_path(pose_info: dict[str, Any]) -> Path | None:
    for key in ("image_path", "image"):
        val = pose_info.get(key)
        if not val:
            continue
        p = Path(str(val))
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _prepare_nailong_panel(
    pose_info: dict[str, Any],
    panel_size: tuple[int, int],
    accent: tuple[int, int, int],
) -> Image.Image:
    """Load, scale (including 4K), and alpha-composite Nailong art onto a warm light backdrop."""
    w, h = panel_size
    # Soft warm-cream vertical gradient backdrop so transparent RGBA PNGs look crisp and bright
    backdrop = Image.new("RGBA", (w, h), (*_WARM_CARD_BG, 255))
    bg_draw = ImageDraw.Draw(backdrop)
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(_WARM_CARD_BG[0] * (1 - 0.04 * t) + accent[0] * (0.04 * t))
        g = int(_WARM_CARD_BG[1] * (1 - 0.04 * t) + accent[1] * (0.04 * t))
        b = int(_WARM_CARD_BG[2] * (1 - 0.06 * t) + accent[2] * (0.06 * t))
        bg_draw.line([(0, y), (w, y)], fill=(min(255, r), min(255, g), min(255, b), 255))

    img_path = _resolve_pose_image_path(pose_info)
    if img_path is None:
        rgb_fallback = backdrop.convert("RGB")
        draw = ImageDraw.Draw(rgb_fallback)
        font = load_chinese_font(34, bold=True)
        draw.text(
            (w // 2, h // 2),
            str(pose_info.get("name", "奶龙姿势")),
            fill=(51, 65, 85),
            font=font,
            anchor="mm",
        )
        return rgb_fallback

    try:
        with Image.open(img_path) as raw:
            # Speed up decoding for huge 4K JPEGs before high-quality LANCZOS resize
            if raw.format == "JPEG" and max(raw.size) > 1600:
                try:
                    raw.draft("RGB", (w * 2, h * 2))
                except Exception:
                    pass
            raw = ImageOps.exif_transpose(raw)
            src_rgba = raw.convert("RGBA")

        # Keep a small inner padding so the full character silhouette is visible without cropping
        pad = 12
        max_w = max(1, w - pad * 2)
        max_h = max(1, h - pad * 2)
        src_rgba.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        offset_x = (w - src_rgba.width) // 2
        offset_y = (h - src_rgba.height) // 2

        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        layer.paste(src_rgba, (offset_x, offset_y))
        composited = Image.alpha_composite(backdrop, layer)
        return composited.convert("RGB")
    except Exception as exc:
        logger.warning("Failed to load pose image %s: %s", img_path, exc)
        return backdrop.convert("RGB")


def _truncate_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: Any,
    max_width: int,
) -> str:
    if not text:
        return ""
    bbox = draw.textbbox((0, 0), text, font=font)
    if bbox[2] - bbox[0] <= max_width:
        return text
    ellipsis = "..."
    trimmed = text
    while trimmed:
        trimmed = trimmed[:-1]
        cand = trimmed + ellipsis
        bbox = draw.textbbox((0, 0), cand, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return cand
    return ellipsis


def compose_match_photo(
    frame_bgr: np.ndarray,
    pose_info: dict[str, Any],
    confidence: float,
    output_dir: Path,
) -> str:
    """Compose a 1280x680 dual-column comparison poster and save it as JPEG.

    Returns the generated filename (e.g. '20261122_102030_01_kungfu.jpg').
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    ts_file = now.strftime("%Y%m%d_%H%M%S")
    ts_display = now.strftime("%Y-%m-%d %H:%M:%S")

    pose_id = str(pose_info.get("id", "pose"))
    pose_name = str(pose_info.get("name", pose_id))
    subtitle = str(pose_info.get("subtitle") or "")
    speech = str(pose_info.get("speech") or "")
    is_hidden = bool(pose_info.get("is_hidden") or pose_id == "10_rainbow")
    accent = parse_hex_color(pose_info.get("color"), fallback=(250, 204, 21))
    conf_pct = max(0.0, min(100.0, float(confidence) * 100.0))

    # 1. Background canvas (sleek cyber warm-dark gradient)
    poster = Image.new("RGB", (POSTER_WIDTH, POSTER_HEIGHT), (15, 23, 42))
    draw = ImageDraw.Draw(poster)
    for y in range(POSTER_HEIGHT):
        t = y / (POSTER_HEIGHT - 1)
        r = int(15 * (1 - t) + (24 + accent[0] * 0.08) * t)
        g = int(23 * (1 - t) + (22 + accent[1] * 0.08) * t)
        b = int(42 * (1 - t) + (36 + accent[2] * 0.08) * t)
        draw.line([(0, y), (POSTER_WIDTH, y)], fill=(min(255, r), min(255, g), min(255, b)))

    # Subtle outer accent border (multi-ring rainbow accent for SSR hidden variant)
    if is_hidden:
        draw.rounded_rectangle([6, 6, POSTER_WIDTH - 6, POSTER_HEIGHT - 6], radius=26, outline=(244, 114, 182), width=3)
        draw.rounded_rectangle([11, 11, POSTER_WIDTH - 11, POSTER_HEIGHT - 11], radius=23, outline=(250, 204, 21), width=3)
        draw.rounded_rectangle([16, 16, POSTER_WIDTH - 16, POSTER_HEIGHT - 16], radius=20, outline=(56, 189, 248), width=2)
    else:
        draw.rounded_rectangle(
            [10, 10, POSTER_WIDTH - 10, POSTER_HEIGHT - 10],
            radius=24,
            outline=accent,
            width=3,
        )

    # 2. Top header bar
    title_font = load_chinese_font(26, bold=True)
    meta_font = load_chinese_font(18, bold=False)
    badge_font = load_chinese_font(20, bold=True)

    draw.rounded_rectangle([32, 22, 46, 56], radius=6, fill=accent)
    header_title = "★ SSR 隐藏款降临 · 奶龙家人同框认证 ★" if is_hidden else "奶龙家人鉴定 · 同框认证海报"
    draw.text(
        (60, 39),
        header_title,
        fill=(254, 240, 138) if is_hidden else (248, 250, 252),
        font=title_font,
        anchor="lm",
    )

    # Top-right timestamp & match badge
    conf_text = f"匹配度 {conf_pct:.1f}%"
    conf_bbox = draw.textbbox((0, 0), conf_text, font=badge_font)
    conf_w = (conf_bbox[2] - conf_bbox[0]) + 34
    conf_x0 = POSTER_WIDTH - 34 - conf_w
    draw.rounded_rectangle(
        [conf_x0, 22, POSTER_WIDTH - 34, 56],
        radius=17,
        fill=accent,
    )
    draw.text(
        (conf_x0 + conf_w // 2, 39),
        conf_text,
        fill=(15, 23, 42),
        font=badge_font,
        anchor="mm",
    )
    draw.text(
        (conf_x0 - 18, 39),
        ts_display,
        fill=(148, 163, 184),
        font=meta_font,
        anchor="rm",
    )

    # 3. Dual-column photo panels (Left: live frame, Right: Nailong image)
    panel_w, panel_h = 582, 444
    panel_y = 74
    left_x = 34
    right_x = POSTER_WIDTH - 34 - panel_w
    panel_radius = 22
    mask = _rounded_mask((panel_w, panel_h), panel_radius)

    live_panel = _prepare_live_panel(frame_bgr, (panel_w, panel_h))
    nailong_panel = _prepare_nailong_panel(pose_info, (panel_w, panel_h), accent)

    poster.paste(live_panel, (left_x, panel_y), mask)
    poster.paste(nailong_panel, (right_x, panel_y), mask)

    # Panel frames
    draw.rounded_rectangle(
        [left_x, panel_y, left_x + panel_w, panel_y + panel_h],
        radius=panel_radius,
        outline=(100, 116, 139),
        width=3,
    )
    draw.rounded_rectangle(
        [right_x, panel_y, right_x + panel_w, panel_y + panel_h],
        radius=panel_radius,
        outline=accent,
        width=4,
    )

    # Corner pills inside panels
    pill_font = load_chinese_font(16, bold=True)
    draw.rounded_rectangle([left_x + 14, panel_y + 14, left_x + 116, panel_y + 44], radius=14, fill=(15, 23, 42))
    draw.text((left_x + 65, panel_y + 29), "实拍定格", fill=(248, 250, 252), font=pill_font, anchor="mm")

    draw.rounded_rectangle([right_x + 14, panel_y + 14, right_x + 132, panel_y + 44], radius=14, fill=accent)
    draw.text((right_x + 73, panel_y + 29), "认证奶蛙", fill=(15, 23, 42), font=pill_font, anchor="mm")

    # Center "VS" / sync medallion between panels
    mid_x = POSTER_WIDTH // 2
    mid_y = panel_y + panel_h // 2
    draw.ellipse([mid_x - 30, mid_y - 30, mid_x + 30, mid_y + 30], fill=accent, outline=(15, 23, 42), width=5)
    vs_font = load_chinese_font(20, bold=True)
    draw.text((mid_x, mid_y), "SYNC", fill=(15, 23, 42), font=vs_font, anchor="mm")

    # 4. Bottom info card
    card_y0, card_y1 = 536, 654
    draw.rounded_rectangle(
        [34, card_y0, POSTER_WIDTH - 34, card_y1],
        radius=20,
        fill=(30, 41, 59),
        outline=(71, 85, 105),
        width=2,
    )
    # Left color bar
    draw.rounded_rectangle([48, card_y0 + 18, 58, card_y1 - 18], radius=5, fill=accent)

    name_font = load_chinese_font(36, bold=True)
    sub_font = load_chinese_font(20, bold=False)
    speech_font = load_chinese_font(22, bold=False)
    score_font = load_chinese_font(44, bold=True)
    score_lbl_font = load_chinese_font(16, bold=False)

    # Pose title & subtitle
    draw.text((78, card_y0 + 36), pose_name, fill=(255, 255, 255), font=name_font, anchor="lm")
    name_bbox = draw.textbbox((78, card_y0 + 36), pose_name, font=name_font, anchor="lm")
    if subtitle:
        sub_x = min(name_bbox[2] + 20, POSTER_WIDTH - 360)
        sub_trimmed = _truncate_text(draw, subtitle, sub_font, max_width=max(120, POSTER_WIDTH - 260 - sub_x))
        draw.text((sub_x, card_y0 + 40), sub_trimmed, fill=accent, font=sub_font, anchor="lm")

    # Speech quote line
    speech_display = f"「{speech}」" if speech else f"动作代号：{pose_id}"
    speech_trimmed = _truncate_text(draw, speech_display, speech_font, max_width=POSTER_WIDTH - 330)
    draw.text((78, card_y0 + 86), speech_trimmed, fill=(226, 232, 240), font=speech_font, anchor="lm")

    # Right score block inside bottom card
    draw.line([(POSTER_WIDTH - 236, card_y0 + 20), (POSTER_WIDTH - 236, card_y1 - 20)], fill=(71, 85, 105), width=2)
    draw.text(
        (POSTER_WIDTH - 135, card_y0 + 44),
        f"{conf_pct:.0f}%",
        fill=accent,
        font=score_font,
        anchor="mm",
    )
    draw.text(
        (POSTER_WIDTH - 135, card_y0 + 88),
        "动作同步率",
        fill=(148, 163, 184),
        font=score_lbl_font,
        anchor="mm",
    )

    # 5. Save JPEG cleanly (avoid filename collision within the same second)
    safe_pose_id = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in pose_id)
    filename = f"{ts_file}_{safe_pose_id}.jpg"
    dest = output_path / filename
    if dest.exists():
        filename = f"{ts_file}_{now.microsecond // 1000:03d}_{safe_pose_id}.jpg"
        dest = output_path / filename

    poster.save(dest, format="JPEG", quality=92, optimize=True)
    logger.info("Saved match poster: %s", dest)
    return filename
