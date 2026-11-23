"""Pose feature extraction with translation & torso-scale invariance.

Extracts a 46-dimensional feature vector from MediaPipe Pose 33 landmarks:
- 34 dims: 17 core keypoints' normalized 2D coordinates relative to pelvis midpoint
           and scaled by torso length.
- 12 dims: 8 normalized joint angles (divided by pi, in [0, 1]) + 4 scale-invariant
           distances / relative height ratios.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# 17 main body keypoints (MediaPipe Pose 33-landmark indices):
#  0: nose
#  1: 11 left_shoulder    2: 12 right_shoulder
#  3: 13 left_elbow       4: 14 right_elbow
#  5: 15 left_wrist       6: 16 right_wrist
#  7: 19 left_index       8: 20 right_index
#  9: 23 left_hip        10: 24 right_hip
# 11: 25 left_knee       12: 26 right_knee
# 13: 27 left_ankle      14: 28 right_ankle
# 15: 31 left_foot_index 16: 32 right_foot_index
KEYPOINT_INDICES: list[int] = [
    0,
    11,
    12,
    13,
    14,
    15,
    16,
    19,
    20,
    23,
    24,
    25,
    26,
    27,
    28,
    31,
    32,
]

# Local-index (0..16) skeleton connections for frontend stick-figure rendering.
SKELETON_EDGES: list[tuple[int, int]] = [
    # Head to shoulders
    (0, 1),
    (0, 2),
    # Torso
    (1, 2),
    (1, 9),
    (2, 10),
    (9, 10),
    # Left arm
    (1, 3),
    (3, 5),
    (5, 7),
    # Right arm
    (2, 4),
    (4, 6),
    (6, 8),
    # Left leg
    (9, 11),
    (11, 13),
    (13, 15),
    # Right leg
    (10, 12),
    (12, 14),
    (14, 16),
]

_GEOM_FEATURE_NAMES: list[str] = [
    "ang_l_elbow",
    "ang_r_elbow",
    "ang_l_shoulder",
    "ang_r_shoulder",
    "ang_l_hip",
    "ang_r_hip",
    "ang_l_knee",
    "ang_r_knee",
    "dist_wrists",
    "dist_ankles",
    "rel_l_wrist_nose_y",
    "rel_r_wrist_nose_y",
]

FEATURE_DIM: int = len(KEYPOINT_INDICES) * 2 + len(_GEOM_FEATURE_NAMES)  # 46

_CORE_TORSO_MP_INDICES: tuple[int, int, int, int] = (11, 12, 23, 24)
_MIN_CORE_VISIBILITY: float = 0.3
_MIN_TORSO_LENGTH: float = 1e-4
_MIN_LANDMARK_COUNT: int = 33


def get_feature_names() -> list[str]:
    """Return the deterministic 46-element list of feature column names."""
    names: list[str] = []
    for mp_idx in KEYPOINT_INDICES:
        names.append(f"kp{mp_idx}_x")
        names.append(f"kp{mp_idx}_y")
    names.extend(_GEOM_FEATURE_NAMES)
    return names


def _parse_landmark(pt: Any) -> tuple[float, float, float] | None:
    """Extract (x, y, visibility) from a MediaPipe landmark object or dict."""
    if pt is None:
        return None

    if isinstance(pt, Mapping):
        raw_x = pt.get("x")
        raw_y = pt.get("y")
        raw_v = pt.get("visibility", pt.get("v", 1.0))
    else:
        raw_x = getattr(pt, "x", None)
        raw_y = getattr(pt, "y", None)
        raw_v = getattr(pt, "visibility", 1.0)

    if raw_x is None or raw_y is None:
        return None

    try:
        x = float(raw_x)
        y = float(raw_y)
        v = 0.0 if raw_v is None else float(raw_v)
    except (TypeError, ValueError):
        return None

    if not (np.isfinite(x) and np.isfinite(y)):
        return None

    if not np.isfinite(v):
        v = 0.0
    else:
        v = float(np.clip(v, 0.0, 1.0))

    return x, y, v


def _angle_norm(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute interior angle ABC divided by pi, normalized to [0, 1]."""
    ba = a - b
    bc = c - b
    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0
    cos_val = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cos_val = float(np.clip(cos_val, -1.0, 1.0))
    return float(np.arccos(cos_val) / np.pi)


def extract_features(
    landmarks: Any,
) -> tuple[np.ndarray | None, list[dict[str, float | int]]]:
    """Extract a 46-dim invariant pose feature vector and 17 skeleton points.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList.landmark (or
            NormalizedLandmarkList itself), or a sequence of >= 33 dicts with
            keys ``"x"``, ``"y"``, ``"visibility"``.

    Returns:
        ``(feat_vec, skeleton_points)`` where ``feat_vec`` is a ``float32``
        ndarray of shape ``(46,)`` with no NaN/Inf, and ``skeleton_points`` is
        a list of 17 dicts ``{"idx": int, "x": float, "y": float, "v": float}``.
        Returns ``(None, [])`` on invalid input, low core visibility (< 0.3),
        or degenerate torso length (< 1e-4).
    """
    if landmarks is None:
        return None, []

    # Allow passing MediaPipe NormalizedLandmarkList wrapper directly.
    if hasattr(landmarks, "landmark"):
        landmarks = landmarks.landmark

    if isinstance(landmarks, (str, bytes)) or not isinstance(landmarks, Sequence):
        # Support MediaPipe RepeatedCompositeContainer (not a Python Sequence subclass in some versions).
        try:
            n_landmarks = len(landmarks)
        except TypeError:
            return None, []
    else:
        n_landmarks = len(landmarks)

    if n_landmarks < _MIN_LANDMARK_COUNT:
        return None, []

    coords: dict[int, np.ndarray] = {}
    visibilities: dict[int, float] = {}
    skeleton_points: list[dict[str, float | int]] = []

    for mp_idx in KEYPOINT_INDICES:
        try:
            parsed = _parse_landmark(landmarks[mp_idx])
        except (IndexError, KeyError, TypeError):
            return None, []

        if parsed is None:
            return None, []

        x, y, v = parsed
        coords[mp_idx] = np.array([x, y], dtype=np.float64)
        visibilities[mp_idx] = v
        skeleton_points.append(
            {
                "idx": int(mp_idx),
                "x": round(x, 4),
                "y": round(y, 4),
                "v": round(v, 4),
            }
        )

    # Core torso visibility gate (left/right shoulders & hips)
    core_vis = sum(visibilities[idx] for idx in _CORE_TORSO_MP_INDICES) / 4.0
    if core_vis < _MIN_CORE_VISIBILITY:
        return None, []

    # Pelvis midpoint origin and torso length scale
    origin = 0.5 * (coords[23] + coords[24])
    shoulder_mid = 0.5 * (coords[11] + coords[12])
    torso_len = float(np.linalg.norm(shoulder_mid - origin))
    if not np.isfinite(torso_len) or torso_len < _MIN_TORSO_LENGTH:
        return None, []

    # 34 normalized 2D relative coordinates
    rel_coords: list[float] = []
    for mp_idx in KEYPOINT_INDICES:
        rel_pt = (coords[mp_idx] - origin) / torso_len
        rel_coords.append(float(rel_pt[0]))
        rel_coords.append(float(rel_pt[1]))

    # 8 normalized joint angles (in [0, 1])
    ang_l_elbow = _angle_norm(coords[11], coords[13], coords[15])
    ang_r_elbow = _angle_norm(coords[12], coords[14], coords[16])
    ang_l_shoulder = _angle_norm(coords[23], coords[11], coords[13])
    ang_r_shoulder = _angle_norm(coords[24], coords[12], coords[14])
    ang_l_hip = _angle_norm(coords[11], coords[23], coords[25])
    ang_r_hip = _angle_norm(coords[12], coords[24], coords[26])
    ang_l_knee = _angle_norm(coords[23], coords[25], coords[27])
    ang_r_knee = _angle_norm(coords[24], coords[26], coords[28])

    # 4 scale-invariant distances and relative vertical offsets
    dist_wrists = float(np.linalg.norm(coords[15] - coords[16]) / torso_len)
    dist_ankles = float(np.linalg.norm(coords[27] - coords[28]) / torso_len)
    rel_l_wrist_nose_y = float((coords[15][1] - coords[0][1]) / torso_len)
    rel_r_wrist_nose_y = float((coords[16][1] - coords[0][1]) / torso_len)

    geom_features = [
        ang_l_elbow,
        ang_r_elbow,
        ang_l_shoulder,
        ang_r_shoulder,
        ang_l_hip,
        ang_r_hip,
        ang_l_knee,
        ang_r_knee,
        dist_wrists,
        dist_ankles,
        rel_l_wrist_nose_y,
        rel_r_wrist_nose_y,
    ]

    feat_vec = np.asarray(rel_coords + geom_features, dtype=np.float32)
    if feat_vec.shape != (FEATURE_DIM,) or not np.all(np.isfinite(feat_vec)):
        return None, []

    return feat_vec, skeleton_points
