"""Unit tests for src.features."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.features import (
    FEATURE_DIM,
    KEYPOINT_INDICES,
    SKELETON_EDGES,
    extract_features,
    get_feature_names,
)


def _make_pose_dicts(
    scale: float = 1.0,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    visibility: float = 0.95,
) -> list[dict[str, float]]:
    """Build a realistic 33-point MediaPipe landmark list in dict format."""
    # Base coordinates in [0, 1] space
    base: dict[int, tuple[float, float]] = {
        0: (0.50, 0.18),  # nose
        11: (0.42, 0.30),  # left shoulder
        12: (0.58, 0.30),  # right shoulder
        13: (0.36, 0.44),  # left elbow
        14: (0.66, 0.42),  # right elbow
        15: (0.28, 0.38),  # left wrist
        16: (0.74, 0.50),  # right wrist
        19: (0.25, 0.36),  # left index
        20: (0.77, 0.52),  # right index
        23: (0.44, 0.58),  # left hip
        24: (0.56, 0.58),  # right hip
        25: (0.42, 0.76),  # left knee
        26: (0.58, 0.75),  # right knee
        27: (0.40, 0.92),  # left ankle
        28: (0.60, 0.92),  # right ankle
        31: (0.37, 0.95),  # left foot index
        32: (0.63, 0.95),  # right foot index
    }
    landmarks: list[dict[str, float]] = []
    for idx in range(33):
        bx, by = base.get(idx, (0.5, 0.5))
        landmarks.append(
            {
                "x": bx * scale + shift_x,
                "y": by * scale + shift_y,
                "z": 0.0,
                "visibility": visibility,
            }
        )
    return landmarks


def _dicts_to_objects(lms: list[dict[str, float]]) -> list[SimpleNamespace]:
    return [SimpleNamespace(**pt) for pt in lms]


def test_constants_and_feature_names() -> None:
    expected_indices = [0, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28, 31, 32]
    assert KEYPOINT_INDICES == expected_indices
    assert len(KEYPOINT_INDICES) == 17
    assert FEATURE_DIM == 46

    names = get_feature_names()
    assert len(names) == FEATURE_DIM
    assert len(set(names)) == FEATURE_DIM
    assert names[0] == "kp0_x"
    assert names[1] == "kp0_y"
    assert names[32] == "kp32_x"
    assert names[33] == "kp32_y"
    assert "ang_l_elbow" in names[34:]
    assert "ang_r_knee" in names[34:]


def test_skeleton_edges_validity() -> None:
    n_points = len(KEYPOINT_INDICES)
    assert len(SKELETON_EDGES) > 0

    seen_undirected: set[tuple[int, int]] = set()
    covered_nodes: set[int] = set()
    for edge in SKELETON_EDGES:
        assert isinstance(edge, tuple) and len(edge) == 2
        u, v = edge
        assert isinstance(u, int) and isinstance(v, int)
        assert 0 <= u < n_points
        assert 0 <= v < n_points
        assert u != v
        canon = (min(u, v), max(u, v))
        assert canon not in seen_undirected
        seen_undirected.add(canon)
        covered_nodes.add(u)
        covered_nodes.add(v)

    assert covered_nodes == set(range(n_points))


def test_extract_features_output_contract_and_input_formats() -> None:
    lms_dict = _make_pose_dicts()
    lms_obj = _dicts_to_objects(lms_dict)
    lms_wrapper = SimpleNamespace(landmark=lms_obj)

    feat_dict, skel_dict = extract_features(lms_dict)
    feat_obj, skel_obj = extract_features(lms_obj)
    feat_wrap, skel_wrap = extract_features(lms_wrapper)

    assert feat_dict is not None
    assert isinstance(feat_dict, np.ndarray)
    assert feat_dict.shape == (FEATURE_DIM,)
    assert feat_dict.dtype == np.float32
    assert np.all(np.isfinite(feat_dict))

    np.testing.assert_allclose(feat_dict, feat_obj, atol=1e-6)
    np.testing.assert_allclose(feat_dict, feat_wrap, atol=1e-6)

    assert len(skel_dict) == 17
    assert skel_dict == skel_obj == skel_wrap
    for expected_mp_idx, item in zip(KEYPOINT_INDICES, skel_dict):
        assert set(item.keys()) == {"idx", "x", "y", "v"}
        assert item["idx"] == expected_mp_idx
        assert isinstance(item["x"], float)
        assert isinstance(item["y"], float)
        assert isinstance(item["v"], float)


@pytest.mark.parametrize(
    ("shift_x", "shift_y"),
    [(0.25, -0.17), (-1.3, 2.8), (0.001, 0.002)],
)
def test_translation_invariance(shift_x: float, shift_y: float) -> None:
    feat_base, _ = extract_features(_make_pose_dicts())
    feat_shifted, _ = extract_features(_make_pose_dicts(shift_x=shift_x, shift_y=shift_y))

    assert feat_base is not None and feat_shifted is not None
    np.testing.assert_allclose(feat_shifted, feat_base, atol=1e-5)


@pytest.mark.parametrize("scale", [0.25, 0.6, 2.5, 8.0])
def test_scale_invariance(scale: float) -> None:
    feat_base, _ = extract_features(_make_pose_dicts())
    feat_scaled, _ = extract_features(_make_pose_dicts(scale=scale))

    assert feat_base is not None and feat_scaled is not None
    np.testing.assert_allclose(feat_scaled, feat_base, atol=1e-5)


def test_combined_scale_and_translation_invariance() -> None:
    feat_base, _ = extract_features(_make_pose_dicts())
    feat_transformed, _ = extract_features(_make_pose_dicts(scale=3.2, shift_x=-0.75, shift_y=1.4))

    assert feat_base is not None and feat_transformed is not None
    np.testing.assert_allclose(feat_transformed, feat_base, atol=1e-5)


def test_geometric_feature_values() -> None:
    lms = _make_pose_dicts()
    # Set shoulders at y=0.3, hips at y=0.5 -> torso_len = 0.2, origin = (0.5, 0.5)
    lms[11]["x"], lms[11]["y"] = 0.4, 0.3
    lms[12]["x"], lms[12]["y"] = 0.6, 0.3
    lms[23]["x"], lms[23]["y"] = 0.4, 0.5
    lms[24]["x"], lms[24]["y"] = 0.6, 0.5
    lms[0]["x"], lms[0]["y"] = 0.5, 0.2

    # Left elbow = exact 90 deg: shoulder(0.4,0.3) -> elbow(0.4,0.4) -> wrist(0.5,0.4)
    lms[13]["x"], lms[13]["y"] = 0.4, 0.4
    lms[15]["x"], lms[15]["y"] = 0.5, 0.4

    # Right elbow = exact 180 deg (straight): shoulder(0.6,0.3) -> elbow(0.6,0.4) -> wrist(0.6,0.5)
    lms[14]["x"], lms[14]["y"] = 0.6, 0.4
    lms[16]["x"], lms[16]["y"] = 0.6, 0.5

    # Ankles separated by 0.4 -> dist_ankles = 0.4 / 0.2 = 2.0
    lms[27]["x"], lms[27]["y"] = 0.3, 0.9
    lms[28]["x"], lms[28]["y"] = 0.7, 0.9

    feat, _ = extract_features(lms)
    assert feat is not None
    names = get_feature_names()
    fmap = dict(zip(names, feat.tolist()))

    assert fmap["ang_l_elbow"] == pytest.approx(0.5, abs=1e-5)
    assert fmap["ang_r_elbow"] == pytest.approx(1.0, abs=1e-5)
    assert fmap["dist_ankles"] == pytest.approx(2.0, abs=1e-5)
    # Left wrist y (0.4) - nose y (0.2) = 0.2 -> / torso_len(0.2) = 1.0
    assert fmap["rel_l_wrist_nose_y"] == pytest.approx(1.0, abs=1e-5)
    # Right wrist y (0.5) - nose y (0.2) = 0.3 -> / torso_len(0.2) = 1.5
    assert fmap["rel_r_wrist_nose_y"] == pytest.approx(1.5, abs=1e-5)

    # All 8 angles must lie in [0, 1]
    for angle_key in names[34:42]:
        assert 0.0 <= fmap[angle_key] <= 1.0


def test_coincident_limb_points_no_nan() -> None:
    lms = _make_pose_dicts()
    # Collapse left elbow onto left shoulder and wrist
    lms[13]["x"], lms[13]["y"] = lms[11]["x"], lms[11]["y"]
    lms[15]["x"], lms[15]["y"] = lms[11]["x"], lms[11]["y"]

    feat, skel = extract_features(lms)
    assert feat is not None
    assert len(skel) == 17
    assert np.all(np.isfinite(feat))


@pytest.mark.parametrize("bad_input", [None, [], _make_pose_dicts()[:32], "invalid", 123])
def test_invalid_inputs_return_none(bad_input: object) -> None:
    feat, skel = extract_features(bad_input)
    assert feat is None
    assert skel == []


def test_low_core_visibility_returns_none() -> None:
    lms = _make_pose_dicts(visibility=0.95)
    for idx in (11, 12, 23, 24):
        lms[idx]["visibility"] = 0.25
    feat, skel = extract_features(lms)
    assert feat is None
    assert skel == []

    # Just above threshold should succeed
    for idx in (11, 12, 23, 24):
        lms[idx]["visibility"] = 0.31
    feat_ok, skel_ok = extract_features(lms)
    assert feat_ok is not None
    assert len(skel_ok) == 17


def test_zero_and_tiny_torso_length_returns_none() -> None:
    lms_zero = _make_pose_dicts()
    # Put shoulders exactly on hips
    lms_zero[11]["x"], lms_zero[11]["y"] = lms_zero[23]["x"], lms_zero[23]["y"]
    lms_zero[12]["x"], lms_zero[12]["y"] = lms_zero[24]["x"], lms_zero[24]["y"]
    feat, skel = extract_features(lms_zero)
    assert feat is None
    assert skel == []

    lms_tiny = _make_pose_dicts()
    lms_tiny[11]["x"], lms_tiny[11]["y"] = 0.44, 0.58 - 1e-5
    lms_tiny[12]["x"], lms_tiny[12]["y"] = 0.56, 0.58 - 1e-5
    feat2, skel2 = extract_features(lms_tiny)
    assert feat2 is None
    assert skel2 == []


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_inf_coordinates_handled_safely(bad_val: float) -> None:
    lms_core = _make_pose_dicts()
    lms_core[11]["x"] = bad_val
    feat, skel = extract_features(lms_core)
    assert feat is None
    assert skel == []

    lms_limb = _make_pose_dicts()
    lms_limb[15]["y"] = bad_val
    feat2, skel2 = extract_features(lms_limb)
    assert feat2 is None
    assert skel2 == []


def test_nan_visibility_on_core_points_rejected() -> None:
    lms = _make_pose_dicts()
    for idx in (11, 12, 23, 24):
        lms[idx]["visibility"] = float("nan")
    feat, skel = extract_features(lms)
    assert feat is None
    assert skel == []
