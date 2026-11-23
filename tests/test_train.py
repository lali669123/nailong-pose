"""Unit tests for src.train."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_DIM, get_feature_names
from src.train import main as train_cli_main
from src.train import train_and_evaluate

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _write_synthetic_csv(
    csv_path: Path,
    class_specs: list[tuple[str, list[str], int]],
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic samples.csv with well-separated class clusters.

    Args:
        csv_path: Target CSV file path.
        class_specs: List of ``(label, session_ids, samples_per_session)``.
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    feature_names = get_feature_names()
    rows: list[dict[str, object]] = []
    ts = 1732240000.0

    for cls_idx, (label, sessions, n_per_sess) in enumerate(class_specs):
        # Distinct class mean vector in 46-dim space
        center = np.zeros(FEATURE_DIM, dtype=np.float32)
        center[(cls_idx * 5) % FEATURE_DIM : ((cls_idx * 5) % FEATURE_DIM) + 5] = 3.0
        center[34 + (cls_idx % 8)] = 0.2 + 0.25 * cls_idx

        for sess_id in sessions:
            for _ in range(n_per_sess):
                vec = center + rng.normal(0.0, 0.15, size=FEATURE_DIM)
                row: dict[str, object] = {
                    "timestamp": round(ts, 3),
                    "session_id": sess_id,
                    "label": label,
                }
                for name, val in zip(feature_names, vec.tolist()):
                    row[name] = round(float(val), 6)
                rows.append(row)
                ts += 0.05

    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def test_train_and_evaluate_with_stratified_group_kfold(tmp_path: Path) -> None:
    samples_csv = tmp_path / "samples.csv"
    model_file = tmp_path / "pose_clf.joblib"
    plot_file = tmp_path / "eval_confusion.png"

    # 3 classes across 4 shared recording rounds (all classes in every session)
    rounds = ["round_1", "round_2", "round_3", "round_4"]
    _write_synthetic_csv(
        samples_csv,
        [
            ("idle", rounds, 6),
            ("01_kungfu", rounds, 6),
            ("06_shocked", rounds, 6),
        ],
    )

    result = train_and_evaluate(
        samples_path=samples_csv,
        model_path=model_file,
        plot_path=plot_file,
    )

    expected_keys = {
        "best_model",
        "best_accuracy",
        "best_macro_f1",
        "cv_strategy",
        "comparisons",
        "classes",
        "sample_counts",
        "confusion_matrix_url",
        "trained_at",
    }
    assert expected_keys.issubset(result.keys())
    assert result["best_model"] in {"RandomForest", "MLP"}
    assert result["cv_strategy"].startswith("StratifiedGroupKFold")
    assert 0.9 <= result["best_accuracy"] <= 1.0
    assert 0.9 <= result["best_macro_f1"] <= 1.0
    assert set(result["comparisons"].keys()) == {"RandomForest", "MLP"}
    for metrics in result["comparisons"].values():
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert sorted(result["classes"]) == ["01_kungfu", "06_shocked", "idle"]
    assert result["sample_counts"] == {"01_kungfu": 24, "06_shocked": 24, "idle": 24}
    assert result["confusion_matrix_url"] == "/models/eval_confusion.png"
    assert isinstance(result["trained_at"], str) and len(result["trained_at"]) >= 19

    # Verify joblib bundle reloads cleanly and predicts accurately
    assert model_file.exists() and model_file.stat().st_size > 0
    bundle = joblib.load(model_file)
    for key in ("model", "model_name", "classes", "feature_names", "feature_dim", "metrics", "trained_at"):
        assert key in bundle
    assert bundle["model_name"] == result["best_model"]
    assert bundle["feature_dim"] == FEATURE_DIM
    assert bundle["feature_names"] == get_feature_names()
    assert bundle["classes"] == result["classes"]

    df = pd.read_csv(samples_csv)
    X_sample = df[get_feature_names()].to_numpy(dtype=np.float32)[:5]
    preds = bundle["model"].predict(X_sample)
    probs = bundle["model"].predict_proba(X_sample)
    assert preds.shape == (5,)
    assert probs.shape == (5, 3)
    assert all(p in bundle["classes"] for p in preds)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-5)

    # Verify valid PNG was generated
    assert plot_file.exists() and plot_file.stat().st_size > 1000
    assert plot_file.read_bytes()[:8] == PNG_SIGNATURE


def test_train_per_class_multi_sessions_uses_group_cv(tmp_path: Path) -> None:
    """Realistic app capture: each recording session belongs to one pose, >=3 sessions/class."""
    samples_csv = tmp_path / "samples.csv"
    model_file = tmp_path / "pose_clf.joblib"
    plot_file = tmp_path / "eval_confusion.png"

    _write_synthetic_csv(
        samples_csv,
        [
            ("idle", ["idle_s1", "idle_s2", "idle_s3"], 6),
            ("01_kungfu", ["kf_s1", "kf_s2", "kf_s3"], 6),
            ("09_heart", ["ht_s1", "ht_s2", "ht_s3"], 6),
        ],
    )

    result = train_and_evaluate(
        samples_path=samples_csv,
        model_path=model_file,
        plot_path=plot_file,
    )
    assert result["cv_strategy"].startswith("StratifiedGroupKFold")
    assert result["best_accuracy"] >= 0.9
    assert model_file.exists()
    assert plot_file.exists()


def test_train_fallback_to_stratified_kfold_when_sessions_insufficient(tmp_path: Path) -> None:
    """When each class only has 1 session, must fall back to StratifiedKFold safely."""
    samples_csv = tmp_path / "samples.csv"
    model_file = tmp_path / "pose_clf.joblib"
    plot_file = tmp_path / "eval_confusion.png"

    _write_synthetic_csv(
        samples_csv,
        [
            ("idle", ["only_idle_sess"], 12),
            ("01_kungfu", ["only_kf_sess"], 12),
            ("02_arms_crossed", ["only_ac_sess"], 12),
        ],
    )

    result = train_and_evaluate(
        samples_path=samples_csv,
        model_path=model_file,
        plot_path=plot_file,
    )
    assert result["cv_strategy"].startswith("StratifiedKFold")
    assert result["best_accuracy"] >= 0.9
    assert model_file.exists()
    assert plot_file.read_bytes()[:8] == PNG_SIGNATURE


def test_insufficient_samples_and_classes_raise_value_error(tmp_path: Path) -> None:
    model_file = tmp_path / "model.joblib"
    plot_file = tmp_path / "cm.png"

    # 1. Missing file
    with pytest.raises(ValueError, match="样本文件"):
        train_and_evaluate(tmp_path / "missing.csv", model_file, plot_file)

    # 2. Empty file
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        train_and_evaluate(empty_csv, model_file, plot_file)

    # 3. Only 1 class
    one_class_csv = tmp_path / "one_class.csv"
    _write_synthetic_csv(one_class_csv, [("idle", ["s1", "s2"], 6)])
    with pytest.raises(ValueError, match="2 个"):
        train_and_evaluate(one_class_csv, model_file, plot_file)

    # 4. One class has fewer than 5 samples
    few_samples_csv = tmp_path / "few_samples.csv"
    _write_synthetic_csv(
        few_samples_csv,
        [
            ("idle", ["s1"], 10),
            ("01_kungfu", ["s2"], 4),
        ],
    )
    with pytest.raises(ValueError, match="5 条"):
        train_and_evaluate(few_samples_csv, model_file, plot_file)

    # 5. Missing required feature columns
    bad_cols_csv = tmp_path / "bad_cols.csv"
    pd.DataFrame({"label": ["idle", "01_kungfu"], "kp0_x": [0.1, 0.2]}).to_csv(
        bad_cols_csv, index=False
    )
    with pytest.raises(ValueError, match="缺少必要列"):
        train_and_evaluate(bad_cols_csv, model_file, plot_file)


def test_cli_entry_point(tmp_path: Path) -> None:
    samples_csv = tmp_path / "cli_samples.csv"
    model_file = tmp_path / "cli_model.joblib"
    plot_file = tmp_path / "cli_cm.png"

    _write_synthetic_csv(
        samples_csv,
        [
            ("idle", ["s1", "s2", "s3"], 5),
            ("01_kungfu", ["s1", "s2", "s3"], 5),
        ],
    )

    rc = train_cli_main(
        [
            "--samples",
            str(samples_csv),
            "--model",
            str(model_file),
            "--plot",
            str(plot_file),
        ]
    )
    assert rc == 0
    assert model_file.exists()
    assert plot_file.exists()

    rc_fail = train_cli_main(["--samples", str(tmp_path / "nonexistent.csv")])
    assert rc_fail == 1
