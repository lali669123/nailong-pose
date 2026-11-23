"""Model training, leakage-safe cross-validation, and evaluation plotting.

Compares RandomForest and StandardScaler+MLP on extracted 46-dim pose features,
preferring StratifiedGroupKFold over ``session_id`` to prevent temporal leakage
across consecutive video frames, and exports the winning joblib bundle plus a
high-contrast confusion matrix PNG.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PATH = PROJECT_ROOT / "data" / "samples.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "pose_clf.joblib"
CONFUSION_IMG_PATH = PROJECT_ROOT / "models" / "eval_confusion.png"

_CACHE_ROOT = PROJECT_ROOT / ".cache"
_MPL_CACHE_DIR = _CACHE_ROOT / "mpl"
_TMP_CACHE_DIR = _CACHE_ROOT / "tmp"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_TMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE_DIR)
os.environ["TMP"] = str(_TMP_CACHE_DIR)
os.environ["TEMP"] = str(_TMP_CACHE_DIR)
tempfile.tempdir = str(_TMP_CACHE_DIR)

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config_loader import load_config
from src.features import FEATURE_DIM, get_feature_names

_MIN_CLASSES = 2
_MIN_SAMPLES_PER_CLASS = 5


def _load_non_trainable_labels(config_path: Path | None = None) -> set[str]:
    """Return pose IDs marked trainable=False in poses.yaml (e.g. hidden variant 10_rainbow)."""
    try:
        cfg = load_config(config_path)
    except Exception:
        return set()
    excluded: set[str] = set()
    for pose in cfg.get("poses") or []:
        if not bool(pose.get("trainable", True)):
            pid = str(pose.get("id") or "").strip()
            if pid:
                excluded.add(pid)
    return excluded


def _build_candidate_models() -> dict[str, object]:
    """Instantiate candidate classifiers specified by SPEC.md."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=500,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def _select_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    classes: list[str],
    min_class_count: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Choose StratifiedGroupKFold when session diversity allows, else StratifiedKFold."""
    n_classes = len(classes)

    if groups is not None:
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        if n_groups >= 3:
            max_splits = min(5, n_groups)
            for n_splits in range(max_splits, 2, -1):
                sgkf = StratifiedGroupKFold(n_splits=n_splits)
                try:
                    candidate_splits = list(sgkf.split(X, y, groups))
                except ValueError:
                    continue

                if len(candidate_splits) < 2:
                    continue

                all_valid = all(
                    len(tr) > 0
                    and len(te) > 0
                    and len(np.unique(y[tr])) == n_classes
                    and len(np.unique(y[te])) >= 2
                    for tr, te in candidate_splits
                )
                if all_valid:
                    return (
                        candidate_splits,
                        f"StratifiedGroupKFold(n_splits={n_splits})",
                    )

    n_splits = min(5, min_class_count)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    return splits, f"StratifiedKFold(n_splits={n_splits})"


def _save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    best_model: str,
    best_accuracy: float,
    best_macro_f1: float,
    cv_strategy: str,
    plot_path: Path,
) -> None:
    """Render and save a clean, dark-themed confusion matrix PNG."""
    _MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE_DIR)

    plot_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        np.where(row_sums == 0, 1, row_sums),
    )

    n_classes = len(classes)
    fig_size = max(5.6, min(12.0, 1.05 * n_classes + 2.8))

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    cmap = LinearSegmentedColormap.from_list(
        "nailong_amber",
        ["#111827", "#334155", "#B45309", "#F59E0B", "#FACC15"],
    )

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88), dpi=160)
    fig.patch.set_facecolor("#0B1120")
    ax.set_facecolor("#111827")

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row Recall", color="#CBD5E1", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#94A3B8", labelcolor="#CBD5E1", labelsize=9)
    cbar.outline.set_edgecolor("#334155")

    ticks = np.arange(n_classes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(
        classes,
        rotation=35 if n_classes > 3 else 0,
        ha="right" if n_classes > 3 else "center",
        color="#E2E8F0",
        fontsize=9.5,
    )
    ax.set_yticklabels(classes, color="#E2E8F0", fontsize=9.5)

    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    # Subtle grid borders between cells
    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which="minor", color="#1E293B", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate cells with count and row-normalized percentage
    text_size = 9.5 if n_classes <= 6 else (8.0 if n_classes <= 10 else 7.0)
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_norm[i, j]
            count = int(cm[i, j])
            txt_color = "#0B1120" if val >= 0.58 else "#F8FAFC"
            label_str = f"{count}\n({val * 100:.0f}%)"
            ax.text(
                j,
                i,
                label_str,
                ha="center",
                va="center",
                color=txt_color,
                fontsize=text_size,
                fontweight="bold" if i == j else "normal",
            )

    ax.set_xlabel("Predicted Label", color="#FACC15", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Label", color="#FACC15", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        f"Confusion Matrix - {best_model}\n"
        f"Acc: {best_accuracy * 100:.1f}%  |  Macro-F1: {best_macro_f1 * 100:.1f}%  |  {cv_strategy}",
        color="#F8FAFC",
        fontsize=11.5,
        fontweight="bold",
        pad=14,
    )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def train_and_evaluate(
    samples_path: Path | None = None,
    model_path: Path | None = None,
    plot_path: Path | None = None,
    exclude_labels: set[str] | list[str] | None = None,
    config_path: Path | None = None,
) -> dict:
    """Train RandomForest & MLP classifiers, evaluate via CV, and export artifacts.

    Args:
        samples_path: Path to ``samples.csv`` (defaults to ``SAMPLES_PATH``).
        model_path: Output path for joblib bundle (defaults to ``MODEL_PATH``).
        plot_path: Output path for confusion matrix PNG (defaults to ``CONFUSION_IMG_PATH``).
        exclude_labels: Optional labels to exclude from training (defaults to poses marked
            ``trainable: false`` in ``poses.yaml``, such as hidden variant ``10_rainbow``).
        config_path: Optional path to ``poses.yaml`` used when resolving default excluded labels.

    Returns:
        Dictionary of evaluation metrics, comparisons, class stats, and metadata.

    Raises:
        ValueError: If samples file is missing/malformed or has < 2 classes or
            < 5 samples in any class.
    """
    resolved_samples = Path(samples_path) if samples_path is not None else SAMPLES_PATH
    resolved_model = Path(model_path) if model_path is not None else MODEL_PATH
    resolved_plot = Path(plot_path) if plot_path is not None else CONFUSION_IMG_PATH
    resolved_excluded = (
        {str(x).strip() for x in exclude_labels if str(x).strip()}
        if exclude_labels is not None
        else _load_non_trainable_labels(config_path)
    )

    if not resolved_samples.exists():
        raise ValueError(f"未找到样本文件：{resolved_samples}，请先采集动作数据后再训练。")

    try:
        df = pd.read_csv(resolved_samples)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("样本文件为空，请至少采集 2 个动作类别、每类至少 5 帧样本。") from exc
    except Exception as exc:
        raise ValueError(f"读取样本文件失败：{exc}") from exc

    if df.empty:
        raise ValueError("样本文件不包含任何数据行，请先采集动作样本。")

    feature_names = get_feature_names()
    missing_cols = [c for c in ["label", *feature_names] if c not in df.columns]
    if missing_cols:
        preview = ", ".join(missing_cols[:5])
        suffix = "..." if len(missing_cols) > 5 else ""
        raise ValueError(f"样本文件缺少必要列（共 {len(missing_cols)} 列）：{preview}{suffix}")

    # Clean labels and coerce feature columns to numeric
    labels_series = df["label"].astype(str).str.strip()
    valid_label_mask = df["label"].notna() & (labels_series != "") & (labels_series.str.lower() != "nan")

    feature_df = df[feature_names].apply(pd.to_numeric, errors="coerce")
    feature_arr = feature_df.to_numpy(dtype=np.float64)
    valid_feat_mask = np.all(np.isfinite(feature_arr), axis=1)

    valid_mask = valid_label_mask.to_numpy() & valid_feat_mask
    if not np.any(valid_mask):
        raise ValueError("样本文件中没有有效的数值特征行，请重新采集数据。")

    X = feature_arr[valid_mask].astype(np.float32)
    y = labels_series.to_numpy()[valid_mask]

    if "session_id" in df.columns:
        raw_sessions = df.loc[valid_mask, "session_id"].fillna("default").astype(str).str.strip()
        groups = np.where(raw_sessions.to_numpy() == "", "default", raw_sessions.to_numpy())
    else:
        groups = None

    actually_excluded: list[str] = []
    if resolved_excluded:
        keep_mask = np.array([lbl not in resolved_excluded for lbl in y], dtype=bool)
        if not np.all(keep_mask):
            actually_excluded = sorted({str(lbl) for lbl in y[~keep_mask]})
            X = X[keep_mask]
            y = y[keep_mask]
            if groups is not None:
                groups = groups[keep_mask]
            if len(y) == 0:
                raise ValueError(
                    f"过滤非训练隐藏类（{', '.join(actually_excluded)}）后无剩余有效样本，请先采集常规动作数据。"
                )

    unique_labels, counts = np.unique(y, return_counts=True)
    classes = [str(c) for c in unique_labels]
    sample_counts = {str(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)}

    if len(classes) < _MIN_CLASSES:
        raise ValueError(
            f"至少需要 {_MIN_CLASSES} 个不同动作类别才能训练，当前仅检测到 {len(classes)} 类"
            f"（{', '.join(classes) if classes else '无'}）。"
        )

    insufficient = [
        f"{lbl}({cnt}/{_MIN_SAMPLES_PER_CLASS})"
        for lbl, cnt in sample_counts.items()
        if cnt < _MIN_SAMPLES_PER_CLASS
    ]
    if insufficient:
        raise ValueError(
            f"每个动作类别至少需要 {_MIN_SAMPLES_PER_CLASS} 条有效样本，当前样本不足：{', '.join(insufficient)}。"
        )

    min_class_count = int(min(sample_counts.values()))
    splits, cv_strategy = _select_cv_splits(X, y, groups, classes, min_class_count)

    candidates = _build_candidate_models()
    comparisons: dict[str, dict[str, float]] = {}
    oof_predictions: dict[str, np.ndarray] = {}

    for name, estimator in candidates.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            y_pred = cross_val_predict(clone(estimator), X, y, cv=splits)
        acc = round(float(accuracy_score(y, y_pred)), 4)
        macro_f1 = round(float(f1_score(y, y_pred, average="macro", zero_division=0)), 4)
        comparisons[name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
        }
        oof_predictions[name] = y_pred

    best_name = max(
        comparisons.keys(),
        key=lambda k: (comparisons[k]["macro_f1"], comparisons[k]["accuracy"]),
    )
    best_accuracy = comparisons[best_name]["accuracy"]
    best_macro_f1 = comparisons[best_name]["macro_f1"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        best_estimator = clone(candidates[best_name]).fit(X, y)

    fitted_classes = [str(c) for c in best_estimator.classes_]
    trained_at = datetime.now().astimezone().isoformat(timespec="seconds")

    metrics_dict = {
        "best_model": best_name,
        "best_accuracy": best_accuracy,
        "best_macro_f1": best_macro_f1,
        "cv_strategy": cv_strategy,
        "comparisons": comparisons,
        "sample_counts": sample_counts,
    }

    bundle = {
        "model": best_estimator,
        "model_name": best_name,
        "classes": fitted_classes,
        "feature_names": feature_names,
        "feature_dim": FEATURE_DIM,
        "metrics": metrics_dict,
        "trained_at": trained_at,
    }

    resolved_model.parent.mkdir(parents=True, exist_ok=True)
    tmp_model_path = resolved_model.with_suffix(resolved_model.suffix + ".tmp")
    joblib.dump(bundle, tmp_model_path)
    os.replace(tmp_model_path, resolved_model)

    _save_confusion_matrix_plot(
        y_true=y,
        y_pred=oof_predictions[best_name],
        classes=fitted_classes,
        best_model=best_name,
        best_accuracy=best_accuracy,
        best_macro_f1=best_macro_f1,
        cv_strategy=cv_strategy,
        plot_path=resolved_plot,
    )

    return {
        "best_model": best_name,
        "best_accuracy": best_accuracy,
        "best_macro_f1": best_macro_f1,
        "cv_strategy": cv_strategy,
        "comparisons": comparisons,
        "classes": fitted_classes,
        "sample_counts": sample_counts,
        "excluded_labels": actually_excluded,
        "confusion_matrix_url": "/models/eval_confusion.png",
        "trained_at": trained_at,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Nailong Pose classifier.")
    parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help=f"Path to samples CSV (default: {SAMPLES_PATH})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=f"Output joblib model path (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help=f"Output confusion matrix PNG path (default: {CONFUSION_IMG_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = train_and_evaluate(
            samples_path=args.samples,
            model_path=args.model,
            plot_path=args.plot,
        )
    except ValueError as exc:
        print(f"[train] Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
