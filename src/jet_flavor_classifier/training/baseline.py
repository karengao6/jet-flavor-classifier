from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from jet_flavor_classifier.data.features import JET_FEATURE_NAMES, extract_jet_features
from jet_flavor_classifier.data.labels import CLASS_NAMES, encode_labels
from jet_flavor_classifier.data.sampling import limit_indices
from jet_flavor_classifier.data.split import split_by_event


def _build_models(seed: int) -> dict[str, object]:
    """Create the feature-level baseline estimators."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "xgboost": XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASS_NAMES),
            max_depth=8,
            learning_rate=0.08,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def _compute_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, float | int]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )

    metrics: dict[str, dict[str, float | int]] = {}
    for class_index, class_name in enumerate(class_names):
        metrics[class_name] = {
            "precision": float(precision[class_index]),
            "recall": float(recall[class_index]),
            "f1": float(f1[class_index]),
            "support": int(support[class_index]),
        }
    return metrics


def _save_metrics(model_name: str, output_dir: Path, metrics: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{model_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    confusion_matrix_path = output_dir / f"{model_name}_confusion_matrix.csv"
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    df = pd.DataFrame(cm, index=metrics["class_names"], columns=metrics["class_names"])
    df.to_csv(confusion_matrix_path)


def _evaluate_model(
    model_name: str,
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    """Fit one classifier and calculate its test-set metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    per_class = _compute_class_metrics(y_test, y_pred, class_names)
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))

    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    return {
        "model": model_name,
        "feature_names": list(JET_FEATURE_NAMES),
        "class_names": class_names,
        "accuracy": float(accuracy),
        "precision": {
            "macro": float(precision_macro),
            "weighted": float(precision_weighted),
            "per_class": {name: values["precision"] for name, values in per_class.items()},
        },
        "recall": {
            "macro": float(recall_macro),
            "weighted": float(recall_weighted),
            "per_class": {name: values["recall"] for name, values in per_class.items()},
        },
        "f1": {
            "macro": float(f1_macro),
            "weighted": float(f1_weighted),
            "per_class": {name: values["f1"] for name, values in per_class.items()},
        },
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "class_specific": per_class,
    }


def run_baseline(
    data_path: str,
    output_dir: str | Path,
    *,
    seed: int = 42,
    max_samples: int | None = None,
) -> dict[str, dict[str, object]]:
    """Train and evaluate logistic regression and XGBoost jet-feature baselines."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_path, "r") as h5_file:
        jets = np.asarray(h5_file["jets"])

    event_numbers = jets["eventNumber"]
    train_indices, _, test_indices = split_by_event(event_numbers, seed=seed)

    if max_samples is not None:
        train_indices = limit_indices(train_indices, max_samples, seed)
        test_indices = limit_indices(test_indices, max_samples, seed + 1)

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Training or test split is empty after limiting samples.")

    X_train = extract_jet_features(jets[train_indices])
    X_test = extract_jet_features(jets[test_indices])

    y_train = encode_labels(jets["HadronGhostTruthLabelID"][train_indices])
    y_test = encode_labels(jets["HadronGhostTruthLabelID"][test_indices])

    class_names = list(CLASS_NAMES)
    model_outputs: dict[str, dict[str, object]] = {}

    for model_name, model in _build_models(seed).items():
        metrics = _evaluate_model(
            model_name, model, X_train, y_train, X_test, y_test, class_names
        )

        _save_metrics(model_name, output_path, metrics)
        model_outputs[model_name] = metrics

    return model_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline jet-feature models.")
    parser.add_argument("--data", required=True, help="Path to the HDF5 dataset.")
    parser.add_argument(
        "--output-dir",
        default="results/baseline",
        help="Directory where structured metrics artifacts are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible splitting and model training.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of jets per split to keep the baseline lightweight.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_baseline(
        data_path=args.data,
        output_dir=args.output_dir,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    for model_name, metrics in results.items():
        print(
            f"{model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['f1']['macro']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
