"""Evaluation helpers for fitted binary churn classifiers."""

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .modeling import ChurnProbabilities, Predictions, predict_with_probabilities


METRIC_ORDER: Final[tuple[str, ...]] = (
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC AUC",
)


def _validate_evaluation_data(X_test: Any, y_test: ArrayLike) -> NDArray[Any]:
    """Validate binary evaluation labels and row counts."""
    y_true = np.asarray(y_test)
    if y_true.ndim != 1:
        raise ValueError("y_test must be one-dimensional.")
    if len(y_true) == 0:
        raise ValueError("Evaluation data cannot be empty.")
    if len(X_test) != len(y_true):
        raise ValueError("X_test and y_test must contain the same number of rows.")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("y_test must contain only binary labels 0 and 1.")

    return y_true


def _evaluation_outputs(
    model: Any, X_test: Any, y_test: ArrayLike
) -> tuple[NDArray[Any], Predictions, ChurnProbabilities]:
    y_true = _validate_evaluation_data(X_test, y_test)
    y_pred, y_probability = predict_with_probabilities(model, X_test)
    if y_pred.shape != y_true.shape:
        raise ValueError("Model predictions do not match the shape of y_test.")

    return y_true, y_pred, y_probability


def evaluate_model(
    model: Any, X_test: Any, y_test: ArrayLike
) -> dict[str, float]:
    """Calculate the project metrics for a fitted binary churn classifier."""
    y_true, y_pred, y_probability = _evaluation_outputs(
        model, X_test, y_test
    )
    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC AUC requires both active and churn classes in y_test.")

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "F1 Score": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "ROC AUC": float(roc_auc_score(y_true, y_probability)),
    }


def evaluation_dataframe(metrics: Mapping[str, float]) -> pd.DataFrame:
    """Convert a metric mapping into a consistently ordered tidy DataFrame."""
    if not metrics:
        raise ValueError("metrics must be a non-empty mapping.")

    ordered_metrics = [metric for metric in METRIC_ORDER if metric in metrics]
    ordered_metrics.extend(metric for metric in metrics if metric not in METRIC_ORDER)

    return pd.DataFrame(
        {
            "Metric": ordered_metrics,
            "Value": [float(metrics[name]) for name in ordered_metrics],
        }
    )


def confusion_matrix_dataframe(
    model: Any, X_test: Any, y_test: ArrayLike
) -> pd.DataFrame:
    """Return a consistently labelled 2-by-2 binary confusion matrix."""
    y_true, y_pred, _ = _evaluation_outputs(model, X_test, y_test)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return pd.DataFrame(
        matrix,
        index=["Actual Active", "Actual Churn"],
        columns=["Predicted Active", "Predicted Churn"],
    )


def compare_models(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return model evaluation records ranked by descending F1 score."""
    if not results:
        raise ValueError("results must be a non-empty sequence of mappings.")

    comparison = pd.DataFrame(results)
    required_columns = {"Model", "F1 Score"}
    missing_columns = required_columns - set(comparison.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing comparison columns: {missing}")

    comparison["F1 Score"] = pd.to_numeric(
        comparison["F1 Score"], errors="raise"
    )
    return comparison.sort_values(
        by="F1 Score", ascending=False
    ).reset_index(drop=True)
