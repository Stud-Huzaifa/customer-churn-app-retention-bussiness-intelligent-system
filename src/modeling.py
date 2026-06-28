"""Shared validation and inference helpers for binary churn classifiers."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


Predictions = NDArray[Any]
ChurnProbabilities = NDArray[np.float64]


def validate_binary_classifier(model: Any) -> None:
    """Validate the interface required from a fitted churn classifier."""
    required_methods = ("predict", "predict_proba")
    missing_methods = [
        name for name in required_methods if not callable(getattr(model, name, None))
    ]
    if missing_methods:
        missing = ", ".join(missing_methods)
        raise TypeError(f"Model does not support: {missing}")

    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        raise ValueError("The model does not contain the positive churn class (1).")


def predict_with_probabilities(
    model: Any, features: Any
) -> tuple[Predictions, ChurnProbabilities]:
    """Return binary predictions and positive-class probabilities in one pass."""
    validate_binary_classifier(model)

    predictions = np.asarray(model.predict(features))
    probabilities = np.asarray(model.predict_proba(features))
    classes = list(model.classes_)

    if predictions.ndim != 1:
        raise ValueError("Model predictions must be one-dimensional.")
    if probabilities.ndim != 2 or probabilities.shape[0] != len(predictions):
        raise ValueError("Model probabilities do not match its predictions.")

    positive_index = classes.index(1)
    if positive_index >= probabilities.shape[1]:
        raise ValueError("Positive churn probability column is missing.")

    churn_probabilities = probabilities[:, positive_index].astype(
        np.float64, copy=False
    )
    if not np.isfinite(churn_probabilities).all() or not (
        (churn_probabilities >= 0) & (churn_probabilities <= 1)
    ).all():
        raise ValueError("Model returned invalid churn probabilities.")

    return predictions, churn_probabilities
