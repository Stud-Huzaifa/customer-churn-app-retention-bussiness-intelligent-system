"""Inference services for customer churn prediction and risk segmentation."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .config import (
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    MODEL_PATH,
)
from .modeling import (
    ChurnProbabilities,
    Predictions,
    predict_with_probabilities,
    validate_binary_classifier,
)
from .preprocess import ReferenceDate, prepare_features
from .utils import load_csv, load_pickle, save_csv


@lru_cache(maxsize=None)
def _load_model_cached(model_path: Path) -> Any:
    """Load one model instance for each resolved artifact path."""
    model = load_pickle(model_path)
    validate_binary_classifier(model)
    return model


def load_model(model_path: str | Path = MODEL_PATH) -> Any:
    """Return a cached fitted model, loading it only on the first request."""
    return _load_model_cached(Path(model_path).resolve())


def clear_model_cache() -> None:
    """Clear cached models after an artifact is replaced or retrained."""
    _load_model_cached.cache_clear()


def _resolve_model(model: Any | None) -> Any:
    resolved_model = load_model() if model is None else model
    validate_binary_classifier(resolved_model)
    return resolved_model


def _predict_outputs(
    df: pd.DataFrame,
    reference_date: ReferenceDate = None,
    *,
    model: Any | None = None,
) -> tuple[Predictions, ChurnProbabilities]:
    features = prepare_features(df, reference_date=reference_date)
    return predict_with_probabilities(_resolve_model(model), features)


def predict(
    df: pd.DataFrame,
    reference_date: ReferenceDate = None,
    *,
    model: Any | None = None,
) -> Predictions:
    """Predict whether each supplied customer will churn."""
    predictions, _ = _predict_outputs(
        df, reference_date=reference_date, model=model
    )
    return predictions


def predict_probability(
    df: pd.DataFrame,
    reference_date: ReferenceDate = None,
    *,
    model: Any | None = None,
) -> ChurnProbabilities:
    """Predict positive-class churn probabilities for supplied customers."""
    _, probabilities = _predict_outputs(
        df, reference_date=reference_date, model=model
    )
    return probabilities


def assign_risk(probability: float) -> str:
    """Convert one churn probability into its configured business risk level."""
    try:
        value = float(probability)
    except (TypeError, ValueError) as exc:
        raise ValueError("Churn probability must be numeric.") from exc

    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Churn probability must be between 0 and 1.")
    if value <= LOW_RISK_THRESHOLD:
        return "Low Risk"
    if value <= HIGH_RISK_THRESHOLD:
        return "Medium Risk"
    return "High Risk"


def assign_risk_levels(probabilities: ArrayLike) -> NDArray[np.str_]:
    """Vectorize risk assignment for a one-dimensional probability array."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1:
        raise ValueError("Churn probabilities must be one-dimensional.")
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Churn probabilities must be between 0 and 1.")

    return np.select(
        [values <= LOW_RISK_THRESHOLD, values <= HIGH_RISK_THRESHOLD],
        ["Low Risk", "Medium Risk"],
        default="High Risk",
    ).astype(np.str_)


def predict_customers(
    df: pd.DataFrame,
    reference_date: ReferenceDate = None,
    *,
    model: Any | None = None,
) -> pd.DataFrame:
    """Append churn predictions, probabilities, and risk levels to customer data."""
    predictions, probabilities = _predict_outputs(
        df, reference_date=reference_date, model=model
    )

    result = df.copy()
    result["Predicted_Churn"] = predictions
    result["Churn_Probability"] = probabilities
    result["Risk_Level"] = assign_risk_levels(probabilities)
    return result


def predict_csv(
    input_path: str | Path,
    output_path: str | Path | None = None,
    reference_date: ReferenceDate = None,
    *,
    model: Any | None = None,
) -> pd.DataFrame:
    """Predict customers from a CSV file and save the completed report."""
    customers = load_csv(input_path)
    result = predict_customers(
        customers,
        reference_date=reference_date,
        model=model,
    )
    # Use a sensible default output path if none supplied
    out_path = Path(output_path) if output_path is not None else Path.cwd() / "predictions.csv"
    save_csv(result, out_path)
    return result
