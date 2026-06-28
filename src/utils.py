"""Generic file, validation, and prediction-report utilities."""

from pathlib import Path
from typing import Any, Collection, Final

import joblib
import pandas as pd


RISK_LEVELS: Final[tuple[str, ...]] = (
    "Low Risk",
    "Medium Risk",
    "High Risk",
)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Collection[str] = (),
    *,
    allow_empty: bool = True,
) -> None:
    """Validate a DataFrame and its required columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not allow_empty and df.empty:
        raise ValueError("DataFrame cannot be empty.")

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}")


def load_csv(path: str | Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a CSV file after confirming that it exists."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file was not found: {path}")

    return pd.read_csv(path, **read_csv_kwargs)


def save_csv(
    df: pd.DataFrame, path: str | Path, **to_csv_kwargs: Any
) -> Path:
    """Save a DataFrame to CSV, creating its parent directory if needed."""
    validate_dataframe(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_csv_kwargs.setdefault("index", False)
    df.to_csv(path, **to_csv_kwargs)

    return path


def load_pickle(path: str | Path) -> Any:
    """Load a trusted joblib/pickle artifact."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact was not found: {path}")

    return joblib.load(path)


def save_pickle(obj: Any, path: str | Path) -> Path:
    """Save an object as a joblib artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

    return path


def risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize customer counts in business risk order."""
    validate_dataframe(df, required_columns=["Risk_Level"])

    if df["Risk_Level"].isna().any():
        raise ValueError("Risk_Level cannot contain missing values.")

    invalid_levels = set(df["Risk_Level"].dropna()) - set(RISK_LEVELS)
    if invalid_levels:
        invalid = ", ".join(sorted(map(str, invalid_levels)))
        raise ValueError(f"Unknown risk levels: {invalid}")

    counts = df["Risk_Level"].value_counts().reindex(RISK_LEVELS, fill_value=0)

    return counts.rename_axis("Risk Level").reset_index(name="Customers")


def model_summary(df: pd.DataFrame) -> dict[str, int | float]:
    """Calculate dashboard metrics from a prediction report."""
    required_columns = ["Predicted_Churn", "Churn_Probability"]
    validate_dataframe(df, required_columns=required_columns)

    predictions = pd.to_numeric(df["Predicted_Churn"], errors="coerce")
    probabilities = pd.to_numeric(df["Churn_Probability"], errors="coerce")

    if predictions.isna().any() or not predictions.isin([0, 1]).all():
        raise ValueError("Predicted_Churn must contain only 0 and 1 values.")
    if probabilities.isna().any() or not probabilities.between(0, 1).all():
        raise ValueError("Churn_Probability must contain values between 0 and 1.")

    total_customers = len(df)
    predicted_churn = int(predictions.sum())
    active_customers = total_customers - predicted_churn
    churn_rate = (
        round((predicted_churn / total_customers) * 100, 2)
        if total_customers
        else 0.0
    )
    average_probability = (
        round(float(probabilities.mean()) * 100, 2) if total_customers else 0.0
    )

    return {
        "Total Customers": total_customers,
        "Predicted Churn": predicted_churn,
        "Active Customers": active_customers,
        "Churn Rate (%)": churn_rate,
        "Average Churn Probability (%)": average_probability,
    }


def high_risk_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Return high-risk customers ordered by descending churn probability."""
    required_columns = ["Risk_Level", "Churn_Probability"]
    validate_dataframe(df, required_columns=required_columns)

    return (
        df.loc[df["Risk_Level"] == "High Risk"]
        .sort_values("Churn_Probability", ascending=False)
        .copy()
    )


def export_predictions(df: pd.DataFrame, filepath: str | Path) -> Path:
    """Export a prediction report to CSV."""
    required_columns = [
        "Predicted_Churn",
        "Churn_Probability",
        "Risk_Level",
    ]
    validate_dataframe(df, required_columns=required_columns)

    return save_csv(df, filepath)
