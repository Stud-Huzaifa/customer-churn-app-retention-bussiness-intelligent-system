"""Lightweight data-quality and prediction monitoring for the churn workflow."""

from dataclasses import dataclass
from typing import Final

import pandas as pd

from .preprocess import REQUIRED_INPUT_COLUMNS
from .utils import validate_dataframe


NUMERIC_RANGES: Final[dict[str, tuple[float, float | None]]] = {
    "Age": (0, 120),
    "Purchase_Frequency": (0, None),
    "Total_Spending": (0, None),
    "Average_Order_Value": (0, None),
    "Website_Visits": (0, None),
    "Email_Open_Rate": (0, 100),
    "Customer_Support_Tickets": (0, None),
}
DATE_COLUMNS: Final[tuple[str, ...]] = (
    "Registration_Date",
    "Last_Purchase_Date",
)
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = ("Gender", "Location")


@dataclass(frozen=True)
class InputMonitoringReport:
    """Validation results for one uploaded customer dataset."""

    row_count: int
    column_count: int
    missing_columns: tuple[str, ...]
    missing_value_count: int
    invalid_value_count: int
    invalid_values_by_column: dict[str, int]

    @property
    def schema_valid(self) -> bool:
        """Return whether every required input column is present."""
        return not self.missing_columns

    @property
    def values_valid(self) -> bool:
        """Return whether all non-missing values satisfy basic constraints."""
        return self.invalid_value_count == 0


@dataclass(frozen=True)
class PredictionMonitoringReport:
    """Aggregate monitoring metrics for one prediction run."""

    prediction_count: int
    predicted_churn_count: int
    predicted_churn_percentage: float
    high_risk_count: int
    high_risk_percentage: float
    average_churn_probability: float


def monitor_input_data(df: pd.DataFrame) -> InputMonitoringReport:
    """Check schema, missing values, parseability, and sensible value ranges."""
    validate_dataframe(df)
    missing_columns = tuple(
        column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns
    )
    invalid_by_column: dict[str, int] = {}

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        if column not in df.columns:
            continue

        original = df[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        invalid |= numeric.notna() & (numeric < minimum)
        if maximum is not None:
            invalid |= numeric.notna() & (numeric > maximum)

        count = int(invalid.sum())
        if count:
            invalid_by_column[column] = count

    for column in DATE_COLUMNS:
        if column not in df.columns:
            continue

        original = df[column]
        parsed = pd.to_datetime(original, errors="coerce")
        count = int((original.notna() & parsed.isna()).sum())
        if count:
            invalid_by_column[column] = count

    for column in CATEGORICAL_COLUMNS:
        if column not in df.columns:
            continue

        values = df[column].astype("string")
        count = int((values.notna() & values.str.strip().eq("")).sum())
        if count:
            invalid_by_column[column] = count

    return InputMonitoringReport(
        row_count=len(df),
        column_count=len(df.columns),
        missing_columns=missing_columns,
        missing_value_count=int(df.isna().sum().sum()),
        invalid_value_count=sum(invalid_by_column.values()),
        invalid_values_by_column=invalid_by_column,
    )


def monitor_predictions(df: pd.DataFrame) -> PredictionMonitoringReport:
    """Calculate aggregate output and risk metrics for a prediction run."""
    required_columns = [
        "Predicted_Churn",
        "Churn_Probability",
        "Risk_Level",
    ]
    validate_dataframe(df, required_columns=required_columns, allow_empty=False)

    predictions = pd.to_numeric(df["Predicted_Churn"], errors="coerce")
    probabilities = pd.to_numeric(df["Churn_Probability"], errors="coerce")
    if predictions.isna().any() or not predictions.isin([0, 1]).all():
        raise ValueError("Predicted_Churn must contain only 0 and 1 values.")
    if probabilities.isna().any() or not probabilities.between(0, 1).all():
        raise ValueError("Churn_Probability must contain values between 0 and 1.")

    prediction_count = len(df)
    predicted_churn_count = int(predictions.sum())
    high_risk_count = int(df["Risk_Level"].eq("High Risk").sum())

    return PredictionMonitoringReport(
        prediction_count=prediction_count,
        predicted_churn_count=predicted_churn_count,
        predicted_churn_percentage=round(
            predicted_churn_count / prediction_count * 100,
            2,
        ),
        high_risk_count=high_risk_count,
        high_risk_percentage=round(
            high_risk_count / prediction_count * 100,
            2,
        ),
        average_churn_probability=round(float(probabilities.mean()) * 100, 2),
    )
