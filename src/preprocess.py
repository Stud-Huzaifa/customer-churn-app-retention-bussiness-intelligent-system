"""Feature engineering shared by model training and inference."""

from datetime import date, datetime
from typing import Final, TypeAlias

import numpy as np
import pandas as pd

from .utils import validate_dataframe


ReferenceDate: TypeAlias = str | date | datetime | pd.Timestamp | None

RAW_INPUT_COLUMNS: Final[list[str]] = [
    "Customer_ID",
    "Name",
    "Age",
    "Gender",
    "Location",
    "Registration_Date",
    "Last_Purchase_Date",
    "Purchase_Frequency",
    "Total_Spending",
    "Average_Order_Value",
    "Website_Visits",
    "Email_Open_Rate",
    "Customer_Support_Tickets",
]
REQUIRED_INPUT_COLUMNS: Final[list[str]] = [
    "Age",
    "Gender",
    "Location",
    "Registration_Date",
    "Purchase_Frequency",
    "Total_Spending",
    "Average_Order_Value",
    "Website_Visits",
    "Email_Open_Rate",
    "Customer_Support_Tickets",
]

DROP_COLUMNS: Final[list[str]] = [
    "Customer_ID",
    "Name",
    "Registration_Date",
    "Last_Purchase_Date",
    "Age_Group",
    "Never_Purchased",
    "Purchase_Recency_Group",
    "Purchase_Segment",
    "Spending_Segment",
    "AOV_Segment",
    "Website_Engagement",
    "Email_Engagement",
    "Support_Level",
    "Days_Since_Last_Purchase",
    "Churn",
]

# The order must match feature_names_in_ stored in churn_model.pkl.
MODEL_FEATURE_COLUMNS: Final[list[str]] = [
    "Age",
    "Gender",
    "Location",
    "Purchase_Frequency",
    "Total_Spending",
    "Average_Order_Value",
    "Website_Visits",
    "Email_Open_Rate",
    "Customer_Support_Tickets",
    "Customer_Tenure_Days",
    "Total_Spending_Log",
    "Average_Order_Value_Log",
    "Email_Open_Rate_Missing",
]
NUMERIC_FEATURE_COLUMNS: Final[list[str]] = [
    "Age",
    "Purchase_Frequency",
    "Total_Spending",
    "Average_Order_Value",
    "Website_Visits",
    "Email_Open_Rate",
    "Customer_Support_Tickets",
    "Customer_Tenure_Days",
    "Total_Spending_Log",
    "Average_Order_Value_Log",
    "Email_Open_Rate_Missing",
]
CATEGORICAL_FEATURE_COLUMNS: Final[list[str]] = ["Gender", "Location"]

LOCATION_MAPPING: Final[dict[str, str]] = {
    "ny": "new york",
    "newyork": "new york",
    "la": "los angeles",
    "uae-dubai": "dubai",
}


def _get_reference_date(reference_date: ReferenceDate = None) -> pd.Timestamp:
    """Normalize a supplied date or return today's local date."""
    if reference_date is None:
        return pd.Timestamp.today().normalize()

    try:
        timestamp = pd.Timestamp(reference_date)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid reference date: {reference_date!r}") from exc

    if pd.isna(timestamp):
        raise ValueError(f"Invalid reference date: {reference_date!r}")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)

    return timestamp.normalize()


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse a date series and remove timezone metadata when present."""
    dates = pd.to_datetime(series, errors="coerce")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_localize(None)

    return dates


def _add_customer_tenure(
    df: pd.DataFrame, reference_date: ReferenceDate
) -> None:
    if "Registration_Date" not in df.columns:
        raise ValueError("Missing required column: Registration_Date")

    registration_dates = _parse_dates(df["Registration_Date"])
    df["Registration_Date"] = registration_dates
    df["Customer_Tenure_Days"] = (
        _get_reference_date(reference_date) - registration_dates
    ).dt.days


def _add_days_since_last_purchase(
    df: pd.DataFrame, reference_date: ReferenceDate
) -> None:
    if "Last_Purchase_Date" not in df.columns:
        raise ValueError("Missing required column: Last_Purchase_Date")

    today = _get_reference_date(reference_date)
    purchase_dates = _parse_dates(df["Last_Purchase_Date"])
    valid_dates = purchase_dates.where(purchase_dates <= today)
    df["Last_Purchase_Date"] = purchase_dates
    df["Days_Since_Last_Purchase"] = (today - valid_dates).dt.days


def _add_log_features(df: pd.DataFrame) -> None:
    for source, destination in (
        ("Total_Spending", "Total_Spending_Log"),
        ("Average_Order_Value", "Average_Order_Value_Log"),
    ):
        if source in df.columns:
            values = pd.to_numeric(df[source], errors="coerce")
            df[source] = values
            df[destination] = np.log1p(values.where(values >= 0))


def _add_email_missing_flag(df: pd.DataFrame) -> None:
    if "Email_Open_Rate" not in df.columns:
        return

    email_open_rate = pd.to_numeric(df["Email_Open_Rate"], errors="coerce")
    df["Email_Open_Rate"] = email_open_rate
    derived_flag = email_open_rate.isna().astype(int)

    if "Email_Open_Rate_Missing" not in df.columns:
        df["Email_Open_Rate_Missing"] = derived_flag
        return

    existing_flag = pd.to_numeric(
        df["Email_Open_Rate_Missing"], errors="coerce"
    )
    existing_flag = existing_flag.where(existing_flag.isin([0, 1]), 0)
    df["Email_Open_Rate_Missing"] = np.maximum(
        existing_flag, derived_flag
    ).astype(int)


def _normalize_categorical_features(df: pd.DataFrame) -> None:
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype("string").str.strip().str.title()

    if "Location" in df.columns:
        locations = (
            df["Location"]
            .astype("string")
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        df["Location"] = locations.replace(LOCATION_MAPPING)


def create_customer_tenure(
    df: pd.DataFrame, reference_date: ReferenceDate = None
) -> pd.DataFrame:
    """Return a copy with customer tenure calculated as it was in training."""
    validate_dataframe(df)
    result = df.copy()
    _add_customer_tenure(result, reference_date)
    return result


def create_days_since_last_purchase(
    df: pd.DataFrame, reference_date: ReferenceDate = None
) -> pd.DataFrame:
    """Return a copy with purchase recency for valid non-future dates."""
    validate_dataframe(df)
    result = df.copy()
    _add_days_since_last_purchase(result, reference_date)
    return result


def create_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with safe log-transformed spending features."""
    validate_dataframe(df)
    result = df.copy()
    _add_log_features(result)
    return result


def create_email_missing_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the original email missingness indicator preserved."""
    validate_dataframe(df)
    result = df.copy()
    _add_email_missing_flag(result)
    return result


def normalize_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with categories normalized as they were during training."""
    validate_dataframe(df)
    result = df.copy()
    _normalize_categorical_features(result)
    return result


def prepare_features(
    df: pd.DataFrame, reference_date: ReferenceDate = None
) -> pd.DataFrame:
    """Create and order the exact feature set expected by the trained model."""
    validate_dataframe(df, allow_empty=False)
    features = df.copy()

    _add_customer_tenure(features, reference_date)
    if "Last_Purchase_Date" in features.columns:
        # Recency defines the target and is removed below to prevent leakage.
        _add_days_since_last_purchase(features, reference_date)
    _add_log_features(features)
    _add_email_missing_flag(features)
    _normalize_categorical_features(features)

    features = features.drop(columns=DROP_COLUMNS, errors="ignore")
    missing_columns = [
        column for column in MODEL_FEATURE_COLUMNS if column not in features.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required model columns: {missing}")

    return features.loc[:, MODEL_FEATURE_COLUMNS]
