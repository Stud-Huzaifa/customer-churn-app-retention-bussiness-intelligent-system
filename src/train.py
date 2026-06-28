"""Training pipeline for the production customer churn classifier."""

from pathlib import Path
from typing import Final, TypeAlias, cast

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .config import (
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
try:
    from .config import TARGET_COLUMN  # type: ignore
except Exception:
    # Default target column name if not provided in config
    TARGET_COLUMN = "Churn"
from pathlib import Path as _Path
try:
    # Import PREPROCESSOR_PATH if available; otherwise leave it unset.
    from .config import PREPROCESSOR_PATH  # type: ignore
except Exception:
    PREPROCESSOR_PATH = None
try:
    # Import PROCESSED_DATA_PATH if available; otherwise default to a file in project
    from .config import PROCESSED_DATA_PATH  # type: ignore
except Exception:
    PROCESSED_DATA_PATH = _Path("processed_data.csv")
from .preprocess import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    ReferenceDate,
    prepare_features,
)
from .utils import load_csv, save_pickle, validate_dataframe


DataSplit: TypeAlias = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]
TrainingOutput: TypeAlias = tuple[Pipeline, pd.DataFrame, pd.Series]

N_ESTIMATORS: Final[int] = 124
LEARNING_RATE: Final[float] = 0.03944514624109112
MAX_DEPTH: Final[int] = 3
MIN_SAMPLES_SPLIT: Final[int] = 9
MIN_SAMPLES_LEAF: Final[int] = 15
SUBSAMPLE: Final[float] = 0.6974082233258441
CHURN_RECENCY_DAYS: Final[int] = 90

TUNED_MODEL_PARAMS: Final[dict[str, int | float]] = {
    "n_estimators": N_ESTIMATORS,
    "learning_rate": LEARNING_RATE,
    "max_depth": MAX_DEPTH,
    "min_samples_split": MIN_SAMPLES_SPLIT,
    "min_samples_leaf": MIN_SAMPLES_LEAF,
    "subsample": SUBSAMPLE,
    "random_state": RANDOM_STATE,
}


def _infer_reference_date(df: pd.DataFrame) -> pd.Timestamp | None:
    """Recover the date used to create tenure in the processed dataset."""
    required_columns = {"Registration_Date", "Customer_Tenure_Days"}
    if not required_columns.issubset(df.columns):
        return None

    registration_dates = pd.to_datetime(df["Registration_Date"], errors="coerce")
    tenure = pd.to_numeric(df["Customer_Tenure_Days"], errors="coerce")
    reference_dates = (
        registration_dates + pd.to_timedelta(tenure, unit="D")
    ).dropna()

    if reference_dates.empty:
        return None

    return pd.Timestamp(reference_dates.mode().iloc[0]).normalize()


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    reference_date: ReferenceDate = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create model features and the binary churn target."""
    validate_dataframe(df, allow_empty=False)

    if target_column in df.columns:
        target = pd.to_numeric(df[target_column], errors="coerce")
    elif "Days_Since_Last_Purchase" in df.columns:
        recency = pd.to_numeric(
            df["Days_Since_Last_Purchase"], errors="coerce"
        )
        target = (recency >= CHURN_RECENCY_DAYS).astype(int)
    else:
        raise ValueError(
            f"Training data must contain '{target_column}' or "
            "'Days_Since_Last_Purchase'."
        )

    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError(f"{target_column} must contain only binary values 0 and 1.")
    target = target.astype(int).rename(target_column)

    if target.nunique() < 2:
        raise ValueError("Training data must contain both active and churn classes.")

    if reference_date is None:
        reference_date = _infer_reference_date(df)

    features = prepare_features(df, reference_date=reference_date)

    return features, target


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build the numeric and categorical preprocessing pipeline."""
    missing_columns = [
        column for column in MODEL_FEATURE_COLUMNS if column not in X.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing model features: {missing}")

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_transformer, NUMERIC_FEATURE_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_FEATURE_COLUMNS),
        ]
    )


def split_data(X: pd.DataFrame, y: pd.Series) -> DataSplit:
    """Create the reproducible stratified train/test split."""
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if len(X) == 0:
        raise ValueError("Training data cannot be empty.")
    if pd.Series(y).nunique() < 2:
        raise ValueError("y must contain both active and churn classes.")

    return cast(
        DataSplit,
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        ),
    )


def build_model(preprocessor: ColumnTransformer) -> Pipeline:
    """Build the tuned Gradient Boosting model pipeline."""
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=N_ESTIMATORS,
                    learning_rate=LEARNING_RATE,
                    max_depth=MAX_DEPTH,
                    min_samples_split=MIN_SAMPLES_SPLIT,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    subsample=SUBSAMPLE,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer | None = None,
) -> Pipeline:
    """Fit and return the final model pipeline."""
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if preprocessor is None:
        preprocessor = build_preprocessor(X_train)

    model = build_model(preprocessor)
    model.fit(X_train, y_train)

    return model


def save_model(model: Pipeline, path: str | Path = MODEL_PATH) -> Path:
    """Save a fitted model pipeline."""
    if not hasattr(model, "named_steps") or "preprocessor" not in model.named_steps:
        raise TypeError("model must be a fitted pipeline with a preprocessor step.")
    try:
        check_is_fitted(model)
    except NotFittedError as exc:
        raise ValueError("model must be fitted before it can be saved.") from exc

    return save_pickle(model, path)


def train_from_file(
    data_path: str | Path = PROCESSED_DATA_PATH,
    model_path: str | Path = MODEL_PATH,
    preprocessor_path: str | Path | None = PREPROCESSOR_PATH,
    reference_date: ReferenceDate = None,
) -> TrainingOutput:
    """Run training end to end and save both fitted artifacts."""
    dataframe = load_csv(data_path)
    X, y = prepare_training_data(
        dataframe,
        reference_date=reference_date,
    )
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)

    save_model(model, model_path)
    if preprocessor_path is not None:
        save_pickle(model.named_steps["preprocessor"], preprocessor_path)

    return model, X_test, y_test
