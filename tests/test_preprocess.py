import pandas as pd
import pytest

from src.config import SAMPLE_INPUT_PATH
from src.preprocess import MODEL_FEATURE_COLUMNS, prepare_features


def test_prepare_features_matches_model_contract_without_mutating_input() -> None:
    customers = pd.read_csv(SAMPLE_INPUT_PATH)
    original = customers.copy(deep=True)

    features = prepare_features(customers, reference_date="2026-06-26")

    assert features.columns.tolist() == MODEL_FEATURE_COLUMNS
    assert len(features) == len(customers)
    pd.testing.assert_frame_equal(customers, original)


def test_prepare_features_reports_missing_columns() -> None:
    customers = pd.read_csv(SAMPLE_INPUT_PATH).drop(columns=["Age"])

    with pytest.raises(ValueError, match="Missing required model columns: Age"):
        prepare_features(customers, reference_date="2026-06-26")
