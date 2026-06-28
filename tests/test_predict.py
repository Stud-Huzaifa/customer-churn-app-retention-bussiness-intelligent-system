import numpy as np
import pandas as pd

from src.config import SAMPLE_INPUT_PATH, SAMPLE_OUTPUT_PATH
from src.predict import clear_model_cache, load_model, predict_customers


def test_model_is_cached() -> None:
    clear_model_cache()

    first_model = load_model()
    second_model = load_model()

    assert first_model is second_model


def test_sample_predictions_match_expected_output() -> None:
    customers = pd.read_csv(SAMPLE_INPUT_PATH)
    expected = pd.read_csv(SAMPLE_OUTPUT_PATH)

    actual = predict_customers(
        customers,
        reference_date="2026-06-26",
        model=load_model(),
    )

    assert np.array_equal(actual["Predicted_Churn"], expected["Predicted_Churn"])
    assert np.allclose(
        actual["Churn_Probability"],
        expected["Churn_Probability"],
        atol=1e-12,
        rtol=0,
    )
    assert actual["Risk_Level"].tolist() == expected["Risk_Level"].tolist()
