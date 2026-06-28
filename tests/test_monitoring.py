import pandas as pd

from src.config import SAMPLE_INPUT_PATH, SAMPLE_OUTPUT_PATH
from src.monitoring import monitor_input_data, monitor_predictions


def test_valid_sample_input_monitoring() -> None:
    sample = pd.read_csv(SAMPLE_INPUT_PATH)

    report = monitor_input_data(sample)

    assert report.schema_valid
    assert report.values_valid
    assert report.missing_columns == ()
    assert report.missing_value_count == 1
    assert report.invalid_value_count == 0


def test_input_monitoring_detects_schema_and_value_errors() -> None:
    sample = pd.read_csv(SAMPLE_INPUT_PATH)
    sample = sample.drop(columns=["Age"])
    sample.loc[0, "Email_Open_Rate"] = 120
    sample.loc[1, "Registration_Date"] = "not-a-date"

    report = monitor_input_data(sample)

    assert not report.schema_valid
    assert not report.values_valid
    assert report.missing_columns == ("Age",)
    assert report.invalid_values_by_column["Email_Open_Rate"] == 1
    assert report.invalid_values_by_column["Registration_Date"] == 1


def test_prediction_monitoring_summary() -> None:
    predictions = pd.read_csv(SAMPLE_OUTPUT_PATH)

    report = monitor_predictions(predictions)

    assert report.prediction_count == 2
    assert report.predicted_churn_count == 0
    assert report.predicted_churn_percentage == 0.0
    assert report.high_risk_count == 0
    assert report.high_risk_percentage == 0.0
    assert report.average_churn_probability > 0
