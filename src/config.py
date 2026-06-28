from pathlib import Path


# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

# File paths
RAW_DATA_PATH = DATA_DIR / "raw" / "customer_churn_raw.csv"
CLEAN_DATA_PATH = DATA_DIR / "processed" / "customer_churn_clean.csv"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"

# App settings
APP_TITLE = "Customer Churn Prediction"
APP_DESCRIPTION = "Predict customer churn risk for an online fashion store."

# Risk thresholds
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.20