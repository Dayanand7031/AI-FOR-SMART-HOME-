import os
from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).parent.resolve()

# Data Directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model Directories
MODELS_DIR = ROOT_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Output Directories
OUTPUTS_DIR = ROOT_DIR / "outputs"

# NASA POWER API Configuration
# The endpoint changes based on resolution: daily, monthly, yearly
NASA_BASE_URL = "https://power.larc.nasa.gov/api/temporal"
NASA_PARAMS = [
    "ALLSKY_SFC_SW_DWN",
    "T2M",
    "RH2M",
    "WS10M",
    "PRECTOTCORR",
    "PS",
    "CLOUD_AMT"
]

# Default Location and Range (Can be overridden by user)
DEFAULT_LAT = 40.7128  # New York City example
DEFAULT_LON = -74.0060
START_DATE = "20210101" # YYYYMMDD
END_DATE = "20260508"   # YYYYMMDD

# Model Config
CANDIDATE_MODELS = {
    "ML": ["xgboost", "random_forest", "lightgbm"],
    "DL": ["lstm", "gru", "cnn_lstm"]
}
SEQUENCE_LENGTH = 7 # Days for DL models
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42
