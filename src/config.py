from pathlib import Path

# Phase A: Define the main project paths in one place
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"
LABELS_PATH = RAW_DIR / "labeled_anomalies.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Phase B: Store core experiment settings here for easy reuse
RANDOM_STATE = 42
WINDOW_SIZE = 30
WINDOW_STRIDE = 5
SCALER_NAME = "standard"
