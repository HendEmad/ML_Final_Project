from pathlib import Path
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (PROJECT_ROOT/"data"/"telco_customer_churn"/"WA_Fn-UseC_-Telco-Customer-Churn.csv")

TARGET_COLUMN = "Churn"

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 20

ARTIFACTS_DIR = (PROJECT_ROOT/"artifacts"/"churn_prediction")
MODELS_DIR = (ARTIFACTS_DIR/"models")
PREPROCESSORS_DIR = (ARTIFACTS_DIR/"preprocessors")
RESULTS_DIR = (ARTIFACTS_DIR/"results")
LOG_DIR = (PROJECT_ROOT/"logs"/"churn_prediction")

for directory in [MODELS_DIR, PREPROCESSORS_DIR, RESULTS_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)