import random
import tensorflow as tf
from pathlib import Path

TRAIN_DATA_PATH = (r"data\Intel_image_data\seg_train\seg_train")
TEST_DATA_PATH = (r"data\Intel_image_data\seg_test\seg_test")
IMAGE_SIZE = 64
BATCH_SIZE = 32
VALIDATION_SIZE = 0.2
MAX_IMAGES_PER_CLASS = 1000
RANDOM_STATE = 42
CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]
NUMBER_OF_CLASSES = len(CLASS_NAMES)

random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = (PROJECT_ROOT/ "artifacts"/ "image_classification")

MODELS_DIR = (ARTIFACTS_DIR/ "models")

PREPROCESSORS_DIR = (ARTIFACTS_DIR/ "preprocessors")

RESULTS_DIR = (ARTIFACTS_DIR/"results")

LOG_DIR = (PROJECT_ROOT/"logs"/"image_classification")

for directory in [MODELS_DIR, PREPROCESSORS_DIR, RESULTS_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)