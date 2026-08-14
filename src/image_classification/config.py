import random

import tensorflow as tf


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