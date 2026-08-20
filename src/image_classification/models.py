import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Rescaling,
    Flatten,
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Lambda,
    GlobalAveragePooling2D
)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from src.image_classification.config import RANDOM_STATE

# ML models
ml_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=20,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    ),

    "Support Vector Machine": SVC(
        kernel="rbf",
        random_state=RANDOM_STATE
    )
}

# ANN Model
def create_ann_model(img_size, number_of_classes):
    model = Sequential([
        Input(shape=(img_size, img_size, 3)),
        Rescaling(1. / 255),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(number_of_classes, activation="softmax")
    ])
    return model

# CNN Model
def create_cnn_model(image_size, number_of_classes):
    model = Sequential([
        Input(shape=(image_size, image_size, 3)),
        Rescaling(1. / 255),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(number_of_classes, activation="softmax")
    ])

    return model

# VGG16
def create_vgg16_model(
    image_size: int,
    number_of_classes: int
) -> tf.keras.Model:

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(
            image_size,
            image_size,
            3
        )
    )

    base_model.trainable = False

    model = Sequential([
        Input(
            shape=(
                image_size,
                image_size,
                3
            )
        ),

        base_model,

        GlobalAveragePooling2D(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(
            0.3
        ),

        Dense(
            number_of_classes,
            activation="softmax"
        )
    ])

    return model