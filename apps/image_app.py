from pathlib import Path
import cv2
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf

from src.common.streamlit_utils import (load_json, setup_page)
from src.image_classification.preprocessing import extract_hog_features
from tensorflow.keras.applications.vgg16 import preprocess_input

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = (PROJECT_ROOT/"artifacts"/"image_classification")
MODELS_DIR = (ARTIFACTS_DIR/"models")
PREPROCESSORS_DIR = (ARTIFACTS_DIR/"preprocessors")
RESULTS_DIR = (ARTIFACTS_DIR/"results")

setup_page("Scene Classification")
metadata = load_json(ARTIFACTS_DIR/"metadata.json")
CLASS_NAMES = (metadata["class_names"])
IMAGE_SIZE = (metadata["image_size"])

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

model_name = st.selectbox(
    "Choose Model",
    ["Best ML", "CNN", "VGG16"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):
        resized_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        if model_name == "Best ML":
            scaler = joblib.load(PREPROCESSORS_DIR/"hog_scaler.pkl")
            model = joblib.load(MODELS_DIR/"best_ml_model.pkl")

            hog_features = (extract_hog_features(np.array([resized_image])))
            hog_features = (scaler.transform(hog_features))

            prediction = model.predict(hog_features)[0]
        elif model_name == "CNN":
            model = (tf.keras.models.load_model(MODELS_DIR/"cnn_model.keras"))

            input_image = np.expand_dims(resized_image, axis=0)
            probabilities = model.predict(input_image, verbose=0)
            prediction = np.argmax(probabilities, axis=1)[0]
        elif model_name == "VGG16":
            model = tf.keras.models.load_model(MODELS_DIR/"vgg16_model.keras",compile=False)
            input_image = np.expand_dims(resized_image.astype("float32"), axis=0)
            input_image = preprocess_input(input_image)
            probabilities = model.predict(input_image, verbose=0)
            prediction = np.argmax(probabilities, axis=1)[0]

        predicted_class = (CLASS_NAMES[prediction])
        st.success(
            f"Prediction: "
            f"{predicted_class}"
        )        