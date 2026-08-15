import joblib
import json
import pandas as pd
from src.common.evaluation import (model_results, compare_models)
from src.common.logging_config import setup_logger
from src.image_classification.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    NUMBER_OF_CLASSES,
    MODELS_DIR,
    PREPROCESSORS_DIR,
    RESULTS_DIR,
    LOG_DIR,
    CLASS_NAMES
)
from src.image_classification.data import (data_loader,tensorflow_dataset_to_numpy)
from src.image_classification.preprocessing import (extract_hog_features, create_hog_scaler)
from src.image_classification.models import (
    ml_models,
    create_ann_model,
    create_cnn_model,
    create_vgg16_model
)
from src.image_classification.train import (train_ml_model, train_deep_learning_model)

logger = setup_logger(
    logger_name="image_training",
    log_file=str(
        LOG_DIR / "training.log"
    )
)

def main() -> None:
    logger.info(
        "Image classification training started."
    )

    model_results.clear()
    logger.info("Loading datasets.")
    train_dataset, validation_dataset, test_dataset = data_loader(
        train_path=TRAIN_DATA_PATH,
        test_path=TEST_DATA_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    # ------------------ ML Part ------------------
    logger.info("Converting TensorFlow datasets to NumPy.")
    x_train_ml, y_train_ml = (tensorflow_dataset_to_numpy(train_dataset))
    x_validation_ml, y_validation_ml = (tensorflow_dataset_to_numpy(validation_dataset))
    x_test_ml, y_test_ml = (tensorflow_dataset_to_numpy(test_dataset))

    logger.info("Extracting HOG features.")
    x_train_hog = extract_hog_features(x_train_ml)
    x_validation_hog = extract_hog_features(x_validation_ml)
    x_test_hog = extract_hog_features(x_test_ml)

    logger.info("Scaling HOG features.")
    hog_scaler = create_hog_scaler()
    x_train_hog_scaled = (hog_scaler.fit_transform(x_train_hog))
    x_validation_hog_scaled = (hog_scaler.transform(x_validation_hog))
    x_test_hog_scaled = (hog_scaler.transform(x_test_hog))

    scaler_path = (PREPROCESSORS_DIR/"hog_scaler.pkl")
    joblib.dump(hog_scaler, scaler_path)
    logger.info(f"HOG scaler saved to {scaler_path}")

    # ------------------ ML Models training & saving best model ------------------
    logger.info("Training classical ML models.")
    trained_ml_models = {}
    ml_results = {}
    for model_name, model in ml_models.items():
        logger.info(f"Training {model_name}")
        trained_model, predictions, results = train_ml_model(
            model_name=model_name,
            model=model,
            x_train=x_train_hog_scaled,
            y_train=y_train_ml,
            x_test=x_test_hog_scaled,
            y_test=y_test_ml
        )
        trained_ml_models[model_name] = trained_model
        ml_results[model_name] = results

    best_ml_name = max(ml_results, 
                       key=lambda name: (ml_results[name]["F1 Score"]))
    best_ml_model = (trained_ml_models[best_ml_name])
    best_ml_path = (MODELS_DIR/"best_ml_model.pkl")
    joblib.dump(best_ml_model, best_ml_path)
    logger.info(f"Best ML model: {best_ml_name}")
    logger.info(
        f"Best ML model saved to "
        f"{best_ml_path}"
    )

    with open(RESULTS_DIR/"best_ml_model.txt", "w", encoding="utf-8") as file:
        file.write(best_ml_name)

    # ------------------ DL Models training + save them ------------------
    # ------------------ ANN Model ------------------
    logger.info("Training ANN.")
    ann_model = create_ann_model(img_size=IMAGE_SIZE, number_of_classes=NUMBER_OF_CLASSES)
    ann_model, ann_history, ann_predictions, ann_results = train_deep_learning_model(
        model_name="ANN",
        model=ann_model,
        x_train=x_train_ml,
        y_train=y_train_ml,
        x_validation=x_validation_ml,
        y_validation=y_validation_ml,
        x_test=x_test_ml,
        y_test=y_test_ml,
        epochs=15,
        batch_size=BATCH_SIZE
    )

    ann_path = (MODELS_DIR/"ann_model.keras")
    ann_model.save(ann_path)
    logger.info(f"ANN saved to {ann_path}")

    # ------------------ CNN Model ------------------
    logger.info("Training CNN.")
    cnn_model = create_cnn_model(image_size=IMAGE_SIZE, number_of_classes=NUMBER_OF_CLASSES)
    cnn_model, cnn_history, cnn_predictions, cnn_results = train_deep_learning_model(
        model_name="CNN",
        model=cnn_model,
        x_train=x_train_ml,
        y_train=y_train_ml,
        x_validation=x_validation_ml,
        y_validation=y_validation_ml,
        x_test=x_test_ml,
        y_test=y_test_ml,
        epochs=15,
        batch_size=BATCH_SIZE
    )

    cnn_path = (MODELS_DIR/"cnn_model.keras")
    cnn_model.save(cnn_path)
    logger.info(f"CNN saved to {cnn_path}")

    # ------------------ VGG16 model ------------------
    logger.info(
        "Training VGG16 transfer learning model."
    )

    vgg16_model = create_vgg16_model(
        image_size=IMAGE_SIZE,
        number_of_classes=NUMBER_OF_CLASSES
    )

    vgg16_model, vgg16_history, vgg16_predictions, vgg16_results = train_deep_learning_model(
        model_name="VGG16",
        model=vgg16_model,
        x_train=x_train_ml,
        y_train=y_train_ml,
        x_validation=x_validation_ml,
        y_validation=y_validation_ml,
        x_test=x_test_ml,
        y_test=y_test_ml,
        epochs=15,
        batch_size=BATCH_SIZE
    )

    vgg16_path = (MODELS_DIR/"vgg16_model.keras")
    vgg16_model.save(vgg16_path)
    logger.info(f"VGG16 saved to {vgg16_path}")

    # ------------------ Comparison ------------------
    comparison_df = compare_models(model_results)
    results_path = (RESULTS_DIR/"model_comparison.csv")
    comparison_df.to_csv(results_path, index=False)
    logger.info(
        f"Model comparison saved to "
        f"{results_path}"
    )

    best_overall = (comparison_df.iloc[0])
    logger.info(
        "Best overall model: "
        f"{best_overall['Model']} "
        f"| F1={best_overall['F1 Score']}"
    )

    logger.info("Image classification training completed.")

    # ------------------ Metadata ------------------
    metadata = {
        "image_size": IMAGE_SIZE,
        "class_names": CLASS_NAMES,
        "best_ml_model": best_ml_name,
        "ml_preprocessing": "HOG + StandardScaler",
        "ann_model": "ann_model.keras",
        "cnn_model": "cnn_model.keras",
        "transfer_learning_model": "vgg16_model.keras"
    }

    metadata_path = (RESULTS_DIR/"metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)
    logger.info(f"Metadata saved to {metadata_path}")
    # Final log
    best_overall = comparison_df.iloc[0]
    logger.info(
        "Best overall model: "
        f"{best_overall['Model']} "
        f"| F1={best_overall['F1 Score']}"
    )
    logger.info("Image classification training completed.")
    
    if __name__ == "__main__":
        main()