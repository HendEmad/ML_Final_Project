import joblib

from src.common.evaluation import (
    model_results,
    compare_models
)

from src.common.logging_config import (
    setup_logger
)

from src.image_classification.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    MODELS_DIR,
    PREPROCESSORS_DIR,
    RESULTS_DIR,
    LOG_DIR
)

from src.image_classification.data import (
    data_loader,
    tensorflow_dataset_to_numpy
)

from src.image_classification.preprocessing import (
    extract_hog_features,
    create_hog_scaler
)

from src.image_classification.models import (
    ml_models
)

from src.image_classification.train import (
    train_ml_model
)


logger = setup_logger(
    logger_name="image_ml_training",
    log_file=str(
        LOG_DIR / "ml_training.log"
    )
)


def main() -> None:
    logger.info(
        "Classical ML training started."
    )

    model_results.clear()

    train_dataset, _, test_dataset = data_loader(
        train_path=TRAIN_DATA_PATH,
        test_path=TEST_DATA_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    x_train, y_train = tensorflow_dataset_to_numpy(
        train_dataset
    )

    x_test, y_test = tensorflow_dataset_to_numpy(
        test_dataset
    )

    logger.info(
        "Extracting HOG features."
    )

    x_train_hog = extract_hog_features(
        x_train
    )

    x_test_hog = extract_hog_features(
        x_test
    )

    hog_scaler = create_hog_scaler()

    x_train_hog = hog_scaler.fit_transform(
        x_train_hog
    )

    x_test_hog = hog_scaler.transform(
        x_test_hog
    )

    scaler_path = (
        PREPROCESSORS_DIR
        / "hog_scaler.pkl"
    )

    joblib.dump(
        hog_scaler,
        scaler_path
    )

    trained_models = {}
    ml_results = {}

    for model_name, model in ml_models.items():

        logger.info(
            f"Training {model_name}"
        )

        (
            trained_model,
            predictions,
            results
        ) = train_ml_model(
            model_name=model_name,
            model=model,
            x_train=x_train_hog,
            y_train=y_train,
            x_test=x_test_hog,
            y_test=y_test
        )

        trained_models[
            model_name
        ] = trained_model

        ml_results[
            model_name
        ] = results

    best_ml_name = max(
        ml_results,
        key=lambda name: (
            ml_results[name]["F1 Score"]
        )
    )

    best_ml_model = (
        trained_models[
            best_ml_name
        ]
    )

    joblib.dump(
        best_ml_model,
        MODELS_DIR
        / "best_ml_model.pkl"
    )

    with open(
        RESULTS_DIR
        / "best_ml_model.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            best_ml_name
        )

    comparison_df = compare_models(
        model_results
    )

    comparison_df.to_csv(
        RESULTS_DIR
        / "ml_model_comparison.csv",
        index=False
    )

    logger.info(
        f"Best ML model: {best_ml_name}"
    )

    logger.info(
        "Classical ML training completed."
    )


if __name__ == "__main__":
    main()