from src.common.evaluation import (
    model_results
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
    NUMBER_OF_CLASSES,
    MODELS_DIR,
    LOG_DIR
)

from src.image_classification.data import (
    data_loader,
    tensorflow_dataset_to_numpy
)

from src.image_classification.models import (
    create_cnn_model
)

from src.image_classification.train import (
    train_deep_learning_model
)


logger = setup_logger(
    logger_name="image_cnn_training",
    log_file=str(
        LOG_DIR / "cnn_training.log"
    )
)


def main() -> None:
    logger.info(
        "CNN training started."
    )

    model_results.clear()

    (
        train_dataset,
        validation_dataset,
        test_dataset
    ) = data_loader(
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

    (
        x_validation,
        y_validation
    ) = tensorflow_dataset_to_numpy(
        validation_dataset
    )

    x_test, y_test = tensorflow_dataset_to_numpy(
        test_dataset
    )

    model = create_cnn_model(
        IMAGE_SIZE,
        NUMBER_OF_CLASSES
    )

    (
        model,
        history,
        predictions,
        results
    ) = train_deep_learning_model(
        model_name="CNN",
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        epochs=15,
        batch_size=BATCH_SIZE
    )

    model.save(
        MODELS_DIR
        / "cnn_model.keras"
    )

    logger.info(
        f"CNN F1: {results['F1 Score']}"
    )

    logger.info(
        "CNN training completed."
    )


if __name__ == "__main__":
    main()