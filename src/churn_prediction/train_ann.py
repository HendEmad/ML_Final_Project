# python -m src.churn_prediction.train_ann
import joblib

from src.common.evaluation import (
    model_results
)

from src.common.logging_config import (
    setup_logger
)

from src.churn_prediction.config import (
    DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    BATCH_SIZE,
    EPOCHS,
    MODELS_DIR,
    PREPROCESSORS_DIR,
    LOG_DIR
)

from src.churn_prediction.data import (
    load_data,
    clean_data,
    separate_features_target,
    split_data
)

from src.churn_prediction.preprocessing import (
    get_feature_types,
    create_preprocessor
)

from src.churn_prediction.models import (
    create_ann_model
)

from src.churn_prediction.train import (
    train_deep_learning_model
)


logger = setup_logger(
    logger_name="churn_ann_training",
    log_file=str(
        LOG_DIR / "ann_training.log"
    )
)


def main() -> None:
    logger.info(
        "ANN training started."
    )

    model_results.clear()

    data = load_data(
        str(DATA_PATH)
    )

    data = clean_data(
        data
    )

    x, y = separate_features_target(
        data,
        TARGET_COLUMN
    )

    (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test
    ) = split_data(
        x,
        y,
        test_size=TEST_SIZE,
        validation_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    (
        numerical_columns,
        categorical_columns
    ) = get_feature_types(
        x_train
    )

    preprocessor = create_preprocessor(
        numerical_columns,
        categorical_columns
    )

    x_train_processed = (
        preprocessor.fit_transform(
            x_train
        )
    )

    x_validation_processed = (
        preprocessor.transform(
            x_validation
        )
    )

    x_test_processed = (
        preprocessor.transform(
            x_test
        )
    )

    preprocessor_path = (
        PREPROCESSORS_DIR
        / "preprocessor.pkl"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    number_of_features = (
        x_train_processed.shape[1]
    )

    logger.info(
        f"Training ANN with "
        f"{number_of_features} "
        f"input features."
    )

    model = create_ann_model(
        number_of_features
    )

    (
        model,
        history,
        predictions,
        results
    ) = train_deep_learning_model(
        model_name="ANN",
        model=model,
        x_train=x_train_processed,
        y_train=y_train,
        x_validation=x_validation_processed,
        y_validation=y_validation,
        x_test=x_test_processed,
        y_test=y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    model.save(
        MODELS_DIR
        / "ann_model.keras"
    )

    logger.info(
        f"ANN F1: "
        f"{results['F1 Score']}"
    )

    logger.info(
        "ANN training completed."
    )


if __name__ == "__main__":
    main()