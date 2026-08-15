# python -m src.churn_prediction.train_ml
import joblib

from src.common.evaluation import (
    model_results,
    compare_models
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
    MODELS_DIR,
    PREPROCESSORS_DIR,
    RESULTS_DIR,
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
    ml_models
)

from src.churn_prediction.train import (
    train_ml_model
)


logger = setup_logger(
    logger_name="churn_ml_training",
    log_file=str(
        LOG_DIR / "ml_training.log"
    )
)


def main() -> None:
    logger.info(
        "Classical ML training started."
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
        _,
        x_test,
        y_train,
        _,
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

    logger.info(
        "Fitting preprocessing pipeline."
    )

    x_train_processed = (
        preprocessor.fit_transform(
            x_train
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

    logger.info(
        f"Preprocessor saved to "
        f"{preprocessor_path}"
    )

    trained_models = {}
    ml_results = {}

    for model_name, model in (
        ml_models.items()
    ):

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
            x_train=x_train_processed,
            y_train=y_train,
            x_test=x_test_processed,
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
            ml_results[name][
                "F1 Score"
            ]
        )
    )

    best_ml_model = (
        trained_models[
            best_ml_name
        ]
    )

    best_ml_path = (
        MODELS_DIR
        / "best_ml_model.pkl"
    )

    joblib.dump(
        best_ml_model,
        best_ml_path
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