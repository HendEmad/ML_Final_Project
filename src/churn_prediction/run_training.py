import json
import joblib
from src.common.evaluation import (model_results, compare_models)
from src.common.logging_config import setup_logger
from src.churn_prediction.config import (
    DATA_PATH, TARGET_COLUMN,
    TEST_SIZE, VALIDATION_SIZE,
    RANDOM_STATE, BATCH_SIZE,
    EPOCHS, MODELS_DIR,
    PREPROCESSORS_DIR, RESULTS_DIR,
    LOG_DIR
)
from src.churn_prediction.data import (load_data, clean_data, separate_features_target, split_data)
from src.churn_prediction.preprocessing import (get_feature_types, create_preprocessor)
from src.churn_prediction.models import (ml_models, create_ann_model)
from src.churn_prediction.train import (train_ml_model, train_deep_learning_model)

logger = setup_logger(
    logger_name="churn_training",
    log_file=str(LOG_DIR / "training.log")
)

def main() -> None:
    logger.info("Churn prediction training started.")
    model_results.clear()

    logger.info("Loading dataset.")
    data = load_data(str(DATA_PATH))

    logger.info("Cleaning dataset.")
    data = clean_data(data)

    x, y = separate_features_target(data, TARGET_COLUMN)
    logger.info("Splitting dataset.")

    x_train, x_validation, x_test, y_train, y_validation, y_test = split_data(x, y,
                                                                              test_size=TEST_SIZE,
                                                                              validation_size=VALIDATION_SIZE,
                                                                              random_state=RANDOM_STATE)
    numerical_columns, categorical_columns = get_feature_types(x_train)
    logger.info(f"Numerical features: "
                f"{len(numerical_columns)}")
    logger.info(f"Categorical features: "
                f"{len(categorical_columns)}")

    preprocessor = create_preprocessor(numerical_columns, categorical_columns)
    logger.info("Fitting preprocessing pipeline.")
    x_train_processed = (preprocessor.fit_transform(x_train))
    x_validation_processed = (preprocessor.transform(x_validation))
    x_test_processed = (preprocessor.transform(x_test))
    # save processor results
    preprocessor_path = (PREPROCESSORS_DIR/"preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to "
                f"{preprocessor_path}")

    # ------------------ ML Models training ------------------
    logger.info("Training classical ML models.")
    trained_models = {}
    ml_results = {}

    for model_name, model in (ml_models.items()):
        logger.info(f"Training {model_name}")
        trained_model, predictions, results = train_ml_model( 
            model_name=model_name,
            model=model,
            x_train=x_train_processed,
            y_train=y_train,
            x_test=x_test_processed,
            y_test=y_test
        )

        trained_models[model_name] = trained_model
        ml_results[model_name] = results

    # ------------------ Best ML model ------------------
    best_ml_name = max(ml_results, key=lambda name: (
            ml_results[name]["F1 Score"])
    )
    best_ml_model = (trained_models[best_ml_name])
    best_ml_path = (MODELS_DIR/"best_ml_model.pkl")
    joblib.dump(best_ml_model, best_ml_path)
    logger.info(f"Best ML model: "
                f"{best_ml_name}")
    logger.info(f"Best ML model saved to "
                f"{best_ml_path}")

    with open(RESULTS_DIR/"best_ml_model.txt", "w", encoding="utf-8") as file:
        file.write(best_ml_name)

    # ------------------ ANN Model Training ------------------
    number_of_features = (x_train_processed.shape[1])
    logger.info(
        f"Training ANN with "
        f"{number_of_features} "
        f"input features."
    )
    ann_model = create_ann_model(number_of_features)
    ann_model, ann_history, ann_predictions, ann_results = train_deep_learning_model(
        model_name="ANN",
        model=ann_model,
        x_train=x_train_processed,
        y_train=y_train,
        x_validation=(x_validation_processed),
        y_validation=y_validation,
        x_test=x_test_processed,
        y_test=y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    ann_path = (MODELS_DIR/"ann_model.keras")
    ann_model.save(ann_path)
    logger.info(f"ANN saved to {ann_path}")

    comparison_df = compare_models(model_results)
    results_path = (RESULTS_DIR/"model_comparison.csv")
    comparison_df.to_csv(results_path, index=False)
    logger.info(
        f"Model comparison saved to "
        f"{results_path}"
    )

    metadata = {
        "target_column": TARGET_COLUMN,
        "target_classes": {"0": "No Churn", "1": "Churn"},
        "best_ml_model": (best_ml_name),
        "preprocessor": ("preprocessor.pkl"),
        "ann_model": ("ann_model.keras"),
        "number_of_features_after_preprocessing": (number_of_features),
        "numerical_columns": (numerical_columns),
        "categorical_columns": (categorical_columns)
    }

    metadata_path = (RESULTS_DIR/"metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)
    logger.info(
        f"Metadata saved to "
        f"{metadata_path}"
    )

    best_overall = (comparison_df.iloc[0])
    logger.info(
        "Best overall model: "
        f"{best_overall['Model']} "
        f"| F1="
        f"{best_overall['F1 Score']}"
    )
    logger.info("Churn prediction training completed.")

if __name__ == "__main__":
    main()