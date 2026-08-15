import time
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from src.common.evaluation import calculate_metrics

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

def train_ml_model(model_name: str, model, x_train, y_train, x_test, y_test):
    start_time = time.time()
    model.fit(x_train, y_train)
    training_time = (time.time() - start_time)
    predictions = model.predict(x_test)
    results = calculate_metrics(
        model_name=model_name,
        y_true=y_test,
        y_pred=predictions,
        training_time=training_time
    )
    print(f"\n{model_name}")
    print("-" * 50)
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No Churn",
                "Churn"
            ],
            zero_division=0
        )
    )
    return (model, predictions, results)

def train_deep_learning_model(
    model_name: str,
    model: tf.keras.Model,
    x_train,
    y_train,
    x_validation,
    y_validation,
    x_test,
    y_test,
    epochs: int = 15,
    batch_size: int = 32
):
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    start_time = time.time()
    history = model.fit(x_train, y_train, 
                        validation_data=(x_validation, y_validation), 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        callbacks=[early_stopping])
    training_time = (time.time() - start_time)
    probabilities = model.predict(x_test, verbose=0)
    predictions = (probabilities.flatten() >= 0.5).astype(int)
    results = calculate_metrics(model_name=model_name, 
                                y_true=y_test, 
                                y_pred=predictions, 
                                training_time=training_time)

    print(f"\n{model_name}")
    print("-" * 50)
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No Churn",
                "Churn"
            ],
            zero_division=0
        )
    )

    return (model, history, predictions, results)