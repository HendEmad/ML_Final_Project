import time
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from src.common.evaluation import calculate_metrics
from src.image_classification.config import CLASS_NAMES

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

def train_ml_model(model_name, model, x_train, y_train, x_test, y_test):
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
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    return (model, predictions, results)

def train_deep_learning_model(
        model_name, model, x_train, y_train, x_validation, 
        y_validation, x_test, y_test, epochs=15, batch_size=32
):
    model.compile(
        optimizer="adam",
        loss=(
            "sparse_categorical_"
            "crossentropy"
        ),
        metrics=["accuracy"]
    )
    start_time = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )
    training_time = (time.time() - start_time)
    probs = model.predict(x_test, verbose=0)
    predictions = np.argmax(probs, axis=1)
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
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )
    return (model, history, predictions, results)