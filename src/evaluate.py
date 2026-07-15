from pathlib import Path

import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src.data_processing import (
    clean_data,
    load_raw_data,
    validate_data,
)

from src.features import (
    FEATURE_COLUMNS,
    build_modeling_dataset,
)

from src.train import (
    RAW_DATA_PATH,
    chronological_train_val_test_split,
)


# ==================================================
# MODEL PATH
# ==================================================

# Path to the best model selected during validation
MODEL_PATH = Path("models/best_model.joblib")


def load_model(model_path: Path = MODEL_PATH):
    """
    Load the best model selected during validation.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Run 'python -m src.train' first."
        )

    model = joblib.load(model_path)

    print(
        f"Model loaded successfully from: "
        f"{model_path}"
    )

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the selected best model on the
    untouched test dataset.
    """

    predictions = model.predict(X_test)

    # ==================================================
    # CALCULATE EVALUATION METRICS
    # ==================================================

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    # ==================================================
    # DISPLAY FINAL TEST RESULTS
    # ==================================================

    print("\n" + "=" * 55)
    print("FINAL TEST SET EVALUATION")
    print("=" * 55)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


if __name__ == "__main__":

    # ==================================================
    # LOAD AND PREPARE DATA
    # ==================================================

    print("=" * 55)
    print("LOADING AND PREPARING DATA")
    print("=" * 55)

    raw_data = load_raw_data(
        RAW_DATA_PATH
    )

    cleaned_data = clean_data(
        raw_data
    )

    validate_data(
        cleaned_data
    )

    modeling_data = build_modeling_dataset(
        cleaned_data
    )

    # ==================================================
    # RECREATE THE SAME CHRONOLOGICAL SPLIT
    # ==================================================

    train_df, val_df, test_df = (
        chronological_train_val_test_split(
            modeling_data
        )
    )

    # ==================================================
    # PREPARE UNTOUCHED TEST DATA
    # ==================================================

    X_test = test_df[FEATURE_COLUMNS]

    y_test = test_df["target"]

    print("\n" + "=" * 55)
    print("UNTOUCHED TEST DATA")
    print("=" * 55)

    print(
        f"Testing observations: "
        f"{len(test_df)}"
    )

    print(
        "Testing period: "
        f"{test_df['date'].min()} to "
        f"{test_df['date'].max()}"
    )

    # ==================================================
    # LOAD SELECTED BEST MODEL
    # ==================================================

    best_model = load_model()

    # ==================================================
    # FINAL TEST EVALUATION
    # ==================================================

    results = evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    # ==================================================
    # FINAL PERFORMANCE SUMMARY
    # ==================================================

    print("\n" + "=" * 55)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("=" * 55)

    print(
        f"Accuracy:  "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{results['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{results['recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{results['f1_score']:.4f}"
    )