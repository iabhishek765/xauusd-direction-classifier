from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from src.data_processing import (
    clean_data,
    load_raw_data,
    validate_data,
)

from src.features import (
    FEATURE_COLUMNS,
    build_modeling_dataset,
)


RAW_DATA_PATH = Path("data/raw/XAU_1h_data.csv")
MODEL_PATH = Path("models/best_model.joblib")

TEST_SIZE = 0.15
N_SPLITS = 5


def chronological_train_val_test_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
):
    """
    Split the modeling dataset chronologically into train,
    validation, and test sets.

    This function is preserved for compatibility with
    evaluate.py.
    """

    train_end = int(len(df) * train_size)
    val_end = int(len(df) * (train_size + val_size))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def create_development_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
):
    """
    Split the dataset chronologically into development
    and untouched test datasets.

    The development dataset is used for time-series
    cross-validation and model selection.

    The final test dataset remains untouched.
    """

    test_start = int(len(df) * (1 - test_size))

    development_df = df.iloc[:test_start].copy()
    test_df = df.iloc[test_start:].copy()

    return development_df, test_df


def create_logistic_model():
    """
    Create a fresh Logistic Regression pipeline.
    """

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def create_baseline_model():
    """
    Create a fresh baseline classifier.
    """

    return DummyClassifier(
        strategy="most_frequent"
    )


def create_gradient_boosting_model(parameters):
    """
    Create a fresh HistGradientBoostingClassifier
    using the supplied hyperparameters.
    """

    return HistGradientBoostingClassifier(
        learning_rate=parameters["learning_rate"],
        max_iter=parameters["max_iter"],
        max_leaf_nodes=parameters["max_leaf_nodes"],
        l2_regularization=parameters["l2_regularization"],
        random_state=42,
    )


def evaluate_model_with_time_series_cv(
    model,
    X,
    y,
    n_splits: int = N_SPLITS,
):
    """
    Evaluate a model using expanding-window
    time-series cross-validation.

    Returns all fold accuracies, mean accuracy,
    and standard deviation.
    """

    time_series_split = TimeSeriesSplit(
        n_splits=n_splits
    )

    fold_accuracies = []

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        time_series_split.split(X),
        start=1,
    ):

        fold_model = clone(model)

        X_train_fold = X.iloc[train_indices]
        y_train_fold = y.iloc[train_indices]

        X_validation_fold = X.iloc[validation_indices]
        y_validation_fold = y.iloc[validation_indices]

        fold_model.fit(
            X_train_fold,
            y_train_fold,
        )

        predictions = fold_model.predict(
            X_validation_fold
        )

        accuracy = accuracy_score(
            y_validation_fold,
            predictions,
        )

        fold_accuracies.append(accuracy)

        print(
            f"Fold {fold_number} accuracy: "
            f"{accuracy:.4f}"
        )

    mean_accuracy = np.mean(fold_accuracies)

    standard_deviation = np.std(fold_accuracies)

    return (
        fold_accuracies,
        mean_accuracy,
        standard_deviation,
    )


def evaluate_baseline_with_time_series_cv(
    X,
    y,
):
    """
    Evaluate the baseline classifier using
    time-series cross-validation.
    """

    print("\n" + "=" * 55)
    print("BASELINE TIME-SERIES CROSS-VALIDATION")
    print("=" * 55)

    baseline_model = create_baseline_model()

    (
        fold_accuracies,
        mean_accuracy,
        standard_deviation,
    ) = evaluate_model_with_time_series_cv(
        baseline_model,
        X,
        y,
    )

    print(
        f"\nBaseline mean CV accuracy: "
        f"{mean_accuracy:.4f}"
    )

    print(
        f"Baseline CV standard deviation: "
        f"{standard_deviation:.4f}"
    )

    return mean_accuracy


def evaluate_logistic_with_time_series_cv(
    X,
    y,
):
    """
    Evaluate Logistic Regression using
    time-series cross-validation.
    """

    print("\n" + "=" * 55)
    print("LOGISTIC REGRESSION TIME-SERIES CROSS-VALIDATION")
    print("=" * 55)

    logistic_model = create_logistic_model()

    (
        fold_accuracies,
        mean_accuracy,
        standard_deviation,
    ) = evaluate_model_with_time_series_cv(
        logistic_model,
        X,
        y,
    )

    print(
        f"\nLogistic Regression mean CV accuracy: "
        f"{mean_accuracy:.4f}"
    )

    print(
        f"Logistic Regression CV standard deviation: "
        f"{standard_deviation:.4f}"
    )

    return (
        logistic_model,
        mean_accuracy,
        standard_deviation,
    )


def tune_gradient_boosting_with_time_series_cv(
    X,
    y,
):
    """
    Tune HistGradientBoostingClassifier using
    time-series cross-validation.

    The untouched final test dataset is never
    used during hyperparameter tuning.
    """

    parameter_combinations = [
        {
            "learning_rate": 0.03,
            "max_iter": 100,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 150,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 15,
            "l2_regularization": 2.0,
        },
        {
            "learning_rate": 0.03,
            "max_iter": 150,
            "max_leaf_nodes": 31,
            "l2_regularization": 2.0,
        },
    ]

    best_parameters = None
    best_mean_accuracy = -np.inf
    best_standard_deviation = None

    print("\n" + "=" * 55)
    print("GRADIENT BOOSTING TIME-SERIES CV TUNING")
    print("=" * 55)

    for experiment_number, parameters in enumerate(
        parameter_combinations,
        start=1,
    ):

        print("\n" + "-" * 55)
        print(f"EXPERIMENT {experiment_number}")
        print("-" * 55)

        print(
            f"Learning rate:       "
            f"{parameters['learning_rate']}"
        )

        print(
            f"Maximum iterations:  "
            f"{parameters['max_iter']}"
        )

        print(
            f"Maximum leaf nodes:  "
            f"{parameters['max_leaf_nodes']}"
        )

        print(
            f"L2 regularization:   "
            f"{parameters['l2_regularization']}"
        )

        model = create_gradient_boosting_model(
            parameters
        )

        (
            fold_accuracies,
            mean_accuracy,
            standard_deviation,
        ) = evaluate_model_with_time_series_cv(
            model,
            X,
            y,
        )

        print(
            f"\nMean CV accuracy: "
            f"{mean_accuracy:.4f}"
        )

        print(
            f"CV standard deviation: "
            f"{standard_deviation:.4f}"
        )

        if mean_accuracy > best_mean_accuracy:

            best_mean_accuracy = mean_accuracy

            best_standard_deviation = (
                standard_deviation
            )

            best_parameters = parameters.copy()

    print("\n" + "=" * 55)
    print("BEST GRADIENT BOOSTING CV CONFIGURATION")
    print("=" * 55)

    print(
        f"Learning rate:       "
        f"{best_parameters['learning_rate']}"
    )

    print(
        f"Maximum iterations:  "
        f"{best_parameters['max_iter']}"
    )

    print(
        f"Maximum leaf nodes:  "
        f"{best_parameters['max_leaf_nodes']}"
    )

    print(
        f"L2 regularization:   "
        f"{best_parameters['l2_regularization']}"
    )

    print(
        f"\nBest mean CV accuracy: "
        f"{best_mean_accuracy:.4f}"
    )

    print(
        f"Best CV standard deviation: "
        f"{best_standard_deviation:.4f}"
    )

    best_model = create_gradient_boosting_model(
        best_parameters
    )

    return (
        best_model,
        best_mean_accuracy,
        best_standard_deviation,
        best_parameters,
    )


def select_best_model(
    logistic_model,
    logistic_mean_accuracy,
    logistic_standard_deviation,
    gradient_boosting_model,
    gradient_boosting_mean_accuracy,
    gradient_boosting_standard_deviation,
):
    """
    Select the best model using mean time-series
    cross-validation accuracy.

    Standard deviation is displayed as an indicator
    of model stability across time periods.
    """

    print("\n" + "=" * 55)
    print("FINAL TIME-SERIES CV MODEL COMPARISON")
    print("=" * 55)

    print(
        "Logistic Regression:"
    )

    print(
        f"Mean CV accuracy:       "
        f"{logistic_mean_accuracy:.4f}"
    )

    print(
        f"CV standard deviation:  "
        f"{logistic_standard_deviation:.4f}"
    )

    print(
        "\nGradient Boosting:"
    )

    print(
        f"Mean CV accuracy:       "
        f"{gradient_boosting_mean_accuracy:.4f}"
    )

    print(
        f"CV standard deviation:  "
        f"{gradient_boosting_standard_deviation:.4f}"
    )

    if (
        gradient_boosting_mean_accuracy
        > logistic_mean_accuracy
    ):
        best_model = gradient_boosting_model
        best_model_name = "Gradient Boosting"
        best_mean_accuracy = (
            gradient_boosting_mean_accuracy
        )

    else:
        best_model = logistic_model
        best_model_name = "Logistic Regression"
        best_mean_accuracy = (
            logistic_mean_accuracy
        )

    print(
        f"\nSelected model: "
        f"{best_model_name}"
    )

    print(
        f"Selected mean CV accuracy: "
        f"{best_mean_accuracy:.4f}"
    )

    return (
        best_model,
        best_model_name,
    )


def save_model(
    model,
    model_path: Path = MODEL_PATH,
):
    """
    Save the selected and fully trained model.
    """

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )


if __name__ == "__main__":

    # ==================================================
    # LOAD AND PREPARE DATA
    # ==================================================

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
    # CREATE DEVELOPMENT AND UNTOUCHED TEST SETS
    # ==================================================

    development_df, test_df = (
        create_development_test_split(
            modeling_data
        )
    )

    X_development = development_df[
        FEATURE_COLUMNS
    ]

    y_development = development_df[
        "target"
    ]

    print("=" * 55)
    print("DEVELOPMENT AND TEST DATA SPLIT")
    print("=" * 55)

    print(
        f"Development observations: "
        f"{len(development_df)}"
    )

    print(
        f"Testing observations:     "
        f"{len(test_df)}"
    )

    print(
        "\nDevelopment period: "
        f"{development_df['date'].min()} "
        f"to {development_df['date'].max()}"
    )

    print(
        "Testing period: "
        f"{test_df['date'].min()} "
        f"to {test_df['date'].max()}"
    )

    # ==================================================
    # BASELINE CROSS-VALIDATION
    # ==================================================

    baseline_mean_accuracy = (
        evaluate_baseline_with_time_series_cv(
            X_development,
            y_development,
        )
    )

    # ==================================================
    # LOGISTIC REGRESSION CROSS-VALIDATION
    # ==================================================

    (
        logistic_model,
        logistic_mean_accuracy,
        logistic_standard_deviation,
    ) = evaluate_logistic_with_time_series_cv(
        X_development,
        y_development,
    )

    # ==================================================
    # GRADIENT BOOSTING CROSS-VALIDATION AND TUNING
    # ==================================================

    (
        gradient_boosting_model,
        gradient_boosting_mean_accuracy,
        gradient_boosting_standard_deviation,
        best_gradient_boosting_parameters,
    ) = tune_gradient_boosting_with_time_series_cv(
        X_development,
        y_development,
    )

    # ==================================================
    # SELECT BEST MODEL
    # ==================================================

    best_model, best_model_name = select_best_model(
        logistic_model,
        logistic_mean_accuracy,
        logistic_standard_deviation,
        gradient_boosting_model,
        gradient_boosting_mean_accuracy,
        gradient_boosting_standard_deviation,
    )

    # ==================================================
    # RETRAIN SELECTED MODEL ON ALL DEVELOPMENT DATA
    # ==================================================

    print("\n" + "=" * 55)
    print("RETRAINING SELECTED MODEL")
    print("=" * 55)

    best_model.fit(
        X_development,
        y_development,
    )

    print(
        f"{best_model_name} retrained successfully "
        "on the complete development dataset."
    )

    # ==================================================
    # SAVE FINAL MODEL
    # ==================================================

    save_model(
        best_model
    )

    # ==================================================
    # TRAINING SUMMARY
    # ==================================================

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED")
    print("=" * 55)

    print(
        f"Baseline mean CV accuracy: "
        f"{baseline_mean_accuracy:.4f}"
    )

    print(
        f"Logistic Regression mean CV accuracy: "
        f"{logistic_mean_accuracy:.4f}"
    )

    print(
        f"Gradient Boosting mean CV accuracy: "
        f"{gradient_boosting_mean_accuracy:.4f}"
    )

    print(
        f"\nSelected model: "
        f"{best_model_name}"
    )

    print(
        f"Model saved to: "
        f"{MODEL_PATH}"
    )

    print(
        "\nThe untouched test dataset was not used "
        "during cross-validation or model selection."
    )