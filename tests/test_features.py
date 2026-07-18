import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    create_features,
)


def test_create_features_generates_all_columns():

    df = pd.read_csv(
        "data/processed/xauusd_h1_clean.csv"
    )

    df = df.iloc[:50].copy()

    df["date"] = pd.to_datetime(df["date"])

    feature_df = create_features(df)

    for column in FEATURE_COLUMNS:
        assert column in feature_df.columns


def test_latest_feature_vector_has_no_missing_values():

    df = pd.read_csv(
        "data/processed/xauusd_h1_clean.csv"
    )

    df = df.iloc[:50].copy()

    df["date"] = pd.to_datetime(df["date"])

    feature_df = create_features(df)

    latest = feature_df.iloc[-1]

    assert latest[FEATURE_COLUMNS].isnull().sum() == 0