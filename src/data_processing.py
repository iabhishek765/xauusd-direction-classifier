from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/XAU_1h_data.csv")
PROCESSED_DATA_PATH = Path("data/processed/xauusd_h1_clean.csv")


def load_raw_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw hourly XAUUSD OHLC data."""
    return pd.read_csv(file_path, sep=";")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the raw XAUUSD dataset."""

    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.lower()

    # Convert date column to datetime
    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y.%m.%d %H:%M"
    )

    # Sort observations chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Remove duplicate timestamps if present
    df = df.drop_duplicates(subset="date", keep="first")

    # Remove rows containing missing OHLC values
    df = df.dropna(subset=["open", "high", "low", "close"])

    return df


def validate_data(df: pd.DataFrame) -> None:
    """Run basic integrity checks on cleaned OHLC data."""

    required_columns = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if not df["date"].is_monotonic_increasing:
        raise ValueError("Dataset is not sorted chronologically.")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    if df[["open", "high", "low", "close"]].isnull().any().any():
        raise ValueError("Missing OHLC values detected.")

    # Validate basic OHLC relationships
    invalid_high = (
        df["high"]
        < df[["open", "close", "low"]].max(axis=1)
    )

    invalid_low = (
        df["low"]
        > df[["open", "close", "high"]].min(axis=1)
    )

    if invalid_high.any():
        raise ValueError(
            f"{invalid_high.sum()} rows contain invalid high prices."
        )

    if invalid_low.any():
        raise ValueError(
            f"{invalid_low.sum()} rows contain invalid low prices."
        )


def save_processed_data(
    df: pd.DataFrame,
    file_path: Path = PROCESSED_DATA_PATH,
) -> None:
    """Save cleaned data for downstream analysis."""

    file_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(file_path, index=False)


if __name__ == "__main__":

    data = load_raw_data()

    print(f"Raw dataset shape: {data.shape}")

    cleaned_data = clean_data(data)

    validate_data(cleaned_data)

    save_processed_data(cleaned_data)

    print(f"Cleaned dataset shape: {cleaned_data.shape}")
    print(
        f"Date range: {cleaned_data['date'].min()} "
        f"to {cleaned_data['date'].max()}"
    )

    print("Data validation completed successfully.")
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")