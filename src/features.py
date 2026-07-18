import pandas as pd


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the binary target for next-candle direction.

    target = 1 if the next candle closes above
    the current candle close, otherwise 0.
    """

    df = df.copy()

    df["next_close"] = df["close"].shift(-1)

    df["target"] = (
        df["next_close"] > df["close"]
    ).astype(int)

    # The final row has no future candle available
    df = df.iloc[:-1].copy()

    return df


def calculate_rsi(
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    using current and historical prices only.
    """

    price_change = close.diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.rolling(window=window).mean()
    average_loss = losses.rolling(window=window).mean()

    relative_strength = (
        average_gain / average_loss.replace(0, 1e-10)
    )

    rsi = 100 - (
        100 / (1 + relative_strength)
    )

    return rsi


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features using current and historical
    information only.
    """

    df = df.copy()

    # ==================================================
    # RETURN FEATURES
    # ==================================================

    df["return_1h"] = df["close"].pct_change(periods=1)
    df["return_3h"] = df["close"].pct_change(periods=3)
    df["return_6h"] = df["close"].pct_change(periods=6)
    df["return_12h"] = df["close"].pct_change(periods=12)
    df["return_24h"] = df["close"].pct_change(periods=24)

    # ==================================================
    # MOMENTUM FEATURES
    # ==================================================
    # Momentum measures how far the current price is
    # trading above or below its recent moving average.
    # Unlike percentage returns, these features capture
    # the current strength of price relative to its
    # recent trend.

    momentum_ma_3 = df["close"].rolling(window=3).mean()
    momentum_ma_6 = df["close"].rolling(window=6).mean()
    momentum_ma_12 = df["close"].rolling(window=12).mean()
    momentum_ma_24 = df["close"].rolling(window=24).mean()

    df["momentum_3h"] = (
        df["close"] / momentum_ma_3
    ) - 1

    df["momentum_6h"] = (
        df["close"] / momentum_ma_6
    ) - 1

    df["momentum_12h"] = (
        df["close"] / momentum_ma_12
    ) - 1

    df["momentum_24h"] = (
        df["close"] / momentum_ma_24
    ) - 1

    # ==================================================
    # CANDLE FEATURES
    # ==================================================

    df["body_size"] = (
        (df["close"] - df["open"]) / df["open"]
    )

    df["range_size"] = (
        (df["high"] - df["low"]) / df["open"]
    )

    # Prevent division by zero for zero-range candles
    candle_range = (
        df["high"] - df["low"]
    ).replace(0, 1e-10)

    # Upper wick relative to the complete candle range
    df["upper_wick_ratio"] = (
        df["high"] - df[["open", "close"]].max(axis=1)
    ) / candle_range

    # Lower wick relative to the complete candle range
    df["lower_wick_ratio"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    ) / candle_range

    # Closing price position within the candle range
    df["close_position"] = (
        df["close"] - df["low"]
    ) / candle_range

    # ==================================================
    # VOLATILITY FEATURES
    # ==================================================

    df["volatility_6h"] = (
        df["return_1h"]
        .rolling(window=6)
        .std()
    )

    df["volatility_12h"] = (
        df["return_1h"]
        .rolling(window=12)
        .std()
    )

    df["volatility_24h"] = (
        df["return_1h"]
        .rolling(window=24)
        .std()
    )

    # ==================================================
    # ATR-STYLE VOLATILITY FEATURE
    # ==================================================

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_14 = true_range.rolling(window=14).mean()

    # Normalize ATR relative to the current closing price
    df["atr_14_ratio"] = atr_14 / df["close"]

    # ==================================================
    # MOVING AVERAGE FEATURES
    # ==================================================

    sma_6h = (
        df["close"]
        .rolling(window=6)
        .mean()
    )

    sma_12h = (
        df["close"]
        .rolling(window=12)
        .mean()
    )

    sma_24h = (
        df["close"]
        .rolling(window=24)
        .mean()
    )

    df["sma_6h_ratio"] = (
        df["close"] / sma_6h
    ) - 1

    df["sma_12h_ratio"] = (
        df["close"] / sma_12h
    ) - 1

    df["sma_24h_ratio"] = (
        df["close"] / sma_24h
    ) - 1

    # ==================================================
    # TREND FEATURE
    # ==================================================

    # Difference between short-term and longer-term
    # moving averages, normalized by the longer SMA
    df["trend_strength_24h"] = (
        sma_6h - sma_24h
    ) / sma_24h

    # ==================================================
    # RSI FEATURE
    # ==================================================

    df["rsi_14"] = calculate_rsi(
        df["close"],
        window=14,
    )

    df["rsi_14"] = df["rsi_14"] / 100

    # ==================================================
    # ROLLING MARKET DIRECTION FEATURE
    # ==================================================

    up_candle = (
        df["close"] > df["open"]
    ).astype(int)

    df["up_candle_ratio_12h"] = (
        up_candle
        .rolling(window=12)
        .mean()
    )

    # ==================================================
    # TIME FEATURES
    # ==================================================

    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek

    return df


FEATURE_COLUMNS = [
    "return_1h",
    "return_3h",
    "return_6h",
    "return_12h",
    "return_24h",
    "momentum_3h",
    "momentum_6h",
    "momentum_12h",
    "momentum_24h",
    "body_size",
    "range_size",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position",
    "volatility_6h",
    "volatility_12h",
    "volatility_24h",
    "atr_14_ratio",
    "sma_6h_ratio",
    "sma_12h_ratio",
    "sma_24h_ratio",
    "trend_strength_24h",
    "rsi_14",
    "up_candle_ratio_12h",
    "hour",
    "day_of_week",
]


def build_modeling_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create features and target, then remove rows
    that cannot be used for model training.
    """

    df = create_features(df)

    df = create_target(df)

    # Remove rows with NaN values created by
    # percentage changes and rolling calculations
    df = df.dropna(
        subset=FEATURE_COLUMNS + ["target"]
    )

    return df.reset_index(drop=True)