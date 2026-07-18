from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features import (
    FEATURE_COLUMNS,
    create_features,
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="XAU/USD Direction Classifier API",
    description=(
        "Machine learning API for predicting the hourly "
        "direction of XAU/USD."
    ),
    version="1.0.0",
)


class Candle(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictionInput(BaseModel):
    candles: list[Candle]


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "XAU/USD Direction Classifier API",
        "status": "running",
    }


# --------------------------------------------------
# HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "number_of_features": len(FEATURE_COLUMNS),
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict(input_data: PredictionInput):

    try:

        # Require enough historical candles to compute
        # rolling features safely.
        if len(input_data.candles) < 30:
            raise HTTPException(
                status_code=400,
                detail="At least 30 historical candles are required.",
            )

        # Convert request into a DataFrame
        input_df = pd.DataFrame(
            [candle.model_dump() for candle in input_data.candles]
        )

        # Convert date column to datetime
        input_df["date"] = pd.to_datetime(input_df["date"])

        # Ensure chronological order
        input_df = (
            input_df
            .sort_values("date")
            .reset_index(drop=True)
        )

        # Compute engineered features
        feature_df = create_features(input_df)

        # Remove rows where rolling features are unavailable
        feature_df = feature_df.dropna(
    subset=FEATURE_COLUMNS
)
        
        if feature_df.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Not enough valid historical data to compute "
                    "model features."
                ),
            )

        # Select the latest feature vector
        latest_features = (
            feature_df.iloc[[-1]][FEATURE_COLUMNS]
        )

        prediction = int(
            model.predict(latest_features)[0]
        )

        prediction_probability = float(
            model.predict_proba(latest_features)[0][prediction]
        )

        direction = "UP" if prediction == 1 else "DOWN"

        return {
            "predicted_class": prediction,
            "predicted_direction": direction,
            "confidence": round(prediction_probability, 4),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )