from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features import FEATURE_COLUMNS


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


# --------------------------------------------------
# INPUT SCHEMA
# --------------------------------------------------

class PredictionInput(BaseModel):
    features: dict[str, float]


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

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in input_data.features
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Required model features are missing.",
                "missing_features": missing_features,
            },
        )

    try:
        input_df = pd.DataFrame(
            [
                {
                    feature: input_data.features[feature]
                    for feature in FEATURE_COLUMNS
                }
            ]
        )

        prediction = int(model.predict(input_df)[0])

        prediction_probability = float(
            model.predict_proba(input_df)[0][prediction]
        )

        direction = "UP" if prediction == 1 else "DOWN"

        return {
            "predicted_class": prediction,
            "predicted_direction": direction,
            "confidence": round(prediction_probability, 4),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )