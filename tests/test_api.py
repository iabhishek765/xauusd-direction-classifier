from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["number_of_features"] == 26

    

def test_predict_endpoint():
    csv_path = Path("data/processed/xauusd_h1_clean.csv")

    df = pd.read_csv(csv_path)

    # Use 30 consecutive candles after enough history exists
    sample = df.tail(30)

    candles = []

    for _, row in sample.iterrows():
        candles.append(
            {
                "date": row["date"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )

    response = client.post(
        "/predict",
        json={"candles": candles},
    )

    data = response.json()

    assert response.status_code == 200
    assert "predicted_class" in data
    assert "predicted_direction" in data
    assert "confidence" in data

    assert data["predicted_direction"] in ["UP", "DOWN"]
    assert 0 <= data["confidence"] <= 1