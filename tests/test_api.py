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
    sample_features = {
        "return_1h": -0.01710613018663254,
        "return_3h": -0.0003464303242011191,
        "return_6h": -0.03672455589556112,
        "return_12h": -0.0526073047135317,
        "return_24h": -0.08630029063302214,
        "momentum_3h": -0.0003464303242011191,
        "momentum_6h": -0.03672455589556112,
        "momentum_12h": -0.0526073047135317,
        "momentum_24h": -0.08630029063302214,
        "body_size": -0.01715594842267454,
        "range_size": 0.02503852080123267,
        "upper_wick_ratio": 0.0,
        "lower_wick_ratio": 0.3148178137651831,
        "close_position": 0.3148178137651831,
        "volatility_6h": 0.015080124142968266,
        "volatility_12h": 0.014401339201012517,
        "volatility_24h": 0.012358700806787831,
        "atr_14_ratio": 0.025095728648695168,
        "sma_6h_ratio": -0.015561588237105961,
        "sma_12h_ratio": -0.029323515778380216,
        "sma_24h_ratio": -0.05500596983406303,
        "trend_strength_24h": -0.04006790178607679,
        "rsi_14": 0.3152245766276849,
        "up_candle_ratio_12h": 0.5,
        "hour": 22.0,
        "day_of_week": 4.0,
    }

    response = client.post(
        "/predict",
        json={"features": sample_features},
    )

    data = response.json()

    assert response.status_code == 200
    assert "predicted_class" in data
    assert "predicted_direction" in data
    assert "confidence" in data
    assert data["predicted_direction"] in ["UP", "DOWN"]
    assert 0 <= data["confidence"] <= 1