from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_endpoint():
    payload = {
        "ride_distance_km": 8.5,
        "estimated_duration_min": 28.0,
        "estimated_price": 1850.0,
        "hour_of_day": 18,
        "day_of_week": 4,
        "is_weekend": 0,
        "is_rush_hour": 1,
        "surge_multiplier": 1.5,
        "waiting_time_min": 7.0,
        "driver_rating": 4.6,
        "user_rating": 4.8,
        "user_past_cancellations": 2,
        "driver_past_cancellations": 1,
        "pickup_area": "city_center",
        "dropoff_area": "residential",
        "payment_method": "cash",
        "weather_condition": "rain",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "cancellation_probability" in data
    assert "risk_label" in data

    assert data["prediction"] in [0, 1]
    assert 0 <= data["cancellation_probability"] <= 1
    assert data["risk_label"] in ["low", "high"]