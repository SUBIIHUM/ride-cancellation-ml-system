import os
import joblib
import pandas as pd


MODEL_PATH = "models/model.joblib"


SAMPLE_RIDE = {
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


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run python src/train.py first."
        )

    return joblib.load(model_path)


def predict_cancellation(ride_data: dict) -> dict:
    model = load_model()

    input_df = pd.DataFrame([ride_data])

    cancellation_probability = model.predict_proba(input_df)[0][1]
    prediction = int(cancellation_probability >= 0.5)

    return {
        "prediction": prediction,
        "cancellation_probability": round(float(cancellation_probability), 4),
        "risk_label": "high" if prediction == 1 else "low",
    }


def main() -> None:
    result = predict_cancellation(SAMPLE_RIDE)

    print("Input ride:")
    print(SAMPLE_RIDE)

    print("\nPrediction result:")
    print(result)


if __name__ == "__main__":
    main()