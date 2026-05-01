import os
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = "models/model.joblib"


class RideRequest(BaseModel):
    ride_distance_km: float = Field(..., ge=0)
    estimated_duration_min: float = Field(..., ge=0)
    estimated_price: float = Field(..., ge=0)

    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)

    is_weekend: int = Field(..., ge=0, le=1)
    is_rush_hour: int = Field(..., ge=0, le=1)

    surge_multiplier: float = Field(..., ge=1)
    waiting_time_min: float = Field(..., ge=0)

    driver_rating: float = Field(..., ge=0, le=5)
    user_rating: float = Field(..., ge=0, le=5)

    user_past_cancellations: int = Field(..., ge=0)
    driver_past_cancellations: int = Field(..., ge=0)

    pickup_area: str
    dropoff_area: str
    payment_method: str
    weather_condition: str


class PredictionResponse(BaseModel):
    prediction: int
    cancellation_probability: float
    risk_label: str


app = FastAPI(
    title="Ride Cancellation Prediction API",
    description="A production-style ML API for predicting ride cancellation risk.",
    version="1.0.0",
)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run python src/train.py first."
        )

    return joblib.load(MODEL_PATH)


model = load_model()


@app.get("/")
def root():
    return {
        "message": "Ride Cancellation Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: RideRequest):
    input_df = pd.DataFrame([request.model_dump()])

    cancellation_probability = model.predict_proba(input_df)[0][1]
    prediction = int(cancellation_probability >= 0.5)

    risk_label = "high" if prediction == 1 else "low"

    return PredictionResponse(
        prediction=prediction,
        cancellation_probability=round(float(cancellation_probability), 4),
        risk_label=risk_label,
    )