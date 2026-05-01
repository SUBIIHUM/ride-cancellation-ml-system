import os
import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_SAMPLES = 20000


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_ride_data(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    np.random.seed(seed)

    ride_distance_km = np.random.gamma(shape=2.2, scale=3.0, size=n_samples)
    ride_distance_km = np.clip(ride_distance_km, 0.5, 35)

    estimated_duration_min = ride_distance_km * np.random.uniform(2.0, 4.5, size=n_samples)
    estimated_duration_min += np.random.normal(0, 4, size=n_samples)
    estimated_duration_min = np.clip(estimated_duration_min, 3, 120)

    hour_of_day = np.random.randint(0, 24, size=n_samples)
    day_of_week = np.random.randint(0, 7, size=n_samples)

    is_weekend = np.isin(day_of_week, [5, 6]).astype(int)
    is_rush_hour = np.isin(hour_of_day, [7, 8, 9, 17, 18, 19]).astype(int)

    surge_multiplier = np.random.choice(
        [1.0, 1.2, 1.5, 2.0],
        size=n_samples,
        p=[0.68, 0.18, 0.10, 0.04]
    )

    base_price = 500
    price_per_km = 120
    estimated_price = (base_price + ride_distance_km * price_per_km) * surge_multiplier
    estimated_price += np.random.normal(0, 150, size=n_samples)
    estimated_price = np.clip(estimated_price, 300, 12000)

    waiting_time_min = np.random.exponential(scale=3.0, size=n_samples)
    waiting_time_min += is_rush_hour * np.random.uniform(1, 5, size=n_samples)
    waiting_time_min = np.clip(waiting_time_min, 0, 30)

    driver_rating = np.random.normal(4.75, 0.25, size=n_samples)
    driver_rating = np.clip(driver_rating, 3.0, 5.0)

    user_rating = np.random.normal(4.7, 0.3, size=n_samples)
    user_rating = np.clip(user_rating, 3.0, 5.0)

    user_past_cancellations = np.random.poisson(lam=1.2, size=n_samples)
    driver_past_cancellations = np.random.poisson(lam=0.7, size=n_samples)

    pickup_area = np.random.choice(
        ["city_center", "residential", "airport", "suburb", "business_district"],
        size=n_samples,
        p=[0.30, 0.30, 0.10, 0.15, 0.15]
    )

    dropoff_area = np.random.choice(
        ["city_center", "residential", "airport", "suburb", "business_district"],
        size=n_samples,
        p=[0.28, 0.32, 0.10, 0.15, 0.15]
    )

    payment_method = np.random.choice(
        ["cash", "card", "wallet"],
        size=n_samples,
        p=[0.45, 0.40, 0.15]
    )

    weather_condition = np.random.choice(
        ["clear", "rain", "snow", "fog"],
        size=n_samples,
        p=[0.65, 0.20, 0.10, 0.05]
    )

    # Cancellation probability logic
    risk_score = (
        -2.2
        + 0.10 * waiting_time_min
        + 0.45 * is_rush_hour
        + 0.25 * is_weekend
        + 0.35 * (surge_multiplier - 1)
        + 0.18 * user_past_cancellations
        + 0.12 * driver_past_cancellations
        - 0.65 * (driver_rating - 4.5)
        - 0.25 * (user_rating - 4.5)
        + 0.015 * estimated_duration_min
    )

    risk_score += np.where(weather_condition == "rain", 0.35, 0)
    risk_score += np.where(weather_condition == "snow", 0.55, 0)
    risk_score += np.where(weather_condition == "fog", 0.25, 0)
    risk_score += np.where(pickup_area == "airport", 0.30, 0)
    risk_score += np.where(payment_method == "cash", 0.15, 0)

    cancellation_probability = sigmoid(risk_score)
    is_cancelled = np.random.binomial(1, cancellation_probability)

    df = pd.DataFrame({
        "ride_distance_km": ride_distance_km.round(2),
        "estimated_duration_min": estimated_duration_min.round(2),
        "estimated_price": estimated_price.round(2),
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "surge_multiplier": surge_multiplier,
        "waiting_time_min": waiting_time_min.round(2),
        "driver_rating": driver_rating.round(2),
        "user_rating": user_rating.round(2),
        "user_past_cancellations": user_past_cancellations,
        "driver_past_cancellations": driver_past_cancellations,
        "pickup_area": pickup_area,
        "dropoff_area": dropoff_area,
        "payment_method": payment_method,
        "weather_condition": weather_condition,
        "is_cancelled": is_cancelled
    })

    return df


def main() -> None:
    os.makedirs("data/raw", exist_ok=True)

    df = generate_ride_data()
    output_path = "data/raw/rides.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print("\nTarget distribution:")
    print(df["is_cancelled"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()