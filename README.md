# Ride Cancellation Prediction System

End-to-end machine learning system for predicting ride cancellation risk in a ride-hailing scenario.

This project demonstrates a production-oriented ML workflow: synthetic data generation, data preprocessing, model training, model evaluation, model serialization, API deployment with FastAPI, Docker packaging, and basic API testing.

## Business Problem

Ride-hailing platforms may lose revenue and user trust when rides are cancelled after matching. A cancellation risk prediction system can help identify high-risk rides before or shortly after dispatch and support better operational decisions.

The goal of this project is to build a machine learning service that predicts the probability of ride cancellation based on ride, user, driver, pricing, timing, and context-related features.

## Objective

Build a production-style machine learning pipeline that predicts whether a ride is likely to be cancelled.

The project covers:

- data generation;
- data preprocessing;
- feature engineering;
- model training;
- model evaluation;
- model serialization;
- API inference service;
- input validation;
- basic API testing;
- Docker-based deployment setup.

## ML Task

Binary classification:

- `1` — ride is likely to be cancelled;
- `0` — ride is likely to be completed.

The model returns both:

- binary prediction;
- cancellation probability;
- risk label: `low` or `high`.

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- RandomForestClassifier
- FastAPI
- Pydantic
- Uvicorn
- pytest
- Docker
- joblib

## Project Structure

```text
ride-cancellation-ml-system/
│
├── data/
│   └── raw/
│       └── rides.csv
│
├── models/
│   └── model.joblib
│
├── reports/
│   └── metrics.json
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py
│   ├── train.py
│   ├── predict.py
│   └── api.py
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

## Dataset

Real ride-hailing cancellation data is usually proprietary, so this project uses a synthetic dataset designed to simulate common ride-hailing patterns.

The dataset includes features such as:

- ride distance;
- estimated ride duration;
- estimated price;
- hour of day;
- day of week;
- weekend indicator;
- rush hour indicator;
- surge multiplier;
- waiting time;
- driver rating;
- user rating;
- user past cancellations;
- driver past cancellations;
- pickup area;
- dropoff area;
- payment method;
- weather condition.

Target variable:

```text
is_cancelled
```

## Feature Logic

The synthetic target is generated using realistic cancellation risk assumptions:

- longer waiting time increases cancellation risk;
- rush hour increases cancellation risk;
- bad weather increases cancellation risk;
- higher surge multiplier increases cancellation risk;
- users with more past cancellations have higher cancellation probability;
- drivers with lower rating may increase cancellation probability;
- airport pickup and cash payment may slightly increase cancellation risk.

## System Architecture

```text
Synthetic Ride Data
        │
        ▼
Data Generation
src/generate_data.py
        │
        ▼
Training Pipeline
src/train.py
        │
        ├── Data preprocessing
        ├── Feature encoding
        ├── Model training
        └── Model evaluation
        │
        ▼
Serialized ML Pipeline
models/model.joblib
        │
        ▼
FastAPI Inference Service
src/api.py
        │
        ├── Input validation with Pydantic
        ├── Probability prediction
        └── Risk label generation
        │
        ▼
Prediction Response
low / high cancellation risk
```

## ML Pipeline

The machine learning pipeline includes:

1. Synthetic ride data generation.
2. Train-test split with stratification.
3. Numeric feature scaling.
4. Categorical feature one-hot encoding.
5. Model training with `RandomForestClassifier`.
6. Evaluation using classification metrics.
7. Saving the full preprocessing + model pipeline with `joblib`.
8. Serving predictions through FastAPI.

The preprocessing and model are saved together as a single sklearn pipeline, which reduces the risk of training-serving skew.

## Model Performance

The model is evaluated on a hold-out test set using standard binary classification metrics.

| Metric | Value |
|---|---:|
| Accuracy | 0.6715 |
| Precision | 0.4565 |
| Recall | 0.4884 |
| F1-score | 0.4719 |
| ROC-AUC | 0.6582 |
| PR-AUC | 0.4502 |

Metrics are saved in:

```text
reports/metrics.json
```

## How to Run Locally

### 1. Clone repository

```bash
git clone https://github.com/SUBIIHUM/ride-cancellation-ml-system.git
cd ride-cancellation-ml-system
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

For Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate synthetic data

```bash
python src/generate_data.py
```

### 5. Train model

```bash
python src/train.py
```

### 6. Run local prediction script

```bash
python src/predict.py
```

### 7. Run FastAPI service

```bash
uvicorn src.api:app --reload
```

Open API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Prediction Endpoint

```text
POST /predict
```

Example request:

```json
{
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
  "weather_condition": "rain"
}
```

Example response:

```json
{
  "prediction": 1,
  "cancellation_probability": 0.6234,
  "risk_label": "high"
}
```

## Run with Docker

Build Docker image:

```bash
docker build -t ride-cancellation-api .
```

Run container:

```bash
docker run -p 8000:8000 ride-cancellation-api
```

Open API documentation:

```text
http://127.0.0.1:8000/docs
```

## Testing

The project includes basic API tests for:

- health check endpoint;
- prediction endpoint;
- response schema;
- prediction probability range;
- risk label validity.

Run tests:

```bash
pytest -v
```

Expected output:

```text
tests/test_api.py::test_health_check PASSED
tests/test_api.py::test_predict_endpoint PASSED
```

## Production Considerations

Although this project uses synthetic data, it is designed to follow production-oriented ML engineering practices:

- preprocessing and model are saved as a single sklearn pipeline;
- API input is validated using Pydantic schemas;
- inference returns both prediction and probability;
- project includes basic API tests;
- Dockerfile is provided for containerized deployment;
- model metrics are saved as reproducible artifacts;
- monitoring requirements are documented for future production usage.

## Monitoring Plan

In a real production environment, the following should be monitored:

### Model Monitoring

- prediction probability distribution;
- cancellation rate by time, region, and user segment;
- feature distribution drift;
- model performance after delayed ground-truth labels become available;
- precision, recall, ROC-AUC, PR-AUC over time.

### Data Monitoring

- missing values;
- invalid feature values;
- unseen categorical values;
- changes in input feature distribution;
- data freshness.

### API Monitoring

- API latency;
- request volume;
- error rate;
- failed validation requests;
- model loading errors.

## Business Value

A cancellation risk prediction model can support several product and operational decisions:

- identifying high-risk ride requests before dispatch;
- improving driver-passenger matching reliability;
- reducing failed matches;
- improving customer experience;
- supporting cancellation reduction experiments;
- helping operations teams monitor cancellation risk across different time periods and regions.

## Limitations

This project uses synthetic data, so the model should not be interpreted as a real production model for an actual ride-hailing platform.

The goal of the project is to demonstrate an end-to-end ML engineering workflow rather than to optimize real-world cancellation prediction performance.

Main limitations:

- synthetic data does not fully represent real user behavior;
- no real-time feature store is used;
- no online monitoring dashboard is implemented;
- no automated retraining pipeline is included;
- no A/B testing framework is connected.

## Future Improvements

Possible improvements:

- replace synthetic data with real ride-hailing data;
- add MLflow experiment tracking;
- add model registry;
- add batch inference pipeline;
- add Airflow or Prefect orchestration;
- add data validation with Pandera or Great Expectations;
- add model drift monitoring;
- add CI/CD pipeline;
- add deployment to cloud platform;
- add dashboard for model and API monitoring;
- add retraining workflow.

## Author

**Abilmansur Muratbay**

Machine Learning / Data Science Master's student focused on applied ML, MLOps, and production-ready machine learning systems.