# Ride Cancellation Prediction System

An end-to-end machine learning project for predicting ride cancellation risk in a ride-hailing scenario.

The project demonstrates the full ML delivery cycle: synthetic data generation, data preprocessing, model training, evaluation, model serialization, and deployment as a FastAPI inference service.

## Business Problem

Ride-hailing platforms may lose revenue and user trust when rides are cancelled after matching. Predicting cancellation risk before or shortly after matching can help platforms improve dispatching, user experience, and operational reliability.

## Objective

Build a production-style ML pipeline that predicts whether a ride is likely to be cancelled.

## ML Task

Binary classification:

- `1` — ride is likely to be cancelled
- `0` — ride is likely to be completed

## Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- RandomForestClassifier
- FastAPI
- Pydantic
- Uvicorn
- Docker
- joblib

## Project Structure

```text
ride-cancellation-ml-system/
│
├── data/
│   └── raw/
│
├── models/
│   └── model.joblib
│
├── reports/
│   └── metrics.json
│
├── src/
│   ├── generate_data.py
│   ├── train.py
│   ├── predict.py
│   └── api.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md