# Bike-Sharing Demand Forecasting MLOps Exercise

This repository scaffolds an end-to-end Python MLOps project around the UCI Bike Sharing hourly dataset. It trains a simple baseline and a stronger gradient boosting model with a strict time-based split, logs runs to MLflow, registers the best model, serves predictions through FastAPI, and generates an Evidently drift report comparing the training reference set to a holdout batch.

## Stack

- `scikit-learn` for modeling
- `MLflow` for experiment tracking and model registry
- `FastAPI` for online inference
- `Evidently` for drift reporting
- `pytest` for tests
- `Docker` and `docker-compose` for containerized local development
- `GitHub Actions` for CI

## Project layout

```text
.
|-- .github/workflows/ci.yml
|-- Dockerfile
|-- Makefile
|-- docker-compose.yml
|-- scripts/
|-- src/bike_demand/
`-- tests/
```

## Local setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
make install
```

## Run locally

Train the models, log both runs to MLflow, register the best model, save monitoring batches, and generate `drift_report.html`:

```bash
make train
```

Start the FastAPI service:

```bash
make serve
```

Run batch inference from a CSV and write predictions to a new CSV:

```bash
make batch-predict
```

Run the drift report again manually if you want to refresh it:

```bash
make drift
```

Run tests:

```bash
make test
```

## MLflow

The local workflow uses a SQLite-backed MLflow tracking URI by default and stores artifacts under `./mlruns`. After `make train`, inspect runs with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

The best model is registered under `bike-demand-forecaster` and tagged with the alias `champion`.

The serving manifest pins the exact registered model version so the API and batch jobs keep serving the same artifact even if the alias moves later.

## API endpoints

### Health

```bash
curl http://localhost:8000/health
```

### Model info

```bash
curl http://localhost:8000/model-info
```

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "season": 2,
        "yr": 1,
        "mnth": 7,
        "hr": 8,
        "holiday": 0,
        "weekday": 2,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.62,
        "atemp": 0.58,
        "hum": 0.44,
        "windspeed": 0.12
      }
    ]
  }'
```

## Batch inference

The batch job expects a CSV containing the model feature columns:

- `season`
- `yr`
- `mnth`
- `hr`
- `holiday`
- `weekday`
- `workingday`
- `weathersit`
- `temp`
- `atemp`
- `hum`
- `windspeed`

Run it with defaults:

```bash
make batch-predict
```

Or override the input and output paths:

```bash
make batch-predict INPUT=data/processed/holdout_batch.csv OUTPUT=data/processed/predictions/my_predictions.csv
```

Direct script usage:

```bash
python scripts/batch_predict.py --input data/processed/holdout_batch.csv --output data/processed/predictions/my_predictions.csv
```

## Docker

Build and start the API plus an MLflow server:

```bash
docker compose up --build
```

To trigger training against the MLflow server running in Docker:

```bash
docker compose run --rm api make train
```

## Modeling workflow

1. Download `hour.csv` from the UCI Bike Sharing dataset archive.
2. Sort observations by timestamp and split them into train, validation, and holdout segments in chronological order.
3. Train a baseline `DummyRegressor`.
4. Train a stronger `HistGradientBoostingRegressor`.
5. Select the best model by validation RMSE.
6. Refit the winner on train plus validation data.
7. Log both candidates to MLflow and register the winner as the `champion` alias.
8. Pin the serving manifest to the exact registered model version selected during training.
9. Save `data/processed/training_reference.csv`, `data/processed/holdout_batch.csv`, and `drift_report.html`.

## Notes

- The API loads model metadata from `models/serving_manifest.json`, which is created by the training workflow.
- Training, serving, monitoring, and batch inference emit structured JSON logs to stdout.
- If `/model-info` or `/predict` returns `503`, run `make train` first so the serving manifest and registered model exist.
