from __future__ import annotations

from fastapi.testclient import TestClient

from bike_demand.api.app import create_app


class DummyService:
    def predict(self, instances):
        return [42.0 for _ in instances]

    def model_info(self):
        return {
            "model_name": "bike-demand-forecaster",
            "model_version": "7",
            "model_alias": "champion",
            "serving_model_uri": "models:/bike-demand-forecaster/7",
            "alias_model_uri": "models:/bike-demand-forecaster@champion",
            "fallback_model_uri": "runs:/run-123/model",
            "selected_model_label": "gradient_boosting",
            "best_run_id": "run-123",
            "metrics": {"test_rmse": 12.3},
            "feature_columns": ["season", "yr", "mnth"],
            "tracking_uri": "sqlite:///mlflow.db",
            "manifest": {"selected_model_label": "gradient_boosting"},
        }


def test_predict_endpoint_returns_predictions() -> None:
    app = create_app(service=DummyService())
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "instances": [
                {
                    "season": 1,
                    "yr": 1,
                    "mnth": 6,
                    "hr": 8,
                    "holiday": 0,
                    "weekday": 2,
                    "workingday": 1,
                    "weathersit": 1,
                    "temp": 0.62,
                    "atemp": 0.58,
                    "hum": 0.45,
                    "windspeed": 0.12,
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {"predictions": [42.0]}


def test_health_and_model_info_endpoints() -> None:
    app = create_app(service=DummyService())
    client = TestClient(app)

    health_response = client.get("/health")
    model_info_response = client.get("/model-info")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok", "model_ready": True}
    assert model_info_response.status_code == 200
    assert model_info_response.json()["model_alias"] == "champion"
    assert model_info_response.json()["serving_model_uri"] == "models:/bike-demand-forecaster/7"
