from __future__ import annotations

from fastapi import FastAPI, HTTPException

from bike_demand.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from bike_demand.api.service import PredictorService
from bike_demand.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(service: PredictorService | None = None) -> FastAPI:
    configure_logging("bike-demand-api")
    app = FastAPI(title="Bike Demand Forecast API", version="0.1.0")

    try:
        app.state.service = service or PredictorService.from_settings()
    except FileNotFoundError:
        app.state.service = None
        logger.warning(
            "API started without a loaded model.",
            extra={"event": "api_started_without_model", "extra_fields": {}},
        )
    else:
        logger.info(
            "API started with a loaded model.",
            extra={
                "event": "api_started_with_model",
                "extra_fields": {
                    "serving_model_uri": app.state.service.manifest.get("pinned_model_uri")
                    or app.state.service.manifest.get("model_uri")
                },
            },
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        logger.info(
            "Health check requested.",
            extra={
                "event": "health_checked",
                "extra_fields": {"model_ready": app.state.service is not None},
            },
        )
        return HealthResponse(status="ok", model_ready=app.state.service is not None)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        if app.state.service is None:
            raise HTTPException(status_code=503, detail="Model is not available.")
        logger.info(
            "Model info requested.",
            extra={"event": "model_info_requested", "extra_fields": {}},
        )
        return ModelInfoResponse(**app.state.service.model_info())

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest) -> PredictionResponse:
        if app.state.service is None:
            raise HTTPException(status_code=503, detail="Model is not available.")
        instances = [instance.model_dump() for instance in payload.instances]
        predictions = app.state.service.predict(instances)
        return PredictionResponse(predictions=predictions)

    return app


app = create_app()
