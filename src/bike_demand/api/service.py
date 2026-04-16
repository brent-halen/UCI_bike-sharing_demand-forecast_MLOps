from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd

from bike_demand.config import Settings, get_settings
from bike_demand.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PredictorService:
    model: Any
    manifest: dict[str, Any]

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "PredictorService":
        settings = settings or get_settings()
        manifest_path = settings.serving_manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(
                "Serving manifest not found. Run `python scripts/train.py` first."
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_registry_uri(settings.mlflow_tracking_uri)

        model_uri = manifest.get("pinned_model_uri") or manifest.get("model_uri")
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(
                "Loaded serving model from pinned URI.",
                extra={
                    "event": "serving_model_loaded",
                    "extra_fields": {
                        "model_uri": model_uri,
                        "registered_model_version": manifest.get("registered_model_version"),
                    },
                },
            )
        except Exception:
            fallback_uri = manifest.get("fallback_model_uri")
            if not fallback_uri:
                raise
            model = mlflow.pyfunc.load_model(fallback_uri)
            logger.warning(
                "Pinned model URI load failed; using fallback run URI.",
                extra={
                    "event": "serving_model_fallback_loaded",
                    "extra_fields": {
                        "pinned_model_uri": model_uri,
                        "fallback_model_uri": fallback_uri,
                    },
                },
            )
        return cls(model=model, manifest=manifest)

    def predict(self, instances: list[dict[str, Any]]) -> list[float]:
        frame = pd.DataFrame(instances)
        predictions = self.model.predict(frame)
        output = [float(value) for value in predictions]
        logger.info(
            "Generated predictions.",
            extra={
                "event": "predictions_generated",
                "extra_fields": {
                    "instance_count": len(instances),
                    "serving_model_uri": self.manifest.get("pinned_model_uri") or self.manifest.get("model_uri"),
                },
            },
        )
        return output

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.manifest.get("registered_model_name", "unregistered"),
            "model_version": self.manifest.get("registered_model_version"),
            "model_alias": self.manifest.get("registered_model_alias"),
            "serving_model_uri": self.manifest.get("pinned_model_uri") or self.manifest.get("model_uri"),
            "alias_model_uri": self.manifest.get("alias_model_uri"),
            "fallback_model_uri": self.manifest.get("fallback_model_uri"),
            "selected_model_label": self.manifest.get("selected_model_label"),
            "best_run_id": self.manifest.get("best_run_id"),
            "metrics": self.manifest.get("metrics", {}),
            "feature_columns": self.manifest.get("feature_columns", []),
            "tracking_uri": self.manifest.get("tracking_uri"),
            "manifest": self.manifest,
        }
