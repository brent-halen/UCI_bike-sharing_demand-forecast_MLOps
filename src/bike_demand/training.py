from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from bike_demand.config import Settings, ensure_project_dirs, get_settings
from bike_demand.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    feature_target_split,
    load_dataset,
    save_monitoring_batches,
    time_based_split,
)
from bike_demand.modeling import (
    build_candidate_models,
    fit_final_model,
    fit_for_selection,
    regression_metrics,
)
from bike_demand.logging_utils import get_logger

logger = get_logger(__name__)

def configure_mlflow(settings: Settings) -> str:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(settings.experiment_name)
    if experiment is None:
        artifact_location = settings.mlflow_artifacts_dir.resolve().as_uri()
        experiment_id = client.create_experiment(
            name=settings.experiment_name,
            artifact_location=artifact_location,
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(settings.experiment_name)
    logger.info(
        "Configured MLflow.",
        extra={
            "event": "mlflow_configured",
            "extra_fields": {
                "tracking_uri": settings.mlflow_tracking_uri,
                "experiment_name": settings.experiment_name,
                "experiment_id": experiment_id,
            },
        },
    )
    return experiment_id


def _wait_for_model_version(
    client: MlflowClient,
    model_name: str,
    model_version: str,
    timeout_seconds: int = 60,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        version = client.get_model_version(model_name, model_version)
        if version.status == "READY":
            return
        time.sleep(1)


def register_best_run(
    run_id: str,
    settings: Settings,
) -> dict[str, str]:
    client = MlflowClient()
    try:
        client.create_registered_model(settings.registered_model_name)
    except Exception:
        pass

    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=settings.registered_model_name,
    )
    _wait_for_model_version(client, settings.registered_model_name, registered.version)
    client.set_registered_model_alias(
        settings.registered_model_name,
        settings.champion_alias,
        registered.version,
    )
    pinned_model_uri = f"models:/{settings.registered_model_name}/{registered.version}"
    alias_model_uri = f"models:/{settings.registered_model_name}@{settings.champion_alias}"

    logger.info(
        "Registered best model version.",
        extra={
            "event": "model_registered",
            "extra_fields": {
                "registered_model_name": settings.registered_model_name,
                "registered_model_version": str(registered.version),
                "registered_model_alias": settings.champion_alias,
                "pinned_model_uri": pinned_model_uri,
                "alias_model_uri": alias_model_uri,
            },
        },
    )

    return {
        "registered_model_name": settings.registered_model_name,
        "registered_model_version": str(registered.version),
        "registered_model_alias": settings.champion_alias,
        "model_uri": pinned_model_uri,
        "pinned_model_uri": pinned_model_uri,
        "alias_model_uri": alias_model_uri,
        "fallback_model_uri": model_uri,
    }


def _write_serving_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def train_and_register(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    ensure_project_dirs(settings)
    configure_mlflow(settings)
    logger.info(
        "Starting training workflow.",
        extra={
            "event": "training_started",
            "extra_fields": {
                "registered_model_name": settings.registered_model_name,
                "tracking_uri": settings.mlflow_tracking_uri,
            },
        },
    )

    dataset = load_dataset(settings)
    split = time_based_split(dataset)
    save_monitoring_batches(split, settings)
    logger.info(
        "Prepared time-based splits.",
        extra={
            "event": "time_split_completed",
            "extra_fields": {
                "train_rows": len(split.train),
                "validation_rows": len(split.validation),
                "test_rows": len(split.test),
            },
        },
    )

    x_train, y_train = feature_target_split(split.train)
    x_validation, y_validation = feature_target_split(split.validation)
    x_test, y_test = feature_target_split(split.test)
    x_train_validation = pd.concat([x_train, x_validation], ignore_index=True)
    y_train_validation = pd.concat([y_train, y_validation], ignore_index=True)

    run_summaries: list[dict[str, Any]] = []
    best_run: dict[str, Any] | None = None

    with mlflow.start_run(run_name="bike-demand-training") as parent_run:
        mlflow.log_params(
            {
                "dataset": "UCI Bike Sharing Dataset (hour.csv)",
                "train_ratio": 0.7,
                "validation_ratio": 0.15,
                "test_ratio": 0.15,
                "feature_count": len(FEATURE_COLUMNS),
                "target_column": TARGET_COLUMN,
            }
        )

        for model_label, estimator in build_candidate_models().items():
            with mlflow.start_run(run_name=model_label, nested=True) as child_run:
                logger.info(
                    "Training candidate model.",
                    extra={
                        "event": "candidate_training_started",
                        "extra_fields": {"model_label": model_label},
                    },
                )
                selection_model = fit_for_selection(estimator, x_train, y_train)
                validation_predictions = selection_model.predict(x_validation)
                validation_metrics = regression_metrics(
                    y_validation,
                    validation_predictions,
                    prefix="val",
                )

                final_model = fit_final_model(estimator, x_train_validation, y_train_validation)
                test_predictions = final_model.predict(x_test)
                test_metrics = regression_metrics(y_test, test_predictions, prefix="test")
                metrics = {**validation_metrics, **test_metrics}

                mlflow.log_params({"model_label": model_label})
                mlflow.log_metrics(metrics)
                mlflow.set_tags(
                    {
                        "dataset_source": "uci-bike-sharing-hourly",
                        "selection_metric": "val_rmse",
                    }
                )

                sample_input = x_train_validation.head(5)
                sample_predictions = final_model.predict(sample_input)
                signature = infer_signature(sample_input, sample_predictions)
                mlflow.sklearn.log_model(
                    sk_model=final_model,
                    artifact_path="model",
                    signature=signature,
                    input_example=sample_input,
                )

                summary = {
                    "model_label": model_label,
                    "run_id": child_run.info.run_id,
                    "metrics": metrics,
                }
                run_summaries.append(summary)
                logger.info(
                    "Finished candidate model.",
                    extra={
                        "event": "candidate_training_completed",
                        "extra_fields": {
                            "model_label": model_label,
                            "run_id": child_run.info.run_id,
                            **metrics,
                        },
                    },
                )

                if best_run is None or metrics["val_rmse"] < best_run["metrics"]["val_rmse"]:
                    best_run = summary

        if best_run is None:
            raise RuntimeError("No model candidates were trained.")

        registration = register_best_run(best_run["run_id"], settings)
        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "tracking_uri": settings.mlflow_tracking_uri,
            "experiment_name": settings.experiment_name,
            "parent_run_id": parent_run.info.run_id,
            "best_run_id": best_run["run_id"],
            "selected_model_label": best_run["model_label"],
            "metrics": best_run["metrics"],
            "candidate_runs": run_summaries,
            "feature_columns": FEATURE_COLUMNS,
            "reference_data_path": str(settings.training_reference_path.resolve()),
            "holdout_data_path": str(settings.holdout_batch_path.resolve()),
            **registration,
        }
        mlflow.log_dict(manifest, artifact_file="serving_manifest.json")
        _write_serving_manifest(manifest, settings.serving_manifest_path)
        logger.info(
            "Training workflow completed.",
            extra={
                "event": "training_completed",
                "extra_fields": {
                    "best_run_id": best_run["run_id"],
                    "selected_model_label": best_run["model_label"],
                    "registered_model_version": manifest["registered_model_version"],
                    "manifest_path": str(settings.serving_manifest_path.resolve()),
                },
            },
        )
        return manifest
