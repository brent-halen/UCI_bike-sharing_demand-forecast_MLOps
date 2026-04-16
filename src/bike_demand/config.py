from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    models_dir: Path
    mlflow_artifacts_dir: Path
    mlflow_db_path: Path
    report_path: Path
    raw_dataset_path: Path
    training_reference_path: Path
    holdout_batch_path: Path
    batch_predictions_dir: Path
    serving_manifest_path: Path
    dataset_url: str
    experiment_name: str
    registered_model_name: str
    champion_alias: str
    mlflow_tracking_uri: str


def _default_tracking_uri(project_root: Path) -> str:
    db_path = project_root.joinpath("mlflow.db").resolve()
    return f"sqlite:///{db_path.as_posix()}"


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    models_dir = project_root / "models"
    mlflow_artifacts_dir = project_root / "mlruns"

    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        models_dir=models_dir,
        mlflow_artifacts_dir=mlflow_artifacts_dir,
        mlflow_db_path=project_root / "mlflow.db",
        report_path=project_root / "drift_report.html",
        raw_dataset_path=raw_data_dir / "hour.csv",
        training_reference_path=processed_data_dir / "training_reference.csv",
        holdout_batch_path=processed_data_dir / "holdout_batch.csv",
        batch_predictions_dir=processed_data_dir / "predictions",
        serving_manifest_path=models_dir / "serving_manifest.json",
        dataset_url="https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "bike-demand-forecasting"),
        registered_model_name=os.getenv("REGISTERED_MODEL_NAME", "bike-demand-forecaster"),
        champion_alias=os.getenv("MODEL_ALIAS", "champion"),
        mlflow_tracking_uri=os.getenv(
            "MLFLOW_TRACKING_URI",
            _default_tracking_uri(project_root),
        ),
    )


def ensure_project_dirs(settings: Settings) -> None:
    for path in (
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.batch_predictions_dir,
        settings.models_dir,
        settings.mlflow_artifacts_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
