from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from bike_demand.config import Settings
from bike_demand.logging_utils import get_logger

FEATURE_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
TARGET_COLUMN = "cnt"
TIMESTAMP_COLUMN = "timestamp"
logger = get_logger(__name__)


@dataclass(frozen=True)
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def download_hourly_dataset(settings: Settings, force: bool = False) -> Path:
    if settings.raw_dataset_path.exists() and not force:
        logger.info(
            "Using cached dataset.",
            extra={
                "event": "dataset_cache_hit",
                "extra_fields": {"dataset_path": str(settings.raw_dataset_path.resolve())},
            },
        )
        return settings.raw_dataset_path

    logger.info(
        "Downloading hourly bike-sharing dataset.",
        extra={
            "event": "dataset_download_started",
            "extra_fields": {"dataset_url": settings.dataset_url},
        },
    )
    with urlopen(settings.dataset_url) as response:
        archive_bytes = response.read()

    with ZipFile(BytesIO(archive_bytes)) as archive:
        with archive.open("hour.csv") as dataset_file:
            settings.raw_dataset_path.write_bytes(dataset_file.read())

    logger.info(
        "Dataset downloaded.",
        extra={
            "event": "dataset_download_completed",
            "extra_fields": {"dataset_path": str(settings.raw_dataset_path.resolve())},
        },
    )
    return settings.raw_dataset_path


def load_dataset(settings: Settings) -> pd.DataFrame:
    dataset_path = download_hourly_dataset(settings)
    frame = pd.read_csv(dataset_path)
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame["dteday"]) + pd.to_timedelta(
        frame["hr"], unit="h"
    )
    frame = frame.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    logger.info(
        "Dataset loaded.",
        extra={
            "event": "dataset_loaded",
            "extra_fields": {
                "row_count": len(frame),
                "column_count": len(frame.columns),
                "dataset_path": str(dataset_path.resolve()),
            },
        },
    )
    return frame


def time_based_split(
    frame: pd.DataFrame, train_ratio: float = 0.7, validation_ratio: float = 0.15
) -> DataSplit:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1.")

    total_rows = len(frame)
    train_end = int(total_rows * train_ratio)
    validation_end = int(total_rows * (train_ratio + validation_ratio))

    return DataSplit(
        train=frame.iloc[:train_end].copy(),
        validation=frame.iloc[train_end:validation_end].copy(),
        test=frame.iloc[validation_end:].copy(),
    )


def feature_target_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.loc[:, FEATURE_COLUMNS].copy(), frame[TARGET_COLUMN].copy()


def save_monitoring_batches(split: DataSplit, settings: Settings) -> None:
    reference = split.train.loc[:, FEATURE_COLUMNS + [TARGET_COLUMN]]
    holdout = split.test.loc[:, FEATURE_COLUMNS + [TARGET_COLUMN]]
    reference.to_csv(settings.training_reference_path, index=False)
    holdout.to_csv(settings.holdout_batch_path, index=False)
    logger.info(
        "Saved monitoring batches.",
        extra={
            "event": "monitoring_batches_saved",
            "extra_fields": {
                "reference_rows": len(reference),
                "holdout_rows": len(holdout),
                "reference_path": str(settings.training_reference_path.resolve()),
                "holdout_path": str(settings.holdout_batch_path.resolve()),
            },
        },
    )
