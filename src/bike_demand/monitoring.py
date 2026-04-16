from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from bike_demand.config import Settings, get_settings
from bike_demand.logging_utils import get_logger

logger = get_logger(__name__)


def generate_drift_report(
    settings: Settings | None = None,
    output_path: Path | None = None,
) -> Path:
    settings = settings or get_settings()
    report_path = output_path or settings.report_path

    reference = pd.read_csv(settings.training_reference_path)
    current = pd.read_csv(settings.holdout_batch_path)
    logger.info(
        "Generating drift report.",
        extra={
            "event": "drift_report_started",
            "extra_fields": {
                "reference_rows": len(reference),
                "current_rows": len(current),
                "reference_path": str(settings.training_reference_path.resolve()),
                "current_path": str(settings.holdout_batch_path.resolve()),
            },
        },
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(report_path))
    logger.info(
        "Drift report generated.",
        extra={
            "event": "drift_report_completed",
            "extra_fields": {"report_path": str(report_path.resolve())},
        },
    )
    return report_path
