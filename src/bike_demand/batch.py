from __future__ import annotations

from pathlib import Path

import pandas as pd

from bike_demand.api.service import PredictorService
from bike_demand.config import Settings, get_settings
from bike_demand.data import FEATURE_COLUMNS
from bike_demand.logging_utils import get_logger

logger = get_logger(__name__)


def batch_predict(
    input_path: Path,
    output_path: Path,
    settings: Settings | None = None,
    service: PredictorService | None = None,
) -> Path:
    settings = settings or get_settings()
    predictor = service or PredictorService.from_settings(settings)

    frame = pd.read_csv(input_path)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"Input file is missing required feature columns: {', '.join(missing_columns)}"
        )

    predictions = predictor.predict(frame.loc[:, FEATURE_COLUMNS].to_dict(orient="records"))
    result = frame.copy()
    result["predicted_cnt"] = predictions

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(
        "Batch predictions written.",
        extra={
            "event": "batch_predictions_completed",
            "extra_fields": {
                "input_path": str(input_path.resolve()),
                "output_path": str(output_path.resolve()),
                "row_count": len(result),
            },
        },
    )
    return output_path
