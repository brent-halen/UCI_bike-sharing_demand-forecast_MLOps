from __future__ import annotations

import pandas as pd

from bike_demand.data import TIMESTAMP_COLUMN, time_based_split
from bike_demand.modeling import enrich_features


def test_time_based_split_preserves_chronology() -> None:
    frame = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.date_range("2024-01-01", periods=20, freq="h"),
            "season": [1] * 20,
            "yr": [0] * 20,
            "mnth": [1] * 20,
            "hr": list(range(20)),
            "holiday": [0] * 20,
            "weekday": [0] * 20,
            "workingday": [1] * 20,
            "weathersit": [1] * 20,
            "temp": [0.3] * 20,
            "atemp": [0.3] * 20,
            "hum": [0.5] * 20,
            "windspeed": [0.2] * 20,
            "cnt": list(range(20)),
        }
    )

    split = time_based_split(frame)

    assert len(split.train) == 14
    assert len(split.validation) == 3
    assert len(split.test) == 3
    assert split.train[TIMESTAMP_COLUMN].max() < split.validation[TIMESTAMP_COLUMN].min()
    assert split.validation[TIMESTAMP_COLUMN].max() < split.test[TIMESTAMP_COLUMN].min()


def test_enrich_features_adds_cyclical_columns() -> None:
    frame = pd.DataFrame(
        {
            "season": [1],
            "yr": [0],
            "mnth": [6],
            "hr": [12],
            "holiday": [0],
            "weekday": [3],
            "workingday": [1],
            "weathersit": [1],
            "temp": [0.5],
            "atemp": [0.4],
            "hum": [0.6],
            "windspeed": [0.2],
        }
    )

    enriched = enrich_features(frame)

    for column in [
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
    ]:
        assert column in enriched.columns
