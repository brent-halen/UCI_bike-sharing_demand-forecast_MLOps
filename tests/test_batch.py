from __future__ import annotations

from pathlib import Path

import pandas as pd

from bike_demand.batch import batch_predict


class DummyService:
    def predict(self, instances):
        return [float(index) for index, _ in enumerate(instances, start=1)]


def test_batch_predict_writes_predictions(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame(
        [
            {
                "season": 1,
                "yr": 0,
                "mnth": 1,
                "hr": 8,
                "holiday": 0,
                "weekday": 1,
                "workingday": 1,
                "weathersit": 1,
                "temp": 0.3,
                "atemp": 0.31,
                "hum": 0.4,
                "windspeed": 0.1,
            },
            {
                "season": 2,
                "yr": 1,
                "mnth": 7,
                "hr": 17,
                "holiday": 0,
                "weekday": 2,
                "workingday": 1,
                "weathersit": 2,
                "temp": 0.6,
                "atemp": 0.62,
                "hum": 0.5,
                "windspeed": 0.15,
            },
        ]
    ).to_csv(input_path, index=False)

    batch_predict(input_path=input_path, output_path=output_path, service=DummyService())

    output = pd.read_csv(output_path)
    assert "predicted_cnt" in output.columns
    assert output["predicted_cnt"].tolist() == [1.0, 2.0]
