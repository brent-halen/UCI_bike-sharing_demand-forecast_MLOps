from __future__ import annotations

import argparse
from pathlib import Path

from bike_demand.batch import batch_predict
from bike_demand.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch bike demand inference from a CSV file.")
    parser.add_argument("--input", required=True, help="Path to an input CSV containing model features.")
    parser.add_argument("--output", required=True, help="Path to write the predictions CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging("bike-demand-batch")
    args = parse_args()
    output_path = batch_predict(input_path=Path(args.input), output_path=Path(args.output))
    print(f"Batch predictions written to {output_path}")
