from bike_demand.monitoring import generate_drift_report
from bike_demand.logging_utils import configure_logging


if __name__ == "__main__":
    configure_logging("bike-demand-monitoring")
    output_path = generate_drift_report()
    print(f"Drift report written to {output_path}")
