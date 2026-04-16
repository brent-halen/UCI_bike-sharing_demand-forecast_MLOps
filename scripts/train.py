from bike_demand.monitoring import generate_drift_report
from bike_demand.logging_utils import configure_logging
from bike_demand.training import train_and_register


if __name__ == "__main__":
    configure_logging("bike-demand-training")
    manifest = train_and_register()
    report_path = generate_drift_report()
    print(f"Best model: {manifest['selected_model_label']}")
    print(f"Registered as: {manifest['registered_model_name']}@{manifest['registered_model_alias']}")
    print(f"Drift report: {report_path}")
