PYTHON ?= python
UVICORN ?= uvicorn
HOST ?= 0.0.0.0
PORT ?= 8000
INPUT ?= data/processed/holdout_batch.csv
OUTPUT ?= data/processed/predictions/batch_predictions.csv

.PHONY: install train serve drift test batch-predict

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

train:
	PYTHONPATH=src $(PYTHON) scripts/train.py

serve:
	PYTHONPATH=src $(UVICORN) bike_demand.api.app:app --host $(HOST) --port $(PORT)

drift:
	PYTHONPATH=src $(PYTHON) scripts/generate_drift_report.py

batch-predict:
	PYTHONPATH=src $(PYTHON) scripts/batch_predict.py --input "$(INPUT)" --output "$(OUTPUT)"

test:
	PYTHONPATH=src $(PYTHON) -m pytest
