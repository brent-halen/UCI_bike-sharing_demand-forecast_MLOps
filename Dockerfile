FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts
COPY tests /app/tests
COPY Makefile /app/Makefile

RUN python -m pip install --upgrade pip && \
    python -m pip install -e ".[dev]"

EXPOSE 8000

CMD ["uvicorn", "bike_demand.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
