from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BikeDemandFeatures(BaseModel):
    season: int = Field(ge=1, le=4)
    yr: int = Field(ge=0, le=1)
    mnth: int = Field(ge=1, le=12)
    hr: int = Field(ge=0, le=23)
    holiday: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=6)
    workingday: int = Field(ge=0, le=1)
    weathersit: int = Field(ge=1, le=4)
    temp: float
    atemp: float
    hum: float
    windspeed: float


class PredictionRequest(BaseModel):
    instances: list[BikeDemandFeatures]


class PredictionResponse(BaseModel):
    predictions: list[float]


class HealthResponse(BaseModel):
    status: str
    model_ready: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str | None = None
    model_alias: str | None = None
    serving_model_uri: str | None = None
    alias_model_uri: str | None = None
    fallback_model_uri: str | None = None
    selected_model_label: str | None = None
    best_run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    feature_columns: list[str] = Field(default_factory=list)
    tracking_uri: str | None = None
    manifest: dict[str, Any] = Field(default_factory=dict)
