from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from bike_demand.data import FEATURE_COLUMNS

CATEGORICAL_COLUMNS = ["season", "weathersit"]
PASSTHROUGH_NUMERIC_COLUMNS = [
    "yr",
    "holiday",
    "workingday",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]


def enrich_features(frame: pd.DataFrame) -> pd.DataFrame:
    feature_frame = frame.loc[:, FEATURE_COLUMNS].copy()
    feature_frame["month_sin"] = np.sin(2 * np.pi * feature_frame["mnth"] / 12.0)
    feature_frame["month_cos"] = np.cos(2 * np.pi * feature_frame["mnth"] / 12.0)
    feature_frame["hour_sin"] = np.sin(2 * np.pi * feature_frame["hr"] / 24.0)
    feature_frame["hour_cos"] = np.cos(2 * np.pi * feature_frame["hr"] / 24.0)
    feature_frame["weekday_sin"] = np.sin(2 * np.pi * feature_frame["weekday"] / 7.0)
    feature_frame["weekday_cos"] = np.cos(2 * np.pi * feature_frame["weekday"] / 7.0)
    return feature_frame


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", "passthrough", PASSTHROUGH_NUMERIC_COLUMNS),
        ]
    )


def build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_enrichment", FunctionTransformer(enrich_features, validate=False)),
            ("preprocess", build_preprocessor()),
            ("regressor", estimator),
        ]
    )


def build_candidate_models() -> dict[str, Any]:
    return {
        "baseline_dummy": DummyRegressor(strategy="mean"),
        "gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=400,
            min_samples_leaf=20,
            random_state=42,
        ),
    }


def fit_for_selection(
    estimator: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    pipeline = build_pipeline(clone(estimator))
    pipeline.fit(x_train, y_train)
    return pipeline


def fit_final_model(
    estimator: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    pipeline = build_pipeline(clone(estimator))
    pipeline.fit(x_train, y_train)
    return pipeline


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
    }
