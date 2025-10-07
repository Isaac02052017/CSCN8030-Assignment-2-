


"""
Model utilities for Crop Yield Prediction (No GenAI prototype).
"""
from __future__ import annotations
import math
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_linear_regression(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_random_forest(
    X: pd.DataFrame, y: pd.Series,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    random_state: int = 42
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def save_model(model, path: str) -> None:
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)
