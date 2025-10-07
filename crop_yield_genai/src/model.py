

"""
Model utilities (GenAI-assisted): adds a compact RF hyperparameter grid.
"""
from __future__ import annotations
import math
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
import joblib

def train_linear_regression(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    m = LinearRegression()
    m.fit(X, y)
    return m

def suggested_rf_grid(small_data: bool = True) -> dict:
    """Compact grid, safe for ~150–500 rows."""
    if small_data:
        return {
            "n_estimators": [200, 350, 500],
            "max_depth": [None, 8, 14],
            "min_samples_split": [2, 5, 10],
        }
    else:
        return {
            "n_estimators": [300, 600, 900],
            "max_depth": [None, 12, 18],
            "min_samples_split": [2, 5, 10],
        }

def train_rf_gridsearch(
    X: pd.DataFrame, y: pd.Series,
    grid: dict | None = None, cv_splits: int = 5, random_state: int = 42
) -> Tuple[RandomForestRegressor, dict]:
    if grid is None:
        grid = suggested_rf_grid(small_data=True)
    base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    gs = GridSearchCV(base, param_grid=grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_

def evaluate(y_true, y_pred) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def save_model(model, path: str) -> None:
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)
