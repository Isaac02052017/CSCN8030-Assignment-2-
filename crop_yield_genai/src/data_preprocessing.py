

from __future__ import annotations
from typing import Tuple, List
import pandas as pd


"""
Data preprocessing & feature engineering (GenAI-assisted).
Adds Growing Degree Days (GDD_approx), categorical encoding,
robust feature selection, and time-based splitting with Year handling.
"""



import sys, importlib, inspect
import src.data_preprocessing as dp
importlib.reload(dp)




print("Loaded from:", dp.__file__)
print("Has ensure_year_column?", hasattr(dp, "ensure_year_column"))
print("Functions available:", [n for n, v in inspect.getmembers(dp) if inspect.isfunction(v)])







# -----------------------------
# Data loading and cleaning
# -----------------------------

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV, strip column whitespace, and drop fully-empty rows.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all")
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce numerics, clip to plausible ranges, and drop rows with missing target.
    """
    df = df.copy()
    num_cols = ["Rainfall_mm", "Temperature_C", "Fertilizer_kg", "Yield_kg_ha"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Rainfall_mm" in df.columns:
        df["Rainfall_mm"] = df["Rainfall_mm"].clip(0, 3000)
    if "Temperature_C" in df.columns:
        df["Temperature_C"] = df["Temperature_C"].clip(-10, 45)
    if "Fertilizer_kg" in df.columns:
        df["Fertilizer_kg"] = df["Fertilizer_kg"].clip(0, 400)

    if "Yield_kg_ha" in df.columns:
        df = df.dropna(subset=["Yield_kg_ha"])
    return df


# -----------------------------
# Feature engineering & encoding
# -----------------------------

def add_gdd_feature(df: pd.DataFrame, base_temp: float = 10.0) -> pd.DataFrame:
    """
    Approximate seasonal Growing Degree Days using mean Temperature_C.
    GDD_approx ≈ max(0, Temperature_C - base_temp) * 120
    (Assumes ~120-day growing season; adjust as needed.)
    """
    df = df.copy()
    if "Temperature_C" in df.columns:
        gdd_daily = (pd.to_numeric(df["Temperature_C"], errors="coerce") - float(base_temp)).clip(lower=0)
        df["GDD_approx"] = gdd_daily * 120.0
    else:
        df["GDD_approx"] = 0.0
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode Region and Crop as categorical codes (adds Region_cat, Crop_cat).
    """
    df = df.copy()
    for col in ["Region", "Crop"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
            df[col + "_cat"] = df[col].cat.codes
    return df


def select_features(
    df: pd.DataFrame,
    features: List[str] | None = None,
    target: str = "Yield_kg_ha",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Select default or provided features and return (X, y, features).
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    if features is None:
        # Prefer to include engineered GDD_approx if present
        candidate = ["Rainfall_mm", "Temperature_C", "Fertilizer_kg"]
        if "GDD_approx" in df.columns:
            candidate.append("GDD_approx")
        if "Region_cat" in df.columns:
            candidate.append("Region_cat")
        if "Crop_cat" in df.columns:
            candidate.append("Crop_cat")
        features = candidate

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in DataFrame: {missing}")

    X = df[features].copy()
    y = df[target].copy()
    return X, y, features


# -----------------------------
# Robust Year handling & split
# -----------------------------

def ensure_year_column(df: pd.DataFrame, preferred_name: str = "Year") -> pd.DataFrame:
    """
    Ensure a 'Year' column exists:
      1) If a case-insensitive 'year' column exists, rename to 'Year'
      2) Else, try deriving from a date-like column (Date/Timestamp/Datetime)
      3) Else, fabricate a Year sequence so the pipeline can run
    """
    df = df.copy()

    # 1) Case-insensitive rename if present
    for c in df.columns:
        if c == preferred_name or c.lower() == preferred_name.lower():
            if c != preferred_name:
                df = df.rename(columns={c: preferred_name})
            return df

    # 2) Derive from a date-like column
    date_candidates = [c for c in df.columns if c.lower() in ("date", "timestamp", "datetime", "date_time")]
    for c in date_candidates:
        ser = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        if ser.notna().any():
            df[preferred_name] = ser.dt.year
            return df

    # 3) Fabricate a Year if none found
    start_year = 2016
    df[preferred_name] = start_year + (pd.RangeIndex(len(df)) % 5)
    return df


def time_based_split(
    df: pd.DataFrame,
    time_col: str = "Year",
    test_years: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train/test by the last `test_years` worth of years.
    Robust to missing `Year`: will ensure/derive/fabricate it first.
    """
    df = ensure_year_column(df, preferred_name=time_col).copy()

    years = df[time_col].dropna().astype(int).sort_values().unique().tolist()
    if not years:
        raise ValueError("No valid years available for time-based split.")

    cutoff = years[0] if test_years >= len(years) else years[-test_years]
    train_df = df[df[time_col] < cutoff].copy()
    test_df = df[df[time_col] >= cutoff].copy()
    return train_df, test_df
