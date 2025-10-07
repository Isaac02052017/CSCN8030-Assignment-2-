# app.py — robust feature matching for saved models (GenAI & No-GenAI)
import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.model import load_model
from src.data_preprocessing import encode_categoricals, add_gdd_feature

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾")

# ---- Paths ----
BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

st.title("🌾 Crop Yield Predictor")
if not any(MODELS_DIR.glob("*.pkl")):
    st.warning("No saved models found. Train a model first (run the notebook).")
    st.stop()

# ---- Choose model ----
model_files = sorted(MODELS_DIR.glob("*.pkl"))
chosen_name = st.selectbox("Choose model", [f.name for f in model_files], index=0)
model_path = MODELS_DIR / chosen_name
model = load_model(str(model_path))

# ---- Load required feature order if available ----
features_path = model_path.with_suffix(".features.json")
required_features = None

if features_path.exists():
    try:
        required_features = json.loads(features_path.read_text(encoding="utf-8"))
    except Exception:
        required_features = None

# fallback: try scikit-learn metadata
if required_features is None:
    if hasattr(model, "feature_names_in_"):
        required_features = list(model.feature_names_in_)
    elif hasattr(model, "n_features_in_"):
        # We will match by length later if .features.json is missing
        required_features = None

# ---- UI inputs ----
col1, col2, col3 = st.columns(3)
with col1:
    rainfall = st.number_input("Rainfall (mm)", 0.0, 3000.0, 520.0, 10.0)
with col2:
    temperature = st.number_input("Temperature (°C)", -10.0, 45.0, 18.0, 0.1)
with col3:
    fertilizer = st.number_input("Fertilizer (kg/ha)", 0.0, 400.0, 120.0, 1.0)

region = st.selectbox("Region", [f"R{i}" for i in range(1, 13)], index=0)
crop = st.selectbox("Crop", ["Wheat", "Maize"], index=0)

# Build one-row DataFrame and apply the same preprocessing
row = pd.DataFrame({
    "Rainfall_mm": [rainfall],
    "Temperature_C": [temperature],
    "Fertilizer_kg": [fertilizer],
    "Region": [region],
    "Crop": [crop]
})
row = encode_categoricals(row)
row = add_gdd_feature(row, base_temp=10.0)

# Candidate feature sets commonly used in the notebooks
candidates = [
    ["Rainfall_mm", "Temperature_C", "Fertilizer_kg", "GDD_approx", "Region_cat", "Crop_cat"],  # GenAI full
    ["Rainfall_mm", "Temperature_C", "Fertilizer_kg", "Region_cat", "Crop_cat"],               # No-GenAI with cats
    ["Rainfall_mm", "Temperature_C", "Fertilizer_kg"],                                         # bare numeric
]

def pick_features(row_df, model, required):
    # 1) If we have an explicit training feature list — use it exactly
    if required is not None:
        missing = [f for f in required if f not in row_df.columns]
        if missing:
            raise RuntimeError(
                "Your model was trained with features not present in the app input: "
                + ", ".join(missing)
            )
        return required

    # 2) Otherwise, try to match model.n_features_in_ with our candidates
    n_req = getattr(model, "n_features_in_", None)
    if n_req is not None:
        for feats in candidates:
            if all(f in row_df.columns for f in feats) and len(feats) == int(n_req):
                return feats

    # 3) As a last resort, use any candidate that exists fully
    for feats in candidates:
        if all(f in row_df.columns for f in feats):
            return feats

    raise RuntimeError("Could not determine feature set for this model. Retrain and save with features.json.")

try:
    feats = pick_features(row, model, required_features)
    pred = float(model.predict(row[feats].values)[0])
    st.success(f"Predicted Yield: {pred:,.0f} kg/ha")
    st.caption(f"Features used: {feats}")
except Exception as e:
    st.error(f"Feature mismatch with the loaded model: {e}")
    st.info(
        "Fix: Re-run your training notebook and ensure it saves the features JSON alongside the model. "
        "This app will automatically use it on next run."
    )
