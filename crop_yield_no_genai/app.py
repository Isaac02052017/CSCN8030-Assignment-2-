

# Minimal Streamlit demo for the No-GenAI prototype
import streamlit as st
import pandas as pd
from pathlib import Path
from src.model import load_model
from src.data_preprocessing import encode_categoricals

# ---------- UI setup ----------
st.set_page_config(page_title="Crop Yield Predictor (No-GenAI)", page_icon="🌾")
st.title("🌾 Crop Yield Predictor — No-GenAI Baseline")
st.write("Load a saved model from `outputs/models/` (run the training notebook first).")

# ---------- Robust model discovery & selection ----------
# Resolve models directory reliably from this file's location:
models_dir = (Path(__file__).parent / "outputs" / "models").resolve()

if not models_dir.exists():
    st.warning(f"`{models_dir}` does not exist. Train and save a model first.")
    st.stop()

# Find all .pkl models and pick the newest by default
candidates = sorted(models_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)

if not candidates:
    st.warning(f"No `.pkl` models found in `{models_dir}`. "
               "Run the notebook to train and save one (e.g., baseline_rf.pkl).")
    st.stop()

# Map option labels to full paths
options = {p.name: p for p in candidates}
default_index = 0  # newest first due to sorting above

chosen_label = st.selectbox("Choose model", list(options.keys()), index=default_index)
chosen_path = options[chosen_label]

st.caption(f"Using model: `{chosen_path}`")
model = load_model(str(chosen_path))  # load_model expects a path-like string

# ---------- Inputs ----------
col1, col2, col3 = st.columns(3)
with col1:
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=3000.0, value=500.0, step=10.0)
with col2:
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=45.0, value=18.0, step=0.1)
with col3:
    fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=0.0, max_value=400.0, value=120.0, step=1.0)

region = st.selectbox("Region", [f"R{i}" for i in range(1, 13)], index=0)
crop = st.selectbox("Crop", ["Wheat", "Maize"], index=0)

# ---------- Row -> features ----------
row = pd.DataFrame({
    "Rainfall_mm": [rainfall],
    "Temperature_C": [temperature],
    "Fertilizer_kg": [fertilizer],
    "Region": [region],
    "Crop": [crop],
})

# Encode categoricals if your preprocessing provides these helpers
row = encode_categoricals(row)

# Try multiple feature sets to be tolerant to model training variations
feature_candidates = [
    ["Rainfall_mm", "Temperature_C", "Fertilizer_kg", "Region_cat", "Crop_cat"],
    ["Rainfall_mm", "Temperature_C", "Fertilizer_kg"],
]

pred = None
used = None
for feats in feature_candidates:
    if all(f in row.columns for f in feats):
        try:
            pred = float(model.predict(row[feats].values)[0])
            used = feats
            break
        except Exception:
            continue

# ---------- Output ----------
if pred is None:
    st.error("Feature mismatch with the loaded model. Retrain and save a compatible model in `outputs/models/`.")
else:
    st.success(f"Predicted Yield: {pred:,.0f} kg/ha")
    st.caption(f"Features used: {used}")
