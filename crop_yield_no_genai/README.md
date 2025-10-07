# Crop Yield Prediction — Prototype A (No‑GenAI)

A **ready‑to‑run traditional ML baseline** for predicting crop yield (kg/ha) using seasonal features.
This prototype avoids generative AI and focuses on a clean, reproducible pipeline with a
time‑based train/test split, baseline models (Linear Regression, Random Forest), and clear metrics.

---

## 📁 Folder Structure (expected)

```
crop_yield_no_genai/
├─ data/
│  └─ crop_yield_sample.csv        # your dataset (see schema below)
├─ notebooks/
│  └─ 02_baseline_no_genai.ipynb   # run this to train & evaluate
├─ outputs/
│  ├─ figures/                     # auto‑saved plots
│  └─ models/                      # auto‑saved best model (.pkl)
├─ src/
│  ├─ data_preprocessing.py
│  ├─ model.py
│  └─ visualization.py
├─ app.py                          # optional Streamlit demo
├─ README.md                       # this file
├─ requirements.txt
└─ .gitignore
```

> If you don’t have all the folders yet, create them. The notebook will create `outputs/` automatically.

---

## 🚀 Quick Start

### 1) Create & activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Add your data
Place a CSV at: `data/crop_yield_sample.csv` with the following minimum columns:

| Column          | Type  | Description                              |
|-----------------|-------|------------------------------------------|
| Year            | int   | Crop season year                         |
| Region          | str   | Region identifier (e.g., R1, R2, …)      |
| Crop            | str   | Crop type (e.g., Wheat, Maize)           |
| Rainfall_mm     | float | Seasonal cumulative rainfall             |
| Temperature_C   | float | Seasonal mean temperature                |
| Fertilizer_kg   | float | Fertilizer rate (kg/ha)                  |
| Yield_kg_ha     | float | **Target**: observed yield (kg/ha)       |

> If your column names differ, update `src/data_preprocessing.py::select_features()`.

### 4) Train & evaluate (Notebook)
Open and **run all cells** in:
```
notebooks/02_baseline_no_genai.ipynb
```
This will:
- Load & clean data; encode `Region`/`Crop` categories
- Perform a **time‑based split** (last 2 years = test)
- Train: Dummy (mean), Linear Regression, Random Forest
- Save plots to `outputs/figures/` and **best model** to `outputs/models/`

### 5) (Optional) Demo app
```bash
streamlit run app.py
```
- Select the saved model (`baseline_rf.pkl` or `baseline_lr.pkl`)
- Provide inputs (rainfall, temperature, fertilizer, region, crop) → predicted yield

---

## 📊 Metrics & Model Selection

We report **MAE**, **RMSE**, and **R²** on the held‑out test set.  
The notebook picks the **lowest RMSE** model and saves it to `outputs/models/`.

| Model            | What it does                            |
|------------------|-----------------------------------------|
| Naïve (mean)     | Predicts the mean of `y_train`          |
| Linear Regression| Linear baseline                         |
| Random Forest    | Handles nonlinearity & interactions     |

---

## 🧰 Troubleshooting

- **No model found in app** → Run the notebook first to save one.
- **Feature mismatch** → Ensure the app’s input features match those used to train the saved model. If you changed features, retrain and save again.
- **Data types** → Make sure numeric columns are numbers (no stray text).

---

## ✅ Deliverables Checklist

- [ ] Executed notebook with metrics table & plots
- [ ] Saved best model in `outputs/models/`
- [ ] `README.md`, `requirements.txt`, `.gitignore`
- [ ] (Optional) Streamlit demo screenshot or short video

---

## 📎 Notes

- Keep random seeds fixed for reproducibility.
- This prototype intentionally avoids generative AI.
- Next steps: add agronomic features (e.g., GDD, NDVI), try gradient boosting (XGBoost/LightGBM), and add explainability (e.g., SHAP).
