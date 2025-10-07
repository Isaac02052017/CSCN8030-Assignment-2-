# Crop Yield Prediction — Prototype B (GenAI‑Assisted)

A **GenAI‑assisted** baseline for predicting crop yield (kg/ha). This prototype keeps the same supervised
learning pipeline as the traditional version but uses **generative AI** to accelerate:
- **Feature ideas** (adds `GDD_approx` — a simple growing‑degree proxy)
- **Hyperparameter search** (compact, small‑data friendly **RandomForest** grid via GridSearchCV)
- **EDA narration & docs** (short draft text captured in `outputs/eda_narrative_genai.md` and `prompts_log.md`)

No external GenAI APIs are called at runtime; GenAI contributions are embedded in the repository and curated by you.

---

## 📁 Folder Structure (expected)

```
crop_yield_genai/
├─ data/
│  └─ crop_yield_sample.csv            # your dataset
├─ notebooks/
│  ├─ 03_genai_assisted.ipynb          # main GenAI‑assisted build
│  └─ 04_eval_compare.ipynb            # optional: compare vs No‑GenAI models
├─ outputs/
│  ├─ figures/                         # auto‑saved plots
│  ├─ models/                          # auto‑saved best model (genai_rf.pkl)
│  └─ eda_narrative_genai.md           # draft EDA notes (editable)
├─ src/
│  ├─ data_preprocessing.py            # includes add_gdd_feature()
│  ├─ model.py                         # suggested_rf_grid() + GridSearchCV
│  └─ visualization.py
├─ app.py                               # optional Streamlit demo
├─ prompts_log.md                       # selected prompts for transparency
├─ README.md                            # this file
├─ requirements.txt
└─ .gitignore
```

> If folders are missing, create them. The notebooks will create `outputs/` automatically.

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
Place a CSV at: `data/crop_yield_sample.csv` with at least these columns:

| Column          | Type  | Description                              |
|-----------------|-------|------------------------------------------|
| Year            | int   | Crop season year                         |
| Region          | str   | Region identifier (e.g., R1, R2, …)      |
| Crop            | str   | Crop type (e.g., Wheat, Maize)           |
| Rainfall_mm     | float | Seasonal cumulative rainfall             |
| Temperature_C   | float | Seasonal mean temperature                |
| Fertilizer_kg   | float | Fertilizer rate (kg/ha)                  |
| Yield_kg_ha     | float | **Target**: observed yield (kg/ha)       |

> The GenAI‑assisted pipeline will also compute **`GDD_approx = max(0, Temperature_C − 10) × 120`**.

### 4) Train the GenAI‑assisted model
Open and **run all cells** in:
```
notebooks/03_genai_assisted.ipynb
```
This will:
- Load & clean data, **add `GDD_approx`**, and encode categoricals
- Perform a **time‑based split** (last 2 years used for test)
- Train: Linear Regression (reference) + **RandomForest with a compact RF grid** via GridSearchCV
- Save plots to `outputs/figures/`, a short **EDA narrative** to `outputs/eda_narrative_genai.md`,
  and the selected model to `outputs/models/genai_rf.pkl`

### 5) (Optional) Compare with your No‑GenAI prototype
After you have models from both prototypes saved in `outputs/models/`, open:
```
notebooks/04_eval_compare.ipynb
```
It will load any available models (baseline LR/RF, GenAI RF) and report **MAE / RMSE / R²** on the same test split.

### 6) (Optional) Demo app
```bash
streamlit run app.py
```
- Choose `genai_rf.pkl` (or other available models)
- Enter rainfall, temperature, fertilizer, region, crop → get predicted yield

---

## 🧠 What is “GenAI‑assisted” here?

- **Feature idea**: `GDD_approx` based on mean seasonal temperature (no external calls)
- **Hyperparameters**: a **small** grid tuned for ≤500 rows to speed up search
- **Narration**: draft EDA text and docstring polish (you review & edit)
- **Transparency**: `prompts_log.md` keeps a record of prompts used

**Guardrails**
- Verify all code & metrics manually
- Don’t evaluate on synthetic data
- Keep held‑out test data truly unseen

---

## 📊 Metrics & Model Selection

We report **MAE**, **RMSE**, and **R²** on the held‑out test set.  
The notebook selects the **lowest RMSE** model and saves it to `outputs/models/genai_rf.pkl`.

---

## 🧰 Troubleshooting

- **No model found in app** → Run the notebook first to save one.
- **Feature mismatch** → Ensure the app’s inputs match training features; retrain & save again if you changed features.
- **Slow grid search** → Reduce `n_estimators` options or folds in `GridSearchCV`.

---

## ✅ Deliverables Checklist

- [ ] Executed `03_genai_assisted.ipynb` with results
- [ ] `outputs/models/genai_rf.pkl` saved
- [ ] `outputs/figures/*` + `outputs/eda_narrative_genai.md`
- [ ] `prompts_log.md` updated
- [ ] (Optional) `04_eval_compare.ipynb` run & screenshot
- [ ] (Optional) Streamlit demo screenshot

---

## 📎 Notes & Next Steps

- Consider adding soil/NDVI features and trying gradient boosting (XGBoost/LightGBM).
- Add model explainability (e.g., SHAP) and partial dependence plots.
- Track **time saved** and **iteration count** vs. the No‑GenAI prototype for your report.
