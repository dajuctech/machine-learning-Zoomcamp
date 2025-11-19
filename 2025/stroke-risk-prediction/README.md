# 🧠 Stroke Risk Prediction (ML Zoomcamp Midterm)

End-to-end stroke risk classifier following the ML Zoomcamp midterm template: notebooks for exploration, reproducible training scripts, saved models, and a small serving stack (Flask API + Streamlit UI).

---

## Project Highlights

- **Objective:** classify whether a patient will suffer a stroke using demographic and clinical indicators.
- **Dataset:** [Healthcare Dataset Stroke Data](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (5,110 rows, 12 columns) stored in `data/raw/healthcare-dataset-stroke-data.csv`.
- **Target:** `stroke` ∈ {0,1} with 4.87 % positives (see `reports/eda_summary.txt`).
- **Primary metric:** ROC AUC (less sensitive to imbalance than accuracy).
- **Stack:** pandas, scikit-learn (Gradient Boosting + preprocessing pipeline), Flask, Streamlit.
- **Deliverables:** notebooks (`notebooks/`), training & inference scripts (`src/`), serialized models (`models/`), API/UI (`app/`), data splits (`data/processed/`).

---

## Repository Map

| Path | Description |
| --- | --- |
| `app/main.py` | Flask REST API exposing `/predict`. |
| `app/streamlit_app.py` | Streamlit front-end that calls the API. |
| `data/raw/` | Kaggle CSV plus other raw datasets used elsewhere. |
| `data/processed/` | Stratified `train.csv`, `val.csv`, `test.csv`. |
| `models/` | Pickled models (`model.bin` is the latest GB pipeline) + `model_comparison.csv`. |
| `notebooks/01_eda.ipynb` → `05_trees_and_tuning.ipynb` | Step-by-step EDA, baselines, tuning. |
| `reports/eda_summary.txt` | Quick stats extracted from notebooks. |
| `scripts/download_data.py` | Example script for downloading raw data. |
| `src/train.py` | Main training pipeline (loads raw CSV, preprocesses, trains Gradient Boosting, saves `models/model.bin`). |
| `src/predict.py` | Reusable `predict_single` helper for batch/CLI inference. |
| `requirements.txt` | Minimal deps (pandas, numpy, scikit-learn, fastapi, uvicorn, joblib, matplotlib, seaborn). |

---

## Data & EDA

- **Records:** 5,110, **stroke cases:** 249 (4.87 %).  
- **Missingness:** `bmi` has 201 nulls, imputed with the training median (`28.1`).  
- **Risk signals:** Hypertension (13 % stroke rate) and heart disease (17 %) heavily correlate with the target; smokers show slightly higher incidence.  
- **Splits:** `train/val/test = 60/20/20` using stratified `train_test_split`, stored in `data/processed/`.  
- Full exploration in `notebooks/01_eda.ipynb`; metrics summarized in `reports/eda_summary.txt`.

---

## Modeling Workflow

1. **EDA & Baseline** (`notebooks/01_eda.ipynb`, `02_baseline_logreg.ipynb`) – logistic regression baseline with ROC curves.
2. **Model Comparison** (`03_models_classification.ipynb`, `05_trees_and_tuning.ipynb`) – Decision Tree, Random Forest, Gradient Boosting; metrics exported to `models/model_comparison.csv`.
3. **Production Training** (`src/train.py`)  
   - Drops `id`, imputes BMI with train median.  
   - Preprocessing via `ColumnTransformer`: `StandardScaler` on numeric (`age`, `avg_glucose_level`, `bmi`) + `OneHotEncoder(handle_unknown='ignore')` on categoricals (`gender`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`).  
   - Classifier: `GradientBoostingClassifier` with the tuned params in the script (`n_estimators=200`, `learning_rate=0.1`, `max_depth=5`, `min_samples_split=50`, `min_samples_leaf=20`).  
   - Saves the full pipeline to `models/model.bin`.  
4. **Inference** (`src/predict.py`) – loads `model.bin`, exposes `predict_single(patient_dict)` and a CLI demo.

---

## Metrics

From `models/model_comparison.csv`:

| Model | Accuracy | ROC AUC |
| --- | --- | --- |
| Logistic Regression | 0.756 | 0.838 |
| Random Forest | 0.734 | 0.837 |
| Decision Tree | 0.694 | 0.805 |

Final Gradient Boosting run (`src/train.py`):

- **Validation:** Accuracy 0.946, ROC AUC 0.821.  
- **Test:** Accuracy 0.941, ROC AUC 0.796, Precision/Recall/F1 for the positive class currently 0 because of the 0.5 threshold.  
- ✅ Next action: tune class weights or lower the decision threshold to recover recall without hurting ROC AUC.

---

## How to Reproduce

> All paths assume the repo root `home work/2025/stroke-risk-prediction`.

### 1. Create environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Serving stack
pip install flask streamlit requests
```

### 2. Prepare data

The Kaggle CSV is already committed (`data/raw/healthcare-dataset-stroke-data.csv`). If you rebuild from scratch, download it manually or adapt `scripts/download_data.py`. Splits are generated on the fly by `src/train.py` and cached in `data/processed/`.

### 3. Train model

```bash
cd src
python3 train.py
```

Run from `src/` so the relative data path (`../data/raw/...`) resolves correctly. Metrics print to stdout and the pipeline saves to `../models/model.bin`.

### 4. Batch inference

```bash
cd src
python3 predict.py
```

Edit the `test_patient_*` dictionaries or import `predict_single` in your own scripts.

---

## Serving the Model

### Flask API (`app/main.py`)

```bash
cd app
python3 main.py  # starts on http://0.0.0.0:9696
```

Sample request:

```bash
curl -X POST http://0.0.0.0:9696/predict \
     -H "Content-Type: application/json" \
     -d '{
           "gender":"Female",
           "age":67,
           "hypertension":1,
           "heart_disease":1,
           "ever_married":"Yes",
           "work_type":"Private",
           "Residence_type":"Urban",
           "avg_glucose_level":228.7,
           "bmi":36.6,
           "smoking_status":"formerly smoked"
         }'
```

Response:

```json
{
  "stroke_probability": 0.71,
  "stroke_prediction": 1,
  "risk_level": "VERY HIGH",
  "model": "Gradient Boosting",
  "patient_data": { ... }
}
```

### Streamlit Dashboard (`app/streamlit_app.py`)

```bash
streamlit run app/streamlit_app.py
```

- Two-column form that collects the ten model inputs.
- Sends them to the Flask API and displays probability, binary prediction, and qualitative risk level (LOW → VERY HIGH) with recommendations.
- Includes quick-test buttons (high- vs low-risk profiles) and an expandable log of the payload.

---

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `01_eda.ipynb` | Data overview, imbalance, feature exploration. |
| `02_baseline_logreg.ipynb` | Logistic regression baseline and ROC analysis. |
| `03_models_classification.ipynb` | Additional models + comparisons. |
| `04_evaluation.ipynb` | Hold-out diagnostics, confusion matrices. |
| `05_trees_and_tuning.ipynb` | Gradient Boosting tuning & feature importances. |

These notebooks document every step before the code was moved to `src/`.

---

## Next Steps

1. Tune probability thresholds / class weights to recover recall for the 5 % minority class.
2. Calibrate predicted probabilities (Platt or isotonic) for better clinical interpretation.
3. Automate evaluation and add regression/unit tests for the training + inference scripts.
4. Containerize Flask + Streamlit for deployment to Render/Fly.io/Cloud Run, or switch to FastAPI for async support.

---

Questions or suggestions? Open an issue or ping me! 💬
