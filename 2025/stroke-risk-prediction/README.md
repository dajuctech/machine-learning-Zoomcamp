# 🧠 Stroke Risk Prediction – ML Zoomcamp Midterm

Predicting the probability that a patient will suffer a stroke by combining demographic variables with basic clinical measurements. The project follows the ML Zoomcamp midterm guidelines: start with a notebook-driven exploration, move to reproducible scripts, and expose the final model behind an API + lightweight UI.

---

## 🎯 Project Overview

- **Problem**: Binary classification (`stroke` ∈ {0,1}) for early risk stratification using routinely available features.
- **Dataset**: [Kaggle – Healthcare Dataset Stroke Data](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (5,110 encounters, 249 positives).
- **Primary metric**: ROC AUC (robust to class imbalance, focuses on ranking ability).
- **Stack**: Pandas, scikit-learn (Gradient Boosting), Flask API, Streamlit UI.
- **Deliverables**: Reusable training/inference script (`src/`), saved model (`models/model.bin`), REST API (`app/main.py`), interactive dashboard (`app/streamlit_app.py`), notebooks for transparency (`notebooks/`).

---

## 📊 Dataset

| Feature | Type | Notes |
| --- | --- | --- |
| `id` | integer | Unique encounter ID (dropped before modeling). |
| `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status` | categorical | One-hot encoded, `smoking_status` contains `"Unknown"`. |
| `age`, `avg_glucose_level`, `bmi` | numeric | `bmi` has 201 missing values (3.9%). |
| `hypertension`, `heart_disease` | binary | Indicators already encoded as {0,1}. |
| `stroke` | binary target | 1 = stroke event, 4.9 % positives. |

Additional facts (see `reports/eda_summary.txt`):

- Total rows: **5,110**, columns: **12**.
- Class imbalance: **19.5 : 1** (negative:positive).
- BMI median: **28.1**, 201 missing entries filled with the training median.
- Stroke rate jumps to 13 % with hypertension and 17 % with heart disease.
- Smokers and self-employed patients show a slightly higher risk signal.

Raw CSV lives in `data/raw/healthcare-dataset-stroke-data.csv`; stratified train/val/test splits are cached in `data/processed/`.

---

## 📁 Repository Layout

```
home work/2025/stroke-risk-prediction
├── app/                # Flask API + Streamlit dashboard
├── data/               # Raw CSV + stratified splits
├── models/             # Serialized models & comparison table
├── notebooks/          # 01_eda … 05_trees_and_tuning
├── reports/            # Text summaries (EDA, metrics)
├── scripts/            # Utility scripts
├── src/                # Training + batch prediction code
└── requirements.txt
```

---

## 🔄 Modeling Pipeline

1. **Split before touching the data** – 60 / 20 / 20 (train/val/test) with stratification via `train_test_split`.
2. **Preprocessing** – drop `id`, median-impute BMI, scale numeric columns (`age`, `avg_glucose_level`, `bmi`), one-hot encode categoricals (`OneHotEncoder(handle_unknown='ignore')`).
3. **Baseline** – Logistic Regression in `notebooks/02_baseline_logreg.ipynb` to sanity-check features and the metric pipeline.
4. **Model search** – Evaluate Decision Tree, Random Forest, Gradient Boosting in `notebooks/03_*` and `05_trees_and_tuning.ipynb`, log comparisons to `models/model_comparison.csv`.
5. **Production script** – `src/train.py` rebuilds the best preprocessing + Gradient Boosting pipeline, fits it, prints metrics, and saves `models/model.bin`.
6. **Batch inference** – `src/predict.py` exposes a reusable `predict_single` helper (used by API/UI).

---

## 📈 Model Performance (test split)

| Model | Accuracy | ROC AUC | Notes |
| --- | --- | --- | --- |
| Logistic Regression | 0.756 | 0.838 | Balanced baseline, best ranking ability so far. |
| Random Forest | 0.734 | 0.837 | 300-tree RF, slightly lower accuracy but comparable ROC AUC. |
| Decision Tree | 0.694 | 0.805 | Overfits quickly, serves as reference. |
| Gradient Boosting (tuned) | 0.941 | 0.796 | `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`, accuracy inflated by class imbalance – needs threshold tuning to recover recall. |

Key insight: ROC AUC remains the most honest metric under the 5 % positive rate. Gradient Boosting brings tighter calibration but, with the default 0.5 cutoff, misses rare strokes; the next iteration should focus on probability calibration and cost-sensitive thresholding.

---

## 🛠️ Reproducing the Project

### 1. Environment

```bash
cd "home work/2025/stroke-risk-prediction"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# API/UI helpers (if not already installed)
pip install flask streamlit requests
```

### 2. Data

1. Download the Kaggle CSV and place it at `data/raw/healthcare-dataset-stroke-data.csv` (already present in this repo snapshot).
2. Optional: regenerate stratified splits by rerunning `src/train.py` – the script recreates temporary dataframes each run, so no extra preprocessing file is required.

### 3. Train the model

```bash
cd src
python3 train.py
```

Outputs include validation/test metrics and writes `../models/model.bin`. Run the script from within `src/` so the relative data path resolves correctly.

### 4. Batch inference from the CLI

```bash
cd src
python3 predict.py
```

The script loads the saved pipeline once and prints probabilities for two sample patients. You can also import `predict_single` in another module to score custom dictionaries.

---

## 🌐 Serving the Model

### Flask REST API

```bash
cd app
python3 main.py
```

Sample request:

```bash
curl -X POST http://0.0.0.0:9696/predict \
     -H "Content-Type: application/json" \
     -d '{
           "gender":"Male",
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

Response fields:

```json
{
  "stroke_probability": 0.71,
  "stroke_prediction": 1,
  "risk_level": "VERY HIGH",
  "model": "Gradient Boosting",
  "patient_data": { ... }
}
```

### Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

- Runs a two-column form with the ten model features.
- Talks to the Flask API (`http://localhost:9696/predict`), so keep the API server running in another terminal.
- Shows probability, binary flag, and a qualitative risk tier (LOW → VERY HIGH), plus recommendations.

---

## 📓 Notebooks

| Notebook | Purpose |
| --- | --- |
| `01_eda.ipynb` | Initial exploration, class imbalance checks, feature distributions. |
| `02_baseline_logreg.ipynb` | Baseline logistic regression, ROC curves, metric discussion. |
| `03_models_classification.ipynb` | Additional classifiers + cross-validation experiments. |
| `04_evaluation.ipynb` | Hold-out test analysis, confusion matrices. |
| `05_trees_and_tuning.ipynb` | Gradient Boosting hyperparameter search and feature importance. |

Notebooks are deliberately verbose to document the reasoning behind every modeling decision before code moved into `src/`.

---



Feel free to open an issue or reach out if you want to extend the project or discuss modeling trade-offs! 💬
