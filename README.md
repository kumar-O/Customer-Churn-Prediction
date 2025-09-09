# Customer Churn Prediction
Customer churn prediction with Python using synthetic datasets. Includes data generation, feature engineering, and training with Logistic Regression, Random Forest, and Gradient Boosting. Improved pipeline applies hyperparameter tuning and threshold optimization to boost recall. Outputs metrics, reports, and charts.

Predict customer churn on a synthetic dataset using Python. Includes data generation, feature engineering, model training (Logistic Regression, Random Forest, Gradient Boosting), and **improved training with hyperparameter search + threshold tuning** for better recall on churners. Outputs metrics, reports, and visualizations.

---

## Features
- Synthetic customer dataset with realistic behavior signals
- Models: Logistic Regression, Random Forest, Gradient Boosting
- hyperparameter optimization (RandomizedSearchCV), class weighting, PR-AUC–based model selection, and F2-based threshold tuning
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Visuals: ROC curves, Precision-Recall curves, confusion matrix, feature importance
- Saved artifacts: best model (`joblib`) & metrics

---

## Project Structure
```
customer-churn-prediction/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ .gitignore
├─ data/
│  └─ generate_customers.py
├─ src/
│  ├─ train_models.py            
│  └─ utils.py
└─ outputs/
   └─ figures & reports
```

---

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Generate Synthetic Data
```bash
python data/generate_customers.py --n 10000 --seed 42 --out data/customers.csv
```

---

## Train & Evaluate

### Standard Training
```bash
python src/train_models.py --input data/customers.csv --outdir outputs --test-size 0.2 --seed 42
```

### Improved Training
```bash
python src/train_models.py --input data/customers.csv --outdir outputs --test-size 0.2 --val-size 0.2 --seed 42
```

**Outputs**
- `metrics_improved.json` – model selection, tuned threshold, test metrics
- `classification_report_improved.txt`
- `roc_curve_improved.png`
- `pr_curve_improved.png`
- `confusion_matrix_improved.png`
- `feature_importance.png`
- `best_model_improved.joblib`

---

## Results

### Key Metrics (Improved Model: Logistic Regression)
| Metric        | Value |
|---------------|-------|
| Accuracy      | **83.9%** |
| ROC-AUC       | **0.823** |
| PR-AUC (AP)   | **0.562** |
| Recall (Churn)| **0.501** (↑ from 0.316) |
| Precision (Churn) | 0.526 |

➡️ Recall for churners improved by ~60%, making the model much better at catching at-risk customers.

---

### Confusion Matrix
<img width="1120" height="1120" alt="confusion_matrix_improved" src="https://github.com/user-attachments/assets/cc1dc38f-0c83-4b48-89f1-46ec57891168" />

### ROC Curve
<img width="1280" height="960" alt="roc_curve_improved" src="https://github.com/user-attachments/assets/8b35aa42-d40f-4ca8-bbab-7722a6ed2756" />

### Precision-Recall Curve
<img width="1280" height="960" alt="pr_curve_improved" src="https://github.com/user-attachments/assets/d4d19655-140a-47ce-a42c-c4c328a8c6fd" />

---

## 📑 Data Schema
| column                    | description                          |
|---------------------------|--------------------------------------|
| customer_id               | unique customer ID                   |
| age                       | customer age                         |
| region                    | {North, South, East, West}           |
| tenure_months             | months since signup                  |
| is_premium                | premium plan (0/1)                   |
| monthly_spend             | average monthly spend                |
| avg_txn_value             | average transaction value            |
| txns_last_30d             | transactions in last 30 days         |
| days_since_last_purchase  | recency (days)                       |
| customer_service_calls    | support calls in last 90 days        |
| discounts_used_90d        | discounts used in last 90 days       |
| complaints_90d            | complaint count                      |
| churn                     | target label (0/1)                   |
