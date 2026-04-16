# Fraud Detection System

A machine learning web application that detects fraudulent financial transactions in real time using an **XGBoost classifier** trained on highly imbalanced payment data.

---
## 🚀 Live Demo

You can try the live application here:  
[https://your-streamlit-app-link](https://fraudd-detection.streamlit.app/)

Or run locally:


streamlit run app.py
----

## 📌 Project Overview

Financial fraud is a critical challenge in digital payment systems. This project builds an end-to-end fraud detection pipeline that:

- Trains and compares **3 ML models** (XGBoost, Random Forest, Logistic Regression)
- Handles **severe class imbalance** (~0.13% fraud rate) using `scale_pos_weight` and threshold tuning
- Selects the **best model** (XGBoost) based on PR-AUC
- Deploys an interactive **Streamlit web app** for real-time transaction risk scoring

---

## 🗂️ Project Structure

```
FRAUD_DETECTION/
│
├── app.py                                        # Streamlit web application
├── best_fraud_model.pkl                          # Saved XGBoost model + metadata
├── fraud-detection-using-machine-learning.ipynb  # Full ML pipeline notebook
└── Fraud.csv                                     # Raw dataset (not tracked in Git — see below)
```

---

## ML Pipeline (Notebook)

### 1. Exploratory Data Analysis
- Transaction amount distribution (log scale)
- Fraud rate by transaction type → only `CASH_OUT` and `TRANSFER` contain fraud
- Fraud rate by hour of day (time-based patterns)
- Correlation heatmap of balance features

### 2. Feature Engineering

| Feature | Description |
|---|---|
| `log_amount` | log₁p transform to reduce right-skew |
| `is_high_amount` | Binary flag for top 1% transaction amounts (p99) |
| `hour` | Hour of day derived from `step` (step % 24) |
| `is_night` | Binary flag for hours 22–23 and 0–5 |
| `balance_diff_orig` | Net change in sender balance (oldbalanceOrg − newbalanceOrig) |
| `balance_diff_dest` | Net change in receiver balance (newbalanceDest − oldbalanceDest) |
| `type_enc` | Label-encoded transaction type |

### 3. Models Trained

| Model | Strategy |
|---|---|
| Logistic Regression | `class_weight='balanced'`, C=0.3 |
| Random Forest | `class_weight='balanced'`, deep regularization |
| **XGBoost (Best)** | `scale_pos_weight`, tuned depth & learning rate |

### 4. Model Selection & Threshold Tuning
- Best model selected by **PR-AUC** (better metric for imbalanced data than accuracy)
- Custom decision threshold: chosen from the Precision–Recall curve to ensure **Fraud Recall ≥ 90%**

### 5. Final Results (XGBoost — tuned)

| Metric | Score |
|---|---|
| ROC-AUC | **0.9823** |
| PR-AUC | **0.8512** |
| Fraud Recall | ≥ 90% |



##  Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | ML models, preprocessing, metrics |
| XGBoost | Best performing classifier |
| Joblib | Model serialization |
| Streamlit | Web app deployment |
| Matplotlib / Seaborn | Visualization |

## 📦 Dataset

The dataset used is the **PaySim Synthetic Financial Dataset** available on Kaggle:

🔗 [Fraud Dataset — Kaggle](https://www.kaggle.com/datasets/athangpatil/fraud-dataset)

> **Note:** `Fraud.csv` is not tracked in this repository due to its large size.  
> Download it from the link above and place it in the project root before running the notebook.

Add this to `.gitignore`:
```
Fraud.csv
```

---
