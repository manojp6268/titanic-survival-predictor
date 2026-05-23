# 🚢 Titanic Survival Predictor

![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0.3-000000?style=flat-square&logo=flask)
![Model](https://img.shields.io/badge/Model-Logistic_Regression-steelblue?style=flat-square)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.865-orange?style=flat-square)

> *Predict whether a Titanic passenger would have survived - powered by a machine learning model trained on historical data.*

---

## 🔗 Live Demo
👉 **xxxx(#)** #TBD

---

## What this is

A fully deployed end-to-end machine learning application. Enter passenger
details - class, age, sex, fare, family size - and the model predicts
survival probability with a confidence score.

Built as the deployment stage of a complete ML pipeline covering EDA,
feature engineering, model comparison, hyperparameter tuning, and
SHAP explainability.

---

## How it works
User inputs passenger details via web form
│
▼
Flask receives and preprocesses inputs
│
▼
StandardScaler transforms continuous features
│
▼
Logistic Regression predicts survival probability
│
▼
Result displayed with probability bar

---

## Model details

| Metric | Score |
|---|---|
| Accuracy | 82.12% |
| F1 Score | 0.7612 |
| ROC-AUC | **0.8651** |

**Why Logistic Regression?**
After comparing four models (LR, Random Forest, XGBoost, SVM),
Logistic Regression achieved the highest ROC-AUC - the correct
primary metric given the 62/38 class imbalance in the dataset.
Feature engineering did the heavy lifting, not model complexity.

---

## Features engineered

| Feature | Description |
|---|---|
| `Title` | Extracted from passenger name — proxy for age, sex, social status |
| `FamilySize` | SibSp + Parch + 1 - captures travelling group context |
| `IsChild` | Binary flag for passengers under 12 |
| `Fare_log` | Log transformation of Fare - reduces right skewness |
| `HasCabin` | Binary flag for whether cabin was recorded |

---

## Tech stack

`Python` `Flask` `Scikit-learn` `Pandas` `NumPy` `SHAP` `Gunicorn`

---

## Project structure

titanic-survival-predictor/
├── app.py                 <- Flask application
├── requirements.txt       <- Dependencies
├── Procfile               <- Render deployment config
├── models/
│   ├── titanic_model.pkl  <- Trained Logistic Regression
│   ├── scaler.pkl         <- Fitted StandardScaler
│   └── feature_names.pkl <- Ordered feature list
└── templates/
├── index.html         <- Input form
└── result.html        <- Prediction result page

---

## Full ML notebook

The complete pipeline - EDA, feature engineering, four model comparison,
hyperparameter tuning, and SHAP explainability:

👉 [titanic-ml-pipeline](https://github.com/manojp6268/titanic-ml-pipeline)

---

## Run locally

```bash
git clone https://github.com/manojp6268/titanic-survival-predictor.git
cd titanic-survival-predictor
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`

---

## Author

**Manoj Prakash** - Data Scientist & AI/ML Engineer
M.Sc. Data Science @ Universität Trier · ex-Oracle Cerner · ex-Huawei

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/manoj-p-a95b7b1a2)
[![GitHub](https://img.shields.io/badge/GitHub-manojp6268-181717?style=flat-square&logo=github)](https://github.com/manojp6268)
