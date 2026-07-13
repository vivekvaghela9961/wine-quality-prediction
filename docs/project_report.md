# Wine Quality Prediction — Final Project Report

## Executive Summary
This report summarizes the design, training, and deployment of a production-grade machine learning model to predict wine quality scores based on chemical properties. We compiled red and white wine records, optimized a Random Forest Regressor model using Optuna, logged tracking facts with MLflow, and deployed the model via a FastAPI backend and a Streamlit dashboard.

---

## 1. Exploratory Data Analysis (EDA) & Data Cleaning
- **Raw Data**: 6,497 samples combining red (1,599) and white (4,898) wine properties from the UCI Machine Learning Repository.
- **Handling Duplicates**: Found and dropped **1,177 duplicated records** to prevent data leakage during train/test splits.
- **Outlier Capping**: Handled outliers in properties like residual sugar and free sulfur dioxide using IQR-based bounds clipping (1.5x IQR).
- **Target Variable Distribution**: Quality scores follow a bell curve concentrated between 5 and 7.

---

## 2. Feature Engineering & Scaling
To capture physical attributes, three domain-specific features were engineered:
1. **Total Acidity**: `fixed acidity` + `volatile acidity` + `citric acid`
2. **Acid-to-Sugar Ratio**: `total acidity` / (`residual sugar` + 1e-5)
3. **Bound Sulfur Dioxide**: `total sulfur dioxide` - `free sulfur dioxide`
4. **Categorical Encoding**: Wine type was converted to a binary column `is_red` (Red = 1, White = 0).
5. **Feature Scaling**: Scaled all numerical features using `StandardScaler` fitted solely on the training split.

---

## 3. Modeling & Hyperparameter Optimization
We compared four architectures using a 3-fold cross validation grid:
1. **Linear Regression** (Baseline RMSE: 0.7171)
2. **XGBoost** (Baseline RMSE: 0.6952)
3. **LightGBM** (Baseline RMSE: 0.6742)
4. **Random Forest** (Baseline RMSE: 0.6643)

### Random Forest Tuning (Optuna)
Using Optuna optimization over 10 trials, the hyperparameter space was tuned:
- `n_estimators`: 148
- `max_depth`: 15
- `min_samples_split`: 4
- `min_samples_leaf`: 4
- **Final Evaluation (Holdout Test)**: MAE of **0.5142** and RMSE of **0.6687**.

---

## 4. System Architecture
The system consists of the following core layers:
- **Database Layer**: SQLite database logging incoming client requests and predictions.
- **Security & Auth Layer**: JWT bearer token authentication using `PyJWT` and PBKDF2 password hashing.
- **API Backend**: FastAPI service implementing `/predict`, `/predict_batch`, `/model_info`, `/metrics`, and `/predictions_history`.
- **Frontend Dashboard**: Streamlit interface containing graphical sliders, file upload for batch runs, historical tables, and feature importance charts.
- **Containerization**: Orchestrated using Docker and Docker Compose.

```
[Browser] ---> [Streamlit App (8501)] ---> [FastAPI API (8000)] ---> [SQLite Database]
                                                 |
                                                 v
                                           [Saved Model / Scaler]
```

---

## 5. Model Monitoring & Maintenance
- **Latency Monitoring**: Middleware registers endpoint roundtrip timings to console and `logs/api.log`.
- **Drift Detection**: Run `python -m src.drift_detection` to compare baseline feature distributions against live logs using Kolmogorov-Smirnov tests.
