# Wine Quality Prediction & Monitoring Dashboard

An end-to-end production-grade machine learning project that predicts wine quality based on physicochemical properties. Tracked with **MLflow**, served via a secure **FastAPI** backend with a **SQLite** audit database, and visualised with a modern **Streamlit** dashboard.

## 📊 System Architecture

```
                       +----------------------------+
                       |     Streamlit Frontend     |
                       |        (Port 8501)         |
                       +--------------+-------------+
                                      |
                                      | HTTP Requests (Bearer Token)
                                      v
                       +----------------------------+
                       |      FastAPI Backend       |
                       |        (Port 8000)         |
                       +-------+------------+-------+
                               |            |
                      ORM Logs |            | Load Pickle
                               v            v
               +---------------+--+      +--+------------------+
               |  SQLite Database |      | Tuned Random Forest |
               |   (db.sqlite3)   |      |   Model & Scaler    |
               +------------------+      +---------------------+
```

## ✨ Features
1. **Data Engineering**: Data cleaning, median imputation, outlier IQR-based clipping, scaling, and feature engineering (total acidity, acid-to-sugar ratio, bound SO2).
2. **Machine Learning & Tuning**: Optuna hyperparameter optimization tuning Random Forest, comparing XGBoost, LightGBM, and Linear Regression.
3. **MLOps Tracking**: MLflow tracking metrics (MAE, RMSE, R²), parameter parameters, and model artifacts.
4. **Secure API**: FastAPI service utilizing JWT auth and password hashing. Ends include `/predict` (single), `/predict_batch` (CSV streams), `/predictions_history`, `/model_info`, and `/metrics`.
5. **Interactive Frontend**: Streamlit dashboard with chemical input forms, batch CSV predictions, database history log views, and global feature importance charts.
6. **Logging & Monitoring**: Request processing latency middleware logs requests to rotating console/file handlers (`logs/api.log`). Statistical feature drift detection using Kolmogorov-Smirnov tests.
7. **Containerization**: Full multi-container setups using Docker and Docker Compose.
8. **CI/CD**: GitHub Actions checking code formatting (Black, Flake8), running unit tests, and validating Docker builds on merges.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional)

### 2. Local Setup
1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd "Wine Predictions Project"
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Environment Variables**:
   ```bash
   cp .env.example .env
   ```

---

## 🛠️ Execution & Testing

### Run Tests and Coverage Reports
Run the test suite with PowerShell helper or pytest directly:
```powershell
./run_tests.ps1
```
Or:
```bash
pytest --cov=api --cov=src --cov-report=term-missing tests/
```

### Start API and Streamlit Dashboard Locally
1. **Start the API Server**:
   ```bash
   uvicorn api.main:app --reload
   ```
2. **Start the Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🐳 Running with Docker Compose

Orchestrate the entire stack with a single command:
```bash
docker-compose up --build
```
- **FastAPI API**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Frontend**: [http://localhost:8501](http://localhost:8501)

---

## 🔍 Data Drift Detection
To check if live inference data matches the baseline training distribution, execute:
```bash
python -m src.drift_detection
```
This runs Kolmogorov-Smirnov tests on your SQLite logs and training reference values, printing outputs and saving details to `docs/drift_report.json`.
