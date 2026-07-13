# Wine Quality Prediction

A machine learning project that predicts wine quality based on physicochemical properties. Tracked with MLflow, served via a FastAPI backend with a SQLite database, and accessed via a Streamlit dashboard.

## System Architecture

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

## Features
1. **Data Engineering**: Data cleaning, median imputation, outlier clipping, scaling, and feature engineering.
2. **Machine Learning & Tuning**: Optuna hyperparameter optimization for a Random Forest model.
3. **MLOps Tracking**: MLflow tracking metrics (MAE, RMSE, R²), parameters, and model artifacts.
4. **API**: FastAPI service utilizing JWT auth and password hashing. Endpoints include `/predict`, `/predict_batch`, `/predictions_history`, `/model_info`, and `/metrics`.
5. **Frontend**: Streamlit dashboard with chemical input forms, batch CSV predictions, and database history log views.
6. **Logging & Monitoring**: Request processing latency middleware. Statistical feature drift detection using Kolmogorov-Smirnov tests.
7. **Containerization**: Multi-container setup using Docker and Docker Compose.
8. **CI/CD**: GitHub Actions checking code formatting, running unit tests, and validating Docker builds on merges.

---

## Getting Started

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional)

### Local Setup
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd "Wine Predictions Project"
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Initialize Environment Variables:
   ```bash
   cp .env.example .env
   ```

---

## Execution & Testing

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
1. Start the API Server:
   ```bash
   uvicorn api.main:app --reload
   ```
2. Start the Streamlit Dashboard:
   ```bash
   streamlit run app.py
   ```
   Open http://localhost:8501 in your browser.

---

## Running with Docker Compose

Run the stack with a single command:
```bash
docker-compose up --build
```
- FastAPI API: http://localhost:8000
- Interactive Swagger Docs: http://localhost:8000/docs
- Streamlit Frontend: http://localhost:8501

---

## Data Drift Detection
To check if live inference data matches the baseline training distribution, execute:
```bash
python -m src.drift_detection
```
This runs Kolmogorov-Smirnov tests on the SQLite logs and training reference values, printing outputs and saving details to `docs/drift_report.json`.
