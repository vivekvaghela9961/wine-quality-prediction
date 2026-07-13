# Wine Quality Prediction

An end-to-end machine learning project to predict wine quality based on physicochemical properties using machine learning models, tracked with MLflow, served via FastAPI, and containerized with Docker.

## Project Overview

This repository builds a production-grade machine learning application. It utilizes the UCI Wine Quality Dataset to predict the quality of wines (red and white) from chemical features.

Key features:
- **Data Engineering**: Data cleaning, scaling, and feature engineering.
- **Machine Learning**: Linear Regression, Random Forest, XGBoost, and LightGBM models.
- **MLOps**: MLflow for tracking parameters, metrics, and models.
- **API**: FastAPI with endpoints for single predictions, batch predictions (CSV), health monitoring, and SQLite logs database.
- **Auth**: JWT-based login/signup for secure API access.
- **Frontend**: Streamlit application for interactive prediction forms, batch CSV analysis, and historical logs.
- **Docker**: Complete Docker Compose setup for localized service orchestration.

## Getting Started

### Prerequisites
- Python 3.9+
- Git
- Docker & Docker Compose (optional, for Phase 9+)

### Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd "Wine Predictions Project"
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Folder Structure
```
├── data/           # Raw and processed datasets
├── src/            # Python source code for data cleaning, training, and evaluation
├── api/            # FastAPI application files
├── models/         # Saved pickle model and scaler files
├── tests/          # Unit and integration tests
├── docs/           # Project reports and documentation
├── app.py          # Streamlit frontend app
└── docker-compose.yml
```

## Running the API

You can start the FastAPI application using `uvicorn`:
```bash
# From the project root directory
.venv\Scripts\uvicorn api.main:app --reload
```
Once started, the API will be available at `http://127.0.0.1:8000`.

### Interactive Documentation (Swagger)
FastAPI automatically generates interactive Swagger UI documentation at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### API Endpoints and Usage Examples

#### 1. Single Wine Prediction `/predict`
Send chemical properties of a single wine to predict its quality.

**Request**:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "fixed_acidity": 7.4,
       "volatile_acidity": 0.7,
       "citric_acid": 0.0,
       "residual_sugar": 1.9,
       "chlorides": 0.076,
       "free_sulfur_dioxide": 11.0,
       "total_sulfur_dioxide": 34.0,
       "density": 0.9978,
       "pH": 3.51,
       "sulphates": 0.56,
       "alcohol": 9.4,
       "type": "red"
     }'
```

**Response**:
```json
{
  "prediction": 5.234,
  "rounded_prediction": 5,
  "wine_type": "red"
}
```

#### 2. Batch Prediction `/predict_batch`
Upload a CSV file containing multiple wine records and get a CSV containing predictions in return.

**Request**:
```bash
curl -X POST "http://127.0.0.1:8000/predict_batch" \
     -F "file=@path/to/your/wines.csv" \
     --output predictions.csv
```

#### 3. Model Info `/model_info`
Retrieve information about the loaded model, including hyperparameters and expected features.

```bash
curl -X GET "http://127.0.0.1:8000/model_info"
```

#### 4. Model Performance Metrics `/metrics`
Retrieve performance metrics (MAE, RMSE, R²) computed during validation.

```bash
curl -X GET "http://127.0.0.1:8000/metrics"
```

## Running with Docker Compose

For a complete production-grade deployment, you can orchestrate both the FastAPI backend and Streamlit frontend services using Docker Compose.

### Quickstart

1. **Build and Start Container Services**:
   ```bash
   docker-compose up --build
   ```

2. **Access the Services**:
   - **FastAPI API Server**: [http://localhost:8000](http://localhost:8000)
   - **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Streamlit Frontend Dashboard**: [http://localhost:8501](http://localhost:8501)

3. **Shutdown Services**:
   ```bash
   docker-compose down
   ```


