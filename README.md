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
