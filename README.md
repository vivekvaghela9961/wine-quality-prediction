<div align="center">

# 🍷 Vinum Predict

### *Premium Wine Quality Intelligence — Powered by Machine Learning*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-gold?style=for-the-badge)](LICENSE)

<br/>

> *Predict the quality of red and white wines using physicochemical parameters — served through a production-grade REST API with a luxurious, glassmorphic web dashboard.*

<br/>

</div>

---

## ✨ What Is This?

**Vinum Predict** is an end-to-end machine learning system that takes the chemical properties of a wine — acidity, sugar, alcohol content, pH — and predicts its quality score (1–10).

It is built like a real production system:

| Layer | Technology |
|---|---|
| 🧠 **ML Model** | Optuna-tuned Random Forest Regressor |
| 🔁 **Experiment Tracking** | MLflow (metrics, params, artifacts) |
| ⚡ **API** | FastAPI with JWT auth, rate limiting & middleware |
| 💾 **Database** | SQLite via SQLAlchemy ORM |
| 🌐 **Frontend** | Premium Glassmorphic HTML/JS/CSS app |
| 🐳 **Containerisation** | Docker + Docker Compose |
| 🔬 **Testing** | pytest with 22 test cases & coverage reporting |
| 🚀 **CI/CD** | GitHub Actions (lint → test → docker build) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              🌐  Web Browser (Port 8000)             │
│          Premium Glassmorphic HTML/JS/CSS UI         │
└───────────────────────┬─────────────────────────────┘
                        │  HTTP/REST  (Bearer JWT Token)
                        ▼
┌─────────────────────────────────────────────────────┐
│              ⚡  FastAPI Backend (Port 8000)          │
│                                                     │
│   /predict         → Single wine prediction         │
│   /predict_batch   → Bulk CSV predictions           │
│   /predictions_history → Logged past results        │
│   /model_info      → Feature list & params          │
│   /metrics         → MAE, RMSE, R² scores           │
│   /auth/login      → JWT token issuance             │
│   /auth/signup     → User registration              │
└──────────┬──────────────────────────┬───────────────┘
           │  SQLAlchemy ORM           │  Pickle Load
           ▼                          ▼
┌──────────────────┐       ┌──────────────────────────┐
│  💾 SQLite DB    │       │  🧠 Tuned Random Forest  │
│   (db.sqlite3)   │       │   Model + StandardScaler │
│  Users & Logs    │       │   (models/model.pkl)     │
└──────────────────┘       └──────────────────────────┘
```

---

## 📊 Model Performance

The Random Forest Regressor was tuned with **Optuna** and evaluated on a held-out test set:

| Metric | Score | What it means |
|---|---|---|
| **MAE** | `0.5115` | Average error of ±0.51 quality points |
| **RMSE** | `0.6655` | Root mean squared error |
| **R² Score** | `0.4109` | Model explains ~41% of quality variance |
| **CV RMSE** | `0.7423` | Stable across 5-fold cross-validation |

> 📌 Wine quality is a notoriously subjective dataset — an R² of ~0.41 is competitive with published benchmarks on this dataset.

---

## 🚀 Quick Start

### ✅ Prerequisites

- Python `3.10+`
- Git
- Docker & Docker Compose *(optional, for containerised deployment)*

---

### 🖥️ Option 1 — Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/vivekvaghela9961/wine-quality-prediction.git
cd "wine-quality-prediction"
```

**2. Create a virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
```

**5. Start the API + Frontend**
```bash
uvicorn api.main:app --port 8000
```

🎉 Open **[http://localhost:8000](http://localhost:8000)** in your browser — the full app is served right there!

---

### 🐳 Option 2 — Run with Docker Compose

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| 🌐 Web App & API | http://localhost:8000 |
| 📖 Swagger Docs | http://localhost:8000/docs |

---

## 🔑 API Reference

All prediction endpoints require a **JWT Bearer token**. First, create an account and log in:

```bash
# 1. Sign up
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"username": "your_user", "password": "your_pass"}'

# 2. Log in and grab the token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_user", "password": "your_pass"}'
```

Then use the returned `access_token` as a Bearer token on all prediction calls:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "red",
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
    "alcohol": 9.4
  }'
```

**Response:**
```json
{
  "prediction": 5.287,
  "rounded_prediction": 5,
  "wine_type": "red"
}
```

---

## 🧪 Testing

Run the full test suite (22 test cases covering API, authentication, batch predictions, and data processing):

```bash
# With coverage report
pytest --cov=api --cov=src --cov-report=term-missing tests/

# PowerShell shortcut
./run_tests.ps1
```

---

## 🔬 Data Drift Detection

Monitor whether live inference data is drifting away from the training distribution using **Kolmogorov-Smirnov** tests:

```bash
python -m src.drift_detection
```

Results are printed to the console and saved to `docs/drift_report.json`.

---

## 🗂️ Project Structure

```
wine-quality-prediction/
│
├── 🌐 frontend/              # Premium glassmorphic UI
│   ├── index.html            # App structure
│   ├── styles.css            # Luxury dark theme + glassmorphism
│   ├── app.js                # API calls, auth, charts
│   └── images/               # Wine cellar & vineyard backgrounds
│
├── ⚡ api/                   # FastAPI backend
│   ├── main.py               # Routes, middleware, static files
│   ├── auth.py               # JWT authentication
│   ├── models.py             # SQLAlchemy ORM models
│   ├── database.py           # DB engine & session
│   └── logging_config.py     # Structured logging
│
├── 🧠 src/                   # ML pipeline
│   ├── download_data.py      # Dataset download
│   ├── preprocess.py         # Cleaning, feature engineering
│   ├── train.py              # Model training + Optuna tuning
│   ├── evaluate.py           # Metrics & MLflow logging
│   └── drift_detection.py    # KS test drift monitoring
│
├── 🔬 tests/                 # 22 pytest test cases
├── 📦 models/                # Trained model & scaler pickles
├── 📋 docs/                  # Contributing guide, drift reports
├── 🐳 docker-compose.yml     # Multi-container orchestration
└── ⚙️ .github/workflows/     # CI/CD pipeline (lint, test, build)
```

---

## 🤝 Contributing

Contributions are welcome! Please read [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for setup instructions, code standards, and the branching strategy.

---

<div align="center">

Made with 🍷 by **Vivek Vaghela**

*If you found this project interesting, consider starring ⭐ the repo!*

</div>
