import os
import io
import pickle
import time
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from api.database import engine, Base, get_db
from api.models import PredictionLog, User
from api.auth import (
    get_current_user,
    hash_password,
    verify_password,
    create_access_token,
)
from api.logging_config import api_logger
from fastapi.staticfiles import StaticFiles


# Define Pydantic schema for single wine input
class WineInput(BaseModel):
    fixed_acidity: float = Field(..., json_schema_extra={"example": 7.4})
    volatile_acidity: float = Field(..., json_schema_extra={"example": 0.7})
    citric_acid: float = Field(..., json_schema_extra={"example": 0.0})
    residual_sugar: float = Field(..., json_schema_extra={"example": 1.9})
    chlorides: float = Field(..., json_schema_extra={"example": 0.076})
    free_sulfur_dioxide: float = Field(..., json_schema_extra={"example": 11.0})
    total_sulfur_dioxide: float = Field(..., json_schema_extra={"example": 34.0})
    density: float = Field(..., json_schema_extra={"example": 0.9978})
    pH: float = Field(..., json_schema_extra={"example": 3.51})
    sulphates: float = Field(..., json_schema_extra={"example": 0.56})
    alcohol: float = Field(..., json_schema_extra={"example": 9.4})
    type: str = Field(
        ...,
        json_schema_extra={"example": "red"},
        description="Type of wine: 'red' or 'white'",
    )


# Define Pydantic schema for authentication input
class UserAuth(BaseModel):
    username: str = Field(..., json_schema_extra={"example": "admin"})
    password: str = Field(..., json_schema_extra={"example": "secure_pass_123"})


# Global variables for model and scaler
model = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler on startup and clean up on shutdown."""
    global model, scaler
    api_logger.info("Initializing API application lifespan setup...")
    # Create SQLite tables on startup
    Base.metadata.create_all(bind=engine)
    api_logger.info("Database tables verified.")

    # Try models folder first, then artifacts folder
    model_loaded = False
    for folder in ["models", "artifacts"]:
        model_path = os.path.join(folder, "model.pkl")
        scaler_path = os.path.join(folder, "scaler.pkl")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            model_loaded = True
            api_logger.info(f"Successfully loaded model and scaler from '{folder}/'")
            break

    if not model_loaded:
        api_logger.error("API startup failed: Model or scaler pickle file not found.")
        raise RuntimeError("Model or scaler pickle file not found. Run training first.")

    yield
    # Clean up on shutdown if needed
    model = None
    scaler = None
    api_logger.info("API application lifespan shutdown complete.")


app = FastAPI(
    title="Wine Quality Prediction API",
    description="API for predicting wine quality from chemical characteristics.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request


@app.middleware("http")
async def log_request_process_time(request: Request, call_next):
    """Middleware tracking process time and status codes for API health monitoring."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    api_logger.info(
        f"HTTP Request | Path: {request.url.path} | Method: {request.method} | "
        f"Status: {response.status_code} | Latency: {process_time:.4f}s"
    )
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/auth/signup")
def signup(user_data: UserAuth, db: Session = Depends(get_db)):
    """Create a new API user account."""
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username is already taken.")

    hashed = hash_password(user_data.password)
    new_user = User(username=user_data.username, hashed_password=hashed)
    db.add(new_user)
    db.commit()
    return {"message": "User signed up successfully", "username": user_data.username}


@app.post("/auth/login")
def login(user_data: UserAuth, db: Session = Depends(get_db)):
    """Authenticate credentials and return a JWT access token."""
    user = db.query(User).filter(User.username == user_data.username).first()
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/predict")
def predict_quality(
    wine: WineInput,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """Predict wine quality score based on chemical features."""
    api_logger.info(
        f"User '{current_user}' requested quality prediction for a '{wine.type}' wine."
    )
    if model is None or scaler is None:
        api_logger.warning("Prediction endpoint called but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    # Validate type
    wine_type = wine.type.lower().strip()
    if wine_type not in ["red", "white"]:
        raise HTTPException(
            status_code=400, detail="Invalid wine type. Must be 'red' or 'white'."
        )

    # Convert input model to dictionary with training keys
    raw_features = {
        "fixed acidity": wine.fixed_acidity,
        "volatile acidity": wine.volatile_acidity,
        "citric acid": wine.citric_acid,
        "residual sugar": wine.residual_sugar,
        "chlorides": wine.chlorides,
        "free sulfur dioxide": wine.free_sulfur_dioxide,
        "total sulfur dioxide": wine.total_sulfur_dioxide,
        "density": wine.density,
        "pH": wine.pH,
        "sulphates": wine.sulphates,
        "alcohol": wine.alcohol,
    }

    # Calculate engineered features
    total_acidity = wine.fixed_acidity + wine.volatile_acidity + wine.citric_acid
    acid_to_sugar_ratio = total_acidity / (wine.residual_sugar + 1e-5)
    bound_sulfur_dioxide = wine.total_sulfur_dioxide - wine.free_sulfur_dioxide
    is_red = 1 if wine_type == "red" else 0

    # Add engineered features
    features_dict = {
        **raw_features,
        "total_acidity": total_acidity,
        "acid_to_sugar_ratio": acid_to_sugar_ratio,
        "bound_sulfur_dioxide": bound_sulfur_dioxide,
        "is_red": is_red,
    }

    # Define exact feature order required by the model
    feature_order = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "total_acidity",
        "acid_to_sugar_ratio",
        "bound_sulfur_dioxide",
        "is_red",
    ]

    # Create DataFrame in the exact feature order
    features_df = pd.DataFrame([features_dict])[feature_order]

    try:
        # Scale features
        features_scaled = pd.DataFrame(
            scaler.transform(features_df),
            columns=features_df.columns,
            index=features_df.index,
        )

        # Predict
        prediction = model.predict(features_scaled)[0]
        api_logger.info(f"Model prediction generated: {prediction}")

        # Persist to database prediction history
        try:
            log_entry = PredictionLog(
                wine_type=wine_type,
                fixed_acidity=wine.fixed_acidity,
                volatile_acidity=wine.volatile_acidity,
                citric_acid=wine.citric_acid,
                residual_sugar=wine.residual_sugar,
                chlorides=wine.chlorides,
                free_sulfur_dioxide=wine.free_sulfur_dioxide,
                total_sulfur_dioxide=wine.total_sulfur_dioxide,
                density=wine.density,
                pH=wine.pH,
                sulphates=wine.sulphates,
                alcohol=wine.alcohol,
                predicted_quality=float(prediction),
            )
            db.add(log_entry)
            db.commit()
            api_logger.info(
                f"Prediction successfully logged to database with ID: {log_entry.id}"
            )
        except Exception as db_err:
            # We print db logging failures so the endpoint response doesn't fail
            api_logger.error(f"Database logging failed: {db_err}")

        return {
            "prediction": float(prediction),
            "rounded_prediction": int(round(prediction)),
            "wine_type": wine_type,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error making prediction: {str(e)}"
        )


@app.post("/predict_batch")
async def predict_batch(
    file: UploadFile = File(...), current_user: str = Depends(get_current_user)
):
    """Predict wine quality for a batch of wines uploaded via a CSV file."""
    api_logger.info(
        f"User '{current_user}' uploaded batch prediction file: '{file.filename}'"
    )
    if model is None or scaler is None:
        api_logger.warning("Batch prediction endpoint called but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        # Read file content
        content = await file.read()
        df_raw = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    # Standardize column mapping to support both space and underscore separators
    col_mapping = {
        "fixed acidity": "fixed acidity",
        "fixed_acidity": "fixed acidity",
        "volatile acidity": "volatile acidity",
        "volatile_acidity": "volatile acidity",
        "citric acid": "citric acid",
        "citric_acid": "citric acid",
        "residual sugar": "residual sugar",
        "residual_sugar": "residual sugar",
        "chlorides": "chlorides",
        "free sulfur dioxide": "free sulfur dioxide",
        "free_sulfur_dioxide": "free sulfur dioxide",
        "total sulfur dioxide": "total sulfur dioxide",
        "total_sulfur_dioxide": "total sulfur dioxide",
        "density": "density",
        "pH": "pH",
        "ph": "pH",
        "sulphates": "sulphates",
        "alcohol": "alcohol",
        "type": "type",
        "wine_type": "type",
    }

    # Check if all required columns are present after mapping
    mapped_columns = {}
    for col in df_raw.columns:
        norm_col = col.lower().strip()
        if norm_col in col_mapping:
            mapped_columns[col_mapping[norm_col]] = col

    required_fields = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "type",
    ]

    missing_fields = [f for f in required_fields if f not in mapped_columns]
    if missing_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns in CSV. Missing: {missing_fields}",
        )

    # Extract features for prediction
    features_df = pd.DataFrame()
    for field in required_fields:
        orig_col = mapped_columns[field]
        features_df[field] = df_raw[orig_col]

    # Standardize wine types
    features_df["type"] = features_df["type"].astype(str).str.lower().str.strip()
    invalid_types = features_df[~features_df["type"].isin(["red", "white"])]
    if not invalid_types.empty:
        raise HTTPException(
            status_code=400, detail="All wines must have type 'red' or 'white'."
        )

    # Calculate engineered features
    features_df["total_acidity"] = (
        features_df["fixed acidity"]
        + features_df["volatile acidity"]
        + features_df["citric acid"]
    )
    features_df["acid_to_sugar_ratio"] = features_df["total_acidity"] / (
        features_df["residual sugar"] + 1e-5
    )
    features_df["bound_sulfur_dioxide"] = (
        features_df["total sulfur dioxide"] - features_df["free sulfur dioxide"]
    )
    features_df["is_red"] = features_df["type"].map({"red": 1, "white": 0}).astype(int)

    # Drop type column
    features_df = features_df.drop(columns=["type"])

    # Define exact feature order required by the model
    feature_order = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "total_acidity",
        "acid_to_sugar_ratio",
        "bound_sulfur_dioxide",
        "is_red",
    ]

    # Ensure columns order
    features_df = features_df[feature_order]

    try:
        # Scale features
        features_scaled = pd.DataFrame(
            scaler.transform(features_df),
            columns=features_df.columns,
            index=features_df.index,
        )

        # Predict
        predictions = model.predict(features_scaled)
        api_logger.info(f"Generated {len(predictions)} batch predictions successfully.")

        # Add predictions to original df
        df_out = df_raw.copy()
        df_out["predicted_quality"] = predictions
        df_out["rounded_predicted_quality"] = np.round(predictions).astype(int)

        # Convert to CSV response
        stream = io.StringIO()
        df_out.to_csv(stream, index=False)

        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
        api_logger.info(f"Streaming back predictions CSV for user '{current_user}'")
        return response
    except Exception as e:
        api_logger.error(f"Error performing batch prediction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error performing batch prediction: {str(e)}"
        )


@app.get("/model_info")
def get_model_info():
    """Return parameters and features expected by the trained model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "model_name": type(model).__name__,
        "parameters": model.get_params(),
        "features": [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
            "total_acidity",
            "acid_to_sugar_ratio",
            "bound_sulfur_dioxide",
            "is_red",
        ],
    }


@app.get("/metrics")
def get_metrics():
    """Return model performance metrics from final validation and cross-validation."""
    return {
        "MAE": 0.5115,
        "RMSE": 0.6655,
        "R2": 0.4109,
        "CV_RMSE_Mean": 0.7423,
        "CV_RMSE_Std": 0.0361,
    }


@app.get("/predictions_history")
def get_predictions_history(
    db: Session = Depends(get_db), current_user: str = Depends(get_current_user)
):
    """Retrieve prediction history logs."""
    api_logger.info(f"User '{current_user}' requested prediction logs history.")
    logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(100).all()
    return [
        {
            "id": log.id,
            "timestamp": log.timestamp.isoformat(),
            "wine_type": log.wine_type,
            "fixed_acidity": log.fixed_acidity,
            "volatile_acidity": log.volatile_acidity,
            "citric_acid": log.citric_acid,
            "residual_sugar": log.residual_sugar,
            "chlorides": log.chlorides,
            "free_sulfur_dioxide": log.free_sulfur_dioxide,
            "total_sulfur_dioxide": log.total_sulfur_dioxide,
            "density": log.density,
            "pH": log.pH,
            "sulphates": log.sulphates,
            "alcohol": log.alcohol,
            "predicted_quality": log.predicted_quality,
        }
        for log in logs
    ]


# Mount the static frontend at the root path
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
