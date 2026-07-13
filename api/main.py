import os
import io
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Define Pydantic schema for single wine input
class WineInput(BaseModel):
    fixed_acidity: float = Field(..., example=7.4)
    volatile_acidity: float = Field(..., example=0.7)
    citric_acid: float = Field(..., example=0.0)
    residual_sugar: float = Field(..., example=1.9)
    chlorides: float = Field(..., example=0.076)
    free_sulfur_dioxide: float = Field(..., example=11.0)
    total_sulfur_dioxide: float = Field(..., example=34.0)
    density: float = Field(..., example=0.9978)
    pH: float = Field(..., example=3.51)
    sulphates: float = Field(..., example=0.56)
    alcohol: float = Field(..., example=9.4)
    type: str = Field(..., example="red", description="Type of wine: 'red' or 'white'")

# Global variables for model and scaler
model = None
scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler on startup and clean up on shutdown."""
    global model, scaler
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
            print(f"Successfully loaded model and scaler from '{folder}/'")
            break
            
    if not model_loaded:
        raise RuntimeError("Model or scaler pickle file not found. Run training first.")
        
    yield
    # Clean up on shutdown if needed
    model = None
    scaler = None

app = FastAPI(
    title="Wine Quality Prediction API",
    description="An end-to-end API for predicting wine quality from chemical characteristics.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    """Root endpoint welcoming the user."""
    return {
        "message": "Welcome to the Wine Quality Prediction API",
        "docs_url": "/docs",
        "status": "running"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict")
def predict_quality(wine: WineInput):
    """Predict wine quality score based on chemical features."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    # Validate type
    wine_type = wine.type.lower().strip()
    if wine_type not in ["red", "white"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid wine type. Must be 'red' or 'white'."
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
        "alcohol": wine.alcohol
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
        "is_red": is_red
    }
    
    # Define exact feature order required by the model
    feature_order = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol", "total_acidity", "acid_to_sugar_ratio",
        "bound_sulfur_dioxide", "is_red"
    ]
    
    # Create DataFrame in the exact feature order
    features_df = pd.DataFrame([features_dict])[feature_order]
    
    try:
        # Scale features
        features_scaled = pd.DataFrame(
            scaler.transform(features_df),
            columns=features_df.columns,
            index=features_df.index
        )
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        return {
            "prediction": float(prediction),
            "rounded_prediction": int(round(prediction)),
            "wine_type": wine_type
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error making prediction: {str(e)}"
        )

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Predict wine quality for a batch of wines uploaded via a CSV file."""
    if model is None or scaler is None:
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
        "wine_type": "type"
    }
    
    # Check if all required columns are present after mapping
    mapped_columns = {}
    for col in df_raw.columns:
        norm_col = col.lower().strip()
        if norm_col in col_mapping:
            mapped_columns[col_mapping[norm_col]] = col
            
    required_fields = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol", "type"
    ]
    
    missing_fields = [f for f in required_fields if f not in mapped_columns]
    if missing_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns in CSV. Missing: {missing_fields}"
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
            status_code=400,
            detail="All wines must have type 'red' or 'white'."
        )
        
    # Calculate engineered features
    features_df["total_acidity"] = (
        features_df["fixed acidity"] + features_df["volatile acidity"] + features_df["citric acid"]
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
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol", "total_acidity", "acid_to_sugar_ratio",
        "bound_sulfur_dioxide", "is_red"
    ]
    
    # Ensure columns order
    features_df = features_df[feature_order]
    
    try:
        # Scale features
        features_scaled = pd.DataFrame(
            scaler.transform(features_df),
            columns=features_df.columns,
            index=features_df.index
        )
        
        # Predict
        predictions = model.predict(features_scaled)
        
        # Add predictions to original df
        df_out = df_raw.copy()
        df_out["predicted_quality"] = predictions
        df_out["rounded_predicted_quality"] = np.round(predictions).astype(int)
        
        # Convert to CSV response
        stream = io.StringIO()
        df_out.to_csv(stream, index=False)
        
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing batch prediction: {str(e)}"
        )
