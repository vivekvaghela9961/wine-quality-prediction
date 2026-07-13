import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.feature_engineering import scale_features

def load_processed_data(filepath: str):
    """Load the processed wine quality dataset."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=["quality"])
    y = df["quality"]
    return X, y

def main():
    print("Loading processed dataset...")
    X, y = load_processed_data("data/processed/winequality-processed.csv")
    
    # Train-test split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    print("Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        preds = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })
        
    # Display results as a DataFrame
    results_df = pd.DataFrame(results)
    print("\n--- Model Comparison ---")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
