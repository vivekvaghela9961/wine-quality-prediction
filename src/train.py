import os
import pickle
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.feature_engineering import scale_features

# Disable Optuna verbose logging by default to keep output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_processed_data(filepath: str):
    """Load the processed wine quality dataset."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=["quality"])
    y = df["quality"]
    return X, y

def tune_random_forest(X_train, y_train):
    """Tune RandomForestRegressor using Optuna."""
    print("Running Optuna hyperparameter optimization for Random Forest...")
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state": 42
        }
        model = RandomForestRegressor(**params)
        # Use 3-fold cross validation for speed
        score = cross_val_score(model, X_train, y_train, cv=3, scoring="neg_root_mean_squared_error").mean()
        return -score  # Minimize RMSE

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    print(f"Best trial: RMSE {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params

def save_artifacts(model, scaler):
    """Save the model and scaler to models/ and artifacts/ folders."""
    for folder in ["models", "artifacts"]:
        os.makedirs(folder, exist_ok=True)
        
        model_path = os.path.join(folder, "model.pkl")
        scaler_path = os.path.join(folder, "scaler.pkl")
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
            
        print(f"Saved artifacts to {folder}/")

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
        print(f"Training baseline {name}...")
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
        
    results_df = pd.DataFrame(results)
    print("\n--- Model Comparison (Baseline) ---")
    print(results_df.to_string(index=False))
    
    # Tune the best model (Random Forest)
    best_params = tune_random_forest(X_train_scaled, y_train)
    
    # Train final model with best params
    print("\nTraining final tuned RandomForest model...")
    best_model = RandomForestRegressor(**best_params, random_state=42)
    best_model.fit(X_train_scaled, y_train)
    
    final_preds = best_model.predict(X_test_scaled)
    final_mae = mean_absolute_error(y_test, final_preds)
    final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    final_r2 = r2_score(y_test, final_preds)
    
    print("\n--- Tuned Random Forest Evaluation ---")
    print(f"Mean Absolute Error (MAE): {final_mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {final_rmse:.4f}")
    print(f"R2 Score: {final_r2:.4f}")
    
    # Save best model and scaler
    save_artifacts(best_model, scaler)

if __name__ == "__main__":
    main()
