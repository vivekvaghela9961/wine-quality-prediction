import os
import json
import pandas as pd
from scipy.stats import ks_2samp
from api.database import SessionLocal
from api.models import PredictionLog

def detect_drift(reference_path: str, min_records: int = 10):
    """
    Detect statistical drift in incoming prediction features compared to training reference data.
    Uses Kolmogorov-Smirnov test. If p-value < 0.05, we reject the null hypothesis that
    the sample distributions are from the same distribution (indicating drift).
    """
    print("Starting data drift detection analysis...")
    
    if not os.path.exists(reference_path):
        print(f"Reference training data not found at {reference_path}. Cannot perform drift check.")
        return None
        
    # 1. Load baseline training features
    df_ref = pd.read_csv(reference_path)
    
    # 2. Load prediction logs from DB
    db = SessionLocal()
    try:
        logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(500).all()
    finally:
        db.close()
        
    if len(logs) < min_records:
        print(f"Insufficient prediction logs (found {len(logs)}, need at least {min_records}) to analyze drift.")
        return {"status": "insufficient_data", "records_found": len(logs)}
        
    # Convert logs to pandas dataframe
    log_data = []
    for log in logs:
        log_data.append({
            "fixed acidity": log.fixed_acidity,
            "volatile acidity": log.volatile_acidity,
            "citric acid": log.citric_acid,
            "residual sugar": log.residual_sugar,
            "chlorides": log.chlorides,
            "free sulfur dioxide": log.free_sulfur_dioxide,
            "total sulfur dioxide": log.total_sulfur_dioxide,
            "density": log.density,
            "pH": log.pH,
            "sulphates": log.sulphates,
            "alcohol": log.alcohol
        })
    df_pred = pd.DataFrame(log_data)
    
    features = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]
    
    drift_results = {}
    drift_detected = False
    
    print("\n--- Kolmogorov-Smirnov Drift Test Results ---")
    for feat in features:
        # Normalize name for dataframe lookup (if baseline has underscores)
        ref_feat = feat
        if feat not in df_ref.columns and feat.replace(" ", "_") in df_ref.columns:
            ref_feat = feat.replace(" ", "_")
            
        ref_dist = df_ref[ref_feat].dropna()
        pred_dist = df_pred[feat].dropna()
        
        # Run Kolmogorov-Smirnov test
        stat, p_val = ks_2samp(ref_dist, pred_dist)
        
        # Check if drifted (alpha = 0.05)
        is_drifted = bool(p_val < 0.05)
        if is_drifted:
            drift_detected = True
            
        drift_results[feat] = {
            "ks_statistic": float(stat),
            "p_value": float(p_val),
            "drift_detected": is_drifted,
            "baseline_mean": float(ref_dist.mean()),
            "logged_mean": float(pred_dist.mean())
        }
        
        status_str = "DRIFT DETECTED [WARNING]" if is_drifted else "Normal"
        print(f"Feature: {feat:<22} | p-value: {p_val:<10.4f} | Status: {status_str}")
        
    report = {
        "status": "success",
        "drift_detected": drift_detected,
        "total_records_analyzed": len(df_pred),
        "results": drift_results
    }
    
    # Save report
    os.makedirs("docs", exist_ok=True)
    report_path = "docs/drift_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\nDrift report successfully saved to {report_path}")
    return report

if __name__ == "__main__":
    detect_drift("data/processed/winequality-processed.csv", min_records=5)
