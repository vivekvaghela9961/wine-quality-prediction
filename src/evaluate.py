import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_model_and_scaler(models_dir="models"):
    """Load model.pkl and scaler.pkl from the models directory."""
    with open(os.path.join(models_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def load_processed_data(filepath="data/processed/winequality-processed.csv"):
    """Load the processed wine quality dataset."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=["quality"])
    y = df["quality"]
    return X, y


def main():
    print("Loading data...")
    X, y = load_processed_data()

    print("Loading model and scaler...")
    model, scaler = load_model_and_scaler()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    # Evaluation on Test Split
    print("Evaluating model on test split...")
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n--- Model Evaluation on Test Split ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Cross Validation
    print("\nRunning 5-fold cross-validation on full dataset...")
    cv_scores = cross_val_score(
        model, X_scaled, y, cv=5, scoring="neg_root_mean_squared_error"
    )
    cv_rmse = -cv_scores
    print(f"5-Fold CV RMSE Scores: {cv_rmse}")
    print(f"Mean CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")

    # Plot residuals
    print("\nGenerating residual plot...")
    residuals = y_test - preds

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=preds, y=residuals, alpha=0.5, color="#800020")
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Residual Plot (Predicted vs Residuals)")
    plt.xlabel("Predicted Quality")
    plt.ylabel("Residuals")

    os.makedirs("docs", exist_ok=True)
    plot_path = "docs/residuals.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Residual plot saved to {plot_path}")


if __name__ == "__main__":
    # Prevent matplotlib from blocking on execution
    os.environ["MPLBACKEND"] = "Agg"
    main()
