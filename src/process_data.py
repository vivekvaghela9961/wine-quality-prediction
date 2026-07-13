import os
import pandas as pd
from src.data_cleaning import clean_dataset
from src.feature_engineering import create_features, encode_type


def process_data(red_path: str, white_path: str, output_path: str):
    """Run data cleaning and feature engineering pipelines and save results."""
    print("Loading raw data...")
    red_df = pd.read_csv(red_path, sep=";")
    white_df = pd.read_csv(white_path, sep=";")

    # Add type indicator
    red_df["type"] = "red"
    white_df["type"] = "white"

    # Combine datasets
    df = pd.concat([red_df, white_df], ignore_index=True)
    print(f"Combined raw dataset shape: {df.shape}")

    # 1. Clean data (missing values, duplicates, outliers capping)
    print("Cleaning dataset...")
    df_cleaned = clean_dataset(df)
    print(f"Cleaned dataset shape: {df_cleaned.shape}")

    # 2. Feature engineering (create features)
    print("Engineering features...")
    df_engineered = create_features(df_cleaned)

    # 3. Categorical encoding (type -> is_red)
    print("Encoding wine type...")
    df_processed = encode_type(df_engineered)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path} with shape {df_processed.shape}")


if __name__ == "__main__":
    red_raw = "data/raw/winequality-red.csv"
    white_raw = "data/raw/winequality-white.csv"
    processed_out = "data/processed/winequality-processed.csv"

    process_data(red_raw, white_raw, processed_out)
