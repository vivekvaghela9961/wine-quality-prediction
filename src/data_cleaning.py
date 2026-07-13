import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """Load wine quality dataset with semicolon separator."""
    return pd.read_csv(filepath, sep=";")

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by filling numerical NaNs with column median."""
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().any():
            median = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median)
    return df_clean

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""
    return df.drop_duplicates().reset_index(drop=True)

def cap_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """Cap outliers using the Interquartile Range (IQR) method (clipping)."""
    df_clean = df.copy()
    for col in columns:
        if col not in df_clean.columns:
            continue
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    return df_clean

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Perform the full cleaning pipeline on the input dataset."""
    df_clean = handle_missing_values(df)
    df_clean = remove_duplicates(df_clean)
    
    # Cap outliers for numeric columns (excluding 'quality' target if it exists)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if 'quality' in numeric_cols:
        numeric_cols.remove('quality')
        
    df_clean = cap_outliers_iqr(df_clean, numeric_cols)
    return df_clean
