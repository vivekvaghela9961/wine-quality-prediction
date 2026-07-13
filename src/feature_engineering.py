import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new domain-specific features for wine quality."""
    df_feat = df.copy()

    # 1. Total acidity
    df_feat["total_acidity"] = (
        df_feat["fixed acidity"] + df_feat["volatile acidity"] + df_feat["citric acid"]
    )

    # 2. Ratio of acidity to sugar (avoid division by zero)
    df_feat["acid_to_sugar_ratio"] = df_feat["total_acidity"] / (
        df_feat["residual sugar"] + 1e-5
    )

    # 3. Bound sulfur dioxide (total SO2 - free SO2)
    df_feat["bound_sulfur_dioxide"] = (
        df_feat["total sulfur dioxide"] - df_feat["free sulfur dioxide"]
    )

    return df_feat


def encode_type(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the string 'type' column: 'red' -> 1, 'white' -> 0."""
    df_enc = df.copy()
    if "type" in df_enc.columns:
        # If already numeric, do nothing
        if not pd.api.types.is_numeric_dtype(df_enc["type"]):
            df_enc["is_red"] = (
                df_enc["type"].map({"red": 1, "white": 0}).fillna(0).astype(int)
            )
            df_enc = df_enc.drop(columns=["type"])
    return df_enc


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame = None, scaler: StandardScaler = None
):
    """
    Scale numeric features using StandardScaler.

    If scaler is not provided, fit a new one on X_train.
    Returns:
        scaled_X_train, scaled_X_test (if X_test provided), scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled_train_arr = scaler.fit_transform(X_train)
    else:
        scaled_train_arr = scaler.transform(X_train)

    scaled_train = pd.DataFrame(
        scaled_train_arr, columns=X_train.columns, index=X_train.index
    )

    if X_test is not None:
        scaled_test_arr = scaler.transform(X_test)
        scaled_test = pd.DataFrame(
            scaled_test_arr, columns=X_test.columns, index=X_test.index
        )
        return scaled_train, scaled_test, scaler

    return scaled_train, scaler
