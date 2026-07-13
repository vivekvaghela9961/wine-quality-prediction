import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.data_cleaning import handle_missing_values, remove_duplicates, cap_outliers_iqr, clean_dataset
from src.feature_engineering import create_features, encode_type, scale_features

@pytest.fixture
def mock_raw_df():
    """Create a mock DataFrame with missing values, duplicates, and outliers."""
    return pd.DataFrame({
        "fixed acidity": [7.0, 7.0, 7.0, 15.0, np.nan],  # NaN and duplicate, outlier 15.0
        "volatile acidity": [0.27, 0.27, 0.27, 0.3, 0.4], # duplicates
        "citric acid": [0.36, 0.36, 0.36, 0.2, 0.1],
        "residual sugar": [20.7, 20.7, 20.7, 1.0, 2.0],
        "chlorides": [0.045, 0.045, 0.045, 0.05, 0.06],
        "free sulfur dioxide": [45.0, 45.0, 45.0, 50.0, 60.0],
        "total sulfur dioxide": [170.0, 170.0, 170.0, 180.0, 190.0],
        "density": [1.001, 1.001, 1.001, 0.999, 0.998],
        "pH": [3.0, 3.0, 3.0, 3.2, 3.1],
        "sulphates": [0.45, 0.45, 0.45, 0.5, 0.65],
        "alcohol": [8.8, 8.8, 8.8, 11.0, 12.0],
        "quality": [6, 6, 6, 5, 7],
        "type": ["white", "white", "white", "red", "red"]
    })

def test_handle_missing_values(mock_raw_df):
    # Verify NaN exists first
    assert mock_raw_df["fixed acidity"].isnull().any()
    
    df_clean = handle_missing_values(mock_raw_df)
    # Verify no NaN exists after function
    assert not df_clean["fixed acidity"].isnull().any()
    # Fill value should be median of non-nulls (7.0, 7.0, 7.0, 15.0 -> median is 7.0)
    assert df_clean.loc[4, "fixed acidity"] == 7.0

def test_remove_duplicates(mock_raw_df):
    assert len(mock_raw_df) == 5
    df_no_dup = remove_duplicates(mock_raw_df)
    # The first 3 rows are identical, so 2 should be dropped -> length 3
    assert len(df_no_dup) == 3

def test_cap_outliers_iqr(mock_raw_df):
    # Let's clean missing values first so IQR works on non-nulls
    df_no_nan = handle_missing_values(mock_raw_df)
    # Volatile acidity values: [0.27, 0.27, 0.27, 0.3, 0.4] -> Q1=0.27, Q3=0.3, IQR=0.03
    # Upper bound = 0.3 + 1.5 * 0.03 = 0.345
    # 0.4 is an outlier and should be capped at 0.345
    df_capped = cap_outliers_iqr(df_no_nan, ["volatile acidity"], factor=1.5)
    assert df_capped.loc[4, "volatile acidity"] == pytest.approx(0.345)

def test_create_features(mock_raw_df):
    df_feat = create_features(mock_raw_df)
    # Check new columns exist
    assert "total_acidity" in df_feat.columns
    assert "acid_to_sugar_ratio" in df_feat.columns
    assert "bound_sulfur_dioxide" in df_feat.columns
    
    # Verify bound_sulfur_dioxide value (170 - 45 = 125)
    assert df_feat.loc[0, "bound_sulfur_dioxide"] == 125.0

def test_encode_type(mock_raw_df):
    df_enc = encode_type(mock_raw_df)
    assert "is_red" in df_enc.columns
    assert "type" not in df_enc.columns
    # white -> 0, red -> 1
    assert df_enc.loc[0, "is_red"] == 0
    assert df_enc.loc[3, "is_red"] == 1

def test_scale_features():
    train_df = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0, 5.0]})
    test_df = pd.DataFrame({"feat": [2.0, 6.0]})
    
    scaled_train, scaled_test, scaler = scale_features(train_df, test_df)
    
    assert isinstance(scaler, StandardScaler)
    # Mean of scaled_train should be 0, std should be 1
    assert np.mean(scaled_train["feat"]) == pytest.approx(0.0, abs=1e-7)
    assert np.std(scaled_train["feat"]) == pytest.approx(1.0, abs=1e-7)
    
    # Verify we can also transform test set
    assert scaled_test.loc[0, "feat"] == scaled_train.loc[1, "feat"]
