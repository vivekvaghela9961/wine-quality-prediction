# Model Card — Wine Quality Predictor

This model card details the performance, architecture, and intent of the Wine Quality Predictor model.

## Model Details

- **Model Type**: Random Forest Regressor (Scikit-Learn)
- **Trained By**: Vivek
- **Date**: July 2026
- **Hyperparameters**:
  - `n_estimators`: 120
  - `max_depth`: 16
  - `min_samples_split`: 5
  - `min_samples_leaf`: 1
  - `random_state`: 42

## Intended Use

- **Primary Use Case**: Predict the quality score of wine (on a scale of 0 to 10) based on chemical properties.
- **Intended Users**: Winemakers, quality control labs, retailers, and researchers.
- **Out of Scope**: Evaluating non-grape wines, spirits, or predicting pricing, flavor profiles, or user preference changes.

## Factors & Feature Importance

The model utilizes 15 features (including engineered ratios and wine type encoding).

Based on SHAP interpretability analysis, the most significant drivers of predicted wine quality are:
1. **alcohol**: Positive correlation (higher alcohol is generally associated with higher quality rating).
2. **volatile acidity**: Negative correlation (high acetic acid levels decrease rating due to vinegar-like taste).
3. **free sulfur dioxide** & **total sulfur dioxide**: Crucial for freshness and preservation.
4. **sulphates**: Additives that affect SO2 levels.

## Quantitative Evaluation

The model was evaluated using a 20% test split and 5-fold cross-validation.

| Metric | Value |
| :--- | :--- |
| **Mean Absolute Error (MAE)** | 0.5115 |
| **Root Mean Squared Error (RMSE)** | 0.6655 |
| **R² Score** | 0.4109 |
| **5-Fold CV Mean RMSE** | 0.7423 (+/- 0.0361) |

## Training Data

- **Source**: UCI Wine Quality Dataset (Red and White wine).
- **Processing**:
  - Missing values handled (none present in raw dataset, but logic exists).
  - Duplicate records removed (1,177 rows dropped to avoid leakage).
  - Outliers capped using IQR thresholding.
  - New features engineered: `total_acidity`, `acid_to_sugar_ratio`, `bound_sulfur_dioxide`, and `is_red` (binary type).

## Limitations & Ethical Considerations

- **Limitations**: The model is trained on objective chemical parameters. It does not account for subjective aspects like tasting notes, bottle variation, aging potential, or temperature of service.
- **Ethics**: Wine consumption is restricted by age and health factors. Model predictions are strictly for quality control and research and should not promote irresponsible alcohol use.
