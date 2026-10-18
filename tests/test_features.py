"""Deterministic tests for feature engineering transformers and preprocessing pipelines."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from predictiq.features.pipeline import build_model_pipeline, create_preprocessor
from predictiq.features.transformers import DomainFeatureEngineer


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Sample DataFrame with raw feature columns."""
    return pd.DataFrame({
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "No"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [1, 24, 60],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 95.50, 20.05],
        "TotalCharges": [29.85, 2292.0, 1203.0],
    })


def test_domain_feature_engineer_columns_added(sample_feature_df: pd.DataFrame) -> None:
    """Verify that expected engineered features are generated."""
    engineer = DomainFeatureEngineer()
    engineer.fit(sample_feature_df)
    transformed = engineer.transform(sample_feature_df)

    expected_cols = [
        "service_count",
        "streaming_services_count",
        "has_tech_support",
        "has_online_security",
        "is_month_to_month",
        "is_fiber_optic",
        "is_electronic_check",
        "tenure_group",
    ]
    for col in expected_cols:
        assert col in transformed.columns, f"Missing engineered column: {col}"

    assert len(transformed) == len(sample_feature_df)


def test_domain_feature_engineer_service_counts(sample_feature_df: pd.DataFrame) -> None:
    """Verify exact service count logic."""
    engineer = DomainFeatureEngineer()
    transformed = engineer.transform(sample_feature_df)

    # Row 0: Phone(1) + OnlineBackup(1) + DSL(1) = 3 active services
    assert transformed.loc[0, "service_count"] == 3
    assert transformed.loc[0, "has_tech_support"] == 0
    assert transformed.loc[0, "is_month_to_month"] == 1
    assert transformed.loc[0, "is_electronic_check"] == 1

    # Row 1: Phone(1) + Multi(1) + Fiber(1) + Sec(1) + Dev(1) + Tech(1) + TV(1) + Mov(1) = 8
    assert transformed.loc[1, "service_count"] == 8
    assert transformed.loc[1, "has_tech_support"] == 1
    assert transformed.loc[1, "is_month_to_month"] == 0
    assert transformed.loc[1, "streaming_services_count"] == 2


def test_domain_feature_engineer_type_error_on_array() -> None:
    """Raise TypeError if raw numpy array is passed."""
    engineer = DomainFeatureEngineer()
    arr = np.array([[1, 2], [3, 4]])
    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
        engineer.transform(arr)


def test_create_preprocessor_unseen_category_handling(sample_feature_df: pd.DataFrame) -> None:
    """Verify that unseen categorical levels during transform are safely ignored."""
    engineer = DomainFeatureEngineer()
    train_eng = engineer.fit_transform(sample_feature_df)

    preprocessor = create_preprocessor(scale_numeric=True)
    preprocessor.fit(train_eng)

    # Create test row with novel/unseen categorical values
    unseen_df = sample_feature_df.copy()
    unseen_df.loc[0, "PaymentMethod"] = "Cryptocurrency Token"
    unseen_df.loc[0, "Contract"] = "Decade Plan"
    unseen_eng = engineer.transform(unseen_df)

    # Should transform without raising errors or creating NaNs
    transformed = preprocessor.transform(unseen_eng)
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == len(unseen_df)


def test_create_preprocessor_missing_value_imputation(sample_feature_df: pd.DataFrame) -> None:
    """Verify missing values are imputed using training medians and modes."""
    engineer = DomainFeatureEngineer()
    train_eng = engineer.fit_transform(sample_feature_df)

    preprocessor = create_preprocessor(scale_numeric=True)
    preprocessor.fit(train_eng)

    # Introduce NaNs in test row
    nan_df = sample_feature_df.copy()
    nan_df.loc[0, "MonthlyCharges"] = np.nan
    nan_df.loc[0, "InternetService"] = np.nan
    nan_eng = engineer.transform(nan_df)

    transformed = preprocessor.transform(nan_eng)
    assert not np.isnan(transformed).any()


def test_build_model_pipeline_end_to_end(sample_feature_df: pd.DataFrame) -> None:
    """Verify end-to-end pipeline fit and predict."""
    y = pd.Series([0, 1, 0])
    pipeline = build_model_pipeline(DummyClassifier(strategy="most_frequent"))

    pipeline.fit(sample_feature_df, y)
    preds = pipeline.predict(sample_feature_df)

    assert len(preds) == len(sample_feature_df)
    assert set(preds).issubset({0, 1})
