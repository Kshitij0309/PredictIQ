"""Deterministic tests for hyperparameter optimization module."""

import numpy as np
import pandas as pd
import pytest

from predictiq.models.tuning import (
    TuningResult,
    get_hyperparameter_search_spaces,
    tune_model,
)


@pytest.fixture
def synthetic_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate small synthetic training set for fast, deterministic tuning tests."""
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        "gender": ["Female", "Male"] * 40,
        "SeniorCitizen": [0, 1] * 40,
        "Partner": ["Yes", "No"] * 40,
        "Dependents": ["No", "Yes"] * 40,
        "tenure": list(range(n)),
        "PhoneService": ["Yes"] * n,
        "MultipleLines": ["No", "Yes"] * 40,
        "InternetService": ["DSL", "Fiber optic"] * 40,
        "OnlineSecurity": ["No", "Yes"] * 40,
        "OnlineBackup": ["Yes", "No"] * 40,
        "DeviceProtection": ["No", "Yes"] * 40,
        "TechSupport": ["No", "Yes"] * 40,
        "StreamingTV": ["No", "Yes"] * 40,
        "StreamingMovies": ["No", "Yes"] * 40,
        "Contract": ["Month-to-month", "One year"] * 40,
        "PaperlessBilling": ["Yes", "No"] * 40,
        "PaymentMethod": ["Electronic check", "Mailed check"] * 40,
        "MonthlyCharges": [30.0 + i * 0.8 for i in range(n)],
        "TotalCharges": [100.0 + i * 25.0 for i in range(n)],
    })
    y = pd.Series([0] * 58 + [1] * 22)
    return df, y


def test_search_spaces_definition() -> None:
    """Verify defined search spaces contain LightGBM, Random Forest, and Logistic Regression."""
    spaces = get_hyperparameter_search_spaces()
    assert "LightGBM" in spaces
    assert "Random Forest" in spaces
    assert "Logistic Regression" in spaces

    for name, (pipe, grid) in spaces.items():
        assert pipe is not None
        assert len(grid) > 0


def test_tune_model_logistic_regression(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify tuning executes cleanly on Logistic Regression and returns TuningResult."""
    X, y = synthetic_training_data
    result: TuningResult = tune_model(
        model_name="Logistic Regression",
        X_train=X,
        y_train=y,
        n_iter=3,
        cv_folds=3,
        random_state=42,
    )

    assert result.model_name == "Logistic Regression"
    assert result.best_estimator is not None
    assert 0.0 <= result.best_cv_score <= 1.0
    assert result.cv_folds == 3
    assert result.runtime_seconds > 0.0
    assert "classifier__C" in result.best_params


def test_tune_model_unknown_candidate(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify ValueError is raised for unrecognized candidate name."""
    X, y = synthetic_training_data
    with pytest.raises(ValueError, match="Unknown model_name"):
        tune_model(
            model_name="NonExistentModel",
            X_train=X,
            y_train=y,
        )
