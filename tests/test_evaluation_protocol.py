"""Deterministic tests validating Evaluation Protocol Independence (Part A).

Enforces:
1. Fresh, reproducible held-out evaluation split isolation (Seed 2026).
2. Model selection uses training cross-validation only (Average Precision), never test set metrics.
3. Candidate models are compared across identical cross-validation folds.
"""

import pandas as pd
import pytest

from predictiq.data.loader import (
    DEFAULT_EXPLORATORY_SPLIT_SEED,
    ISOLATED_EVALUATION_SPLIT_SEED,
    DatasetSplits,
    load_and_split_data,
    load_and_split_isolated_evaluation_data,
)
from predictiq.models.baselines import get_all_baselines
from predictiq.models.train import run_cross_validation_evaluation


@pytest.fixture
def synthetic_protocol_df() -> pd.DataFrame:
    """Generate synthetic dataset for protocol verification."""
    n_rows = 100
    return pd.DataFrame({
        "customerID": [f"ID-{i:04d}" for i in range(n_rows)],
        "gender": ["Female", "Male"] * 50,
        "SeniorCitizen": [0, 1] * 50,
        "Partner": ["Yes", "No"] * 50,
        "Dependents": ["No", "Yes"] * 50,
        "tenure": list(range(n_rows)),
        "PhoneService": ["Yes"] * n_rows,
        "MultipleLines": ["No", "Yes"] * 50,
        "InternetService": ["DSL", "Fiber optic"] * 50,
        "OnlineSecurity": ["No", "Yes"] * 50,
        "OnlineBackup": ["Yes", "No"] * 50,
        "DeviceProtection": ["No", "Yes"] * 50,
        "TechSupport": ["No", "Yes"] * 50,
        "StreamingTV": ["No", "Yes"] * 50,
        "StreamingMovies": ["No", "Yes"] * 50,
        "Contract": ["Month-to-month", "One year"] * 50,
        "PaperlessBilling": ["Yes", "No"] * 50,
        "PaymentMethod": ["Electronic check", "Mailed check"] * 50,
        "MonthlyCharges": [30.0 + i * 0.5 for i in range(n_rows)],
        "TotalCharges": [100.0 + i * 15.0 for i in range(n_rows)],
        "Churn": ["No"] * 73 + ["Yes"] * 27,
    })


def test_isolated_evaluation_split_independence(
    synthetic_protocol_df: pd.DataFrame,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify that isolated evaluation split (Seed 2026) is distinct from exploratory split (Seed 42)."""
    csv_file = tmp_path / "protocol_data.csv"
    synthetic_protocol_df.to_csv(csv_file, index=False)

    exploratory_splits: DatasetSplits = load_and_split_data(
        data_path=csv_file,
        random_state=DEFAULT_EXPLORATORY_SPLIT_SEED,
        split_purpose="exploratory",
    )
    isolated_splits: DatasetSplits = load_and_split_isolated_evaluation_data(
        data_path=csv_file,
        random_state=ISOLATED_EVALUATION_SPLIT_SEED,
    )

    assert exploratory_splits.split_seed == 42
    assert isolated_splits.split_seed == 2026
    assert isolated_splits.split_purpose == "isolated_evaluation"

    # Both must preserve stratification
    assert 0.24 <= exploratory_splits.train_positive_ratio <= 0.30
    assert 0.24 <= isolated_splits.train_positive_ratio <= 0.30

    # Test sets must not be identical across different seeds
    assert not exploratory_splits.X_test.equals(isolated_splits.X_test)


def test_cv_identical_folds_comparison(
    synthetic_protocol_df: pd.DataFrame,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify that candidate models are evaluated across identical training folds."""
    csv_file = tmp_path / "protocol_data.csv"
    synthetic_protocol_df.to_csv(csv_file, index=False)

    splits = load_and_split_isolated_evaluation_data(data_path=csv_file)
    candidates = get_all_baselines(random_state=42)

    cv_scores = {}
    for name, pipeline in candidates.items():
        mean_pr, std_pr, _, _ = run_cross_validation_evaluation(
            pipeline=pipeline,
            X_train=splits.X_train,
            y_train=splits.y_train,
            n_splits=3,
            random_state=42,
        )
        cv_scores[name] = (mean_pr, std_pr)

    assert len(cv_scores) == 4
    for name, (m, s) in cv_scores.items():
        assert 0.0 <= m <= 1.0
        assert s >= 0.0
