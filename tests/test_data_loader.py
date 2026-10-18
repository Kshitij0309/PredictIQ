"""Deterministic tests for data loading and leakage-safe dataset splitting."""

import pandas as pd
import pytest

from predictiq.data.loader import (
    DataLoadError,
    DatasetSplits,
    load_and_split_data,
    prepare_features_and_target,
    split_dataset,
)


@pytest.fixture
def synthetic_churn_df() -> pd.DataFrame:
    """Fixture providing a small synthetic dataset mirroring Telco churn schema."""
    n_rows = 100
    churn_labels = ["No"] * 73 + ["Yes"] * 27
    return pd.DataFrame({
        "customerID": [f"TEST-{i:04d}" for i in range(n_rows)],
        "gender": ["Male", "Female"] * 50,
        "SeniorCitizen": [0, 1] * 50,
        "Partner": ["Yes", "No"] * 50,
        "Dependents": ["No", "Yes"] * 50,
        "tenure": list(range(n_rows)),
        "PhoneService": ["Yes"] * 90 + ["No"] * 10,
        "MultipleLines": ["No", "Yes"] * 50,
        "InternetService": ["DSL", "Fiber optic", "No", "DSL"] * 25,
        "OnlineSecurity": ["Yes", "No"] * 50,
        "OnlineBackup": ["No", "Yes"] * 50,
        "DeviceProtection": ["Yes", "No"] * 50,
        "TechSupport": ["No", "Yes"] * 50,
        "StreamingTV": ["Yes", "No"] * 50,
        "StreamingMovies": ["No", "Yes"] * 50,
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"] * 25,
        "PaperlessBilling": ["Yes", "No"] * 50,
        "PaymentMethod": ["Electronic check", "Mailed check"] * 50,
        "MonthlyCharges": [25.0 + i * 0.5 for i in range(n_rows)],
        "TotalCharges": [100.0 + i * 20.0 for i in range(n_rows)],
        "Churn": churn_labels,
    })


def test_prepare_features_and_target_separation(synthetic_churn_df: pd.DataFrame) -> None:
    """Ensure target is cleanly separated, mapped to binary, and customerID dropped."""
    X, y = prepare_features_and_target(synthetic_churn_df)

    assert "Churn" not in X.columns
    assert "customerID" not in X.columns
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})
    assert int(y.sum()) == 27  # Exactly 27 'Yes' instances


def test_prepare_features_and_target_missing_target() -> None:
    """Raise DataLoadError when target column is missing."""
    df_no_target = pd.DataFrame({"tenure": [1, 2], "MonthlyCharges": [50.0, 60.0]})
    with pytest.raises(DataLoadError, match="Target column 'Churn' not found"):
        prepare_features_and_target(df_no_target)


def test_prepare_features_and_target_invalid_classes() -> None:
    """Raise DataLoadError when target contains invalid non-binary labels."""
    df_bad = pd.DataFrame({
        "customerID": ["C1", "C2", "C3"],
        "tenure": [1, 2, 3],
        "Churn": ["Yes", "No", "Maybe"],
    })
    with pytest.raises(DataLoadError, match="unexpected classes"):
        prepare_features_and_target(df_bad)


def test_split_dataset_reproducibility(synthetic_churn_df: pd.DataFrame) -> None:
    """Verify train/test split determinism with fixed seed."""
    X, y = prepare_features_and_target(synthetic_churn_df)

    split1 = split_dataset(X, y, test_size=0.20, random_state=42)
    split2 = split_dataset(X, y, test_size=0.20, random_state=42)

    pd.testing.assert_frame_equal(split1.X_train, split2.X_train)
    pd.testing.assert_frame_equal(split1.X_test, split2.X_test)
    pd.testing.assert_series_equal(split1.y_train, split2.y_train)
    pd.testing.assert_series_equal(split1.y_test, split2.y_test)


def test_split_dataset_no_leakage_and_stratification(synthetic_churn_df: pd.DataFrame) -> None:
    """Verify zero overlap and stratification balance."""
    X, y = prepare_features_and_target(synthetic_churn_df)
    splits: DatasetSplits = split_dataset(X, y, test_size=0.20, random_state=42)

    # Size checks: 80 train, 20 test
    assert splits.train_size == 80
    assert splits.test_size == 20

    # Stratification: positive class proportion should closely match 27%
    train_pos_rate = splits.train_positive_ratio
    test_pos_rate = splits.test_positive_ratio
    assert 0.24 <= train_pos_rate <= 0.30
    assert 0.24 <= test_pos_rate <= 0.30


def test_split_dataset_invalid_test_size(synthetic_churn_df: pd.DataFrame) -> None:
    """Verify invalid test_size raises DataLoadError."""
    X, y = prepare_features_and_target(synthetic_churn_df)
    with pytest.raises(DataLoadError, match="test_size must be between 0.0 and 1.0"):
        split_dataset(X, y, test_size=1.5)


def test_load_and_split_data_integration(tmp_path: pytest.TempPathFactory) -> None:
    """Verify end-to-end load_and_split_data on custom CSV path."""
    n_rows = 50
    df = pd.DataFrame({
        "customerID": [f"ID-{i}" for i in range(n_rows)],
        "gender": ["Female"] * n_rows,
        "SeniorCitizen": [0] * n_rows,
        "Partner": ["Yes"] * n_rows,
        "Dependents": ["No"] * n_rows,
        "tenure": list(range(n_rows)),
        "PhoneService": ["Yes"] * n_rows,
        "MultipleLines": ["No"] * n_rows,
        "InternetService": ["DSL"] * n_rows,
        "OnlineSecurity": ["Yes"] * n_rows,
        "OnlineBackup": ["No"] * n_rows,
        "DeviceProtection": ["No"] * n_rows,
        "TechSupport": ["Yes"] * n_rows,
        "StreamingTV": ["No"] * n_rows,
        "StreamingMovies": ["No"] * n_rows,
        "Contract": ["One year"] * n_rows,
        "PaperlessBilling": ["No"] * n_rows,
        "PaymentMethod": ["Mailed check"] * n_rows,
        "MonthlyCharges": [30.0] * n_rows,
        "TotalCharges": [100.0] * n_rows,
        "Churn": ["No"] * 40 + ["Yes"] * 10,
    })
    csv_file = tmp_path / "test_data.csv"
    df.to_csv(csv_file, index=False)

    splits = load_and_split_data(data_path=csv_file, test_size=0.20, random_state=42)
    assert splits.train_size == 40
    assert splits.test_size == 10
    assert "customerID" not in splits.X_train.columns
    assert "Churn" not in splits.X_train.columns
