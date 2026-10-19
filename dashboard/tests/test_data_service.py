"""Unit tests for the dashboard data service."""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard.services.data_service import (
    compute_breakdown,
    compute_overview_metrics,
    filter_dataset,
    load_customer_data,
)


@pytest.fixture
def sample_customer_df() -> pd.DataFrame:
    """Create a minimal synthetic customer dataframe for testing."""
    return pd.DataFrame([
        {
            "customerID": "001",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 5,
            "tenure_cohort": "0-12 mos",
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 80.0,
            "TotalCharges": 400.0,
            "Churn": "Yes",
        },
        {
            "customerID": "002",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 30,
            "tenure_cohort": "25-48 mos",
            "Contract": "One year",
            "InternetService": "DSL",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 50.0,
            "TotalCharges": 1500.0,
            "Churn": "No",
        },
        {
            "customerID": "003",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 60,
            "tenure_cohort": "49-72 mos",
            "Contract": "Two year",
            "InternetService": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 20.0,
            "TotalCharges": 1200.0,
            "Churn": "No",
        },
        {
            "customerID": "004",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 2,
            "tenure_cohort": "0-12 mos",
            "Contract": "Month-to-month",
            "InternetService": "DSL",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 40.0,
            "TotalCharges": 80.0,
            "Churn": "Yes",
        },
    ])


def test_load_customer_data_real_or_fallback():
    df, meta = load_customer_data()
    # If the real file exists in the repo, it should be loaded with > 7000 rows
    if meta["available"]:
        assert len(df) >= 7000
        assert "Churn" in df.columns
        assert "MonthlyCharges" in df.columns
    else:
        assert df.empty
        assert "error" in meta


def test_load_customer_data_nonexistent_path():
    df, meta = load_customer_data("non_existent_path_to_churn_file.csv")
    assert df.empty
    assert meta["available"] is False
    assert "not found" in meta["error"].lower()


def test_compute_overview_metrics(sample_customer_df):
    metrics = compute_overview_metrics(sample_customer_df)
    assert metrics["total_customers"] == 4
    assert metrics["churn_count"] == 2
    assert metrics["retained_count"] == 2
    assert metrics["churn_pct"] == 50.0
    assert metrics["retained_pct"] == 50.0
    assert metrics["avg_monthly_charges"] == 47.5
    assert metrics["total_monthly_revenue"] == 190.0


def test_compute_overview_metrics_empty():
    metrics = compute_overview_metrics(pd.DataFrame())
    assert metrics["total_customers"] == 0
    assert metrics["churn_count"] == 0
    assert metrics["churn_pct"] == 0.0


def test_compute_breakdown(sample_customer_df):
    contract_summary = compute_breakdown(sample_customer_df, "Contract")
    assert not contract_summary.empty
    assert "total_customers" in contract_summary.columns
    assert "churn_rate_pct" in contract_summary.columns

    # Month-to-month has 2 customers, both churned (100% churn rate)
    m2m_row = contract_summary[contract_summary["Contract"] == "Month-to-month"].iloc[0]
    assert m2m_row["total_customers"] == 2
    assert m2m_row["churn_count"] == 2
    assert m2m_row["churn_rate_pct"] == 100.0


def test_filter_dataset(sample_customer_df):
    # Filter by contract
    filtered = filter_dataset(sample_customer_df, contracts=["One year"])
    assert len(filtered) == 1
    assert filtered.iloc[0]["customerID"] == "002"

    # Filter by tenure range
    filtered_tenure = filter_dataset(sample_customer_df, tenure_range=(1, 10))
    assert len(filtered_tenure) == 2
    assert set(filtered_tenure["customerID"]) == {"001", "004"}

    # Filter by monthly charges range
    filtered_charges = filter_dataset(sample_customer_df, monthly_charges_range=(10.0, 30.0))
    assert len(filtered_charges) == 1
    assert filtered_charges.iloc[0]["customerID"] == "003"


def test_validate_dataset_schema(sample_customer_df):
    from dashboard.services.data_service import validate_dataset_schema

    # Valid dataframe
    is_valid, err = validate_dataset_schema(sample_customer_df)
    assert is_valid is True
    assert err is None

    # Empty dataframe
    is_valid, err = validate_dataset_schema(pd.DataFrame())
    assert is_valid is False
    assert "empty" in err.lower()

    # Missing required column
    incomplete_df = sample_customer_df.drop(columns=["Churn"])
    is_valid, err = validate_dataset_schema(incomplete_df)
    assert is_valid is False
    assert "missing" in err.lower()

    # Invalid non-binary churn values
    bad_churn_df = sample_customer_df.copy()
    bad_churn_df["Churn"] = ["Unknown", "Maybe", "Invalid", "N/A"]
    is_valid, err = validate_dataset_schema(bad_churn_df)
    assert is_valid is False
    assert "binary" in err.lower()


def test_load_customer_data_rejects_unrelated_csv(tmp_path):
    # Create an unrelated CSV (e.g. stock or weather data)
    unrelated_csv = tmp_path / "stocks.csv"
    pd.DataFrame({"ticker": ["AAPL", "GOOG"], "price": [150.0, 2800.0]}).to_csv(unrelated_csv, index=False)

    df, meta = load_customer_data(unrelated_csv)
    assert df.empty
    assert meta["available"] is False
    assert "not found" in meta["error"].lower()


def test_load_customer_data_with_custom_predictiq_data_dir(tmp_path, monkeypatch):
    from dashboard.services.data_service import get_dataset_paths

    fake_data_dir = tmp_path / "my_custom_data"
    fake_data_dir.mkdir()
    interim_dir = fake_data_dir / "interim"
    interim_dir.mkdir()

    # Create dummy canonical dataset
    dummy_csv = interim_dir / "Telco-Customer-Churn-cleaned.csv"
    pd.DataFrame({
        "customerID": ["C1"],
        "gender": ["Female"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [10],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["DSL"],
        "OnlineSecurity": ["Yes"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [70.0],
        "TotalCharges": [700.0],
        "Churn": ["Yes"],
    }).to_csv(dummy_csv, index=False)

    monkeypatch.setenv("PREDICTIQ_DATA_DIR", str(fake_data_dir))
    paths = get_dataset_paths()
    assert dummy_csv.resolve() in paths

    df, meta = load_customer_data()
    assert meta["available"] is True
    assert meta["dataset_type"] == "canonical_cleaned"
    assert len(df) == 1

