"""Unit tests for the CSV batch validator."""

from __future__ import annotations

import io

import pandas as pd

from dashboard.utils.csv_validator import (
    MAX_BATCH_SIZE,
    REQUIRED_FEATURE_COLUMNS,
    generate_sample_batch_csv,
    validate_batch_dataframe,
)


def _make_valid_row(customer_id: str = "CID-001") -> dict:
    return {
        "customerID": customer_id,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 45.5,
        "TotalCharges": 546.0,
    }


def test_validate_batch_dataframe_valid():
    df = pd.DataFrame([_make_valid_row("C-1"), _make_valid_row("C-2")])
    is_valid, records, cids, err = validate_batch_dataframe(df)

    assert is_valid is True
    assert err is None
    assert len(records) == 2
    assert cids == ["C-1", "C-2"]
    assert records[0]["tenure"] == 12
    assert records[0]["MonthlyCharges"] == 45.5


def test_validate_batch_dataframe_missing_column():
    row = _make_valid_row()
    del row["Contract"]
    df = pd.DataFrame([row])

    is_valid, records, cids, err = validate_batch_dataframe(df)
    assert is_valid is False
    assert records is None
    assert "Missing 1 required column(s)" in err
    assert "Contract" in err


def test_validate_batch_dataframe_exceeds_max_limit():
    rows = [_make_valid_row(f"C-{i}") for i in range(MAX_BATCH_SIZE + 1)]
    df = pd.DataFrame(rows)

    is_valid, records, cids, err = validate_batch_dataframe(df)
    assert is_valid is False
    assert f"exceeds maximum limit of {MAX_BATCH_SIZE}" in err


def test_validate_batch_dataframe_empty():
    df = pd.DataFrame()
    is_valid, records, cids, err = validate_batch_dataframe(df)
    assert is_valid is False
    assert "empty" in err.lower()


def test_validate_batch_dataframe_negative_tenure():
    row = _make_valid_row()
    row["tenure"] = -5
    df = pd.DataFrame([row])

    is_valid, records, cids, err = validate_batch_dataframe(df)
    assert is_valid is False
    assert "must be non-negative" in err


def test_generate_sample_batch_csv():
    csv_str = generate_sample_batch_csv()
    assert isinstance(csv_str, str)
    df = pd.read_csv(io.StringIO(csv_str))
    assert len(df) == 3
    for col in REQUIRED_FEATURE_COLUMNS:
        assert col in df.columns
