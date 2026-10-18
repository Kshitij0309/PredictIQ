"""Deterministic unit tests for data validation engine."""

import numpy as np
import pandas as pd
import pytest

from predictiq.data.validation import (
    validate_dataset,
)


@pytest.fixture
def valid_df() -> pd.DataFrame:
    """Produce a small, valid synthetic Telco Churn DataFrame."""
    return pd.DataFrame({
        "customerID": ["0001-TEST", "0002-TEST", "0003-TEST"],
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "No"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [12, 24, 6],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service"],
        "DeviceProtection": ["No", "No", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service"],
        "StreamingTV": ["No", "No", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)"],
        "MonthlyCharges": [29.85, 75.50, 20.00],
        "TotalCharges": ["358.20", "1812.00", "120.00"],
        "Churn": ["No", "Yes", "No"],
    })


class TestSchemaValidation:
    """Validate schema compliance checks."""

    def test_valid_dataset_passes(self, valid_df: pd.DataFrame):
        report = validate_dataset(valid_df)
        assert report.is_valid is True
        assert report.total_rows == 3
        assert report.total_columns == 21
        assert len(report.missing_columns) == 0
        assert report.target_distribution == {"No": 2, "Yes": 1}

    def test_missing_required_columns_fails(self, valid_df: pd.DataFrame):
        df_missing = valid_df.drop(columns=["tenure", "Churn"])
        report = validate_dataset(df_missing)
        assert report.is_valid is False
        assert "tenure" in report.missing_columns
        assert "Churn" in report.missing_columns
        critical_rules = [i.rule for i in report.issues if i.severity == "CRITICAL"]
        assert "SCHEMA_MISSING_COLUMNS" in critical_rules

    def test_unexpected_columns_warning(self, valid_df: pd.DataFrame):
        df_extra = valid_df.copy()
        df_extra["extra_column"] = 123
        report = validate_dataset(df_extra)
        assert report.is_valid is True  # Warning does not invalidate dataset
        assert "extra_column" in report.unexpected_columns
        rules = [i.rule for i in report.issues if i.severity == "WARNING"]
        assert "SCHEMA_UNEXPECTED_COLUMNS" in rules


class TestIntegrityAndConstraints:
    """Validate data integrity, duplicate IDs, target labels, and bounds."""

    def test_duplicate_identifiers_fails(self, valid_df: pd.DataFrame):
        df_dup = valid_df.copy()
        df_dup.loc[1, "customerID"] = df_dup.loc[0, "customerID"]
        report = validate_dataset(df_dup)
        assert report.is_valid is False
        assert report.duplicate_id_count == 1
        critical_rules = [i.rule for i in report.issues if i.severity == "CRITICAL"]
        assert "DUPLICATE_IDENTIFIERS" in critical_rules

    def test_invalid_target_labels_fails(self, valid_df: pd.DataFrame):
        df_invalid = valid_df.copy()
        df_invalid.loc[0, "Churn"] = "Maybe"
        report = validate_dataset(df_invalid)
        assert report.is_valid is False
        critical_rules = [i.rule for i in report.issues if i.severity == "CRITICAL"]
        assert "INVALID_TARGET_LABELS" in critical_rules

    def test_missing_target_values_fails(self, valid_df: pd.DataFrame):
        df_missing_target = valid_df.copy()
        df_missing_target.loc[0, "Churn"] = np.nan
        report = validate_dataset(df_missing_target)
        assert report.is_valid is False
        critical_rules = [i.rule for i in report.issues if i.severity == "CRITICAL"]
        assert "MISSING_TARGET_VALUES" in critical_rules

    def test_whitespace_as_null_reporting(self, valid_df: pd.DataFrame):
        df_ws = valid_df.copy()
        df_ws.loc[0, "TotalCharges"] = "   "
        report = validate_dataset(df_ws)
        assert report.whitespace_values_per_column["TotalCharges"] == 1
        rules = [i.rule for i in report.issues]
        assert "WHITESPACE_AS_NULL" in rules

    def test_negative_tenure_fails(self, valid_df: pd.DataFrame):
        df_neg = valid_df.copy()
        df_neg.loc[0, "tenure"] = -1
        report = validate_dataset(df_neg)
        assert report.is_valid is False
        critical_rules = [i.rule for i in report.issues if i.severity == "CRITICAL"]
        assert "NUMERICAL_OUT_OF_BOUNDS" in critical_rules

    def test_negative_charges_fails(self, valid_df: pd.DataFrame):
        df_neg = valid_df.copy()
        df_neg.loc[0, "MonthlyCharges"] = -15.0
        report = validate_dataset(df_neg)
        assert report.is_valid is False
        critical_rules = [i.rule for i in report.issues if i.severity == "CRITICAL"]
        assert "NUMERICAL_OUT_OF_BOUNDS" in critical_rules


class TestDomainConsistency:
    """Validate domain business rules."""

    def test_phone_service_inconsistency_warning(self, valid_df: pd.DataFrame):
        df_phone = valid_df.copy()
        # PhoneService='No' but MultipleLines='Yes'
        df_phone.loc[2, "MultipleLines"] = "Yes"
        report = validate_dataset(df_phone)
        warning_rules = [i.rule for i in report.issues if i.severity == "WARNING"]
        assert "DOMAIN_INCONSISTENCY_PHONE" in warning_rules

    def test_internet_service_inconsistency_warning(self, valid_df: pd.DataFrame):
        df_net = valid_df.copy()
        # InternetService='No' but OnlineSecurity='Yes'
        df_net.loc[2, "OnlineSecurity"] = "Yes"
        report = validate_dataset(df_net)
        warning_rules = [i.rule for i in report.issues if i.severity == "WARNING"]
        assert "DOMAIN_INCONSISTENCY_ONLINESECURITY" in warning_rules


class TestReportFormatting:
    """Validate report serialization and markdown presentation."""

    def test_report_dict_and_markdown(self, valid_df: pd.DataFrame):
        report = validate_dataset(valid_df)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["is_valid"] is True
        assert d["total_rows"] == 3

        md = report.to_markdown()
        assert "# PredictIQ Dataset Validation Report" in md
        assert "73.46%" not in md  # 2/3 is 66.67%
        assert "66.67%" in md
