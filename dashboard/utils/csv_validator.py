"""Validation utilities for batch CSV uploads to PredictIQ API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

MAX_BATCH_SIZE: int = 500

REQUIRED_FEATURE_COLUMNS: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def validate_batch_dataframe(
    df: pd.DataFrame,
) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[List[Optional[str]]], Optional[str]]:
    """Validate uploaded DataFrame against API batch schema.

    Args:
        df: Input pandas DataFrame parsed from uploaded CSV.

    Returns:
        Tuple:
            - is_valid (bool): True if valid, False otherwise.
            - records (List[Dict] | None): Cleaned list of record dictionaries for /predict/batch.
            - customer_ids (List[str | None] | None): Extracted customer identifiers for row association.
            - error_message (str | None): Descriptive failure message if invalid.
    """
    if df is None or df.empty:
        return False, None, None, "Uploaded CSV file is empty. Please upload a file containing customer records."

    if len(df) > MAX_BATCH_SIZE:
        return (
            False,
            None,
            None,
            f"Batch size exceeds maximum limit of {MAX_BATCH_SIZE} records. "
            f"The uploaded file contains {len(df):,} records.",
        )

    # Check for missing required columns (case-sensitive check with fallback)
    missing_cols = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        return (
            False,
            None,
            None,
            f"Missing {len(missing_cols)} required column(s): {', '.join(missing_cols)}. "
            "Please ensure all required customer feature columns are present.",
        )

    # Extract customerID for row association if present
    customer_ids: List[Optional[str]] = []
    if "customerID" in df.columns:
        customer_ids = [
            str(val).strip() if pd.notna(val) and str(val).strip() != "" else None
            for val in df["customerID"]
        ]
    else:
        customer_ids = [None] * len(df)

    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        try:
            # Tenure validation
            tenure_val = int(row["tenure"])
            if tenure_val < 0:
                return (
                    False,
                    None,
                    None,
                    f"Row {idx + 1}: 'tenure' must be non-negative (got {tenure_val}).",
                )

            # MonthlyCharges validation
            monthly_val = float(row["MonthlyCharges"])
            if monthly_val < 0.0:
                return (
                    False,
                    None,
                    None,
                    f"Row {idx + 1}: 'MonthlyCharges' must be non-negative (got {monthly_val}).",
                )

            # TotalCharges validation
            raw_tc = row["TotalCharges"]
            if pd.isna(raw_tc) or str(raw_tc).strip() == "":
                total_val: float = 0.0
            else:
                total_val = float(str(raw_tc).strip())
                if total_val < 0.0:
                    return (
                        False,
                        None,
                        None,
                        f"Row {idx + 1}: 'TotalCharges' must be non-negative (got {total_val}).",
                    )

            # SeniorCitizen validation
            senior_val = int(row["SeniorCitizen"])
            if senior_val not in (0, 1):
                senior_val = 1 if senior_val > 0 else 0

            record: Dict[str, Any] = {
                "gender": str(row["gender"]).strip(),
                "SeniorCitizen": senior_val,
                "Partner": str(row["Partner"]).strip(),
                "Dependents": str(row["Dependents"]).strip(),
                "tenure": tenure_val,
                "PhoneService": str(row["PhoneService"]).strip(),
                "MultipleLines": str(row["MultipleLines"]).strip(),
                "InternetService": str(row["InternetService"]).strip(),
                "OnlineSecurity": str(row["OnlineSecurity"]).strip(),
                "OnlineBackup": str(row["OnlineBackup"]).strip(),
                "DeviceProtection": str(row["DeviceProtection"]).strip(),
                "TechSupport": str(row["TechSupport"]).strip(),
                "StreamingTV": str(row["StreamingTV"]).strip(),
                "StreamingMovies": str(row["StreamingMovies"]).strip(),
                "Contract": str(row["Contract"]).strip(),
                "PaperlessBilling": str(row["PaperlessBilling"]).strip(),
                "PaymentMethod": str(row["PaymentMethod"]).strip(),
                "MonthlyCharges": monthly_val,
                "TotalCharges": total_val,
            }

            if customer_ids[idx] is not None:
                record["customerID"] = customer_ids[idx]

            records.append(record)

        except (ValueError, TypeError) as err:
            return (
                False,
                None,
                None,
                f"Row {idx + 1} contains invalid numeric data: {err}",
            )

    return True, records, customer_ids, None


def generate_sample_batch_csv() -> str:
    """Generate a valid sample CSV string for users to download as template."""
    sample_data = pd.DataFrame([
        {
            "customerID": "7590-VHVEG",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85,
        },
        {
            "customerID": "5575-GNVDE",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 34,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 56.95,
            "TotalCharges": 1889.50,
        },
        {
            "customerID": "3668-QPYBK",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 53.85,
            "TotalCharges": 108.15,
        },
    ])
    return sample_data.to_csv(index=False)
