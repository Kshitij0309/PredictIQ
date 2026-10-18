"""Data cleaning and type normalization for PredictIQ.

Normalizes raw data types, addresses hidden missing values (e.g. whitespace in TotalCharges),
and produces validated interim data matrices for exploratory analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from predictiq.config import get_config
from predictiq.data.validation import validate_dataset

logger = logging.getLogger(__name__)


def clean_telco_churn_data(
    df: pd.DataFrame,
    impute_total_charges: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Clean and normalize raw Telco customer churn dataframe.

    Operations:
    1. Strips leading/trailing whitespace from string fields.
    2. Identifies blank strings in TotalCharges.
    3. If impute_total_charges=True, sets TotalCharges=0.0 where tenure=0 (new accounts).
    4. Casts TotalCharges to float64 and tenure to int64.
    5. Normalizes SeniorCitizen into a binary flag or category.
    6. Adds tenure_cohort category for exploratory analysis.

    Returns:
        Tuple[pd.DataFrame, dict]: Cleaned dataframe and summary of cleaning actions.
    """
    cleaned = df.copy()
    actions: dict = {
        "original_rows": len(cleaned),
        "total_charges_imputed_count": 0,
        "types_converted": {},
    }

    # 1. Strip whitespace from all string columns
    str_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    # 2. Address TotalCharges whitespace
    if "TotalCharges" in cleaned.columns:
        whitespace_mask = cleaned["TotalCharges"] == ""
        ws_count = int(whitespace_mask.sum())
        actions["total_charges_imputed_count"] = ws_count

        if impute_total_charges and ws_count > 0:
            logger.info(f"Imputing {ws_count} blank TotalCharges records with 0.0 (tenure=0 new accounts)")
            cleaned.loc[whitespace_mask, "TotalCharges"] = "0.0"

        # Convert to float
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
        actions["types_converted"]["TotalCharges"] = "float64"

    # 3. Numeric types
    if "tenure" in cleaned.columns:
        cleaned["tenure"] = pd.to_numeric(cleaned["tenure"], errors="coerce").fillna(0).astype(int)
        actions["types_converted"]["tenure"] = "int64"

    if "MonthlyCharges" in cleaned.columns:
        cleaned["MonthlyCharges"] = pd.to_numeric(cleaned["MonthlyCharges"], errors="coerce").astype(float)
        actions["types_converted"]["MonthlyCharges"] = "float64"

    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = pd.to_numeric(cleaned["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
        actions["types_converted"]["SeniorCitizen"] = "int64"

    # 4. Tenure cohorts for EDA
    if "tenure" in cleaned.columns:
        bins = [-1, 12, 24, 48, 72]
        labels = ["0-12 mos", "13-24 mos", "25-48 mos", "49-72 mos"]
        cleaned["tenure_cohort"] = pd.cut(cleaned["tenure"], bins=bins, labels=labels)
        actions["added_columns"] = ["tenure_cohort"]

    return cleaned, actions


def process_raw_to_interim(
    raw_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
) -> Path:
    """Read raw dataset, validate, clean, and write to data/interim/."""
    config = get_config()
    raw_file = Path(raw_path) if raw_path else (config.paths.raw_data_dir / "Telco-Customer-Churn.csv")
    out_file = (
        Path(output_path)
        if output_path
        else (config.paths.interim_data_dir / "Telco-Customer-Churn-cleaned.csv")
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not raw_file.is_file():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_file}")

    # Validate raw data before cleaning
    report = validate_dataset(raw_file)
    logger.info(f"Raw validation status: {report.summary()}")

    raw_df = pd.read_csv(raw_file)
    cleaned_df, actions = clean_telco_churn_data(raw_df)

    # Save to interim as csv
    cleaned_df.to_csv(out_file, index=False)

    # Save parquet if pyarrow/fastparquet engine is available
    try:
        parquet_out = out_file.with_suffix(".parquet")
        cleaned_df.to_parquet(parquet_out, index=False)
        logger.info(f"Parquet interim copy saved: {parquet_out}")
    except (ImportError, ValueError, Exception):
        pass

    logger.info(f"Cleaned dataset saved to: {out_file} ({len(cleaned_df)} rows)")
    return out_file
