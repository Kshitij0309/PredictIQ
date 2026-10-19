"""Data loading, caching, and aggregation service for customer analytics.

Provides deterministic path resolution via PREDICTIQ_DATA_DIR and PREDICTIQ_ROOT_DIR,
canonical cleaned dataset prioritization, schema validation, and defensive fallback
to raw dataset without creating synthetic data.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Required columns for customer analytics aggregation and filtering
REQUIRED_ANALYTICS_COLUMNS: List[str] = [
    "Churn",
    "Contract",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "InternetService",
    "PaymentMethod",
]


def _find_repo_root() -> Path:
    """Resolve repository root directory from environment or file location."""
    env_root = os.getenv("PREDICTIQ_ROOT_DIR") or os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).is_dir():
        return Path(env_root).resolve()
    # dashboard/services/data_service.py -> 3 levels up to root
    return Path(__file__).resolve().parent.parent.parent


def _resolve_data_dir() -> Path:
    """Resolve data directory from environment or fallback to repo root."""
    env_data_dir = os.getenv("PREDICTIQ_DATA_DIR") or os.getenv("DATA_DIR")
    if env_data_dir:
        p = Path(env_data_dir).resolve()
        if p.is_dir():
            return p

    root = _find_repo_root()
    return (root / "data").resolve()


def get_dataset_paths() -> List[Path]:
    """Return candidate paths for the customer churn dataset in strict priority order.

    Priority order:
    1. Canonical cleaned dataset in configured data_dir (interim/Telco-Customer-Churn-cleaned.csv)
    2. Cleaned dataset directly in data_dir (Telco-Customer-Churn-cleaned.csv)
    3. Raw dataset fallback in data_dir (raw/Telco-Customer-Churn.csv)
    4. Raw dataset directly in data_dir (Telco-Customer-Churn.csv)
    5. Fallback paths from repository root (if data_dir differed)
    """
    data_dir = _resolve_data_dir()
    repo_root = _find_repo_root()
    root_data_dir = (repo_root / "data").resolve()

    candidates: List[Path] = [
        (data_dir / "interim" / "Telco-Customer-Churn-cleaned.csv").resolve(),
        (data_dir / "Telco-Customer-Churn-cleaned.csv").resolve(),
        (data_dir / "raw" / "Telco-Customer-Churn.csv").resolve(),
        (data_dir / "Telco-Customer-Churn.csv").resolve(),
    ]

    if root_data_dir != data_dir:
        candidates.extend([
            (root_data_dir / "interim" / "Telco-Customer-Churn-cleaned.csv").resolve(),
            (root_data_dir / "Telco-Customer-Churn-cleaned.csv").resolve(),
            (root_data_dir / "raw" / "Telco-Customer-Churn.csv").resolve(),
            (root_data_dir / "Telco-Customer-Churn.csv").resolve(),
        ])

    # De-duplicate while preserving strict priority order
    seen: set[Path] = set()
    unique_candidates: List[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    return unique_candidates


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Validate that the loaded customer dataframe contains required columns and valid target."""
    if df.empty:
        return False, "Dataset file is completely empty."

    missing = [c for c in REQUIRED_ANALYTICS_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Dataset is missing required analytics columns: {missing}"

    # Target variable validation: ensure Churn has binary values
    churn_unique = set(df["Churn"].dropna().astype(str).str.strip().str.lower().unique())
    if not churn_unique.intersection({"yes", "no", "1", "0", "true", "false"}):
        return False, f"Target column 'Churn' lacks expected binary values (found: {churn_unique})"

    return True, None


def load_customer_data(file_path: Optional[str | Path] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and validate the IBM Telco customer dataset from disk.

    Args:
        file_path: Optional custom path to CSV file.

    Returns:
        Tuple of (df, metadata)
        - df: Loaded pandas DataFrame (empty if missing or invalid)
        - metadata: Dictionary indicating availability, source path, record count, and schema details.
    """
    if file_path:
        candidates = [Path(file_path).resolve()]
    else:
        candidates = get_dataset_paths()

    checked_paths = [str(p) for p in candidates]

    for p in candidates:
        if p and p.is_file():
            try:
                df = pd.read_csv(p)

                # Validate schema before adopting this CSV
                is_valid, validation_error = validate_dataset_schema(df)
                if not is_valid:
                    logger.warning(f"File {p} failed schema validation: {validation_error}")
                    continue

                # Ensure TotalCharges is numeric (coercing blank strings from raw data to 0.0)
                if "TotalCharges" in df.columns:
                    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

                # Ensure tenure_cohort exists for analytics
                if "tenure_cohort" not in df.columns and "tenure" in df.columns:
                    bins = [-1, 12, 24, 48, 72]
                    labels = ["0-12 mos", "13-24 mos", "25-48 mos", "49-72 mos"]
                    df["tenure_cohort"] = pd.cut(df["tenure"], bins=bins, labels=labels)

                # Standardize Churn column to canonical 'Yes'/'No' string
                if "Churn" in df.columns:
                    churn_series = df["Churn"].astype(str).str.strip()
                    df["Churn"] = churn_series.map(
                        lambda x: "Yes"
                        if x.lower() in ("yes", "1", "true")
                        else ("No" if x.lower() in ("no", "0", "false") else x)
                    )

                is_canonical = "cleaned" in p.name.lower() or "interim" in str(p).lower()

                return df, {
                    "available": True,
                    "source_path": str(p),
                    "total_records": len(df),
                    "columns": list(df.columns),
                    "dataset_type": "canonical_cleaned" if is_canonical else "raw_fallback",
                    "churn_distribution": df["Churn"].value_counts().to_dict(),
                }
            except Exception as err:
                logger.error(f"Error reading dataset from {p}: {err}")
                continue

    error_msg = (
        f"Dataset not found. Checked candidate paths: {checked_paths}. "
        f"Configured PREDICTIQ_DATA_DIR='{os.getenv('PREDICTIQ_DATA_DIR', '')}', "
        f"PREDICTIQ_ROOT_DIR='{os.getenv('PREDICTIQ_ROOT_DIR', '')}'."
    )
    return pd.DataFrame(), {
        "available": False,
        "error": error_msg,
        "checked_paths": checked_paths,
        "total_records": 0,
    }


def compute_overview_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level customer and churn key performance indicators.

    Args:
        df: Input customer DataFrame.

    Returns:
        Dictionary of overview metrics.
    """
    if df.empty or "Churn" not in df.columns:
        return {
            "total_customers": 0,
            "churn_count": 0,
            "churn_pct": 0.0,
            "retained_count": 0,
            "retained_pct": 0.0,
            "avg_monthly_charges": 0.0,
            "total_monthly_revenue": 0.0,
            "median_tenure": 0.0,
        }

    total = len(df)
    churn_mask = df["Churn"].isin(["Yes", "1", 1])
    churn_count = int(churn_mask.sum())
    retained_count = total - churn_count

    churn_pct = (churn_count / total * 100.0) if total > 0 else 0.0
    retained_pct = (retained_count / total * 100.0) if total > 0 else 0.0

    avg_monthly = float(df["MonthlyCharges"].mean()) if "MonthlyCharges" in df.columns else 0.0
    total_monthly = float(df["MonthlyCharges"].sum()) if "MonthlyCharges" in df.columns else 0.0
    median_tenure = float(df["tenure"].median()) if "tenure" in df.columns else 0.0

    return {
        "total_customers": total,
        "churn_count": churn_count,
        "churn_pct": churn_pct,
        "retained_count": retained_count,
        "retained_pct": retained_pct,
        "avg_monthly_charges": avg_monthly,
        "total_monthly_revenue": total_monthly,
        "median_tenure": median_tenure,
    }


def compute_breakdown(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "Churn",
) -> pd.DataFrame:
    """Compute customer volume, churn count, and churn rate by category."""
    if df.empty or group_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame()

    df_work = df.copy()
    df_work["is_churn"] = df_work[target_col].isin(["Yes", "1", 1]).astype(int)

    summary = (
        df_work.groupby(group_col, observed=True)
        .agg(
            total_customers=("is_churn", "count"),
            churn_count=("is_churn", "sum"),
        )
        .reset_index()
    )
    summary["retained_count"] = summary["total_customers"] - summary["churn_count"]
    summary["churn_rate_pct"] = (summary["churn_count"] / summary["total_customers"] * 100.0).round(1)
    summary["retained_rate_pct"] = (summary["retained_count"] / summary["total_customers"] * 100.0).round(1)

    return summary.sort_values(by="total_customers", ascending=False)


def filter_dataset(
    df: pd.DataFrame,
    contracts: Optional[List[str]] = None,
    internet_services: Optional[List[str]] = None,
    payment_methods: Optional[List[str]] = None,
    senior_citizen: Optional[str] = None,
    partner: Optional[str] = None,
    dependents: Optional[str] = None,
    tenure_range: Optional[Tuple[int, int]] = None,
    monthly_charges_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Filter customer DataFrame by user-selected attributes."""
    if df.empty:
        return df

    filtered = df.copy()

    if contracts and "Contract" in filtered.columns:
        filtered = filtered[filtered["Contract"].isin(contracts)]

    if internet_services and "InternetService" in filtered.columns:
        filtered = filtered[filtered["InternetService"].isin(internet_services)]

    if payment_methods and "PaymentMethod" in filtered.columns:
        filtered = filtered[filtered["PaymentMethod"].isin(payment_methods)]

    if senior_citizen and senior_citizen != "All" and "SeniorCitizen" in filtered.columns:
        val = 1 if senior_citizen == "Yes" else 0
        filtered = filtered[filtered["SeniorCitizen"] == val]

    if partner and partner != "All" and "Partner" in filtered.columns:
        filtered = filtered[filtered["Partner"] == partner]

    if dependents and dependents != "All" and "Dependents" in filtered.columns:
        filtered = filtered[filtered["Dependents"] == dependents]

    if tenure_range and "tenure" in filtered.columns:
        low, high = tenure_range
        filtered = filtered[(filtered["tenure"] >= low) & (filtered["tenure"] <= high)]

    if monthly_charges_range and "MonthlyCharges" in filtered.columns:
        low_m, high_m = monthly_charges_range
        filtered = filtered[(filtered["MonthlyCharges"] >= low_m) & (filtered["MonthlyCharges"] <= high_m)]

    return filtered
