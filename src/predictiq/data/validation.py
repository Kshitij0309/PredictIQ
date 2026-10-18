"""Data validation engine for PredictIQ.

Performs schema verification, missing-value auditing, duplicate detection,
numerical boundary checks, and domain-specific consistency validation without
silently modifying or deleting records.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Expected canonical schema definition for IBM Telco Customer Churn
EXPECTED_COLUMNS: List[str] = [
    "customerID",
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
    "Churn",
]

TARGET_COLUMN: str = "Churn"
ALLOWED_TARGET_VALUES: Set[str] = {"Yes", "No"}
ID_COLUMN: str = "customerID"

ALLOWED_CATEGORICAL_VALUES: Dict[str, Set[Any]] = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {0, 1, "0", "1"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"No", "Yes", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"No", "Yes", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"No", "Yes", "No internet service"},
    "TechSupport": {"No", "Yes", "No internet service"},
    "StreamingTV": {"No", "Yes", "No internet service"},
    "StreamingMovies": {"No", "Yes", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}


class DataValidationError(ValueError):
    """Raised when critical validation errors prevent dataset processing."""
    pass


@dataclass
class ValidationIssue:
    """Detailed record of a data quality or schema issue."""
    severity: str  # "CRITICAL", "WARNING", "INFO"
    rule: str
    column: Optional[str]
    description: str
    affected_count: int
    recommended_action: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationReport:
    """Comprehensive, non-destructive audit report for a tabular dataset."""
    is_valid: bool
    total_rows: int
    total_columns: int
    columns_present: List[str]
    missing_columns: List[str]
    unexpected_columns: List[str]
    duplicate_id_count: int
    target_distribution: Dict[str, int]
    target_proportions: Dict[str, float]
    missing_values_per_column: Dict[str, int]
    whitespace_values_per_column: Dict[str, int]
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def has_critical_errors(self) -> bool:
        """True if any issue has CRITICAL severity."""
        return any(i.severity == "CRITICAL" for i in self.issues)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "columns_present": self.columns_present,
            "missing_columns": self.missing_columns,
            "unexpected_columns": self.unexpected_columns,
            "duplicate_id_count": self.duplicate_id_count,
            "target_distribution": self.target_distribution,
            "target_proportions": self.target_proportions,
            "missing_values_per_column": self.missing_values_per_column,
            "whitespace_values_per_column": self.whitespace_values_per_column,
            "issues": [i.to_dict() for i in self.issues],
        }

    def summary(self) -> str:
        """Concise one-line summary string."""
        status = "PASSED" if self.is_valid else "FAILED"
        return (
            f"ValidationReport({status}): {self.total_rows} rows, {self.total_columns} columns, "
            f"{len(self.issues)} issues detected ({sum(1 for i in self.issues if i.severity == 'CRITICAL')} critical)"
        )

    def to_markdown(self) -> str:
        """Format the validation results as GitHub-flavored Markdown."""
        lines = [
            "# PredictIQ Dataset Validation Report",
            "",
            f"- **Status:** {'VALID' if self.is_valid else 'INVALID'}",
            f"- **Shape:** {self.total_rows:,} rows × {self.total_columns} columns",
            f"- **Duplicates:** {self.duplicate_id_count} duplicate IDs",
            "",
            "## Target Distribution",
            "| Class | Count | Proportion |",
            "|---|---|---|",
        ]
        for label, count in self.target_distribution.items():
            prop = self.target_proportions.get(label, 0.0)
            lines.append(f"| {label} | {count:,} | {prop:.2%} |")

        lines.extend([
            "",
            "## Data Quality Issues & Recommendations",
            "| Severity | Rule | Column | Description | Count | Recommended Action |",
            "|---|---|---|---|---|---|",
        ])

        if not self.issues:
            lines.append("| INFO | - | - | Zero validation issues detected. | 0 | None |")
        else:
            for issue in self.issues:
                col = issue.column or "N/A"
                lines.append(
                    f"| **{issue.severity}** | `{issue.rule}` | `{col}` | {issue.description} | {issue.affected_count:,} | {issue.recommended_action} |"
                )

        return "\n".join(lines)


class DatasetValidator:
    """Validates schemas, data constraints, and business domain rules."""

    def __init__(
        self,
        expected_columns: Optional[List[str]] = None,
        target_column: str = TARGET_COLUMN,
        id_column: str = ID_COLUMN,
    ):
        self.expected_columns = expected_columns or EXPECTED_COLUMNS
        self.target_column = target_column
        self.id_column = id_column

    def validate(self, data: pd.DataFrame | Path | str) -> ValidationReport:
        """Run full validation suite on dataframe or CSV file path."""
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.is_file():
                raise FileNotFoundError(f"Dataset file not found at: {path}")
            df = pd.read_csv(path, dtype=str)  # Read all as string first to detect whitespace
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(f"Unsupported data input type: {type(data)}")

        issues: List[ValidationIssue] = []
        total_rows, total_cols = df.shape
        cols_present = list(df.columns)

        # 1. Schema check: missing and unexpected columns
        missing_cols = [c for c in self.expected_columns if c not in cols_present]
        unexpected_cols = [c for c in cols_present if c not in self.expected_columns]

        if missing_cols:
            issues.append(
                ValidationIssue(
                    severity="CRITICAL",
                    rule="SCHEMA_MISSING_COLUMNS",
                    column=", ".join(missing_cols),
                    description=f"Dataset is missing required columns: {missing_cols}",
                    affected_count=len(missing_cols),
                    recommended_action="Ensure raw ingestion pipeline provides complete canonical dataset.",
                )
            )

        if unexpected_cols:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    rule="SCHEMA_UNEXPECTED_COLUMNS",
                    column=", ".join(unexpected_cols),
                    description=f"Dataset contains unexpected extra columns: {unexpected_cols}",
                    affected_count=len(unexpected_cols),
                    recommended_action="Inspect upstream source for schema evolution or unwanted metadata.",
                )
            )

        # 2. Row count check
        if total_rows == 0:
            issues.append(
                ValidationIssue(
                    severity="CRITICAL",
                    rule="EMPTY_DATASET",
                    column=None,
                    description="Dataset contains 0 rows.",
                    affected_count=0,
                    recommended_action="Verify upstream data pipeline ingestion.",
                )
            )

        # 3. Duplicate ID check
        dup_count = 0
        if self.id_column in df.columns:
            dups = df[self.id_column].duplicated()
            dup_count = int(dups.sum())
            if dup_count > 0:
                issues.append(
                    ValidationIssue(
                        severity="CRITICAL",
                        rule="DUPLICATE_IDENTIFIERS",
                        column=self.id_column,
                        description=f"Found {dup_count} duplicate {self.id_column} records.",
                        affected_count=dup_count,
                        recommended_action="Deduplicate by latest customer interaction timestamp or investigate identity merging.",
                    )
                )

        # 4. Target variable validation
        target_dist: Dict[str, int] = {}
        target_prop: Dict[str, float] = {}

        if self.target_column in df.columns:
            raw_target_series = df[self.target_column].dropna().astype(str).str.strip()
            unique_targets = set(raw_target_series.unique())
            invalid_targets = unique_targets - ALLOWED_TARGET_VALUES

            if invalid_targets:
                invalid_count = int((~raw_target_series.isin(ALLOWED_TARGET_VALUES)).sum())
                issues.append(
                    ValidationIssue(
                        severity="CRITICAL",
                        rule="INVALID_TARGET_LABELS",
                        column=self.target_column,
                        description=f"Target column contains invalid labels: {invalid_targets}",
                        affected_count=invalid_count,
                        recommended_action="Normalize target labels to binary 'Yes'/'No' or investigate data corruption.",
                    )
                )

            # Target distribution
            for val, cnt in raw_target_series.value_counts().items():
                target_dist[str(val)] = int(cnt)
                target_prop[str(val)] = float(cnt / total_rows) if total_rows > 0 else 0.0

            # Target missingness check
            target_nulls = int(df[self.target_column].isna().sum())
            if target_nulls > 0:
                issues.append(
                    ValidationIssue(
                        severity="CRITICAL",
                        rule="MISSING_TARGET_VALUES",
                        column=self.target_column,
                        description=f"Found {target_nulls} records with missing target labels.",
                        affected_count=target_nulls,
                        recommended_action="Quarantine rows without targets from supervised training.",
                    )
                )

        # 5. Missing values and suspicious whitespace checks
        missing_dict: Dict[str, int] = {}
        whitespace_dict: Dict[str, int] = {}

        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            missing_dict[col] = null_count

            # Check for strings that are entirely whitespace (hidden nulls)
            str_series = series.astype(str)
            whitespace_mask = (~series.isna()) & (str_series.str.strip() == "")
            ws_count = int(whitespace_mask.sum())
            whitespace_dict[col] = ws_count

            if ws_count > 0:
                # TotalCharges with tenure=0 is a known domain finding; others are warnings
                sev = "INFO" if col == "TotalCharges" else "WARNING"
                action = (
                    "Impute TotalCharges = 0.0 for tenure=0 new accounts."
                    if col == "TotalCharges"
                    else "Strip whitespace and treat as explicit null."
                )
                issues.append(
                    ValidationIssue(
                        severity=sev,
                        rule="WHITESPACE_AS_NULL",
                        column=col,
                        description=f"Column '{col}' has {ws_count} blank/whitespace-only values.",
                        affected_count=ws_count,
                        recommended_action=action,
                    )
                )

        # 6. Categorical allowed value checks
        for col, allowed in ALLOWED_CATEGORICAL_VALUES.items():
            if col in df.columns:
                valid_mask = df[col].isin(allowed) | df[col].isna()
                # Also treat whitespace strings as already reported
                valid_mask = valid_mask | (df[col].astype(str).str.strip() == "")
                invalid_cnt = int((~valid_mask).sum())
                if invalid_cnt > 0:
                    sample_invalids = list(df.loc[~valid_mask, col].unique()[:3])
                    issues.append(
                        ValidationIssue(
                            severity="CRITICAL",
                            rule="INVALID_CATEGORICAL_VALUE",
                            column=col,
                            description=f"Column '{col}' contains unrecognized categories (e.g. {sample_invalids}).",
                            affected_count=invalid_cnt,
                            recommended_action=f"Filter or remap to permitted set: {sorted(list(allowed))}.",
                        )
                    )

        # 7. Numerical ranges & types
        # tenure: integer >= 0
        if "tenure" in df.columns:
            try:
                numeric_tenure = pd.to_numeric(df["tenure"], errors="coerce")
                neg_tenure = int((numeric_tenure < 0).sum())
                if neg_tenure > 0:
                    issues.append(
                        ValidationIssue(
                            severity="CRITICAL",
                            rule="NUMERICAL_OUT_OF_BOUNDS",
                            column="tenure",
                            description=f"Found {neg_tenure} negative tenure records.",
                            affected_count=neg_tenure,
                            recommended_action="Quarantine records with impossible negative tenure.",
                        )
                    )
            except Exception as e:
                logger.warning(f"Could not validate tenure numerically: {e}")

        # MonthlyCharges: float > 0
        if "MonthlyCharges" in df.columns:
            try:
                numeric_mc = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
                neg_mc = int((numeric_mc <= 0).sum())
                if neg_mc > 0:
                    issues.append(
                        ValidationIssue(
                            severity="CRITICAL",
                            rule="NUMERICAL_OUT_OF_BOUNDS",
                            column="MonthlyCharges",
                            description=f"Found {neg_mc} MonthlyCharges records <= 0.",
                            affected_count=neg_mc,
                            recommended_action="Inspect billing data for negative or zero charges.",
                        )
                    )
            except Exception as e:
                logger.warning(f"Could not validate MonthlyCharges numerically: {e}")

        # 8. Domain Consistency Checks
        # PhoneService == 'No' -> MultipleLines must be 'No phone service'
        if "PhoneService" in df.columns and "MultipleLines" in df.columns:
            inconsistent_phone = df[(df["PhoneService"] == "No") & (df["MultipleLines"] != "No phone service")]
            if len(inconsistent_phone) > 0:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        rule="DOMAIN_INCONSISTENCY_PHONE",
                        column="MultipleLines",
                        description=f"Found {len(inconsistent_phone)} rows where PhoneService='No' but MultipleLines!='No phone service'.",
                        affected_count=len(inconsistent_phone),
                        recommended_action="Synchronize MultipleLines to 'No phone service'.",
                    )
                )

        # InternetService == 'No' -> Internet sub-services must be 'No internet service'
        internet_features = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        if "InternetService" in df.columns:
            for feat in internet_features:
                if feat in df.columns:
                    inconsistent_net = df[(df["InternetService"] == "No") & (df[feat] != "No internet service")]
                    if len(inconsistent_net) > 0:
                        issues.append(
                            ValidationIssue(
                                severity="WARNING",
                                rule=f"DOMAIN_INCONSISTENCY_{feat.upper()}",
                                column=feat,
                                description=f"Found {len(inconsistent_net)} rows where InternetService='No' but {feat}!='No internet service'.",
                                affected_count=len(inconsistent_net),
                                recommended_action=f"Synchronize {feat} to 'No internet service'.",
                            )
                        )

        # Overall validity: valid if no CRITICAL issues
        has_critical = any(i.severity == "CRITICAL" for i in issues)

        return ValidationReport(
            is_valid=(not has_critical),
            total_rows=total_rows,
            total_columns=total_cols,
            columns_present=cols_present,
            missing_columns=missing_cols,
            unexpected_columns=unexpected_cols,
            duplicate_id_count=dup_count,
            target_distribution=target_dist,
            target_proportions=target_prop,
            missing_values_per_column=missing_dict,
            whitespace_values_per_column=whitespace_dict,
            issues=issues,
        )


def validate_dataset(data: pd.DataFrame | Path | str) -> ValidationReport:
    """Convenience helper to validate a dataset with default rules."""
    validator = DatasetValidator()
    return validator.validate(data)
