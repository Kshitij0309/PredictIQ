"""Data loading and leakage-safe dataset splitting for PredictIQ.

Provides structured dataset loading, target separation, customer identifier exclusion,
and reproducible stratified train-test splitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from predictiq.config import get_config
from predictiq.data.cleaning import clean_telco_churn_data, process_raw_to_interim

logger = logging.getLogger(__name__)

# Canonical target and identifier column definitions
TARGET_COLUMN: str = "Churn"
POSITIVE_TARGET_LABEL: str = "Yes"
NEGATIVE_TARGET_LABEL: str = "No"
IDENTIFIER_COLUMNS: Tuple[str, ...] = ("customerID",)


class DataLoadError(ValueError):
    """Raised when data loading, schema requirements, or split preconditions fail."""
    pass


@dataclass(frozen=True)
class DatasetSplits:
    """Immutable container for leakage-safe train and test splits."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    target_name: str = TARGET_COLUMN
    positive_label: int = 1
    negative_label: int = 0

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def test_size(self) -> int:
        return len(self.X_test)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def train_positive_ratio(self) -> float:
        return float(self.y_train.mean())

    @property
    def test_positive_ratio(self) -> float:
        return float(self.y_test.mean())


def load_telco_churn_data(
    data_path: Optional[Path | str] = None,
    clean_if_raw: bool = True,
) -> pd.DataFrame:
    """Load the Telco Churn dataset from interim or raw data directory.

    If interim cleaned data exists, it is loaded directly. If only raw data exists
    and clean_if_raw=True, it is cleaned and returned.

    Args:
        data_path: Explicit path to CSV file, or None to use default paths.
        clean_if_raw: Whether to clean raw data if loaded from raw path.

    Returns:
        pd.DataFrame: Cleaned dataset.

    Raises:
        DataLoadError: If dataset cannot be found or contains missing target.
    """
    config = get_config()

    if data_path is not None:
        file_path = Path(data_path)
    else:
        interim_path = config.paths.interim_data_dir / "Telco-Customer-Churn-cleaned.csv"
        raw_path = config.paths.raw_data_dir / "Telco-Customer-Churn.csv"

        if interim_path.is_file():
            file_path = interim_path
        elif raw_path.is_file():
            file_path = raw_path
        else:
            raise DataLoadError(
                f"No dataset found at interim '{interim_path}' or raw '{raw_path}'. "
                "Run data acquisition first."
            )

    if not file_path.is_file():
        raise DataLoadError(f"Dataset file does not exist at: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as err:
        raise DataLoadError(f"Failed to read CSV at {file_path}: {err}") from err

    if TARGET_COLUMN not in df.columns:
        raise DataLoadError(
            f"Required target column '{TARGET_COLUMN}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Check if raw data needs cleaning
    is_interim = "cleaned" in file_path.name
    if not is_interim and clean_if_raw:
        logger.info("Cleaning raw dataset on load...")
        df, _ = clean_telco_churn_data(df)

    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    drop_identifiers: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and binary target vector y.

    Performs validation:
    1. Ensures target exists and has valid binary classes.
    2. Maps target ('Yes' -> 1, 'No' -> 0).
    3. Drops identifier columns (e.g. 'customerID') to prevent identifier leakage.
    4. Drops any transient EDA columns that shouldn't be raw features (e.g. tenure_cohort if recomputed).

    Args:
        df: Input dataframe.
        target_col: Name of target column (default 'Churn').
        drop_identifiers: Whether to drop identifier columns.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) where y is integer binary (0 or 1).
    """
    if target_col not in df.columns:
        raise DataLoadError(f"Target column '{target_col}' not found in dataframe.")

    target_raw = df[target_col]

    # Validate target classes
    unique_targets = set(target_raw.dropna().unique())
    valid_binary_labels = {POSITIVE_TARGET_LABEL, NEGATIVE_TARGET_LABEL}
    valid_int_labels = {0, 1}

    if unique_targets.issubset(valid_binary_labels):
        y = (target_raw == POSITIVE_TARGET_LABEL).astype(int)
    elif unique_targets.issubset(valid_int_labels):
        y = target_raw.astype(int)
    else:
        raise DataLoadError(
            f"Target column contains unexpected classes: {unique_targets}. "
            f"Expected binary labels: {valid_binary_labels} or {valid_int_labels}"
        )

    if y.isnull().any():
        raise DataLoadError(f"Target column contains {y.isnull().sum()} null values.")

    # Prepare features X: drop target and identifiers
    cols_to_drop = [target_col]
    if drop_identifiers:
        for id_col in IDENTIFIER_COLUMNS:
            if id_col in df.columns:
                cols_to_drop.append(id_col)

    # If tenure_cohort was previously added in interim data, drop it so that feature
    # engineering creates features cleanly and consistently
    if "tenure_cohort" in df.columns:
        cols_to_drop.append("tenure_cohort")

    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns]).copy()

    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> DatasetSplits:
    """Create a reproducible, leakage-safe stratified train/test split.

    Guarantees:
    - Fixed random state (default 42).
    - Preserves class distribution across train and test sets when stratify=True.
    - Zero index overlap between train and test.
    - Target is completely separated.

    Args:
        X: Feature matrix.
        y: Target series (binary 0/1).
        test_size: Proportion of dataset for held-out test (default 0.20).
        random_state: Fixed random seed (default 42).
        stratify: Whether to perform stratified split (default True).

    Returns:
        DatasetSplits: Named immutable container with X_train, X_test, y_train, y_test.
    """
    if len(X) != len(y):
        raise DataLoadError(f"X length ({len(X)}) must match y length ({len(y)})")

    if not (0.0 < test_size < 1.0):
        raise DataLoadError(f"test_size must be between 0.0 and 1.0, got {test_size}")

    stratify_target = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    # Verify zero index overlap
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    overlap = train_indices.intersection(test_indices)
    if overlap:
        raise DataLoadError(f"Fatal data split error: train and test indices overlap: {overlap}")

    logger.info(
        f"Dataset split complete: Train={len(X_train)} ({y_train.mean():.2%} positive), "
        f"Test={len(X_test)} ({y_test.mean():.2%} positive)"
    )

    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def load_and_split_data(
    data_path: Optional[Path | str] = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> DatasetSplits:
    """End-to-end convenience loader: loads data, validates, separates X/y, and splits."""
    df = load_telco_churn_data(data_path=data_path)
    X, y = prepare_features_and_target(df)
    return split_dataset(X, y, test_size=test_size, random_state=random_state, stratify=True)
