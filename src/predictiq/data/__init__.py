"""PredictIQ Data Layer.

Provides data acquisition, schema validation, and dataset integrity checking.
"""

from predictiq.data.cleaning import (
    clean_telco_churn_data,
    process_raw_to_interim,
)
from predictiq.data.ingestion import (
    ChecksumMismatchError,
    DataAcquisitionError,
    DatasetMetadata,
    acquire_dataset,
    compute_sha256,
    verify_dataset_file,
)
from predictiq.data.loader import (
    DEFAULT_EXPLORATORY_SPLIT_SEED,
    IDENTIFIER_COLUMNS,
    ISOLATED_EVALUATION_SPLIT_SEED,
    NEGATIVE_TARGET_LABEL,
    POSITIVE_TARGET_LABEL,
    TARGET_COLUMN,
    DataLoadError,
    DatasetSplits,
    load_and_split_data,
    load_and_split_isolated_evaluation_data,
    load_telco_churn_data,
    prepare_features_and_target,
    split_dataset,
)
from predictiq.data.validation import (
    DataValidationError,
    DatasetValidator,
    ValidationIssue,
    ValidationReport,
    validate_dataset,
)

__all__ = [
    "acquire_dataset",
    "compute_sha256",
    "verify_dataset_file",
    "DataAcquisitionError",
    "ChecksumMismatchError",
    "DatasetMetadata",
    "DatasetValidator",
    "DataValidationError",
    "ValidationIssue",
    "ValidationReport",
    "validate_dataset",
    "clean_telco_churn_data",
    "process_raw_to_interim",
    "TARGET_COLUMN",
    "POSITIVE_TARGET_LABEL",
    "NEGATIVE_TARGET_LABEL",
    "IDENTIFIER_COLUMNS",
    "DEFAULT_EXPLORATORY_SPLIT_SEED",
    "ISOLATED_EVALUATION_SPLIT_SEED",
    "DataLoadError",
    "DatasetSplits",
    "load_telco_churn_data",
    "prepare_features_and_target",
    "split_dataset",
    "load_and_split_data",
    "load_and_split_isolated_evaluation_data",
]
