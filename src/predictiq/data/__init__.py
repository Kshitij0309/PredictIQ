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
]
