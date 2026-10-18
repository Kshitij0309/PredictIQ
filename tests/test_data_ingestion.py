"""Deterministic unit tests for data acquisition and provenance tracking."""

import hashlib
from pathlib import Path

import pytest

from predictiq.data.ingestion import (
    ChecksumMismatchError,
    DataAcquisitionError,
    DatasetMetadata,
    acquire_dataset,
    compute_sha256,
    verify_dataset_file,
)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a minimal synthetic CSV file for test fixtures."""
    file_path = tmp_path / "mock_dataset.csv"
    file_path.write_text(
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,"
        "InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
        "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn\n"
        "0001-TEST,Female,0,Yes,No,12,Yes,No,DSL,Yes,No,No,No,No,No,Month-to-month,Yes,Electronic check,50.0,600.0,No\n",
        encoding="utf-8",
    )
    return file_path


class TestHashAndVerification:
    """Validate SHA-256 calculation and file integrity verification."""

    def test_compute_sha256_matches_known_string(self, tmp_path: Path):
        content = b"PredictIQ deterministic checksum test payload\n"
        test_file = tmp_path / "hash_test.txt"
        test_file.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        assert compute_sha256(test_file) == expected

    def test_compute_sha256_missing_file_raises(self, tmp_path: Path):
        non_existent = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            compute_sha256(non_existent)

    def test_verify_dataset_file(self, sample_csv: Path):
        expected_hash = compute_sha256(sample_csv)
        assert verify_dataset_file(sample_csv, expected_sha256=expected_hash) is True
        assert verify_dataset_file(sample_csv, expected_sha256="incorrect_hash") is False
        assert verify_dataset_file(sample_csv.parent / "ghost.csv") is False

    def test_verify_dataset_file_empty_fails(self, tmp_path: Path):
        empty_file = tmp_path / "empty.csv"
        empty_file.touch()
        assert verify_dataset_file(empty_file) is False


class TestDatasetMetadata:
    """Validate metadata serialization, deserialization, and integrity."""

    def test_metadata_save_and_load(self, tmp_path: Path):
        meta_path = tmp_path / "metadata.json"
        meta = DatasetMetadata(
            dataset_name="Test Dataset",
            filename="test.csv",
            source_url="https://example.com/test.csv",
            provenance_repo="https://github.com/example/test",
            license="Apache-2.0",
            retrieval_timestamp_utc="2026-10-18T12:00:00+00:00",
            file_size_bytes=1024,
            sha256_checksum="abc123def456",
            acquisition_method="automated_download",
            target_path=str(tmp_path / "test.csv"),
        )
        saved = meta.save(meta_path)
        assert saved.is_file()

        loaded = DatasetMetadata.load(meta_path)
        assert loaded.dataset_name == "Test Dataset"
        assert loaded.sha256_checksum == "abc123def456"
        assert loaded.acquisition_method == "automated_download"

    def test_metadata_load_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            DatasetMetadata.load(tmp_path / "non_existent_meta.json")


class TestDatasetAcquisition:
    """Validate acquisition flows using local fixtures without external network calls."""

    def test_acquire_from_local_source(self, sample_csv: Path, tmp_path: Path):
        dest_dir = tmp_path / "raw"
        meta_dir = tmp_path / "metadata"

        dest_file, metadata = acquire_dataset(
            destination_dir=dest_dir,
            filename="imported.csv",
            metadata_dir=meta_dir,
            local_source_file=sample_csv,
        )

        assert dest_file.is_file()
        assert dest_file.name == "imported.csv"
        assert metadata.acquisition_method == "manual_import"
        assert metadata.file_size_bytes == sample_csv.stat().st_size
        assert (meta_dir / "imported_metadata.json").is_file()

    def test_acquire_from_local_missing_raises(self, tmp_path: Path):
        with pytest.raises(DataAcquisitionError):
            acquire_dataset(
                destination_dir=tmp_path / "raw",
                local_source_file=tmp_path / "missing_source.csv",
            )

    def test_acquire_checksum_mismatch_raises(self, sample_csv: Path, tmp_path: Path):
        with pytest.raises(ChecksumMismatchError):
            acquire_dataset(
                destination_dir=tmp_path / "raw",
                filename="mismatch.csv",
                local_source_file=sample_csv,
                expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
            )
