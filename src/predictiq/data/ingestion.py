"""Dataset acquisition and provenance tracking for PredictIQ.

Downloads public datasets from verified upstream repositories, calculates
cryptographic hashes (SHA-256), and generates reproducible provenance metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from predictiq.config import get_config

logger = logging.getLogger(__name__)

# Verified upstream IBM repository and dataset location
DEFAULT_DATASET_URL: str = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
DEFAULT_PROVENANCE_URL: str = (
    "https://github.com/IBM/telco-customer-churn-on-icp4d"
)
DEFAULT_FILENAME: str = "Telco-Customer-Churn.csv"
DEFAULT_DATASET_NAME: str = "IBM Telco Customer Churn"
DEFAULT_LICENSE: str = "Apache-2.0"

# Recorded cryptographic identity for the official repository version
VERIFIED_SHA256: str = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)


class DataAcquisitionError(Exception):
    """Raised when dataset acquisition fails due to network, filesystem, or protocol issues."""
    pass


class ChecksumMismatchError(DataAcquisitionError):
    """Raised when the calculated file digest does not match the expected hash."""
    pass


@dataclass(frozen=True)
class DatasetMetadata:
    """Provenance and cryptographic identity metadata for acquired datasets."""
    dataset_name: str
    filename: str
    source_url: str
    provenance_repo: str
    license: str
    retrieval_timestamp_utc: str
    file_size_bytes: int
    sha256_checksum: str
    acquisition_method: str  # 'automated_download' or 'manual_import'
    target_path: str

    def to_dict(self) -> dict:
        """Convert metadata to dictionary representation."""
        return asdict(self)

    def save(self, metadata_path: Path | str) -> Path:
        """Save metadata to a JSON file."""
        target = Path(metadata_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target

    @classmethod
    def load(cls, metadata_path: Path | str) -> DatasetMetadata:
        """Load metadata from an existing JSON file."""
        path = Path(metadata_path)
        if not path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def compute_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 cryptographic digest of a local file in memory-efficient blocks."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_dataset_file(
    file_path: Path | str,
    expected_sha256: Optional[str] = None,
) -> bool:
    """Verify that a dataset file exists, is non-empty, and optionally matches expected hash."""
    path = Path(file_path)
    if not path.is_file():
        return False
    if path.stat().st_size == 0:
        return False
    if expected_sha256 is not None:
        computed = compute_sha256(path)
        if computed.lower() != expected_sha256.lower():
            return False
    return True


def acquire_dataset(
    source_url: str = DEFAULT_DATASET_URL,
    provenance_repo: str = DEFAULT_PROVENANCE_URL,
    destination_dir: Optional[Path | str] = None,
    filename: str = DEFAULT_FILENAME,
    metadata_dir: Optional[Path | str] = None,
    force_download: bool = False,
    timeout_seconds: int = 30,
    expected_sha256: Optional[str] = None,
    local_source_file: Optional[Path | str] = None,
) -> tuple[Path, DatasetMetadata]:
    """Acquire the dataset either via verified HTTP download or verified local import.

    Args:
        source_url: HTTP(S) URL to download the dataset from.
        provenance_repo: Upstream public repository tracking provenance.
        destination_dir: Target directory (defaults to config.paths.raw_data_dir).
        filename: Destination filename.
        metadata_dir: Directory where provenance metadata JSON will be written.
        force_download: Overwrite existing local file if True.
        timeout_seconds: Network connection and read timeout.
        expected_sha256: Optional hash to enforce identity.
        local_source_file: If provided, copies from this manual path instead of downloading.

    Returns:
        tuple[Path, DatasetMetadata]: Local dataset file path and its metadata record.
    """
    config = get_config()
    dest_dir = Path(destination_dir) if destination_dir else config.paths.raw_data_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination_path = dest_dir / filename

    meta_dir = (
        Path(metadata_dir)
        if metadata_dir
        else (config.paths.root_dir / "data" / "metadata")
    )
    meta_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = meta_dir / f"{Path(filename).stem}_metadata.json"

    # Reuse existing verified raw file if present and not force
    if destination_path.is_file() and not force_download and not local_source_file:
        logger.info(f"Existing dataset file found at '{destination_path}'. Verifying identity...")
        actual_hash = compute_sha256(destination_path)
        if expected_sha256 and actual_hash.lower() != expected_sha256.lower():
            raise ChecksumMismatchError(
                f"Existing file '{destination_path}' hash ({actual_hash}) does not match expected ({expected_sha256})"
            )

        if metadata_file.is_file():
            metadata = DatasetMetadata.load(metadata_file)
            return destination_path, metadata

        # Generate metadata for pre-existing file
        metadata = DatasetMetadata(
            dataset_name=DEFAULT_DATASET_NAME,
            filename=filename,
            source_url=source_url,
            provenance_repo=provenance_repo,
            license=DEFAULT_LICENSE,
            retrieval_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            file_size_bytes=destination_path.stat().st_size,
            sha256_checksum=actual_hash,
            acquisition_method="pre_existing_verified",
            target_path=str(destination_path),
        )
        metadata.save(metadata_file)
        return destination_path, metadata

    # Branch: Manual local file import
    if local_source_file:
        src = Path(local_source_file)
        if not src.is_file():
            raise DataAcquisitionError(f"Local source file does not exist: {src}")

        shutil.copy2(src, destination_path)
        actual_hash = compute_sha256(destination_path)
        if expected_sha256 and actual_hash.lower() != expected_sha256.lower():
            raise ChecksumMismatchError(
                f"Imported file '{src}' hash ({actual_hash}) does not match expected ({expected_sha256})"
            )

        metadata = DatasetMetadata(
            dataset_name=DEFAULT_DATASET_NAME,
            filename=filename,
            source_url=f"file://{src.resolve()}",
            provenance_repo=provenance_repo,
            license=DEFAULT_LICENSE,
            retrieval_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            file_size_bytes=destination_path.stat().st_size,
            sha256_checksum=actual_hash,
            acquisition_method="manual_import",
            target_path=str(destination_path),
        )
        metadata.save(metadata_file)
        return destination_path, metadata

    # Branch: Automated verified HTTP download
    logger.info(f"Downloading dataset from '{source_url}'...")
    req = urllib.request.Request(
        source_url,
        headers={"User-Agent": "PredictIQ-DataAcquisition/0.1.0"},
    )

    tmp_path = destination_path.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise DataAcquisitionError(
                    f"HTTP request to '{source_url}' returned unexpected status {response.status}"
                )
            with open(tmp_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as err:
        if tmp_path.is_file():
            tmp_path.unlink()
        raise DataAcquisitionError(
            f"Failed to acquire dataset from '{source_url}': {err}"
        ) from err

    # Compute hash and validate
    actual_hash = compute_sha256(tmp_path)
    if expected_sha256 and actual_hash.lower() != expected_sha256.lower():
        tmp_path.unlink()
        raise ChecksumMismatchError(
            f"Downloaded file digest '{actual_hash}' does not match expected '{expected_sha256}'"
        )

    # Move temporary file to final raw destination
    if destination_path.is_file():
        destination_path.unlink()
    tmp_path.rename(destination_path)

    metadata = DatasetMetadata(
        dataset_name=DEFAULT_DATASET_NAME,
        filename=filename,
        source_url=source_url,
        provenance_repo=provenance_repo,
        license=DEFAULT_LICENSE,
        retrieval_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        file_size_bytes=destination_path.stat().st_size,
        sha256_checksum=actual_hash,
        acquisition_method="automated_download",
        target_path=str(destination_path),
    )
    metadata.save(metadata_file)
    logger.info(
        f"Dataset successfully acquired: {destination_path} ({metadata.file_size_bytes} bytes, SHA-256: {metadata.sha256_checksum})"
    )
    return destination_path, metadata


def main() -> None:
    """CLI runner for dataset acquisition and verification."""
    parser = argparse.ArgumentParser(description="PredictIQ Dataset Acquisition Tool")
    parser.add_argument(
        "--url",
        default=DEFAULT_DATASET_URL,
        help="Source download URL",
    )
    parser.add_argument(
        "--dest-dir",
        default=None,
        help="Target output directory (defaults to data/raw)",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help="Destination filename",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing dataset file without downloading",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.verify_only:
        config = get_config()
        target = (Path(args.dest_dir) if args.dest_dir else config.paths.raw_data_dir) / args.filename
        if verify_dataset_file(target, expected_sha256=VERIFIED_SHA256):
            print(f"VERIFIED: {target} matches expected hash ({VERIFIED_SHA256})")
        else:
            print(f"FAILED: {target} is missing or has incorrect checksum.")
            exit(1)
        return

    path, meta = acquire_dataset(
        source_url=args.url,
        destination_dir=args.dest_dir,
        filename=args.filename,
        force_download=args.force,
        expected_sha256=VERIFIED_SHA256,
    )
    print(f"SUCCESS: Acquired {meta.dataset_name}")
    print(f"Location: {path}")
    print(f"Size: {meta.file_size_bytes} bytes")
    print(f"SHA-256: {meta.sha256_checksum}")
    print(f"License: {meta.license}")


if __name__ == "__main__":
    main()
