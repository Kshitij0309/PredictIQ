"""Model pipeline persistence and metadata serialization for PredictIQ.

Provides joblib-based serialization of complete fitted preprocessing-plus-classifier pipelines,
paired with schema metadata, training seeds, feature lists, and evaluation summaries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
from sklearn.pipeline import Pipeline

from predictiq.config import get_config

logger = logging.getLogger(__name__)


class ModelPersistenceError(RuntimeError):
    """Raised when serializing or deserializing a model pipeline fails."""
    pass


def save_model_pipeline(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    output_dir: Optional[Path | str] = None,
    model_name: str = "churn_baseline_model",
) -> Tuple[Path, Path]:
    """Save a fitted model pipeline and accompanying metadata to disk.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        metadata: Dictionary containing training parameters, metrics, feature schema, etc.
        output_dir: Target directory for artifacts. Defaults to <root>/models.
        model_name: Base filename prefix.

    Returns:
        Tuple[Path, Path]: Path to saved .joblib file and path to .json metadata file.

    Raises:
        ModelPersistenceError: If serialization fails.
    """
    config = get_config()
    out_dir = Path(output_dir) if output_dir else (config.paths.root_dir / "models")
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = out_dir / f"{model_name}.joblib"
    metadata_path = out_dir / f"{model_name}_metadata.json"

    try:
        # Save complete pipeline
        joblib.dump(pipeline, joblib_path, compress=3)
        logger.info(f"Model pipeline saved to: {joblib_path}")

        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Model metadata saved to: {metadata_path}")

        return joblib_path, metadata_path

    except Exception as err:
        raise ModelPersistenceError(f"Failed to save model pipeline to {out_dir}: {err}") from err


def load_model_pipeline(
    model_path: Optional[Path | str] = None,
    metadata_path: Optional[Path | str] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Load a serialized model pipeline and its metadata from disk.

    Args:
        model_path: Path to .joblib file. Defaults to <root>/models/churn_baseline_model.joblib.
        metadata_path: Path to metadata .json. Defaults to same base path with _metadata.json suffix.

    Returns:
        Tuple[Pipeline, Dict[str, Any]]: Deserialized pipeline and metadata dictionary.

    Raises:
        FileNotFoundError: If files do not exist.
        ModelPersistenceError: If deserialization fails.
    """
    config = get_config()
    default_dir = config.paths.root_dir / "models"

    if model_path:
        m_path = Path(model_path)
    else:
        champion_path = default_dir / "churn_champion_model.joblib"
        baseline_path = default_dir / "churn_baseline_model.joblib"
        cwd_champion = Path.cwd() / "models" / "churn_champion_model.joblib"
        cwd_baseline = Path.cwd() / "models" / "churn_baseline_model.joblib"
        if champion_path.is_file():
            m_path = champion_path
        elif cwd_champion.is_file():
            m_path = cwd_champion
        elif baseline_path.is_file():
            m_path = baseline_path
        elif cwd_baseline.is_file():
            m_path = cwd_baseline
        else:
            m_path = champion_path
    meta_path = (
        Path(metadata_path)
        if metadata_path
        else m_path.with_name(f"{m_path.stem}_metadata.json")
    )

    if not m_path.is_file():
        raise FileNotFoundError(f"Model pipeline file does not exist: {m_path}")

    try:
        pipeline = joblib.load(m_path)
    except Exception as err:
        raise ModelPersistenceError(f"Failed to deserialize pipeline from {m_path}: {err}") from err

    metadata: Dict[str, Any] = {}
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as err:
            logger.warning(f"Could not load metadata from {meta_path}: {err}")

    return pipeline, metadata
