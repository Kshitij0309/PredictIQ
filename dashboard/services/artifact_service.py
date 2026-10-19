"""Service for loading genuine model evaluation metrics, reports, and diagnostic figures."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _find_repo_root() -> Path:
    """Resolve repository root directory from environment or file location."""
    env_root = os.getenv("PREDICTIQ_ROOT_DIR") or os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).is_dir():
        return Path(env_root).resolve()
    return Path(__file__).resolve().parent.parent.parent


def load_model_metadata() -> Tuple[Dict[str, Any], bool]:
    """Load model champion metadata JSON."""
    root = _find_repo_root()
    candidates = [
        root / "models" / "churn_champion_model_metadata.json",
        Path("models/churn_champion_model_metadata.json").resolve(),
    ]
    for p in candidates:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data, True
            except Exception as err:
                logger.error(f"Error reading metadata from {p}: {err}")
                return {"error": str(err)}, False
    return {"error": "Metadata file models/churn_champion_model_metadata.json not found."}, False


def load_cv_model_comparison() -> Tuple[pd.DataFrame, bool]:
    """Load 5-fold cross-validation candidate model comparison CSV."""
    root = _find_repo_root()
    candidates = [
        root / "reports" / "phase4_cv_model_comparison.csv",
        Path("reports/phase4_cv_model_comparison.csv").resolve(),
    ]
    for p in candidates:
        if p.is_file():
            try:
                df = pd.read_csv(p)
                return df, True
            except Exception as err:
                logger.error(f"Error reading CV comparison from {p}: {err}")
                return pd.DataFrame(), False
    return pd.DataFrame(), False


def load_final_test_summary() -> Tuple[pd.DataFrame, bool]:
    """Load final held-out evaluation summary CSV."""
    root = _find_repo_root()
    candidates = [
        root / "reports" / "phase4_final_test_summary.csv",
        Path("reports/phase4_final_test_summary.csv").resolve(),
    ]
    for p in candidates:
        if p.is_file():
            try:
                df = pd.read_csv(p)
                return df, True
            except Exception as err:
                logger.error(f"Error reading final test summary from {p}: {err}")
                return pd.DataFrame(), False
    return pd.DataFrame(), False


def get_figure_path(figure_name: str) -> Optional[Path]:
    """Resolve path to a diagnostic evaluation figure if it exists."""
    root = _find_repo_root()
    candidates = [
        root / "reports" / "figures" / figure_name,
        Path("reports/figures") / figure_name,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def list_available_figures() -> Dict[str, Optional[Path]]:
    """Return dictionary of standard figures and their verified paths."""
    figure_names = [
        "calibration_curve.png",
        "champion_heldout_curves.png",
        "confusion_matrices.png",
        "pr_curves.png",
        "roc_curves.png",
        "shap_summary.png",
        "shap_local_high_risk.png",
        "shap_local_low_risk.png",
        "threshold_tradeoff.png",
    ]
    return {name: get_figure_path(name) for name in figure_names}
