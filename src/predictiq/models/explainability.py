"""Model explainability and SHAP attribution analysis for PredictIQ.

Generates global feature attribution summaries and localized archetype decision explanations
using TreeSHAP without exposing customer identifiers or claiming causal influence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class SHAPExplanationResult:
    """Summary of global and local SHAP explanations."""
    global_importance_df: pd.DataFrame
    top_10_features: List[str]
    summary_plot_path: Optional[Path] = None
    local_high_risk_plot_path: Optional[Path] = None
    local_low_risk_plot_path: Optional[Path] = None
    archetype_insights: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> dict:
        return {
            "top_10_features": self.top_10_features,
            "summary_plot_path": str(self.summary_plot_path) if self.summary_plot_path else None,
            "local_high_risk_plot_path": (
                str(self.local_high_risk_plot_path) if self.local_high_risk_plot_path else None
            ),
            "local_low_risk_plot_path": (
                str(self.local_low_risk_plot_path) if self.local_low_risk_plot_path else None
            ),
            "archetype_insights": self.archetype_insights,
        }


def extract_preprocessed_features(
    pipeline: Any,
    X: pd.DataFrame,
) -> tuple[np.ndarray, List[str], Any]:
    """Transform raw dataframe through pipeline preprocessing steps and return matrix and names."""
    from sklearn.calibration import CalibratedClassifierCV

    # Unwrap top-level CalibratedClassifierCV if passed directly
    base_pipe = pipeline
    if isinstance(pipeline, CalibratedClassifierCV):
        if hasattr(pipeline, "estimator") and pipeline.estimator is not None:
            base_pipe = pipeline.estimator
        elif hasattr(pipeline, "calibrated_classifiers_") and len(pipeline.calibrated_classifiers_) > 0:
            base_pipe = pipeline.calibrated_classifiers_[0].estimator

    # 1. Feature Engineering step
    if hasattr(base_pipe, "named_steps") and "features" in base_pipe.named_steps:
        X_feat = base_pipe.named_steps["features"].transform(X)
    else:
        X_feat = X

    # 2. Preprocessor step
    preprocessor = base_pipe.named_steps["preprocessor"]
    X_trans = preprocessor.transform(X_feat)

    # 3. Feature names out
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feat_{i}" for i in range(X_trans.shape[1])]

    # 4. Underlying classifier
    classifier = base_pipe.named_steps["classifier"]

    # Handle inner CalibratedClassifierCV if wrapped inside pipeline step
    if isinstance(classifier, CalibratedClassifierCV):
        if hasattr(classifier, "calibrated_classifiers_") and len(classifier.calibrated_classifiers_) > 0:
            classifier = classifier.calibrated_classifiers_[0].estimator
        else:
            classifier = classifier.estimator

    return X_trans, feature_names, classifier


def generate_shap_explanations(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    reports_dir: Optional[Path | str] = None,
    max_eval_samples: int = 500,
) -> SHAPExplanationResult:
    """Generate global TreeSHAP attributions and illustrative local customer explanations.

    Safety & Integrity Guardrails:
    - Identifiers and PII are excluded.
    - SHAP values quantify empirical model feature impact, NOT real-world causal mechanisms.

    Args:
        pipeline: Fitted model pipeline (e.g. LightGBM, Random Forest).
        X_sample: Representative sample of data for explanation.
        reports_dir: Output directory for plots (e.g. reports/).
        max_eval_samples: Maximum rows to compute SHAP values for (default 500 for performance).

    Returns:
        SHAPExplanationResult: Ranked feature importance and artifact paths.
    """
    logger.info("Computing SHAP feature attributions on representative cohort...")
    sample_eval = X_sample.head(max_eval_samples).copy()

    X_trans, feature_names, tree_model = extract_preprocessed_features(pipeline, sample_eval)
    X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

    # 1. Initialize TreeExplainer
    explainer = shap.TreeExplainer(tree_model)
    shap_raw = explainer.shap_values(X_trans_df)

    # Scikit-learn Random Forest returns list [values_class_0, values_class_1]
    # LightGBM returns ndarray (N, n_features) or list
    if isinstance(shap_raw, list):
        # Binary classification positive class is index 1
        shap_values = shap_raw[1] if len(shap_raw) > 1 else shap_raw[0]
    elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        # Array of shape (N, n_features, 2)
        shap_values = shap_raw[:, :, 1]
    else:
        shap_values = shap_raw

    # 2. Compute Global Importance (Mean Absolute SHAP Value)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    top_10 = list(importance_df["feature"].head(10))
    logger.info(f"Top 5 SHAP predictive drivers: {top_10[:5]}")

    # 3. Generate Visual Artifacts
    fig_dir = (Path(reports_dir) / "figures") if reports_dir else Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Global summary plot (Bar plot of top 15 features)
    summary_plot_path = fig_dir / "shap_summary.png"
    plt.figure(figsize=(10, 6), dpi=300)
    top_15 = importance_df.head(15).iloc[::-1]  # Invert for horizontal bar plot
    plt.barh(top_15["feature"], top_15["mean_abs_shap"], color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.xlabel("Mean Absolute SHAP Value (Impact on Churn Model Output)", fontsize=11, fontweight="bold")
    plt.title("Global Feature Attribution (TreeSHAP Summary)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()

    # 4. Local Explanations for Archetypes
    probas = pipeline.predict_proba(sample_eval)[:, 1]
    high_risk_idx = int(np.argmax(probas))
    low_risk_idx = int(np.argmin(probas))

    # Local High-Risk Customer Plot
    local_high_path = fig_dir / "shap_local_high_risk.png"
    _plot_local_archetype_shap(
        shap_values=shap_values[high_risk_idx],
        feature_names=feature_names,
        churn_proba=float(probas[high_risk_idx]),
        archetype_title="Archetype A: High Churn Risk Customer",
        output_path=local_high_path,
    )

    # Local Low-Risk Customer Plot
    local_low_path = fig_dir / "shap_local_low_risk.png"
    _plot_local_archetype_shap(
        shap_values=shap_values[low_risk_idx],
        feature_names=feature_names,
        churn_proba=float(probas[low_risk_idx]),
        archetype_title="Archetype B: Low Churn Risk Customer",
        output_path=local_low_path,
    )

    archetype_insights = [
        {
            "archetype": "High Churn Risk",
            "predicted_probability": round(float(probas[high_risk_idx]), 4),
            "top_positive_drivers": _get_top_local_drivers(shap_values[high_risk_idx], feature_names, positive=True),
        },
        {
            "archetype": "Low Churn Risk",
            "predicted_probability": round(float(probas[low_risk_idx]), 4),
            "top_retention_drivers": _get_top_local_drivers(shap_values[low_risk_idx], feature_names, positive=False),
        },
    ]

    return SHAPExplanationResult(
        global_importance_df=importance_df,
        top_10_features=top_10,
        summary_plot_path=summary_plot_path,
        local_high_risk_plot_path=local_high_path,
        local_low_risk_plot_path=local_low_path,
        archetype_insights=archetype_insights,
    )


def _plot_local_archetype_shap(
    shap_values: np.ndarray,
    feature_names: List[str],
    churn_proba: float,
    archetype_title: str,
    output_path: Path,
) -> None:
    """Plot local feature contributions for an archetype customer."""
    # Top 8 contributors by magnitude
    top_indices = np.argsort(np.abs(shap_values))[-8:]
    top_names = [feature_names[i] for i in top_indices]
    top_vals = [shap_values[i] for i in top_indices]

    colors = ["crimson" if val > 0 else "royalblue" for val in top_vals]

    plt.figure(figsize=(9, 5), dpi=300)
    plt.barh(top_names, top_vals, color=colors, edgecolor="black", alpha=0.85)
    plt.axvline(0, color="gray", linestyle="--", lw=1)
    plt.xlabel("SHAP Value (Contribution to Log-Odds of Churn)", fontsize=10, fontweight="bold")
    plt.title(f"{archetype_title}\n(Predicted Churn Probability: {churn_proba:.1%})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _get_top_local_drivers(
    shap_vals: np.ndarray,
    feature_names: List[str],
    positive: bool = True,
    k: int = 3,
) -> List[str]:
    """Retrieve top k feature names pushing probability up (positive=True) or down (positive=False)."""
    if positive:
        indices = np.argsort(shap_vals)[-k:][::-1]
    else:
        indices = np.argsort(shap_vals)[:k]
    return [feature_names[i] for i in indices]
