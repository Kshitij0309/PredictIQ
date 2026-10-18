"""Model evaluation metrics, visualization, and comparison reporting for PredictIQ.

Computes comprehensive classification metrics for binary churn classification,
safely extracts probability-based metrics (ROC-AUC, PR-AUC), and produces publication-quality
evaluation plots and comparison tables.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluationResult:
    """Standardized evaluation results container for a single model."""
    model_name: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    average_precision: Optional[float]
    brier_score: Optional[float]
    confusion_matrix: List[List[int]]
    classification_report_dict: Dict[str, Any]
    cv_mean_pr_auc: Optional[float] = None
    cv_std_pr_auc: Optional[float] = None
    cv_mean_roc_auc: Optional[float] = None
    cv_std_roc_auc: Optional[float] = None
    y_true: Optional[List[int]] = None
    y_pred: Optional[List[int]] = None
    y_proba: Optional[List[float]] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return scalar metrics suitable for tabular comparison and logging."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "average_precision": (
                round(self.average_precision, 4) if self.average_precision is not None else None
            ),
            "brier_score": round(self.brier_score, 4) if self.brier_score is not None else None,
            "cv_mean_pr_auc": (
                round(self.cv_mean_pr_auc, 4) if self.cv_mean_pr_auc is not None else None
            ),
            "cv_mean_roc_auc": (
                round(self.cv_mean_roc_auc, 4) if self.cv_mean_roc_auc is not None else None
            ),
        }


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    model_name: str = "Model",
    model_type: str = "Classifier",
    cv_mean_pr_auc: Optional[float] = None,
    cv_std_pr_auc: Optional[float] = None,
    cv_mean_roc_auc: Optional[float] = None,
    cv_std_roc_auc: Optional[float] = None,
    store_predictions: bool = True,
) -> ModelEvaluationResult:
    """Evaluate a fitted model on held-out test data.

    Calculates:
    - Accuracy, Precision, Recall, F1 for the positive class (1 = Churn).
    - ROC-AUC and Average Precision (PR-AUC) using probabilities where supported.
    - Confusion Matrix and full classification report.

    Args:
        model: Fitted scikit-learn estimator or pipeline.
        X_test: Test features.
        y_test: Test true labels (binary 0/1).
        model_name: Descriptive name for the model.
        model_type: Architecture/algorithm category.
        cv_mean_pr_auc: Optional training cross-validation PR-AUC mean.
        cv_std_pr_auc: Optional training cross-validation PR-AUC std.
        cv_mean_roc_auc: Optional training cross-validation ROC-AUC mean.
        cv_std_roc_auc: Optional training cross-validation ROC-AUC std.
        store_predictions: Whether to store raw prediction vectors for plotting.

    Returns:
        ModelEvaluationResult: Rich evaluation metrics container.
    """
    y_true = np.asarray(y_test, dtype=int)
    y_pred = np.asarray(model.predict(X_test), dtype=int)

    # Probability extraction
    y_proba: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_test)
            if probas.shape[1] >= 2:
                y_proba = probas[:, 1]
            else:
                # Single class prediction case (e.g. DummyClassifier with 1 class)
                y_proba = probas[:, 0]
        except Exception as err:
            logger.warning(f"Could not extract predict_proba for {model_name}: {err}")

    # Standard metrics
    acc = float(accuracy_score(y_true, y_pred))
    # Handle zero division safely for dummy or extreme models
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Ranking metrics (require probability scores)
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    brier: Optional[float] = None

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
            pr_auc = float(average_precision_score(y_true, y_proba))
            brier = float(brier_score_loss(y_true, y_proba))
        except Exception as err:
            logger.warning(f"Error computing ranking metrics for {model_name}: {err}")

    cm = confusion_matrix(y_true, y_pred).tolist()
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return ModelEvaluationResult(
        model_name=model_name,
        model_type=model_type,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        roc_auc=roc_auc,
        average_precision=pr_auc,
        brier_score=brier,
        confusion_matrix=cm,
        classification_report_dict=report_dict,
        cv_mean_pr_auc=cv_mean_pr_auc,
        cv_std_pr_auc=cv_std_pr_auc,
        cv_mean_roc_auc=cv_mean_roc_auc,
        cv_std_roc_auc=cv_std_roc_auc,
        y_true=y_true.tolist() if store_predictions else None,
        y_pred=y_pred.tolist() if store_predictions else None,
        y_proba=y_proba.tolist() if (store_predictions and y_proba is not None) else None,
    )


def build_comparison_dataframe(results: List[ModelEvaluationResult]) -> pd.DataFrame:
    """Build a consolidated comparison table across multiple evaluated models."""
    records = [res.to_summary_dict() for res in results]
    df = pd.DataFrame.from_records(records)
    # Sort descending by primary metric: average_precision (PR-AUC)
    if "average_precision" in df.columns and df["average_precision"].notnull().any():
        df = df.sort_values(by="average_precision", ascending=False).reset_index(drop=True)
    return df


def plot_roc_curves(
    results: List[ModelEvaluationResult],
    output_path: Optional[Path | str] = None,
) -> Path:
    """Generate and save comparative ROC curves."""
    plt.figure(figsize=(8, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for res in results:
        if res.y_true is not None and res.y_proba is not None:
            fpr, tpr, _ = roc_curve(res.y_true, res.y_proba)
            auc_str = f"{res.roc_auc:.3f}" if res.roc_auc is not None else "N/A"
            plt.plot(fpr, tpr, lw=2, label=f"{res.model_name} (AUC = {auc_str})")

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    out_file = Path(output_path) if output_path else Path("reports/figures/roc_curves.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_pr_curves(
    results: List[ModelEvaluationResult],
    output_path: Optional[Path | str] = None,
) -> Path:
    """Generate and save comparative Precision-Recall curves."""
    plt.figure(figsize=(8, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Baseline positive prevalence line
    prevalence = None
    for res in results:
        if res.y_true is not None and prevalence is None:
            prevalence = float(np.mean(res.y_true))

    for res in results:
        if res.y_true is not None and res.y_proba is not None:
            precision, recall, _ = precision_recall_curve(res.y_true, res.y_proba)
            ap_str = f"{res.average_precision:.3f}" if res.average_precision is not None else "N/A"
            plt.plot(recall, precision, lw=2, label=f"{res.model_name} (PR-AUC = {ap_str})")

    if prevalence is not None:
        plt.axhline(
            y=prevalence,
            color="gray",
            linestyle="--",
            lw=1.5,
            label=f"Baseline Prevalence ({prevalence:.1%})",
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall (PR) Curve Comparison (Primary Metric)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    out_file = Path(output_path) if output_path else Path("reports/figures/pr_curves.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_confusion_matrices(
    results: List[ModelEvaluationResult],
    output_path: Optional[Path | str] = None,
) -> Path:
    """Generate side-by-side heatmaps of confusion matrices."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), dpi=300)
    if n_models == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = np.array(res.confusion_matrix)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
        )
        ax.set_title(f"{res.model_name}\n(F1 = {res.f1_score:.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("Actual Label", fontsize=10)

    plt.tight_layout()
    out_file = Path(output_path) if output_path else Path("reports/figures/confusion_matrices.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def save_evaluation_artifacts(
    results: List[ModelEvaluationResult],
    reports_dir: Optional[Path | str] = None,
) -> Dict[str, Path]:
    """Save all evaluation tables and plots into reports/ directory."""
    rep_dir = Path(reports_dir) if reports_dir else Path("reports")
    fig_dir = rep_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Comparison DataFrame
    df = build_comparison_dataframe(results)
    csv_path = rep_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)

    # 2. JSON summary metrics
    json_path = rep_dir / "model_comparison.json"
    clean_results = []
    for r in results:
        d = asdict(r)
        # Drop large arrays from JSON
        d.pop("y_true", None)
        d.pop("y_pred", None)
        d.pop("y_proba", None)
        clean_results.append(d)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)

    # 3. Plots
    roc_path = plot_roc_curves(results, fig_dir / "roc_curves.png")
    pr_path = plot_pr_curves(results, fig_dir / "pr_curves.png")
    cm_path = plot_confusion_matrices(results, fig_dir / "confusion_matrices.png")

    return {
        "comparison_csv": csv_path,
        "comparison_json": json_path,
        "roc_curve": roc_path,
        "pr_curve": pr_path,
        "confusion_matrix": cm_path,
    }
