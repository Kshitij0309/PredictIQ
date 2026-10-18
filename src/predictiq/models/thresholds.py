"""Decision threshold optimization and business trade-off analysis for PredictIQ.

Evaluates candidate classification thresholds strictly on training-only out-of-fold predictions.
Provides transparent precision-recall trade-off tables and supports configurable business cost matrices
without inventing unverified customer lifetime value or savings claims.
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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


@dataclass
class BusinessCostMatrix:
    """Configurable cost matrix parameters for decision threshold optimization.

    If costs are unknown or not supplied, threshold selection falls back to empirical
    F1-maximization or constrained recall analysis.
    """
    cost_false_positive: float  # e.g., cost of retention incentive wasted on non-churner
    cost_false_negative: float  # e.g., loss of revenue from an unmitigated churner


@dataclass
class ThresholdOptimizationResult:
    """Outcome of training-only threshold optimization."""
    optimal_threshold: float
    selection_strategy: str
    metrics_at_optimal: Dict[str, Any]
    tradeoff_dataframe: pd.DataFrame
    cost_matrix_applied: Optional[Dict[str, float]] = None
    plot_path: Optional[Path] = None

    def to_dict(self) -> dict:
        return {
            "optimal_threshold": round(self.optimal_threshold, 4),
            "selection_strategy": self.selection_strategy,
            "metrics_at_optimal": self.metrics_at_optimal,
            "cost_matrix_applied": self.cost_matrix_applied,
            "plot_path": str(self.plot_path) if self.plot_path else None,
        }


def compute_threshold_tradeoff_table(
    y_true: np.ndarray | pd.Series,
    y_probas: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    cost_matrix: Optional[BusinessCostMatrix] = None,
) -> pd.DataFrame:
    """Compute precision, recall, F1, and cost across a spectrum of candidate thresholds.

    Operates strictly on training out-of-fold probability estimates.

    Args:
        y_true: Ground truth binary target (0/1).
        y_probas: Probability scores for the positive class.
        thresholds: Array of candidate thresholds. Defaults to np.linspace(0.10, 0.90, 41).
        cost_matrix: Optional BusinessCostMatrix.

    Returns:
        pd.DataFrame: Table of metrics across all candidate thresholds.
    """
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_probas, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 41)

    records: List[Dict[str, Any]] = []

    for t in thresholds:
        preds = (p_arr >= t).astype(int)
        cm = confusion_matrix(y_arr, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        prec = float(precision_score(y_arr, preds, zero_division=0))
        rec = float(recall_score(y_arr, preds, zero_division=0))
        f1 = float(f1_score(y_arr, preds, zero_division=0))

        row: Dict[str, Any] = {
            "threshold": round(float(t), 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

        if cost_matrix is not None:
            total_cost = (fp * cost_matrix.cost_false_positive) + (fn * cost_matrix.cost_false_negative)
            row["total_cost"] = round(total_cost, 2)

        records.append(row)

    return pd.DataFrame.from_records(records)


def optimize_decision_threshold(
    y_train_oof_true: np.ndarray | pd.Series,
    y_train_oof_probas: np.ndarray,
    cost_matrix: Optional[BusinessCostMatrix] = None,
    min_recall_constraint: Optional[float] = None,
    reports_dir: Optional[Path | str] = None,
) -> ThresholdOptimizationResult:
    """Find the optimal decision threshold using training data out-of-fold predictions.

    Selection Logic:
    1. If cost_matrix is supplied: selects threshold minimizing total business loss.
    2. If min_recall_constraint is supplied: selects threshold maximizing precision subject to recall >= constraint.
    3. Default (unspecified business costs): selects threshold maximizing F1-score on training folds.

    Args:
        y_train_oof_true: Ground truth training targets.
        y_train_oof_probas: Out-of-fold probability estimates on training folds.
        cost_matrix: Optional BusinessCostMatrix.
        min_recall_constraint: Optional recall lower bound (e.g. 0.80).
        reports_dir: Directory to save visualization figures.

    Returns:
        ThresholdOptimizationResult: Chosen frozen threshold and analysis table.
    """
    df = compute_threshold_tradeoff_table(
        y_true=y_train_oof_true,
        y_probas=y_train_oof_probas,
        cost_matrix=cost_matrix,
    )

    strategy_desc = "F1-Score Maximization"
    if cost_matrix is not None:
        optimal_idx = int(df["total_cost"].idxmin())
        strategy_desc = (
            f"Cost-Minimization (Cost_FP={cost_matrix.cost_false_positive}, "
            f"Cost_FN={cost_matrix.cost_false_negative})"
        )
    elif min_recall_constraint is not None:
        eligible = df[df["recall"] >= min_recall_constraint]
        if not eligible.empty:
            optimal_idx = int(eligible["precision"].idxmax())
            strategy_desc = f"Constrained Recall Maximization (Recall >= {min_recall_constraint:.0%})"
        else:
            optimal_idx = int(df["f1_score"].idxmax())
    else:
        # Default empirical maximum F1 on training folds
        optimal_idx = int(df["f1_score"].idxmax())
        strategy_desc = "Empirical F1-Score Maximization (Training OOF)"

    optimal_row = df.iloc[optimal_idx].to_dict()
    best_threshold = float(optimal_row["threshold"])

    logger.info(
        f"Optimal threshold selected: {best_threshold:.4f} via [{strategy_desc}] -> "
        f"OOF F1={optimal_row['f1_score']:.4f}, Precision={optimal_row['precision']:.4f}, "
        f"Recall={optimal_row['recall']:.4f}"
    )

    # Plot threshold trade-off curve
    plot_path: Optional[Path] = None
    if reports_dir is not None:
        rep_dir = Path(reports_dir)
        fig_dir = rep_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_path = fig_dir / "threshold_tradeoff.png"
        plot_threshold_curves(
            df_tradeoff=df,
            optimal_threshold=best_threshold,
            output_path=plot_path,
        )

    return ThresholdOptimizationResult(
        optimal_threshold=best_threshold,
        selection_strategy=strategy_desc,
        metrics_at_optimal=optimal_row,
        tradeoff_dataframe=df,
        cost_matrix_applied=(
            {"cost_fp": cost_matrix.cost_false_positive, "cost_fn": cost_matrix.cost_false_negative}
            if cost_matrix
            else None
        ),
        plot_path=plot_path,
    )


def plot_threshold_curves(
    df_tradeoff: pd.DataFrame,
    optimal_threshold: float,
    output_path: Path | str,
) -> Path:
    """Generate trade-off plot illustrating Precision, Recall, and F1 across decision thresholds."""
    plt.figure(figsize=(9, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    plt.plot(df_tradeoff["threshold"], df_tradeoff["precision"], "b-", lw=2, label="Precision")
    plt.plot(df_tradeoff["threshold"], df_tradeoff["recall"], "g-", lw=2, label="Recall")
    plt.plot(df_tradeoff["threshold"], df_tradeoff["f1_score"], "r-", lw=2.5, label="F1-Score")

    # Mark frozen optimal threshold
    plt.axvline(
        x=optimal_threshold,
        color="black",
        linestyle="--",
        lw=1.5,
        label=f"Selected Threshold ($p^* = {optimal_threshold:.2f}$)",
    )

    # Standard reference threshold 0.50
    plt.axvline(
        x=0.50,
        color="gray",
        linestyle=":",
        lw=1.2,
        label="Standard Default ($p = 0.50$)",
    )

    plt.xlabel("Classification Decision Threshold ($p$)", fontsize=11, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=11, fontweight="bold")
    plt.title("Decision Threshold Trade-Off Analysis (Training Out-Of-Fold)", fontsize=13, fontweight="bold")
    plt.legend(loc="best", frameon=True)
    plt.xlim([0.10, 0.90])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file
