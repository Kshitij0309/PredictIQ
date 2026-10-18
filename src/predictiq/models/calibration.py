"""Probability calibration and reliability diagnostics for PredictIQ.

Evaluates and corrects predicted probability distortions (particularly pronounced in class-weighted models)
using training-only Platt scaling (sigmoid calibration) and Brier score loss minimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Summary of probability calibration performance and artifacts."""
    method: str
    calibrated_pipeline: CalibratedClassifierCV
    brier_score_uncalibrated: float
    brier_score_calibrated: float
    brier_improvement_pct: float
    plot_path: Optional[Path] = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "brier_score_uncalibrated": round(self.brier_score_uncalibrated, 4),
            "brier_score_calibrated": round(self.brier_score_calibrated, 4),
            "brier_improvement_pct": round(self.brier_improvement_pct, 2),
            "plot_path": str(self.plot_path) if self.plot_path else None,
        }


def calibrate_model_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "sigmoid",
    cv: int = 5,
    reports_dir: Optional[Path | str] = None,
) -> CalibrationResult:
    """Calibrate model probability predictions using training data only.

    Rationale:
    Class-weighted models (e.g. class_weight='balanced') artificially shift raw predicted
    probabilities upward to prioritize recall. Calibrating probabilities with sigmoid
    (Platt scaling) via internal cross-validation restores empirical alignment with true risk rates.

    Args:
        pipeline: Base pipeline (can be unfitted or fitted; CalibratedClassifierCV refits across folds).
        X_train: Training features only.
        y_train: Training labels only.
        method: 'sigmoid' (Platt scaling) or 'isotonic'. Defaults to 'sigmoid'.
        cv: Number of cross-validation folds for calibration fitting.
        reports_dir: Optional output directory for calibration plots.

    Returns:
        CalibrationResult: Calibrated estimator and diagnostic metrics.
    """
    logger.info(f"Calibrating model probabilities using method='{method}' ({cv}-fold CV on train)...")

    # 1. Fit uncalibrated pipeline on train to compute baseline uncalibrated Brier score on CV folds
    from sklearn.model_selection import cross_val_predict

    raw_probas = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")
    p_uncal = raw_probas[:, 1]
    brier_uncal = float(brier_score_loss(y_train, p_uncal))

    # 2. Fit CalibratedClassifierCV on training data only
    calibrated_clf = CalibratedClassifierCV(
        estimator=pipeline,
        method=method,
        cv=cv,
        n_jobs=-1,
    )
    calibrated_clf.fit(X_train, y_train)

    # 3. Compute calibrated cross-val probabilities on train
    cal_probas = cross_val_predict(calibrated_clf, X_train, y_train, cv=cv, method="predict_proba")
    p_cal = cal_probas[:, 1]
    brier_cal = float(brier_score_loss(y_train, p_cal))

    improvement_pct = ((brier_uncal - brier_cal) / brier_uncal) * 100.0
    logger.info(
        f"Calibration complete: Brier score {brier_uncal:.4f} -> {brier_cal:.4f} "
        f"({improvement_pct:+.2f}% improvement)"
    )

    # 4. Generate diagnostic Reliability Diagram (Calibration Curve)
    plot_path: Optional[Path] = None
    if reports_dir is not None:
        rep_dir = Path(reports_dir)
        fig_dir = rep_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_path = fig_dir / "calibration_curve.png"
        plot_calibration_curve(
            y_train=y_train,
            p_uncalibrated=p_uncal,
            p_calibrated=p_cal,
            brier_uncal=brier_uncal,
            brier_cal=brier_cal,
            output_path=plot_path,
        )

    return CalibrationResult(
        method=method,
        calibrated_pipeline=calibrated_clf,
        brier_score_uncalibrated=brier_uncal,
        brier_score_calibrated=brier_cal,
        brier_improvement_pct=improvement_pct,
        plot_path=plot_path,
    )


def plot_calibration_curve(
    y_train: pd.Series | np.ndarray,
    p_uncalibrated: np.ndarray,
    p_calibrated: np.ndarray,
    brier_uncal: float,
    brier_cal: float,
    output_path: Path | str,
    n_bins: int = 10,
) -> Path:
    """Plot reliability diagram comparing uncalibrated vs calibrated probability estimates."""
    y_arr = np.asarray(y_train, dtype=int)
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_arr, p_uncalibrated, n_bins=n_bins, strategy="uniform")
    prob_true_cal, prob_pred_cal = calibration_curve(y_arr, p_calibrated, n_bins=n_bins, strategy="uniform")

    plt.figure(figsize=(8, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Perfectly calibrated reference line
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", lw=1.5)

    # Uncalibrated curve
    plt.plot(
        prob_pred_uncal,
        prob_true_uncal,
        "s-",
        color="crimson",
        lw=2,
        label=f"Uncalibrated (Brier = {brier_uncal:.4f})",
    )

    # Calibrated curve
    plt.plot(
        prob_pred_cal,
        prob_true_cal,
        "o-",
        color="royalblue",
        lw=2,
        label=f"Calibrated (Sigmoid) (Brier = {brier_cal:.4f})",
    )

    plt.xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    plt.ylabel("Observed Fraction of Positives (Actual Churn)", fontsize=11, fontweight="bold")
    plt.title("Reliability Diagram: Probability Calibration Analysis", fontsize=13, fontweight="bold")
    plt.legend(loc="upper left", frameon=True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file
