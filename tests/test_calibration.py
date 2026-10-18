"""Deterministic tests for probability calibration and reliability diagrams."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictiq.models.baselines import build_logistic_regression_baseline
from predictiq.models.calibration import (
    CalibrationResult,
    calibrate_model_pipeline,
    plot_calibration_curve,
)


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate small fixture for calibration tests."""
    n = 60
    df = pd.DataFrame({
        "gender": ["Female", "Male"] * 30,
        "SeniorCitizen": [0, 1] * 30,
        "Partner": ["Yes", "No"] * 30,
        "Dependents": ["No", "Yes"] * 30,
        "tenure": list(range(n)),
        "PhoneService": ["Yes"] * n,
        "MultipleLines": ["No", "Yes"] * 30,
        "InternetService": ["DSL", "Fiber optic"] * 30,
        "OnlineSecurity": ["No", "Yes"] * 30,
        "OnlineBackup": ["Yes", "No"] * 30,
        "DeviceProtection": ["No", "Yes"] * 30,
        "TechSupport": ["No", "Yes"] * 30,
        "StreamingTV": ["No", "Yes"] * 30,
        "StreamingMovies": ["No", "Yes"] * 30,
        "Contract": ["Month-to-month", "One year"] * 30,
        "PaperlessBilling": ["Yes", "No"] * 30,
        "PaymentMethod": ["Electronic check", "Mailed check"] * 30,
        "MonthlyCharges": [30.0 + i * 1.0 for i in range(n)],
        "TotalCharges": [100.0 + i * 30.0 for i in range(n)],
    })
    y = pd.Series([0] * 42 + [1] * 18)
    return df, y


def test_calibrate_model_pipeline_execution(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Verify calibration pipeline trains and returns CalibrationResult."""
    X, y = synthetic_data
    base_pipeline = build_logistic_regression_baseline(random_state=42)

    res: CalibrationResult = calibrate_model_pipeline(
        pipeline=base_pipeline,
        X_train=X,
        y_train=y,
        method="sigmoid",
        cv=3,
        reports_dir=tmp_path,
    )

    assert res.method == "sigmoid"
    assert res.calibrated_pipeline is not None
    assert 0.0 <= res.brier_score_calibrated <= 1.0
    assert 0.0 <= res.brier_score_uncalibrated <= 1.0
    assert res.plot_path is not None and res.plot_path.is_file()


def test_plot_calibration_curve(tmp_path: Path) -> None:
    """Verify plot_calibration_curve saves an image file."""
    y = np.array([0, 0, 0, 1, 1, 1])
    p_uncal = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    p_cal = np.array([0.15, 0.25, 0.35, 0.65, 0.75, 0.85])

    plot_file = tmp_path / "test_reliability.png"
    out = plot_calibration_curve(
        y_train=y,
        p_uncalibrated=p_uncal,
        p_calibrated=p_cal,
        brier_uncal=0.15,
        brier_cal=0.12,
        output_path=plot_file,
    )

    assert out.is_file()
    assert out.stat().st_size > 0
