"""Deterministic tests for model explainability and SHAP attribution analysis."""

from pathlib import Path

import pandas as pd
import pytest

from predictiq.models.baselines import build_random_forest_baseline
from predictiq.models.explainability import (
    SHAPExplanationResult,
    generate_shap_explanations,
)


@pytest.fixture
def explainability_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provide synthetic dataset for SHAP tests."""
    n = 40
    df = pd.DataFrame({
        "gender": ["Female", "Male"] * 20,
        "SeniorCitizen": [0, 1] * 20,
        "Partner": ["Yes", "No"] * 20,
        "Dependents": ["No", "Yes"] * 20,
        "tenure": list(range(n)),
        "PhoneService": ["Yes"] * n,
        "MultipleLines": ["No", "Yes"] * 20,
        "InternetService": ["DSL", "Fiber optic"] * 20,
        "OnlineSecurity": ["No", "Yes"] * 20,
        "OnlineBackup": ["Yes", "No"] * 20,
        "DeviceProtection": ["No", "Yes"] * 20,
        "TechSupport": ["No", "Yes"] * 20,
        "StreamingTV": ["No", "Yes"] * 20,
        "StreamingMovies": ["No", "Yes"] * 20,
        "Contract": ["Month-to-month", "One year"] * 20,
        "PaperlessBilling": ["Yes", "No"] * 20,
        "PaymentMethod": ["Electronic check", "Mailed check"] * 20,
        "MonthlyCharges": [30.0 + i * 1.5 for i in range(n)],
        "TotalCharges": [100.0 + i * 40.0 for i in range(n)],
    })
    y = pd.Series([0] * 30 + [1] * 10)
    return df, y


def test_generate_shap_explanations(
    explainability_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Verify SHAP attributions, feature ranking, and plot generation."""
    X, y = explainability_data
    pipeline = build_random_forest_baseline(random_state=42, n_estimators=10, max_depth=3)
    pipeline.fit(X, y)

    res: SHAPExplanationResult = generate_shap_explanations(
        pipeline=pipeline,
        X_sample=X,
        reports_dir=tmp_path,
        max_eval_samples=40,
    )

    assert len(res.top_10_features) == 10
    assert not res.global_importance_df.empty
    assert res.summary_plot_path is not None and res.summary_plot_path.is_file()
    assert res.local_high_risk_plot_path is not None and res.local_high_risk_plot_path.is_file()
    assert res.local_low_risk_plot_path is not None and res.local_low_risk_plot_path.is_file()
    assert len(res.archetype_insights) == 2
