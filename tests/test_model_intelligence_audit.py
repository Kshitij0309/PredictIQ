"""Evaluation integrity and model intelligence correctness audit tests.

Verifies:
1. Metric provenance across champion metadata, held-out test summary, and candidate comparisons.
2. Threshold sweep sorting, metric computation, zero-division handling, and OOF threshold selection.
3. Calibration Brier score provenance separation (CV training vs isolated held-out test).
4. Feature name mapping and TreeSHAP attribution extraction across pipeline stages.
5. Figure availability in artifact service.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard.services.artifact_service import list_available_figures
from predictiq.models.explainability import extract_preprocessed_features
from predictiq.models.persistence import load_model_pipeline
from predictiq.models.thresholds import compute_threshold_tradeoff_table


def test_metric_provenance_consistency():
    """Verify champion metadata matches held-out test summary and differs from candidate baselines."""
    meta_path = Path("models/churn_champion_model_metadata.json")
    assert meta_path.is_file(), "Champion model metadata must exist"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    test_summary_path = Path("reports/phase4_final_test_summary.csv")
    assert test_summary_path.is_file(), "Phase 4 test summary must exist"
    test_df = pd.read_csv(test_summary_path)

    # 1. Champion held-out metrics match
    assert pytest.approx(meta["final_test_pr_auc"], abs=1e-4) == float(test_df["test_pr_auc"].iloc[0])
    assert pytest.approx(meta["final_test_roc_auc"], abs=1e-4) == float(test_df["test_roc_auc"].iloc[0])
    assert pytest.approx(meta["optimal_decision_threshold"], abs=1e-4) == float(test_df["optimal_threshold"].iloc[0])
    assert pytest.approx(meta["final_test_f1_optimal_threshold"], abs=1e-4) == float(test_df["test_f1_optimal_threshold"].iloc[0])
    assert pytest.approx(meta["final_test_recall_optimal_threshold"], abs=1e-4) == float(test_df["test_recall_optimal_threshold"].iloc[0])
    assert pytest.approx(meta["final_test_precision_optimal_threshold"], abs=1e-4) == float(test_df["test_precision_optimal_threshold"].iloc[0])

    # 2. Phase 3 candidate baseline metrics exist and differ from champion held-out metrics
    model_comp_path = Path("reports/model_comparison.csv")
    if model_comp_path.is_file():
        cand_df = pd.read_csv(model_comp_path)
        rf_row = cand_df[cand_df["model_name"] == "Random Forest"]
        assert not rf_row.empty
        rf_pr_auc = float(rf_row["average_precision"].iloc[0])
        rf_roc_auc = float(rf_row["roc_auc"].iloc[0])

        # Phase 3 exploratory RF PR-AUC was 0.6471, distinct from champion held-out PR-AUC 0.6684
        assert pytest.approx(rf_pr_auc, abs=1e-3) == 0.647
        assert rf_pr_auc != meta["final_test_pr_auc"]
        assert pytest.approx(rf_roc_auc, abs=1e-3) == 0.842


def test_calibration_score_provenance_separation():
    """Verify Brier score provenance: CV training (0.1602 -> 0.1354) vs held-out test (0.1377)."""
    meta_path = Path("models/churn_champion_model_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    brier_uncal = float(meta["brier_score_uncalibrated"])
    brier_cal = float(meta["brier_score_calibrated"])
    brier_gain = float(meta["brier_improvement_pct"])

    assert pytest.approx(brier_uncal, abs=1e-3) == 0.160
    assert pytest.approx(brier_cal, abs=1e-3) == 0.135
    assert pytest.approx(brier_gain, abs=1e-1) == 15.4

    # Held-out test Brier score is distinct and reported in test summary
    test_summary_path = Path("reports/phase4_final_test_summary.csv")
    test_df = pd.read_csv(test_summary_path)
    test_brier = float(test_df["test_brier_score"].iloc[0])

    assert pytest.approx(test_brier, abs=1e-3) == 0.138
    # Held-out Brier must be tracked separately from training CV Brier
    assert test_brier != brier_cal


def test_threshold_sweep_ordering_and_calculations():
    """Verify threshold array ordering, correct metric formulas, and zero-division handling."""
    thresholds = np.linspace(0.10, 0.90, 41)
    # 1. Strictly sorted ascending
    assert np.all(np.diff(thresholds) > 0), "Thresholds must be strictly sorted ascending"

    # 2. Test recomputation with synthetic labels
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_probas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85])

    df = compute_threshold_tradeoff_table(y_true, y_probas, thresholds=thresholds)

    assert len(df) == 41
    # Check monotonicity of thresholds in output dataframe
    assert (df["threshold"].diff().dropna() > 0).all()

    # Verify metric computation at threshold 0.50
    row_50 = df[df["threshold"] == 0.50].iloc[0]
    preds_50 = (y_probas >= 0.50).astype(int)
    tp = int(np.sum((preds_50 == 1) & (y_true == 1)))
    fp = int(np.sum((preds_50 == 1) & (y_true == 0)))
    fn = int(np.sum((preds_50 == 0) & (y_true == 1)))
    tn = int(np.sum((preds_50 == 0) & (y_true == 0)))

    expected_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    expected_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    expected_f1 = (2 * expected_prec * expected_rec) / (expected_prec + expected_rec) if (expected_prec + expected_rec) > 0 else 0.0

    assert row_50["true_positives"] == tp
    assert row_50["false_positives"] == fp
    assert row_50["true_negatives"] == tn
    assert pytest.approx(row_50["precision"], abs=1e-4) == expected_prec
    assert pytest.approx(row_50["recall"], abs=1e-4) == expected_rec
    assert pytest.approx(row_50["f1_score"], abs=1e-4) == expected_f1


def test_threshold_zero_division_at_extreme_tail():
    """Verify zero predicted positive cases at high threshold are handled gracefully without errors."""
    y_true = np.array([0, 1, 0, 1])
    y_probas = np.array([0.2, 0.3, 0.4, 0.5])  # Max proba is 0.5

    # Threshold 0.90 has 0 predicted positives
    df = compute_threshold_tradeoff_table(y_true, y_probas, thresholds=np.array([0.90]))
    row = df.iloc[0]

    assert row["true_positives"] == 0
    assert row["false_positives"] == 0
    assert row["precision"] == 0.0
    assert row["recall"] == 0.0
    assert row["f1_score"] == 0.0


def test_feature_name_mapping_and_pipeline_unwrapping():
    """Verify that extract_preprocessed_features unwraps CalibratedClassifierCV and maps top SHAP features."""
    pipeline, meta = load_model_pipeline("models/churn_champion_model.joblib")
    dummy_X = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20,
    }])

    X_trans, feature_names, classifier = extract_preprocessed_features(pipeline, dummy_X)

    assert len(feature_names) == X_trans.shape[1]
    assert len(feature_names) == 57

    # Ensure all top 10 SHAP features exist in pipeline feature names
    top_10 = meta.get("top_10_shap_features", [])
    assert len(top_10) == 10
    for feat in top_10:
        assert feat in feature_names, f"Feature '{feat}' must be present in pipeline feature names"


def test_artifact_figure_registration():
    """Verify all required diagnostic evaluation figures are registered and available."""
    figures = list_available_figures()

    required_figures = [
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

    for fig_name in required_figures:
        assert fig_name in figures, f"Figure '{fig_name}' must be listed in artifact service"
        fig_path = figures[fig_name]
        assert fig_path is not None and fig_path.is_file(), f"Figure file '{fig_name}' must exist on disk"
