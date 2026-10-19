"""Unit tests for the artifact loader service."""

from __future__ import annotations

from dashboard.services.artifact_service import (
    get_figure_path,
    list_available_figures,
    load_cv_model_comparison,
    load_final_test_summary,
    load_model_metadata,
)


def test_load_model_metadata():
    data, is_loaded = load_model_metadata()
    if is_loaded:
        assert "model_name" in data
        assert "optimal_decision_threshold" in data
        assert data["optimal_decision_threshold"] == 0.34
    else:
        assert "error" in data


def test_load_cv_model_comparison():
    df, is_loaded = load_cv_model_comparison()
    if is_loaded:
        assert not df.empty
        assert "model_name" in df.columns
        assert "cv_mean_pr_auc" in df.columns


def test_load_final_test_summary():
    df, is_loaded = load_final_test_summary()
    if is_loaded:
        assert not df.empty
        assert "optimal_threshold" in df.columns


def test_get_figure_path_nonexistent():
    path = get_figure_path("definitely_nonexistent_figure_12345.png")
    assert path is None


def test_list_available_figures():
    figures = list_available_figures()
    assert isinstance(figures, dict)
    assert "calibration_curve.png" in figures
    assert "champion_heldout_curves.png" in figures
    assert "threshold_tradeoff.png" in figures
    assert "shap_summary.png" in figures
