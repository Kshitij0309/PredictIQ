"""Deterministic tests for decision threshold optimization and business trade-offs."""

from pathlib import Path

import numpy as np
import pytest

from predictiq.models.thresholds import (
    BusinessCostMatrix,
    ThresholdOptimizationResult,
    compute_threshold_tradeoff_table,
    optimize_decision_threshold,
)


@pytest.fixture
def threshold_sample() -> tuple[np.ndarray, np.ndarray]:
    """Provide synthetic labels and predicted probabilities."""
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    probas = np.array([0.05, 0.12, 0.25, 0.35, 0.45, 0.55, 0.40, 0.65, 0.80, 0.95])
    return y, probas


def test_compute_threshold_tradeoff_table(
    threshold_sample: tuple[np.ndarray, np.ndarray],
) -> None:
    """Verify table outputs precision, recall, F1 across candidates."""
    y, probas = threshold_sample
    df = compute_threshold_tradeoff_table(y, probas, thresholds=np.array([0.20, 0.50, 0.80]))

    assert len(df) == 3
    assert "threshold" in df.columns
    assert "precision" in df.columns
    assert "recall" in df.columns
    assert "f1_score" in df.columns
    # At low threshold, recall should be high
    assert df.loc[df["threshold"] == 0.20, "recall"].values[0] >= df.loc[df["threshold"] == 0.80, "recall"].values[0]


def test_optimize_decision_threshold_f1(
    threshold_sample: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    """Verify threshold optimization selects F1-maximizing threshold when costs are unspecified."""
    y, probas = threshold_sample
    res: ThresholdOptimizationResult = optimize_decision_threshold(
        y_train_oof_true=y,
        y_train_oof_probas=probas,
        reports_dir=tmp_path,
    )

    assert 0.10 <= res.optimal_threshold <= 0.90
    assert "F1-Score Maximization" in res.selection_strategy
    assert res.plot_path is not None and res.plot_path.is_file()


def test_optimize_decision_threshold_with_cost_matrix(
    threshold_sample: tuple[np.ndarray, np.ndarray],
) -> None:
    """Verify cost matrix influences optimal threshold selection."""
    y, probas = threshold_sample
    # If false negative cost is very high relative to false positive, optimal threshold should decrease to boost recall
    high_fn_costs = BusinessCostMatrix(cost_false_positive=10.0, cost_false_negative=500.0)
    res_high_fn = optimize_decision_threshold(y, probas, cost_matrix=high_fn_costs)

    # If false positive cost is very high relative to false negative, optimal threshold should increase to boost precision
    high_fp_costs = BusinessCostMatrix(cost_false_positive=500.0, cost_false_negative=10.0)
    res_high_fp = optimize_decision_threshold(y, probas, cost_matrix=high_fp_costs)

    assert res_high_fn.optimal_threshold <= res_high_fp.optimal_threshold
