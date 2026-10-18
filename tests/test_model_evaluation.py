"""Deterministic tests for model evaluation, metrics, and visualization artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from predictiq.models.evaluation import (
    ModelEvaluationResult,
    build_comparison_dataframe,
    evaluate_model,
    save_evaluation_artifacts,
)


@pytest.fixture
def evaluation_fixture() -> tuple[pd.DataFrame, pd.Series]:
    """Provide synthetic features and targets for evaluation testing."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "num1": np.random.randn(n),
        "num2": np.random.randn(n),
    })
    y = pd.Series([0] * 40 + [1] * 20)
    return X, y


def test_evaluate_model_with_probabilities(
    evaluation_fixture: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify evaluation metrics on a model that outputs probabilities."""
    X, y = evaluation_fixture
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    res: ModelEvaluationResult = evaluate_model(
        model=model,
        X_test=X,
        y_test=y,
        model_name="TestLogistic",
        model_type="LogisticRegression",
    )

    assert res.model_name == "TestLogistic"
    assert 0.0 <= res.accuracy <= 1.0
    assert 0.0 <= res.precision <= 1.0
    assert 0.0 <= res.recall <= 1.0
    assert 0.0 <= res.f1_score <= 1.0
    assert res.roc_auc is not None and 0.0 <= res.roc_auc <= 1.0
    assert res.average_precision is not None and 0.0 <= res.average_precision <= 1.0
    assert res.brier_score is not None and 0.0 <= res.brier_score <= 1.0
    assert len(res.confusion_matrix) == 2
    assert "accuracy" in res.classification_report_dict


def test_evaluate_model_dummy_baseline(
    evaluation_fixture: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify evaluation handles models with degenerate probabilities gracefully."""
    X, y = evaluation_fixture
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)

    res = evaluate_model(
        model=model,
        X_test=X,
        y_test=y,
        model_name="TestDummy",
        model_type="DummyClassifier",
    )

    assert res.accuracy == 40 / 60  # Most frequent class is 0 (40/60)
    assert res.recall == 0.0  # Positive class 1 was never predicted
    assert res.precision == 0.0


def test_build_comparison_dataframe() -> None:
    """Verify comparison table aggregates and sorts models by primary metric (PR-AUC)."""
    res1 = ModelEvaluationResult(
        model_name="ModelA",
        model_type="Linear",
        accuracy=0.80,
        precision=0.70,
        recall=0.60,
        f1_score=0.65,
        roc_auc=0.85,
        average_precision=0.68,
        brier_score=0.15,
        confusion_matrix=[[30, 5], [10, 15]],
        classification_report_dict={},
    )
    res2 = ModelEvaluationResult(
        model_name="ModelB",
        model_type="Tree",
        accuracy=0.82,
        precision=0.75,
        recall=0.65,
        f1_score=0.70,
        roc_auc=0.88,
        average_precision=0.74,
        brier_score=0.12,
        confusion_matrix=[[32, 3], [8, 17]],
        classification_report_dict={},
    )

    df = build_comparison_dataframe([res1, res2])
    assert len(df) == 2
    # ModelB has higher average_precision (0.74 > 0.68), should be row 0
    assert df.iloc[0]["model_name"] == "ModelB"
    assert df.iloc[1]["model_name"] == "ModelA"


def test_save_evaluation_artifacts(tmp_path: Path) -> None:
    """Verify generation of comparison artifacts (CSV, JSON, PNGs)."""
    res = ModelEvaluationResult(
        model_name="TestModel",
        model_type="TestClassifier",
        accuracy=0.85,
        precision=0.80,
        recall=0.70,
        f1_score=0.75,
        roc_auc=0.88,
        average_precision=0.72,
        brier_score=0.14,
        confusion_matrix=[[40, 5], [8, 17]],
        classification_report_dict={"1": {"f1-score": 0.75}},
        y_true=[0] * 45 + [1] * 25,
        y_pred=[0] * 42 + [1] * 28,
        y_proba=[0.1] * 45 + [0.8] * 25,
    )

    artifacts = save_evaluation_artifacts([res], reports_dir=tmp_path)
    assert artifacts["comparison_csv"].is_file()
    assert artifacts["comparison_json"].is_file()
    assert artifacts["roc_curve"].is_file()
    assert artifacts["pr_curve"].is_file()
    assert artifacts["confusion_matrix"].is_file()
