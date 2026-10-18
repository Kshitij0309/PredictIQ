"""Deterministic tests for MLflow experiment tracking manager."""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from predictiq.models.evaluation import ModelEvaluationResult
from predictiq.tracking.manager import (
    MLflowRunResult,
    MLflowTrackingManager,
)


def test_mlflow_local_directory_fallback(tmp_path: Path) -> None:
    """Verify fallback to local directory when remote server is unreachable."""
    unreachable_uri = "http://localhost:59999"  # Port not running
    local_dir = tmp_path / "test_mlruns"

    manager = MLflowTrackingManager(
        tracking_uri=unreachable_uri,
        experiment_name="test-fallback-exp",
        local_fallback_dir=local_dir,
    )

    assert manager.tracking_mode == "local_directory"
    assert local_dir.is_dir()
    assert manager.experiment_name == "test-fallback-exp"


def test_mlflow_log_model_run_locally(tmp_path: Path) -> None:
    """Verify end-to-end logging to local MLflow store without network dependencies."""
    local_dir = tmp_path / "test_mlruns"
    manager = MLflowTrackingManager(
        tracking_uri=local_dir.as_uri(),
        experiment_name="test-local-logging",
        local_fallback_dir=local_dir,
    )

    dummy_pipe = Pipeline([("dummy", DummyClassifier(strategy="most_frequent"))])
    X = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    dummy_pipe.fit(X, y)

    eval_result = ModelEvaluationResult(
        model_name="TestDummy",
        model_type="DummyClassifier",
        accuracy=0.66,
        precision=0.0,
        recall=0.0,
        f1_score=0.0,
        roc_auc=0.50,
        average_precision=0.33,
        brier_score=0.22,
        confusion_matrix=[[2, 0], [1, 0]],
        classification_report_dict={},
    )

    run_res: MLflowRunResult = manager.log_model_run(
        model_name="TestDummy",
        pipeline=dummy_pipe,
        eval_result=eval_result,
        hyperparameters={"strategy": "most_frequent"},
        X_sample=X,
        dataset_meta={"split_seed": 42},
    )

    assert run_res.status == "logged"
    assert len(run_res.run_id) > 0
    assert run_res.tracking_mode == "local_directory"
