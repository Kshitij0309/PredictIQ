"""Deterministic tests for model persistence, deserialization, and inference engine."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from predictiq.features.pipeline import build_model_pipeline
from predictiq.inference.predictor import (
    ChurnPredictor,
    InferenceValidationError,
)
from predictiq.models.persistence import (
    load_model_pipeline,
    save_model_pipeline,
)


@pytest.fixture
def sample_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Provide minimal training dataset with all raw features."""
    n = 20
    df = pd.DataFrame({
        "gender": ["Female", "Male"] * 10,
        "SeniorCitizen": [0, 1] * 10,
        "Partner": ["Yes", "No"] * 10,
        "Dependents": ["No", "Yes"] * 10,
        "tenure": list(range(n)),
        "PhoneService": ["Yes"] * n,
        "MultipleLines": ["No", "Yes"] * 10,
        "InternetService": ["DSL", "Fiber optic"] * 10,
        "OnlineSecurity": ["No", "Yes"] * 10,
        "OnlineBackup": ["Yes", "No"] * 10,
        "DeviceProtection": ["No", "Yes"] * 10,
        "TechSupport": ["No", "Yes"] * 10,
        "StreamingTV": ["No", "Yes"] * 10,
        "StreamingMovies": ["No", "Yes"] * 10,
        "Contract": ["Month-to-month", "One year"] * 10,
        "PaperlessBilling": ["Yes", "No"] * 10,
        "PaymentMethod": ["Electronic check", "Mailed check"] * 10,
        "MonthlyCharges": [30.0 + i * 2.0 for i in range(n)],
        "TotalCharges": [100.0 + i * 50.0 for i in range(n)],
    })
    y = pd.Series([0, 1] * 10)
    return df, y


def test_pipeline_persistence_and_reload(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Verify saved pipeline produces identical predictions after reloading."""
    X, y = sample_dataset
    pipeline = build_model_pipeline(LogisticRegression(random_state=42))
    pipeline.fit(X, y)

    # Initial predictions
    preds_before = pipeline.predict(X)
    probas_before = pipeline.predict_proba(X)

    # Save
    meta = {"model": "LogisticRegression", "author": "PredictIQ Test"}
    joblib_path, meta_path = save_model_pipeline(
        pipeline=pipeline,
        metadata=meta,
        output_dir=tmp_path,
        model_name="test_model",
    )

    assert joblib_path.is_file()
    assert meta_path.is_file()

    # Reload
    reloaded_pipeline, loaded_meta = load_model_pipeline(
        model_path=joblib_path,
        metadata_path=meta_path,
    )

    preds_after = reloaded_pipeline.predict(X)
    probas_after = reloaded_pipeline.predict_proba(X)

    np.testing.assert_array_equal(preds_before, preds_after)
    np.testing.assert_allclose(probas_before, probas_after, rtol=1e-5)
    assert loaded_meta["author"] == "PredictIQ Test"


def test_churn_predictor_inference_interface(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify ChurnPredictor methods: predict, predict_proba, predict_detailed."""
    X, y = sample_dataset
    pipeline = build_model_pipeline(LogisticRegression(random_state=42))
    pipeline.fit(X, y)

    predictor = ChurnPredictor(pipeline=pipeline)

    preds = predictor.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})

    probas = predictor.predict_proba(X)
    assert len(probas) == len(X)
    assert ((probas >= 0.0) & (probas <= 1.0)).all()

    detailed = predictor.predict_detailed(X)
    assert "predicted_churn" in detailed.columns
    assert "churn_probability" in detailed.columns
    assert "risk_tier" in detailed.columns
    assert set(detailed["risk_tier"].unique()).issubset({"Low Risk", "Medium Risk", "High Risk"})


def test_churn_predictor_handles_unseen_categories(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify inference gracefully handles completely new categorical levels."""
    X, y = sample_dataset
    pipeline = build_model_pipeline(LogisticRegression(random_state=42))
    pipeline.fit(X, y)

    predictor = ChurnPredictor(pipeline=pipeline)

    novel_input = X.head(2).copy()
    novel_input.loc[0, "PaymentMethod"] = "Novel Payment System"
    novel_input.loc[0, "Contract"] = "Unseen Contract Option"

    preds = predictor.predict(novel_input)
    assert len(preds) == 2
    assert not np.isnan(preds).any()


def test_churn_predictor_input_validation(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify ChurnPredictor rejects empty or missing columns."""
    X, y = sample_dataset
    pipeline = build_model_pipeline(LogisticRegression(random_state=42))
    pipeline.fit(X, y)

    predictor = ChurnPredictor(pipeline=pipeline)

    # Empty DataFrame
    with pytest.raises(InferenceValidationError, match="contains zero records"):
        predictor.predict(pd.DataFrame())

    # Empty list
    with pytest.raises(InferenceValidationError, match="record list is empty"):
        predictor.predict([])

    # Missing core column (e.g. MonthlyCharges dropped)
    bad_df = X.drop(columns=["MonthlyCharges"])
    with pytest.raises(InferenceValidationError, match="missing 1 required features"):
        predictor.predict(bad_df)
