"""Baseline model definitions and pipeline builders for PredictIQ.

Constructs:
1. Naive Baseline: DummyClassifier (most frequent class strategy).
2. Linear Baseline: LogisticRegression with L2 regularization and balanced class weighting.
3. Tree-based Baseline: RandomForestClassifier with bounded depth and balanced class weighting.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from predictiq.features.pipeline import build_model_pipeline

logger = logging.getLogger(__name__)


def build_dummy_baseline(strategy: str = "most_frequent") -> Pipeline:
    """Build naive baseline pipeline using scikit-learn DummyClassifier.

    Serves as the empirical floor: any viable predictive model must surpass this baseline.
    """
    model = DummyClassifier(strategy=strategy)
    return build_model_pipeline(classifier=model, scale_numeric=False)


def build_logistic_regression_baseline(
    random_state: int = 42,
    max_iter: int = 1000,
    C: float = 1.0,
    class_weight: Optional[str] = "balanced",
    solver: str = "lbfgs",
) -> Pipeline:
    """Build interpretable linear baseline using Logistic Regression.

    Configuration:
    - Numerical standardization enabled (essential for L2 regularization).
    - Class weight 'balanced' to adjust cost function inversely proportional to class frequencies.
    - Fixed random_state for reproducible optimization.
    """
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        solver=solver,
        random_state=random_state,
    )
    return build_model_pipeline(classifier=model, scale_numeric=True)


def build_random_forest_baseline(
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: Optional[int] = 8,
    min_samples_leaf: int = 5,
    class_weight: Optional[str] = "balanced",
) -> Pipeline:
    """Build non-linear tree ensemble baseline using Random Forest.

    Configuration:
    - Bounded max_depth=8 and min_samples_leaf=5 to prevent overfitting on 7,043 rows.
    - Class weight 'balanced' to improve recall on minority churn instances.
    - Fixed random_state for exact determinism.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    return build_model_pipeline(classifier=model, scale_numeric=False)


def get_all_baselines(random_state: int = 42) -> Dict[str, Pipeline]:
    """Retrieve all baseline pipelines ready for training and comparative evaluation."""
    return {
        "Dummy (Most Frequent)": build_dummy_baseline(strategy="most_frequent"),
        "Logistic Regression": build_logistic_regression_baseline(random_state=random_state),
        "Random Forest": build_random_forest_baseline(random_state=random_state),
    }
