"""Hyperparameter optimization and model search for PredictIQ.

Executes reproducible, training-only cross-validation tuning optimizing Average Precision (PR-AUC).
Guarantees that held-out test evaluation data remains strictly isolated.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from predictiq.models.baselines import (
    build_lightgbm_baseline,
    build_logistic_regression_baseline,
    build_random_forest_baseline,
)

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Container for hyperparameter search results."""
    model_name: str
    best_estimator: Pipeline
    best_params: Dict[str, Any]
    best_cv_score: float  # Mean Average Precision across folds
    cv_std_score: float   # Std of Average Precision across folds
    search_space: Dict[str, Any]
    total_candidates_evaluated: int
    runtime_seconds: float
    cv_folds: int
    random_seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "best_cv_pr_auc_mean": round(self.best_cv_score, 4),
            "best_cv_pr_auc_std": round(self.cv_std_score, 4),
            "total_candidates": self.total_candidates_evaluated,
            "runtime_seconds": round(self.runtime_seconds, 2),
            "cv_folds": self.cv_folds,
            "random_seed": self.random_seed,
        }


def get_hyperparameter_search_spaces() -> Dict[str, tuple[Pipeline, Dict[str, Any]]]:
    """Define sensible, computationally reasonable search spaces for candidates."""
    spaces = {
        "LightGBM": (
            build_lightgbm_baseline(),
            {
                "classifier__n_estimators": [60, 100, 140],
                "classifier__learning_rate": [0.03, 0.05, 0.08],
                "classifier__num_leaves": [15, 25, 31],
                "classifier__max_depth": [4, 5, 6],
                "classifier__min_child_samples": [15, 25, 35],
            },
        ),
        "Random Forest": (
            build_random_forest_baseline(),
            {
                "classifier__n_estimators": [80, 120, 160],
                "classifier__max_depth": [6, 8, 10],
                "classifier__min_samples_leaf": [3, 5, 8],
                "classifier__max_features": ["sqrt", 0.7],
            },
        ),
        "Logistic Regression": (
            build_logistic_regression_baseline(),
            {
                "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
                "classifier__solver": ["lbfgs", "saga"],
            },
        ),
    }
    return spaces


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 12,
    cv_folds: int = 5,
    random_state: int = 42,
) -> TuningResult:
    """Tune a single model pipeline on training data using Stratified K-Fold CV.

    Optimization Objective: Average Precision / PR-AUC.

    Args:
        model_name: One of 'LightGBM', 'Random Forest', 'Logistic Regression'.
        X_train: Training features only.
        y_train: Training labels only.
        n_iter: Number of parameter combinations to sample.
        cv_folds: Number of stratified folds.
        random_state: Random seed for exact determinism.

    Returns:
        TuningResult: Structured tuning results with best pipeline.
    """
    spaces = get_hyperparameter_search_spaces()
    if model_name not in spaces:
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {list(spaces.keys())}")

    base_pipeline, param_grid = spaces[model_name]
    start_time = time.time()

    # Stratified K-Fold ensures identical fold assignments across candidate parameter runs
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Scorer targeting PR-AUC (Average Precision) on probability outputs
    scorer = make_scorer(
        average_precision_score,
        response_method="predict_proba",
    )

    # Bounded randomized search to keep execution deterministic and computationally reasonable
    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_grid,
        n_iter=min(n_iter, _calculate_grid_size(param_grid)),
        scoring=scorer,
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        refit=True,  # Refit best estimator on full training set
        verbose=0,
    )

    logger.info(f"Starting hyperparameter tuning for '{model_name}' ({search.n_iter} trials, {cv_folds} folds)...")
    search.fit(X_train, y_train)
    runtime = time.time() - start_time

    best_idx = search.best_index_
    best_mean = float(search.cv_results_["mean_test_score"][best_idx])
    best_std = float(search.cv_results_["std_test_score"][best_idx])

    logger.info(
        f"Tuned '{model_name}' complete in {runtime:.1f}s -> Best CV PR-AUC: {best_mean:.4f} (+/- {best_std:.4f})"
    )

    return TuningResult(
        model_name=model_name,
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        best_cv_score=best_mean,
        cv_std_score=best_std,
        search_space=param_grid,
        total_candidates_evaluated=search.n_iter,
        runtime_seconds=runtime,
        cv_folds=cv_folds,
        random_seed=random_state,
    )


def tune_all_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 12,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, TuningResult]:
    """Execute hyperparameter optimization across all candidate models on training data only."""
    results: Dict[str, TuningResult] = {}
    for name in ["Logistic Regression", "Random Forest", "LightGBM"]:
        results[name] = tune_model(
            model_name=name,
            X_train=X_train,
            y_train=y_train,
            n_iter=n_iter,
            cv_folds=cv_folds,
            random_state=random_state,
        )
    return results


def _calculate_grid_size(param_grid: Dict[str, list]) -> int:
    """Calculate total combinatorial grid space size."""
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size
