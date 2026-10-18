"""End-to-end model training, cross-validation, and MLflow orchestration for PredictIQ.

Executes leakage-safe training, 5-fold cross-validation, held-out test evaluation,
artifact generation, MLflow experiment logging, and best-model persistence.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from predictiq.config import get_config
from predictiq.data.loader import DatasetSplits, load_and_split_data
from predictiq.models.baselines import get_all_baselines
from predictiq.models.evaluation import (
    ModelEvaluationResult,
    build_comparison_dataframe,
    evaluate_model,
    save_evaluation_artifacts,
)
from predictiq.models.persistence import save_model_pipeline
from predictiq.tracking.manager import MLflowTrackingManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_cross_validation_evaluation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[float, float, float, float]:
    """Execute leakage-safe Stratified K-Fold cross-validation on training data only.

    Returns:
        Tuple[float, float, float, float]: (mean_pr_auc, std_pr_auc, mean_roc_auc, std_roc_auc)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pr_aucs: List[float] = []
    roc_aucs: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Fit fresh pipeline clone strictly on fold training data
        from sklearn.base import clone

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_tr, y_tr)

        if hasattr(fold_pipeline, "predict_proba"):
            try:
                probas = fold_pipeline.predict_proba(X_val)
                p1 = probas[:, 1] if probas.shape[1] >= 2 else probas[:, 0]
                pr_aucs.append(float(average_precision_score(y_val, p1)))
                roc_aucs.append(float(roc_auc_score(y_val, p1)))
            except Exception:
                pr_aucs.append(float(np.mean(y_val)))
                roc_aucs.append(0.5)
        else:
            pr_aucs.append(float(np.mean(y_val)))
            roc_aucs.append(0.5)

    return (
        float(np.mean(pr_aucs)),
        float(np.std(pr_aucs)),
        float(np.mean(roc_aucs)),
        float(np.std(roc_aucs)),
    )


def run_baseline_training_pipeline(
    data_path: Optional[Path | str] = None,
    test_size: float = 0.20,
    random_state: int = 42,
    cv_splits: int = 5,
    primary_metric: str = "average_precision",
    save_reports: bool = True,
    log_to_mlflow: bool = True,
    persist_best_model: bool = True,
) -> Dict[str, Any]:
    """Execute complete Phase 3 baseline modeling workflow.

    Workflow:
    1. Load and split dataset (80% train, 20% test, stratified, seed 42).
    2. Build naive Dummy baseline, Logistic Regression, and Random Forest pipelines.
    3. Run 5-fold cross-validation on train data to estimate out-of-fold generalization.
    4. Fit pipelines on full train set and evaluate on held-out test set.
    5. Generate comparison table, ROC/PR curves, and confusion matrices.
    6. Log runs to MLflow (using local fallback if server is offline).
    7. Persist best model pipeline and metadata according to primary metric (PR-AUC).

    Returns:
        Dict[str, Any]: Comprehensive summary dictionary.
    """
    logger.info("=" * 70)
    logger.info("PredictIQ Phase 3: Baseline Model Training & Experiment Tracking")
    logger.info("=" * 70)

    # 1. Load data
    logger.info(f"Loading and splitting dataset (test_size={test_size}, random_state={random_state})...")
    splits: DatasetSplits = load_and_split_data(
        data_path=data_path,
        test_size=test_size,
        random_state=random_state,
    )
    logger.info(
        f"Data prepared: Train={splits.train_size} (Churn={splits.train_positive_ratio:.1%}), "
        f"Test={splits.test_size} (Churn={splits.test_positive_ratio:.1%}), "
        f"Features={splits.n_features}"
    )

    # 2. Initialize tracking manager
    tracking_manager = MLflowTrackingManager() if log_to_mlflow else None

    # 3. Model candidate definitions
    model_candidates = get_all_baselines(random_state=random_state)
    evaluation_results: List[ModelEvaluationResult] = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    # 4. Train, CV, and evaluate each candidate
    for model_name, pipeline in model_candidates.items():
        logger.info(f"--- Processing Candidate: {model_name} ---")

        # Cross-validation on train data only
        logger.info(f"Running {cv_splits}-fold Stratified CV on training set...")
        mean_pr, std_pr, mean_roc, std_roc = run_cross_validation_evaluation(
            pipeline=pipeline,
            X_train=splits.X_train,
            y_train=splits.y_train,
            n_splits=cv_splits,
            random_state=random_state,
        )
        logger.info(f"CV PR-AUC: {mean_pr:.4f} (+/- {std_pr:.4f}) | CV ROC-AUC: {mean_roc:.4f}")

        # Fit on full training set
        logger.info("Fitting on complete training set...")
        pipeline.fit(splits.X_train, splits.y_train)
        fitted_pipelines[model_name] = pipeline

        # Evaluate on held-out test set
        logger.info("Evaluating on held-out test set...")
        model_type = pipeline.named_steps["classifier"].__class__.__name__
        eval_result = evaluate_model(
            model=pipeline,
            X_test=splits.X_test,
            y_test=splits.y_test,
            model_name=model_name,
            model_type=model_type,
            cv_mean_pr_auc=mean_pr,
            cv_std_pr_auc=std_pr,
            cv_mean_roc_auc=mean_roc,
            cv_std_roc_auc=std_roc,
            store_predictions=True,
        )
        evaluation_results.append(eval_result)
        logger.info(
            f"Test Metrics -> PR-AUC: {eval_result.average_precision} | "
            f"ROC-AUC: {eval_result.roc_auc} | F1: {eval_result.f1_score:.4f} | "
            f"Acc: {eval_result.accuracy:.4f} | Rec: {eval_result.recall:.4f}"
        )

    # 5. Save evaluation artifacts
    artifact_paths: Dict[str, Path] = {}
    if save_reports:
        logger.info("Generating comparison tables and figures...")
        config = get_config()
        rep_dir = config.paths.root_dir / "reports"
        artifact_paths = save_evaluation_artifacts(evaluation_results, reports_dir=rep_dir)
        logger.info(f"Evaluation artifacts saved to: {rep_dir}")

    # 6. MLflow logging
    mlflow_runs: Dict[str, Any] = {}
    if tracking_manager is not None:
        logger.info(f"Logging runs to MLflow (Mode: {tracking_manager.tracking_mode})...")
        for res in evaluation_results:
            pipe = fitted_pipelines[res.model_name]
            classifier = pipe.named_steps["classifier"]
            params = classifier.get_params()
            params["scale_numeric"] = pipe.named_steps["preprocessor"].transformers[0][1].steps[-1][0] == "scaler"

            run_res = tracking_manager.log_model_run(
                model_name=res.model_name,
                pipeline=pipe,
                eval_result=res,
                hyperparameters=params,
                X_sample=splits.X_train.head(5),
                dataset_meta={
                    "train_size": splits.train_size,
                    "test_size": splits.test_size,
                    "split_seed": random_state,
                    "target": splits.target_name,
                },
                artifact_paths=artifact_paths,
            )
            mlflow_runs[res.model_name] = run_res

    # 7. Model selection based on primary metric
    comp_df = build_comparison_dataframe(evaluation_results)
    best_row = comp_df.iloc[0]
    best_model_name = str(best_row["model_name"])
    best_pipeline = fitted_pipelines[best_model_name]
    logger.info(
        f"Selected Best Model: '{best_model_name}' based on {primary_metric} = "
        f"{best_row.get(primary_metric, 'N/A')}"
    )

    # 8. Model persistence
    saved_model_paths: Optional[Tuple[Path, Path]] = None
    if persist_best_model:
        metadata = {
            "selected_model_name": best_model_name,
            "primary_metric": primary_metric,
            "primary_metric_score": best_row.get(primary_metric),
            "test_summary": best_row.to_dict(),
            "train_size": splits.train_size,
            "test_size": splits.test_size,
            "random_seed": random_state,
            "cv_splits": cv_splits,
            "features_in": list(splits.X_train.columns),
            "python_version": sys.version,
        }
        saved_model_paths = save_model_pipeline(
            pipeline=best_pipeline,
            metadata=metadata,
            model_name="churn_baseline_model",
        )

    logger.info("=" * 70)
    logger.info("Phase 3 Baseline Modeling Workflow Completed Successfully.")
    logger.info("=" * 70)

    return {
        "comparison_dataframe": comp_df,
        "evaluation_results": evaluation_results,
        "best_model_name": best_model_name,
        "best_model_metrics": best_row.to_dict(),
        "saved_model_paths": saved_model_paths,
        "artifact_paths": artifact_paths,
        "mlflow_runs": mlflow_runs,
    }


if __name__ == "__main__":
    results = run_baseline_training_pipeline()
    print("\n--- Model Comparison Summary ---")
    print(results["comparison_dataframe"].to_string(index=False))
