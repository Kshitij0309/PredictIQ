"""End-to-end model training, cross-validation, and MLflow orchestration for PredictIQ.

Enforces Evaluation Protocol Independence:
1. Model selection and hyperparameter optimization strictly use training-only cross-validation.
2. Probability calibration and decision threshold selection are performed on training folds.
3. The selected model, calibration, and threshold are frozen before evaluating the isolated test set.
4. The held-out test evaluation is performed exactly once for final unbiased reporting.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from predictiq.config import get_config
from predictiq.data.loader import (
    DEFAULT_EXPLORATORY_SPLIT_SEED,
    ISOLATED_EVALUATION_SPLIT_SEED,
    DatasetSplits,
    load_and_split_data,
    load_and_split_isolated_evaluation_data,
)
from predictiq.models.baselines import get_all_baselines
from predictiq.models.calibration import calibrate_model_pipeline
from predictiq.models.evaluation import (
    ModelEvaluationResult,
    build_comparison_dataframe,
    evaluate_model,
    save_evaluation_artifacts,
)
from predictiq.models.explainability import generate_shap_explanations
from predictiq.models.persistence import save_model_pipeline
from predictiq.models.thresholds import BusinessCostMatrix, optimize_decision_threshold
from predictiq.models.tuning import tune_all_candidate_models
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

    for _, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

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
    random_state: int = DEFAULT_EXPLORATORY_SPLIT_SEED,
    cv_splits: int = 5,
    save_reports: bool = True,
    log_to_mlflow: bool = True,
    persist_best_model: bool = True,
) -> Dict[str, Any]:
    """Execute baseline modeling workflow adhering to training-only selection.

    Model selection is performed STRICTLY based on training cross-validation Average Precision,
    preserving evaluation protocol independence.
    """
    logger.info("=" * 70)
    logger.info("PredictIQ Baseline Model Training (Training-Only CV Selection)")
    logger.info("=" * 70)

    splits: DatasetSplits = load_and_split_data(
        data_path=data_path,
        test_size=test_size,
        random_state=random_state,
        split_purpose="exploratory",
    )

    tracking_manager = MLflowTrackingManager() if log_to_mlflow else None
    model_candidates = get_all_baselines(random_state=random_state)
    evaluation_results: List[ModelEvaluationResult] = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    for model_name, pipeline in model_candidates.items():
        logger.info(f"--- Candidate: {model_name} ---")
        mean_pr, std_pr, mean_roc, std_roc = run_cross_validation_evaluation(
            pipeline=pipeline,
            X_train=splits.X_train,
            y_train=splits.y_train,
            n_splits=cv_splits,
            random_state=random_state,
        )
        logger.info(f"Training CV PR-AUC: {mean_pr:.4f} (+/- {std_pr:.4f}) | CV ROC-AUC: {mean_roc:.4f}")

        pipeline.fit(splits.X_train, splits.y_train)
        fitted_pipelines[model_name] = pipeline

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

    config = get_config()
    rep_dir = config.paths.root_dir / "reports"
    artifact_paths: Dict[str, Path] = {}
    if save_reports:
        artifact_paths = save_evaluation_artifacts(evaluation_results, reports_dir=rep_dir)

    mlflow_runs: Dict[str, Any] = {}
    if tracking_manager is not None:
        for res in evaluation_results:
            pipe = fitted_pipelines[res.model_name]
            classifier = pipe.named_steps["classifier"]
            params = classifier.get_params()
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

    # Select champion strictly by Training CV Mean PR-AUC (NOT test score)
    comp_df = build_comparison_dataframe(evaluation_results)
    best_row = comp_df.sort_values(by="cv_mean_pr_auc", ascending=False).iloc[0]
    best_model_name = str(best_row["model_name"])
    best_pipeline = fitted_pipelines[best_model_name]

    logger.info(
        f"Selected Best Baseline: '{best_model_name}' based on Training CV PR-AUC = "
        f"{best_row.get('cv_mean_pr_auc'):.4f} (+/- {best_row.get('cv_std_pr_auc', 0):.4f})"
    )

    saved_model_paths: Optional[Tuple[Path, Path]] = None
    if persist_best_model:
        metadata = {
            "selected_model_name": best_model_name,
            "selection_criterion": "highest_training_cv_mean_pr_auc",
            "cv_mean_pr_auc": best_row.get("cv_mean_pr_auc"),
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

    return {
        "comparison_dataframe": comp_df,
        "evaluation_results": evaluation_results,
        "best_model_name": best_model_name,
        "best_model_metrics": best_row.to_dict(),
        "saved_model_paths": saved_model_paths,
        "artifact_paths": artifact_paths,
        "mlflow_runs": mlflow_runs,
    }


def run_phase4_modeling_pipeline(
    data_path: Optional[Path | str] = None,
    isolated_eval_seed: int = ISOLATED_EVALUATION_SPLIT_SEED,
    cv_splits: int = 5,
    n_tuning_iter: int = 10,
    cost_matrix: Optional[BusinessCostMatrix] = None,
    save_reports: bool = True,
    log_to_mlflow: bool = True,
    persist_champion: bool = True,
) -> Dict[str, Any]:
    """Execute complete Phase 4 modeling, tuning, calibration, thresholding & explainability workflow.

    Protocol Guarantees:
    1. Dataset is split using isolated_eval_seed (2026). The test set is locked until final evaluation.
    2. Candidate models (Logistic Regression, Random Forest, LightGBM) are tuned and compared on training CV.
    3. The champion model is selected and frozen strictly based on training CV PR-AUC.
    4. Probabilities are calibrated on training folds via Platt scaling (sigmoid).
    5. The decision threshold p* is optimized on training out-of-fold predictions and frozen.
    6. Held-out test set is evaluated exactly once with the frozen champion model and frozen threshold.
    7. TreeSHAP explanations are computed on a representative training sample.
    """
    logger.info("=" * 75)
    logger.info("PredictIQ Phase 4: Advanced Modeling, Tuning, Calibration & Explainability")
    logger.info("=" * 75)

    # Step 1: Load and isolate evaluation data
    logger.info(f"[1/7] Partitioning dataset into isolated split (seed={isolated_eval_seed})...")
    eval_splits: DatasetSplits = load_and_split_isolated_evaluation_data(
        data_path=data_path,
        test_size=0.20,
        random_state=isolated_eval_seed,
    )
    logger.info(
        f"Data partitioned: Training={eval_splits.train_size} rows, "
        f"Held-Out Evaluation={eval_splits.test_size} rows (LOCKED)."
    )

    config = get_config()
    rep_dir = config.paths.root_dir / "reports"
    tracking_manager = MLflowTrackingManager() if log_to_mlflow else None

    # Step 2: Hyperparameter Tuning on Training Data Only
    logger.info(f"[2/7] Running training-only Stratified {cv_splits}-Fold CV hyperparameter tuning...")
    tuning_results = tune_all_candidate_models(
        X_train=eval_splits.X_train,
        y_train=eval_splits.y_train,
        n_iter=n_tuning_iter,
        cv_folds=cv_splits,
        random_state=isolated_eval_seed,
    )

    # Step 3: Model Selection on Training CV Only (Frozen Pre-Test)
    logger.info("[3/7] Selecting and freezing champion model based strictly on Training CV PR-AUC...")
    best_candidate_name = max(tuning_results.keys(), key=lambda k: tuning_results[k].best_cv_score)
    best_tuning_result = tuning_results[best_candidate_name]
    frozen_pipeline = best_tuning_result.best_estimator

    logger.info(
        f"Champion Selected: '{best_candidate_name}' | "
        f"CV PR-AUC: {best_tuning_result.best_cv_score:.4f} (+/- {best_tuning_result.cv_std_score:.4f}) | "
        f"Best Params: {best_tuning_result.best_params}"
    )

    # Step 4: Probability Calibration (Training Folds Only)
    logger.info("[4/7] Calibrating champion probabilities via Sigmoid Platt Scaling on training folds...")
    calibration_res = calibrate_model_pipeline(
        pipeline=frozen_pipeline,
        X_train=eval_splits.X_train,
        y_train=eval_splits.y_train,
        method="sigmoid",
        cv=cv_splits,
        reports_dir=rep_dir if save_reports else None,
    )
    calibrated_pipeline = calibration_res.calibrated_pipeline

    # Step 5: Decision Threshold Optimization (Training Out-Of-Fold Only)
    logger.info("[5/7] Optimizing decision threshold on training out-of-fold predictions...")
    oof_probas = cross_val_predict(
        calibrated_pipeline,
        eval_splits.X_train,
        eval_splits.y_train,
        cv=cv_splits,
        method="predict_proba",
    )[:, 1]

    threshold_res = optimize_decision_threshold(
        y_train_oof_true=eval_splits.y_train,
        y_train_oof_probas=oof_probas,
        cost_matrix=cost_matrix,
        reports_dir=rep_dir if save_reports else None,
    )
    frozen_threshold = threshold_res.optimal_threshold
    logger.info(f"Frozen Decision Threshold: p* = {frozen_threshold:.4f} (Selection: {threshold_res.selection_strategy})")

    # Step 6: Single Final Evaluation on Isolated Held-Out Test Set
    logger.info("[6/7] Evaluating frozen champion model on isolated held-out test data (ONCE)...")
    calibrated_pipeline.fit(eval_splits.X_train, eval_splits.y_train)

    test_probas = calibrated_pipeline.predict_proba(eval_splits.X_test)[:, 1]
    test_preds_default = (test_probas >= 0.50).astype(int)
    test_preds_optimal = (test_probas >= frozen_threshold).astype(int)

    # Evaluate standard metrics (probability based)
    final_eval_result = evaluate_model(
        model=calibrated_pipeline,
        X_test=eval_splits.X_test,
        y_test=eval_splits.y_test,
        model_name=f"{best_candidate_name} (Calibrated)",
        model_type=best_candidate_name,
        cv_mean_pr_auc=best_tuning_result.best_cv_score,
        cv_std_pr_auc=best_tuning_result.cv_std_score,
        store_predictions=True,
    )

    from sklearn.metrics import f1_score, precision_score, recall_score

    # Compute metric comparisons at default 0.50 vs optimal threshold
    f1_default = float(f1_score(eval_splits.y_test, test_preds_default, zero_division=0))
    rec_default = float(recall_score(eval_splits.y_test, test_preds_default, zero_division=0))
    prec_default = float(precision_score(eval_splits.y_test, test_preds_default, zero_division=0))

    f1_opt = float(f1_score(eval_splits.y_test, test_preds_optimal, zero_division=0))
    rec_opt = float(recall_score(eval_splits.y_test, test_preds_optimal, zero_division=0))
    prec_opt = float(precision_score(eval_splits.y_test, test_preds_optimal, zero_division=0))

    logger.info(
        f"Final Test Metrics -> PR-AUC: {final_eval_result.average_precision:.4f} | "
        f"ROC-AUC: {final_eval_result.roc_auc:.4f} | Brier: {final_eval_result.brier_score:.4f}"
    )
    logger.info(
        f"Threshold Impact -> Default (p=0.50): F1={f1_default:.4f}, Rec={rec_default:.4f}, Prec={prec_default:.4f} | "
        f"Optimal (p*={frozen_threshold:.2f}): F1={f1_opt:.4f}, Rec={rec_opt:.4f}, Prec={prec_opt:.4f}"
    )

    # Step 7: Explainability & SHAP
    logger.info("[7/7] Computing TreeSHAP global and local archetype explanations...")
    shap_res = generate_shap_explanations(
        pipeline=frozen_pipeline,  # Use base tree pipeline for fast TreeSHAP attribution
        X_sample=eval_splits.X_train,
        reports_dir=rep_dir if save_reports else None,
    )

    # Save artifacts & export reports
    artifact_paths: Dict[str, Path] = {}
    if save_reports:
        # Comparison table of CV scores across candidates
        cv_records = []
        for name, tr in tuning_results.items():
            cv_records.append({
                "model_name": name,
                "cv_mean_pr_auc": round(tr.best_cv_score, 4),
                "cv_std_pr_auc": round(tr.cv_std_score, 4),
                "runtime_seconds": round(tr.runtime_seconds, 2),
                "best_parameters": str(tr.best_params),
            })
        cv_df = pd.DataFrame(cv_records).sort_values(by="cv_mean_pr_auc", ascending=False)
        cv_df.to_csv(rep_dir / "phase4_cv_model_comparison.csv", index=False)

        # Final test evaluation summary
        test_summary_df = pd.DataFrame([{
            "model_name": f"{best_candidate_name} (Calibrated)",
            "test_pr_auc": round(final_eval_result.average_precision, 4),
            "test_roc_auc": round(final_eval_result.roc_auc, 4),
            "test_brier_score": round(final_eval_result.brier_score, 4),
            "test_f1_optimal_threshold": round(f1_opt, 4),
            "test_recall_optimal_threshold": round(rec_opt, 4),
            "test_precision_optimal_threshold": round(prec_opt, 4),
            "optimal_threshold": round(frozen_threshold, 4),
            "test_f1_default_threshold": round(f1_default, 4),
            "test_recall_default_threshold": round(rec_default, 4),
            "test_precision_default_threshold": round(prec_default, 4),
        }])
        test_summary_df.to_csv(rep_dir / "phase4_final_test_summary.csv", index=False)

        artifact_paths = {
            "cv_comparison_csv": rep_dir / "phase4_cv_model_comparison.csv",
            "test_summary_csv": rep_dir / "phase4_final_test_summary.csv",
        }
        if calibration_res.plot_path:
            artifact_paths["calibration_curve"] = calibration_res.plot_path
        if threshold_res.plot_path:
            artifact_paths["threshold_tradeoff"] = threshold_res.plot_path
        if shap_res.summary_plot_path:
            artifact_paths["shap_summary"] = shap_res.summary_plot_path
        if shap_res.local_high_risk_plot_path:
            artifact_paths["shap_high_risk"] = shap_res.local_high_risk_plot_path
        if shap_res.local_low_risk_plot_path:
            artifact_paths["shap_low_risk"] = shap_res.local_low_risk_plot_path

    # Persist champion model pipeline and metadata
    saved_paths: Optional[Tuple[Path, Path]] = None
    if persist_champion:
        meta = {
            "model_name": f"{best_candidate_name} (Calibrated)",
            "base_estimator": best_candidate_name,
            "best_hyperparameters": best_tuning_result.best_params,
            "optimal_decision_threshold": frozen_threshold,
            "threshold_selection_strategy": threshold_res.selection_strategy,
            "brier_score_uncalibrated": calibration_res.brier_score_uncalibrated,
            "brier_score_calibrated": calibration_res.brier_score_calibrated,
            "brier_improvement_pct": calibration_res.brier_improvement_pct,
            "cv_mean_pr_auc": best_tuning_result.best_cv_score,
            "cv_std_pr_auc": best_tuning_result.cv_std_score,
            "final_test_pr_auc": final_eval_result.average_precision,
            "final_test_roc_auc": final_eval_result.roc_auc,
            "final_test_f1_optimal_threshold": f1_opt,
            "final_test_recall_optimal_threshold": rec_opt,
            "final_test_precision_optimal_threshold": prec_opt,
            "split_seed": isolated_eval_seed,
            "split_purpose": "isolated_evaluation",
            "top_10_shap_features": shap_res.top_10_features,
            "python_version": sys.version,
        }
        saved_paths = save_model_pipeline(
            pipeline=calibrated_pipeline,
            metadata=meta,
            model_name="churn_champion_model",
        )

    # Log to MLflow
    if tracking_manager is not None:
        tracking_manager.log_model_run(
            model_name=f"champion_{best_candidate_name.lower().replace(' ', '_')}",
            pipeline=calibrated_pipeline,
            eval_result=final_eval_result,
            hyperparameters={
                **best_tuning_result.best_params,
                "optimal_threshold": frozen_threshold,
                "calibration_method": "sigmoid",
            },
            X_sample=eval_splits.X_train.head(5),
            dataset_meta={
                "split_seed": isolated_eval_seed,
                "split_purpose": "isolated_evaluation",
                "train_size": eval_splits.train_size,
                "test_size": eval_splits.test_size,
            },
            artifact_paths=artifact_paths,
            tags={"phase": "phase_4_champion", "status": "frozen_evaluated"},
        )

    logger.info("=" * 75)
    logger.info("Phase 4 Pipeline Completed Successfully.")
    logger.info("=" * 75)

    return {
        "champion_model_name": best_candidate_name,
        "tuning_results": tuning_results,
        "best_tuning_result": best_tuning_result,
        "calibration_result": calibration_res,
        "threshold_result": threshold_res,
        "final_test_metrics": {
            "pr_auc": final_eval_result.average_precision,
            "roc_auc": final_eval_result.roc_auc,
            "brier_score": final_eval_result.brier_score,
            "f1_optimal": f1_opt,
            "recall_optimal": rec_opt,
            "precision_optimal": prec_opt,
            "f1_default": f1_default,
            "recall_default": rec_default,
            "precision_default": prec_default,
        },
        "shap_result": shap_res,
        "saved_model_paths": saved_paths,
        "artifact_paths": artifact_paths,
    }


if __name__ == "__main__":
    results = run_phase4_modeling_pipeline()
    print("\n--- Champion Model Selected ---")
    print(f"Model: {results['champion_model_name']}")
    print(f"Optimal Threshold: p* = {results['threshold_result'].optimal_threshold:.4f}")
    print(f"Test PR-AUC: {results['final_test_metrics']['pr_auc']:.4f}")
    print(f"Test F1 at p*: {results['final_test_metrics']['f1_optimal']:.4f}")
