"""Executable script to run PredictIQ Phase 4 advanced tuning, calibration, and explainability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project src to sys.path if running as standalone script
project_root = Path(__file__).resolve().parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from predictiq.data.loader import ISOLATED_EVALUATION_SPLIT_SEED
from predictiq.models.thresholds import BusinessCostMatrix
from predictiq.models.train import run_phase4_modeling_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PredictIQ Phase 4 advanced modeling, tuning, calibration, and explainability."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional custom path to Telco churn CSV dataset.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=ISOLATED_EVALUATION_SPLIT_SEED,
        help=f"Random seed for the isolated evaluation split (default: {ISOLATED_EVALUATION_SPLIT_SEED}).",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds on training data (default: 5).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Number of hyperparameter search iterations per candidate (default: 10).",
    )
    parser.add_argument(
        "--cost-fp",
        type=float,
        default=None,
        help="Optional cost of a false positive for threshold optimization.",
    )
    parser.add_argument(
        "--cost-fn",
        type=float,
        default=None,
        help="Optional cost of a false negative for threshold optimization.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment logging.",
    )

    args = parser.parse_args()

    cost_matrix = None
    if args.cost_fp is not None and args.cost_fn is not None:
        cost_matrix = BusinessCostMatrix(
            cost_false_positive=args.cost_fp,
            cost_false_negative=args.cost_fn,
        )

    results = run_phase4_modeling_pipeline(
        data_path=args.data_path,
        isolated_eval_seed=args.eval_seed,
        cv_splits=args.cv_splits,
        n_tuning_iter=args.n_iter,
        cost_matrix=cost_matrix,
        log_to_mlflow=not args.no_mlflow,
    )

    print("\n" + "=" * 75)
    print("PHASE 4 CHAMPION MODEL RESULTS SUMMARY")
    print("=" * 75)
    print(f"Champion Model Architecture:  {results['champion_model_name']}")
    print(f"Training CV PR-AUC (Mean):    {results['best_tuning_result'].best_cv_score:.4f} (+/- {results['best_tuning_result'].cv_std_score:.4f})")
    print(f"Brier Score Reduction:        {results['calibration_result'].brier_score_uncalibrated:.4f} -> {results['calibration_result'].brier_score_calibrated:.4f} ({results['calibration_result'].brier_improvement_pct:+.2f}%)")
    print(f"Frozen Optimal Threshold:     p* = {results['threshold_result'].optimal_threshold:.4f}")
    print(f"Final Test PR-AUC (Isolated): {results['final_test_metrics']['pr_auc']:.4f}")
    print(f"Final Test ROC-AUC (Isolated):{results['final_test_metrics']['roc_auc']:.4f}")
    print(f"Final Test F1 at p*:          {results['final_test_metrics']['f1_optimal']:.4f} (Recall: {results['final_test_metrics']['recall_optimal']:.4f}, Precision: {results['final_test_metrics']['precision_optimal']:.4f})")
    print(f"Final Test F1 at default 0.5: {results['final_test_metrics']['f1_default']:.4f} (Recall: {results['final_test_metrics']['recall_default']:.4f}, Precision: {results['final_test_metrics']['precision_default']:.4f})")
    if results.get("saved_model_paths"):
        print(f"Saved Champion Pipeline:      {results['saved_model_paths'][0]}")
        print(f"Saved Champion Metadata:      {results['saved_model_paths'][1]}")
    print("=" * 75 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
